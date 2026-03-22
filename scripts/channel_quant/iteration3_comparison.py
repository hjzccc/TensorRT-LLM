#!/usr/bin/env python3
# pyright: basic, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

from iteration2_budget_sweep import (
    DEFAULT_LOGIT_CHUNK,
    LayerMetricBundle,
    build_projection_plan as build_iteration2_projection_plan,
    build_expert_only_plan as build_iteration2_expert_only_plan,
    compute_gate_aware_projection_metrics,
    estimate_memory_gb,
    fp8_weights_from_masks,
    load_json,
    load_prefix_dataset_tokens,
    quantize_gate_up_with_pair_mask,
    resolve_non_expert_bytes,
    topk_mask_from_scores,
)
from new_spike1_channel_metrics import (
    EPS,
    quantize_down_proj_with_mask,
    quantize_linear_weight,
    resolve_terminal_keys,
    run_calibration_forward,
)
from spike1_ground_truth import (
    MODEL_ID,
    WeightStore,
    build_causal_mask,
    build_text_config,
    dtype_from_name,
    full_attention_forward,
    layer_keys,
    linear_attention_forward,
    load_root_config,
    move_tensor,
    quantize_to_fp8,
    quantize_to_nvfp4_columns,
    release_tensors,
    rms_norm_qwen3_next,
    shorten_layer_tensors,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "iteration3_results.json"
DEFAULT_EXPLORATION_MD = SCRIPT_DIR / "exploration.md"
DEFAULT_ITERATION2_JSON = RESULTS_DIR / "iteration2_results.json"
DEFAULT_CALIBRATION_TOKENS = 128
DEFAULT_EVAL_TOKENS = 512
DEFAULT_EVAL_CHUNK_TOKENS = 1024
DEFAULT_RANDOM_SEEDS = (0, 1, 2)
DEFAULT_FGMP_SAMPLES_PER_LAYER = 65536
SECTION_MARKER = "## [9] Iteration 3 - Related Work Comparison + Alternative Strategies"
TARGET_FP8_FRACTION = 0.25
W1_PAIR_WEIGHT_COST = 4096
W2_CHANNEL_WEIGHT_COST = 512
K_BLOCK_SIZE = 16


@dataclass
class EvaluationPlan:
    name: str
    description: str
    mode: str
    memory_gb: float
    fp8_weights: int
    promoted_experts: dict[int, set[int]] | None = None
    w1_pair_masks: dict[int, dict[int, torch.Tensor]] | None = None
    w2_channel_masks: dict[int, dict[int, torch.Tensor]] | None = None
    w1_projection_fp8: dict[int, torch.Tensor] | None = None
    w2_projection_fp8: dict[int, torch.Tensor] | None = None
    fgmp_block_threshold: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--calibration-tokens", type=int, default=DEFAULT_CALIBRATION_TOKENS)
    parser.add_argument("--eval-tokens", type=int, default=DEFAULT_EVAL_TOKENS)
    parser.add_argument("--eval-chunk-tokens", type=int, default=DEFAULT_EVAL_CHUNK_TOKENS)
    parser.add_argument("--logit-chunk-size", type=int, default=DEFAULT_LOGIT_CHUNK)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--exploration-md", type=Path, default=DEFAULT_EXPLORATION_MD)
    parser.add_argument("--iteration2-json", type=Path, default=DEFAULT_ITERATION2_JSON)
    parser.add_argument("--fgmp-samples-per-layer", type=int, default=DEFAULT_FGMP_SAMPLES_PER_LAYER)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def atomic_json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp_path.replace(path)


def empty_payload(model_id: str, calibration: dict[str, Any], evaluation: dict[str, Any], fgmp_samples_per_layer: int) -> dict[str, Any]:
    return {
        "metadata": {
            "model": model_id,
            "calibration": calibration,
            "evaluation": evaluation,
            "target_fp8_fraction": TARGET_FP8_FRACTION,
            "metric": "gate_aware_act_qerror",
            "fgmp_block_proxy": "weight_variance_per_16_block",
            "fgmp_samples_per_layer": fgmp_samples_per_layer,
            "random_seeds": list(DEFAULT_RANDOM_SEEDS),
            "note": "W1 uses pair-wise gate_up_proj assignment when channel-level masks are used.",
        },
        "related_work": {},
        "alternative_strategies": {},
    }


def upsert_result(payload: dict[str, Any], category: str, key: str, value: dict[str, Any]) -> None:
    payload[category][key] = value


def maybe_get_saved_result(payload: dict[str, Any], category: str, key: str) -> dict[str, Any] | None:
    bucket = payload.get(category, {})
    if not isinstance(bucket, dict):
        return None
    saved = bucket.get(key)
    return saved if isinstance(saved, dict) else None


def flatten_metric_cache(metric_cache: dict[int, LayerMetricBundle]) -> tuple[torch.Tensor, torch.Tensor]:
    ordered_layers = [metric_cache[layer_idx] for layer_idx in sorted(metric_cache)]
    w1_flat = torch.cat([bundle.w1_pair_scores.reshape(-1) for bundle in ordered_layers], dim=0)
    w2_flat = torch.cat([bundle.w2_channel_scores.reshape(-1) for bundle in ordered_layers], dim=0)
    return w1_flat.to(torch.float32), w2_flat.to(torch.float32)


def percentile_threshold(values: torch.Tensor, percentile: float) -> torch.Tensor:
    flat = values.reshape(-1)
    if int(flat.numel()) == 0:
        raise ValueError("Cannot compute percentile on empty tensor")
    bounded = min(100.0, max(0.0, percentile))
    if bounded <= 0.0:
        return flat.min()
    if bounded >= 100.0:
        return flat.max()
    rank = int(math.floor((bounded / 100.0) * max(int(flat.numel()) - 1, 0))) + 1
    return torch.kthvalue(flat, rank).values


def quantize_weight_with_kblock_mask(weight: torch.Tensor, fp8_block_mask: torch.Tensor) -> torch.Tensor:
    if fp8_block_mask.ndim != 2:
        raise ValueError(f"Expected 2D K-block mask, got {tuple(fp8_block_mask.shape)}")
    out_features, in_features = weight.shape
    expected_shape = (out_features, in_features // K_BLOCK_SIZE)
    if tuple(fp8_block_mask.shape) != expected_shape:
        raise ValueError(f"Expected K-block mask {expected_shape}, got {tuple(fp8_block_mask.shape)}")
    if not bool(fp8_block_mask.any()):
        return quantize_linear_weight(weight, "fp4")
    if bool(fp8_block_mask.all()):
        return quantize_linear_weight(weight, "fp8")
    weight_t = weight.transpose(0, 1).contiguous()
    fp4_t = quantize_to_nvfp4_columns(weight_t)
    fp8_t = quantize_to_fp8(weight_t)
    in_blocks = weight_t.shape[0] // K_BLOCK_SIZE
    fp4_blocks = fp4_t.view(in_blocks, K_BLOCK_SIZE, weight_t.shape[1])
    fp8_blocks = fp8_t.view(in_blocks, K_BLOCK_SIZE, weight_t.shape[1])
    block_mask = fp8_block_mask.to(device=weight_t.device, dtype=torch.bool).transpose(0, 1).contiguous().view(in_blocks, 1, weight_t.shape[1])
    mixed = torch.where(block_mask, fp8_blocks, fp4_blocks).view_as(weight_t)
    return mixed.transpose(0, 1).contiguous()


def block_variance_scores(weight: torch.Tensor) -> torch.Tensor:
    if weight.shape[-1] % K_BLOCK_SIZE != 0:
        raise ValueError(f"Expected K dimension multiple of {K_BLOCK_SIZE}, got {tuple(weight.shape)}")
    blocks = weight.float().view(*weight.shape[:-1], weight.shape[-1] // K_BLOCK_SIZE, K_BLOCK_SIZE)
    return blocks.var(dim=-1, correction=0)


def sample_block_variance_scores(weight: torch.Tensor, sample_count: int, generator: torch.Generator) -> torch.Tensor:
    scores = block_variance_scores(weight)
    flat = scores.reshape(-1)
    if sample_count >= int(flat.numel()):
        return flat
    indices = torch.randint(0, int(flat.numel()), (sample_count,), generator=generator, dtype=torch.int64)
    return flat.index_select(0, indices)


def maybe_reuse_iteration2_result(
    iteration2_payload: dict[str, Any] | None,
    key: str,
    calibration_tokens: int,
    eval_tokens: int,
) -> dict[str, Any] | None:
    if iteration2_payload is None:
        return None
    metadata = iteration2_payload.get("metadata", {})
    calibration = metadata.get("calibration", {}) if isinstance(metadata, dict) else {}
    evaluation = metadata.get("evaluation", {}) if isinstance(metadata, dict) else {}
    if int(calibration.get("actual_tokens", -1)) != calibration_tokens:
        return None
    if int(evaluation.get("actual_tokens", -1)) != eval_tokens:
        return None
    hier = iteration2_payload.get("hierarchy_comparison", {})
    value = hier.get(key) if isinstance(hier, dict) else None
    return value if isinstance(value, dict) else None


def build_zero_w1_masks(config: Any) -> dict[int, dict[int, torch.Tensor]]:
    return {
        layer_idx: {
            expert_idx: torch.zeros(config.moe_intermediate_size, dtype=torch.bool)
            for expert_idx in range(config.num_experts)
        }
        for layer_idx in range(config.num_hidden_layers)
    }


def build_uniform_random_masks(config: Any, seed: int) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    w1_pair_masks: dict[int, dict[int, torch.Tensor]] = {}
    w2_channel_masks: dict[int, dict[int, torch.Tensor]] = {}
    k_w1 = int(round(TARGET_FP8_FRACTION * config.moe_intermediate_size))
    k_w2 = int(round(TARGET_FP8_FRACTION * config.hidden_size))
    for layer_idx in range(config.num_hidden_layers):
        layer_w1: dict[int, torch.Tensor] = {}
        layer_w2: dict[int, torch.Tensor] = {}
        for expert_idx in range(config.num_experts):
            w1_mask = torch.zeros(config.moe_intermediate_size, dtype=torch.bool)
            w2_mask = torch.zeros(config.hidden_size, dtype=torch.bool)
            if k_w1 > 0:
                w1_perm = torch.randperm(config.moe_intermediate_size, generator=generator)
                w1_mask[w1_perm[:k_w1]] = True
            if k_w2 > 0:
                w2_perm = torch.randperm(config.hidden_size, generator=generator)
                w2_mask[w2_perm[:k_w2]] = True
            layer_w1[expert_idx] = w1_mask
            layer_w2[expert_idx] = w2_mask
        w1_pair_masks[layer_idx] = layer_w1
        w2_channel_masks[layer_idx] = layer_w2
    return w1_pair_masks, w2_channel_masks


def build_global_threshold_plan(
    name: str,
    description: str,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
    metric_cache: dict[int, LayerMetricBundle],
    percentile: float,
) -> EvaluationPlan:
    w1_flat, w2_flat = flatten_metric_cache(metric_cache)
    all_scores = torch.cat([w1_flat, w2_flat], dim=0)
    threshold = percentile_threshold(all_scores, percentile)
    w1_pair_masks: dict[int, dict[int, torch.Tensor]] = {}
    w2_channel_masks: dict[int, dict[int, torch.Tensor]] = {}
    for layer_idx, bundle in metric_cache.items():
        w1_pair_masks[layer_idx] = {}
        w2_channel_masks[layer_idx] = {}
        for expert_idx in range(config.num_experts):
            w1_pair_masks[layer_idx][expert_idx] = bundle.w1_pair_scores[expert_idx] > threshold
            w2_channel_masks[layer_idx][expert_idx] = bundle.w2_channel_scores[expert_idx] > threshold
    fp8_weights = fp8_weights_from_masks(config, w1_pair_masks, w2_channel_masks)
    return EvaluationPlan(
        name=name,
        description=description,
        mode="mixed_channel",
        memory_gb=estimate_memory_gb(non_expert_bytes, total_expert_elems, fp8_weights),
        fp8_weights=fp8_weights,
        w1_pair_masks=w1_pair_masks,
        w2_channel_masks=w2_channel_masks,
    )


def cluster_high_mask(scores: torch.Tensor, max_iters: int = 32) -> torch.Tensor:
    values = scores.to(torch.float32)
    if values.numel() == 0:
        return torch.zeros(0, dtype=torch.bool)
    low = float(values.min().item())
    high = float(values.max().item())
    if not math.isfinite(low) or not math.isfinite(high) or abs(high - low) <= EPS:
        return torch.zeros(values.numel(), dtype=torch.bool)
    center_a = low
    center_b = high
    mask = torch.zeros(values.numel(), dtype=torch.bool)
    for _ in range(max_iters):
        assign_b = (values - center_b).abs() <= (values - center_a).abs()
        if not bool(assign_b.any()) or bool(assign_b.all()):
            return topk_mask_from_scores(values, values.numel() // 2)
        new_a = float(values[~assign_b].mean().item())
        new_b = float(values[assign_b].mean().item())
        mask = assign_b if new_b >= new_a else ~assign_b
        if abs(new_a - center_a) <= 1e-6 and abs(new_b - center_b) <= 1e-6:
            break
        center_a, center_b = new_a, new_b
    if not bool(mask.any()) or bool(mask.all()):
        return topk_mask_from_scores(values, values.numel() // 2)
    return mask.cpu()


def gap_split_mask(scores: torch.Tensor) -> torch.Tensor:
    values = scores.to(torch.float32)
    if values.numel() < 2 or float(values.max().item() - values.min().item()) <= EPS:
        return values > torch.quantile(values, 0.5)
    sorted_values, _ = torch.sort(values)
    diffs = sorted_values[1:] - sorted_values[:-1]
    positive = diffs[diffs > 0]
    if int(positive.numel()) == 0:
        return values > torch.quantile(values, 0.5)
    best_idx = int(torch.argmax(diffs).item())
    best_gap = float(diffs[best_idx].item())
    median_gap = float(torch.quantile(positive, 0.5).item())
    if best_gap <= max(1e-8, 2.0 * median_gap):
        return values > torch.quantile(values, 0.5)
    cutoff = float(sorted_values[best_idx].item())
    mask = values > cutoff
    if not bool(mask.any()) or bool(mask.all()):
        return values > torch.quantile(values, 0.5)
    return mask.cpu()


def build_kmeans_plan(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> EvaluationPlan:
    w1_pair_masks = build_zero_w1_masks(config)
    w2_channel_masks: dict[int, dict[int, torch.Tensor]] = {}
    for layer_idx, bundle in metric_cache.items():
        layer_masks: dict[int, torch.Tensor] = {}
        for expert_idx in range(config.num_experts):
            layer_masks[expert_idx] = cluster_high_mask(bundle.w2_channel_scores[expert_idx])
        w2_channel_masks[layer_idx] = layer_masks
    fp8_weights = fp8_weights_from_masks(config, w1_pair_masks, w2_channel_masks)
    return EvaluationPlan(
        name="kmeans_2cluster",
        description="RPTQ-style per-expert 2-means on W2 gate-aware scores; W1 stays FP4.",
        mode="mixed_channel",
        memory_gb=estimate_memory_gb(non_expert_bytes, total_expert_elems, fp8_weights),
        fp8_weights=fp8_weights,
        w1_pair_masks=w1_pair_masks,
        w2_channel_masks=w2_channel_masks,
    )


def build_gap_plan(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> EvaluationPlan:
    w1_pair_masks: dict[int, dict[int, torch.Tensor]] = {}
    w2_channel_masks: dict[int, dict[int, torch.Tensor]] = {}
    for layer_idx, bundle in metric_cache.items():
        layer_w1: dict[int, torch.Tensor] = {}
        layer_w2: dict[int, torch.Tensor] = {}
        for expert_idx in range(config.num_experts):
            layer_w1[expert_idx] = gap_split_mask(bundle.w1_pair_scores[expert_idx])
            layer_w2[expert_idx] = gap_split_mask(bundle.w2_channel_scores[expert_idx])
        w1_pair_masks[layer_idx] = layer_w1
        w2_channel_masks[layer_idx] = layer_w2
    fp8_weights = fp8_weights_from_masks(config, w1_pair_masks, w2_channel_masks)
    return EvaluationPlan(
        name="sensitivity_gap",
        description="OWQ-style largest-gap split per expert/projection with median fallback.",
        mode="mixed_channel",
        memory_gb=estimate_memory_gb(non_expert_bytes, total_expert_elems, fp8_weights),
        fp8_weights=fp8_weights,
        w1_pair_masks=w1_pair_masks,
        w2_channel_masks=w2_channel_masks,
    )


def build_mxmoe_plan(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> EvaluationPlan:
    target_fp8_weights = int(round(TARGET_FP8_FRACTION * total_expert_elems))
    items: list[tuple[float, int, int, int, str]] = []
    for layer_idx, bundle in metric_cache.items():
        for expert_idx in range(config.num_experts):
            w1_score = float(bundle.w1_pair_scores[expert_idx].sum().item())
            w2_score = float(bundle.w2_channel_scores[expert_idx].sum().item())
            items.append((w1_score / (2.0 * config.moe_intermediate_size * config.hidden_size + EPS), 2 * config.moe_intermediate_size * config.hidden_size, layer_idx, expert_idx, "w1"))
            items.append((w2_score / (config.hidden_size * config.moe_intermediate_size + EPS), config.hidden_size * config.moe_intermediate_size, layer_idx, expert_idx, "w2"))
    items.sort(key=lambda item: (item[0], -item[1], -item[2], -item[3], item[4]), reverse=True)

    w1_projection_fp8 = {layer_idx: torch.zeros(config.num_experts, dtype=torch.bool) for layer_idx in range(config.num_hidden_layers)}
    w2_projection_fp8 = {layer_idx: torch.zeros(config.num_experts, dtype=torch.bool) for layer_idx in range(config.num_hidden_layers)}
    fp8_weights = 0
    for _ratio, cost, layer_idx, expert_idx, projection in items:
        if fp8_weights + cost > target_fp8_weights:
            continue
        if projection == "w1":
            w1_projection_fp8[layer_idx][expert_idx] = True
        else:
            w2_projection_fp8[layer_idx][expert_idx] = True
        fp8_weights += cost
        if fp8_weights >= target_fp8_weights:
            break
    return EvaluationPlan(
        name="mxmoe_per_block",
        description="MxMoE-style greedy knapsack over per-expert W1/W2 projection blocks.",
        mode="projection_block",
        memory_gb=estimate_memory_gb(non_expert_bytes, total_expert_elems, fp8_weights),
        fp8_weights=fp8_weights,
        w1_projection_fp8=w1_projection_fp8,
        w2_projection_fp8=w2_projection_fp8,
    )


def build_scalebits_plan(
    metric_cache: dict[int, LayerMetricBundle],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> EvaluationPlan:
    w1_flat, w2_flat = flatten_metric_cache(metric_cache)
    w1_ratio = w1_flat / float(W1_PAIR_WEIGHT_COST)
    w2_ratio = w2_flat / float(W2_CHANNEL_WEIGHT_COST)
    all_ratio = torch.cat([w1_ratio, w2_ratio], dim=0)
    w1_count = int(w1_flat.numel())
    sorted_indices = torch.argsort(all_ratio, descending=True)
    sorted_is_w1 = sorted_indices < w1_count
    sorted_cost = torch.where(sorted_is_w1, torch.full_like(sorted_indices, 8, dtype=torch.int32), torch.ones_like(sorted_indices, dtype=torch.int32))
    budget_units = int(round(TARGET_FP8_FRACTION * total_expert_elems / W2_CHANNEL_WEIGHT_COST))
    cumulative_cost = torch.cumsum(sorted_cost, dim=0)
    selected_count = int(torch.searchsorted(cumulative_cost, torch.tensor(budget_units, dtype=torch.int32), right=True).item())
    selected_indices = sorted_indices[:selected_count]
    actual_units = int(cumulative_cost[selected_count - 1].item()) if selected_count > 0 else 0

    w1_mask_flat = torch.zeros(w1_count, dtype=torch.bool)
    w2_mask_flat = torch.zeros(int(w2_flat.numel()), dtype=torch.bool)
    chosen_w1 = selected_indices[selected_indices < w1_count]
    chosen_w2 = selected_indices[selected_indices >= w1_count] - w1_count
    if int(chosen_w1.numel()) > 0:
        w1_mask_flat[chosen_w1] = True
    if int(chosen_w2.numel()) > 0:
        w2_mask_flat[chosen_w2] = True

    w1_pair_masks: dict[int, dict[int, torch.Tensor]] = {}
    w2_channel_masks: dict[int, dict[int, torch.Tensor]] = {}
    w1_per_layer = config.num_experts * config.moe_intermediate_size
    w2_per_layer = config.num_experts * config.hidden_size
    for layer_idx in range(config.num_hidden_layers):
        w1_layer_flat = w1_mask_flat[layer_idx * w1_per_layer : (layer_idx + 1) * w1_per_layer].view(config.num_experts, config.moe_intermediate_size)
        w2_layer_flat = w2_mask_flat[layer_idx * w2_per_layer : (layer_idx + 1) * w2_per_layer].view(config.num_experts, config.hidden_size)
        w1_pair_masks[layer_idx] = {expert_idx: w1_layer_flat[expert_idx].clone() for expert_idx in range(config.num_experts)}
        w2_channel_masks[layer_idx] = {expert_idx: w2_layer_flat[expert_idx].clone() for expert_idx in range(config.num_experts)}

    fp8_weights = int(actual_units * W2_CHANNEL_WEIGHT_COST)
    return EvaluationPlan(
        name="scalebits_submodular",
        description="ScaleBITS-style global greedy over W1 pairs and W2 channels by sensitivity per byte.",
        mode="mixed_channel",
        memory_gb=estimate_memory_gb(non_expert_bytes, total_expert_elems, fp8_weights),
        fp8_weights=fp8_weights,
        w1_pair_masks=w1_pair_masks,
        w2_channel_masks=w2_channel_masks,
    )


def estimate_fgmp_threshold(
    store: WeightStore,
    config: Any,
    samples_per_layer: int,
) -> float:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(12345)
    samples: list[torch.Tensor] = []
    for layer_idx in range(config.num_hidden_layers):
        gate_key = f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"
        down_key = f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj"
        raw = store.load_tensors([gate_key, down_key])
        print(f"[fgmp] sampling layer {layer_idx:02d}/{config.num_hidden_layers - 1:02d}", flush=True)
        gate_sample = sample_block_variance_scores(raw[gate_key], samples_per_layer, generator)
        down_sample = sample_block_variance_scores(raw[down_key], samples_per_layer, generator)
        samples.append(gate_sample.to(torch.float32))
        samples.append(down_sample.to(torch.float32))
        del raw, gate_sample, down_sample
        gc.collect()
    all_samples = torch.cat(samples, dim=0)
    return float(torch.quantile(all_samples, 0.75).item())


def count_fgmp_fp8_weights(store: WeightStore, config: Any, threshold: float) -> int:
    total_blocks = 0
    for layer_idx in range(config.num_hidden_layers):
        gate_key = f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"
        down_key = f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj"
        raw = store.load_tensors([gate_key, down_key])
        print(f"[fgmp] counting layer {layer_idx:02d}/{config.num_hidden_layers - 1:02d}", flush=True)
        total_blocks += int((block_variance_scores(raw[gate_key]) > threshold).sum().item())
        total_blocks += int((block_variance_scores(raw[down_key]) > threshold).sum().item())
        del raw
        gc.collect()
    return int(total_blocks * K_BLOCK_SIZE)


def build_fgmp_plan(
    store: WeightStore,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
    samples_per_layer: int,
) -> EvaluationPlan:
    threshold = estimate_fgmp_threshold(store, config, samples_per_layer)
    fp8_weights = count_fgmp_fp8_weights(store, config, threshold)
    return EvaluationPlan(
        name="fgmp_per_16block",
        description="FGMP-style sampled global threshold over per-16 K-block weight variance.",
        mode="fgmp_block16",
        memory_gb=estimate_memory_gb(non_expert_bytes, total_expert_elems, fp8_weights),
        fp8_weights=fp8_weights,
        fgmp_block_threshold=threshold,
    )


def build_random_plan(
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
    seed: int,
) -> EvaluationPlan:
    w1_pair_masks, w2_channel_masks = build_uniform_random_masks(config, seed)
    fp8_weights = fp8_weights_from_masks(config, w1_pair_masks, w2_channel_masks)
    return EvaluationPlan(
        name=f"random_seed_{seed}",
        description=f"Uniform random 25% FP8 W1/W2 assignment, seed={seed}.",
        mode="mixed_channel",
        memory_gb=estimate_memory_gb(non_expert_bytes, total_expert_elems, fp8_weights),
        fp8_weights=fp8_weights,
        w1_pair_masks=w1_pair_masks,
        w2_channel_masks=w2_channel_masks,
    )


def moe_forward_with_plan(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config: Any,
    plan: EvaluationPlan,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    router_logits = F.linear(flat, tensors["gate.weight"]).float()
    routing_probs = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(routing_probs, config.num_experts_per_tok, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
    gate_up_proj = tensors["experts.gate_up_proj"]
    down_proj = tensors["experts.down_proj"]
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)

    layer_promoted = plan.promoted_experts.get(layer_idx) if plan.promoted_experts is not None else None
    layer_w1_masks = plan.w1_pair_masks.get(layer_idx) if plan.w1_pair_masks is not None else None
    layer_w2_masks = plan.w2_channel_masks.get(layer_idx) if plan.w2_channel_masks is not None else None
    layer_w1_proj = plan.w1_projection_fp8.get(layer_idx) if plan.w1_projection_fp8 is not None else None
    layer_w2_proj = plan.w2_projection_fp8.get(layer_idx) if plan.w2_projection_fp8 is not None else None

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        gate_up_weight = gate_up_proj[expert_idx]
        down_weight = down_proj[expert_idx]

        if plan.mode == "expert_only":
            expert_mode = "fp8" if layer_promoted is not None and expert_idx in layer_promoted else "fp4"
            quant_gate_up = quantize_linear_weight(gate_up_weight, expert_mode)
            quant_down = quantize_linear_weight(down_weight, expert_mode)
        elif plan.mode == "projection_block":
            w1_mode = "fp8" if layer_w1_proj is not None and bool(layer_w1_proj[expert_idx]) else "fp4"
            w2_mode = "fp8" if layer_w2_proj is not None and bool(layer_w2_proj[expert_idx]) else "fp4"
            quant_gate_up = quantize_linear_weight(gate_up_weight, w1_mode)
            quant_down = quantize_linear_weight(down_weight, w2_mode)
        elif plan.mode == "mixed_channel":
            pair_mask = layer_w1_masks[expert_idx] if layer_w1_masks is not None else None
            channel_mask = layer_w2_masks[expert_idx] if layer_w2_masks is not None else None
            if pair_mask is None:
                quant_gate_up = quantize_linear_weight(gate_up_weight, "fp4")
            else:
                quant_gate_up = quantize_gate_up_with_pair_mask(gate_up_weight, pair_mask.to(device=gate_up_weight.device))
            if channel_mask is None or not bool(channel_mask.any()):
                quant_down = quantize_linear_weight(down_weight, "fp4")
            elif bool(channel_mask.all()):
                quant_down = quantize_linear_weight(down_weight, "fp8")
            else:
                quant_down = quantize_down_proj_with_mask(down_weight, channel_mask.to(device=down_weight.device))
        elif plan.mode == "fgmp_block16":
            if plan.fgmp_block_threshold is None:
                raise ValueError("FGMP plan missing threshold")
            gate_mask = block_variance_scores(gate_up_weight) > plan.fgmp_block_threshold
            down_mask = block_variance_scores(down_weight) > plan.fgmp_block_threshold
            quant_gate_up = quantize_weight_with_kblock_mask(gate_up_weight, gate_mask)
            quant_down = quantize_weight_with_kblock_mask(down_weight, down_mask)
        else:
            raise ValueError(f"Unsupported plan mode={plan.mode}")

        gate_up = F.linear(current_state, quant_gate_up)
        gate, up = gate_up.chunk(2, dim=-1)
        current_hidden = F.linear(F.silu(gate) * up, quant_down)
        current_hidden = current_hidden * routing_weights[token_idx, route_pos].unsqueeze(-1)
        final_hidden_states.index_add_(0, token_idx, current_hidden.to(hidden_states.dtype))

    shared = F.linear(flat, tensors["shared_expert.gate_proj.weight"])
    shared = F.silu(shared) * F.linear(flat, tensors["shared_expert.up_proj.weight"])
    shared = F.linear(shared, tensors["shared_expert.down_proj.weight"])
    shared_gate = torch.sigmoid(F.linear(flat, tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_gate * shared
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim), expert_counts


def evaluate_chunked_perplexity(
    store: WeightStore,
    config: Any,
    input_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    logit_chunk_size: int,
    eval_chunk_tokens: int,
    plan: EvaluationPlan,
) -> tuple[float, float]:
    embed_key, norm_key, lm_head_key = resolve_terminal_keys(store.weight_map)
    root_tensors = store.load_tensors([embed_key, norm_key, lm_head_key])
    embed_weight = move_tensor(root_tensors[embed_key], device, dtype)
    final_norm = move_tensor(root_tensors[norm_key], device, dtype)
    lm_head = move_tensor(root_tensors[lm_head_key], device, dtype)
    del root_tensors

    total_token_count = int(input_ids.shape[1])
    chunk_starts = list(range(0, total_token_count - 1, eval_chunk_tokens))
    total_nll = 0.0
    total_scored_tokens = 0

    with torch.inference_mode():
        for chunk_idx, begin in enumerate(chunk_starts, start=1):
            end = min(begin + eval_chunk_tokens + 1, total_token_count)
            if end - begin < 2:
                continue
            chunk_ids = input_ids[:, begin:end]
            print(
                f"[eval:{plan.name}] chunk {chunk_idx:03d}/{len(chunk_starts):03d} tokens={begin}:{end} mode={plan.mode}",
                flush=True,
            )
            hidden_states = F.embedding(chunk_ids.to(device), embed_weight)
            seq_len = int(chunk_ids.shape[1])
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            causal_mask = build_causal_mask(seq_len, device)
            rotary = Qwen3NextRotaryEmbedding(config=config, device=device)
            position_embeddings = rotary(hidden_states, position_ids)

            for layer_idx in range(config.num_hidden_layers):
                layer_type = config.layer_types[layer_idx]
                print(
                    f"[eval:{plan.name}] chunk {chunk_idx:03d}/{len(chunk_starts):03d} layer {layer_idx:02d}/{config.num_hidden_layers - 1:02d} ({layer_type})",
                    flush=True,
                )
                raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
                tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
                del raw_tensors

                residual = hidden_states
                hidden_norm = rms_norm_qwen3_next(hidden_states, tensors["input_layernorm.weight"], config.rms_norm_eps)
                if layer_type == "linear_attention":
                    mixed = linear_attention_forward(hidden_norm, {
                        "A_log": tensors["linear_attn.A_log"],
                        "conv1d.weight": tensors["linear_attn.conv1d.weight"],
                        "dt_bias": tensors["linear_attn.dt_bias"],
                        "in_proj_a.weight": tensors["linear_attn.in_proj_a.weight"],
                        "in_proj_b.weight": tensors["linear_attn.in_proj_b.weight"],
                        "in_proj_qkv.weight": tensors["linear_attn.in_proj_qkv.weight"],
                        "in_proj_z.weight": tensors["linear_attn.in_proj_z.weight"],
                        "norm.weight": tensors["linear_attn.norm.weight"],
                        "out_proj.weight": tensors["linear_attn.out_proj.weight"],
                    }, config)
                else:
                    mixed = full_attention_forward(hidden_norm, {
                        "q_norm.weight": tensors["self_attn.q_norm.weight"],
                        "k_norm.weight": tensors["self_attn.k_norm.weight"],
                        "q_proj.weight": tensors["self_attn.q_proj.weight"],
                        "k_proj.weight": tensors["self_attn.k_proj.weight"],
                        "v_proj.weight": tensors["self_attn.v_proj.weight"],
                        "o_proj.weight": tensors["self_attn.o_proj.weight"],
                    }, config, position_embeddings, causal_mask)

                hidden_states = residual + mixed
                mlp_input = rms_norm_qwen3_next(hidden_states, tensors["post_attention_layernorm.weight"], config.rms_norm_eps)
                mlp_out, _ = moe_forward_with_plan(
                    mlp_input,
                    {
                        "experts.gate_up_proj": tensors["mlp.experts.gate_up_proj"],
                        "experts.down_proj": tensors["mlp.experts.down_proj"],
                        "gate.weight": tensors["mlp.gate.weight"],
                        "shared_expert.gate_proj.weight": tensors["mlp.shared_expert.gate_proj.weight"],
                        "shared_expert.up_proj.weight": tensors["mlp.shared_expert.up_proj.weight"],
                        "shared_expert.down_proj.weight": tensors["mlp.shared_expert.down_proj.weight"],
                        "shared_expert_gate.weight": tensors["mlp.shared_expert_gate.weight"],
                    },
                    config,
                    plan,
                    layer_idx,
                )
                hidden_states = hidden_states + mlp_out
                release_tensors(tensors)

            hidden_states = rms_norm_qwen3_next(hidden_states, final_norm, config.rms_norm_eps)
            targets = chunk_ids[:, 1:].to(device)
            for start in range(0, seq_len - 1, logit_chunk_size):
                end = min(seq_len - 1, start + logit_chunk_size)
                logits = F.linear(hidden_states[:, start:end, :], lm_head)
                loss_sum = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    targets[:, start:end].reshape(-1),
                    reduction="sum",
                )
                total_nll += float(loss_sum.item())
                total_scored_tokens += int(targets[:, start:end].numel())
                del logits

            del chunk_ids, hidden_states, targets, position_ids, causal_mask, rotary, position_embeddings
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    release_tensors({"embed_weight": embed_weight, "final_norm": final_norm, "lm_head": lm_head})
    avg_nll = total_nll / max(total_scored_tokens, 1)
    return avg_nll, float(math.exp(avg_nll))


def render_markdown_table(rows: list[tuple[str, float, float]]) -> list[str]:
    lines = ["| Method | PPL | Memory (GB) |", "|--------|-----|-------------|"]
    for name, ppl, memory_gb in rows:
        lines.append(f"| {name} | {ppl:.4f} | {memory_gb:.3f} |")
    return lines


def render_exploration_section(payload: dict[str, Any]) -> str:
    related = payload.get("related_work", {})
    alternatives = payload.get("alternative_strategies", {})
    related_rows = [
        ("MxMoE-style per-block", related["mxmoe_per_block"]["ppl"], related["mxmoe_per_block"]["memory_gb"]),
        ("DynaExq per-expert", related["dynaexq_per_expert"]["ppl"], related["dynaexq_per_expert"]["memory_gb"]),
        ("FGMP per-16-block", related["fgmp_per_16block"]["ppl"], related["fgmp_per_16block"]["memory_gb"]),
        ("ScaleBITS submodular", related["scalebits_submodular"]["ppl"], related["scalebits_submodular"]["memory_gb"]),
        ("Random baseline (3-seed mean)", related["random_baseline"]["ppl"], related["random_baseline"]["memory_gb"]),
        ("Our two-level hierarchy", related["ours_two_level"]["ppl"], related["ours_two_level"]["memory_gb"]),
    ]
    alt_rows = [
        ("Threshold @ median", alternatives["threshold_median"]["ppl"], alternatives["threshold_median"]["memory_gb"]),
        ("Threshold @ p75", alternatives["threshold_p75"]["ppl"], alternatives["threshold_p75"]["memory_gb"]),
        ("Threshold @ p90", alternatives["threshold_p90"]["ppl"], alternatives["threshold_p90"]["memory_gb"]),
        ("K-means 2-cluster", alternatives["kmeans_2cluster"]["ppl"], alternatives["kmeans_2cluster"]["memory_gb"]),
        ("Sensitivity gap", alternatives["sensitivity_gap"]["ppl"], alternatives["sensitivity_gap"]["memory_gb"]),
    ]
    baseline_related_items = [(name, value) for name, value in related.items() if name != "ours_two_level"]
    best_related_name, best_related = min(baseline_related_items, key=lambda item: float(item[1]["ppl"]))
    best_alt_name, best_alt = min(alternatives.items(), key=lambda item: float(item[1]["ppl"]))

    lines = [SECTION_MARKER, ""]
    lines.append(
        "**Approach**: Reused the Iteration 2 streamed perplexity loop, Spike 1 quantizers, and the gate-aware activation metric, then added two follow-ups: a related-work simulation at the 25% average-FP8 budget and a set of unconstrained alternative assignment rules. Calibration remains a single 128-token WikiText-2 train pass, evaluation stays on the first 512 WikiText-2 test tokens, and all methods stream one layer at a time so the full model is never resident at once."
    )
    lines.append("")
    lines.append(
        f"**Related work comparison**: The strongest prior-work baseline in this run is `{best_related_name}` at PPL {best_related['ppl']:.4f} and {best_related['memory_gb']:.3f} GB, while our reused two-level hierarchy stays at PPL {related['ours_two_level']['ppl']:.4f} and {related['ours_two_level']['memory_gb']:.3f} GB."
    )
    lines.append("")
    lines.extend(render_markdown_table(related_rows))
    lines.append("")
    lines.append(
        f"**Alternative strategies**: The best unconstrained strategy here is `{best_alt_name}` at PPL {best_alt['ppl']:.4f} and {best_alt['memory_gb']:.3f} GB. Threshold rules expose how much the score distribution itself wants to spend, the k-means split tests adaptive per-expert cluster sizes, and the gap rule checks whether sharp elbows exist in the score spectra."
    )
    lines.append("")
    lines.extend(render_markdown_table(alt_rows))
    lines.append("")
    lines.append(
        "**Verdict**: This iteration answers two practical questions: whether the gain of the two-level hierarchy survives comparison against prior mixed-precision assignment ideas, and whether a different decision rule on the same gate-aware signal can outperform simple top-k routing-aware allocation. The related-work rows isolate granularity and allocation policy effects, while the alternative rows show whether the score distribution prefers fixed-budget, thresholded, clustered, or gap-based splits."
    )
    return "\n".join(lines)


def upsert_exploration_section(path: Path, section_body: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if SECTION_MARKER in existing:
        prefix = existing.split(SECTION_MARKER, maxsplit=1)[0].rstrip()
        updated = f"{prefix}\n\n{section_body}\n"
    else:
        updated = existing.rstrip()
        if updated:
            updated += "\n\n---\n\n"
        updated += f"{section_body}\n"
    path.write_text(updated, encoding="utf-8")


def summarize_random_results(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    ppl_values = [float(row["ppl"]) for row in seed_results]
    avg_nll_values = [float(row["avg_nll"]) for row in seed_results]
    memory_values = [float(row["memory_gb"]) for row in seed_results]
    return {
        "avg_nll": sum(avg_nll_values) / len(avg_nll_values),
        "ppl": sum(ppl_values) / len(ppl_values),
        "memory_gb": sum(memory_values) / len(memory_values),
        "seed_results": seed_results,
    }


def main() -> None:
    args = parse_args()
    start_time = time.time()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    calibration_ids, calibration_info = load_prefix_dataset_tokens(tokenizer, "train", args.calibration_tokens)
    eval_ids, eval_info = load_prefix_dataset_tokens(tokenizer, "test", args.eval_tokens)
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    config = build_text_config(root_config)
    store = WeightStore(args.model_id, snapshot_dir, weight_map)
    total_bf16_bytes = int(root_config.get("total_size", 0) or load_json(snapshot_dir / "model.safetensors.index.json")["metadata"]["total_size"])
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, config)

    payload = empty_payload(args.model_id, calibration_info, eval_info, args.fgmp_samples_per_layer)
    if args.output_json.exists():
        try:
            payload = load_json(args.output_json)
            print(f"[resume] loaded existing results from {args.output_json}", flush=True)
        except Exception:
            print(f"[resume] ignoring unreadable existing file at {args.output_json}", flush=True)

    iteration2_payload: dict[str, Any] | None = None
    if args.iteration2_json.exists():
        try:
            iteration2_payload = load_json(args.iteration2_json)
            print(f"[reuse] loaded iteration2 results from {args.iteration2_json}", flush=True)
        except Exception:
            iteration2_payload = None

    print("[calibration] collecting routed activations", flush=True)
    captures = run_calibration_forward(store, config, calibration_ids, device, dtype)
    print("[metrics] computing W1/W2 gate-aware scores", flush=True)
    metric_cache = compute_gate_aware_projection_metrics(store, config, captures, device, dtype)

    related_queue: list[tuple[str, EvaluationPlan | None, dict[str, Any] | None]] = []
    alternative_queue: list[tuple[str, EvaluationPlan]] = []

    reuse_expert_only = maybe_reuse_iteration2_result(
        iteration2_payload,
        "expert_only",
        args.calibration_tokens,
        args.eval_tokens,
    )
    reuse_two_level = maybe_reuse_iteration2_result(
        iteration2_payload,
        "two_level",
        args.calibration_tokens,
        args.eval_tokens,
    )
    if reuse_expert_only is not None:
        related_queue.append(("dynaexq_per_expert", None, reuse_expert_only))
    else:
        expert_only_plan = build_iteration2_expert_only_plan(metric_cache, config, non_expert_bytes, total_expert_elems, TARGET_FP8_FRACTION)
        related_queue.append(("dynaexq_per_expert", EvaluationPlan(
            name="dynaexq_per_expert",
            description=expert_only_plan.description,
            mode=expert_only_plan.mode,
            memory_gb=expert_only_plan.memory_gb,
            fp8_weights=expert_only_plan.fp8_weights,
            promoted_experts=expert_only_plan.promoted_experts,
        ), None))
    if reuse_two_level is not None:
        related_queue.append(("ours_two_level", None, reuse_two_level))
    else:
        two_level_plan = build_iteration2_projection_plan(
            "ours_two_level",
            "Freq-weighted per-channel 25% FP8 in W1 and 25% in W2.",
            metric_cache,
            config,
            non_expert_bytes,
            total_expert_elems,
            "freq_weighted",
            0.25,
            0.25,
        )
        related_queue.append((
            "ours_two_level",
            EvaluationPlan(
                name="ours_two_level",
                description=two_level_plan.description,
                mode=two_level_plan.mode,
                memory_gb=two_level_plan.memory_gb,
                fp8_weights=two_level_plan.fp8_weights,
                w1_pair_masks=two_level_plan.w1_pair_masks,
                w2_channel_masks=two_level_plan.w2_channel_masks,
            ),
            None,
        ))

    related_queue.extend(
        [
            ("mxmoe_per_block", build_mxmoe_plan(metric_cache, config, non_expert_bytes, total_expert_elems), None),
            ("fgmp_per_16block", build_fgmp_plan(store, config, non_expert_bytes, total_expert_elems, args.fgmp_samples_per_layer), None),
            ("scalebits_submodular", build_scalebits_plan(metric_cache, config, non_expert_bytes, total_expert_elems), None),
        ]
    )

    alternative_queue.extend(
        [
            ("threshold_median", build_global_threshold_plan("threshold_median", "Global median threshold over combined W1/W2 gate-aware scores.", config, non_expert_bytes, total_expert_elems, metric_cache, 50.0)),
            ("threshold_p75", build_global_threshold_plan("threshold_p75", "Global p75 threshold over combined W1/W2 gate-aware scores.", config, non_expert_bytes, total_expert_elems, metric_cache, 75.0)),
            ("threshold_p90", build_global_threshold_plan("threshold_p90", "Global p90 threshold over combined W1/W2 gate-aware scores.", config, non_expert_bytes, total_expert_elems, metric_cache, 90.0)),
            ("kmeans_2cluster", build_kmeans_plan(metric_cache, config, non_expert_bytes, total_expert_elems)),
            ("sensitivity_gap", build_gap_plan(metric_cache, config, non_expert_bytes, total_expert_elems)),
        ]
    )

    random_seed_results: list[dict[str, Any]] = []
    if maybe_get_saved_result(payload, "related_work", "random_baseline") is None:
        for seed in DEFAULT_RANDOM_SEEDS:
            random_plan = build_random_plan(config, non_expert_bytes, total_expert_elems, seed)
            print(
                f"\n=== Running related_work:random_seed_{seed} | {random_plan.description} | memory~{random_plan.memory_gb:.3f} GB ===",
                flush=True,
            )
            avg_nll, ppl = evaluate_chunked_perplexity(
                store,
                config,
                eval_ids,
                device,
                dtype,
                args.logit_chunk_size,
                args.eval_chunk_tokens,
                random_plan,
            )
            seed_result = {
                "seed": seed,
                "avg_nll": avg_nll,
                "ppl": ppl,
                "memory_gb": random_plan.memory_gb,
            }
            random_seed_results.append(seed_result)
            payload.setdefault("metadata", {})["runtime_seconds"] = round(time.time() - start_time, 3)
            atomic_json_dump(args.output_json, payload)
            print(f"[done] related_work:random_seed_{seed} -> ppl={ppl:.4f} memory={random_plan.memory_gb:.3f} GB", flush=True)
        random_summary = summarize_random_results(random_seed_results)
        upsert_result(payload, "related_work", "random_baseline", random_summary)
        atomic_json_dump(args.output_json, payload)

    for key, plan, reused in related_queue:
        if maybe_get_saved_result(payload, "related_work", key) is not None:
            print(f"[skip] related_work:{key} already present", flush=True)
            continue
        if reused is not None:
            print(f"[reuse] related_work:{key} <- iteration2", flush=True)
            upsert_result(payload, "related_work", key, reused)
            atomic_json_dump(args.output_json, payload)
            continue
        if plan is None:
            raise RuntimeError(f"Missing plan for {key}")
        print(
            f"\n=== Running related_work:{key} | {plan.description} | memory~{plan.memory_gb:.3f} GB ===",
            flush=True,
        )
        avg_nll, ppl = evaluate_chunked_perplexity(
            store,
            config,
            eval_ids,
            device,
            dtype,
            args.logit_chunk_size,
            args.eval_chunk_tokens,
            plan,
        )
        upsert_result(payload, "related_work", key, {"avg_nll": avg_nll, "ppl": ppl, "memory_gb": plan.memory_gb})
        payload.setdefault("metadata", {})["runtime_seconds"] = round(time.time() - start_time, 3)
        atomic_json_dump(args.output_json, payload)
        print(f"[done] related_work:{key} -> ppl={ppl:.4f} memory={plan.memory_gb:.3f} GB", flush=True)

    for key, plan in alternative_queue:
        if maybe_get_saved_result(payload, "alternative_strategies", key) is not None:
            print(f"[skip] alternative_strategies:{key} already present", flush=True)
            continue
        print(
            f"\n=== Running alternative_strategies:{key} | {plan.description} | memory~{plan.memory_gb:.3f} GB ===",
            flush=True,
        )
        avg_nll, ppl = evaluate_chunked_perplexity(
            store,
            config,
            eval_ids,
            device,
            dtype,
            args.logit_chunk_size,
            args.eval_chunk_tokens,
            plan,
        )
        upsert_result(payload, "alternative_strategies", key, {"avg_nll": avg_nll, "ppl": ppl, "memory_gb": plan.memory_gb})
        payload.setdefault("metadata", {})["runtime_seconds"] = round(time.time() - start_time, 3)
        atomic_json_dump(args.output_json, payload)
        print(f"[done] alternative_strategies:{key} -> ppl={ppl:.4f} memory={plan.memory_gb:.3f} GB", flush=True)

    section = render_exploration_section(payload)
    upsert_exploration_section(args.exploration_md, section)
    payload.setdefault("metadata", {})["runtime_seconds"] = round(time.time() - start_time, 3)
    atomic_json_dump(args.output_json, payload)
    print(f"Saved results -> {args.output_json}", flush=True)
    print(f"Updated exploration log -> {args.exploration_md}", flush=True)


if __name__ == "__main__":
    main()

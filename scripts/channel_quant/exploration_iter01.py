#!/usr/bin/env python3
# pyright: basic, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from baselines_comparison import (
    DEFAULT_MINI_BATCH_TOKENS,
    EvaluationPlan,
    LayerMetricBundle,
    build_two_level_masks,
    compute_mc_moe_importance,
    compute_mxmoe_projection_deltas,
    estimate_mixed_memory_gb,
    evaluate_plan,
    fp8_weights_from_masks,
    load_prefix_dataset_tokens,
    quantize_linear_weight,
    resolve_non_expert_bytes,
    run_calibration_pass,
)
from spike1_ground_truth import (
    MODEL_ID,
    WeightStore,
    build_text_config,
    load_root_config,
    move_tensor,
    release_tensors,
)

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "exploration_iter01.json"
DEFAULT_EXPLORATION_MD = SCRIPT_DIR / "exploration.md"
DEFAULT_CALIBRATION_TOKENS = 128
DEFAULT_EVAL_TOKENS = 512
DEFAULT_TARGET_LAYERS = (5, 20, 35)
DEFAULT_TOP_HOT_EXPERTS = 3
TARGET_FP8_FRACTION = 0.25
EPS = 1e-10
CURRENT_BEST_PPL = 5.3374
CURRENT_BEST_MEMORY_GB = 27.6
EXPLORATION_MARKER = "## [10] Full Sensitivity Metric Shootout"

CHANNEL_METRICS = (
    "weight_l1",
    "weight_kurtosis",
    "activation_magnitude",
    "activation_variance",
    "hessian_diag",
    "slim_salience",
    "gate_aware_act_qerror",
    "activation_kurtosis",
    "output_perturbation",
)
PROXY_CHANNEL_METRICS = tuple(metric for metric in CHANNEL_METRICS if metric != "output_perturbation")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--calibration-tokens", type=int, default=DEFAULT_CALIBRATION_TOKENS)
    parser.add_argument("--eval-tokens", type=int, default=DEFAULT_EVAL_TOKENS)
    parser.add_argument("--mini-batch-tokens", type=int, default=DEFAULT_MINI_BATCH_TOKENS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--exploration-md", type=Path, default=DEFAULT_EXPLORATION_MD)
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


def expert_label(layer_idx: int, expert_idx: int) -> str:
    return f"L{layer_idx:02d}_E{expert_idx:03d}"


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")
    if left.size == 0:
        return 0.0
    if spearmanr is not None:
        result = spearmanr(left, right)
        value_raw = getattr(result, "statistic", result[0])
        if value_raw is None or not np.isscalar(value_raw):
            return 0.0
        value_arr = np.asarray(value_raw, dtype=np.float64).reshape(-1)
        if int(value_arr.size) != 1:
            return 0.0
        value = float(value_arr[0])
        if not np.isfinite(value):
            return 0.0
        return value
    left_rank = left.argsort().argsort().astype(np.float64)
    right_rank = right.argsort().argsort().astype(np.float64)
    if float(left_rank.std()) <= 0.0 or float(right_rank.std()) <= 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def gini(values: np.ndarray) -> float:
    sorted_values = np.sort(np.asarray(values, dtype=np.float64))
    if sorted_values.size == 0:
        return 0.0
    total = float(sorted_values.sum())
    if total <= 0.0:
        return 0.0
    index = np.arange(1, sorted_values.size + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * sorted_values) / (sorted_values.size * total)) - ((sorted_values.size + 1.0) / sorted_values.size))


def vector_kurtosis(values: torch.Tensor, dim: int) -> torch.Tensor:
    values_f = values.float()
    if values_f.shape[dim] <= 1:
        shape = list(values_f.shape)
        del shape[dim]
        return torch.zeros(shape, dtype=torch.float32, device=values_f.device)
    mean = values_f.mean(dim=dim, keepdim=True)
    centered = values_f - mean
    var = centered.square().mean(dim=dim)
    central4 = centered.pow(4).mean(dim=dim)
    kurt = central4 / (var.square() + EPS)
    kurt = torch.where(torch.isfinite(kurt), kurt, torch.zeros_like(kurt))
    return kurt.to(torch.float32)


def target_hot_experts(captures: dict[int, Any]) -> dict[int, list[int]]:
    chosen: dict[int, list[int]] = {}
    for layer_idx in DEFAULT_TARGET_LAYERS:
        capture = captures[layer_idx]
        active = torch.nonzero(capture.expert_counts > 0, as_tuple=False).flatten().tolist()
        ranked = sorted(active, key=lambda expert_idx: (-int(capture.expert_counts[expert_idx]), expert_idx))
        chosen[layer_idx] = ranked[:DEFAULT_TOP_HOT_EXPERTS]
    return chosen


def collect_metric_bundles(
    store: WeightStore,
    config: Any,
    captures: dict[int, Any],
    gt_targets: dict[int, list[int]],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, dict[int, LayerMetricBundle]], dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    print("\n=== Phase B: All channel metrics ===", flush=True)
    metric_bundles: dict[str, dict[int, LayerMetricBundle]] = {metric: {} for metric in CHANNEL_METRICS}
    output_perturbation_gt: dict[str, torch.Tensor] = {}
    gt_metadata: dict[str, dict[str, Any]] = {}

    for layer_idx in range(config.num_hidden_layers):
        capture = captures[layer_idx]
        gate_key = f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"
        down_key = f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj"
        print(f"[metrics] layer {layer_idx:02d}/{config.num_hidden_layers - 1:02d}", flush=True)
        raw = store.load_tensors([gate_key, down_key])
        gate_up_proj = move_tensor(raw[gate_key], device, dtype)
        down_proj = move_tensor(raw[down_key], device, dtype)
        del raw

        layer_w1_scores = {
            metric: torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32)
            for metric in PROXY_CHANNEL_METRICS
        }
        layer_w2_scores = {
            metric: torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32)
            for metric in PROXY_CHANNEL_METRICS
        }
        layer_output_gt = torch.zeros((config.num_experts, config.hidden_size), dtype=torch.float32)
        active_experts = torch.nonzero(capture.expert_counts > 0, as_tuple=False).flatten().tolist()

        for expert_pos, expert_idx in enumerate(active_experts, start=1):
            if expert_pos == 1 or expert_pos % 16 == 0 or expert_pos == len(active_experts):
                print(f"  [metrics] expert {expert_pos}/{len(active_experts)} (E{expert_idx})", flush=True)
            token_idx, route_pos = torch.where(capture.selected_experts == expert_idx)
            if int(token_idx.numel()) == 0:
                continue
            route_weights = capture.routing_weights[token_idx, route_pos].to(device=device, dtype=torch.float32)
            denom = route_weights.sum().clamp_min(EPS)
            x_in = capture.mlp_input[token_idx].to(device=device, dtype=dtype)
            x_in_f = x_in.float()

            mean_abs_x = x_in_f.abs().mean(dim=0)
            var_x = x_in_f.var(dim=0, correction=0) if int(x_in_f.shape[0]) > 1 else torch.zeros_like(mean_abs_x)
            second_x = x_in_f.square().mean(dim=0)
            abs_mean_x = x_in_f.mean(dim=0).abs()
            kurt_x = vector_kurtosis(x_in_f, dim=0)
            weighted_abs_x = (x_in_f.abs() * route_weights.unsqueeze(-1)).sum(dim=0) / denom

            gate_up_weight = gate_up_proj[expert_idx]
            gate_up_fp4 = quantize_linear_weight(gate_up_weight, "fp4")
            gate_weight = gate_up_weight[: config.moe_intermediate_size].float()
            up_weight = gate_up_weight[config.moe_intermediate_size :].float()
            gate_diff = gate_weight - gate_up_fp4[: config.moe_intermediate_size].float()
            up_diff = up_weight - gate_up_fp4[config.moe_intermediate_size :].float()
            pair_abs = gate_weight.abs() + up_weight.abs()
            pair_diff_abs = gate_diff.abs() + up_diff.abs()
            pair_diff_sq = gate_diff.square() + up_diff.square()
            pair_values = torch.cat([gate_weight, up_weight], dim=1)

            layer_w1_scores["weight_l1"][expert_idx] = pair_abs.sum(dim=1).cpu()
            layer_w1_scores["weight_kurtosis"][expert_idx] = vector_kurtosis(pair_values, dim=1).cpu()
            layer_w1_scores["activation_magnitude"][expert_idx] = torch.matmul(pair_abs, mean_abs_x).cpu()
            layer_w1_scores["activation_variance"][expert_idx] = torch.matmul(pair_abs, var_x).cpu()
            layer_w1_scores["hessian_diag"][expert_idx] = torch.matmul(pair_diff_sq, second_x).cpu()
            layer_w1_scores["slim_salience"][expert_idx] = torch.matmul(pair_abs, abs_mean_x).cpu()
            layer_w1_scores["gate_aware_act_qerror"][expert_idx] = torch.matmul(pair_diff_abs, weighted_abs_x).cpu()
            layer_w1_scores["activation_kurtosis"][expert_idx] = torch.matmul(pair_diff_abs, kurt_x).cpu()

            gate_up = F.linear(x_in, gate_up_weight)
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = (F.silu(gate.float()) * up.float()).to(torch.float32)

            mean_abs_inter = intermediate.abs().mean(dim=0)
            var_inter = intermediate.var(dim=0, correction=0) if int(intermediate.shape[0]) > 1 else torch.zeros_like(mean_abs_inter)
            second_inter = intermediate.square().mean(dim=0)
            abs_mean_inter = intermediate.mean(dim=0).abs()
            kurt_inter = vector_kurtosis(intermediate, dim=0)
            weighted_abs_inter = (intermediate.abs() * route_weights.unsqueeze(-1)).sum(dim=0) / denom

            down_weight = down_proj[expert_idx]
            down_fp4 = quantize_linear_weight(down_weight, "fp4")
            down_weight_f = down_weight.float()
            down_diff = down_weight_f - down_fp4.float()
            down_abs = down_weight_f.abs()
            down_diff_abs = down_diff.abs()
            down_diff_sq = down_diff.square()

            layer_w2_scores["weight_l1"][expert_idx] = down_abs.sum(dim=1).cpu()
            layer_w2_scores["weight_kurtosis"][expert_idx] = vector_kurtosis(down_weight_f, dim=1).cpu()
            layer_w2_scores["activation_magnitude"][expert_idx] = torch.matmul(down_abs, mean_abs_inter).cpu()
            layer_w2_scores["activation_variance"][expert_idx] = torch.matmul(down_abs, var_inter).cpu()
            layer_w2_scores["hessian_diag"][expert_idx] = torch.matmul(down_diff_sq, second_inter).cpu()
            layer_w2_scores["slim_salience"][expert_idx] = torch.matmul(down_abs, abs_mean_inter).cpu()
            layer_w2_scores["gate_aware_act_qerror"][expert_idx] = torch.matmul(down_diff_abs, weighted_abs_inter).cpu()
            layer_w2_scores["activation_kurtosis"][expert_idx] = torch.matmul(down_diff_abs, kurt_inter).cpu()

            if expert_idx in gt_targets.get(layer_idx, []):
                gt_delta = F.linear(intermediate, down_weight_f - down_fp4.float()).float()
                gt_scores = torch.linalg.vector_norm(gt_delta, dim=0).cpu()
                label = expert_label(layer_idx, expert_idx)
                layer_output_gt[expert_idx] = gt_scores
                output_perturbation_gt[label] = gt_scores
                gt_metadata[label] = {
                    "layer_idx": int(layer_idx),
                    "expert_idx": int(expert_idx),
                    "token_count": int(token_idx.numel()),
                    "activation_count": int(capture.expert_counts[expert_idx].item()),
                }

            del route_weights, x_in, x_in_f, gate_up_weight, gate_up_fp4, gate_weight, up_weight
            del gate_diff, up_diff, pair_abs, pair_diff_abs, pair_diff_sq, pair_values
            del gate_up, gate, up, intermediate, mean_abs_inter, var_inter, second_inter, abs_mean_inter, kurt_inter, weighted_abs_inter
            del down_weight, down_fp4, down_weight_f, down_diff, down_abs, down_diff_abs, down_diff_sq
            gc.collect()

        for metric_name in PROXY_CHANNEL_METRICS:
            metric_bundles[metric_name][layer_idx] = LayerMetricBundle(
                routing_counts=capture.expert_counts.clone(),
                w1_pair_scores=layer_w1_scores[metric_name],
                w2_channel_scores=layer_w2_scores[metric_name],
            )
        metric_bundles["output_perturbation"][layer_idx] = LayerMetricBundle(
            routing_counts=capture.expert_counts.clone(),
            w1_pair_scores=torch.zeros((config.num_experts, config.moe_intermediate_size), dtype=torch.float32),
            w2_channel_scores=layer_output_gt,
        )
        release_tensors({"gate_up_proj": gate_up_proj, "down_proj": down_proj})

    return metric_bundles, output_perturbation_gt, gt_metadata


def summarize_channel_metrics(
    metric_bundles: dict[str, dict[int, LayerMetricBundle]],
    captures: dict[int, Any],
    output_perturbation_gt: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics_summary: dict[str, Any] = {}
    ranking: list[dict[str, Any]] = []

    for metric_name in CHANNEL_METRICS:
        gini_per_expert: dict[str, float] = {}
        spearman_per_expert: dict[str, float] = {}
        for layer_idx, bundle in metric_bundles[metric_name].items():
            active_experts = torch.nonzero(captures[layer_idx].expert_counts > 0, as_tuple=False).flatten().tolist()
            for expert_idx in active_experts:
                label = expert_label(layer_idx, expert_idx)
                scores = bundle.w2_channel_scores[expert_idx].numpy().astype(np.float64)
                if metric_name == "output_perturbation" and label not in output_perturbation_gt:
                    continue
                gini_per_expert[label] = gini(scores)
                if label in output_perturbation_gt:
                    target = output_perturbation_gt[label].numpy().astype(np.float64)
                    spearman_per_expert[label] = 1.0 if metric_name == "output_perturbation" else safe_spearman(scores, target)
        mean_gini = float(np.mean(list(gini_per_expert.values()), dtype=np.float64)) if gini_per_expert else None
        mean_spearman = float(np.mean(list(spearman_per_expert.values()), dtype=np.float64)) if spearman_per_expert else None
        metrics_summary[metric_name] = {
            "scope": "channel",
            "mean_gini": mean_gini,
            "mean_spearman": mean_spearman,
            "gini_per_expert": gini_per_expert,
            "spearman_per_expert": spearman_per_expert,
        }
        ranking.append({
            "metric": metric_name,
            "mean_spearman": mean_spearman,
            "mean_gini": mean_gini,
        })

    ranking.sort(key=lambda row: (float("inf") if row["mean_spearman"] is None else -float(row["mean_spearman"]), row["metric"]))
    return metrics_summary, ranking


def summarize_mxmoe_metric(
    w1_deltas: dict[int, torch.Tensor],
    w2_deltas: dict[int, torch.Tensor],
    captures: dict[int, Any],
) -> dict[str, Any]:
    per_expert: dict[str, dict[str, float]] = {}
    values: list[float] = []
    for layer_idx, capture in captures.items():
        active_experts = torch.nonzero(capture.expert_counts > 0, as_tuple=False).flatten().tolist()
        for expert_idx in active_experts:
            label = expert_label(layer_idx, expert_idx)
            w1_value = float(w1_deltas[layer_idx][expert_idx].item())
            w2_value = float(w2_deltas[layer_idx][expert_idx].item())
            per_expert[label] = {"w1": w1_value, "w2": w2_value}
            values.extend([w1_value, w2_value])
    return {
        "scope": "projection",
        "mean_gini": None,
        "mean_spearman": None,
        "gini_per_expert": {},
        "projection_delta_per_expert": per_expert,
        "mean_projection_delta": float(np.mean(values, dtype=np.float64)) if values else 0.0,
    }


def summarize_mc_moe_metric(mc_scores: dict[int, torch.Tensor], captures: dict[int, Any]) -> dict[str, Any]:
    per_expert: dict[str, float] = {}
    values: list[float] = []
    for layer_idx, capture in captures.items():
        active_experts = torch.nonzero(capture.expert_counts > 0, as_tuple=False).flatten().tolist()
        for expert_idx in active_experts:
            label = expert_label(layer_idx, expert_idx)
            value = float(mc_scores[layer_idx][expert_idx].item())
            per_expert[label] = value
            values.append(value)
    return {
        "scope": "expert",
        "mean_gini": None,
        "mean_spearman": None,
        "gini_per_expert": {},
        "expert_scores": per_expert,
        "mean_expert_importance": float(np.mean(values, dtype=np.float64)) if values else 0.0,
    }


def top_proxy_metrics(metric_ranking: list[dict[str, Any]], top_k: int = 3) -> list[str]:
    ordered = [
        row["metric"]
        for row in metric_ranking
        if row["metric"] in PROXY_CHANNEL_METRICS and row["mean_spearman"] is not None
    ]
    return ordered[:top_k]


def build_eval_plan_for_metric(
    metric_name: str,
    metric_bundle_map: dict[int, LayerMetricBundle],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> EvaluationPlan:
    w1_pair_masks, w2_channel_masks = build_two_level_masks(metric_bundle_map, config)
    fp8_weights = fp8_weights_from_masks(config, w1_pair_masks, w2_channel_masks)
    memory_gb = estimate_mixed_memory_gb(non_expert_bytes, total_expert_elems, fp8_weights)
    return EvaluationPlan(
        name=f"iter01_{metric_name}",
        description=f"Two-level routing-aware sort-and-split using {metric_name}.",
        mode="mixed_channel",
        memory_gb=memory_gb,
        fp8_weights=fp8_weights,
        w1_pair_masks=w1_pair_masks,
        w2_channel_masks=w2_channel_masks,
    )


def build_exploration_section(results: dict[str, Any]) -> str:
    ranking = results["metric_ranking"]
    best_proxy = next((row for row in ranking if row["metric"] in PROXY_CHANNEL_METRICS), None)
    best_metric = str(results["best_metric"])
    best_ppl = float(results["best_ppl"])
    top_names = ", ".join(item["metric_name"] for item in results["ppl_results"].values())
    best_proxy_text = "n/a"
    if best_proxy is not None:
        best_proxy_text = f"{best_proxy['metric']} (mean rho={float(best_proxy['mean_spearman']):.4f}, mean Gini={float(best_proxy['mean_gini']):.4f})"
    beat_text = "beats" if best_ppl < CURRENT_BEST_PPL else "does not beat"
    return "\n".join([
        EXPLORATION_MARKER,
        "**Approach**: Tested 11 sensitivity metrics in one shared 128-token WikiText-2 train calibration pass, computed per-channel W1/W2 proxy scores for every active expert, used W2 output perturbation on 9 hot experts as ground truth, then reran 512-token WikiText-2 test PPL with the top-3 proxy metrics under the same two-level routing-aware sort-and-split eval used by `baselines_comparison.py`.",
        f"**Result**: The strongest proxy by mean Spearman is {best_proxy_text}. The top PPL sweep evaluated {top_names}; the best end-to-end result is `{best_metric}` at PPL {best_ppl:.4f} and {results['ppl_results'][next(key for key, value in results['ppl_results'].items() if value['metric_name'] == best_metric)]['memory_gb']:.3f} GB, which {beat_text} the current MxMoE per-block baseline (PPL {CURRENT_BEST_PPL:.4f} at {CURRENT_BEST_MEMORY_GB:.1f} GB).",
        "**Insight**: Channel discrimination and end-to-end perplexity are not identical objectives: the best ground-truth-correlated proxy is the right candidate set, but routing-aware expert budgeting still determines whether the within-expert ranking gain survives at fixed memory.",
        "**Next**: Take the best proxy from this shootout and sweep W1/W2 split ratios plus larger eval spans to see whether its gain is robust beyond the 512-token matched-budget comparison.",
        "",
    ])


def upsert_exploration_section(path: Path, section: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
    else:
        existing = ""
    if EXPLORATION_MARKER in existing:
        prefix = existing.split(EXPLORATION_MARKER, 1)[0].rstrip()
        updated = prefix + "\n\n" + section
    else:
        updated = existing.rstrip() + "\n\n" + section if existing.strip() else section
    path.write_text(updated.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    start_time = time.time()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    print("=== Full Sensitivity Metric Shootout ===", flush=True)
    print(f"Model: {args.model_id}", flush=True)
    print(f"Device: {device} | dtype: {dtype}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    calib_ids, calib_info = load_prefix_dataset_tokens(tokenizer, "train", args.calibration_tokens)
    eval_ids, eval_info = load_prefix_dataset_tokens(tokenizer, "test", args.eval_tokens)

    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)
    index_path = snapshot_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index_payload = json.load(handle)
    total_bf16_bytes = int(root_config.get("total_size", 0) or index_payload["metadata"]["total_size"])
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)

    payload: dict[str, Any] = {
        "metadata": {
            "model": args.model_id,
            "device": str(device),
            "dtype": args.dtype,
            "calibration": calib_info,
            "evaluation": eval_info,
            "calibration_tokens": int(calib_ids.shape[1]),
            "eval_tokens": int(eval_ids.shape[1]),
            "mini_batch_tokens": int(args.mini_batch_tokens),
            "target_fp8_fraction": TARGET_FP8_FRACTION,
            "target_layers_for_ground_truth": list(DEFAULT_TARGET_LAYERS),
            "ground_truth_experts_per_layer": DEFAULT_TOP_HOT_EXPERTS,
            "current_best_to_beat": {
                "name": "MxMoE per-block with output perturbation",
                "ppl": CURRENT_BEST_PPL,
                "memory_gb": CURRENT_BEST_MEMORY_GB,
            },
        },
        "metrics": {},
        "metric_ranking": [],
        "ppl_results": {},
        "best_metric": None,
        "best_ppl": None,
        "runtime_seconds": None,
    }
    atomic_json_dump(args.output_json, payload)

    store = WeightStore(args.model_id, snapshot_dir, weight_map)

    captures = run_calibration_pass(store, text_config, calib_ids, args.mini_batch_tokens, device, dtype)
    gt_targets = target_hot_experts(captures)
    payload["metadata"]["ground_truth_hot_experts"] = {
        str(layer_idx): [
            {"expert_id": int(expert_idx), "token_count": int(captures[layer_idx].expert_counts[expert_idx].item())}
            for expert_idx in expert_ids
        ]
        for layer_idx, expert_ids in gt_targets.items()
    }
    atomic_json_dump(args.output_json, payload)

    metric_bundles, output_perturbation_gt, gt_metadata = collect_metric_bundles(store, text_config, captures, gt_targets, device, dtype)
    payload["metadata"]["output_perturbation_targets"] = gt_metadata
    metrics_summary, metric_ranking = summarize_channel_metrics(metric_bundles, captures, output_perturbation_gt)

    mxmoe_w1_deltas, mxmoe_w2_deltas = compute_mxmoe_projection_deltas(store, text_config, captures, device, dtype)
    mc_moe_scores = compute_mc_moe_importance(store, text_config, captures, device, dtype)
    metrics_summary["mxmoe_block_perturbation"] = summarize_mxmoe_metric(mxmoe_w1_deltas, mxmoe_w2_deltas, captures)
    metrics_summary["mc_moe_importance"] = summarize_mc_moe_metric(mc_moe_scores, captures)

    payload["metrics"] = metrics_summary
    payload["metric_ranking"] = metric_ranking + [
        {
            "metric": "mxmoe_block_perturbation",
            "mean_spearman": None,
            "mean_gini": None,
        },
        {
            "metric": "mc_moe_importance",
            "mean_spearman": None,
            "mean_gini": None,
        },
    ]
    atomic_json_dump(args.output_json, payload)

    eval_metric_names = top_proxy_metrics(metric_ranking, top_k=3)
    print("\n=== Phase D: PPL with top proxy metrics ===", flush=True)
    for idx, metric_name in enumerate(eval_metric_names, start=1):
        print(f"[eval] {idx}/{len(eval_metric_names)} metric={metric_name}", flush=True)
        plan = build_eval_plan_for_metric(metric_name, metric_bundles[metric_name], text_config, non_expert_bytes, total_expert_elems)
        t0 = time.time()
        ppl, nll = evaluate_plan(plan, eval_ids, text_config, weight_map, snapshot_dir, device, dtype)
        elapsed = time.time() - t0
        payload["ppl_results"][f"top_metric_{idx}"] = {
            "metric_name": metric_name,
            "ppl": round(ppl, 6),
            "nll": round(nll, 6),
            "memory_gb": plan.memory_gb,
            "fp8_weights": int(plan.fp8_weights),
            "time_s": round(elapsed, 1),
        }
        atomic_json_dump(args.output_json, payload)
        print(f"  -> PPL={ppl:.4f} | NLL={nll:.6f} | memory={plan.memory_gb:.3f} GB | time={elapsed:.1f}s", flush=True)

    if payload["ppl_results"]:
        best_name, best_row = min(payload["ppl_results"].items(), key=lambda item: float(item[1]["ppl"]))
        del best_name
        payload["best_metric"] = best_row["metric_name"]
        payload["best_ppl"] = float(best_row["ppl"])
    payload["runtime_seconds"] = round(time.time() - start_time, 3)
    atomic_json_dump(args.output_json, payload)

    section = build_exploration_section(payload)
    upsert_exploration_section(args.exploration_md, section)
    print(f"\nSaved results -> {args.output_json}", flush=True)
    print(f"Updated log -> {args.exploration_md}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportImplicitStringConcatenation=false, reportReturnType=false, reportCallIssue=false, reportArgumentType=false
"""Compute Spike 2 proxy sensitivity metrics for Qwen3.5-35B-A3B MoE experts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


DEFAULT_REPO_ID = "Qwen/Qwen3.5-35B-A3B"
DEFAULT_REVISION = "ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"
DEFAULT_SNAPSHOT = Path(
    "/home/jerry/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/"
    "ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"
)
DEFAULT_OUTPUT = Path(
    "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/channel_quant/results/"
    "spike2_metrics.json"
)

HIDDEN_SIZE = 2048
MOE_INTERMEDIATE_SIZE = 512
NUM_EXPERTS = 256
NUM_LAYERS = 40
GROUP_SIZE = 16
DEFAULT_LAYERS = (5, 20, 35)
DEFAULT_NUM_TOKENS = 128

E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
E2M1_THRESHOLDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
METRIC_ORDER = [
    "weight_l2",
    "weight_l1",
    "weight_kurtosis",
    "weight_max_abs",
    "weight_variance",
    "within_group_var",
    "sinq_column_scale",
    "awq_activation",
    "smoothquant_max",
    "owq_hessian_diag",
    "slim_llm_salience",
    "owq_sensitivity",
]


@dataclass
class ModelPaths:
    repo_id: str
    revision: str
    snapshot_dir: Path


class Timer:

    def __init__(self) -> None:
        self.timings: dict[str, float] = defaultdict(float)

    def timed(self, name: str):
        class _Ctx:

            def __init__(self, outer: Timer, key: str) -> None:
                self.outer = outer
                self.key = key
                self.start = 0.0

            def __enter__(self) -> None:
                self.start = time.perf_counter()

            def __exit__(self, exc_type, exc, tb) -> None:
                self.outer.timings[self.key] += time.perf_counter() - self.start

        return _Ctx(self, name)


class LazySafetensorLoader:

    def __init__(self, model_paths: ModelPaths) -> None:
        self.model_paths = model_paths
        self.index_path = self._ensure_file("model.safetensors.index.json")
        with open(self.index_path, "r", encoding="utf-8") as handle:
            self.weight_map = json.load(handle)["weight_map"]
        self._handles: dict[str, Any] = {}

    def _ensure_file(self, filename: str) -> str:
        local_path = self.model_paths.snapshot_dir / filename
        if local_path.exists():
            return str(local_path)
        return hf_hub_download(
            repo_id=self.model_paths.repo_id,
            filename=filename,
            revision=self.model_paths.revision,
        )

    def get_tensor(self, key: str) -> torch.Tensor:
        filename = self.weight_map[key]
        resolved_path = self._ensure_file(filename)
        if resolved_path not in self._handles:
            self._handles[resolved_path] = safe_open(resolved_path, framework="pt", device="cpu")
        return self._handles[resolved_path].get_tensor(key)

    def has_complete_local_snapshot(self) -> bool:
        filenames = set(self.weight_map.values())
        return all((self.model_paths.snapshot_dir / filename).exists() for filename in filenames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo id")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Model revision / snapshot hash")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help="Preferred local snapshot directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(DEFAULT_LAYERS),
        help="MoE layer indices to analyze",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=DEFAULT_NUM_TOKENS,
        help="Calibration token count or Gaussian token count",
    )
    parser.add_argument(
        "--activation-source",
        choices=("auto", "real", "gaussian"),
        default="auto",
        help="Prefer real Wikitext2 activations or Gaussian fallback",
    )
    parser.add_argument(
        "--allow-full-model-download",
        action="store_true",
        help="Allow full-model download for real activation capture",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device",
    )
    return parser.parse_args()


def resolve_attr_path(root: Any, path: str) -> Any | None:
    current = root
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def tensor_to_lists(tensor: torch.Tensor) -> list[list[float]]:
    return tensor.detach().cpu().to(torch.float32).tolist()


def atomic_json_dump(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    tmp_path.replace(output_path)


def normalize_rms(x: torch.Tensor) -> torch.Tensor:
    rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    return x / rms


def gaussian_layer_inputs(
    layers: list[int],
    num_tokens: int,
    hidden_size: int,
    seed: int,
) -> dict[int, torch.Tensor]:
    inputs: dict[int, torch.Tensor] = {}
    for layer_idx in layers:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + layer_idx)
        inputs[layer_idx] = normalize_rms(
            torch.randn(num_tokens, hidden_size, generator=generator, dtype=torch.float32)
        )
    return inputs


def try_capture_real_inputs(
    args: argparse.Namespace,
    loader: LazySafetensorLoader,
) -> tuple[dict[int, torch.Tensor] | None, dict[str, Any]]:
    info: dict[str, Any] = {"source": "real", "attempted": True, "success": False}
    if args.activation_source == "gaussian":
        info["reason"] = "gaussian-forced"
        return None, info
    if importlib.util.find_spec("bitsandbytes") is None:
        info["reason"] = "bitsandbytes-not-installed"
        return None, info
    if not args.allow_full_model_download and not loader.has_complete_local_snapshot():
        info["reason"] = "local-snapshot-incomplete-and-full-download-disabled"
        return None, info
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer, BitsAndBytesConfig
        import transformers
    except Exception as exc:
        info["reason"] = f"import-failed: {exc}"
        return None, info

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = None
    load_errors: list[str] = []
    model_classes = []
    for class_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModel"):
        model_class = getattr(transformers, class_name, None)
        if model_class is not None:
            model_classes.append(model_class)

    for model_class in model_classes:
        try:
            model = model_class.from_pretrained(
                args.repo_id,
                revision=args.revision,
                device_map="auto",
                quantization_config=bnb_config,
                trust_remote_code=True,
            )
            break
        except Exception as exc:
            load_errors.append(f"{model_class.__name__}: {exc}")
    if model is None:
        info["reason"] = "model-load-failed"
        info["load_errors"] = load_errors
        return None, info

    layer_stack = (
        resolve_attr_path(model, "model.language_model.layers")
        or resolve_attr_path(model, "language_model.layers")
        or resolve_attr_path(model, "model.layers")
        or resolve_attr_path(model, "layers")
    )
    if layer_stack is None:
        info["reason"] = "layer-stack-not-found"
        del model
        return None, info

    tokenizer = AutoTokenizer.from_pretrained(args.repo_id, revision=args.revision, trust_remote_code=True)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    joined_text = "\n\n".join(text for text in dataset["text"] if text.strip())
    encoded = tokenizer(joined_text, return_tensors="pt", truncation=True, max_length=args.num_tokens)
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def _hook(_module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
            if layer_idx in captured:
                return
            hidden = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float().cpu()
            captured[layer_idx] = normalize_rms(hidden[: args.num_tokens])

        return _hook

    for layer_idx in args.layers:
        handles.append(layer_stack[layer_idx].mlp.register_forward_pre_hook(make_hook(layer_idx)))

    try:
        with torch.inference_mode():
            device = next(model.parameters()).device
            model(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
                use_cache=False,
            )
    except Exception as exc:
        info["reason"] = f"forward-failed: {exc}"
        captured.clear()
    finally:
        for handle in handles:
            handle.remove()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(captured) != len(args.layers):
        info["reason"] = info.get("reason", "missing-layer-captures")
        info["captured_layers"] = sorted(captured.keys())
        return None, info

    info["success"] = True
    info["captured_layers"] = sorted(captured.keys())
    return captured, info


def aggregate_moments(parts: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    sample_count = sum(part.shape[-1] for part in parts)
    s1 = parts[0].sum(dim=-1)
    s2 = (parts[0] * parts[0]).sum(dim=-1)
    s3 = (parts[0] * parts[0] * parts[0]).sum(dim=-1)
    s4 = (parts[0] * parts[0] * parts[0] * parts[0]).sum(dim=-1)
    for part in parts[1:]:
        s1 = s1 + part.sum(dim=-1)
        s2 = s2 + (part * part).sum(dim=-1)
        s3 = s3 + (part * part * part).sum(dim=-1)
        s4 = s4 + (part * part * part * part).sum(dim=-1)
    return s1, s2, s3, s4, sample_count


def variance_from_raw(s1: torch.Tensor, s2: torch.Tensor, sample_count: int) -> torch.Tensor:
    mean = s1 / sample_count
    centered = s2 - sample_count * mean * mean
    return centered / max(sample_count - 1, 1)


def kurtosis_from_raw(
    s1: torch.Tensor,
    s2: torch.Tensor,
    s3: torch.Tensor,
    s4: torch.Tensor,
    sample_count: int,
) -> torch.Tensor:
    mean = s1 / sample_count
    ex2 = s2 / sample_count
    ex3 = s3 / sample_count
    ex4 = s4 / sample_count
    variance = variance_from_raw(s1, s2, sample_count).clamp_min(0.0)
    central4 = ex4 - 4.0 * mean * ex3 + 6.0 * mean.square() * ex2 - 3.0 * mean.pow(4)
    return central4 / (variance.square() + 1e-10)


def simulate_nvfp4(weight: torch.Tensor, block_size: int = GROUP_SIZE) -> torch.Tensor:
    if weight.shape[-1] % block_size != 0:
        raise ValueError(f"Last dimension {weight.shape[-1]} must be divisible by {block_size}")
    values = E2M1_VALUES.to(device=weight.device)
    thresholds = E2M1_THRESHOLDS.to(device=weight.device)
    reshaped = weight.float().reshape(*weight.shape[:-1], weight.shape[-1] // block_size, block_size)
    absmax = reshaped.abs().amax(dim=-1, keepdim=True)
    safe_scale = torch.where(absmax > 0, absmax / 6.0, torch.ones_like(absmax))
    normalized = reshaped / safe_scale
    bucket = torch.bucketize(normalized.abs().contiguous(), thresholds)
    quantized = values[bucket] * normalized.sign()
    dequantized = torch.where(absmax > 0, quantized * safe_scale, torch.zeros_like(quantized))
    return dequantized.reshape_as(weight)


def compute_w1_metrics(
    gate_up_weight: torch.Tensor,
    layer_input: torch.Tensor,
    timer: Timer,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    gate = gate_up_weight[:, :MOE_INTERMEDIATE_SIZE, :]
    up = gate_up_weight[:, MOE_INTERMEDIATE_SIZE:, :]
    metrics: dict[str, torch.Tensor] = {}

    with timer.timed("weight_l2"):
        metrics["weight_l2"] = torch.sqrt(gate.square().sum(dim=-1) + up.square().sum(dim=-1))
    with timer.timed("weight_l1"):
        metrics["weight_l1"] = gate.abs().sum(dim=-1) + up.abs().sum(dim=-1)
    with timer.timed("weight_variance"):
        s1, s2, s3, s4, count = aggregate_moments([gate, up])
        metrics["weight_variance"] = variance_from_raw(s1, s2, count)
    with timer.timed("weight_kurtosis"):
        metrics["weight_kurtosis"] = kurtosis_from_raw(s1, s2, s3, s4, count)
    with timer.timed("weight_max_abs"):
        metrics["weight_max_abs"] = torch.maximum(gate.abs().amax(dim=-1), up.abs().amax(dim=-1))
    with timer.timed("within_group_var"):
        gate_var = gate.reshape(NUM_EXPERTS, MOE_INTERMEDIATE_SIZE, -1, GROUP_SIZE).var(
            dim=-1, correction=1
        )
        up_var = up.reshape(NUM_EXPERTS, MOE_INTERMEDIATE_SIZE, -1, GROUP_SIZE).var(
            dim=-1, correction=1
        )
        metrics["within_group_var"] = (gate_var.mean(dim=-1) + up_var.mean(dim=-1)) * 0.5
    with timer.timed("sinq_column_scale"):
        metrics["sinq_column_scale"] = metrics["weight_l2"] / (2 * HIDDEN_SIZE) ** 0.5

    with timer.timed("w1_forward_outputs"):
        output = torch.einsum("tk,enk->etn", layer_input, gate_up_weight)
        gate_out = output[:, :, :MOE_INTERMEDIATE_SIZE]
        up_out = output[:, :, MOE_INTERMEDIATE_SIZE:]
        inter = F.silu(gate_out) * up_out
    with timer.timed("awq_activation"):
        metrics["awq_activation"] = gate_out.abs().mean(dim=1) + up_out.abs().mean(dim=1)
    with timer.timed("smoothquant_max"):
        metrics["smoothquant_max"] = torch.maximum(gate_out.abs().amax(dim=1), up_out.abs().amax(dim=1))
    with timer.timed("owq_hessian_diag"):
        metrics["owq_hessian_diag"] = 2.0 * (gate_out.square().mean(dim=1) + up_out.square().mean(dim=1))
    with timer.timed("slim_llm_salience"):
        input_mean_abs = layer_input.abs().mean(dim=0)
        metrics["slim_llm_salience"] = torch.einsum("enk,k->en", gate.abs() + up.abs(), input_mean_abs)
    with timer.timed("owq_sensitivity"):
        quantized = simulate_nvfp4(gate_up_weight)
        delta = gate_up_weight - quantized
        delta_out = torch.einsum("tk,enk->etn", layer_input, delta)
        delta_gate = delta_out[:, :, :MOE_INTERMEDIATE_SIZE]
        delta_up = delta_out[:, :, MOE_INTERMEDIATE_SIZE:]
        metrics["owq_sensitivity"] = (delta_gate.square() + delta_up.square()).mean(dim=1)
    return metrics, inter


def compute_w2_metrics(
    down_weight: torch.Tensor,
    intermediate_input: torch.Tensor,
    timer: Timer,
) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {}

    with timer.timed("weight_l2"):
        metrics["weight_l2"] = down_weight.norm(p=2, dim=-1)
    with timer.timed("weight_l1"):
        metrics["weight_l1"] = down_weight.abs().sum(dim=-1)
    with timer.timed("weight_variance"):
        s1, s2, s3, s4, count = aggregate_moments([down_weight])
        metrics["weight_variance"] = variance_from_raw(s1, s2, count)
    with timer.timed("weight_kurtosis"):
        metrics["weight_kurtosis"] = kurtosis_from_raw(s1, s2, s3, s4, count)
    with timer.timed("weight_max_abs"):
        metrics["weight_max_abs"] = down_weight.abs().amax(dim=-1)
    with timer.timed("within_group_var"):
        metrics["within_group_var"] = down_weight.reshape(NUM_EXPERTS, HIDDEN_SIZE, -1, GROUP_SIZE).var(
            dim=-1, correction=1
        ).mean(dim=-1)
    with timer.timed("sinq_column_scale"):
        metrics["sinq_column_scale"] = metrics["weight_l2"] / (MOE_INTERMEDIATE_SIZE**0.5)

    with timer.timed("w2_forward_outputs"):
        output = torch.einsum("etk,enk->etn", intermediate_input, down_weight)
    with timer.timed("awq_activation"):
        metrics["awq_activation"] = output.abs().mean(dim=1)
    with timer.timed("smoothquant_max"):
        metrics["smoothquant_max"] = output.abs().amax(dim=1)
    with timer.timed("owq_hessian_diag"):
        metrics["owq_hessian_diag"] = 2.0 * output.square().mean(dim=1)
    with timer.timed("slim_llm_salience"):
        input_mean_abs = intermediate_input.abs().mean(dim=1)
        metrics["slim_llm_salience"] = torch.einsum("enk,ek->en", down_weight.abs(), input_mean_abs)
    with timer.timed("owq_sensitivity"):
        quantized = simulate_nvfp4(down_weight)
        delta = down_weight - quantized
        delta_out = torch.einsum("etk,enk->etn", intermediate_input, delta)
        metrics["owq_sensitivity"] = delta_out.square().mean(dim=1)
    return metrics


def correlation_summary(metric_arrays: dict[str, dict[str, list[np.ndarray]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for projection, metric_map in metric_arrays.items():
        ordered_names = [name for name in METRIC_ORDER if name in metric_map]
        if not ordered_names:
            continue
        stacked = np.stack([np.concatenate(metric_map[name], axis=0) for name in ordered_names], axis=1)
        if spearmanr is not None:
            corr, pval = spearmanr(stacked, axis=0)
            corr = np.asarray(corr, dtype=np.float64)
            pval = np.asarray(pval, dtype=np.float64)
        else:
            ranks = np.stack([stacked[:, idx].argsort().argsort() for idx in range(stacked.shape[1])], axis=1)
            corr = np.corrcoef(ranks, rowvar=False)
            pval = np.full_like(corr, np.nan, dtype=np.float64)
        summary[projection] = {
            "metric_order": ordered_names,
            "spearman_rho": corr.tolist(),
            "p_value": pval.tolist(),
        }
    return summary


def print_correlations(correlations: dict[str, Any]) -> None:
    print("\n=== Spearman Rank Correlations ===")
    for projection, payload in correlations.items():
        print(f"\n[{projection}]")
        names = payload["metric_order"]
        rho = payload["spearman_rho"]
        pval = payload["p_value"]
        for i, first in enumerate(names):
            for j, second in enumerate(names):
                corr = rho[i][j]
                pvalue = pval[i][j]
                pvalue_str = "nan" if np.isnan(pvalue) else f"{pvalue:.2e}"
                print(f"{first} vs {second}: rho={corr:.3f}, p={pvalue_str}")


def print_timings(label: str, timings: dict[str, float]) -> None:
    print(f"\n{label} timings:")
    for name in sorted(timings):
        print(f"  {name:20s} {timings[name]:8.3f}s")


def serialize_group_results(
    results: dict[str, Any],
    layer_idx: int,
    projection: str,
    metric_tensors: dict[str, torch.Tensor],
) -> None:
    for expert_idx in range(NUM_EXPERTS):
        group_key = f"layer_{layer_idx}_expert_{expert_idx}_{projection}"
        results[group_key] = {
            metric_name: metric_tensors[metric_name][expert_idx].detach().cpu().to(torch.float32).tolist()
            for metric_name in METRIC_ORDER
        }


def append_correlation_inputs(
    store: dict[str, list[np.ndarray]],
    metric_tensors: dict[str, torch.Tensor],
) -> None:
    for metric_name in METRIC_ORDER:
        store[metric_name].append(metric_tensors[metric_name].detach().cpu().numpy().reshape(-1))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device)

    invalid_layers = [layer for layer in args.layers if layer < 0 or layer >= NUM_LAYERS]
    if invalid_layers:
        raise ValueError(f"Invalid layer indices: {invalid_layers}")

    model_paths = ModelPaths(
        repo_id=args.repo_id,
        revision=args.revision,
        snapshot_dir=args.snapshot_dir,
    )
    loader = LazySafetensorLoader(model_paths)

    print("=== Spike 2 Proxy Metrics ===")
    print(f"Model: {args.repo_id}@{args.revision}")
    print(f"Layers: {args.layers}")
    print(f"Experts: {NUM_EXPERTS}")
    print(f"Device: {device}")
    print(f"Output: {args.output}")

    real_inputs, activation_info = try_capture_real_inputs(args, loader)
    if real_inputs is None:
        print(f"Falling back to Gaussian activations ({activation_info.get('reason', 'unknown')}).")
        layer_inputs = gaussian_layer_inputs(args.layers, args.num_tokens, HIDDEN_SIZE, args.seed)
        activation_info = {
            "source": "gaussian",
            "attempted_real": activation_info,
            "success": True,
            "num_tokens": args.num_tokens,
        }
    else:
        print("Using real Wikitext2 calibration activations captured from one 4-bit forward pass.")
        layer_inputs = real_inputs
        activation_info["num_tokens"] = args.num_tokens

    results: dict[str, Any] = {
        "_metadata": {
            "repo_id": args.repo_id,
            "revision": args.revision,
            "layers": args.layers,
            "num_experts": NUM_EXPERTS,
            "hidden_size": HIDDEN_SIZE,
            "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
            "num_tokens": args.num_tokens,
            "activation_capture": activation_info,
            "metric_order": METRIC_ORDER,
            "completed_layers": [],
        }
    }
    correlation_inputs: dict[str, dict[str, list[np.ndarray]]] = {
        "w1": defaultdict(list),
        "w2": defaultdict(list),
    }

    overall_start = time.perf_counter()
    for layer_idx in args.layers:
        layer_start = time.perf_counter()
        print(f"\n--- Layer {layer_idx} ---")
        w1_key = f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"
        w2_key = f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj"

        print("Loading expert tensors...")
        gate_up_weight = loader.get_tensor(w1_key).to(device=device, dtype=torch.float32)
        down_weight = loader.get_tensor(w2_key).to(device=device, dtype=torch.float32)
        layer_input = layer_inputs[layer_idx].to(device=device, dtype=torch.float32)

        print(f"  gate_up_proj: {tuple(gate_up_weight.shape)}")
        print(f"  down_proj:    {tuple(down_weight.shape)}")
        print(f"  layer_input:  {tuple(layer_input.shape)}")

        w1_timer = Timer()
        w1_metrics, intermediate_input = compute_w1_metrics(gate_up_weight, layer_input, w1_timer)
        serialize_group_results(results, layer_idx, "w1", w1_metrics)
        append_correlation_inputs(correlation_inputs["w1"], w1_metrics)
        print_timings(f"Layer {layer_idx} / w1", w1_timer.timings)

        w2_timer = Timer()
        w2_metrics = compute_w2_metrics(down_weight, intermediate_input, w2_timer)
        serialize_group_results(results, layer_idx, "w2", w2_metrics)
        append_correlation_inputs(correlation_inputs["w2"], w2_metrics)
        print_timings(f"Layer {layer_idx} / w2", w2_timer.timings)

        results["_metadata"]["completed_layers"].append(layer_idx)
        atomic_json_dump(results, args.output)
        print(f"Saved intermediate results after layer {layer_idx} -> {args.output}")

        del gate_up_weight, down_weight, layer_input, intermediate_input, w1_metrics, w2_metrics
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Layer {layer_idx} completed in {time.perf_counter() - layer_start:.2f}s")

    correlations = correlation_summary(correlation_inputs)
    results["_correlations"] = correlations
    results["_metadata"]["elapsed_seconds"] = time.perf_counter() - overall_start
    atomic_json_dump(results, args.output)
    print_correlations(correlations)
    print(f"\nDone in {results['_metadata']['elapsed_seconds']:.2f}s")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()

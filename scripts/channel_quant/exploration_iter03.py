#!/usr/bin/env python3
# pyright: basic, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from baselines_comparison import (
    DEFAULT_MINI_BATCH_TOKENS,
    EvaluationPlan,
    allocate_weighted_counts,
    evaluate_plan,
    expert_full_weight_count,
    load_prefix_dataset_tokens,
    resolve_non_expert_bytes,
    run_calibration_pass,
    topk_mask_from_scores,
)
from exploration_iter02 import (
    MXMOE_TARGET_PPL,
    build_plan,
    build_topk_two_level_masks,
    compute_metric_caches,
    dtype_from_name,
    metric_cache_from_iter01_payload,
    total_channel_fraction,
    total_pair_fraction,
)
from spike1_ground_truth import MODEL_ID, WeightStore, build_text_config, load_root_config


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_ITER01_JSON = RESULTS_DIR / "exploration_iter01.json"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "exploration_iter03.json"
DEFAULT_EXPLORATION_MD = SCRIPT_DIR / "exploration.md"
DEFAULT_CALIBRATION_TOKENS = 128
DEFAULT_EVAL_TOKENS = 512
SECTION_MARKER = "## [12] Per-Projection Split Optimization"
BASE_AVERAGE_FRACTION = 0.25
DEFAULT_SPLIT_SPECS = (
    (0, 50),
    (5, 45),
    (10, 40),
    (15, 35),
    (20, 30),
    (25, 25),
    (30, 20),
    (40, 10),
    (50, 0),
)
DEFAULT_BUDGET_SWEEP_PCTS = (10, 15, 20, 25, 30, 40, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--iter01-json", type=Path, default=DEFAULT_ITER01_JSON)
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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ratio_key(w1_pct: int, w2_pct: int) -> str:
    return f"w1_{w1_pct:02d}_w2_{w2_pct:02d}"


def budget_key(total_pct: int) -> str:
    return f"total_{total_pct:02d}"


def mean_budget_fraction(w1_fraction: float, w2_fraction: float) -> float:
    return 0.5 * (w1_fraction + w2_fraction)


def realized_fp8_fraction(config: Any, w1_fraction: float, w2_fraction: float) -> float:
    w1_weight = 2.0 * config.moe_intermediate_size * config.hidden_size
    w2_weight = 1.0 * config.hidden_size * config.moe_intermediate_size
    total_weight = w1_weight + w2_weight
    return ((w1_fraction * w1_weight) + (w2_fraction * w2_weight)) / total_weight


def split_weight_shares(config: Any, w1_fraction: float, w2_fraction: float) -> tuple[float, float]:
    w1_weight = 2.0 * config.moe_intermediate_size * config.hidden_size * w1_fraction
    w2_weight = 1.0 * config.hidden_size * config.moe_intermediate_size * w2_fraction
    total = w1_weight + w2_weight
    if total <= 0.0:
        return 0.0, 0.0
    return w1_weight / total, w2_weight / total


def build_split_plan(
    metric_cache: dict[int, Any],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
    w1_fraction: float,
    w2_fraction: float,
) -> EvaluationPlan:
    w1_pair_masks, w2_channel_masks = build_topk_two_level_masks(metric_cache, config, w1_fraction, w2_fraction)
    return build_plan(
        name=ratio_key(int(round(w1_fraction * 100.0)), int(round(w2_fraction * 100.0))),
        description=(
            f"Routing-aware per-projection split with W1={int(round(w1_fraction * 100.0))}% FP8 "
            f"and W2={int(round(w2_fraction * 100.0))}% FP8 using activation_kurtosis."
        ),
        config=config,
        non_expert_bytes=non_expert_bytes,
        total_expert_elems=total_expert_elems,
        w1_pair_masks=w1_pair_masks,
        w2_channel_masks=w2_channel_masks,
    )


def build_three_level_plan(
    metric_cache: dict[int, Any],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
    w1_fraction: float,
    w2_fraction: float,
) -> EvaluationPlan:
    overall_fraction = realized_fp8_fraction(config, w1_fraction, w2_fraction)
    target_fp8_weights = int(round(total_expert_elems * overall_fraction))
    expert_capacity = expert_full_weight_count(config)
    w1_pair_cost = 2 * config.hidden_size
    w2_channel_cost = config.moe_intermediate_size
    w1_share, _w2_share = split_weight_shares(config, w1_fraction, w2_fraction)

    w1_pair_masks: dict[int, dict[int, torch.Tensor]] = {}
    w2_channel_masks: dict[int, dict[int, torch.Tensor]] = {}
    ordered_layers = sorted(metric_cache)
    layer_weights = [float(metric_cache[layer_idx].routing_counts.sum().item()) for layer_idx in ordered_layers]
    layer_budgets = allocate_weighted_counts(
        target_fp8_weights,
        [expert_capacity * config.num_experts] * len(ordered_layers),
        layer_weights,
    )

    for layer_idx, layer_budget in zip(ordered_layers, layer_budgets, strict=True):
        bundle = metric_cache[layer_idx]
        weights = [float(value) for value in bundle.routing_counts.tolist()]
        expert_budgets = allocate_weighted_counts(
            int(layer_budget),
            [expert_capacity] * config.num_experts,
            weights,
        )
        layer_w1: dict[int, torch.Tensor] = {}
        layer_w2: dict[int, torch.Tensor] = {}
        for expert_idx, expert_budget in enumerate(expert_budgets):
            w1_budget = int(round(expert_budget * w1_share))
            w2_budget = expert_budget - w1_budget
            w1_pairs = min(config.moe_intermediate_size, int(round(w1_budget / max(w1_pair_cost, 1))))
            w2_channels = min(config.hidden_size, int(round(w2_budget / max(w2_channel_cost, 1))))
            total_cost = (w1_pairs * w1_pair_cost) + (w2_channels * w2_channel_cost)
            while total_cost > expert_budget and (w1_pairs > 0 or w2_channels > 0):
                if w1_pairs > 0 and (w2_channels == 0 or (total_cost - expert_budget) >= w1_pair_cost):
                    w1_pairs -= 1
                elif w2_channels > 0:
                    w2_channels -= 1
                total_cost = (w1_pairs * w1_pair_cost) + (w2_channels * w2_channel_cost)
            layer_w1[expert_idx] = topk_mask_from_scores(bundle.w1_pair_scores[expert_idx], w1_pairs)
            layer_w2[expert_idx] = topk_mask_from_scores(bundle.w2_channel_scores[expert_idx], w2_channels)
        w1_pair_masks[layer_idx] = layer_w1
        w2_channel_masks[layer_idx] = layer_w2

    return build_plan(
        name="three_level_hierarchy",
        description=(
            "Three-level hierarchy: routing-aware expert budget, per-projection split inside each expert, "
            "then activation_kurtosis top-k channel selection."
        ),
        config=config,
        non_expert_bytes=non_expert_bytes,
        total_expert_elems=total_expert_elems,
        w1_pair_masks=w1_pair_masks,
        w2_channel_masks=w2_channel_masks,
    )


def summarize_plan_result(
    plan: EvaluationPlan,
    config: Any,
    total_expert_elems: int,
    elapsed: float,
    ppl: float,
    nll: float,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "description": plan.description,
        "ppl": round(ppl, 6),
        "nll": round(nll, 6),
        "memory_gb": plan.memory_gb,
        "fp8_weights": int(plan.fp8_weights),
        "fp8_fraction": round(float(plan.fp8_weights) / float(total_expert_elems), 6),
        "w1_pair_fraction": round(total_pair_fraction(config, plan.w1_pair_masks or {}), 6),
        "w2_channel_fraction": round(total_channel_fraction(config, plan.w2_channel_masks or {}), 6),
        "time_s": round(elapsed, 1),
    }
    if extras:
        row.update(extras)
    return row


def maybe_eval_plan(
    payload: dict[str, Any],
    bucket: str,
    key: str,
    plan: EvaluationPlan,
    config: Any,
    total_expert_elems: int,
    eval_ids: torch.Tensor,
    snapshot_dir: Path,
    weight_map: dict[str, str],
    device: torch.device,
    dtype: torch.dtype,
    output_json: Path,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing_bucket = payload.setdefault(bucket, {})
    if key in existing_bucket:
        print(f"[skip] {bucket}:{key} already present", flush=True)
        return existing_bucket[key]
    print(f"\n=== Eval {bucket}:{key} ===", flush=True)
    t0 = time.time()
    ppl, nll = evaluate_plan(plan, eval_ids, config, weight_map, snapshot_dir, device, dtype)
    elapsed = time.time() - t0
    row = summarize_plan_result(plan, config, total_expert_elems, elapsed, ppl, nll, extras)
    existing_bucket[key] = row
    atomic_json_dump(output_json, payload)
    print(
        f"  -> PPL={ppl:.4f} | NLL={nll:.6f} | memory={plan.memory_gb:.3f} GB | fp8={row['fp8_fraction']:.4f} | time={elapsed:.1f}s",
        flush=True,
    )
    return row


def best_row(rows: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    return min(rows.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))


def print_split_ratio_table(rows: dict[str, dict[str, Any]]) -> None:
    print("\n" + "=" * 118, flush=True)
    print("Iteration 3 split ratio sweep (activation_kurtosis, 512 WikiText-2 test tokens)", flush=True)
    print("=" * 118, flush=True)
    print(
        f"{'W1 %':>6} {'W2 %':>6} {'Avg %':>7} {'PPL':>9} {'NLL':>10} {'Memory GB':>11} {'FP8 frac':>10} {'W1 frac':>10} {'W2 frac':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 118, flush=True)
    ordered = sorted(rows.values(), key=lambda row: (float(row["mean_budget_pct"]), float(row["w1_pct"]), float(row["w2_pct"])))
    for row in ordered:
        print(
            f"{int(row['w1_pct']):>6} {int(row['w2_pct']):>6} {float(row['mean_budget_pct']):>7.1f} {float(row['ppl']):>9.4f} {float(row['nll']):>10.6f} {float(row['memory_gb']):>11.3f} {float(row['fp8_fraction']):>10.4f} {float(row['w1_pair_fraction']):>10.4f} {float(row['w2_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
            flush=True,
        )


def print_budget_sweep_table(rows: dict[str, dict[str, Any]]) -> None:
    print("\n" + "=" * 118, flush=True)
    print("Iteration 3 budget sweep (best split ratio held fixed)", flush=True)
    print("=" * 118, flush=True)
    print(
        f"{'Total %':>8} {'W1 %':>6} {'W2 %':>6} {'PPL':>9} {'NLL':>10} {'Memory GB':>11} {'FP8 frac':>10} {'W1 frac':>10} {'W2 frac':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 118, flush=True)
    ordered = sorted(rows.values(), key=lambda row: int(row["total_budget_pct"]))
    for row in ordered:
        print(
            f"{int(row['total_budget_pct']):>8} {float(row['w1_pct']):>6.1f} {float(row['w2_pct']):>6.1f} {float(row['ppl']):>9.4f} {float(row['nll']):>10.6f} {float(row['memory_gb']):>11.3f} {float(row['fp8_fraction']):>10.4f} {float(row['w1_pair_fraction']):>10.4f} {float(row['w2_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
            flush=True,
        )


def build_exploration_section(payload: dict[str, Any]) -> str:
    split_rows = payload["split_ratio_sweep"]
    budget_rows = payload["budget_sweep"]
    hierarchy_row = payload.get("three_level_hierarchy")
    best_split_key, best_split = best_row(split_rows)
    best_budget_key, best_budget = best_row(budget_rows)

    split_table = [
        "| W1 % | W2 % | PPL | Memory (GB) | FP8 fraction |",
        "|------|------|-----|-------------|--------------|",
    ]
    for row in sorted(split_rows.values(), key=lambda item: (float(item["ppl"]), float(item["memory_gb"]), int(item["w1_pct"]))):
        split_table.append(
            f"| {int(row['w1_pct'])} | {int(row['w2_pct'])} | {float(row['ppl']):.4f} | {float(row['memory_gb']):.3f} | {float(row['fp8_fraction']):.4f} |"
        )

    budget_table = [
        "| Total % | W1 % | W2 % | PPL | Memory (GB) |",
        "|---------|------|------|-----|-------------|",
    ]
    for row in sorted(budget_rows.values(), key=lambda item: int(item["total_budget_pct"])):
        budget_table.append(
            f"| {int(row['total_budget_pct'])} | {float(row['w1_pct']):.1f} | {float(row['w2_pct']):.1f} | {float(row['ppl']):.4f} | {float(row['memory_gb']):.3f} |"
        )

    hierarchy_text = ""
    if isinstance(hierarchy_row, dict):
        hierarchy_gap = float(hierarchy_row["ppl"]) - float(best_split["ppl"])
        hierarchy_text = (
            f"**Three-level hierarchy**: The explicit expert-budget -> projection-budget -> per-channel plan lands at PPL {float(hierarchy_row['ppl']):.4f} and {float(hierarchy_row['memory_gb']):.3f} GB, "
            f"which is {'better' if hierarchy_gap < 0.0 else 'worse'} than the direct split sweep by {abs(hierarchy_gap):.4f} PPL."
        )

    return "\n".join([
        SECTION_MARKER,
        "**Approach**: Reused the Iteration 2 streamed 40-layer evaluation loop and the same activation_kurtosis calibration metric, then ran two follow-ups around the new Iteration 2 winner: a W1/W2 split sweep at the nominal 25% average budget and a budget sweep that scales the winning ratio from 10% to 50% average FP8. I also added a stricter three-level hierarchy that allocates total FP8 weights to hot experts first, then splits each expert budget across W1/W2 before doing within-expert top-k channel selection.",
        f"**Split sweep**: The best fixed-average split is `{best_split_key}` at PPL {float(best_split['ppl']):.4f}, {float(best_split['memory_gb']):.3f} GB, and realized FP8 fraction {float(best_split['fp8_fraction']):.4f}. This directly answers whether the Iteration 2 `{ratio_key(10, 40)}` winner was a local optimum or whether pushing still more budget toward W2 helps.",
        "\n".join(split_table),
        f"**Budget sweep**: Holding the winning ratio fixed, the best total budget point is `{best_budget_key}` at PPL {float(best_budget['ppl']):.4f} and {float(best_budget['memory_gb']):.3f} GB. That row is the current Pareto candidate because it isolates whether the new ratio should be run leaner or richer than the original 25% average budget.",
        "\n".join(budget_table),
        hierarchy_text,
        f"**Verdict**: The new reference to beat is PPL {float(best_budget['ppl']):.4f} at {float(best_budget['memory_gb']):.3f} GB. Compared with the 5.337 MxMoE target, the margin is {float(best_budget['ppl']) - MXMOE_TARGET_PPL:+.4f} PPL.",
        "",
    ]).rstrip()


def upsert_exploration_section(path: Path, section: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if SECTION_MARKER in existing:
        prefix = existing.split(SECTION_MARKER, 1)[0].rstrip()
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
    dtype = dtype_from_name(args.dtype)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    print("=== Iteration 3: Per-Projection Split Optimization ===", flush=True)
    print(f"Model: {args.model_id}", flush=True)
    print(f"Device: {device} | dtype: {dtype}", flush=True)

    iter01_payload = load_json(args.iter01_json)
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
            "metric": "activation_kurtosis",
            "iter01_json": str(args.iter01_json),
            "base_average_budget_pct": int(BASE_AVERAGE_FRACTION * 100.0),
            "split_ratio_candidates_pct": [list(spec) for spec in DEFAULT_SPLIT_SPECS],
            "budget_sweep_total_pct": list(DEFAULT_BUDGET_SWEEP_PCTS),
            "mxmoe_target_ppl": MXMOE_TARGET_PPL,
            "iter02_best": {
                "name": "per_projection_split",
                "w1_pct": 10,
                "w2_pct": 40,
                "ppl": 5.330317,
                "memory_gb": 26.806,
            },
        },
        "split_ratio_sweep": {},
        "budget_sweep": {},
        "three_level_hierarchy": None,
        "best_split": None,
        "best_budget": None,
        "runtime_seconds": None,
    }
    if args.output_json.exists():
        try:
            existing_payload = load_json(args.output_json)
            if isinstance(existing_payload, dict):
                payload.update(existing_payload)
                payload.setdefault("split_ratio_sweep", {})
                payload.setdefault("budget_sweep", {})
                print(f"[resume] loaded existing results from {args.output_json}", flush=True)
        except Exception:
            print(f"[resume] ignoring unreadable existing file at {args.output_json}", flush=True)
    atomic_json_dump(args.output_json, payload)

    cached_metrics = metric_cache_from_iter01_payload(iter01_payload)
    metric_source = "iter01_json"
    store = WeightStore(args.model_id, snapshot_dir, weight_map)
    if cached_metrics is None:
        metric_source = "recomputed_from_calibration"
        captures = run_calibration_pass(store, text_config, calib_ids, args.mini_batch_tokens, device, dtype)
        metric_caches = compute_metric_caches(store, text_config, captures, device, dtype)
    else:
        metric_caches = cached_metrics
    activation_cache = metric_caches["activation_kurtosis"]
    payload.setdefault("metadata", {})["metric_source"] = metric_source
    atomic_json_dump(args.output_json, payload)

    for w1_pct, w2_pct in DEFAULT_SPLIT_SPECS:
        w1_fraction = w1_pct / 100.0
        w2_fraction = w2_pct / 100.0
        key = ratio_key(w1_pct, w2_pct)
        plan = build_split_plan(activation_cache, text_config, non_expert_bytes, total_expert_elems, w1_fraction, w2_fraction)
        maybe_eval_plan(
            payload=payload,
            bucket="split_ratio_sweep",
            key=key,
            plan=plan,
            config=text_config,
            total_expert_elems=total_expert_elems,
            eval_ids=eval_ids,
            snapshot_dir=snapshot_dir,
            weight_map=weight_map,
            device=device,
            dtype=dtype,
            output_json=args.output_json,
            extras={
                "w1_pct": w1_pct,
                "w2_pct": w2_pct,
                "mean_budget_pct": 100.0 * mean_budget_fraction(w1_fraction, w2_fraction),
                "target_realized_fp8_fraction": round(realized_fp8_fraction(text_config, w1_fraction, w2_fraction), 6),
            },
        )

    best_split_key, best_split_row = best_row(payload["split_ratio_sweep"])
    payload["best_split"] = {"key": best_split_key, **best_split_row}
    atomic_json_dump(args.output_json, payload)

    best_w1_pct = int(best_split_row["w1_pct"])
    best_w2_pct = int(best_split_row["w2_pct"])
    for total_pct in DEFAULT_BUDGET_SWEEP_PCTS:
        scale = total_pct / (BASE_AVERAGE_FRACTION * 100.0)
        scaled_w1_pct = best_w1_pct * scale
        scaled_w2_pct = best_w2_pct * scale
        w1_fraction = scaled_w1_pct / 100.0
        w2_fraction = scaled_w2_pct / 100.0
        key = budget_key(total_pct)
        plan = build_split_plan(activation_cache, text_config, non_expert_bytes, total_expert_elems, w1_fraction, w2_fraction)
        maybe_eval_plan(
            payload=payload,
            bucket="budget_sweep",
            key=key,
            plan=plan,
            config=text_config,
            total_expert_elems=total_expert_elems,
            eval_ids=eval_ids,
            snapshot_dir=snapshot_dir,
            weight_map=weight_map,
            device=device,
            dtype=dtype,
            output_json=args.output_json,
            extras={
                "total_budget_pct": total_pct,
                "w1_pct": round(scaled_w1_pct, 4),
                "w2_pct": round(scaled_w2_pct, 4),
                "ratio_reference": f"{best_w1_pct}:{best_w2_pct}",
                "target_realized_fp8_fraction": round(realized_fp8_fraction(text_config, w1_fraction, w2_fraction), 6),
            },
        )

    best_budget_key, best_budget_row = best_row(payload["budget_sweep"])
    payload["best_budget"] = {"key": best_budget_key, **best_budget_row}
    atomic_json_dump(args.output_json, payload)

    hierarchy_plan = build_three_level_plan(
        activation_cache,
        text_config,
        non_expert_bytes,
        total_expert_elems,
        best_w1_pct / 100.0,
        best_w2_pct / 100.0,
    )
    existing_hierarchy = payload.get("three_level_hierarchy")
    if not isinstance(existing_hierarchy, dict):
        print("\n=== Eval three_level_hierarchy ===", flush=True)
        t0 = time.time()
        ppl, nll = evaluate_plan(hierarchy_plan, eval_ids, text_config, weight_map, snapshot_dir, device, dtype)
        elapsed = time.time() - t0
        payload["three_level_hierarchy"] = summarize_plan_result(
            hierarchy_plan,
            text_config,
            total_expert_elems,
            elapsed,
            ppl,
            nll,
            {
                "w1_pct": best_w1_pct,
                "w2_pct": best_w2_pct,
                "mean_budget_pct": float(best_split_row["mean_budget_pct"]),
                "ratio_reference": f"{best_w1_pct}:{best_w2_pct}",
                "target_realized_fp8_fraction": round(
                    realized_fp8_fraction(text_config, best_w1_pct / 100.0, best_w2_pct / 100.0),
                    6,
                ),
            },
        )
        atomic_json_dump(args.output_json, payload)
        print(
            f"  -> PPL={ppl:.4f} | NLL={nll:.6f} | memory={hierarchy_plan.memory_gb:.3f} GB | fp8={payload['three_level_hierarchy']['fp8_fraction']:.4f} | time={elapsed:.1f}s",
            flush=True,
        )
    else:
        print("[skip] three_level_hierarchy already present", flush=True)

    payload["runtime_seconds"] = round(time.time() - start_time, 3)
    atomic_json_dump(args.output_json, payload)

    print_split_ratio_table(payload["split_ratio_sweep"])
    print_budget_sweep_table(payload["budget_sweep"])
    if isinstance(payload.get("three_level_hierarchy"), dict):
        row = payload["three_level_hierarchy"]
        print(
            f"\nThree-level hierarchy -> PPL={float(row['ppl']):.4f} | memory={float(row['memory_gb']):.3f} GB | fp8={float(row['fp8_fraction']):.4f}",
            flush=True,
        )

    upsert_exploration_section(args.exploration_md, build_exploration_section(payload))
    print(f"\nSaved results -> {args.output_json}", flush=True)
    print(f"Updated log -> {args.exploration_md}", flush=True)


if __name__ == "__main__":
    main()

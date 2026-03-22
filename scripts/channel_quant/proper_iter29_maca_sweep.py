#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from baselines_comparison import resolve_non_expert_bytes
from proper_eval import SEQLEN, atomic_json_dump, dtype_from_name, load_gptq_standard_data
from proper_iter01 import build_plan_from_masks
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from proper_iter11_push_router_affinity import evaluate_and_record_plan
from proper_iter25_learned_correction import maybe_slice_eval_chunks, resolve_requested_plans
from proper_iter26_maca_calibration import (
    MACA_PAD_LENGTH,
    VariableLengthCalibrationSet,
    load_wikitext_train_ids,
    run_maca_calibration,
    summarize_calibration_artifacts,
)
from proper_iter28_maca_plus_correction import BaseVariant, evaluate_and_record_compound_plan, fit_scalar_corrections_for_variant
from spike1_ground_truth import MODEL_ID, WeightStore, build_text_config, load_root_config


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter29_maca_sweep.json"
TOTAL_CALIBRATION_CHUNKS = 128
CORRECTION_NAME = "maca_best_plus_correction"


@dataclass(frozen=True)
class CalibrationMixConfig:
    name: str
    length_counts: tuple[tuple[int, int], ...]
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--configs",
        default="",
        help=(
            "Comma-separated subset of configs to run. Default runs all five base MaCa mixes and the conditional "
            "best-plus-correction row."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-max-chunks",
        type=int,
        default=0,
        help="Optional cap on WikiText-2 evaluation chunks for smoke tests; 0 means full 145-chunk test.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_sweep_configs() -> list[CalibrationMixConfig]:
    return [
        CalibrationMixConfig(
            name="maca_long_heavy",
            length_counts=((128, 16), (512, 16), (2048, 32), (4096, 64)),
            description="Bias the 128-chunk MaCa pool toward longer contexts while keeping the same joint_w1w2_with_topup rebuild and GPTQ-standard evaluation.",
        ),
        CalibrationMixConfig(
            name="maca_with_8k",
            length_counts=((128, 24), (512, 24), (2048, 24), (4096, 24), (8192, 32)),
            description="Inject nominal 8K samples into the 128-chunk MaCa pool, truncating each 8K chunk to the first 4096 tokens before calibration statistics are accumulated.",
        ),
        CalibrationMixConfig(
            name="maca_short_heavy",
            length_counts=((128, 64), (512, 32), (2048, 16), (4096, 16)),
            description="Bias the 128-chunk MaCa pool toward shorter contexts while keeping the same mask builder and GPTQ-standard evaluation path.",
        ),
        CalibrationMixConfig(
            name="maca_uniform_4k",
            length_counts=((4096, 128),),
            description="Use only 4096-token calibration chunks to test whether length diversity matters more than simply emphasizing long-context statistics.",
        ),
        CalibrationMixConfig(
            name="maca_original",
            length_counts=((128, 32), (512, 32), (2048, 32), (4096, 32)),
            description="Reproduce the original Iteration 26 MaCa multi-scale calibration mix as the comparison anchor for the sweep.",
        ),
    ]


def serialize_length_counts(length_counts: tuple[tuple[int, int], ...]) -> list[dict[str, int]]:
    return [{"length": int(length), "count": int(count)} for length, count in length_counts]


def validate_mix_config(mix: CalibrationMixConfig) -> None:
    total = sum(int(count) for _length, count in mix.length_counts)
    if total != TOTAL_CALIBRATION_CHUNKS:
        raise ValueError(f"{mix.name} must use exactly {TOTAL_CALIBRATION_CHUNKS} calibration chunks, got {total}")


def build_variable_length_calibration_set(
    tokenizer: Any,
    train_ids: torch.Tensor,
    seed: int,
    mix: CalibrationMixConfig,
) -> VariableLengthCalibrationSet:
    validate_mix_config(mix)
    rng = random.Random(f"iter29:{seed}:{mix.name}")
    padded_chunks: list[torch.Tensor] = []
    actual_lengths: list[int] = []
    requested_lengths: list[int] = []
    effective_length_counts: dict[int, int] = {}
    requested_length_counts: dict[int, int] = {}

    for requested_length, count in mix.length_counts:
        requested_length = int(requested_length)
        count = int(count)
        effective_length = min(requested_length, MACA_PAD_LENGTH)
        max_start = int(train_ids.shape[1]) - requested_length - 1
        if max_start < 0:
            raise RuntimeError(f"WikiText-2 train is too short for requested calibration length {requested_length}")

        requested_length_counts[requested_length] = requested_length_counts.get(requested_length, 0) + count
        effective_length_counts[effective_length] = effective_length_counts.get(effective_length, 0) + count
        for _ in range(count):
            start = rng.randint(0, max_start)
            sample = train_ids[:, start : start + requested_length]
            padded = torch.zeros((1, MACA_PAD_LENGTH), dtype=torch.long)
            padded[:, :effective_length] = sample[:, :effective_length]
            padded_chunks.append(padded)
            actual_lengths.append(effective_length)
            requested_lengths.append(requested_length)

    chunks = torch.cat(padded_chunks, dim=0).contiguous()
    actual_lengths_tensor = torch.as_tensor(actual_lengths, dtype=torch.long)
    requested_lengths_tensor = torch.as_tensor(requested_lengths, dtype=torch.long)
    actual_token_total = int(actual_lengths_tensor.sum().item())
    requested_token_total = int(requested_lengths_tensor.sum().item())
    padded_token_total = int(chunks.numel())
    metadata = {
        "dataset": "wikitext/wikitext-2-raw-v1",
        "split": "train",
        "tokenizer": tokenizer.name_or_path,
        "base_seed": int(seed),
        "sampling_seed": f"iter29:{seed}:{mix.name}",
        "config_name": mix.name,
        "description": mix.description,
        "requested_length_counts": [{"length": int(length), "count": int(count)} for length, count in sorted(requested_length_counts.items())],
        "effective_length_counts": [{"length": int(length), "count": int(count)} for length, count in sorted(effective_length_counts.items())],
        "samples": int(chunks.shape[0]),
        "sample_shape": [int(chunks.shape[0]), int(chunks.shape[1])],
        "padding_value": 0,
        "context_window": int(MACA_PAD_LENGTH),
        "requested_token_total": requested_token_total,
        "actual_token_total": actual_token_total,
        "padded_token_total": padded_token_total,
        "requested_to_actual_token_drop": int(requested_token_total - actual_token_total),
        "real_token_fraction": round(float(actual_token_total) / float(max(padded_token_total, 1)), 6),
        "truncated_requested_lengths": sorted({int(length) for length in requested_lengths if int(length) > MACA_PAD_LENGTH}),
        "train_tokens": int(train_ids.shape[1]),
        "standard_reference_tokens": int(SEQLEN * chunks.shape[0]),
    }
    return VariableLengthCalibrationSet(chunks=chunks, actual_lengths=actual_lengths_tensor, metadata=metadata)


def build_joint_plan_for_mix(
    mix: CalibrationMixConfig,
    calibration: Any,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> tuple[Any, dict[str, Any]]:
    w1_masks, w2_masks, joint_meta = build_joint_with_topup_masks(
        calibration,
        calibration.activation_cache,
        config,
        JOINT_MEDIUM_TOPUP_FRACTION,
    )
    plan = build_plan_from_masks(
        mix.name,
        mix.description,
        config,
        non_expert_bytes,
        total_expert_elems,
        w1_masks,
        w2_masks,
    )
    return plan, {
        **joint_meta,
        "calibration_variant": mix.name,
        "channel_metric_requested": "activation_kurtosis",
        "channel_metric_effective": "activation_kurtosis",
        "channel_metric_fallback_used": False,
        "source_base": "joint_w1w2_with_topup",
        "requested_length_counts": serialize_length_counts(mix.length_counts),
        "joint_topup_fraction": float(JOINT_MEDIUM_TOPUP_FRACTION),
    }


def build_selection_summary(results: dict[str, Any], configs: list[CalibrationMixConfig]) -> dict[str, Any]:
    config_names = {config.name for config in configs}
    new_mix_names = [config.name for config in configs if config.name != "maca_original"]
    available_rows = {name: row for name, row in results.items() if name in config_names and "ppl" in row}
    original_row = available_rows.get("maca_original")
    best_new_name: str | None = None
    best_new_row: dict[str, Any] | None = None
    if new_mix_names:
        candidate_rows = [(name, available_rows[name]) for name in new_mix_names if name in available_rows]
        if candidate_rows:
            best_new_name, best_new_row = min(candidate_rows, key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))

    summary: dict[str, Any] = {
        "evaluated_base_configs": sorted(available_rows),
        "missing_base_configs": sorted(config_names.difference(available_rows)),
        "original": None,
        "best_new_mix": None,
        "should_run_correction": False,
    }
    if original_row is not None:
        summary["original"] = {
            "config": "maca_original",
            "ppl": float(original_row["ppl"]),
            "memory_gb": float(original_row["memory_gb"]),
        }
    if best_new_name is not None and best_new_row is not None:
        delta_vs_original = None if original_row is None else round(float(best_new_row["ppl"]) - float(original_row["ppl"]), 6)
        summary["best_new_mix"] = {
            "config": best_new_name,
            "ppl": float(best_new_row["ppl"]),
            "memory_gb": float(best_new_row["memory_gb"]),
            "ppl_delta_vs_original": delta_vs_original,
        }
        summary["should_run_correction"] = bool(original_row is not None and float(best_new_row["ppl"]) < float(original_row["ppl"]))
    return summary


def materialize_base_variant(
    mix: CalibrationMixConfig,
    tokenizer: Any,
    train_ids: torch.Tensor,
    seed: int,
    store: WeightStore,
    payload: dict[str, Any],
    output_json: Path,
    calibration_summaries: dict[str, Any],
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[BaseVariant, dict[str, Any]]:
    print(f"\n=== Build {mix.name} calibration set ===", flush=True)
    calibration_set = build_variable_length_calibration_set(tokenizer, train_ids, seed, mix)
    payload["calibration_sets"][mix.name] = calibration_set.metadata
    atomic_json_dump(output_json, payload)

    calibration = run_maca_calibration(store, config, calibration_set.chunks, calibration_set.actual_lengths, device, dtype)
    calibration_summaries[mix.name] = summarize_calibration_artifacts(calibration)
    payload["metadata"]["calibration_summaries"] = calibration_summaries
    atomic_json_dump(output_json, payload)

    plan, extras = build_joint_plan_for_mix(mix, calibration, config, non_expert_bytes, total_expert_elems)
    return BaseVariant(key=mix.name, plan=plan, extras=extras, calibration_set=calibration_set), calibration_summaries


def print_results_table(payload: dict[str, Any], configs: list[CalibrationMixConfig]) -> None:
    results = payload.get("results", {})
    selection = payload.get("selection", {})
    original_ppl = None
    original_row = results.get("maca_original")
    if isinstance(original_row, dict) and "ppl" in original_row:
        original_ppl = float(original_row["ppl"])

    ordered_names = [config.name for config in configs if config.name in results]
    if CORRECTION_NAME in results:
        ordered_names.append(CORRECTION_NAME)

    print("\n" + "=" * 168, flush=True)
    print("proper_iter29_maca_sweep | MaCa calibration length sweep on joint_w1w2_with_topup | WikiText-2 full eval | GPTQ-standard", flush=True)
    print("=" * 168, flush=True)
    print(
        f"{'Config':<28} {'PPL':>10} {'Delta vs orig':>14} {'Memory GB':>12} {'Eval chunks':>12} {'Time s':>10}",
        flush=True,
    )
    print("-" * 168, flush=True)
    for name in ordered_names:
        row = results[name]
        delta_text = "n/a" if original_ppl is None else f"{float(row['ppl']) - original_ppl:+.4f}"
        print(
            f"{name:<28} {float(row['ppl']):>10.4f} {delta_text:>14} {float(row['memory_gb']):>12.3f} {int(row.get('eval_chunks', 0)):>12d} {float(row['time_s']):>10.1f}",
            flush=True,
        )
    print("-" * 168, flush=True)
    best_new = selection.get("best_new_mix")
    if isinstance(best_new, dict):
        print(
            f"best new mix: {best_new['config']} | ppl={float(best_new['ppl']):.6f} | delta_vs_original={float(best_new.get('ppl_delta_vs_original') or 0.0):+.6f}",
            flush=True,
        )
    correction = selection.get(CORRECTION_NAME)
    if isinstance(correction, dict) and correction.get("status") == "skipped":
        print(f"{CORRECTION_NAME}: skipped ({correction.get('reason', 'unknown')})", flush=True)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    configs = build_sweep_configs()
    for mix in configs:
        validate_mix_config(mix)

    requested_names = resolve_requested_plans(args.configs)
    all_names = {config.name for config in configs} | {CORRECTION_NAME}
    if requested_names is not None:
        missing = sorted(requested_names.difference(all_names))
        if missing:
            raise ValueError(f"Unknown configs requested: {', '.join(missing)}")

    correction_requested = requested_names is None or CORRECTION_NAME in requested_names
    selected_base_names = {config.name for config in configs} if correction_requested else {config.name for config in configs if requested_names is not None and config.name in requested_names}
    selected_configs = [config for config in configs if config.name in selected_base_names]

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    overall_start = time.time()
    tokenizer, _standard_calib_chunks, test_ids_full, calib_info, eval_info = load_gptq_standard_data(args.model_id)
    test_ids, eval_slice = maybe_slice_eval_chunks(test_ids_full, int(args.eval_max_chunks))
    eval_info = {**eval_info, **eval_slice}
    train_ids = load_wikitext_train_ids(tokenizer)

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
            "device": args.device,
            "dtype": args.dtype,
            "seed": int(args.seed),
            "evaluation": eval_info,
            "standard_calibration_reference": calib_info,
            "experiment": "Iteration 29 MaCa calibration length sweep",
            "quantization": "simulated quantization: each config rebuilds joint_w1w2_with_topup masks from its own MaCa calibration mix; optional scalar correction reuses the Iteration 28 all-layer repair on the best new mix only.",
            "protocol": {
                "reference": str(SCRIPT_DIR / "proper_eval.py"),
                "full_test_join": True,
                "seqlen": SEQLEN,
                "non_overlapping": True,
                "loss_accumulation": "loss.float() * seqlen",
                "calibration_total_chunks": TOTAL_CALIBRATION_CHUNKS,
                "maca_context_window": int(MACA_PAD_LENGTH),
            },
            "configs": [
                {"name": config.name, "description": config.description, "length_counts": serialize_length_counts(config.length_counts)}
                for config in configs
            ],
            "requested_configs": sorted(requested_names) if requested_names is not None else sorted(all_names),
        },
        "calibration_sets": {},
        "fitting": {},
        "selection": {},
        "results": {},
    }
    if args.output_json.exists():
        try:
            existing = load_json(args.output_json)
            if isinstance(existing, dict):
                payload["metadata"] = {**existing.get("metadata", {}), **payload["metadata"]}
                payload["calibration_sets"] = {**existing.get("calibration_sets", {}), **payload["calibration_sets"]}
                payload["fitting"] = {**existing.get("fitting", {}), **payload["fitting"]}
                payload["selection"] = {**existing.get("selection", {}), **payload["selection"]}
                payload["results"] = {**existing.get("results", {}), **payload["results"]}
                print(f"[resume] loaded existing results from {args.output_json}", flush=True)
        except Exception:
            print(f"[resume] ignoring unreadable existing results at {args.output_json}", flush=True)
    atomic_json_dump(args.output_json, payload)

    store = WeightStore(args.model_id, snapshot_dir, weight_map)
    base_variants: dict[str, BaseVariant] = {}
    calibration_summaries = dict(payload["metadata"].get("calibration_summaries", {}))

    for mix in selected_configs:
        existing_row = payload.get("results", {}).get(mix.name)
        if isinstance(existing_row, dict) and "ppl" in existing_row:
            print(f"[skip] {mix.name} already present in {args.output_json}", flush=True)
            continue

        variant, calibration_summaries = materialize_base_variant(
            mix,
            tokenizer,
            train_ids,
            int(args.seed),
            store,
            payload,
            args.output_json,
            calibration_summaries,
            text_config,
            non_expert_bytes,
            total_expert_elems,
            device,
            dtype,
        )
        base_variants[mix.name] = variant
        evaluate_and_record_plan(
            payload,
            variant.plan,
            variant.extras,
            args.output_json,
            text_config,
            total_expert_elems,
            test_ids,
            snapshot_dir,
            weight_map,
            device,
            dtype,
        )

    selection = build_selection_summary(payload.get("results", {}), selected_configs)
    payload["selection"] = {**payload.get("selection", {}), **selection}
    atomic_json_dump(args.output_json, payload)

    correction_status: dict[str, Any] | None = None
    best_new_mix = selection.get("best_new_mix")
    if correction_requested and isinstance(best_new_mix, dict) and bool(selection.get("should_run_correction", False)):
        best_name = str(best_new_mix["config"])
        if isinstance(payload.get("results", {}).get(CORRECTION_NAME), dict) and "ppl" in payload["results"][CORRECTION_NAME]:
            correction_status = {
                "status": "already_present",
                "selected_base_variant": best_name,
                "trigger": "best_new_mix_beat_maca_original",
            }
        if correction_status is not None:
            payload.setdefault("selection", {})[CORRECTION_NAME] = correction_status
            payload["metadata"]["elapsed_s"] = round(time.time() - overall_start, 1)
            atomic_json_dump(args.output_json, payload)
            print_results_table(payload, configs)
            print(f"\nSaved results -> {args.output_json}", flush=True)
            del store
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return

        if best_name not in base_variants:
            best_mix = next((mix for mix in configs if mix.name == best_name), None)
            if best_mix is None:
                raise RuntimeError(f"Best new mix `{best_name}` is not a known sweep config")
            variant, calibration_summaries = materialize_base_variant(
                best_mix,
                tokenizer,
                train_ids,
                int(args.seed),
                store,
                payload,
                args.output_json,
                calibration_summaries,
                text_config,
                non_expert_bytes,
                total_expert_elems,
                device,
                dtype,
            )
            base_variants[best_name] = variant
        variant = base_variants[best_name]
        corrections, _top10_layers, fit_payload = fit_scalar_corrections_for_variant(
            args.model_id,
            variant,
            text_config,
            snapshot_dir,
            weight_map,
            device,
            dtype,
        )
        payload["fitting"][CORRECTION_NAME] = {
            **fit_payload,
            "selected_base_variant": best_name,
            "correction_scope": "all_layers",
        }
        atomic_json_dump(args.output_json, payload)

        evaluate_and_record_compound_plan(
            args.model_id,
            payload,
            CORRECTION_NAME,
            variant,
            corrections,
            {
                **variant.extras,
                "base_variant": best_name,
                "correction_type": "scalar_affine",
                "correction_scope": "all_layers",
                "correction_layers": list(range(text_config.num_hidden_layers)),
                "fit_variant": best_name,
                "selection_basis": "best_new_mix_vs_maca_original",
            },
            args.output_json,
            text_config,
            total_expert_elems,
            test_ids,
            snapshot_dir,
            weight_map,
            device,
            dtype,
        )
        correction_status = {
            "status": "evaluated",
            "selected_base_variant": best_name,
            "trigger": "best_new_mix_beat_maca_original",
        }
    elif correction_requested:
        reason = "insufficient_base_results"
        if isinstance(best_new_mix, dict) and not bool(selection.get("should_run_correction", False)):
            reason = "no_new_mix_beat_maca_original"
        correction_status = {
            "status": "skipped",
            "reason": reason,
            "selected_base_variant": None if not isinstance(best_new_mix, dict) else str(best_new_mix["config"]),
        }

    if correction_status is not None:
        payload.setdefault("selection", {})[CORRECTION_NAME] = correction_status

    payload["metadata"]["elapsed_s"] = round(time.time() - overall_start, 1)
    atomic_json_dump(args.output_json, payload)
    print_results_table(payload, configs)
    print(f"\nSaved results -> {args.output_json}", flush=True)

    del store
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

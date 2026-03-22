#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import gc
import itertools
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset

from baselines_comparison import resolve_non_expert_bytes
from proper_eval import SEQLEN, atomic_json_dump, dtype_from_name, load_gptq_standard_data
from proper_iter01 import build_plan_from_masks
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from proper_iter11_push_router_affinity import evaluate_and_record_plan
from proper_iter19_union_residual import load_json, load_reference_rows
from proper_iter25_learned_correction import maybe_slice_eval_chunks, resolve_requested_plans
from proper_iter26_maca_calibration import VariableLengthCalibrationSet, load_wikitext_train_ids, run_maca_calibration, summarize_calibration_artifacts
from proper_iter28_maca_plus_correction import BaseVariant, evaluate_and_record_compound_plan, fit_scalar_corrections_for_variant
from spike1_ground_truth import MODEL_ID, WeightStore, build_text_config, load_root_config


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter30_multilingual_maca.json"

MULTILINGUAL_PLAN = "multilingual_maca_joint_topup"
MULTILINGUAL_CORRECTION_PLAN = "multilingual_maca_joint_topup_plus_correction"
ENGLISH_PLAN = "english_maca_joint_topup"

MULTISCALE_LENGTHS = (128, 512, 2048)
CALIBRATION_TOTAL_CHUNKS = 128
CALIBRATION_PAD_LENGTH = SEQLEN
STREAM_SAMPLE_LIMIT = 5000
MIN_TEXT_CHARS = 200


@dataclass(frozen=True)
class StreamSource:
    key: str
    label: str
    dataset_path: str
    dataset_name: str | None
    text_key: str
    fallback_dataset_path: str | None = None
    fallback_dataset_name: str | None = None
    fallback_text_key: str | None = None


@dataclass(frozen=True)
class TokenizedSource:
    token_ids: torch.Tensor
    metadata: dict[str, Any]


STREAM_SOURCES: dict[str, StreamSource] = {
    "chinese": StreamSource(
        key="chinese",
        label="Chinese Wikipedia",
        dataset_path="wikimedia/wikipedia",
        dataset_name="20231101.zh",
        text_key="text",
        fallback_dataset_path="allenai/c4",
        fallback_dataset_name="zh",
        fallback_text_key="text",
    ),
    "code": StreamSource(
        key="code",
        label="CodeParrot GitHub Code Clean",
        dataset_path="codeparrot/github-code-clean",
        dataset_name=None,
        text_key="code",
        fallback_dataset_path="allenai/c4",
        fallback_dataset_name="zh",
        fallback_text_key="text",
    ),
}


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
            "Comma-separated subset of configs to run. Default runs multilingual base, multilingual all-layer correction, "
            "and the English-only MaCa control."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-max-chunks",
        type=int,
        default=0,
        help="Optional cap on WikiText-2 evaluation chunks for smoke tests; 0 means full 145-chunk test.",
    )
    parser.add_argument(
        "--stream-limit",
        type=int,
        default=STREAM_SAMPLE_LIMIT,
        help="Maximum streamed documents to scan per non-English source before tokenization.",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=MIN_TEXT_CHARS,
        help="Minimum raw text length for streamed Chinese/code examples before tokenization.",
    )
    return parser.parse_args()


def build_config_names() -> set[str]:
    return {MULTILINGUAL_PLAN, MULTILINGUAL_CORRECTION_PLAN, ENGLISH_PLAN}


def flatten_token_ids(token_ids: torch.Tensor) -> torch.Tensor:
    if token_ids.ndim == 2:
        return token_ids[0].contiguous()
    return token_ids.contiguous().view(-1)


def tokenize_joined_text(tokenizer: Any, text: str) -> torch.Tensor:
    encoded = tokenizer(text, return_tensors="pt")
    return flatten_token_ids(torch.as_tensor(encoded.input_ids, dtype=torch.long))


def load_stream_texts(source: StreamSource, limit: int, min_text_chars: int) -> tuple[list[str], dict[str, Any]]:
    attempts = [
        {
            "dataset_path": source.dataset_path,
            "dataset_name": source.dataset_name,
            "text_key": source.text_key,
            "used_fallback": False,
        }
    ]
    if source.fallback_dataset_path is not None:
        attempts.append(
            {
                "dataset_path": source.fallback_dataset_path,
                "dataset_name": source.fallback_dataset_name,
                "text_key": source.fallback_text_key or "text",
                "used_fallback": True,
            }
        )

    errors: list[str] = []
    for attempt in attempts:
        start_time = time.time()
        try:
            dataset_kwargs = {"split": "train", "streaming": True}
            if attempt["dataset_name"] is None:
                dataset = load_dataset(attempt["dataset_path"], **dataset_kwargs)
            else:
                dataset = load_dataset(attempt["dataset_path"], attempt["dataset_name"], **dataset_kwargs)

            texts: list[str] = []
            scanned = 0
            kept = 0
            for item in itertools.islice(dataset, limit):
                scanned += 1
                value = item.get(str(attempt["text_key"]))
                if isinstance(value, str) and len(value) > min_text_chars:
                    texts.append(value)
                    kept += 1
                if scanned % 1000 == 0:
                    print(
                        f"[source:{source.key}] scanned {scanned}/{limit} docs from {attempt['dataset_path']}"
                        f"{'' if attempt['dataset_name'] is None else ':' + str(attempt['dataset_name'])} | kept={kept}",
                        flush=True,
                    )

            if not texts:
                raise RuntimeError(
                    f"no usable texts found in {attempt['dataset_path']}"
                    f"{'' if attempt['dataset_name'] is None else ':' + str(attempt['dataset_name'])}"
                )

            metadata = {
                "dataset": attempt["dataset_path"],
                "dataset_name": attempt["dataset_name"],
                "text_key": attempt["text_key"],
                "used_fallback": bool(attempt["used_fallback"]),
                "scanned_documents": int(scanned),
                "kept_documents": int(kept),
                "stream_limit": int(limit),
                "min_text_chars": int(min_text_chars),
                "load_seconds": round(time.time() - start_time, 3),
            }
            return texts, metadata
        except Exception as exc:
            errors.append(
                f"{attempt['dataset_path']}"
                f"{'' if attempt['dataset_name'] is None else ':' + str(attempt['dataset_name'])} -> {exc}"
            )
            print(
                f"[source:{source.key}] failed on {attempt['dataset_path']}"
                f"{'' if attempt['dataset_name'] is None else ':' + str(attempt['dataset_name'])}: {exc}",
                flush=True,
            )

    raise RuntimeError(f"Unable to load {source.label}. Attempts: {' | '.join(errors)}")


def load_tokenized_source(tokenizer: Any, source_key: str, stream_limit: int, min_text_chars: int) -> TokenizedSource:
    if source_key == "english":
        token_ids = flatten_token_ids(load_wikitext_train_ids(tokenizer))
        metadata = {
            "source": "english",
            "label": "WikiText-2 train",
            "dataset": "wikitext",
            "dataset_name": "wikitext-2-raw-v1",
            "split": "train",
            "token_count": int(token_ids.numel()),
            "used_fallback": False,
        }
        return TokenizedSource(token_ids=token_ids, metadata=metadata)

    if source_key not in STREAM_SOURCES:
        raise ValueError(f"Unknown source key: {source_key}")

    source = STREAM_SOURCES[source_key]
    texts, stream_metadata = load_stream_texts(source, stream_limit, min_text_chars)
    joined_text = "\n\n".join(texts)
    token_ids = tokenize_joined_text(tokenizer, joined_text)
    metadata = {
        "source": source.key,
        "label": source.label,
        **stream_metadata,
        "token_count": int(token_ids.numel()),
    }
    return TokenizedSource(token_ids=token_ids, metadata=metadata)


def allocate_length_counts(total_chunks: int) -> list[tuple[int, int]]:
    base = total_chunks // len(MULTISCALE_LENGTHS)
    remainder = total_chunks - (base * len(MULTISCALE_LENGTHS))
    counts: list[tuple[int, int]] = []
    for idx, length in enumerate(MULTISCALE_LENGTHS):
        count = base + (1 if idx < remainder else 0)
        counts.append((int(length), int(count)))
    return counts


def build_multiscale_calibration_set(
    tokenizer: Any,
    source_cache: dict[str, TokenizedSource],
    source_chunk_counts: tuple[tuple[str, int], ...],
    seed: int,
    config_name: str,
    description: str,
) -> VariableLengthCalibrationSet:
    if sum(int(count) for _source, count in source_chunk_counts) != CALIBRATION_TOTAL_CHUNKS:
        raise ValueError(f"{config_name} must use exactly {CALIBRATION_TOTAL_CHUNKS} calibration chunks")

    rng = random.Random(f"iter30:{seed}:{config_name}")
    padded_chunks: list[torch.Tensor] = []
    actual_lengths: list[int] = []
    source_rows: list[dict[str, Any]] = []

    for source_key, source_chunk_count in source_chunk_counts:
        token_source = source_cache[source_key]
        token_ids = token_source.token_ids
        if int(token_ids.numel()) < min(MULTISCALE_LENGTHS):
            raise RuntimeError(f"Source `{source_key}` is too short for Iteration 30 calibration")

        length_counts = allocate_length_counts(int(source_chunk_count))
        source_actual_tokens = 0
        for length, count in length_counts:
            max_start = int(token_ids.shape[0]) - int(length) - 1
            if max_start < 0:
                raise RuntimeError(f"Source `{source_key}` is too short for calibration length {length}")
            for _ in range(count):
                start = rng.randint(0, max_start)
                chunk = token_ids[start : start + length]
                actual_length = int(chunk.shape[0])
                if actual_length < CALIBRATION_PAD_LENGTH:
                    chunk = F.pad(chunk, (0, CALIBRATION_PAD_LENGTH - actual_length))
                padded_chunks.append(chunk.unsqueeze(0))
                actual_lengths.append(actual_length)
                source_actual_tokens += actual_length

        source_rows.append(
            {
                **token_source.metadata,
                "calibration_chunks": int(source_chunk_count),
                "length_counts": [{"length": int(length), "count": int(count)} for length, count in length_counts],
                "actual_token_total": int(source_actual_tokens),
            }
        )

    chunks = torch.cat(padded_chunks, dim=0).contiguous()
    actual_lengths_tensor = torch.as_tensor(actual_lengths, dtype=torch.long)
    actual_token_total = int(actual_lengths_tensor.sum().item())
    padded_token_total = int(chunks.numel())
    metadata = {
        "dataset": "multisource",
        "split": "train",
        "tokenizer": tokenizer.name_or_path,
        "base_seed": int(seed),
        "sampling_seed": f"iter30:{seed}:{config_name}",
        "config_name": config_name,
        "description": description,
        "lengths": [int(length) for length in MULTISCALE_LENGTHS],
        "samples": int(chunks.shape[0]),
        "sample_shape": [int(chunks.shape[0]), int(chunks.shape[1])],
        "padding_value": 0,
        "context_window": int(CALIBRATION_PAD_LENGTH),
        "actual_token_total": actual_token_total,
        "padded_token_total": padded_token_total,
        "real_token_fraction": round(float(actual_token_total) / float(max(padded_token_total, 1)), 6),
        "standard_reference_tokens": int(SEQLEN * chunks.shape[0]),
        "source_chunk_counts": [{"source": str(source), "count": int(count)} for source, count in source_chunk_counts],
        "sources": source_rows,
    }
    return VariableLengthCalibrationSet(chunks=chunks, actual_lengths=actual_lengths_tensor, metadata=metadata)


def build_joint_plan_for_variant(
    plan_name: str,
    description: str,
    calibration: Any,
    calibration_set: VariableLengthCalibrationSet,
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
        plan_name,
        description,
        config,
        non_expert_bytes,
        total_expert_elems,
        w1_masks,
        w2_masks,
    )
    return plan, {
        **joint_meta,
        "calibration_variant": plan_name,
        "channel_metric_requested": "activation_kurtosis",
        "channel_metric_effective": "activation_kurtosis",
        "channel_metric_fallback_used": False,
        "source_base": "joint_w1w2_with_topup",
        "joint_topup_fraction": float(JOINT_MEDIUM_TOPUP_FRACTION),
        "calibration_sources": [row["source"] for row in calibration_set.metadata["source_chunk_counts"]],
        "calibration_lengths": list(calibration_set.metadata["lengths"]),
        "real_token_fraction": float(calibration_set.metadata["real_token_fraction"]),
    }


def print_results_table(payload: dict[str, Any], references: dict[str, dict[str, float]]) -> None:
    results = payload.get("results", {})
    order = [MULTILINGUAL_CORRECTION_PLAN, MULTILINGUAL_PLAN, ENGLISH_PLAN]
    historical_maca = references.get("maca_joint_topup_no_correction")
    historical_correction = references.get("maca_joint_topup_plus_correction_all")

    print("\n" + "=" * 172, flush=True)
    print("proper_iter30_multilingual_maca | multilingual 128/512/2048 calibration mix | full WikiText-2 | GPTQ-standard", flush=True)
    print("=" * 172, flush=True)
    print(
        f"{'Config':<48} {'PPL':>10} {'Delta vs hist':>14} {'Memory GB':>12} {'Eval chunks':>12} {'Time s':>10}",
        flush=True,
    )
    print("-" * 172, flush=True)

    for name in order:
        if name not in results:
            continue
        row = results[name]
        reference = historical_correction if name == MULTILINGUAL_CORRECTION_PLAN else historical_maca
        delta = "n/a" if reference is None else f"{float(row['ppl']) - float(reference['ppl']):+.4f}"
        print(
            f"{name:<48} {float(row['ppl']):>10.4f} {delta:>14} {float(row['memory_gb']):>12.3f} {int(row.get('eval_chunks', 0)):>12d} {float(row['time_s']):>10.1f}",
            flush=True,
        )

    print("-" * 172, flush=True)
    if historical_maca is not None:
        print(
            f"historical MaCa no-correction reference: maca_joint_topup_no_correction -> ppl={float(historical_maca['ppl']):.6f}",
            flush=True,
        )
    if historical_correction is not None:
        print(
            f"historical MaCa + all-layer scalar reference: maca_joint_topup_plus_correction_all -> ppl={float(historical_correction['ppl']):.6f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    requested_names = resolve_requested_plans(args.configs)
    all_names = build_config_names()
    if requested_names is not None:
        missing = sorted(requested_names.difference(all_names))
        if missing:
            raise ValueError(f"Unknown configs requested: {', '.join(missing)}")

    run_multilingual_base = requested_names is None or MULTILINGUAL_PLAN in requested_names or MULTILINGUAL_CORRECTION_PLAN in requested_names
    run_multilingual_correction = requested_names is None or MULTILINGUAL_CORRECTION_PLAN in requested_names
    run_english_base = requested_names is None or ENGLISH_PLAN in requested_names

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    overall_start = time.time()
    tokenizer, _standard_calib_chunks, test_ids_full, calib_info, eval_info = load_gptq_standard_data(args.model_id)
    test_ids, eval_slice = maybe_slice_eval_chunks(test_ids_full, int(args.eval_max_chunks))
    eval_info = {**eval_info, **eval_slice}

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
            "experiment": "Iteration 30 multilingual + multi-scale MaCa calibration",
            "quantization": "simulated quantization: rebuild joint_w1w2_with_topup masks from multilingual or English-only 128/512/2048 calibration sets; optional all-layer scalar correction reuses the Iteration 28 expert-output repair on the multilingual mask.",
            "protocol": {
                "reference": str(SCRIPT_DIR / "proper_eval.py"),
                "full_test_join": True,
                "seqlen": SEQLEN,
                "non_overlapping": True,
                "loss_accumulation": "loss.float() * seqlen",
                "calibration_total_chunks": CALIBRATION_TOTAL_CHUNKS,
                "maca_context_window": CALIBRATION_PAD_LENGTH,
                "multiscale_lengths": list(MULTISCALE_LENGTHS),
                "stream_limit": int(args.stream_limit),
                "min_text_chars": int(args.min_text_chars),
                "forbidden_lengths": [4096],
            },
            "requested_configs": sorted(requested_names) if requested_names is not None else sorted(all_names),
        },
        "calibration_sets": {},
        "fitting": {},
        "results": {},
    }
    if args.output_json.exists():
        try:
            existing = load_json(args.output_json)
            if isinstance(existing, dict):
                payload["metadata"] = {**existing.get("metadata", {}), **payload["metadata"]}
                payload["calibration_sets"] = {**existing.get("calibration_sets", {}), **payload["calibration_sets"]}
                payload["fitting"] = {**existing.get("fitting", {}), **payload["fitting"]}
                payload["results"] = {**existing.get("results", {}), **payload["results"]}
                print(f"[resume] loaded existing results from {args.output_json}", flush=True)
        except Exception:
            print(f"[resume] ignoring unreadable existing results at {args.output_json}", flush=True)
    atomic_json_dump(args.output_json, payload)

    needed_sources = {"english"}
    if run_multilingual_base:
        needed_sources.update({"chinese", "code"})

    source_cache: dict[str, TokenizedSource] = {}
    for source_key in sorted(needed_sources):
        print(f"\n=== Load source: {source_key} ===", flush=True)
        source_cache[source_key] = load_tokenized_source(tokenizer, source_key, int(args.stream_limit), int(args.min_text_chars))
        print(
            f"[source:{source_key}] tokens={int(source_cache[source_key].token_ids.numel())} | meta={json.dumps(source_cache[source_key].metadata, sort_keys=True)}",
            flush=True,
        )

    store = WeightStore(args.model_id, snapshot_dir, weight_map)
    base_variants: dict[str, BaseVariant] = {}
    calibration_summaries = dict(payload["metadata"].get("calibration_summaries", {}))

    base_specs: list[tuple[str, str, tuple[tuple[str, int], ...]]] = []
    if run_multilingual_base:
        base_specs.append(
            (
                MULTILINGUAL_PLAN,
                "Mix 40% English, 35% Chinese, and 25% code calibration chunks across 128/512/2048 tokens, then rebuild joint_w1w2_with_topup with the standard GPTQ evaluation path unchanged.",
                (("english", 52), ("chinese", 44), ("code", 32)),
            )
        )
    if run_english_base:
        base_specs.append(
            (
                ENGLISH_PLAN,
                "Use the same 128/512/2048 MaCa multi-scale calibration schedule but source all 128 chunks from English WikiText-2 train for a direct control against the multilingual mix.",
                (("english", 128),),
            )
        )

    for plan_name, description, source_chunk_counts in base_specs:
        print(f"\n=== Build {plan_name} calibration set ===", flush=True)
        calibration_set = build_multiscale_calibration_set(
            tokenizer,
            source_cache,
            source_chunk_counts,
            args.seed,
            plan_name,
            description,
        )
        payload["calibration_sets"][plan_name] = calibration_set.metadata
        atomic_json_dump(args.output_json, payload)

        calibration = run_maca_calibration(
            store,
            text_config,
            calibration_set.chunks,
            calibration_set.actual_lengths,
            device,
            dtype,
        )
        calibration_summaries[plan_name] = summarize_calibration_artifacts(calibration)
        payload["metadata"]["calibration_summaries"] = calibration_summaries
        atomic_json_dump(args.output_json, payload)

        plan, extras = build_joint_plan_for_variant(
            plan_name,
            description,
            calibration,
            calibration_set,
            text_config,
            non_expert_bytes,
            total_expert_elems,
        )
        base_variants[plan_name] = BaseVariant(key=plan_name, plan=plan, extras=extras, calibration_set=calibration_set)
        evaluate_and_record_plan(
            payload,
            plan,
            extras,
            args.output_json,
            text_config,
            total_expert_elems,
            test_ids,
            snapshot_dir,
            weight_map,
            device,
            dtype,
        )
        del calibration
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if run_multilingual_correction:
        variant = base_variants.get(MULTILINGUAL_PLAN)
        if variant is None:
            raise RuntimeError("Multilingual base variant was required for correction but was not built")
        corrections, _top_layers, fit_payload = fit_scalar_corrections_for_variant(
            args.model_id,
            variant,
            text_config,
            snapshot_dir,
            weight_map,
            device,
            dtype,
        )
        payload["fitting"][MULTILINGUAL_CORRECTION_PLAN] = {
            **fit_payload,
            "selected_base_variant": MULTILINGUAL_PLAN,
            "correction_scope": "all_layers",
        }
        atomic_json_dump(args.output_json, payload)

        evaluate_and_record_compound_plan(
            args.model_id,
            payload,
            MULTILINGUAL_CORRECTION_PLAN,
            variant,
            corrections,
            {
                **variant.extras,
                "base_variant": MULTILINGUAL_PLAN,
                "correction_type": "scalar_affine",
                "correction_scope": "all_layers",
                "correction_layers": list(range(text_config.num_hidden_layers)),
                "fit_variant": MULTILINGUAL_PLAN,
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

    references = load_reference_rows(args.output_json)
    payload["metadata"]["references"] = {
        name: references[name]
        for name in ["joint_w1w2_with_topup", "maca_joint_topup_no_correction", "maca_joint_topup_plus_correction_all"]
        if name in references
    }
    payload["metadata"]["elapsed_s"] = round(time.time() - overall_start, 1)
    atomic_json_dump(args.output_json, payload)
    print_results_table(payload, references)
    print(f"\nSaved results -> {args.output_json}", flush=True)

    del store
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

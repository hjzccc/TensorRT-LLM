#!/usr/bin/env python3
"""Benchmark inter-token latency (ITL) in single-tile mode.

This script measures ITL for Qwen3-30B-A3B-NVFP4 via TensorRT-LLM's PyTorch
LLM API in two modes:
1) Batched ITL estimate from paired synchronous generate() runs.
2) Optional per-token ITL distribution using generate_async(..., streaming=True).
"""

import argparse
import os
import time
from collections.abc import Iterable
from statistics import mean, stdev
from typing import Any

import torch
import torch.cuda.nvtx as nvtx
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig
from tensorrt_llm.llmapi.llm_args import TorchCompileConfig


def _load_wikitext_batches(
    tokenizer: Any,
    prompt_len: int,
    batch_size: int,
    max_batches: int = 0,
) -> tuple[list[list[list[int]]], int]:
    """Load wikitext-103 test set, tokenize, chunk into batches.

    Returns:
        batches: list of batches, each batch is a list of `batch_size` token-id lists
        total_tokens: total tokens in the dataset
    """
    from datasets import load_dataset

    print("Loading wikitext-103 test set...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    full_text = " ".join([t for t in dataset["text"] if t.strip()])
    all_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    total_tokens = len(all_tokens)
    print(f"  Total tokens in dataset: {total_tokens}")

    # Chunk into prompt_len pieces
    num_chunks = total_tokens // prompt_len
    chunks = [all_tokens[i * prompt_len : (i + 1) * prompt_len] for i in range(num_chunks)]
    print(f"  Chunks of {prompt_len} tokens: {num_chunks}")

    # Group into batches of batch_size
    num_batches = len(chunks) // batch_size
    if max_batches > 0:
        num_batches = min(num_batches, max_batches)

    batches = []
    for bi in range(num_batches):
        batch = [list(chunks[bi * batch_size + j]) for j in range(batch_size)]
        batches.append(batch)

    tokens_covered = num_batches * batch_size * prompt_len
    print(f"  Batches: {num_batches} (batch_size={batch_size})")
    print(f"  Tokens covered: {tokens_covered} / {total_tokens} "
          f"({tokens_covered / total_tokens * 100:.1f}%)")

    return batches, total_tokens


def _build_synthetic_batches(
    tokenizer: Any,
    prompt_len: int,
    batch_size: int,
    num_batches: int,
) -> list[list[list[int]]]:
    """Build synthetic batches by repeating a seed sentence (old behavior)."""
    seed = (
        "TensorRT-LLM MoE benchmark prompt. "
        "We measure prefill and decode performance under controlled settings. "
    )
    seed_ids = tokenizer.encode(seed, add_special_tokens=False)
    if not seed_ids:
        raise RuntimeError("tokenizer produced empty token sequence")
    repeats = (prompt_len + len(seed_ids) - 1) // len(seed_ids)
    full = seed_ids * max(repeats, 1)
    prompt_ids = full[:prompt_len]

    batch = [list(prompt_ids) for _ in range(batch_size)]
    return [batch for _ in range(num_batches)]


def _extract_model_from_llm(llm: LLM) -> Any | None:
    candidate_paths = [
        "_executor.engine.model_engine.model",
        "_executor.engine.model",
        "_executor.model_engine.model",
        "_executor.model",
    ]
    for path in candidate_paths:
        obj: Any = llm
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and hasattr(obj, "named_modules"):
            return obj
    return None


def _find_moe_modules(llm: LLM) -> list[tuple[str, Any]]:
    model = _extract_model_from_llm(llm)
    if model is None:
        return []

    modules: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        if hasattr(module, "use_dual_tile"):
            modules.append((name, module))
    return modules


def _set_dual_tile(
    moe_modules: Iterable[tuple[str, Any]],
    enabled: bool,
    threshold: int,
    tactics: tuple[int, int, int, int],
) -> None:
    g1s, g2s, g1l, g2l = tactics
    for _, module in moe_modules:
        module.use_dual_tile = enabled
        module.dual_tile_threshold = threshold
        module.gemm1_small_tactic = g1s
        module.gemm2_small_tactic = g2s
        module.gemm1_large_tactic = g1l
        module.gemm2_large_tactic = g2l


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, col in enumerate(row):
            widths[i] = max(widths[i], len(col))

    fmt = " | ".join("{" + f":<{w}" + "}" for w in widths)
    sep = "-+-".join("-" * w for w in widths)

    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _measure_batch_itl(
    llm: LLM,
    batches: list[list[list[int]]],
    batch_size: int,
    output_len: int,
    warmup: int,
) -> dict[str, float]:
    prefill_sampling_params = SamplingParams(max_tokens=1, temperature=0.0, top_p=1.0)
    full_sampling_params = SamplingParams(max_tokens=output_len, temperature=0.0, top_p=1.0)

    nvtx.range_push(f"warmup/batch_itl/b{batch_size}")
    for wi in range(warmup):
        _ = llm.generate(batches[0], prefill_sampling_params, use_tqdm=False)
        _ = llm.generate(batches[0], full_sampling_params, use_tqdm=False)
    nvtx.range_pop()

    torch.cuda.synchronize()

    prefill_times_ms: list[float] = []
    total_times_ms: list[float] = []
    decode_times_ms: list[float] = []
    itl_per_batch_ms: list[float] = []
    decode_tokens_per_batch = max(output_len - 1, 0) * batch_size

    nvtx.range_push(f"measure/batch_itl/b{batch_size}_o{output_len}")
    for bi, batch in enumerate(batches):
        nvtx.range_push(f"measure/batch_itl/batch{bi}")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = llm.generate(batch, prefill_sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000.0

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        _ = llm.generate(batch, full_sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t1) * 1000.0
        nvtx.range_pop()

        decode_ms = max(total_ms - prefill_ms, 0.0)
        if output_len > 1:
            avg_itl_ms = decode_ms / (output_len - 1)
        else:
            avg_itl_ms = 0.0
        prefill_times_ms.append(prefill_ms)
        total_times_ms.append(total_ms)
        decode_times_ms.append(decode_ms)
        itl_per_batch_ms.append(avg_itl_ms)

        if (bi + 1) % max(1, len(batches) // 5) == 0:
            print(
                f"    batch {bi + 1}/{len(batches)}: "
                f"prefill={prefill_ms:.1f}ms, total={total_ms:.1f}ms, itl={avg_itl_ms:.4f}ms"
            )
    nvtx.range_pop()

    decode_total_ms = sum(decode_times_ms)
    decode_toks_total = len(batches) * decode_tokens_per_batch
    decode_tok_per_sec = (decode_toks_total / (decode_total_ms / 1000.0)) if decode_total_ms > 0 else 0.0

    return {
        "num_batches": float(len(batches)),
        "avg_prefill_ms": mean(prefill_times_ms),
        "avg_total_ms": mean(total_times_ms),
        "avg_decode_ms": mean(decode_times_ms),
        "avg_itl_ms": mean(itl_per_batch_ms),
        "min_itl_ms": min(itl_per_batch_ms),
        "max_itl_ms": max(itl_per_batch_ms),
        "stdev_itl_ms": stdev(itl_per_batch_ms) if len(itl_per_batch_ms) > 1 else 0.0,
        "decode_tok_per_sec": decode_tok_per_sec,
    }


def _collect_streaming_prompts(
    batches: list[list[list[int]]],
    max_prompts: int,
) -> list[list[int]]:
    prompts: list[list[int]] = []
    for batch in batches:
        for prompt in batch:
            prompts.append(list(prompt))
            if len(prompts) >= max_prompts:
                return prompts
    return prompts


def _measure_streaming_itl(
    llm: LLM,
    prompts: list[list[int]],
    output_len: int,
    warmup: int,
) -> dict[str, float]:
    sampling_params = SamplingParams(max_tokens=output_len, temperature=0.0, top_p=1.0)

    if not prompts:
        return {
            "num_prompts": 0.0,
            "num_itl_samples": 0.0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p99_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "stdev_ms": 0.0,
        }

    nvtx.range_push("warmup/streaming_itl")
    for wi in range(warmup):
        for _ in llm.generate_async(prompts[0], sampling_params, streaming=True):
            pass
    nvtx.range_pop()

    itl_samples_ms: list[float] = []
    nvtx.range_push(f"measure/streaming_itl/prompts{len(prompts)}_o{output_len}")
    for pi, prompt in enumerate(prompts):
        nvtx.range_push(f"measure/streaming_itl/prompt{pi}")
        token_timestamps: list[float] = []
        prev_num_tokens = 0
        for step in llm.generate_async(prompt, sampling_params, streaming=True):
            now = time.perf_counter()
            token_ids_diff = step.outputs[0].token_ids_diff
            num_new_tokens = len(token_ids_diff)
            if num_new_tokens == 0:
                current_num_tokens = len(step.outputs[0].token_ids)
                num_new_tokens = max(current_num_tokens - prev_num_tokens, 0)
                prev_num_tokens = current_num_tokens
            else:
                prev_num_tokens += num_new_tokens
            for _ in range(num_new_tokens):
                token_timestamps.append(now)
        nvtx.range_pop()

        if len(token_timestamps) > 1:
            for i in range(1, len(token_timestamps)):
                itl_samples_ms.append((token_timestamps[i] - token_timestamps[i - 1]) * 1000.0)

    nvtx.range_pop()

    if not itl_samples_ms:
        return {
            "num_prompts": float(len(prompts)),
            "num_itl_samples": 0.0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p99_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "stdev_ms": 0.0,
        }

    return {
        "num_prompts": float(len(prompts)),
        "num_itl_samples": float(len(itl_samples_ms)),
        "mean_ms": mean(itl_samples_ms),
        "p50_ms": _percentile(itl_samples_ms, 50),
        "p90_ms": _percentile(itl_samples_ms, 90),
        "p99_ms": _percentile(itl_samples_ms, 99),
        "min_ms": min(itl_samples_ms),
        "max_ms": max(itl_samples_ms),
        "stdev_ms": stdev(itl_samples_ms) if len(itl_samples_ms) > 1 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Single-tile inter-token latency (ITL) benchmark on Qwen3-30B-A3B-NVFP4"
    )
    parser.add_argument("--model", default="nvidia/Qwen3-30B-A3B-NVFP4")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Number of sequences per batch")
    parser.add_argument("--prompt-len", type=int, default=64,
                        help="Tokens per prompt (wikitext chunk size)")
    parser.add_argument("--output-len", type=int, default=16,
                        help="Max output tokens per sequence")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Max batches to run (0=all from dataset)")
    parser.add_argument("--dataset", choices=["wikitext", "synthetic"],
                        default="wikitext",
                        help="Data source: wikitext-103 (realistic) or synthetic (repeated)")
    parser.add_argument("--synthetic-batches", type=int, default=5,
                        help="Number of batches when using synthetic dataset")
    parser.add_argument("--torch-compile", action="store_true", default=False,
                        help="Enable torch.compile for maximum performance")
    parser.add_argument("--streaming-itl", action="store_true", default=False,
                        help="Enable per-token ITL measurement via generate_async(streaming=True)")
    parser.add_argument("--streaming-samples", type=int, default=16,
                        help="Number of prompts to use for streaming ITL")
    args = parser.parse_args()

    if args.output_len < 1:
        raise ValueError("output_len must be >= 1")

    os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

    print("=" * 110)
    print("Single-Tile ITL Benchmark (Qwen3-30B-A3B-NVFP4)")
    print("=" * 110)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Prompt length: {args.prompt_len}")
    print(f"Output length: {args.output_len}")
    print(f"Warmup: {args.warmup}")
    print(f"Max batches: {args.max_batches if args.max_batches > 0 else 'all'}")
    print(f"Streaming ITL: {args.streaming_itl}")
    print(f"Streaming samples: {args.streaming_samples}")
    print(f"Torch compile: {args.torch_compile}")
    print()

    compile_config = TorchCompileConfig(
        enable_fullgraph=True,
        enable_inductor=False,
        enable_piecewise_cuda_graph=True,
    ) if args.torch_compile else None
    cuda_graph_config = CudaGraphConfig(
        max_batch_size=args.batch_size,
    ) if args.torch_compile else None

    llm = LLM(
        model=args.model,
        backend="pytorch",
        torch_compile_config=compile_config,
        cuda_graph_config=cuda_graph_config,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5),
    )

    try:
        moe_modules = _find_moe_modules(llm)
        if not moe_modules:
            raise RuntimeError(
                "No modules with 'use_dual_tile' were found. "
                "Could not access torch model internals from this LLM executor."
            )

        # Force single-tile mode (same behavior as single_tile_M128 mode).
        _set_dual_tile(
            moe_modules,
            enabled=False,
            threshold=32,
            tactics=(2, 5, 1, 4),
        )
        torch.cuda.synchronize()

        if args.dataset == "wikitext":
            batches, _ = _load_wikitext_batches(
                llm.tokenizer, args.prompt_len, args.batch_size, args.max_batches
            )
        else:
            batches = _build_synthetic_batches(
                llm.tokenizer, args.prompt_len, args.batch_size, args.synthetic_batches
            )

        if not batches:
            raise RuntimeError(
                f"No batches created. Dataset too small for batch_size={args.batch_size} "
                f"and prompt_len={args.prompt_len}"
            )

        print(f"\nTotal batches to run: {len(batches)}")
        print()

        batch_stats = _measure_batch_itl(
            llm=llm,
            batches=batches,
            batch_size=args.batch_size,
            output_len=args.output_len,
            warmup=args.warmup,
        )

        streaming_stats: dict[str, float] | None = None
        if args.streaming_itl:
            prompts = _collect_streaming_prompts(batches, max_prompts=args.streaming_samples)
            streaming_stats = _measure_streaming_itl(
                llm=llm,
                prompts=prompts,
                output_len=args.output_len,
                warmup=args.warmup,
            )

        print()
        print("=" * 110)
        print("BATCH ITL RESULTS")
        print("=" * 110)
        _print_table(
            [
                "batches",
                "avg_itl_ms",
                "avg_prefill_ms",
                "avg_decode_ms",
                "avg_total_ms",
                "decode_tok/s",
                "itl_min_ms",
                "itl_max_ms",
                "itl_stdev_ms",
            ],
            [[
                str(int(batch_stats["num_batches"])),
                f"{batch_stats['avg_itl_ms']:.4f}",
                f"{batch_stats['avg_prefill_ms']:.2f}",
                f"{batch_stats['avg_decode_ms']:.2f}",
                f"{batch_stats['avg_total_ms']:.2f}",
                f"{batch_stats['decode_tok_per_sec']:.0f}",
                f"{batch_stats['min_itl_ms']:.4f}",
                f"{batch_stats['max_itl_ms']:.4f}",
                f"{batch_stats['stdev_itl_ms']:.4f}",
            ]],
        )

        if streaming_stats is not None:
            print()
            print("=" * 110)
            print("STREAMING ITL RESULTS")
            print("=" * 110)
            _print_table(
                [
                    "prompts",
                    "itl_samples",
                    "mean_ms",
                    "p50_ms",
                    "p90_ms",
                    "p99_ms",
                    "min_ms",
                    "max_ms",
                    "stdev_ms",
                ],
                [[
                    str(int(streaming_stats["num_prompts"])),
                    str(int(streaming_stats["num_itl_samples"])),
                    f"{streaming_stats['mean_ms']:.4f}",
                    f"{streaming_stats['p50_ms']:.4f}",
                    f"{streaming_stats['p90_ms']:.4f}",
                    f"{streaming_stats['p99_ms']:.4f}",
                    f"{streaming_stats['min_ms']:.4f}",
                    f"{streaming_stats['max_ms']:.4f}",
                    f"{streaming_stats['stdev_ms']:.4f}",
                ]],
            )

    finally:
        llm.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

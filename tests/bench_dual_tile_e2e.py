#!/usr/bin/env python3
"""Benchmark single-tile vs dual-tile MoE with real end-to-end generation.

Uses wikitext-103 dataset for realistic expert routing patterns.
Runs the full Qwen3-30B-A3B-NVFP4 model via TensorRT-LLM LLM API,
measures real prefill + decode, and toggles dual-tile on live CutlassFusedMoE
instances in the loaded model.
"""

import argparse
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Any

from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig
import torch
import torch.cuda.nvtx as nvtx

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi.llm_args import TorchCompileConfig


@dataclass
class ModeConfig:
    name: str
    dual_tile_enabled: bool
    # For "forced single-tile" modes: use dual-tile machinery with a huge threshold
    # so all experts go to the "small" group, which uses the specified small tactic.
    forced_threshold: int | None = None  # None = use args.dual_tile_threshold
    forced_small_tactics: tuple[int, int] | None = None  # (gemm1, gemm2) for small group

def _parse_int_csv(value: str) -> list[int]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise ValueError("expected at least one integer")
    return [int(x) for x in items]


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


def _format(v: float) -> str:
    return f"{v:.2f}"


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


def _run_dataset_benchmark(
    llm: LLM,
    mode: ModeConfig,
    batches: list[list[list[int]]],
    output_len: int,
    batch_size: int,
    prompt_len: int,
    warmup: int,
) -> dict[str, Any]:
    """Run all dataset batches through the model and collect timing."""

    sampling_params = SamplingParams(
        max_tokens=output_len,
        temperature=0.0,
        top_p=1.0,
    )

    # Warmup with first batch
    for wi in range(warmup):
        nvtx.range_push(f"warmup/{mode.name}/b{batch_size}/{wi}")
        _ = llm.generate(batches[0], sampling_params, use_tqdm=False)
        nvtx.range_pop()

    torch.cuda.synchronize()

    # Run all batches
    batch_times_ms: list[float] = []
    total_start = time.perf_counter()

    for bi, batch in enumerate(batches):
        nvtx.range_push(f"run/{mode.name}/b{batch_size}/batch{bi}")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = llm.generate(batch, sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        nvtx.range_pop()
        batch_times_ms.append(elapsed_ms)

        if (bi + 1) % max(1, len(batches) // 5) == 0:
            avg_so_far = mean(batch_times_ms)
            print(f"    batch {bi + 1}/{len(batches)}: "
                  f"this={elapsed_ms:.1f}ms, avg={avg_so_far:.1f}ms")

    total_ms = (time.perf_counter() - total_start) * 1000.0

    avg_batch_ms = mean(batch_times_ms)
    std_batch_ms = stdev(batch_times_ms) if len(batch_times_ms) > 1 else 0.0
    min_batch_ms = min(batch_times_ms)
    max_batch_ms = max(batch_times_ms)

    total_input_tokens = len(batches) * batch_size * prompt_len
    total_output_tokens = len(batches) * batch_size * output_len
    input_throughput = total_input_tokens / (total_ms / 1000.0)
    output_throughput = total_output_tokens / (total_ms / 1000.0)

    return {
        "mode": mode.name,
        "num_batches": len(batches),
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "output_len": output_len,
        "total_ms": total_ms,
        "avg_batch_ms": avg_batch_ms,
        "std_batch_ms": std_batch_ms,
        "min_batch_ms": min_batch_ms,
        "max_batch_ms": max_batch_ms,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_tok_per_sec": input_throughput,
        "output_tok_per_sec": output_throughput,
        "batch_times_ms": batch_times_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="E2E single-tile vs dual-tile MoE benchmark on Qwen3-30B-A3B-NVFP4"
    )
    parser.add_argument("--model", default="nvidia/Qwen3-30B-A3B-NVFP4")
    parser.add_argument("--batch-size", type=int, default=512,
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
    parser.add_argument("--dual-tile-threshold", type=int, default=32)
    parser.add_argument(
        "--dual-tile-tactics",
        nargs=4,
        type=int,
        metavar=("G1S", "G2S", "G1L", "G2L"),
        default=[2, 5, 1, 4],
    )
    args = parser.parse_args()

    if args.output_len > 512:
        raise ValueError("output_len must be <= 512")

    os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

    print("=" * 110)
    print("Dual-Tile MoE E2E Benchmark (wikitext dataset)")
    print("=" * 110)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Prompt length: {args.prompt_len}")
    print(f"Output length: {args.output_len}")
    print(f"Warmup: {args.warmup}")
    print(f"Max batches: {args.max_batches if args.max_batches > 0 else 'all'}")
    print(
        "Dual-tile config: "
        f"threshold={args.dual_tile_threshold}, tactics={tuple(args.dual_tile_tactics)}"
    )
    print(f"Torch compile: {args.torch_compile}")
    print()

    # Load model
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

    moe_modules = _find_moe_modules(llm)
    if not moe_modules:
        raise RuntimeError(
            "No modules with 'use_dual_tile' were found. "
            "Could not access torch model internals from this LLM executor."
        )

    print(f"Found {len(moe_modules)} MoE modules with dual-tile attributes.")
    print("First few modules:")
    for name, _ in moe_modules[:5]:
        print(f"  - {name}")
    if len(moe_modules) > 5:
        print(f"  ... ({len(moe_modules) - 5} more)")
    print()

    # Load dataset
    if args.dataset == "wikitext":
        batches, total_dataset_tokens = _load_wikitext_batches(
            llm.tokenizer, args.prompt_len, args.batch_size, args.max_batches
        )
    else:
        batches = _build_synthetic_batches(
            llm.tokenizer, args.prompt_len, args.batch_size, args.synthetic_batches
        )
        total_dataset_tokens = args.synthetic_batches * args.batch_size * args.prompt_len

    if not batches:
        raise RuntimeError(
            f"No batches created. Dataset too small for batch_size={args.batch_size} "
            f"and prompt_len={args.prompt_len}"
        )

    print(f"\nTotal batches to run: {len(batches)}")
    print()

    # Modes to test:
    # 1. single_tile_M128: baseline — auto-tuner picks M128 (default for prefill-tuned profile)
    # 2. single_tile_M32:  forced M32 via dual-tile with threshold=999999 (all experts → small group)
    #                      Small overhead from dual-tile path (~0.2%), but gives us M32 tile for all experts.
    # 3. dual_tile:        M32 for small experts + M64 for large experts (two-pass)
    #
    # Tactic mapping (FAST_BUILD SM120):
    #   GEMM1: 0=M128, 1=M64, 2=M32  (no-swap variants)
    #   GEMM2: 0=M128-DEFAULT, 1=M64-DEFAULT, 2=M32-DEFAULT,
    #          3=M128-FINALIZE, 4=M64-FINALIZE, 5=M32-FINALIZE
    g1s, g2s, g1l, g2l = args.dual_tile_tactics
    modes = [
        ModeConfig(name="single_tile_M128", dual_tile_enabled=False),
        ModeConfig(
            name="single_tile_M32",
            dual_tile_enabled=True,
            forced_threshold=999999,  # all experts → small group
            forced_small_tactics=(2, 5),  # GEMM1=M32-noswap, GEMM2=M32-FINALIZE
        ),
        ModeConfig(name="dual_tile", dual_tile_enabled=True),
    ]
    tactics = tuple(args.dual_tile_tactics)

    results: list[dict[str, Any]] = []

    torch.cuda.cudart().cudaProfilerStart()

    for mode in modes:
        print(f"\n{'=' * 80}")
        print(f"MODE: {mode.name}")
        print(f"{'=' * 80}")

        if mode.forced_threshold is not None and mode.forced_small_tactics is not None:
            # Forced single-tile mode: use dual-tile path with huge threshold
            # so all experts go to "small" group with specified tactic.
            fg1s, fg2s = mode.forced_small_tactics
            _set_dual_tile(
                moe_modules,
                enabled=True,
                threshold=mode.forced_threshold,
                tactics=(fg1s, fg2s, g1l, g2l),  # large tactics don't matter (0 experts)
            )
            print(f"  (forcing all experts to small group with threshold={mode.forced_threshold}, "
                  f"small tactics=({fg1s}, {fg2s}))")
        else:
            _set_dual_tile(
                moe_modules,
                enabled=mode.dual_tile_enabled,
                threshold=args.dual_tile_threshold,
                tactics=tactics,
            )
        torch.cuda.synchronize()

        nvtx.range_push(f"dataset/{mode.name}/b{args.batch_size}_p{args.prompt_len}_o{args.output_len}")
        result = _run_dataset_benchmark(
            llm=llm,
            mode=mode,
            batches=batches,
            output_len=args.output_len,
            batch_size=args.batch_size,
            prompt_len=args.prompt_len,
            warmup=args.warmup,
        )
        nvtx.range_pop()

        results.append(result)

        print(f"\n  {mode.name} summary:")
        print(f"    Total time:       {result['total_ms']:.1f} ms")
        print(f"    Avg batch time:   {result['avg_batch_ms']:.1f} ± {result['std_batch_ms']:.1f} ms")
        print(f"    Min/Max batch:    {result['min_batch_ms']:.1f} / {result['max_batch_ms']:.1f} ms")
        print(f"    Input throughput: {result['input_tok_per_sec']:.0f} tok/s")
        print(f"    Output throughput:{result['output_tok_per_sec']:.0f} tok/s")

    torch.cuda.cudart().cudaProfilerStop()

    # ============== Results ==============
    print()
    print("=" * 110)
    print("RESULTS")
    print("=" * 110)

    # Per-mode table
    rows = []
    for r in results:
        rows.append([
            r["mode"],
            str(r["num_batches"]),
            _format(r["total_ms"]),
            _format(r["avg_batch_ms"]),
            _format(r["std_batch_ms"]),
            _format(r["min_batch_ms"]),
            _format(r["max_batch_ms"]),
            f"{r['input_tok_per_sec']:.0f}",
            f"{r['output_tok_per_sec']:.0f}",
        ])

    _print_table(
        [
            "mode",
            "batches",
            "total_ms",
            "avg_batch_ms",
            "std_batch_ms",
            "min_batch_ms",
            "max_batch_ms",
            "input_tok/s",
            "output_tok/s",
        ],
        rows,
    )

    # Pairwise comparison against baseline (first result)
    baseline = results[0]
    if len(results) > 1:
        print()
        print("=" * 110)
        print(f"COMPARISON (all modes vs {baseline['mode']})")
        print("=" * 110)

        for r in results[1:]:
            speedup = baseline["avg_batch_ms"] / r["avg_batch_ms"]
            delta_pct = (1 - r["avg_batch_ms"] / baseline["avg_batch_ms"]) * 100
            sign = "+" if delta_pct > 0 else ""
            print(f"  {r['mode']:>20s} vs {baseline['mode']:>20s}: "
                  f"avg batch {r['avg_batch_ms']:.1f}ms vs {baseline['avg_batch_ms']:.1f}ms  "
                  f"({sign}{delta_pct:.1f}%, {speedup:.4f}x)")

        # Per-batch comparison table
        print()
        print("Per-batch latency comparison (ms):")
        hdr = f"  {'batch':>6s}"
        for r in results:
            hdr += f"  {r['mode']:>18s}"
        print(hdr)
        print(f"  {'-' * (8 + 20 * len(results))}")
        num_batches = min(len(r["batch_times_ms"]) for r in results)
        for bi in range(num_batches):
            line = f"  {bi:>6d}"
            for r in results:
                line += f"  {r['batch_times_ms'][bi]:>18.1f}"
            print(line)

        # Summary verdict
        print()
        for r in results[1:]:
            speedup = baseline["total_ms"] / r["total_ms"]
            if speedup > 1.01:
                print(f">>> {r['mode']} is {(speedup - 1) * 100:.1f}% FASTER than {baseline['mode']}")
            elif speedup < 0.99:
                print(f">>> {r['mode']} is {(1 - speedup) * 100:.1f}% SLOWER than {baseline['mode']}")
            else:
                print(f">>> {r['mode']} vs {baseline['mode']}: NO SIGNIFICANT DIFFERENCE ({speedup:.4f}x)")

    llm.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

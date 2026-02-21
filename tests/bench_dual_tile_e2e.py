#!/usr/bin/env python3
"""Benchmark single-tile vs dual-tile MoE with real end-to-end generation.

This script uses TensorRT-LLM LLM API on the full Qwen3-30B-A3B-NVFP4 model,
measures real prefill + decode, and toggles dual-tile on live CutlassFusedMoE
instances in the loaded model.
"""

import argparse
import itertools
import os
import time
from dataclasses import dataclass
from statistics import mean
from collections.abc import Iterable
from typing import Any

import torch

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.metrics.enums import MetricNames


@dataclass
class ModeConfig:
    name: str
    dual_tile_enabled: bool


@dataclass
class Scenario:
    batch_size: int
    prompt_len: int
    output_len: int


def _parse_int_csv(value: str) -> list[int]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise ValueError("expected at least one integer")
    return [int(x) for x in items]


def _metric(metrics_dict: dict[Any, Any], metric_name: MetricNames) -> float | None:
    if not metrics_dict:
        return None
    if metric_name in metrics_dict:
        return float(metrics_dict[metric_name])
    raw_key = metric_name.value
    if raw_key in metrics_dict:
        return float(metrics_dict[raw_key])
    return None


def _build_prompt_token_ids(tokenizer: Any, target_len: int) -> list[int]:
    seed = (
        "TensorRT-LLM MoE benchmark prompt. "
        "We measure prefill and decode performance under controlled settings. "
    )
    seed_ids = tokenizer.encode(seed, add_special_tokens=False)
    if not seed_ids:
        raise RuntimeError("tokenizer produced empty token sequence")
    repeats = (target_len + len(seed_ids) - 1) // len(seed_ids)
    full = seed_ids * max(repeats, 1)
    return full[:target_len]


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


def _summarize_numbers(values: list[float]) -> tuple[float, float, float]:
    return mean(values), min(values), max(values)


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


def _run_scenario(
    llm: LLM,
    mode: ModeConfig,
    scenario: Scenario,
    warmup: int,
    runs: int,
) -> dict[str, float | int | str]:
    prompt_token_ids = _build_prompt_token_ids(llm.tokenizer, scenario.prompt_len)
    prompts = [list(prompt_token_ids) for _ in range(scenario.batch_size)]

    sampling_params = SamplingParams(
        max_tokens=scenario.output_len,
        temperature=0.0,
        top_p=1.0,
        return_perf_metrics=True,
    )

    for _ in range(warmup):
        _ = llm.generate(prompts, sampling_params, use_tqdm=False)

    batch_latencies_ms: list[float] = []
    req_e2e_ms: list[float] = []
    req_ttft_ms: list[float] = []
    req_decode_tps: list[float] = []

    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        batch_latencies_ms.append(elapsed_ms)

        if not isinstance(outputs, list):
            outputs = [outputs]

        for out in outputs:
            metrics_dict = out.metrics_dict or {}
            e2e_s = _metric(metrics_dict, MetricNames.E2E)
            ttft_s = _metric(metrics_dict, MetricNames.TTFT)

            if e2e_s is None:
                e2e_s = elapsed_ms / 1000.0 / scenario.batch_size
            if ttft_s is None:
                ttft_s = e2e_s

            e2e_ms = e2e_s * 1000.0
            ttft_ms = ttft_s * 1000.0
            req_e2e_ms.append(e2e_ms)
            req_ttft_ms.append(ttft_ms)

            output_tokens = len(out.outputs[0].token_ids or [])
            decode_time_s = max(e2e_s - ttft_s, 1e-9)
            req_decode_tps.append(output_tokens / decode_time_s)

    batch_mean, batch_min, batch_max = _summarize_numbers(batch_latencies_ms)
    req_mean, req_min, req_max = _summarize_numbers(req_e2e_ms)
    ttft_mean, ttft_min, ttft_max = _summarize_numbers(req_ttft_ms)
    decode_tps_mean, _, _ = _summarize_numbers(req_decode_tps)

    return {
        "mode": mode.name,
        "batch_size": scenario.batch_size,
        "prompt_len": scenario.prompt_len,
        "output_len": scenario.output_len,
        "batch_latency_ms_mean": batch_mean,
        "batch_latency_ms_min": batch_min,
        "batch_latency_ms_max": batch_max,
        "req_latency_ms_mean": req_mean,
        "req_latency_ms_min": req_min,
        "req_latency_ms_max": req_max,
        "ttft_ms_mean": ttft_mean,
        "ttft_ms_min": ttft_min,
        "ttft_ms_max": ttft_max,
        "decode_tps_mean": decode_tps_mean,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end single-tile vs dual-tile MoE benchmark on Qwen3-30B-A3B-NVFP4"
    )
    parser.add_argument("--model", default="nvidia/Qwen3-30B-A3B-NVFP4")
    parser.add_argument("--batch-sizes", type=_parse_int_csv, default=[1, 2, 4])
    parser.add_argument("--prompt-lengths", type=_parse_int_csv, default=[64, 256])
    parser.add_argument("--output-lengths", type=_parse_int_csv, default=[32, 64])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--dual-tile-threshold", type=int, default=16)
    parser.add_argument(
        "--dual-tile-tactics",
        nargs=4,
        type=int,
        metavar=("G1S", "G2S", "G1L", "G2L"),
        default=[0, 0, 0, 0],
    )
    args = parser.parse_args()

    if any(x > 512 for x in args.output_lengths):
        raise ValueError("all output lengths must be <= 512")

    os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

    print("=" * 110)
    print("Dual-Tile MoE E2E Benchmark (real generation: prefill + decode)")
    print("=" * 110)
    print(f"Model: {args.model}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Prompt lengths: {args.prompt_lengths}")
    print(f"Output lengths: {args.output_lengths}")
    print(f"Warmup: {args.warmup}, Runs: {args.runs}")
    print(
        "Dual-tile config: "
        f"threshold={args.dual_tile_threshold}, tactics={tuple(args.dual_tile_tactics)}"
    )
    print()

    llm = LLM(model=args.model, backend="pytorch")

    moe_modules = _find_moe_modules(llm)
    if not moe_modules:
        raise RuntimeError(
            "No modules with 'use_dual_tile' were found. "
            "Could not access torch model internals from this LLM executor."
        )

    print(f"Found {len(moe_modules)} MoE modules with dual-tile attributes.")
    print("First few modules:")
    for name, _ in moe_modules[:8]:
        print(f"  - {name}")
    if len(moe_modules) > 8:
        print(f"  ... ({len(moe_modules) - 8} more)")
    print()

    modes = [
        ModeConfig(name="single_tile", dual_tile_enabled=False),
        ModeConfig(name="dual_tile", dual_tile_enabled=True),
    ]

    scenarios = [
        Scenario(batch_size=b, prompt_len=p, output_len=o)
        for b, p, o in itertools.product(
            args.batch_sizes, args.prompt_lengths, args.output_lengths)
    ]

    results: list[dict[str, float | int | str]] = []
    tactics = tuple(args.dual_tile_tactics)

    for scenario in scenarios:
        print(
            f"Running scenario: batch={scenario.batch_size}, "
            f"prompt_len={scenario.prompt_len}, output_len={scenario.output_len}"
        )
        for mode in modes:
            _set_dual_tile(
                moe_modules,
                enabled=mode.dual_tile_enabled,
                threshold=args.dual_tile_threshold,
                tactics=tactics,
            )
            torch.cuda.synchronize()
            print(f"  -> mode={mode.name}")
            result = _run_scenario(
                llm=llm,
                mode=mode,
                scenario=scenario,
                warmup=args.warmup,
                runs=args.runs,
            )
            results.append(result)

    print()
    print("=" * 110)
    print("Per-mode metrics")
    print("=" * 110)

    rows = []
    for item in results:
        rows.append([
            item["mode"],
            str(item["batch_size"]),
            str(item["prompt_len"]),
            str(item["output_len"]),
            _format(float(item["batch_latency_ms_mean"])),
            _format(float(item["req_latency_ms_mean"])),
            _format(float(item["ttft_ms_mean"])),
            _format(float(item["decode_tps_mean"])),
        ])

    _print_table(
        [
            "mode",
            "batch",
            "prompt_tok",
            "output_tok",
            "batch_latency_ms(avg)",
            "req_latency_ms(avg)",
            "ttft_ms(avg)",
            "decode_tok_s(avg)",
        ],
        rows,
    )

    print()
    print("=" * 110)
    print("Dual-tile comparison")
    print("=" * 110)

    by_scenario = {}
    for item in results:
        key = (item["batch_size"], item["prompt_len"], item["output_len"])
        by_scenario.setdefault(key, {})[item["mode"]] = item

    cmp_rows = []
    for key in sorted(by_scenario):
        item = by_scenario[key]
        single = item["single_tile"]
        dual = item["dual_tile"]
        latency_speedup = float(single["batch_latency_ms_mean"]) / float(
            dual["batch_latency_ms_mean"])
        decode_speedup = float(dual["decode_tps_mean"]) / float(
            single["decode_tps_mean"])
        cmp_rows.append([
            str(key[0]),
            str(key[1]),
            str(key[2]),
            _format(float(single["batch_latency_ms_mean"])),
            _format(float(dual["batch_latency_ms_mean"])),
            f"{latency_speedup:.3f}x",
            _format(float(single["ttft_ms_mean"])),
            _format(float(dual["ttft_ms_mean"])),
            f"{decode_speedup:.3f}x",
        ])

    _print_table(
        [
            "batch",
            "prompt_tok",
            "output_tok",
            "single_batch_ms",
            "dual_batch_ms",
            "latency_speedup(single/dual)",
            "single_ttft_ms",
            "dual_ttft_ms",
            "decode_tps_speedup(dual/single)",
        ],
        cmp_rows,
    )

    llm.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

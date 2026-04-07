"""End-to-end pipeline: BF16 -> NVFP4 -> sub-NVFP4 -> NVFP4 -> BF16 -> eval.

Orchestrates the full experiment pipeline with proper artifact separation.

Usage:
    python run.py \
        --model-path /data/junzhou/models/Qwen3-30B-A3B \
        --work-dir /data/junzhou/exp/run01 \
        --tp-size 8 \
        --steps quantize compress dequantize evaluate
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def run_step(script: str, args: list[str], label: str) -> None:
    """Run a pipeline step script."""
    cmd = [sys.executable, str(SCRIPT_DIR / script)] + args
    print(f"\n{'='*60}")
    print(f"Step: {label}")
    print(f"Cmd:  {' '.join(cmd)}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        raise RuntimeError(f"Step '{label}' failed with exit code {result.returncode}")
    print(f"\n{label} completed in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end NVFP4 compression pipeline")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to original BF16 HF model")
    parser.add_argument("--work-dir", type=str, required=True,
                        help="Working directory for all artifacts and outputs")
    parser.add_argument("--quantize-scope", type=str, default="all",
                        choices=["all", "experts-only"],
                        help="Which layers to quantize in the BF16 -> NVFP4 step. Default: all")
    parser.add_argument("--compress-scope", type=str, default="experts-only",
                        choices=["all", "experts-only"],
                        help="Which layers to apply sub-NVFP4 codebook compression to. Default: experts-only")
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--steps", nargs="+",
                        default=["quantize", "compress", "dequantize-nvfp4",
                                 "dequantize-compressed", "evaluate"],
                        help="Which steps to run")
    parser.add_argument("--tasks", nargs="+", default=["mmlu", "gsm8k"],
                        help="Evaluation tasks")
    args = parser.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    model = args.model_path

    # Artifact paths
    nvfp4_artifacts = str(work / "artifacts" / "nvfp4")
    compressed_artifacts = str(work / "artifacts" / "nvfp4_compressed")
    nvfp4_eval_ckpt = str(work / "checkpoints" / "nvfp4-eval")
    compressed_eval_ckpt = str(work / "checkpoints" / "compressed-eval")
    nvfp4_results = str(work / "results" / "nvfp4")
    compressed_results = str(work / "results" / "compressed")

    # Save config
    config = {
        "model_path": model,
        "work_dir": str(work),
        "quantize_scope": args.quantize_scope,
        "compress_scope": args.compress_scope,
        "tp_size": args.tp_size,
        "steps": args.steps,
        "tasks": args.tasks,
    }
    (work / "config.json").write_text(json.dumps(config, indent=2))

    t_total = time.time()

    if "quantize" in args.steps:
        run_step("quantize.py", [
            "--model-path", model,
            "--output-path", nvfp4_artifacts,
            "--scope", args.quantize_scope,
            "--device", args.device,
        ], "BF16 -> NVFP4 quantization")

    if "compress" in args.steps:
        run_step("compress.py", [
            "--input-path", nvfp4_artifacts,
            "--output-path", compressed_artifacts,
            "--input-format", "artifact",
            "--scope", args.compress_scope,
            "--device", args.device,
        ], "NVFP4 -> sub-NVFP4 codebook compression")

    if "dequantize-nvfp4" in args.steps:
        run_step("dequantize.py", [
            "--original-model", model,
            "--artifact-path", nvfp4_artifacts,
            "--output-path", nvfp4_eval_ckpt,
            "--device", args.device,
        ], "NVFP4 -> BF16 eval checkpoint")

    if "dequantize-compressed" in args.steps:
        run_step("dequantize.py", [
            "--original-model", model,
            "--artifact-path", compressed_artifacts,
            "--base-artifact-path", nvfp4_artifacts,
            "--output-path", compressed_eval_ckpt,
            "--device", args.device,
        ], "Compressed -> BF16 eval checkpoint")

    if "evaluate" in args.steps:
        task_args = []
        for t in args.tasks:
            task_args.extend(["--tasks", t])

        # Eval NVFP4 checkpoint
        run_step("evaluate.py", [
            "--model-path", nvfp4_eval_ckpt,
            "--tp-size", str(args.tp_size),
            "--output-path", nvfp4_results,
        ] + task_args, "Evaluate NVFP4")

        # Eval compressed checkpoint
        run_step("evaluate.py", [
            "--model-path", compressed_eval_ckpt,
            "--tp-size", str(args.tp_size),
            "--output-path", compressed_results,
        ] + task_args, "Evaluate Compressed")

    elapsed_total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Pipeline complete. Total time: {elapsed_total:.1f}s")
    print(f"Work directory: {work}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

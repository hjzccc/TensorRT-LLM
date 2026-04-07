"""End-to-end INT3 pipeline runner.

This mirrors the NVFP4 run orchestrator but targets INT3 artifacts.
Evaluation still happens through BF16 checkpoints rebuilt from those artifacts.
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
    cmd = [sys.executable, str(SCRIPT_DIR / script)] + args
    print(f"\n{'=' * 60}")
    print(f"Step: {label}")
    print(f"Cmd:  {' '.join(cmd)}")
    print(f"{'=' * 60}\n")
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(f"Step '{label}' failed with exit code {result.returncode}")
    print(f"\n{label} completed in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end INT3 pipeline")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--work-dir", type=str, required=True)
    parser.add_argument("--method", type=str, default="gptq-sym",
                        choices=["rtn-sym", "gptq-sym", "gptq-asym"])
    parser.add_argument("--scope", type=str, default="all",
                        choices=["all", "experts-only"])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--activation-order", action="store_true", default=False)
    parser.add_argument("--hessian-dir", type=str, default=None)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--steps", nargs='+', default=["quantize", "dequantize", "evaluate"])
    parser.add_argument("--tasks", nargs='+', default=["gsm8k", "mmlu"])
    args = parser.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    artifacts = str(work / 'artifacts' / 'int3')
    eval_ckpt = str(work / 'checkpoints' / 'int3-eval')
    results = str(work / 'results' / 'int3')

    (work / 'config.json').write_text(json.dumps({
        'model_path': args.model_path,
        'method': args.method,
        'scope': args.scope,
        'group_size': args.group_size,
        'activation_order': args.activation_order,
        'hessian_dir': args.hessian_dir,
        'tp_size': args.tp_size,
        'tasks': args.tasks,
    }, indent=2))

    if 'quantize' in args.steps:
        quant_args = [
            '--model-path', args.model_path,
            '--output-path', artifacts,
            '--method', args.method,
            '--scope', args.scope,
            '--group-size', str(args.group_size),
            '--device', args.device,
        ]
        if args.hessian_dir:
            quant_args += ['--hessian-dir', args.hessian_dir]
        if args.activation_order:
            quant_args += ['--activation-order']
        run_step('int3_quantize.py', quant_args, 'BF16 -> INT3 quantization')

    if 'dequantize' in args.steps:
        run_step('int3_dequantize.py', [
            '--original-model', args.model_path,
            '--artifact-path', artifacts,
            '--output-path', eval_ckpt,
            '--device', args.device,
        ], 'INT3 -> BF16 eval checkpoint')

    if 'evaluate' in args.steps:
        task_args = []
        for task in args.tasks:
            task_args += ['--tasks', task]
        run_step('evaluate.py', [
            '--model-path', eval_ckpt,
            '--tp-size', str(args.tp_size),
            '--output-path', results,
        ] + task_args, 'Evaluate INT3 BF16 checkpoint')


if __name__ == '__main__':
    main()

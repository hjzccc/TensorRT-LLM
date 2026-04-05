#!/usr/bin/env python3
"""Run TRT-LLM built-in MMLU evaluation on an NVFP4 checkpoint.

Usage:
  python3 eval_trtllm_mmlu.py --model nvidia/Qwen3-30B-A3B-NVFP4
  python3 eval_trtllm_mmlu.py --model /path/to/nvfp4_checkpoint
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--max-num-tokens", type=int, default=4096)
    parser.add_argument("--kv-fraction", type=float, default=0.15)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"EVAL_TRTLLM: model={args.model}", flush=True)
    print(f"EVAL_TRTLLM: max_batch_size={args.max_batch_size}", flush=True)
    print(f"EVAL_TRTLLM: max_num_tokens={args.max_num_tokens}", flush=True)
    print(f"EVAL_TRTLLM: kv_fraction={args.kv_fraction}", flush=True)

    t0 = time.time()

    print("EVAL_TRTLLM: importing tensorrt_llm...", flush=True)
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import KvCacheConfig
    from tensorrt_llm.evaluate.mmlu import MMLU

    print("EVAL_TRTLLM: loading model...", flush=True)
    kv_config = KvCacheConfig(
        enable_block_reuse=False,
        free_gpu_memory_fraction=args.kv_fraction,
    )
    llm = LLM(
        model=args.model,
        max_batch_size=args.max_batch_size,
        max_num_tokens=args.max_num_tokens,
        kv_cache_config=kv_config,
    )
    load_time = time.time() - t0
    print(f"EVAL_TRTLLM: model loaded in {load_time:.1f}s", flush=True)

    print("EVAL_TRTLLM: running MMLU (5-shot, generation)...", flush=True)
    t1 = time.time()
    evaluator = MMLU()
    score = evaluator.evaluate(llm)
    eval_time = time.time() - t1
    total_time = time.time() - t0

    print(f"EVAL_TRTLLM: MMLU score = {score}", flush=True)
    print(f"EVAL_TRTLLM: eval_time = {eval_time:.1f}s", flush=True)
    print(f"EVAL_TRTLLM: total_time = {total_time:.1f}s", flush=True)

    result = {
        "model": args.model,
        "mmlu_score": score,
        "load_time_s": load_time,
        "eval_time_s": eval_time,
        "total_time_s": total_time,
        "max_batch_size": args.max_batch_size,
        "max_num_tokens": args.max_num_tokens,
        "kv_fraction": args.kv_fraction,
    }

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"EVAL_TRTLLM: results written to {args.output}", flush=True)

    print("EVAL_TRTLLM: DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

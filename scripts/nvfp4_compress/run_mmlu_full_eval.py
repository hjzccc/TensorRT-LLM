#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from lm_eval import evaluator

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.nvfp4_compress.lm_eval_nvfp4 import NVFP4LM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=THIS_DIR / "mmlu_full_results.json",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batch-total-tokens", type=int, default=17760)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--num-fewshot", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print("RUNNER: create model", flush=True)
    model = NVFP4LM(
        batch_size=args.batch_size,
        max_batch_total_tokens=args.max_batch_total_tokens,
    )
    print("RUNNER: before simple_evaluate", flush=True)

    try:
        results = evaluator.simple_evaluate(
            model=model,
            tasks=["mmlu"],
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
        )
        print("RUNNER: after simple_evaluate", flush=True)
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True))
        print(f"RUNNER: wrote results to {args.output}", flush=True)
    except BaseException as exc:
        print(f"RUNNER: exception {type(exc).__name__}: {exc!r}", flush=True)
        traceback.print_exc()
        raise
    finally:
        print("RUNNER: finally reached", flush=True)


if __name__ == "__main__":
    main()

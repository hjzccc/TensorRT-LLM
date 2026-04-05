#!/usr/bin/env python3

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, cast

from datasets import load_dataset

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.nvfp4_compress.lm_eval_nvfp4 import NVFP4LM, resolve_default_ckpt_dir


CHOICES = ["(A)", "(B)", "(C)", "(D)"]
DEFAULT_LOCKFILE = THIS_DIR / ".mmlu_gpu_eval.lock"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--lockfile", type=Path, default=DEFAULT_LOCKFILE)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batch-total-tokens", type=int, default=17760)
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cpu-offload", action="store_true", help="Offload layers to CPU between forward passes (reduces GPU memory usage)")
    parser.add_argument("--streaming-layers", action="store_true", help="Load each layer from disk on-demand (minimal GPU+RAM usage, slower)")
    parser.add_argument(
        "--output",
        type=Path,
        default=THIS_DIR / "mmlu_direct_results.json",
    )
    return parser.parse_args()


def cached_subjects() -> list[str]:
    base = Path("/root/.cache/huggingface/datasets/cais___mmlu")
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def build_prompt(doc: Mapping[str, Any]) -> str:
    question = str(doc["question"]).strip()
    choices = cast(list[str], doc["choices"])
    return (
        f"Q: {question}\n"
        f"(A) {choices[0]} (B) {choices[1]} (C) {choices[2]} (D) {choices[3]}\n"
        "A:"
    )


def main() -> None:
    args = parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if args.subject is not None:
        subjects = [args.subject]
    else:
        subjects = cached_subjects()

    lockfile = args.lockfile
    lockfile.parent.mkdir(parents=True, exist_ok=True)
    if lockfile.exists():
        try:
            lock_pid = int(lockfile.read_text().strip())
            os.kill(lock_pid, 0)
        except (ValueError, ProcessLookupError, OSError):
            lockfile.unlink(missing_ok=True)
        else:
            raise SystemExit(f"Another MMLU eval is active (pid={lock_pid}, lockfile={lockfile})")

    lockfile.write_text(str(os.getpid()))

    def _cleanup_lock() -> None:
        try:
            if lockfile.exists() and lockfile.read_text().strip() == str(os.getpid()):
                lockfile.unlink(missing_ok=True)
        except OSError:
            pass

    atexit.register(_cleanup_lock)

    print(f"DIRECT_MMLU: subjects={len(subjects)}", flush=True)
    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else resolve_default_ckpt_dir()
    model = NVFP4LM(
        ckpt_dir=ckpt_dir,
        batch_size=args.batch_size,
        max_batch_total_tokens=args.max_batch_total_tokens,
        cpu_offload=args.cpu_offload,
        streaming_layers=args.streaming_layers,
    )

    all_correct = 0
    all_total = 0
    results: dict[str, object] = {
        "subjects": {},
        "config": {
            "batch_size": args.batch_size,
            "max_batch_total_tokens": args.max_batch_total_tokens,
            "offline": args.offline,
            "limit": args.limit,
        },
    }
    subject_results = cast(dict[str, object], results["subjects"])

    for subject in subjects:
        t0 = time.time()
        ds = load_dataset("cais/mmlu", subject, split="test")
        if args.limit is not None:
            ds = ds.select(range(min(args.limit, len(ds))))
        docs = list(ds)
        requests: list[tuple[tuple[str, str], list[int], list[int]]] = []
        answer_keys: list[int] = []

        for idx, doc in enumerate(docs):
            doc = cast(Mapping[str, Any], doc)
            prompt = build_prompt(doc)
            context_enc = model.tok_encode(prompt)
            for choice_idx, choice_text in enumerate(CHOICES):
                requests.append(((subject, f"{idx}:{choice_idx}"), context_enc, model.tok_encode(choice_text)))
            answer = doc["answer"]
            answer_keys.append(int(answer) if not isinstance(answer, str) else CHOICES.index(answer))

        scores = model._loglikelihood_tokens(requests)
        correct = 0
        for idx, gold in enumerate(answer_keys):
            choice_scores = [scores[idx * 4 + j][0] for j in range(4)]
            pred = max(range(4), key=lambda j: choice_scores[j])
            correct += int(pred == gold)

        total = len(answer_keys)
        accuracy = correct / total if total else 0.0
        all_correct += correct
        all_total += total
        subject_results[subject] = {
            "acc": accuracy,
            "correct": correct,
            "total": total,
            "elapsed_sec": time.time() - t0,
        }
        results["mmlu_acc"] = all_correct / all_total if all_total else 0.0
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True))
        print(
            f"DIRECT_MMLU: {subject} acc={accuracy:.4f} correct={correct}/{total} elapsed={time.time()-t0:.1f}s overall={results['mmlu_acc']:.4f}",
            flush=True,
        )

    print(json.dumps(results, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Compare logits from two local BF16 checkpoints")
    p.add_argument("--ref", required=True, help="Reference local checkpoint path")
    p.add_argument("--candidate", required=True, help="Candidate local checkpoint path")
    p.add_argument("--prompt", action="append", dest="prompts", default=[], help="Prompt to test; can be repeated")
    p.add_argument("--prompts-file", type=str, default=None, help="Optional text file with one prompt per line")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--atol", type=float, default=5e-2)
    p.add_argument("--rtol", type=float, default=5e-2)
    p.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    p.add_argument("--ref-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--candidate-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    return p.parse_args()


def load_prompts(args):
    prompts = list(args.prompts)
    if args.prompts_file:
        prompts.extend([line for line in Path(args.prompts_file).read_text().splitlines() if line.strip()])
    if not prompts:
        prompts = [
            "What is 2+2? Answer:",
            "The capital of France is",
            "Explain what a transformer model does in one sentence.",
        ]
    return prompts


def load_model(path, dtype_name):
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]
    return AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    ).eval()


def run_model_on_prompts(model_path, tokenized_prompts, dtype_name):
    import gc
    print(f"Loading checkpoint: {model_path} (dtype={dtype_name})", flush=True)
    model = load_model(model_path, dtype_name)
    logits_out = []
    with torch.inference_mode():
        device = next(model.parameters()).device
        for toks in tokenized_prompts:
            inputs = {k: v.to(device) for k, v in toks.items()}
            logits_out.append(model(**inputs).logits.detach().cpu())
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits_out


def summarize_diff(ref_logits, cand_logits):
    diff = (ref_logits.float() - cand_logits.float()).abs()
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "last_token_argmax_match": bool(ref_logits[:, -1, :].argmax(dim=-1).eq(cand_logits[:, -1, :].argmax(dim=-1)).all().item()),
        "all_token_argmax_match": bool(ref_logits.argmax(dim=-1).eq(cand_logits.argmax(dim=-1)).all().item()),
    }


def main():
    args = parse_args()
    prompts = load_prompts(args)

    tokenizer = AutoTokenizer.from_pretrained(args.ref, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_prompts = [
        tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        for prompt in prompts
    ]

    ref_logits_list = run_model_on_prompts(args.ref, tokenized_prompts, args.ref_dtype)
    cand_logits_list = run_model_on_prompts(args.candidate, tokenized_prompts, args.candidate_dtype)

    results = []
    for idx, (prompt, ref_logits, cand_logits) in enumerate(zip(prompts, ref_logits_list, cand_logits_list), start=1):
            summary = summarize_diff(ref_logits, cand_logits)
            summary.update({
                "prompt_index": idx,
                "prompt": prompt,
                "within_tolerance": bool(torch.allclose(ref_logits.float(), cand_logits.float(), atol=args.atol, rtol=args.rtol)),
                "ref_last_token": int(ref_logits[:, -1, :].argmax(dim=-1).item()),
                "candidate_last_token": int(cand_logits[:, -1, :].argmax(dim=-1).item()),
            })
            results.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    overall = {
        "ref": args.ref,
        "candidate": args.candidate,
        "ref_dtype": args.ref_dtype,
        "candidate_dtype": args.candidate_dtype,
        "num_prompts": len(results),
        "all_within_tolerance": all(r["within_tolerance"] for r in results),
        "all_last_token_match": all(r["last_token_argmax_match"] for r in results),
        "max_abs_diff": max(r["max_abs_diff"] for r in results),
        "mean_abs_diff": sum(r["mean_abs_diff"] for r in results) / len(results),
        "results": results,
    }
    print("SUMMARY:", json.dumps(overall, ensure_ascii=False), flush=True)

    if args.output:
        Path(args.output).write_text(json.dumps(overall, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

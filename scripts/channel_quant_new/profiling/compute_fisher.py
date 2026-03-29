#!/usr/bin/env python3
"""Compute per-channel Fisher scores via backward pass on calibration data.

Fisher score for channel j: S_j = E[g_j²] where g_j = ∂L/∂y_j
(gradient of loss w.r.t. expert output channel j)

This captures how much the loss CARES about each channel — fundamentally
different from E[a²]×||w||² which only measures input energy × weight size.

Uses WikiText-2 TRAIN split for calibration. Processes one layer at a time
to stay within 30GB RAM. Accumulates per-expert per-channel g² on CPU.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
OUTPUT_DIR = Path("/code/tensorrt_llm/scripts/channel_quant_new/profiling")
NUM_LAYERS = 40


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nsamples", type=int, default=16,
                   help="Number of calibration samples (reduced for memory)")
    p.add_argument("--seqlen", type=int, default=512,
                   help="Sequence length per sample (reduced for memory)")
    return p.parse_args()


def load_calibration_tokens(tokenizer, nsamples, seqlen):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt")
    all_ids = enc.input_ids
    total = all_ids.numel()
    actual = min(nsamples, total // seqlen)
    return all_ids[:, : actual * seqlen].view(actual, seqlen).contiguous(), actual


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    cal_tokens, nsamples = load_calibration_tokens(tokenizer, args.nsamples, args.seqlen)
    print(f"Calibration: {nsamples} samples × {args.seqlen} tokens from WikiText-2 TRAIN", flush=True)

    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    fisher_scores = {}
    hooks = []

    def make_hook(layer_idx, expert_idx, proj_name, n_channels):
        key = (layer_idx, expert_idx, proj_name)
        fisher_scores[key] = {
            "sum_g2": torch.zeros(n_channels, dtype=torch.float32),
            "count": 0,
        }

        def hook_fn(module, grad_input, grad_output):
            g = grad_output[0]
            if g is None:
                return
            fisher_scores[key]["sum_g2"] += g.float().pow(2).sum(dim=tuple(range(g.dim() - 1))).cpu()
            fisher_scores[key]["count"] += g.shape[0] if g.dim() >= 2 else 1

        return hook_fn

    print("Registering hooks on MoE expert projections...", flush=True)
    lang_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    for li in range(NUM_LAYERS):
        layer = lang_model.layers[li]
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "experts"):
            continue
        experts = layer.mlp.experts
    moe_grad_outputs = {}

    def make_moe_hook(layer_idx):
        def hook_fn(module, grad_input, grad_output):
            g = grad_output[0]
            if g is not None:
                moe_grad_outputs[layer_idx] = g.detach().float().cpu()
        return hook_fn

    for li in range(NUM_LAYERS):
        layer = lang_model.layers[li]
        if hasattr(layer, "mlp"):
            h = layer.mlp.register_full_backward_hook(make_moe_hook(li))
            hooks.append(h)

    print(f"Registered {len(hooks)} MoE hooks", flush=True)

    print("Running forward+backward passes...", flush=True)
    moe_output_grads_accumulated = {}
    t0 = time.time()

    for si in range(nsamples):
        input_ids = cal_tokens[si:si+1].to(model.device)

        model.zero_grad()
        outputs = model(input_ids, labels=input_ids, use_cache=False)
        loss = outputs.loss
        loss.backward()

        for li, grad in moe_grad_outputs.items():
            if li not in moe_output_grads_accumulated:
                moe_output_grads_accumulated[li] = {
                    "sum_g2": torch.zeros(grad.shape[-1], dtype=torch.float32),
                    "count": 0,
                }
            acc = moe_output_grads_accumulated[li]
            acc["sum_g2"] += grad.pow(2).sum(dim=tuple(range(grad.dim() - 1)))
            acc["count"] += grad.shape[0] * (grad.shape[1] if grad.dim() > 2 else 1)

        moe_grad_outputs.clear()

        if (si + 1) % 4 == 0:
            print(f"  Sample {si+1}/{nsamples} ({time.time()-t0:.0f}s)", flush=True)

    for h in hooks:
        h.remove()

    results = {"metadata": {"nsamples": nsamples, "seqlen": args.seqlen, "source": "wikitext2_train"}}
    for li, acc in sorted(moe_output_grads_accumulated.items()):
        fisher = (acc["sum_g2"] / max(acc["count"], 1)).tolist()
        results[str(li)] = {
            "moe_output_fisher": fisher,
            "count": acc["count"],
        }

    out_path = OUTPUT_DIR / "fisher_scores.json"
    with out_path.open("w") as f:
        json.dump(results, f)
    print(f"Done in {time.time()-t0:.0f}s. Saved {out_path}")


if __name__ == "__main__":
    main()

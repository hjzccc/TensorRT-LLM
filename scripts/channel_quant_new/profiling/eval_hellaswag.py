#!/usr/bin/env python3
"""HellaSwag zero-shot evaluation using layer-by-layer exact TRT-LLM kernels.

For each example: compute log-likelihood of context+completion for each of 4 choices.
Pick the highest log-likelihood as the prediction. Compare with ground truth label.

Modes: bf16 (baseline), nvfp4 (uniform quantized).
"""
import argparse
import json
import sys
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

sys.path.insert(0, "/workspace/channel_quant_new")
sys.path.insert(0, "/workspace/channel_quant")
import tensorrt_llm._torch.auto_deploy.custom_ops
import exact_docker_eval as ee
from spike1_ground_truth import (
    build_text_config, layer_keys, load_root_config,
    move_tensor, release_tensors, rms_norm_qwen3_next, shorten_layer_tensors,
)

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
OUTPUT_DIR = "/code/tensorrt_llm/scripts/channel_quant_new/profiling"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["bf16", "nvfp4"], required=True)
    p.add_argument("--num-examples", type=int, default=200)
    return p.parse_args()


def compute_sequence_logprob(token_ids, model_config, weight_store, device, dtype, mode, rotary_cache):
    seqlen = token_ids.shape[1]
    if seqlen > 2048:
        token_ids = token_ids[:, :2048]
        seqlen = 2048

    embed_key = "model.language_model.embed_tokens.weight"
    norm_key = "model.language_model.norm.weight"
    lm_head_key = "lm_head.weight"
    root_t = weight_store.load_tensors([embed_key, norm_key, lm_head_key])
    embed_w = move_tensor(root_t[embed_key], device, dtype)
    norm_w = move_tensor(root_t[norm_key], device, dtype)
    lm_head_w = move_tensor(root_t[lm_head_key], device, dtype)
    del root_t

    hidden = F.embedding(token_ids.to(device), embed_w)

    position_ids = torch.arange(seqlen, device=device).unsqueeze(0)
    if seqlen not in rotary_cache:
        rot = Qwen3NextRotaryEmbedding(config=model_config, device=device)
        dummy = torch.empty((1, seqlen, model_config.hidden_size), device=device, dtype=dtype)
        rotary_cache[seqlen] = rot(dummy, position_ids)
        del dummy
    position_embeddings = rotary_cache[seqlen]

    with torch.inference_mode():
        for layer_idx in range(model_config.num_hidden_layers):
            layer_type = model_config.layer_types[layer_idx]
            raw = weight_store.load_tensors(layer_keys(layer_idx, layer_type))
            ld = shorten_layer_tensors(layer_idx, raw, device, dtype)
            del raw

            residual = hidden
            hidden = rms_norm_qwen3_next(hidden, ld["input_layernorm.weight"], model_config.rms_norm_eps)
            if layer_type == "full_attention":
                at = {k.replace("self_attn.", ""): v for k, v in ld.items() if k.startswith("self_attn.")}
                hidden = ee.full_attention_forward_exact(
                    hidden, at, model_config, position_embeddings,
                    ee.build_causal_mask(seqlen, device), mode=mode, quantized=(mode != "bf16"),
                )
            else:
                at = {k.replace("linear_attn.", ""): v for k, v in ld.items() if k.startswith("linear_attn.")}
                hidden = ee.linear_attention_forward_exact(hidden, at, model_config, mode, "moe_only")
            hidden = residual + hidden

            residual = hidden
            hidden = rms_norm_qwen3_next(hidden, ld["post_attention_layernorm.weight"], model_config.rms_norm_eps)
            mt = {k.replace("mlp.", "", 1): v for k, v in ld.items() if k.startswith("mlp.")}
            moe_out = ee.moe_forward_exact(hidden, mt, model_config, mode, "moe_only")
            hidden = residual + moe_out

            release_tensors(ld)
            del moe_out

        hidden = rms_norm_qwen3_next(hidden, norm_w, model_config.rms_norm_eps)
        logits = F.linear(hidden, lm_head_w)

    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = token_ids[:, 1:].to(device)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    del embed_w, norm_w, lm_head_w, hidden, logits
    torch.cuda.empty_cache()

    return token_log_probs.sum().item()


def main():
    args = parse_args()
    device = torch.device("cuda")
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    snapshot_dir, root_config, weight_map = load_root_config(MODEL_ID)
    model_config = build_text_config(root_config)
    weight_store = ee.WeightStore(MODEL_ID, snapshot_dir, weight_map)

    dataset = load_dataset("Rowan/hellaswag", split="validation")
    num_examples = min(args.num_examples, len(dataset))
    print(f"HellaSwag eval: {num_examples} examples, mode={args.mode}", flush=True)

    rotary_cache = {}
    correct = 0
    total = 0
    t0 = time.time()

    for idx in range(num_examples):
        example = dataset[idx]
        ctx = example["ctx"]
        endings = example["endings"]
        label = int(example["label"])

        choice_scores = []
        for ending in endings:
            text = ctx + " " + ending
            tokens = tokenizer(text, return_tensors="pt").input_ids
            score = compute_sequence_logprob(
                tokens, model_config, weight_store, device, dtype, args.mode, rotary_cache
            )
            choice_scores.append(score)

        prediction = max(range(4), key=lambda i: choice_scores[i])
        if prediction == label:
            correct += 1
        total += 1

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            acc = correct / total * 100
            print(f"  [{idx+1}/{num_examples}] acc={acc:.1f}% ({correct}/{total}) {elapsed:.0f}s", flush=True)

    accuracy = correct / total * 100
    elapsed = time.time() - t0
    print(f"\nFinal: {args.mode} accuracy={accuracy:.1f}% ({correct}/{total}) in {elapsed:.0f}s")

    result = {"mode": args.mode, "accuracy": accuracy, "correct": correct, "total": total, "time_s": elapsed}
    out_path = f"{OUTPUT_DIR}/hellaswag_{args.mode}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

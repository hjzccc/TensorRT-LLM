#!/usr/bin/env python3
"""Fast HellaSwag eval: batch all 4 choices per example, load weights once per layer."""
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
    p.add_argument("--num-examples", type=int, default=100)
    return p.parse_args()


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

    root_keys = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ]
    root_t = weight_store.load_tensors(root_keys)
    embed_w = move_tensor(root_t[root_keys[0]], device, dtype)
    norm_w = move_tensor(root_t[root_keys[1]], device, dtype)
    lm_head_w = move_tensor(root_t[root_keys[2]], device, dtype)
    del root_t

    print(f"HellaSwag eval: {num_examples} examples, mode={args.mode}", flush=True)
    correct = 0
    total = 0
    t0 = time.time()

    for idx in range(num_examples):
        example = dataset[idx]
        ctx = example["ctx"]
        endings = example["endings"]
        label = int(example["label"])

        sequences = []
        for ending in endings:
            text = ctx + " " + ending
            tokens = tokenizer(text, return_tensors="pt").input_ids
            if tokens.shape[1] > 512:
                tokens = tokens[:, :512]
            sequences.append(tokens)

        max_len = max(s.shape[1] for s in sequences)
        padded = torch.zeros(4, max_len, dtype=torch.long)
        lengths = []
        for i, s in enumerate(sequences):
            padded[i, :s.shape[1]] = s[0]
            lengths.append(s.shape[1])

        hidden_bank = F.embedding(padded.to(device), embed_w)

        with torch.inference_mode():
            for layer_idx in range(model_config.num_hidden_layers):
                layer_type = model_config.layer_types[layer_idx]
                raw = weight_store.load_tensors(layer_keys(layer_idx, layer_type))
                ld = shorten_layer_tensors(layer_idx, raw, device, dtype)
                del raw

                for seq_idx in range(4):
                    sl = lengths[seq_idx]
                    h = hidden_bank[seq_idx:seq_idx+1, :sl, :]

                    pid = torch.arange(sl, device=device).unsqueeze(0)
                    rot = Qwen3NextRotaryEmbedding(config=model_config, device=device)
                    dummy = torch.empty((1, sl, model_config.hidden_size), device=device, dtype=dtype)
                    pe = rot(dummy, pid)
                    del dummy

                    residual = h
                    h = rms_norm_qwen3_next(h, ld["input_layernorm.weight"], model_config.rms_norm_eps)
                    if layer_type == "full_attention":
                        at = {k.replace("self_attn.", ""): v for k, v in ld.items() if k.startswith("self_attn.")}
                        h = ee.full_attention_forward_exact(
                            h, at, model_config, pe,
                            ee.build_causal_mask(sl, device),
                            mode=args.mode if args.mode == "bf16" else "bf16",
                            quantized=False,
                        )
                    else:
                        at = {k.replace("linear_attn.", ""): v for k, v in ld.items() if k.startswith("linear_attn.")}
                        h = ee.linear_attention_forward_exact(h, at, model_config, "bf16", "moe_only")
                    h = residual + h

                    residual = h
                    h = rms_norm_qwen3_next(h, ld["post_attention_layernorm.weight"], model_config.rms_norm_eps)
                    mt = {k.replace("mlp.", "", 1): v for k, v in ld.items() if k.startswith("mlp.")}
                    moe_out = ee.moe_forward_exact(h, mt, model_config, args.mode, "moe_only")
                    h = residual + moe_out
                    del moe_out

                    hidden_bank[seq_idx, :sl, :] = h.squeeze(0)
                    del h, residual

                release_tensors(ld)

        choice_scores = []
        for seq_idx in range(4):
            sl = lengths[seq_idx]
            h = hidden_bank[seq_idx:seq_idx+1, :sl, :]
            h = rms_norm_qwen3_next(h, norm_w, model_config.rms_norm_eps)
            logits = F.linear(h, lm_head_w)
            log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
            target = padded[seq_idx, 1:sl].to(device)
            token_lp = log_probs.squeeze(0).gather(1, target.unsqueeze(-1)).squeeze(-1)
            choice_scores.append(token_lp.sum().item())
            del h, logits, log_probs

        prediction = max(range(4), key=lambda i: choice_scores[i])
        if prediction == label:
            correct += 1
        total += 1

        del hidden_bank
        torch.cuda.empty_cache()

        if (idx + 1) % 5 == 0:
            elapsed = time.time() - t0
            acc = correct / total * 100
            eta = elapsed / (idx + 1) * (num_examples - idx - 1)
            print(f"  [{idx+1}/{num_examples}] acc={acc:.1f}% ({correct}/{total}) "
                  f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining", flush=True)

    accuracy = correct / total * 100
    elapsed = time.time() - t0
    print(f"\nFinal: {args.mode} accuracy={accuracy:.1f}% ({correct}/{total}) in {elapsed:.0f}s")

    result = {"mode": args.mode, "accuracy": accuracy, "correct": correct, "total": total, "time_s": elapsed}
    out_path = f"{OUTPUT_DIR}/hellaswag_{args.mode}_{num_examples}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

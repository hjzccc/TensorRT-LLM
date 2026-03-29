#!/usr/bin/env python3
"""HellaSwag zero-shot evaluation with layer-by-layer exact TRT-LLM kernels.

Loads each layer's weights ONCE, processes ALL sequences through it, releases.
Hidden states stored on CPU between layers — same approach as our PPL pipeline.
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
MAX_SEQUENCE_LENGTH = 256
NUM_CHOICES = 4
EMBEDDING_BATCH_SIZE = 16


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bf16", "nvfp4"], required=True)
    parser.add_argument("--num-examples", type=int, default=100)
    return parser.parse_args()


def tokenize_all_choices(dataset, num_examples, tokenizer):
    token_ids_per_sequence = []
    sequence_lengths = []

    for example_idx in range(num_examples):
        example = dataset[example_idx]
        context = example["ctx"]
        for ending in example["endings"]:
            full_text = context + " " + ending
            token_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]
            if len(token_ids) > MAX_SEQUENCE_LENGTH:
                token_ids = token_ids[:MAX_SEQUENCE_LENGTH]
            token_ids_per_sequence.append(token_ids)
            sequence_lengths.append(len(token_ids))

    return token_ids_per_sequence, sequence_lengths


def embed_all_sequences(token_ids_list, sequence_lengths, embedding_weight, hidden_size, dtype, device):
    num_sequences = len(token_ids_list)
    max_length = max(sequence_lengths)

    padded_token_ids = torch.zeros(num_sequences, max_length, dtype=torch.long)
    for i, tokens in enumerate(token_ids_list):
        padded_token_ids[i, :len(tokens)] = tokens

    hidden_states = torch.empty(num_sequences, max_length, hidden_size, dtype=dtype, device="cpu")
    for batch_start in range(0, num_sequences, EMBEDDING_BATCH_SIZE):
        batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, num_sequences)
        batch_tokens = padded_token_ids[batch_start:batch_end].to(device)
        batch_embeddings = F.embedding(batch_tokens, embedding_weight)
        hidden_states[batch_start:batch_end].copy_(batch_embeddings.cpu())
        del batch_tokens, batch_embeddings

    return hidden_states, padded_token_ids


def process_all_sequences_through_layers(
    hidden_states, sequence_lengths, weight_store, model_config,
    quantization_mode, device, dtype, start_time, layers_per_batch=8,
):
    num_sequences = len(sequence_lengths)
    num_layers = model_config.num_hidden_layers

    for layer_batch_start in range(0, num_layers, layers_per_batch):
        layer_batch_end = min(layer_batch_start + layers_per_batch, num_layers)

        loaded_layers = {}
        for layer_idx in range(layer_batch_start, layer_batch_end):
            layer_type = model_config.layer_types[layer_idx]
            raw = weight_store.load_tensors(layer_keys(layer_idx, layer_type))
            loaded_layers[layer_idx] = shorten_layer_tensors(layer_idx, raw, device, dtype)
            del raw

        for seq_idx in range(num_sequences):
            seq_length = sequence_lengths[seq_idx]
            hidden = hidden_states[seq_idx:seq_idx + 1, :seq_length, :].to(device)

            position_ids = torch.arange(seq_length, device=device).unsqueeze(0)
            rotary_embedding = Qwen3NextRotaryEmbedding(config=model_config, device=device)
            rotary_dummy = torch.empty((1, seq_length, model_config.hidden_size), device=device, dtype=dtype)
            position_embeddings = rotary_embedding(rotary_dummy, position_ids)
            del rotary_dummy

            for layer_idx in range(layer_batch_start, layer_batch_end):
                layer_type = model_config.layer_types[layer_idx]
                layer_weights = loaded_layers[layer_idx]

                residual = hidden
                hidden = rms_norm_qwen3_next(hidden, layer_weights["input_layernorm.weight"], model_config.rms_norm_eps)

                if layer_type == "full_attention":
                    attn_w = {k.replace("self_attn.", ""): v for k, v in layer_weights.items() if k.startswith("self_attn.")}
                    hidden = ee.full_attention_forward_exact(
                        hidden, attn_w, model_config, position_embeddings,
                        ee.build_causal_mask(seq_length, device), mode="bf16", quantized=False,
                    )
                else:
                    attn_w = {k.replace("linear_attn.", ""): v for k, v in layer_weights.items() if k.startswith("linear_attn.")}
                    hidden = ee.linear_attention_forward_exact(
                        hidden, attn_w, model_config, "bf16", "moe_only",
                    )
                hidden = residual + hidden

                residual = hidden
                hidden = rms_norm_qwen3_next(hidden, layer_weights["post_attention_layernorm.weight"], model_config.rms_norm_eps)
                moe_w = {k.replace("mlp.", "", 1): v for k, v in layer_weights.items() if k.startswith("mlp.")}
                moe_output = ee.moe_forward_exact(hidden, moe_w, model_config, quantization_mode, "moe_only")
                hidden = residual + moe_output
                del residual, moe_output

            hidden_states[seq_idx, :seq_length, :] = hidden.squeeze(0).cpu()
            del hidden

        for layer_weights in loaded_layers.values():
            release_tensors(layer_weights)
        del loaded_layers
        torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        print(f"  Layers {layer_batch_start}-{layer_batch_end-1}/{num_layers} done ({elapsed:.0f}s)", flush=True)


def compute_choice_log_probabilities(
    hidden_states, sequence_lengths, padded_token_ids,
    final_norm_weight, lm_head_weight, device,
):
    log_probs_per_sequence = []
    for seq_idx in range(len(sequence_lengths)):
        seq_length = sequence_lengths[seq_idx]
        hidden = hidden_states[seq_idx:seq_idx + 1, :seq_length, :].to(device)
        hidden = rms_norm_qwen3_next(hidden, final_norm_weight, 1e-6)
        logits = F.linear(hidden, lm_head_weight)

        shifted_log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        target_tokens = padded_token_ids[seq_idx, 1:seq_length].to(device)
        per_token_log_prob = shifted_log_probs.squeeze(0).gather(1, target_tokens.unsqueeze(-1)).squeeze(-1)
        total_log_prob = per_token_log_prob.sum().item()

        log_probs_per_sequence.append(total_log_prob)
        del hidden, logits, shifted_log_probs

    return log_probs_per_sequence


def score_examples(log_probs, num_examples, dataset):
    correct = 0
    for example_idx in range(num_examples):
        ground_truth = int(dataset[example_idx]["label"])
        choice_start = example_idx * NUM_CHOICES
        choice_scores = log_probs[choice_start:choice_start + NUM_CHOICES]
        prediction = max(range(NUM_CHOICES), key=lambda i: choice_scores[i])
        if prediction == ground_truth:
            correct += 1
    return correct


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

    root_tensors = weight_store.load_tensors([
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ])
    embedding_weight = move_tensor(root_tensors["model.language_model.embed_tokens.weight"], device, dtype)
    final_norm_weight = move_tensor(root_tensors["model.language_model.norm.weight"], device, dtype)
    lm_head_weight = move_tensor(root_tensors["lm_head.weight"], device, dtype)
    del root_tensors

    print(f"HellaSwag: {num_examples} examples, mode={args.mode}, max_seq_len={MAX_SEQUENCE_LENGTH}", flush=True)
    start_time = time.time()

    token_ids_list, sequence_lengths = tokenize_all_choices(dataset, num_examples, tokenizer)
    num_sequences = len(token_ids_list)
    print(f"  {num_sequences} sequences, max_len={max(sequence_lengths)}", flush=True)

    hidden_states, padded_token_ids = embed_all_sequences(
        token_ids_list, sequence_lengths, embedding_weight,
        model_config.hidden_size, dtype, device,
    )
    del embedding_weight, token_ids_list
    torch.cuda.empty_cache()
    print(f"  Hidden bank: {hidden_states.numel() * 2 / 1e9:.2f} GB on CPU", flush=True)

    layers_per_batch = 8
    with torch.inference_mode():
        process_all_sequences_through_layers(
            hidden_states, sequence_lengths, weight_store, model_config,
            args.mode, device, dtype, start_time, layers_per_batch=layers_per_batch,
        )

    log_probs = compute_choice_log_probabilities(
        hidden_states, sequence_lengths, padded_token_ids,
        final_norm_weight, lm_head_weight, device,
    )

    correct = score_examples(log_probs, num_examples, dataset)
    accuracy = correct / num_examples * 100
    elapsed = time.time() - start_time
    print(f"\nFinal: {args.mode} accuracy={accuracy:.1f}% ({correct}/{num_examples}) in {elapsed:.0f}s")

    result = {
        "mode": args.mode,
        "accuracy": round(accuracy, 2),
        "correct": correct,
        "total": num_examples,
        "max_seq_len": MAX_SEQUENCE_LENGTH,
        "time_s": round(elapsed, 1),
    }
    output_path = f"{OUTPUT_DIR}/hellaswag_{args.mode}_{num_examples}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()

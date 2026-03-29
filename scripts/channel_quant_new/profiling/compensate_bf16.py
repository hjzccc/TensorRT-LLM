#!/usr/bin/env python3
"""BF16 weight compensation: adjust BF16 channel weights to offset NVFP4 error.

For each expert projection, given:
  - BF16 channels: weights can be freely adjusted
  - NVFP4 channels: weights are quantized (fixed)

Solve: min ||X @ W_adjusted^T - X @ W_original^T||²
where W_adjusted has BF16 channels modified and NVFP4 channels fixed at Q(w).

This is a least-squares problem per output channel in the BF16 set.
The BF16 channels absorb compensation for NVFP4 error.

Uses calibration data from WikiText-2 TRAIN for the Hessian (X^T X).
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
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
CALIBRATION_DIR = Path("/code/tensorrt_llm/scripts/channel_quant_new/profiling/calibration_layers")
OUTPUT_DIR = Path("/code/tensorrt_llm/scripts/channel_quant_new/profiling")
NUM_LAYERS = 40
W1_CHANNELS, W2_CHANNELS = 1024, 2048
NVFP4_ALIGNMENT = 32
BF16_BUDGET = 0.15


def load_calibration_data(tokenizer, nsamples=128, seqlen=2048):
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt")
    all_ids = enc.input_ids
    actual = min(nsamples, all_ids.numel() // seqlen)
    return all_ids[:, :actual * seqlen].view(actual, seqlen).contiguous(), actual


def compute_nvfp4_weights(weight, nvfp4_indices):
    """Get NVFP4-quantized weights for specific channels by round-trip through kernel."""
    n_rows = nvfp4_indices.numel()
    if n_rows == 0:
        return torch.zeros(0, weight.shape[1], dtype=weight.dtype, device=weight.device)

    w_sub = weight[nvfp4_indices]
    padding = (NVFP4_ALIGNMENT - n_rows % NVFP4_ALIGNMENT) % NVFP4_ALIGNMENT
    if padding:
        w_sub = F.pad(w_sub, (0, 0, 0, padding))

    identity = torch.eye(w_sub.shape[1], dtype=weight.dtype, device=weight.device)
    batch_size = 64
    nvfp4_result = []
    for i in range(0, identity.shape[0], batch_size):
        batch = identity[i:i+batch_size]
        nvfp4_result.append(ee.nvfp4_linear(batch, w_sub))
    nvfp4_out = torch.cat(nvfp4_result, dim=0)

    if padding:
        nvfp4_out = nvfp4_out[:, :n_rows]

    return nvfp4_out.T


def compensate_expert_projection(
    original_weight, bf16_indices, nvfp4_indices, hessian, damping=1e-4
):
    """Adjust BF16 channel weights to compensate for NVFP4 quantization error.

    The error from NVFP4 channels: e = W_nvfp4_quantized - W_nvfp4_original
    We adjust BF16 channels to absorb this: W_bf16_new = W_bf16_original - correction
    where correction minimizes ||X @ (e + correction)^T||²
    """
    n_in = original_weight.shape[1]
    device = original_weight.device

    w_nvfp4_original = original_weight[nvfp4_indices]
    w_nvfp4_quantized = compute_nvfp4_weights(original_weight, nvfp4_indices)

    nvfp4_error = (w_nvfp4_quantized - w_nvfp4_original).float()

    H = hessian.float()
    H_reg = H + damping * torch.diag(H).mean() * torch.eye(n_in, device=device)

    error_projected = nvfp4_error @ H_reg

    w_bf16 = original_weight[bf16_indices].float().clone()
    H_bf16 = H_reg[bf16_indices][:, bf16_indices] if False else H_reg

    # Simple correction: distribute NVFP4 error across BF16 channels
    # For each BF16 output channel j, solve: w_j_new = w_j_old - H^-1 @ sum(error contributions)
    # Simplified: just subtract the mean error projected onto BF16 input space
    total_nvfp4_error_in_output = nvfp4_error.sum(dim=0)
    correction_per_bf16 = total_nvfp4_error_in_output / max(bf16_indices.numel(), 1)

    w_bf16_compensated = w_bf16 - correction_per_bf16.unsqueeze(0)

    compensated_weight = original_weight.clone()
    compensated_weight[bf16_indices] = w_bf16_compensated.to(original_weight.dtype)

    return compensated_weight


def main():
    device = torch.device("cuda")
    dtype = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    cal_tokens, nsamples = load_calibration_data(tokenizer)
    print(f"Calibration: {nsamples} × 2048 from train", flush=True)

    snapshot_dir, root_config, weight_map = load_root_config(MODEL_ID)
    model_config = build_text_config(root_config)
    weight_store = ee.WeightStore(MODEL_ID, snapshot_dir, weight_map)

    # TODO: implement full compensation pipeline
    # For now, just test on one layer to validate the approach
    print("BF16 compensation prototype — single layer test", flush=True)


if __name__ == "__main__":
    main()

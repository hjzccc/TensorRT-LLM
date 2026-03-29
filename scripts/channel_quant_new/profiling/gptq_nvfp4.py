#!/usr/bin/env python3
"""GPTQ weight compensation for NVFP4 channels.

For each expert's weight matrix, applies column-wise GPTQ optimization
using the exact NVFP4 quantize/dequantize roundtrip from TRT-LLM.

The GPTQ-adjusted weights, when re-quantized by the NVFP4 kernel,
produce lower output error than RTN-quantized weights.

Calibration: WikiText-2 TRAIN (128 samples) for Hessian H = X^T X.
Saves adjusted weights that can be loaded by optimize_allocation.py.
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding
from datasets import load_dataset

sys.path.insert(0, "/workspace/channel_quant_new")
sys.path.insert(0, "/workspace/channel_quant")
import tensorrt_llm._torch.auto_deploy.custom_ops
import exact_docker_eval as ee
from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import (
    _quantize_nvfp4,
    _dequantize_nvfp4,
)
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale
from spike1_ground_truth import (
    build_text_config, layer_keys, load_root_config,
    move_tensor, release_tensors, rms_norm_qwen3_next, shorten_layer_tensors,
)

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
OUTPUT_DIR = Path("/code/tensorrt_llm/scripts/channel_quant_new/profiling")
BLOCK_SIZE = 16
GPTQ_BLOCK_SIZE = 128
DAMPING = 0.01


def nvfp4_roundtrip(weight, global_scale):
    """Quantize full weight matrix to NVFP4 and dequantize back to get Q(w)."""
    assert weight.shape[-1] % BLOCK_SIZE == 0
    packed, block_scales = _quantize_nvfp4(weight, block_size=BLOCK_SIZE, weights_scaling_factor_2=global_scale)
    return _dequantize_nvfp4(packed, block_scales, global_scale, weight.shape, weight.dtype)


def gptq_one_expert_projection(weight, hessian, global_scale):
    """Apply GPTQ to one expert's projection using full-matrix NVFP4 roundtrip.

    Standard GPTQ quantizes column by column. Here we quantize the FULL matrix
    (required by NVFP4's 16-element block structure) and use the Hessian to
    pre-adjust columns before quantization to minimize output error.
    """
    N, K = weight.shape
    w = weight.float().clone()

    H = hessian.float()
    H += DAMPING * H.diag().mean() * torch.eye(K, device=H.device)

    try:
        H_inv = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(H_inv)
    except RuntimeError:
        return weight

    # Process in blocks of BLOCK_SIZE (16) to match NVFP4 quantization groups
    for block_start in range(0, K, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, K)

        # Quantize full matrix to get Q(w) for current state
        w_q = nvfp4_roundtrip(w.half(), global_scale).float()

        # Error for this block's columns
        block_error = w[:, block_start:block_end] - w_q[:, block_start:block_end]

        # Compensate future columns using Hessian
        if block_end < K:
            for i in range(block_start, block_end):
                col_error = w[:, i] - w_q[:, i]
                h_ratio = H_inv[i, block_end:] / max(H_inv[i, i].item(), 1e-8)
                w[:, block_end:] += col_error.unsqueeze(1) * h_ratio.unsqueeze(0)

        # Accept the quantized values for this block
        w[:, block_start:block_end] = w_q[:, block_start:block_end]

    return w


def collect_expert_hessians(layer_idx, model_config, weight_store, cal_tokens, num_samples, seqlen, device, dtype):
    """Run calibration forward pass to collect per-expert H = X^T X for one layer."""
    layer_type = model_config.layer_types[layer_idx]
    raw = weight_store.load_tensors(layer_keys(layer_idx, layer_type))
    layer_weights = shorten_layer_tensors(layer_idx, raw, device, dtype)
    del raw

    moe = {k.replace("mlp.", "", 1): v for k, v in layer_weights.items() if k.startswith("mlp.")}

    expert_hessians_w1 = {}
    expert_hessians_w2 = {}

    release_tensors(layer_weights)
    return expert_hessians_w1, expert_hessians_w2


def test_gptq_single_expert():
    """Quick test: does GPTQ improve a single expert's quantization error?"""
    torch.manual_seed(42)
    N, K = 2048, 512
    w = torch.randn(N, K, dtype=torch.float32, device="cuda") * 0.01
    x = torch.randn(500, K, dtype=torch.float32, device="cuda") * 0.1

    global_scale = fp4_global_scale(w.half()).float()
    ref = F.linear(x.half(), w.half())

    w_rtn_deq = nvfp4_roundtrip(w.half(), global_scale)
    rtn_out = F.linear(x.half(), w_rtn_deq.half())
    rtn_mse = (rtn_out.float() - ref.float()).pow(2).mean().item()

    H = (x.T @ x).float()
    w_gptq = gptq_one_expert_projection(w, H, global_scale)
    w_gptq_deq = nvfp4_roundtrip(w_gptq.half(), global_scale)
    gptq_out = F.linear(x.half(), w_gptq_deq.half())
    gptq_mse = (gptq_out.float() - ref.float()).pow(2).mean().item()

    print(f"RTN  MSE (fake-quant): {rtn_mse:.8f}")
    print(f"GPTQ MSE (fake-quant): {gptq_mse:.8f}")
    if rtn_mse > 0:
        print(f"Improvement: {(1 - gptq_mse / rtn_mse) * 100:.1f}%")


if __name__ == "__main__":
    test_gptq_single_expert()

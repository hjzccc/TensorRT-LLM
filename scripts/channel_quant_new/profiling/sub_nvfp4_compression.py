#!/usr/bin/env python3
"""Sub-NVFP4 compression: further compress NVFP4 codes to 3-bit or 2-bit.

Pipeline: BF16 weight → snap to sub-codebook in NVFP4 grid → NVFP4 quantize → evaluate.
Reuses optimize_allocation.py's evaluation infrastructure.
"""
import sys
import time
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/code/tensorrt_llm/scripts/channel_quant_new")
sys.path.insert(0, "/code/tensorrt_llm/scripts/channel_quant")
sys.path.insert(0, "/code/tensorrt_llm/scripts/channel_quant_new/profiling")

import tensorrt_llm._torch.auto_deploy.custom_ops
import exact_docker_eval as ee
from spike1_ground_truth import (
    build_text_config,
    layer_keys,
    load_root_config,
    move_tensor,
    release_tensors,
    rms_norm_qwen3_next,
    shorten_layer_tensors,
)
from real_eval_pipeline import load_eval_data
from transformers import AutoTokenizer
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
BLOCK_SIZE = 16

FP4_VALUES = torch.tensor([-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0, 0.5, 1, 1.5, 2, 3, 4, 6])

CODEBOOKS = {
    '3bit_truncate': torch.tensor([-4, -2, -1, 0, 0, 1, 2, 4], dtype=torch.float32),
    '3bit_uniform':  torch.tensor([-6, -4, -2, 0, 0, 2, 4, 6], dtype=torch.float32),
    '3bit_dense':    torch.tensor([-6, -2, -1, 0, 0, 1, 2, 6], dtype=torch.float32),
    '2bit_sym':      torch.tensor([-6, 0, 0, 6], dtype=torch.float32),
    '2bit_mid':      torch.tensor([-3, 0, 0, 3], dtype=torch.float32),
    '2bit_uniform':  torch.tensor([-6, -2, 2, 6], dtype=torch.float32),
    '2bit_opt1':     torch.tensor([-4, 0, 3, 6], dtype=torch.float32),
    '2bit_opt2':     torch.tensor([-4, 0, 1.5, 4], dtype=torch.float32),
    '2bit_opt3':     torch.tensor([-4, 0, 2, 6], dtype=torch.float32),
    '2bit_sym_zero': torch.tensor([-4, 0, 0, 4], dtype=torch.float32),
    'nvfp4_full':    FP4_VALUES.float(),
}


def snap_weight_to_sub_grid(weight_bf16, codebook, device='cuda'):
    """Snap weight to sub-codebook of NVFP4 grid: block-normalize → round to codebook → denormalize."""
    from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

    weight = weight_bf16.float().to(device)
    rows, cols = weight.shape

    global_scale = fp4_global_scale(weight).to(torch.float32)
    scaled = weight * global_scale

    pad = (BLOCK_SIZE - cols % BLOCK_SIZE) % BLOCK_SIZE
    if pad > 0:
        scaled = F.pad(scaled, (0, pad))

    blocks = scaled.reshape(rows, -1, BLOCK_SIZE)
    block_amax = blocks.abs().amax(dim=2)
    block_scales = (block_amax / 6.0).to(torch.float8_e4m3fn).float()
    safe_scales = block_scales.clone()
    safe_scales[safe_scales == 0] = 1.0

    normalized = blocks / safe_scales.unsqueeze(2)

    cb = codebook.to(device)
    diff = normalized.unsqueeze(-1) - cb.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    nearest_idx = diff.abs().argmin(dim=-1)
    snapped = cb[nearest_idx]

    result = snapped * safe_scales.unsqueeze(2) / global_scale
    result = result.reshape(rows, -1)[:, :cols]

    return result.to(weight_bf16.dtype)


def snap_weight_stochastic_3bit(weight_bf16, codebook, device='cuda'):
    """Stochastic rounding to 3-bit: probability proportional to distance from lower/upper codebook value."""
    from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

    weight = weight_bf16.float().to(device)
    rows, cols = weight.shape

    global_scale = fp4_global_scale(weight).to(torch.float32)
    scaled = weight * global_scale

    pad = (BLOCK_SIZE - cols % BLOCK_SIZE) % BLOCK_SIZE
    if pad > 0:
        scaled = F.pad(scaled, (0, pad))

    blocks = scaled.reshape(rows, -1, BLOCK_SIZE)
    block_amax = blocks.abs().amax(dim=2)
    block_scales = (block_amax / 6.0).to(torch.float8_e4m3fn).float()
    safe_scales = block_scales.clone()
    safe_scales[safe_scales == 0] = 1.0

    normalized = blocks / safe_scales.unsqueeze(2)

    cb = codebook.to(device).sort().values
    diff = normalized.unsqueeze(-1) - cb
    abs_diff = diff.abs()

    nearest_idx = abs_diff.argmin(dim=-1)
    snapped_det = cb[nearest_idx]

    lower_idx = torch.searchsorted(cb, normalized.contiguous().reshape(-1)).reshape(normalized.shape) - 1
    lower_idx = lower_idx.clamp(0, len(cb) - 2)
    upper_idx = lower_idx + 1

    lower_val = cb[lower_idx]
    upper_val = cb[upper_idx]

    span = upper_val - lower_val
    span = torch.where(span == 0, torch.ones_like(span), span)
    p_upper = ((normalized - lower_val) / span).clamp(0, 1)

    rand = torch.rand_like(p_upper)
    use_upper = rand < p_upper
    snapped = torch.where(use_upper, upper_val, lower_val)

    result = snapped * safe_scales.unsqueeze(2) / global_scale
    result = result.reshape(rows, -1)[:, :cols]

    return result.to(weight_bf16.dtype)


CB_3BIT = torch.tensor([-6, -4, -2, 0, 0, 2, 4, 6], dtype=torch.float32)
CB_2BIT = torch.tensor([-4, 0, 3, 6], dtype=torch.float32)
CB_3BIT_SCALE6 = torch.tensor([-6, -4, -2, 0, 0, 2, 4, 6], dtype=torch.float32)
CB_3BIT_SCALE4 = torch.tensor([-4, -2, -1, 0, 0, 1, 2, 4], dtype=torch.float32)


def snap_weight_product_quantized_2bit(weight_bf16, codebooks_list, device='cuda'):
    """Product quantization: split block into 4 sub-vectors of 4, each with 256-entry codebook = 2 bits/elem."""
    from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

    weight = weight_bf16.float().to(device)
    rows, cols = weight.shape

    global_scale = fp4_global_scale(weight).to(torch.float32)
    scaled = weight * global_scale

    pad = (BLOCK_SIZE - cols % BLOCK_SIZE) % BLOCK_SIZE
    if pad > 0:
        scaled = F.pad(scaled, (0, pad))

    blocks = scaled.reshape(rows, -1, BLOCK_SIZE)
    block_amax = blocks.abs().amax(dim=2)
    block_scales = (block_amax / 6.0).to(torch.float8_e4m3fn).float()
    safe_scales = block_scales.clone()
    safe_scales[safe_scales == 0] = 1.0

    normalized = blocks / safe_scales.unsqueeze(2)

    sub_size = 4
    n_subs = BLOCK_SIZE // sub_size
    reconstructed = torch.zeros_like(normalized)

    for si in range(n_subs):
        sub = normalized[:, :, si*sub_size:(si+1)*sub_size]
        cb = codebooks_list[si].to(device)
        flat_sub = sub.reshape(-1, sub_size)
        dists = torch.cdist(flat_sub, cb)
        nearest = dists.argmin(dim=1)
        recon = cb[nearest].reshape(sub.shape)
        reconstructed[:, :, si*sub_size:(si+1)*sub_size] = recon

    result = reconstructed * safe_scales.unsqueeze(2) / global_scale
    result = result.reshape(rows, -1)[:, :cols]
    return result.to(weight_bf16.dtype)


def learn_pq_codebooks(max_experts=20, n_clusters=256, sub_size=4):
    """Learn PQ codebooks from a sample of expert weights using k-means."""
    from safetensors import safe_open

    model_dir = Path("/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots")
    snapshot = list(model_dir.iterdir())[0]
    files = sorted(snapshot.glob("*.safetensors"))

    FP4_UNIQUE = np.array([-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6])
    BOUNDARIES = np.array([-5, -3.5, -2.5, -1.75, -1.25, -0.75, -0.25, 0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])

    n_subs = BLOCK_SIZE // sub_size
    all_subs = [[] for _ in range(n_subs)]
    n_collected = 0

    np.random.seed(42)
    for fpath in files[:5]:
        with safe_open(str(fpath), framework='pt', device='cpu') as f:
            for key in f.keys():
                if 'experts' not in key or ('gate_up_proj' not in key and 'down_proj' not in key):
                    continue
                if 'mtp.' in key or 'shared_expert' in key:
                    continue
                packed = f.get_tensor(key).float().numpy()
                indices = np.random.choice(packed.shape[0], min(max_experts, packed.shape[0]), replace=False)
                for ei in indices:
                    w = packed[ei]
                    rows, cols = w.shape
                    pad = (BLOCK_SIZE - cols % BLOCK_SIZE) % BLOCK_SIZE
                    if pad > 0:
                        w = np.pad(w, ((0,0),(0,pad)))
                    blocks = w.reshape(-1, BLOCK_SIZE)
                    for bi in range(blocks.shape[0]):
                        amax = np.abs(blocks[bi]).max()
                        if amax == 0:
                            continue
                        scale = np.float16(amax / 6.0).astype(np.float32)
                        if scale == 0:
                            scale = 1.0
                        norm = blocks[bi] / scale
                        idx = np.searchsorted(BOUNDARIES, norm)
                        fp4_vals = FP4_UNIQUE[np.clip(idx, 0, len(FP4_UNIQUE)-1)]
                        for si in range(n_subs):
                            all_subs[si].append(fp4_vals[si*sub_size:(si+1)*sub_size])
                    n_collected += blocks.shape[0]
                if n_collected > 200000:
                    break
        if n_collected > 200000:
            break

    print(f"Collected {n_collected:,} blocks for PQ codebook learning", flush=True)

    from sklearn.cluster import MiniBatchKMeans
    codebooks = []
    for si in range(n_subs):
        data = np.array(all_subs[si])
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=4096, random_state=42, n_init=3)
        kmeans.fit(data)
        centroids = kmeans.cluster_centers_
        cb = torch.tensor(centroids, dtype=torch.float32)
        codebooks.append(cb)
        inertia = kmeans.inertia_ / len(data)
        print(f"  Sub-vector {si}: {len(data)} samples, MSE={inertia:.4f}", flush=True)

    return codebooks


CB_1BIT = torch.tensor([-6, 6], dtype=torch.float32)
CB_3BIT_UNIFORM = torch.tensor([-6, -4, -2, 0, 0, 2, 4, 6], dtype=torch.float32)


def snap_weight_mixed_3bit_1bit(weight_bf16, frac_1bit=0.5, device='cuda'):
    """Mixed 3-bit/1-bit: important blocks get 3-bit ±{0,2,4,6}, trivial blocks get 1-bit {-6,6}."""
    from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

    weight = weight_bf16.float().to(device)
    rows, cols = weight.shape

    global_scale = fp4_global_scale(weight).to(torch.float32)
    scaled = weight * global_scale

    pad = (BLOCK_SIZE - cols % BLOCK_SIZE) % BLOCK_SIZE
    if pad > 0:
        scaled = F.pad(scaled, (0, pad))

    blocks = scaled.reshape(rows, -1, BLOCK_SIZE)
    block_amax = blocks.abs().amax(dim=2)
    block_scales = (block_amax / 6.0).to(torch.float8_e4m3fn).float()
    safe_scales = block_scales.clone()
    safe_scales[safe_scales == 0] = 1.0
    normalized = blocks / safe_scales.unsqueeze(2)

    flat_scales = block_scales.reshape(-1)
    n_total = flat_scales.numel()
    n_1bit = int(round(frac_1bit * n_total))

    sorted_idx = flat_scales.argsort()
    is_1bit = torch.zeros(n_total, dtype=torch.bool, device=device)
    is_1bit[sorted_idx[:n_1bit]] = True
    is_1bit = is_1bit.reshape(block_scales.shape)

    cb3 = CB_3BIT_UNIFORM.to(device)
    snap3 = cb3[(normalized.unsqueeze(-1) - cb3).abs().argmin(dim=-1)]

    cb1 = CB_1BIT.to(device)
    snap1 = cb1[(normalized.unsqueeze(-1) - cb1).abs().argmin(dim=-1)]

    snapped = torch.where(is_1bit.unsqueeze(2), snap1, snap3)

    result = snapped * safe_scales.unsqueeze(2) / global_scale
    result = result.reshape(rows, -1)[:, :cols]
    return result.to(weight_bf16.dtype)


ADAPTIVE_2BIT_OPTIONS = [
    (6.0, torch.tensor([-6, 0, 3, 6], dtype=torch.float32)),
    (4.0, torch.tensor([-4, 0, 2, 4], dtype=torch.float32)),
    (3.0, torch.tensor([-3, 0, 1.5, 3], dtype=torch.float32)),
    (2.0, torch.tensor([-2, 0, 1, 2], dtype=torch.float32)),
    (6.0, torch.tensor([-6, -2, 2, 6], dtype=torch.float32)),
    (4.0, torch.tensor([-4, -1.5, 1.5, 4], dtype=torch.float32)),
    (6.0, torch.tensor([-4, 0, 2, 6], dtype=torch.float32)),
    (4.0, torch.tensor([-4, -1, 0, 4], dtype=torch.float32)),
]


def snap_weight_adaptive_2bit(weight_bf16, device='cuda'):
    """Adaptive 2-bit: per block, try multiple (scale_divisor, codebook) combos, keep lowest MSE."""
    from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

    weight = weight_bf16.float().to(device)
    rows, cols = weight.shape

    global_scale = fp4_global_scale(weight).to(torch.float32)
    scaled = weight * global_scale

    pad = (BLOCK_SIZE - cols % BLOCK_SIZE) % BLOCK_SIZE
    if pad > 0:
        scaled = F.pad(scaled, (0, pad))

    blocks = scaled.reshape(rows, -1, BLOCK_SIZE)
    block_amax = blocks.abs().amax(dim=2)

    best_mse = torch.full((rows, blocks.shape[1]), float('inf'), device=device)
    best_recon = torch.zeros_like(blocks)

    for divisor, cb in ADAPTIVE_2BIT_OPTIONS:
        scales = (block_amax / divisor).to(torch.float8_e4m3fn).float()
        safe = torch.where(scales == 0, torch.ones_like(scales), scales)
        normalized = blocks / safe.unsqueeze(2)

        cb_dev = cb.to(device)
        snap = cb_dev[(normalized.unsqueeze(-1) - cb_dev).abs().argmin(dim=-1)]
        recon = snap * safe.unsqueeze(2)
        mse = ((blocks - recon) ** 2).mean(dim=2)

        improved = mse < best_mse
        best_mse = torch.where(improved, mse, best_mse)
        best_recon = torch.where(improved.unsqueeze(2), recon, best_recon)

    result = best_recon / global_scale
    result = result.reshape(rows, -1)[:, :cols]
    return result.to(weight_bf16.dtype)


def snap_weight_adaptive_3bit(weight_bf16, device='cuda'):
    """Adaptive 3-bit Four-Over-Six: per block, try scale-to-6 with ±{0,2,4,6} and scale-to-4 with ±{0,1,2,4}, keep lower MSE."""
    from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

    weight = weight_bf16.float().to(device)
    rows, cols = weight.shape

    global_scale = fp4_global_scale(weight).to(torch.float32)
    scaled = weight * global_scale

    pad = (BLOCK_SIZE - cols % BLOCK_SIZE) % BLOCK_SIZE
    if pad > 0:
        scaled = F.pad(scaled, (0, pad))

    blocks = scaled.reshape(rows, -1, BLOCK_SIZE)
    block_amax = blocks.abs().amax(dim=2)

    scales_6 = (block_amax / 6.0).to(torch.float8_e4m3fn).float()
    scales_4 = (block_amax / 4.0).to(torch.float8_e4m3fn).float()
    safe_6 = torch.where(scales_6 == 0, torch.ones_like(scales_6), scales_6)
    safe_4 = torch.where(scales_4 == 0, torch.ones_like(scales_4), scales_4)

    norm_6 = blocks / safe_6.unsqueeze(2)
    norm_4 = blocks / safe_4.unsqueeze(2)

    cb6 = CB_3BIT_SCALE6.to(device)
    cb4 = CB_3BIT_SCALE4.to(device)

    snap_6 = cb6[(norm_6.unsqueeze(-1) - cb6).abs().argmin(dim=-1)]
    snap_4 = cb4[(norm_4.unsqueeze(-1) - cb4).abs().argmin(dim=-1)]

    recon_6 = snap_6 * safe_6.unsqueeze(2)
    recon_4 = snap_4 * safe_4.unsqueeze(2)

    mse_6 = ((blocks - recon_6) ** 2).sum(dim=2)
    mse_4 = ((blocks - recon_4) ** 2).sum(dim=2)

    use_scale4 = (mse_4 < mse_6).unsqueeze(2)
    snapped_scaled = torch.where(use_scale4, recon_4, recon_6)

    result = snapped_scaled / global_scale
    result = result.reshape(rows, -1)[:, :cols]

    return result.to(weight_bf16.dtype)


def snap_weight_mixed_rate(weight_bf16, frac_2bit, device='cuda'):
    """Mixed bit-rate: rank blocks by scale magnitude, assign bottom frac_2bit to 2-bit, rest to 3-bit."""
    from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

    weight = weight_bf16.float().to(device)
    rows, cols = weight.shape

    global_scale = fp4_global_scale(weight).to(torch.float32)
    scaled = weight * global_scale

    pad = (BLOCK_SIZE - cols % BLOCK_SIZE) % BLOCK_SIZE
    if pad > 0:
        scaled = F.pad(scaled, (0, pad))

    blocks = scaled.reshape(rows, -1, BLOCK_SIZE)
    n_blocks_per_row = blocks.shape[1]
    block_amax = blocks.abs().amax(dim=2)
    block_scales = (block_amax / 6.0).to(torch.float8_e4m3fn).float()
    safe_scales = block_scales.clone()
    safe_scales[safe_scales == 0] = 1.0

    normalized = blocks / safe_scales.unsqueeze(2)

    flat_scales = block_scales.reshape(-1)
    n_total_blocks = flat_scales.numel()
    n_2bit = int(round(frac_2bit * n_total_blocks))

    sorted_indices = flat_scales.argsort()
    is_2bit = torch.zeros(n_total_blocks, dtype=torch.bool, device=device)
    is_2bit[sorted_indices[:n_2bit]] = True
    is_2bit = is_2bit.reshape(rows, n_blocks_per_row)

    cb3 = CB_3BIT.to(device)
    cb2 = CB_2BIT.to(device)

    diff3 = (normalized.unsqueeze(-1) - cb3).abs().argmin(dim=-1)
    snapped3 = cb3[diff3]

    diff2 = (normalized.unsqueeze(-1) - cb2).abs().argmin(dim=-1)
    snapped2 = cb2[diff2]

    snapped = torch.where(is_2bit.unsqueeze(2), snapped2, snapped3)

    result = snapped * safe_scales.unsqueeze(2) / global_scale
    result = result.reshape(rows, -1)[:, :cols]

    return result.to(weight_bf16.dtype)


def evaluate_sub_nvfp4(approach_name, codebook, device='cuda', snap_fn=None):
    snapshot_dir, root_config, weight_map = load_root_config(MODEL_ID)
    model_config = build_text_config(root_config)
    dtype = torch.bfloat16

    n_unique = len(codebook.unique())
    bits = math.log2(n_unique)
    effective_bpe = bits + 0.5  # sub-FP4 codes + FP8 scale overhead (8/16=0.5)

    print(f"\n{'='*60}")
    print(f"Approach: {approach_name}")
    print(f"Codebook: {sorted(codebook.unique().tolist())}")
    print(f"Codes: {n_unique}, Bits/code: {bits:.2f}, Effective: {effective_bpe:.2f} bits/elem")
    print(f"{'='*60}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    eval_ids, num_samples, seqlen = load_eval_data(tokenizer)
    eval_ids = eval_ids.to(device)

    weight_store = ee.WeightStore(MODEL_ID, snapshot_dir, weight_map)

    root_tensors = weight_store.load_tensors([
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ])
    embed_weight = move_tensor(root_tensors["model.language_model.embed_tokens.weight"], device, dtype)
    final_norm = move_tensor(root_tensors["model.language_model.norm.weight"], device, dtype)
    lm_head_weight = move_tensor(root_tensors["lm_head.weight"], device, dtype)
    del root_tensors

    eval_chunks = eval_ids[:, :num_samples * seqlen].view(num_samples, seqlen).contiguous()
    hidden_bank = torch.empty((num_samples, seqlen, model_config.hidden_size), dtype=dtype, device="cpu")
    for i in range(num_samples):
        hidden_bank[i].copy_(
            F.embedding(eval_chunks[i:i+1].to(device), embed_weight).squeeze(0).cpu()
        )

    position_ids = torch.arange(seqlen, device=device).unsqueeze(0)
    rotary = Qwen3NextRotaryEmbedding(config=model_config, device=device)
    dummy = torch.empty((1, seqlen, model_config.hidden_size), device=device, dtype=dtype)
    position_embeddings = rotary(dummy, position_ids)
    del dummy

    negative_log_likelihoods = []
    t0 = time.time()

    with torch.inference_mode():
        for layer_idx in range(model_config.num_hidden_layers):
            layer_type = model_config.layer_types[layer_idx]
            raw = weight_store.load_tensors(layer_keys(layer_idx, layer_type))
            layer_weights = shorten_layer_tensors(layer_idx, raw, device, dtype)
            del raw

            moe = {k.replace("mlp.", "", 1): v for k, v in layer_weights.items() if k.startswith("mlp.")}

            for sample_idx in range(num_samples):
                hidden = hidden_bank[sample_idx:sample_idx+1].to(device)
                residual = hidden

                hidden = rms_norm_qwen3_next(hidden, layer_weights["input_layernorm.weight"], model_config.rms_norm_eps)

                if layer_type == "full_attention":
                    attn_kv = {k.replace("self_attn.", ""): v for k, v in layer_weights.items() if k.startswith("self_attn.")}
                    hidden = ee.full_attention_forward_exact(
                        hidden, attn_kv, model_config, position_embeddings,
                        ee.build_causal_mask(seqlen, device), mode="bf16", quantized=False,
                    )
                else:
                    attn_kv = {k.replace("linear_attn.", ""): v for k, v in layer_weights.items() if k.startswith("linear_attn.")}
                    hidden = ee.linear_attention_forward_exact(hidden, attn_kv, model_config, "bf16", "moe_only")

                hidden = residual + hidden
                residual = hidden
                hidden = rms_norm_qwen3_next(hidden, layer_weights["post_attention_layernorm.weight"], model_config.rms_norm_eps)

                flat = hidden.view(-1, model_config.hidden_size)
                router_logits = ee.bf16_linear(flat, moe["gate.weight"]).float()
                routing_probs = torch.softmax(router_logits, dim=1)
                topk_weights, topk_ids = torch.topk(routing_probs, model_config.num_experts_per_tok, dim=-1)
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
                topk_weights = topk_weights.to(dtype)

                expert_output = torch.zeros_like(flat)
                active_counts = torch.bincount(topk_ids.reshape(-1), minlength=model_config.num_experts)

                for expert_idx in torch.nonzero(active_counts > 0, as_tuple=False).flatten().tolist():
                    token_indices, route_indices = torch.where(topk_ids == expert_idx)
                    expert_input = flat[token_indices]

                    w1 = moe["experts.gate_up_proj"][expert_idx]
                    w2 = moe["experts.down_proj"][expert_idx]

                    if snap_fn is not None:
                        w1_snapped = snap_fn(w1, device)
                        w2_snapped = snap_fn(w2, device)
                    else:
                        w1_snapped = snap_weight_to_sub_grid(w1, codebook, device)
                        w2_snapped = snap_weight_to_sub_grid(w2, codebook, device)

                    gate_up = ee.nvfp4_linear(expert_input, w1_snapped)
                    gate, up = gate_up.chunk(2, dim=-1)
                    intermediate = F.silu(gate) * up
                    expert_out = ee.nvfp4_linear(intermediate, w2_snapped)

                    weighted = expert_out * topk_weights[token_indices, route_indices].unsqueeze(-1)
                    expert_output.index_add_(0, token_indices, weighted.to(dtype))

                shared = ee.bf16_linear(flat, moe["shared_expert.gate_proj.weight"])
                shared = F.silu(shared) * ee.bf16_linear(flat, moe["shared_expert.up_proj.weight"])
                shared = ee.bf16_linear(shared, moe["shared_expert.down_proj.weight"])
                shared_gate = torch.sigmoid(ee.bf16_linear(flat, moe["shared_expert_gate.weight"]))

                hidden = residual + (expert_output + shared_gate * shared).view_as(residual)
                hidden_bank[sample_idx].copy_(hidden.squeeze(0).cpu())

            release_tensors(layer_weights)
            elapsed = time.time() - t0
            print(f"  Layer {layer_idx+1}/{model_config.num_hidden_layers} ({elapsed:.0f}s)", flush=True)

        for sample_idx in range(num_samples):
            hidden = hidden_bank[sample_idx:sample_idx+1].to(device)
            hidden = rms_norm_qwen3_next(hidden, final_norm, model_config.rms_norm_eps)
            logits = F.linear(hidden, lm_head_weight)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = eval_chunks[sample_idx:sample_idx+1, 1:].to(device).contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            negative_log_likelihoods.append(loss.item())

    final_ppl = math.exp(sum(negative_log_likelihoods) / len(negative_log_likelihoods))
    elapsed = time.time() - t0

    BF16_PPL = 6.5896
    NVFP4_PPL = 6.8431

    print(f"\n{'='*60}")
    print(f"RESULT: {approach_name}")
    print(f"  PPL: {final_ppl:.4f}")
    print(f"  BF16={BF16_PPL:.4f} | NVFP4={NVFP4_PPL:.4f}")
    if final_ppl <= NVFP4_PPL:
        recovery = (NVFP4_PPL - final_ppl) / (NVFP4_PPL - BF16_PPL) * 100
        print(f"  Recovery: {recovery:.1f}%")
    else:
        degradation_ppl = final_ppl - NVFP4_PPL
        print(f"  WORSE than NVFP4 by +{degradation_ppl:.4f} PPL")
    print(f"  Bits/elem: {effective_bpe:.2f}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"{'='*60}")

    return {'approach': approach_name, 'ppl': final_ppl, 'bits_per_code': bits,
            'effective_bpe': effective_bpe, 'time': elapsed,
            'codebook': sorted(codebook.unique().tolist())}


def evaluate_mixed_rate(frac_2bit, device='cuda'):
    snapshot_dir, root_config, weight_map = load_root_config(MODEL_ID)
    model_config = build_text_config(root_config)
    dtype = torch.bfloat16

    avg_bits = 3.0 * (1 - frac_2bit) + 2.0 * frac_2bit
    effective_bpe = avg_bits + 0.5 + 1.0/BLOCK_SIZE

    approach_name = f"mixed_rate_{int(frac_2bit*100)}pct_2bit"
    print(f"\n{'='*60}")
    print(f"MIXED BIT-RATE: {frac_2bit*100:.0f}% blocks at 2-bit, rest at 3-bit")
    print(f"3-bit codebook: {sorted(CB_3BIT.unique().tolist())}")
    print(f"2-bit codebook: {sorted(CB_2BIT.unique().tolist())}")
    print(f"Avg bits/code: {avg_bits:.2f}, Effective bits/elem: {effective_bpe:.2f}")
    print(f"{'='*60}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    eval_ids, num_samples, seqlen = load_eval_data(tokenizer)
    eval_ids = eval_ids.to(device)

    weight_store = ee.WeightStore(MODEL_ID, snapshot_dir, weight_map)

    root_tensors = weight_store.load_tensors([
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ])
    embed_weight = move_tensor(root_tensors["model.language_model.embed_tokens.weight"], device, dtype)
    final_norm = move_tensor(root_tensors["model.language_model.norm.weight"], device, dtype)
    lm_head_weight = move_tensor(root_tensors["lm_head.weight"], device, dtype)
    del root_tensors

    eval_chunks = eval_ids[:, :num_samples * seqlen].view(num_samples, seqlen).contiguous()
    hidden_bank = torch.empty((num_samples, seqlen, model_config.hidden_size), dtype=dtype, device="cpu")
    for i in range(num_samples):
        hidden_bank[i].copy_(
            F.embedding(eval_chunks[i:i+1].to(device), embed_weight).squeeze(0).cpu()
        )

    position_ids = torch.arange(seqlen, device=device).unsqueeze(0)
    rotary = Qwen3NextRotaryEmbedding(config=model_config, device=device)
    dummy = torch.empty((1, seqlen, model_config.hidden_size), device=device, dtype=dtype)
    position_embeddings = rotary(dummy, position_ids)
    del dummy

    negative_log_likelihoods = []
    t0 = time.time()

    with torch.inference_mode():
        for layer_idx in range(model_config.num_hidden_layers):
            layer_type = model_config.layer_types[layer_idx]
            raw = weight_store.load_tensors(layer_keys(layer_idx, layer_type))
            layer_weights = shorten_layer_tensors(layer_idx, raw, device, dtype)
            del raw

            moe = {k.replace("mlp.", "", 1): v for k, v in layer_weights.items() if k.startswith("mlp.")}

            for sample_idx in range(num_samples):
                hidden = hidden_bank[sample_idx:sample_idx+1].to(device)
                residual = hidden

                hidden = rms_norm_qwen3_next(hidden, layer_weights["input_layernorm.weight"], model_config.rms_norm_eps)

                if layer_type == "full_attention":
                    attn_kv = {k.replace("self_attn.", ""): v for k, v in layer_weights.items() if k.startswith("self_attn.")}
                    hidden = ee.full_attention_forward_exact(
                        hidden, attn_kv, model_config, position_embeddings,
                        ee.build_causal_mask(seqlen, device), mode="bf16", quantized=False,
                    )
                else:
                    attn_kv = {k.replace("linear_attn.", ""): v for k, v in layer_weights.items() if k.startswith("linear_attn.")}
                    hidden = ee.linear_attention_forward_exact(hidden, attn_kv, model_config, "bf16", "moe_only")

                hidden = residual + hidden
                residual = hidden
                hidden = rms_norm_qwen3_next(hidden, layer_weights["post_attention_layernorm.weight"], model_config.rms_norm_eps)

                flat = hidden.view(-1, model_config.hidden_size)
                router_logits = ee.bf16_linear(flat, moe["gate.weight"]).float()
                routing_probs = torch.softmax(router_logits, dim=1)
                topk_weights, topk_ids = torch.topk(routing_probs, model_config.num_experts_per_tok, dim=-1)
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
                topk_weights = topk_weights.to(dtype)

                expert_output = torch.zeros_like(flat)
                active_counts = torch.bincount(topk_ids.reshape(-1), minlength=model_config.num_experts)

                for expert_idx in torch.nonzero(active_counts > 0, as_tuple=False).flatten().tolist():
                    token_indices, route_indices = torch.where(topk_ids == expert_idx)
                    expert_input = flat[token_indices]

                    w1 = moe["experts.gate_up_proj"][expert_idx]
                    w2 = moe["experts.down_proj"][expert_idx]

                    w1_snapped = snap_weight_mixed_rate(w1, frac_2bit, device)
                    w2_snapped = snap_weight_mixed_rate(w2, frac_2bit, device)

                    gate_up = ee.nvfp4_linear(expert_input, w1_snapped)
                    gate, up = gate_up.chunk(2, dim=-1)
                    intermediate = F.silu(gate) * up
                    expert_out = ee.nvfp4_linear(intermediate, w2_snapped)

                    weighted = expert_out * topk_weights[token_indices, route_indices].unsqueeze(-1)
                    expert_output.index_add_(0, token_indices, weighted.to(dtype))

                shared = ee.bf16_linear(flat, moe["shared_expert.gate_proj.weight"])
                shared = F.silu(shared) * ee.bf16_linear(flat, moe["shared_expert.up_proj.weight"])
                shared = ee.bf16_linear(shared, moe["shared_expert.down_proj.weight"])
                shared_gate = torch.sigmoid(ee.bf16_linear(flat, moe["shared_expert_gate.weight"]))

                hidden = residual + (expert_output + shared_gate * shared).view_as(residual)
                hidden_bank[sample_idx].copy_(hidden.squeeze(0).cpu())

            release_tensors(layer_weights)
            elapsed = time.time() - t0
            print(f"  Layer {layer_idx+1}/{model_config.num_hidden_layers} ({elapsed:.0f}s)", flush=True)

        for sample_idx in range(num_samples):
            hidden = hidden_bank[sample_idx:sample_idx+1].to(device)
            hidden = rms_norm_qwen3_next(hidden, final_norm, model_config.rms_norm_eps)
            logits = F.linear(hidden, lm_head_weight)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = eval_chunks[sample_idx:sample_idx+1, 1:].to(device).contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            negative_log_likelihoods.append(loss.item())

    final_ppl = math.exp(sum(negative_log_likelihoods) / len(negative_log_likelihoods))
    elapsed = time.time() - t0

    BF16_PPL = 6.5896
    NVFP4_PPL = 6.8431

    print(f"\n{'='*60}")
    print(f"RESULT: {approach_name}")
    print(f"  PPL: {final_ppl:.4f}")
    print(f"  BF16={BF16_PPL:.4f} | NVFP4={NVFP4_PPL:.4f}")
    if final_ppl <= NVFP4_PPL:
        print(f"  Recovery: {(NVFP4_PPL - final_ppl) / (NVFP4_PPL - BF16_PPL) * 100:.1f}%")
    else:
        print(f"  WORSE than NVFP4 by +{final_ppl - NVFP4_PPL:.4f} PPL")
    print(f"  Avg bits/code: {avg_bits:.2f}, Effective: {effective_bpe:.2f} bits/elem")
    print(f"  Time: {elapsed:.0f}s")
    print(f"{'='*60}")

    return {'approach': approach_name, 'ppl': final_ppl, 'frac_2bit': frac_2bit,
            'avg_bits_per_code': avg_bits, 'effective_bpe': effective_bpe, 'time': elapsed}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach', type=str, default=None, choices=list(CODEBOOKS.keys()))
    parser.add_argument('--mixed', type=float, default=None)
    parser.add_argument('--adaptive', action='store_true')
    parser.add_argument('--adaptive2bit', action='store_true')
    parser.add_argument('--pq', action='store_true')
    parser.add_argument('--mixed31', type=float, default=None,
                        help='Fraction of blocks at 1-bit (rest at 3-bit)')
    parser.add_argument('--stochastic', action='store_true')
    args = parser.parse_args()

    if args.pq:
        print("Learning PQ codebooks...", flush=True)
        pq_codebooks = learn_pq_codebooks(max_experts=20, n_clusters=256, sub_size=4)
        result = evaluate_sub_nvfp4('pq_2bit', CB_2BIT,
                                    snap_fn=lambda w, d: snap_weight_product_quantized_2bit(w, pq_codebooks, d))
        out_path = Path("/code/tensorrt_llm/scripts/channel_quant_new/profiling/sub_nvfp4_pq_2bit.json")
    elif args.mixed31 is not None:
        avg_bits = 3.0 * (1 - args.mixed31) + 1.0 * args.mixed31
        result = evaluate_sub_nvfp4(f'mixed31_{int(args.mixed31*100)}pct', CB_3BIT_UNIFORM,
                                    snap_fn=lambda w, d: snap_weight_mixed_3bit_1bit(w, args.mixed31, d))
        result['avg_bits_per_code'] = avg_bits
        result['effective_bpe'] = avg_bits + 0.5 + 1.0/BLOCK_SIZE
        out_path = Path(f"/code/tensorrt_llm/scripts/channel_quant_new/profiling/sub_nvfp4_mixed31_{int(args.mixed31*100)}.json")
    elif args.adaptive2bit:
        result = evaluate_sub_nvfp4('adaptive_2bit', CB_2BIT,
                                    snap_fn=lambda w, d: snap_weight_adaptive_2bit(w, d))
        out_path = Path("/code/tensorrt_llm/scripts/channel_quant_new/profiling/sub_nvfp4_adaptive_2bit.json")
    elif args.stochastic:
        cb = CODEBOOKS['3bit_uniform']
        result = evaluate_sub_nvfp4('3bit_stochastic', cb,
                                    snap_fn=lambda w, d: snap_weight_stochastic_3bit(w, cb, d))
        out_path = Path("/code/tensorrt_llm/scripts/channel_quant_new/profiling/sub_nvfp4_3bit_stochastic.json")
    elif args.adaptive:
        CODEBOOKS['adaptive_3bit'] = CB_3BIT_SCALE6
        result = evaluate_sub_nvfp4('adaptive_3bit', CB_3BIT_SCALE6, snap_fn=snap_weight_adaptive_3bit)
        out_path = Path("/code/tensorrt_llm/scripts/channel_quant_new/profiling/sub_nvfp4_adaptive_3bit.json")
    elif args.mixed is not None:
        result = evaluate_mixed_rate(args.mixed)
        out_path = Path(f"/code/tensorrt_llm/scripts/channel_quant_new/profiling/sub_nvfp4_mixed_{int(args.mixed*100)}.json")
    elif args.approach is not None:
        result = evaluate_sub_nvfp4(args.approach, CODEBOOKS[args.approach])
        out_path = Path(f"/code/tensorrt_llm/scripts/channel_quant_new/profiling/sub_nvfp4_{args.approach}.json")
    else:
        parser.error("Specify --approach, --mixed, or --adaptive")
        return

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Phase 3: Per-block optimal codebook selection with library approach.

Key insight: Instead of a single global codebook, use a library of K codebooks.
Each block of 16 FP4 codes selects the best codebook from the library.
Overhead: log2(K) bits per block = log2(K)/16 bits/elem.

For K=16: 4 bits per block = 0.25 bits/elem overhead → 3.25 bits/elem total
For K=256: 8 bits per block = 0.5 bits/elem overhead → 3.5 bits/elem total

Library construction:
1. Sample blocks from the model
2. For each block, find optimal K-element subset of FP4 codes (minimize MSE)
3. Cluster the per-block optimal codebooks into a library using K-means
4. Evaluate PPL with library-based codebook selection

This script:
1. Analyzes FP4 code distribution across the model
2. Builds a library of codebooks using K-means on per-block code distributions
3. Evaluates PPL with library-based codebook selection
"""

import sys
import time
import json
import logging
from pathlib import Path
from itertools import combinations
from collections import Counter
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, "/code/tensorrt_llm/scripts/nvfp4_compress")
sys.path.insert(0, "/workspace/channel_quant_new")
sys.path.insert(0, "/workspace/channel_quant")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

import tensorrt_llm._torch.auto_deploy.custom_ops
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale
from exact_docker_eval import (
    EvalConfig, ensure_runtime_available, flatten_for_linear, restore_linear_shape,
    bf16_linear, exact_linear, quantize_shared_expert, quantize_full_attention,
    full_attention_forward_exact, linear_attention_forward_exact,
    SCALING_VECTOR_SIZE,
)
from real_eval_pipeline import load_eval_data
from spike1_ground_truth import (
    MODEL_ID, WeightStore, build_causal_mask, build_text_config, layer_keys,
    load_root_config, move_tensor, release_tensors, rms_norm_qwen3_next, shorten_layer_tensors,
)
from sub_fp4_compress import E2M1_TABLE, unpack_fp4_codes, repack_fp4_codes, build_code_lut, apply_code_lut
from transformers import AutoTokenizer

BLOCK_SIZE = 16
BF16_PPL = 6.5896
NVFP4_PPL = 6.8431  # MoE-only NVFP4 baseline (from Phase 2 identity)


# ── Codebook library construction ────────────────────────────────────────────

def find_optimal_k_subset(codes: torch.Tensor, k: int) -> list[int]:
    """Find optimal K-element subset of FP4 codes for a block.
    
    Uses frequency-based greedy: start with most frequent codes,
    then add codes that minimize MSE.
    
    Args:
        codes: [N] tensor of FP4 codes (0-15)
        k: number of codes in sub-codebook
    
    Returns:
        list of k code indices (0-15)
    """
    # Get unique codes and their frequencies
    code_counts = Counter(codes.tolist())
    unique_codes = sorted(code_counts.keys(), key=lambda c: -code_counts[c])
    
    # If we have <= k unique codes, use them all (pad with 0 if needed)
    if len(unique_codes) <= k:
        result = unique_codes[:]
        while len(result) < k:
            result.append(0)
        return result[:k]
    
    # Greedy: start with most frequent, add codes that minimize MSE
    selected = [unique_codes[0]]
    
    for _ in range(k - 1):
        best_code = None
        best_mse = float('inf')
        
        for candidate in range(16):
            if candidate in selected:
                continue
            
            trial = selected + [candidate]
            trial_vals = E2M1_TABLE[trial]
            
            # Compute MSE for this trial codebook
            mse = 0.0
            for code in codes:
                src_val = E2M1_TABLE[code.item()]
                dists = (trial_vals - src_val).abs()
                nearest_val = trial_vals[dists.argmin()]
                mse += (src_val - nearest_val).item() ** 2
            
            if mse < best_mse:
                best_mse = mse
                best_code = candidate
        
        selected.append(best_code)
    
    return selected


def build_codebook_library_from_model(
    weight_map: dict,
    snapshot_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    num_sample_layers: int = 5,
    num_sample_experts: int = 10,
    k_per_block: int = 8,
    library_size: int = 16,
) -> list[list[int]]:
    """Build a library of codebooks by analyzing FP4 code distributions.
    
    1. Sample blocks from the model
    2. For each block, find optimal K-element subset
    3. Cluster the per-block codebooks into a library
    
    Returns:
        library: list of library_size codebooks, each a list of k_per_block code indices
    """
    from sklearn.cluster import KMeans
    
    log.info(f"Building codebook library (k={k_per_block}, library_size={library_size})...")
    
    store = WeightStore(MODEL_ID, snapshot_dir, weight_map)
    
    # Collect per-block optimal codebooks
    all_block_codebooks = []  # each entry: list of k_per_block code indices
    
    # Sample from a few layers
    layer_indices = list(range(0, 40, 40 // num_sample_layers))[:num_sample_layers]
    
    for layer_idx in layer_indices:
        log.info(f"  Sampling layer {layer_idx}...")
        
        # Load expert weights for this layer
        gate_up_key = f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"
        down_key = f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj"
        
        if gate_up_key not in weight_map:
            continue
        
        raw = store.load_tensors([gate_up_key, down_key])
        gate_up = move_tensor(raw[gate_up_key], device, dtype)  # [num_experts, out, in]
        down = move_tensor(raw[down_key], device, dtype)
        
        # Sample a few experts
        expert_indices = list(range(0, gate_up.shape[0], gate_up.shape[0] // num_sample_experts))[:num_sample_experts]
        
        for expert_idx in expert_indices:
            for weight in [gate_up[expert_idx], down[expert_idx]]:
                # Quantize to FP4
                s_w2 = fp4_global_scale(weight).to(torch.float32)
                weight_fp4, _ = torch.ops.trtllm.fp4_quantize(weight, s_w2, SCALING_VECTOR_SIZE, False)
                
                # Unpack codes
                codes = unpack_fp4_codes(weight_fp4)  # [out, in]
                
                # Sample blocks
                M, K = codes.shape
                num_blocks = M * (K // BLOCK_SIZE)
                codes_blocked = codes.view(M, K // BLOCK_SIZE, BLOCK_SIZE)
                
                # Sample up to 100 blocks per weight
                sample_size = min(100, num_blocks)
                block_indices = torch.randperm(num_blocks)[:sample_size]
                
                for bi in block_indices:
                    row = bi // (K // BLOCK_SIZE)
                    col = bi % (K // BLOCK_SIZE)
                    block_codes = codes_blocked[row, col]  # [BLOCK_SIZE]
                    
                    # Find optimal K-element subset for this block
                    optimal_cb = find_optimal_k_subset(block_codes.cpu(), k_per_block)
                    all_block_codebooks.append(optimal_cb)
        
        del raw, gate_up, down
        torch.cuda.empty_cache()
    
    log.info(f"  Collected {len(all_block_codebooks)} per-block codebooks")
    
    if len(all_block_codebooks) < library_size:
        log.warning(f"  Not enough blocks ({len(all_block_codebooks)}) for library size {library_size}")
        library_size = len(all_block_codebooks)
    
    # Convert to feature vectors for clustering
    # Feature: binary vector of which codes are in the codebook
    features = np.zeros((len(all_block_codebooks), 16), dtype=np.float32)
    for i, cb in enumerate(all_block_codebooks):
        for code in cb:
            features[i, code] = 1.0
    
    # Cluster into library_size codebooks
    kmeans = KMeans(n_clusters=library_size, n_init=10, random_state=42)
    kmeans.fit(features)
    
    # Convert cluster centers to codebooks
    library = []
    for center in kmeans.cluster_centers_:
        # Select top k_per_block codes by cluster center value
        top_codes = np.argsort(center)[-k_per_block:].tolist()
        library.append(sorted(top_codes))
    
    log.info(f"  Built library of {len(library)} codebooks")
    for i, cb in enumerate(library):
        cb_vals = [E2M1_TABLE[c].item() for c in cb]
        log.info(f"    CB{i:2d}: codes={cb} values={[f'{v:.1f}' for v in cb_vals]}")
    
    return library


def build_per_block_lut(
    weight: torch.Tensor,
    library: list[list[int]],
    device: torch.device,
) -> torch.Tensor:
    """For each block of 16 elements, select the best library codebook.
    
    Returns:
        lut: [M, K//16] tensor of library indices (uint8)
    """
    s_w2 = fp4_global_scale(weight).to(torch.float32)
    weight_fp4, _ = torch.ops.trtllm.fp4_quantize(weight, s_w2, SCALING_VECTOR_SIZE, False)
    codes = unpack_fp4_codes(weight_fp4)  # [M, K]
    
    M, K = codes.shape
    num_blocks_per_row = K // BLOCK_SIZE
    codes_blocked = codes.view(M, num_blocks_per_row, BLOCK_SIZE)
    
    # For each block, find best library codebook
    block_lut = torch.zeros(M, num_blocks_per_row, dtype=torch.uint8, device=device)
    
    # Precompute library codebook values
    lib_vals = []
    for cb in library:
        lib_vals.append(E2M1_TABLE[cb].to(device))
    
    for m in range(M):
        for b in range(num_blocks_per_row):
            block = codes_blocked[m, b]  # [BLOCK_SIZE]
            block_vals = E2M1_TABLE[block.long()].to(device)
            
            best_cb_idx = 0
            best_mse = float('inf')
            
            for cb_idx, cb_vals in enumerate(lib_vals):
                # Compute MSE for this codebook
                mse = 0.0
                for val in block_vals:
                    dists = (cb_vals - val).abs()
                    nearest = cb_vals[dists.argmin()]
                    mse += (val - nearest).item() ** 2
                
                if mse < best_mse:
                    best_mse = mse
                    best_cb_idx = cb_idx
            
            block_lut[m, b] = best_cb_idx
    
    return block_lut


def apply_per_block_codebook(
    weight_fp4: torch.Tensor,
    block_lut: torch.Tensor,
    library: list[list[int]],
    device: torch.device,
) -> torch.Tensor:
    """Apply per-block codebook mapping to packed FP4 weights.
    
    Args:
        weight_fp4: [M, K//2] packed uint8
        block_lut: [M, K//16] library indices
        library: list of codebooks
    
    Returns:
        mapped_fp4: [M, K//2] packed uint8 with codebook mapping applied
    """
    codes = unpack_fp4_codes(weight_fp4)  # [M, K]
    M, K = codes.shape
    num_blocks_per_row = K // BLOCK_SIZE
    codes_blocked = codes.view(M, num_blocks_per_row, BLOCK_SIZE)
    
    # Precompute LUTs for each library codebook
    luts = []
    for cb in library:
        cb_vals = [E2M1_TABLE[c].item() for c in cb]
        luts.append(build_code_lut(cb_vals).to(device))
    
    # Apply per-block mapping
    mapped_blocked = codes_blocked.clone()
    for m in range(M):
        for b in range(num_blocks_per_row):
            cb_idx = block_lut[m, b].item()
            lut = luts[cb_idx]
            mapped_blocked[m, b] = apply_code_lut(codes_blocked[m, b].unsqueeze(0), lut).squeeze(0)
    
    mapped_codes = mapped_blocked.view(M, K)
    return repack_fp4_codes(mapped_codes)


# ── NVFP4 linear with per-block codebook ─────────────────────────────────────

def nvfp4_linear_perblock(
    input: torch.Tensor,
    weight: torch.Tensor,
    library: list[list[int]],
    device: torch.device,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """NVFP4 linear with per-block optimal codebook selection."""
    input_2d, prefix_shape = flatten_for_linear(input)
    s_in2 = fp4_global_scale(input_2d).to(torch.float32)
    s_w2 = fp4_global_scale(weight).to(torch.float32)
    
    weight_fp4, weight_scale = torch.ops.trtllm.fp4_quantize(weight, s_w2, SCALING_VECTOR_SIZE, False)
    
    # Build per-block LUT and apply
    block_lut = build_per_block_lut(weight, library, device)
    weight_fp4_mapped = apply_per_block_codebook(weight_fp4, block_lut, library, device)
    
    alpha = (1.0 / (s_in2 * s_w2)).to(torch.float32)
    out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        input_2d, weight_fp4_mapped, bias=bias,
        input_scale=s_in2, weight_scale=weight_scale, alpha=alpha,
    )
    return restore_linear_shape(out, prefix_shape)


# ── MoE forward with per-block codebook ──────────────────────────────────────

def moe_forward_perblock(
    hidden_states: torch.Tensor,
    tensors: dict,
    config,
    mode: str,
    scope: str,
    library: Optional[list[list[int]]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """MoE forward with per-block codebook selection."""
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)

    router_logits = bf16_linear(flat, tensors["gate.weight"]).float()
    routing_weights = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(routing_weights, config.num_experts_per_tok, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype, device=hidden_states.device
    )
    gate_up_proj = tensors["experts.gate_up_proj"]
    down_proj = tensors["experts.down_proj"]
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]

        if mode == "nvfp4" and library is not None:
            gate_up = nvfp4_linear_perblock(current_state, gate_up_proj[expert_idx], library, device)
        else:
            gate_up = exact_linear(current_state, gate_up_proj[expert_idx], mode=mode)

        gate, up = gate_up.chunk(2, dim=-1)

        if mode == "nvfp4" and library is not None:
            current_hidden = nvfp4_linear_perblock(F.silu(gate) * up, down_proj[expert_idx], library, device)
        else:
            current_hidden = exact_linear(F.silu(gate) * up, down_proj[expert_idx], mode=mode)

        current_hidden = current_hidden * routing_weights[token_idx, route_pos].unsqueeze(-1)
        final_hidden_states.index_add_(0, token_idx, current_hidden.to(hidden_states.dtype))

    if quantize_shared_expert(scope) and mode != "bf16":
        shared_gate_proj = exact_linear(flat, tensors["shared_expert.gate_proj.weight"], mode=mode)
        shared_up_proj = exact_linear(flat, tensors["shared_expert.up_proj.weight"], mode=mode)
        shared = exact_linear(F.silu(shared_gate_proj) * shared_up_proj, tensors["shared_expert.down_proj.weight"], mode=mode)
    else:
        shared = bf16_linear(flat, tensors["shared_expert.gate_proj.weight"])
        shared = F.silu(shared) * bf16_linear(flat, tensors["shared_expert.up_proj.weight"])
        shared = bf16_linear(shared, tensors["shared_expert.down_proj.weight"])

    shared_gate = torch.sigmoid(bf16_linear(flat, tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_gate * shared
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


# ── PPL evaluation ───────────────────────────────────────────────────────────

def evaluate_ppl_perblock(
    eval_ids: torch.Tensor,
    nsamples: int,
    seqlen: int,
    config,
    weight_map: dict,
    snapshot_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    label: str,
    library: Optional[list[list[int]]] = None,
    layer_batch_size: int = 4,
) -> float:
    """Evaluate PPL with per-block codebook selection."""
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

    run_config = EvalConfig(label=label, mode="nvfp4", quant_scope="moe_only")
    store = WeightStore(MODEL_ID, snapshot_dir, weight_map)

    embed_key = "model.language_model.embed_tokens.weight"
    norm_key = "model.language_model.norm.weight"
    lm_head_key = "lm_head.weight"

    root_t = store.load_tensors([embed_key, norm_key, lm_head_key])
    embed_w = move_tensor(root_t[embed_key], device, dtype)
    final_norm_w = move_tensor(root_t[norm_key], device, dtype)
    lm_head_w = move_tensor(root_t[lm_head_key], device, dtype)
    del root_t

    eval_chunks = eval_ids[:, :nsamples * seqlen].view(nsamples, seqlen).contiguous()
    hidden_bank = torch.empty((nsamples, seqlen, config.hidden_size), dtype=dtype, device="cpu")

    for batch_start in range(0, nsamples, layer_batch_size):
        batch_end = min(batch_start + layer_batch_size, nsamples)
        chunk_batch = eval_chunks[batch_start:batch_end].to(device)
        hidden_batch = F.embedding(chunk_batch, embed_w)
        hidden_bank[batch_start:batch_end].copy_(hidden_batch.cpu())
        del chunk_batch, hidden_batch

    causal_mask = build_causal_mask(seqlen, device)
    position_ids = torch.arange(seqlen, device=device).unsqueeze(0)
    rotary_input = torch.empty((1, seqlen, config.hidden_size), device=device, dtype=dtype)
    rotary = Qwen3NextRotaryEmbedding(config=config, device=device)
    position_embeddings = rotary(rotary_input, position_ids)
    del rotary_input

    nlls = []

    with torch.inference_mode():
        for layer_idx in range(config.num_hidden_layers):
            layer_type = config.layer_types[layer_idx]
            raw = store.load_tensors(layer_keys(layer_idx, layer_type))
            layer_tensors = shorten_layer_tensors(layer_idx, raw, device, dtype)
            del raw

            for batch_start in range(0, nsamples, layer_batch_size):
                batch_end = min(batch_start + layer_batch_size, nsamples)
                hidden_states = hidden_bank[batch_start:batch_end].to(device)

                residual = hidden_states
                hidden_states = rms_norm_qwen3_next(hidden_states, layer_tensors["input_layernorm.weight"], config.rms_norm_eps)

                if layer_type == "full_attention":
                    attn_tensors = {k.replace("self_attn.", ""): v for k, v in layer_tensors.items() if k.startswith("self_attn.")}
                    hidden_states = full_attention_forward_exact(
                        hidden_states, attn_tensors, config, position_embeddings, causal_mask,
                        run_config.mode,
                        quantized=(run_config.mode != "bf16" and quantize_full_attention(run_config.quant_scope)),
                    )
                else:
                    attn_tensors = {k.replace("linear_attn.", ""): v for k, v in layer_tensors.items() if k.startswith("linear_attn.")}
                    hidden_states = linear_attention_forward_exact(hidden_states, attn_tensors, config, run_config.mode, run_config.quant_scope)

                hidden_states = residual + hidden_states
                residual = hidden_states
                hidden_states = rms_norm_qwen3_next(hidden_states, layer_tensors["post_attention_layernorm.weight"], config.rms_norm_eps)

                moe_tensors = {k.replace("mlp.", "", 1): v for k, v in layer_tensors.items() if k.startswith("mlp.")}
                moe_out = moe_forward_perblock(
                    hidden_states, moe_tensors, config,
                    run_config.mode, run_config.quant_scope, library, device
                )
                hidden_states = residual + moe_out
                hidden_bank[batch_start:batch_end].copy_(hidden_states.cpu())
                del hidden_states, residual, moe_out

            release_tensors(layer_tensors)
            log.info(f"  [{label}] layer {layer_idx + 1}/{config.num_hidden_layers}")

        for sample_idx in range(nsamples):
            torch.cuda.empty_cache()
            chunk = eval_chunks[sample_idx:sample_idx + 1].to(device)
            hidden_states = hidden_bank[sample_idx:sample_idx + 1].to(device)
            hidden_states = rms_norm_qwen3_next(hidden_states, final_norm_w, config.rms_norm_eps)
            logits = F.linear(hidden_states, lm_head_w)

            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = chunk[:, 1:]
            loss = F.cross_entropy(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            nlls.append(loss.float() * seqlen)
            del logits, shift_logits, hidden_states

            if (sample_idx + 1) % 10 == 0:
                log.info(f"  [{label}] logits {sample_idx + 1}/{nsamples}")

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()
    del embed_w, final_norm_w, lm_head_w, store
    torch.cuda.empty_cache()
    return ppl


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ensure_runtime_available()
    log.info("✓ TRT-LLM NVFP4 runtime available")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    snapshot_dir, config_raw, weight_map = load_root_config(MODEL_ID)
    config = build_text_config(config_raw)
    log.info(f"✓ Model: {MODEL_ID}, layers={config.num_hidden_layers}, experts={config.num_experts}")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot_dir))
    eval_ids, nsamples, seqlen = load_eval_data(tokenizer)
    log.info(f"✓ Eval data: {nsamples} chunks × {seqlen} tokens")

    output_dir = Path("/code/tensorrt_llm/scripts/nvfp4_compress/phase3_results")
    output_dir.mkdir(exist_ok=True)

    results = {}

    # Experiment 3.1: Library of 16 codebooks, 8 codes each
    # Overhead: 4 bits per block = 0.25 bits/elem → 3.25 bits/elem total
    log.info("\n" + "="*60)
    log.info("Experiment 3.1: Library-16 (8 codes/block, 4 bits/block overhead)")
    log.info("Expected bits/elem: 3.25 (3 code bits + 0.25 overhead)")
    log.info("="*60)

    library_16 = build_codebook_library_from_model(
        weight_map, snapshot_dir, device, dtype,
        num_sample_layers=5, num_sample_experts=10,
        k_per_block=8, library_size=16,
    )

    # Save library
    with open(output_dir / "library_16.json", "w") as f:
        json.dump({"library": library_16, "k_per_block": 8, "library_size": 16}, f, indent=2)

    t0 = time.time()
    ppl_lib16 = evaluate_ppl_perblock(
        eval_ids, nsamples, seqlen, config, weight_map,
        snapshot_dir, device, dtype,
        label="library_16",
        library=library_16,
        layer_batch_size=4,
    )
    elapsed = time.time() - t0

    delta = ppl_lib16 - NVFP4_PPL
    log.info(f"✓ Library-16 PPL: {ppl_lib16:.4f} (Δ{delta:+.4f} vs NVFP4)")
    log.info(f"  Bits/elem: 3.25 (3 code + 0.25 overhead)")
    log.info(f"  Time: {elapsed:.1f}s")

    results["library_16"] = {
        "ppl": ppl_lib16,
        "delta_vs_nvfp4": delta,
        "bits_per_elem": 3.25,
        "k_per_block": 8,
        "library_size": 16,
        "time_sec": elapsed,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Experiment 3.2: Library of 256 codebooks, 8 codes each
    # Overhead: 8 bits per block = 0.5 bits/elem → 3.5 bits/elem total
    log.info("\n" + "="*60)
    log.info("Experiment 3.2: Library-256 (8 codes/block, 8 bits/block overhead)")
    log.info("Expected bits/elem: 3.5 (3 code bits + 0.5 overhead)")
    log.info("="*60)

    library_256 = build_codebook_library_from_model(
        weight_map, snapshot_dir, device, dtype,
        num_sample_layers=5, num_sample_experts=10,
        k_per_block=8, library_size=256,
    )

    with open(output_dir / "library_256.json", "w") as f:
        json.dump({"library": library_256, "k_per_block": 8, "library_size": 256}, f, indent=2)

    t0 = time.time()
    ppl_lib256 = evaluate_ppl_perblock(
        eval_ids, nsamples, seqlen, config, weight_map,
        snapshot_dir, device, dtype,
        label="library_256",
        library=library_256,
        layer_batch_size=4,
    )
    elapsed = time.time() - t0

    delta = ppl_lib256 - NVFP4_PPL
    log.info(f"✓ Library-256 PPL: {ppl_lib256:.4f} (Δ{delta:+.4f} vs NVFP4)")
    log.info(f"  Bits/elem: 3.5 (3 code + 0.5 overhead)")
    log.info(f"  Time: {elapsed:.1f}s")

    results["library_256"] = {
        "ppl": ppl_lib256,
        "delta_vs_nvfp4": delta,
        "bits_per_elem": 3.5,
        "k_per_block": 8,
        "library_size": 256,
        "time_sec": elapsed,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    log.info("\n" + "="*60)
    log.info("PHASE 3 SUMMARY")
    log.info("="*60)
    log.info(f"NVFP4 baseline: {NVFP4_PPL:.4f}")
    log.info(f"BF16 baseline:  {BF16_PPL:.4f}")
    for name, res in results.items():
        status = "✓ PASS" if res['delta_vs_nvfp4'] < 0.02 else "✗ FAIL"
        log.info(f"  {name:15s}: PPL={res['ppl']:.4f} (Δ{res['delta_vs_nvfp4']:+.4f}) | bits={res['bits_per_elem']:.2f} | {status}")

    log.info(f"\n✓ Results saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()

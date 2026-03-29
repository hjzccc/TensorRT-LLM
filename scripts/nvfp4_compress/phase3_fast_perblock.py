#!/usr/bin/env python3
"""Phase 3 FAST: Vectorized per-block codebook selection.

The naive per-block approach iterates over every block in Python loops,
which is O(M × K/16) per weight matrix — extremely slow for large expert weights.

This script uses vectorized operations:
1. Build library LUTs as a [library_size, 16] tensor
2. For each block, compute MSE for all library codebooks simultaneously
3. Select best codebook per block using argmin
4. Apply mapping using gather operations

Expected speedup: 100-1000x over naive Python loops.
"""

import sys
import time
import json
import logging
from pathlib import Path
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
from sub_fp4_compress import E2M1_TABLE, unpack_fp4_codes, repack_fp4_codes
from transformers import AutoTokenizer

BLOCK_SIZE = 16
BF16_PPL = 6.5896
NVFP4_PPL_IDENTITY = None  # Will be set from Phase 2 results


# ── Vectorized library operations ────────────────────────────────────────────

def build_library_luts(library: list[list[int]], device: torch.device) -> torch.Tensor:
    """Build [library_size, 16] LUT tensor for vectorized mapping.
    
    luts[lib_idx, src_code] = nearest code in library[lib_idx]
    """
    library_size = len(library)
    luts = torch.zeros(library_size, 16, dtype=torch.uint8, device=device)
    
    for lib_idx, cb_codes in enumerate(library):
        cb_vals = E2M1_TABLE[cb_codes].to(device)  # [k]
        for src_code in range(16):
            src_val = E2M1_TABLE[src_code].to(device)
            dists = (cb_vals - src_val).abs()
            nearest_cb_pos = dists.argmin().item()
            luts[lib_idx, src_code] = cb_codes[nearest_cb_pos]
    
    return luts


def apply_library_perblock_vectorized(
    weight_fp4: torch.Tensor,
    library: list[list[int]],
    device: torch.device,
) -> torch.Tensor:
    """Vectorized per-block codebook selection and mapping.
    
    For each block of 16 codes, selects the library codebook that minimizes MSE,
    then maps all codes in the block to the nearest code in that codebook.
    
    Args:
        weight_fp4: [M, K//2] packed uint8
        library: list of codebooks (each a list of code indices)
        device: torch device
    
    Returns:
        mapped_fp4: [M, K//2] packed uint8
    """
    codes = unpack_fp4_codes(weight_fp4)  # [M, K]
    M, K = codes.shape
    num_blocks_per_row = K // BLOCK_SIZE
    
    # Reshape to blocks: [M * num_blocks_per_row, BLOCK_SIZE]
    codes_blocked = codes.view(M * num_blocks_per_row, BLOCK_SIZE)  # [B, 16]
    num_blocks = codes_blocked.shape[0]
    
    # Convert codes to float values: [B, 16]
    code_vals = E2M1_TABLE.to(device)[codes_blocked.long()]  # [B, 16]
    
    # For each library codebook, compute MSE for each block
    library_size = len(library)
    
    # Build library value tensors: [library_size, k]
    lib_vals_list = []
    for cb_codes in library:
        lib_vals_list.append(E2M1_TABLE[cb_codes].to(device))
    
    # Compute MSE for each block × library combination
    # code_vals: [B, 16], lib_vals: [L, k]
    # For each block b and library l: MSE = mean over 16 elements of min_dist^2
    
    # Expand: code_vals [B, 16, 1], lib_vals [1, 1, L, k]
    # This is memory-intensive for large B, so process in chunks
    
    chunk_size = 1024  # Process 1024 blocks at a time
    best_lib_idx = torch.zeros(num_blocks, dtype=torch.long, device=device)
    
    for chunk_start in range(0, num_blocks, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_blocks)
        chunk_vals = code_vals[chunk_start:chunk_end]  # [C, 16]
        C = chunk_vals.shape[0]
        
        # For each library codebook, compute MSE
        mse_per_lib = torch.zeros(C, library_size, device=device)
        
        for lib_idx, lib_vals in enumerate(lib_vals_list):
            k = len(lib_vals)
            # chunk_vals: [C, 16], lib_vals: [k]
            # For each element, find nearest lib value
            # Expand: [C, 16, 1] vs [1, 1, k] → [C, 16, k]
            dists = (chunk_vals.unsqueeze(-1) - lib_vals.view(1, 1, k)).abs()
            min_dists = dists.min(dim=-1).values  # [C, 16]
            mse_per_lib[:, lib_idx] = (min_dists ** 2).mean(dim=-1)  # [C]
        
        best_lib_idx[chunk_start:chunk_end] = mse_per_lib.argmin(dim=-1)
    
    # Build library LUTs: [library_size, 16]
    luts = build_library_luts(library, device)  # [L, 16]
    
    # Apply per-block mapping using gather
    # best_lib_idx: [B], codes_blocked: [B, 16]
    # For each block b, apply luts[best_lib_idx[b], codes_blocked[b, :]]
    
    # Expand best_lib_idx: [B, 1, 1] → [B, 16, 16] for gather
    # luts: [L, 16], codes_blocked: [B, 16]
    
    # Gather LUT for each block: [B, 16]
    block_luts = luts[best_lib_idx]  # [B, 16] — LUT for each block
    
    # Apply LUT: mapped_codes[b, i] = block_luts[b, codes_blocked[b, i]]
    mapped_codes = torch.gather(block_luts, 1, codes_blocked.long()).to(torch.uint8)  # [B, 16]
    
    # Reshape back: [M, K]
    mapped_codes = mapped_codes.view(M, K)
    
    return repack_fp4_codes(mapped_codes)


# ── NVFP4 linear with fast per-block codebook ─────────────────────────────────

def nvfp4_linear_perblock_fast(
    input: torch.Tensor,
    weight: torch.Tensor,
    library: list[list[int]],
    device: torch.device,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """NVFP4 linear with fast vectorized per-block codebook selection."""
    input_2d, prefix_shape = flatten_for_linear(input)
    s_in2 = fp4_global_scale(input_2d).to(torch.float32)
    s_w2 = fp4_global_scale(weight).to(torch.float32)
    
    weight_fp4, weight_scale = torch.ops.trtllm.fp4_quantize(weight, s_w2, SCALING_VECTOR_SIZE, False)
    weight_fp4_mapped = apply_library_perblock_vectorized(weight_fp4, library, device)
    
    alpha = (1.0 / (s_in2 * s_w2)).to(torch.float32)
    out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        input_2d, weight_fp4_mapped, bias=bias,
        input_scale=s_in2, weight_scale=weight_scale, alpha=alpha,
    )
    return restore_linear_shape(out, prefix_shape)


# ── Build library from model data ────────────────────────────────────────────

def build_library_from_model(
    weight_map: dict,
    snapshot_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    k_per_block: int = 8,
    library_size: int = 16,
    num_sample_layers: int = 5,
    num_sample_experts: int = 20,
    num_sample_blocks: int = 200,
) -> list[list[int]]:
    """Build codebook library from model FP4 code distributions.
    
    Uses K-means clustering on per-block code frequency vectors.
    """
    from sklearn.cluster import KMeans
    
    log.info(f"Building library (k={k_per_block}, size={library_size})...")
    
    store = WeightStore(MODEL_ID, snapshot_dir, weight_map)
    
    # Collect per-block code frequency vectors
    all_freq_vectors = []
    
    layer_step = max(1, 40 // num_sample_layers)
    layer_indices = list(range(0, 40, layer_step))[:num_sample_layers]
    
    for layer_idx in layer_indices:
        gate_up_key = f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"
        if gate_up_key not in weight_map:
            continue
        
        raw = store.load_tensors([gate_up_key])
        gate_up = move_tensor(raw[gate_up_key], device, dtype)
        
        expert_step = max(1, gate_up.shape[0] // num_sample_experts)
        expert_indices = list(range(0, gate_up.shape[0], expert_step))[:num_sample_experts]
        
        for expert_idx in expert_indices:
            weight = gate_up[expert_idx]
            s_w2 = fp4_global_scale(weight).to(torch.float32)
            weight_fp4, _ = torch.ops.trtllm.fp4_quantize(weight, s_w2, SCALING_VECTOR_SIZE, False)
            codes = unpack_fp4_codes(weight_fp4)
            
            M, K = codes.shape
            num_blocks = M * (K // BLOCK_SIZE)
            codes_blocked = codes.view(num_blocks, BLOCK_SIZE)
            
            # Sample blocks
            sample_idx = torch.randperm(num_blocks)[:num_sample_blocks]
            sampled = codes_blocked[sample_idx]  # [S, 16]
            
            # Compute frequency vectors: [S, 16]
            for b in range(sampled.shape[0]):
                freq = torch.bincount(sampled[b].long(), minlength=16).float()
                freq = freq / freq.sum()
                all_freq_vectors.append(freq.cpu().numpy())
        
        del raw, gate_up
        torch.cuda.empty_cache()
    
    log.info(f"  Collected {len(all_freq_vectors)} block frequency vectors")
    
    if len(all_freq_vectors) < library_size:
        log.warning(f"  Not enough blocks, reducing library size to {len(all_freq_vectors)}")
        library_size = len(all_freq_vectors)
    
    # K-means clustering
    features = np.array(all_freq_vectors)
    kmeans = KMeans(n_clusters=library_size, n_init=10, random_state=42, max_iter=100)
    kmeans.fit(features)
    
    # Convert cluster centers to codebooks
    library = []
    for center in kmeans.cluster_centers_:
        # Select top k_per_block codes by frequency
        top_codes = np.argsort(center)[-k_per_block:].tolist()
        # Ensure symmetric: if code c is in, also include its negative (c+8 or c-8)
        library.append(sorted(top_codes))
    
    log.info(f"  Built library of {len(library)} codebooks")
    for i, cb in enumerate(library[:5]):  # Show first 5
        cb_vals = [E2M1_TABLE[c].item() for c in cb]
        log.info(f"    CB{i:2d}: codes={cb} values={[f'{v:.1f}' for v in cb_vals]}")
    
    return library


# ── MoE forward with fast per-block codebook ─────────────────────────────────

def moe_forward_perblock_fast(
    hidden_states: torch.Tensor,
    tensors: dict,
    config,
    mode: str,
    scope: str,
    library: Optional[list[list[int]]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
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
            gate_up = nvfp4_linear_perblock_fast(current_state, gate_up_proj[expert_idx], library, device)
        else:
            gate_up = exact_linear(current_state, gate_up_proj[expert_idx], mode=mode)

        gate, up = gate_up.chunk(2, dim=-1)

        if mode == "nvfp4" and library is not None:
            current_hidden = nvfp4_linear_perblock_fast(F.silu(gate) * up, down_proj[expert_idx], library, device)
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


# ── PPL evaluation ────────────────────────────────────────────────────────────

def evaluate_ppl_perblock_fast(
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
                moe_out = moe_forward_perblock_fast(
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

    # Load Phase 2 results to get NVFP4 baseline
    phase2_results_path = Path("/code/tensorrt_llm/scripts/nvfp4_compress/phase2_fixed_results/results.json")
    nvfp4_ppl = 6.7017  # Default from Phase 2 identity
    if phase2_results_path.exists():
        with open(phase2_results_path) as f:
            p2 = json.load(f)
        if "identity" in p2:
            nvfp4_ppl = p2["identity"]["ppl"]
            log.info(f"✓ NVFP4 baseline from Phase 2: {nvfp4_ppl:.4f}")

    snapshot_dir, config_raw, weight_map = load_root_config(MODEL_ID)
    config = build_text_config(config_raw)
    log.info(f"✓ Model: {MODEL_ID}, layers={config.num_hidden_layers}, experts={config.num_experts}")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot_dir))
    eval_ids, nsamples, seqlen = load_eval_data(tokenizer)
    log.info(f"✓ Eval data: {nsamples} chunks × {seqlen} tokens")

    output_dir = Path("/code/tensorrt_llm/scripts/nvfp4_compress/phase3_fast_results")
    output_dir.mkdir(exist_ok=True)

    results = {}

    # Benchmark vectorized mapping speed
    log.info("\nBenchmarking vectorized mapping speed...")
    w_test = torch.randn(256, 128, dtype=dtype, device=device)
    s_w2 = fp4_global_scale(w_test).to(torch.float32)
    wfp4, _ = torch.ops.trtllm.fp4_quantize(w_test, s_w2, SCALING_VECTOR_SIZE, False)
    test_library = [[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]]
    t0 = time.time()
    for _ in range(10):
        apply_library_perblock_vectorized(wfp4, test_library, device)
    elapsed = (time.time() - t0) / 10
    log.info(f"  Vectorized mapping: {elapsed*1000:.1f}ms per weight (256×128)")
    del w_test, wfp4

    # Experiment 3.1: Library-16 (8 codes/block)
    log.info("\n" + "="*60)
    log.info("Experiment 3.1: Library-16 (8 codes/block, 0.25 bits/elem overhead)")
    log.info("="*60)

    library_16 = build_library_from_model(
        weight_map, snapshot_dir, device, dtype,
        k_per_block=8, library_size=16,
        num_sample_layers=5, num_sample_experts=20, num_sample_blocks=200,
    )

    with open(output_dir / "library_16.json", "w") as f:
        json.dump({"library": library_16, "k_per_block": 8, "library_size": 16}, f, indent=2)

    t0 = time.time()
    ppl_lib16 = evaluate_ppl_perblock_fast(
        eval_ids, nsamples, seqlen, config, weight_map,
        snapshot_dir, device, dtype,
        label="library_16",
        library=library_16,
        layer_batch_size=4,
    )
    elapsed = time.time() - t0

    delta = ppl_lib16 - nvfp4_ppl
    log.info(f"✓ Library-16 PPL: {ppl_lib16:.4f} (Δ{delta:+.4f} vs NVFP4)")
    log.info(f"  Bits/elem: 3.25 | Time: {elapsed:.1f}s")

    results["library_16"] = {
        "ppl": ppl_lib16, "delta_vs_nvfp4": delta,
        "bits_per_elem": 3.25, "k_per_block": 8, "library_size": 16,
        "time_sec": elapsed,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Experiment 3.2: Library-256 (8 codes/block)
    log.info("\n" + "="*60)
    log.info("Experiment 3.2: Library-256 (8 codes/block, 0.5 bits/elem overhead)")
    log.info("="*60)

    library_256 = build_library_from_model(
        weight_map, snapshot_dir, device, dtype,
        k_per_block=8, library_size=256,
        num_sample_layers=5, num_sample_experts=20, num_sample_blocks=200,
    )

    with open(output_dir / "library_256.json", "w") as f:
        json.dump({"library": library_256, "k_per_block": 8, "library_size": 256}, f, indent=2)

    t0 = time.time()
    ppl_lib256 = evaluate_ppl_perblock_fast(
        eval_ids, nsamples, seqlen, config, weight_map,
        snapshot_dir, device, dtype,
        label="library_256",
        library=library_256,
        layer_batch_size=4,
    )
    elapsed = time.time() - t0

    delta = ppl_lib256 - nvfp4_ppl
    log.info(f"✓ Library-256 PPL: {ppl_lib256:.4f} (Δ{delta:+.4f} vs NVFP4)")
    log.info(f"  Bits/elem: 3.5 | Time: {elapsed:.1f}s")

    results["library_256"] = {
        "ppl": ppl_lib256, "delta_vs_nvfp4": delta,
        "bits_per_elem": 3.5, "k_per_block": 8, "library_size": 256,
        "time_sec": elapsed,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    log.info("\n" + "="*60)
    log.info("PHASE 3 SUMMARY")
    log.info("="*60)
    log.info(f"NVFP4 baseline: {nvfp4_ppl:.4f}")
    log.info(f"BF16 baseline:  {BF16_PPL:.4f}")
    for name, res in results.items():
        status = "✓ PASS" if res['delta_vs_nvfp4'] < 0.02 else "✗ FAIL"
        log.info(f"  {name:15s}: PPL={res['ppl']:.4f} (Δ{res['delta_vs_nvfp4']:+.4f}) | bits={res['bits_per_elem']:.2f} | {status}")

    log.info(f"\n✓ Results saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()

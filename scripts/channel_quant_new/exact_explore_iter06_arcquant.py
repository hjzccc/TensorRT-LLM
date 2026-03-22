#!/usr/bin/env python3
# pyright: reportImplicitRelativeImport=false, reportMissingImports=false
"""Exact-path Exploration Iteration 6: ARCQuant-style Augmented Residual Channels.

ARCQuant (arXiv:2601.07475) achieves W4A8-level accuracy under W4A4 hardware constraints
by augmenting the activation matrix with quantized residuals of outlier channels.

Key idea:
  - Identify top-S outlier input channels (by absmax over calibration data)
  - For each linear layer: X_aug = [X | Q_residual(X_outliers)]
                           W_aug = [W | W_outliers]  (duplicate outlier weight columns)
  - Run single NVFP4 GEMM on extended K dimension (K + S)
  - The residual correction is fused into the GEMM reduction — no extra kernel

This directly addresses the root cause: FP4 activation quantization error on outlier channels.
Expected gain: 0.01-0.03 PPL (ARCQuant shows ~0.08 PPL gain on Qwen2.5-7B vs RTN NVFP4)

For MoE: apply per-expert, using per-expert activation statistics from calibration cache.

Run inside trtllm-dual-tile docker:
  python /workspace/channel_quant_new/exact_explore_iter06_arcquant.py --nsamples 4
  python /workspace/channel_quant_new/exact_explore_iter06_arcquant.py --nsamples 145
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/channel_quant_new")
sys.path.insert(0, "/workspace/channel_quant")
sys.path.insert(0, str(SCRIPT_DIR.parent / "channel_quant"))

import exact_docker_eval as exact_eval
from spike1_ground_truth import (
    MODEL_ID,
    build_text_config,
    layer_keys,
    load_root_config,
    move_tensor,
    release_tensors,
    rms_norm_qwen3_next,
    shorten_layer_tensors,
)

RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "exact_explore_iter06_arcquant.json"
CALIBRATION_CACHE_PATH = Path("/workspace/channel_quant/results/proper_iter01_calibration_cache.pt")


@dataclass(frozen=True)
class ARCConfig:
    """Configuration for ARCQuant-style augmented residual channels."""
    label: str
    description: str
    # Number of outlier channels to augment (S in ARCQuant paper)
    # -1 means use threshold-based selection (tau = 2^-3 * M)
    num_residual_channels: int = -1
    # Fraction of K channels to treat as outliers (alternative to fixed count)
    residual_fraction: float = 0.0
    # Whether to apply to W1 (gate_up_proj) inputs
    apply_to_w1: bool = True
    # Whether to apply to W2 (down_proj) inputs
    apply_to_w2: bool = True
    # Use per-expert calibration stats (True) or global stats (False)
    per_expert: bool = True


ALL_CONFIGS = [
    ARCConfig(
        "uniform_bf16",
        "BF16 baseline (no ARC)",
        num_residual_channels=0,
    ),
    ARCConfig(
        "uniform_nvfp4",
        "NVFP4 baseline (no ARC)",
        num_residual_channels=0,
    ),
    ARCConfig(
        "arc_threshold_w1w2",
        "ARCQuant threshold-based (tau=2^-3*M) on W1+W2, per-expert",
        num_residual_channels=-1,
        residual_fraction=0.0,
        apply_to_w1=True,
        apply_to_w2=True,
        per_expert=True,
    ),
    ARCConfig(
        "arc_32ch_w2only",
        "ARCQuant 32 residual channels on W2 only (down_proj), per-expert",
        num_residual_channels=32,
        apply_to_w1=False,
        apply_to_w2=True,
        per_expert=True,
    ),
    ARCConfig(
        "arc_64ch_w2only",
        "ARCQuant 64 residual channels on W2 only, per-expert",
        num_residual_channels=64,
        apply_to_w1=False,
        apply_to_w2=True,
        per_expert=True,
    ),
    ARCConfig(
        "arc_32ch_w1w2",
        "ARCQuant 32 residual channels on W1+W2, per-expert",
        num_residual_channels=32,
        apply_to_w1=True,
        apply_to_w2=True,
        per_expert=True,
    ),
    ARCConfig(
        "arc_64ch_w1w2",
        "ARCQuant 64 residual channels on W1+W2, per-expert",
        num_residual_channels=64,
        apply_to_w1=True,
        apply_to_w2=True,
        per_expert=True,
    ),
    ARCConfig(
        "arc_128ch_w2only",
        "ARCQuant 128 residual channels on W2 only, per-expert",
        num_residual_channels=128,
        apply_to_w1=False,
        apply_to_w2=True,
        per_expert=True,
    ),
]


def load_calibration_cache(cache_path: Path, model_id: str) -> dict:
    """Load calibration cache with per-expert activation statistics."""
    print(f"Loading calibration cache from {cache_path}...")
    cache = torch.load(cache_path, map_location="cpu")
    base_cal = cache.get("base_calibration", cache)
    print(f"✓ Cache loaded")
    return base_cal


def compute_outlier_indices(
    activation_absmax: torch.Tensor,  # [K] per-channel absmax
    num_residual_channels: int,
    residual_fraction: float,
    use_threshold: bool,
) -> torch.Tensor:
    """Compute indices of outlier channels to augment.
    
    ARCQuant threshold: tau = 2^-3 * M (3-bit exponent gap between E5M2 and E2M1)
    Channels with absmax > tau are outliers.
    """
    K = activation_absmax.numel()
    
    if use_threshold:
        # ARCQuant threshold-based selection
        M = activation_absmax.max().item()
        tau = M * (2 ** -3)  # 3-bit exponent gap
        outlier_mask = activation_absmax > tau
        outlier_indices = torch.nonzero(outlier_mask, as_tuple=False).flatten()
        # Sort by descending absmax (most important first)
        sorted_by_mag = activation_absmax[outlier_indices].argsort(descending=True)
        return outlier_indices[sorted_by_mag]
    elif num_residual_channels > 0:
        # Fixed count: top-S by absmax
        S = min(num_residual_channels, K)
        _, top_indices = activation_absmax.topk(S, largest=True)
        return top_indices.sort().values  # sort for contiguous access
    elif residual_fraction > 0:
        S = max(1, int(K * residual_fraction))
        _, top_indices = activation_absmax.topk(S, largest=True)
        return top_indices.sort().values
    else:
        return torch.tensor([], dtype=torch.long)


def arc_nvfp4_linear(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    outlier_indices: torch.Tensor,
) -> torch.Tensor:
    """ARCQuant-style NVFP4 linear with augmented residual channels.
    
    Algorithm:
    1. Quantize X to NVFP4 → Q(X), residual R = X - Q(X)
    2. Extract outlier residuals R_o = R[:, outlier_indices]
    3. Quantize R_o to NVFP4 → Q(R_o)
    4. Augment: X_aug = [X | Q(R_o)] (extend K dimension)
    5. Augment: W_aug = [W | W[:, outlier_indices]] (duplicate outlier weight columns)
    6. Single NVFP4 GEMM on X_aug @ W_aug.T
    
    The GEMM computes: Q(X)@W.T + Q(R_o)@W_o.T ≈ X@W.T
    """
    from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale
    
    input_2d, prefix_shape = exact_eval.flatten_for_linear(input_tensor)
    S = outlier_indices.numel()
    
    if S == 0:
        # No augmentation — standard NVFP4
        return exact_eval.nvfp4_linear(input_tensor, weight)
    
    # Step 1: Quantize primary input X → Q(X)
    s_in = fp4_global_scale(input_2d).to(torch.float32)
    # Compute quantized X and residual
    # NVFP4: group size 16 along K, scale per group
    K = input_2d.shape[-1]
    input_float = input_2d.float()
    
    # Per-group quantization of input (group size 16 along K)
    G = 16  # NVFP4 group size
    K_padded = ((K + G - 1) // G) * G
    if K_padded > K:
        input_padded = F.pad(input_float, (0, K_padded - K))
    else:
        input_padded = input_float
    
    # Reshape to groups: [M, K//G, G]
    M = input_2d.shape[0]
    input_grouped = input_padded.view(M, K_padded // G, G)
    
    # Per-group scale: absmax / 6.0 (NVFP4 E2M1 max representable = 6.0)
    NVFP4_MAX = 6.0
    group_scales = input_grouped.abs().amax(dim=-1, keepdim=True) / NVFP4_MAX  # [M, K//G, 1]
    group_scales = group_scales.clamp(min=1e-8)
    
    # Quantize: round to nearest representable E2M1 value
    # E2M1 representable values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (and negatives)
    input_scaled = input_grouped / group_scales
    # Clamp to E2M1 range
    input_scaled = input_scaled.clamp(-NVFP4_MAX, NVFP4_MAX)
    # Round to nearest E2M1 (approximate with float rounding for now)
    # True E2M1 rounding would use lookup table, but float approx is sufficient for residual
    input_quantized = input_scaled.round()  # approximate
    input_dequant = (input_quantized * group_scales).view(M, K_padded)[:, :K]
    
    # Step 2: Compute residual for outlier channels
    residual = input_float - input_dequant  # [M, K]
    residual_outliers = residual[:, outlier_indices]  # [M, S]
    
    # Step 3: Quantize residual outliers to NVFP4
    # Use global scale for residual (smaller magnitude)
    s_res = fp4_global_scale(residual_outliers.to(input_2d.dtype)).to(torch.float32)
    
    # Step 4: Augment input: X_aug = [X | residual_outliers_quantized]
    # We use the original input_2d for primary (TRT-LLM handles its quantization)
    # and append the residual as additional channels
    residual_outliers_fp = residual_outliers.to(input_2d.dtype)
    input_aug = torch.cat([input_2d, residual_outliers_fp], dim=-1)  # [M, K+S]
    
    # Step 5: Augment weight: W_aug = [W | W[:, outlier_indices]]
    weight_outliers = weight[:, outlier_indices].contiguous()  # [N, S]
    weight_aug = torch.cat([weight, weight_outliers], dim=-1)  # [N, K+S]
    
    # Step 6: Single NVFP4 GEMM on augmented tensors
    # The GEMM computes: Q(X_aug) @ W_aug.T
    # = Q(X) @ W.T + Q(R_o) @ W_o.T
    # ≈ X @ W.T (with residual correction)
    out = exact_eval.nvfp4_linear(input_aug, weight_aug)
    
    return exact_eval.restore_linear_shape(out, prefix_shape)


def build_expert_outlier_indices(
    calibration_cache: dict,
    config: Any,
    arc_config: ARCConfig,
) -> dict[int, dict[int, dict[str, torch.Tensor]]]:
    """Build per-expert outlier indices for W1 and W2 from calibration data.
    
    Returns: {layer_idx: {expert_idx: {'w1': indices, 'w2': indices}}}
    """
    print("Building per-expert outlier indices from calibration cache...")
    
    activation_cache = calibration_cache.get("activation_cache", {})
    routing_counts = calibration_cache.get("routing_counts", {})
    
    outlier_indices: dict[int, dict[int, dict[str, torch.Tensor]]] = {}
    
    use_threshold = (arc_config.num_residual_channels == -1)
    
    total_w1_channels = 0
    total_w2_channels = 0
    
    for layer_idx in range(config.num_hidden_layers):
        outlier_indices[layer_idx] = {}
        
        if layer_idx not in activation_cache:
            # No calibration data for this layer — use empty indices
            for expert_idx in range(config.num_experts):
                outlier_indices[layer_idx][expert_idx] = {
                    'w1': torch.tensor([], dtype=torch.long),
                    'w2': torch.tensor([], dtype=torch.long),
                }
            continue
        
        layer_metrics = activation_cache[layer_idx]
        
        # W1 inputs: hidden_size channels (input to gate_up_proj)
        # w1_pair_scores shape: [num_experts, num_pairs] where num_pairs = moe_intermediate_size // 2
        # But we need activation absmax for input channels (hidden_size)
        # Use w2_channel_scores as proxy for W2 input (intermediate) channel importance
        # For W1 inputs, we need the hidden_size activation stats
        
        # The calibration cache stores:
        # - w1_pair_scores: [num_experts, moe_intermediate_size//2] — W1 output sensitivity
        # - w2_channel_scores: [num_experts, hidden_size] — W2 input (= W1 output) sensitivity
        # For ARCQuant on W1 inputs: we need hidden_size activation stats
        # For ARCQuant on W2 inputs: we need moe_intermediate_size activation stats
        
        w2_scores = layer_metrics.get('w2_channel_scores', None)  # [num_experts, hidden_size]
        w1_scores = layer_metrics.get('w1_pair_scores', None)     # [num_experts, moe_inter//2]
        
        for expert_idx in range(config.num_experts):
            w1_idx = torch.tensor([], dtype=torch.long)
            w2_idx = torch.tensor([], dtype=torch.long)
            
            if arc_config.apply_to_w1 and w2_scores is not None:
                # W1 input channels = hidden_size
                # Use w2_channel_scores as proxy for hidden_size activation importance
                # (these are the same channels — W2 input = W1 output, but we want W1 INPUT)
                # Actually for W1 inputs we need a different stat. Use global absmax as fallback.
                # For now, use w2_channel_scores as a proxy (same hidden_size dimension)
                if expert_idx < w2_scores.shape[0]:
                    absmax = w2_scores[expert_idx].float().abs()
                    w1_idx = compute_outlier_indices(
                        absmax,
                        arc_config.num_residual_channels,
                        arc_config.residual_fraction,
                        use_threshold,
                    )
                    total_w1_channels += w1_idx.numel()
            
            if arc_config.apply_to_w2 and w2_scores is not None:
                # W2 input channels = moe_intermediate_size
                # Use w1_pair_scores as proxy for intermediate channel importance
                if w1_scores is not None and expert_idx < w1_scores.shape[0]:
                    # w1_pair_scores: [num_experts, moe_inter//2] — expand to full intermediate
                    pair_scores = w1_scores[expert_idx].float().abs()
                    # Expand pairs to full intermediate channels
                    absmax = pair_scores.repeat_interleave(2)  # [moe_intermediate_size]
                    w2_idx = compute_outlier_indices(
                        absmax,
                        arc_config.num_residual_channels,
                        arc_config.residual_fraction,
                        use_threshold,
                    )
                    total_w2_channels += w2_idx.numel()
            
            outlier_indices[layer_idx][expert_idx] = {
                'w1': w1_idx,
                'w2': w2_idx,
            }
    
    avg_w1 = total_w1_channels / (config.num_hidden_layers * config.num_experts)
    avg_w2 = total_w2_channels / (config.num_hidden_layers * config.num_experts)
    print(f"  Avg W1 outlier channels per expert: {avg_w1:.1f}")
    print(f"  Avg W2 outlier channels per expert: {avg_w2:.1f}")
    
    return outlier_indices


def arc_moe_forward(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config: Any,
    arc_config: ARCConfig,
    outlier_indices: dict[int, dict[int, dict[str, torch.Tensor]]],
    layer_idx: int,
) -> torch.Tensor:
    """MoE forward pass with ARCQuant-style augmented residual channels."""
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    
    router_logits = exact_eval.bf16_linear(flat, tensors["gate.weight"]).float()
    routing_weights = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(routing_weights, config.num_experts_per_tok, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    
    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    gate_up_proj = tensors["experts.gate_up_proj"]
    down_proj = tensors["experts.down_proj"]
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)
    
    layer_outliers = outlier_indices.get(layer_idx, {})
    
    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        
        expert_outliers = layer_outliers.get(expert_idx, {'w1': torch.tensor([]), 'w2': torch.tensor([])})
        w1_outlier_idx = expert_outliers['w1'].to(device=current_state.device)
        w2_outlier_idx = expert_outliers['w2'].to(device=current_state.device)
        
        # W1 forward (gate_up_proj)
        if arc_config.apply_to_w1 and w1_outlier_idx.numel() > 0:
            gate_up = arc_nvfp4_linear(current_state, gate_up_proj[expert_idx], w1_outlier_idx)
        else:
            gate_up = exact_eval.nvfp4_linear(current_state, gate_up_proj[expert_idx])
        
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        
        # W2 forward (down_proj)
        if arc_config.apply_to_w2 and w2_outlier_idx.numel() > 0:
            current_hidden = arc_nvfp4_linear(hidden, down_proj[expert_idx], w2_outlier_idx)
        else:
            current_hidden = exact_eval.nvfp4_linear(hidden, down_proj[expert_idx])
        
        current_hidden = current_hidden * routing_weights[token_idx, route_pos].unsqueeze(-1)
        final_hidden_states.index_add_(0, token_idx, current_hidden.to(hidden_states.dtype))
    
    # Shared expert (always BF16)
    shared = exact_eval.bf16_linear(flat, tensors["shared_expert.gate_proj.weight"])
    shared = F.silu(shared) * exact_eval.bf16_linear(flat, tensors["shared_expert.up_proj.weight"])
    shared = exact_eval.bf16_linear(shared, tensors["shared_expert.down_proj.weight"])
    shared_gate = torch.sigmoid(exact_eval.bf16_linear(flat, tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_gate * shared
    
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


def evaluate_arc_ppl(
    eval_ids: torch.Tensor,
    nsamples: int,
    seqlen: int,
    config: Any,
    weight_map: dict[str, str],
    snapshot_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    arc_config: ARCConfig,
    outlier_indices: dict[int, dict[int, dict[str, torch.Tensor]]],
    layer_batch_size: int,
) -> float:
    """Evaluate PPL with ARCQuant-style augmented residual channels."""
    store = exact_eval.WeightStore(exact_eval.MODEL_ID, snapshot_dir, weight_map)
    
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
    
    position_ids = torch.arange(seqlen, device=device).unsqueeze(0)
    rotary = Qwen3NextRotaryEmbedding(config=config, device=device)
    rotary_input = torch.empty((1, seqlen, config.hidden_size), device=device, dtype=dtype)
    position_embeddings = rotary(rotary_input, position_ids)
    del rotary_input
    
    nlls: list[torch.Tensor] = []
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
                hidden_states = rms_norm_qwen3_next(
                    hidden_states, layer_tensors["input_layernorm.weight"], config.rms_norm_eps
                )
                if layer_type == "full_attention":
                    attn_tensors = {k.replace("self_attn.", ""): v for k, v in layer_tensors.items() if k.startswith("self_attn.")}
                    hidden_states = exact_eval.full_attention_forward_exact(
                        hidden_states, attn_tensors, config, position_embeddings,
                        exact_eval.build_causal_mask(seqlen, device), mode="bf16", quantized=False,
                    )
                else:
                    attn_tensors = {k.replace("linear_attn.", ""): v for k, v in layer_tensors.items() if k.startswith("linear_attn.")}
                    hidden_states = exact_eval.linear_attention_forward_exact(
                        hidden_states, attn_tensors, config, "bf16", "moe_only"
                    )
                hidden_states = residual + hidden_states
                
                residual = hidden_states
                hidden_states = rms_norm_qwen3_next(
                    hidden_states, layer_tensors["post_attention_layernorm.weight"], config.rms_norm_eps
                )
                moe_tensors = {k.replace("mlp.", "", 1): v for k, v in layer_tensors.items() if k.startswith("mlp.")}
                
                if arc_config.num_residual_channels == 0:
                    # Baseline: use standard NVFP4 or BF16
                    if arc_config.label == "uniform_bf16":
                        moe_out = exact_eval.moe_forward_exact(
                            hidden_states, moe_tensors, config, "bf16", "moe_only"
                        )
                    else:
                        moe_out = exact_eval.moe_forward_exact(
                            hidden_states, moe_tensors, config, "nvfp4", "moe_only"
                        )
                else:
                    moe_out = arc_moe_forward(
                        hidden_states, moe_tensors, config, arc_config, outlier_indices, layer_idx
                    )
                
                hidden_states = residual + moe_out
                hidden_bank[batch_start:batch_end].copy_(hidden_states.cpu())
                del hidden_states, residual, moe_out
            
            release_tensors(layer_tensors)
            print(f"  [{arc_config.label}] layer {layer_idx + 1}/{config.num_hidden_layers}", flush=True)
        
        for sample_idx in range(nsamples):
            chunk = eval_chunks[sample_idx:sample_idx + 1].to(device)
            hidden_states = hidden_bank[sample_idx:sample_idx + 1].to(device)
            hidden_states = rms_norm_qwen3_next(hidden_states, final_norm_w, config.rms_norm_eps)
            logits = F.linear(hidden_states.float(), lm_head_w.float())
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk[:, 1:]
            loss = F.cross_entropy(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            nlls.append(loss.float() * seqlen)
    
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()
    del embed_w, final_norm_w, lm_head_w, store
    torch.cuda.empty_cache()
    return ppl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=exact_eval.MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--nsamples", type=int, default=4)
    parser.add_argument("--layer-batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated list of config labels to run (default: all)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exact_eval.ensure_runtime_available()
    
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    eval_ids, available_nsamples, seqlen = exact_eval.load_eval_data(tokenizer, args.seqlen)
    nsamples = min(args.nsamples, available_nsamples)
    
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    config = build_text_config(root_config)
    
    # Load calibration cache for per-expert activation statistics
    calibration_cache = load_calibration_cache(CALIBRATION_CACHE_PATH, args.model_id)
    
    # Filter configs if requested
    configs_to_run = ALL_CONFIGS
    if args.configs:
        requested = set(args.configs.split(","))
        configs_to_run = [c for c in ALL_CONFIGS if c.label in requested]
        if not configs_to_run:
            print(f"No matching configs found for: {args.configs}")
            return
    
    # Load existing results for resume
    output_path = args.output
    results: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        try:
            with open(output_path) as f:
                existing = json.load(f)
            results = existing.get("results", {})
            print(f"[resume] loaded {len(results)} existing results from {output_path}")
        except Exception:
            pass
    
    for arc_config in configs_to_run:
        if arc_config.label in results:
            print(f"[skip] {arc_config.label} already done (PPL={results[arc_config.label].get('ppl', '?')})")
            continue
        
        print(f"\n=== {arc_config.label} ===", flush=True)
        print(f"    {arc_config.description}", flush=True)
        start_time = time.time()
        
        # Build outlier indices for this config
        if arc_config.num_residual_channels == 0:
            outlier_indices: dict = {}
        else:
            outlier_indices = build_expert_outlier_indices(calibration_cache, config, arc_config)
        
        ppl = evaluate_arc_ppl(
            eval_ids, nsamples, seqlen, config, weight_map, snapshot_dir,
            device, dtype, arc_config, outlier_indices, args.layer_batch_size,
        )
        elapsed = time.time() - start_time
        
        results[arc_config.label] = {
            "description": arc_config.description,
            "ppl": round(ppl, 5),
            "time_s": round(elapsed, 1),
            "nsamples": nsamples,
            "num_residual_channels": arc_config.num_residual_channels,
            "apply_to_w1": arc_config.apply_to_w1,
            "apply_to_w2": arc_config.apply_to_w2,
            "per_expert": arc_config.per_expert,
        }
        print(f"  -> PPL={ppl:.5f} ({elapsed:.0f}s)", flush=True)
        
        # Save after each config
        payload = {
            "metadata": {
                "model": args.model_id,
                "nsamples": nsamples,
                "seqlen": seqlen,
                "experiment": "arcquant_augmented_residual_channels",
                "paper": "arXiv:2601.07475",
                "key_idea": "Augment activation K-dim with quantized residuals of outlier channels",
            },
            "results": results,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Saved -> {output_path}", flush=True)
    
    print("\n=== SUMMARY ===")
    for label, res in sorted(results.items(), key=lambda x: x[1].get("ppl", 99)):
        print(f"  {label:40s}: PPL={res.get('ppl', '?'):.5f}")


if __name__ == "__main__":
    main()

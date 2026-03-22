#!/usr/bin/env python3
# pyright: reportImplicitRelativeImport=false, reportMissingImports=false
"""Exact-path Exploration Iteration 7: Input Channel Mixed Precision.

Key insight from iter01 + ARCQuant analysis:
- Our existing mixed_exact_linear splits OUTPUT channels (rows of W) into FP4/FP8
- But the FP4 ACTIVATION error is the root cause of degradation
- Solution: split INPUT channels (columns of W) into FP4/FP8
  - Outlier input channels → FP8 (both activation and weight columns)
  - Normal input channels → NVFP4

This is different from ARCQuant (which augments K-dim with residuals).
This is simpler: two separate GEMMs on disjoint input channel subsets, then sum.

Y = X @ W.T
  = X[:, fp8_cols] @ W[:, fp8_cols].T  (FP8 GEMM)
  + X[:, fp4_cols] @ W[:, fp4_cols].T  (NVFP4 GEMM)

The FP8 GEMM handles outlier activations with 8-bit precision.
The NVFP4 GEMM handles normal activations with 4-bit precision.

Expected gain: 0.01-0.03 PPL (similar to ARCQuant but simpler implementation)
Memory overhead: none (same weights, just different routing)
Runtime overhead: two GEMMs instead of one (but smaller K each)

Run inside trtllm-dual-tile docker:
  python /workspace/channel_quant_new/exact_explore_iter07_input_channel_mix.py --nsamples 4
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
    build_text_config,
    layer_keys,
    load_root_config,
    move_tensor,
    release_tensors,
    rms_norm_qwen3_next,
    shorten_layer_tensors,
)

RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "exact_explore_iter07_input_channel_mix.json"
CALIBRATION_CACHE_PATH = Path("/workspace/channel_quant/results/proper_iter01_calibration_cache.pt")
METRIC_CACHE_PATH = Path("/workspace/channel_quant/results/proper_iter10_novel_perchannel_metric_cache.pt")


@dataclass(frozen=True)
class InputMixConfig:
    """Configuration for input channel mixed precision."""
    label: str
    description: str
    # Fraction of input channels to keep in FP8 (by absmax)
    fp8_input_fraction: float = 0.0
    # Fixed count of FP8 input channels (-1 = use fraction)
    fp8_input_count: int = -1
    # Apply to W1 (gate_up_proj) inputs
    apply_to_w1: bool = True
    # Apply to W2 (down_proj) inputs
    apply_to_w2: bool = True
    # Use per-expert stats (True) or global stats (False)
    per_expert: bool = True


ALL_CONFIGS = [
    InputMixConfig(
        "uniform_bf16",
        "BF16 baseline",
        fp8_input_fraction=0.0,
    ),
    InputMixConfig(
        "uniform_nvfp4",
        "NVFP4 baseline (no input mixing)",
        fp8_input_fraction=0.0,
    ),
    InputMixConfig(
        "input_mix_w2_32ch",
        "32 FP8 input channels on W2 (down_proj), per-expert",
        fp8_input_count=32,
        apply_to_w1=False,
        apply_to_w2=True,
        per_expert=True,
    ),
    InputMixConfig(
        "input_mix_w2_64ch",
        "64 FP8 input channels on W2, per-expert",
        fp8_input_count=64,
        apply_to_w1=False,
        apply_to_w2=True,
        per_expert=True,
    ),
    InputMixConfig(
        "input_mix_w2_128ch",
        "128 FP8 input channels on W2, per-expert",
        fp8_input_count=128,
        apply_to_w1=False,
        apply_to_w2=True,
        per_expert=True,
    ),
    InputMixConfig(
        "input_mix_w1w2_32ch",
        "32 FP8 input channels on W1+W2, per-expert",
        fp8_input_count=32,
        apply_to_w1=True,
        apply_to_w2=True,
        per_expert=True,
    ),
    InputMixConfig(
        "input_mix_w1w2_64ch",
        "64 FP8 input channels on W1+W2, per-expert",
        fp8_input_count=64,
        apply_to_w1=True,
        apply_to_w2=True,
        per_expert=True,
    ),
    InputMixConfig(
        "input_mix_w2_10pct",
        "10% FP8 input channels on W2 (51 channels), per-expert",
        fp8_input_fraction=0.10,
        apply_to_w1=False,
        apply_to_w2=True,
        per_expert=True,
    ),
    InputMixConfig(
        "input_mix_w2_20pct",
        "20% FP8 input channels on W2 (102 channels), per-expert",
        fp8_input_fraction=0.20,
        apply_to_w1=False,
        apply_to_w2=True,
        per_expert=True,
    ),
]


def load_activation_stats(
    calibration_cache_path: Path,
    metric_cache_path: Path,
    model_id: str,
    config: Any,
) -> dict[int, dict[int, dict[str, torch.Tensor]]]:
    """Load per-expert activation absmax stats for input channel selection.
    
    Returns: {layer_idx: {expert_idx: {'w1_input_absmax': [hidden_size], 'w2_input_absmax': [moe_inter]}}}
    """
    print("Loading activation stats for input channel selection...")
    
    # Load calibration cache
    cache = torch.load(calibration_cache_path, map_location="cpu")
    base_cal = cache.get("base_calibration", cache)
    activation_cache = base_cal.get("activation_cache", {})
    
    # Load metric cache for router-affinity weighted scores
    try:
        mc = torch.load(metric_cache_path, map_location="cpu")
        rac = mc.get("router_affinity_cache", {})
    except Exception:
        rac = {}
    
    stats: dict[int, dict[int, dict[str, torch.Tensor]]] = {}
    
    for layer_idx in range(config.num_hidden_layers):
        stats[layer_idx] = {}
        
        # Use router-affinity cache if available, else fall back to activation cache
        layer_metrics = rac.get(layer_idx, activation_cache.get(layer_idx, {}))
        
        for expert_idx in range(config.num_experts):
            # w2_channel_scores: [num_experts, hidden_size] — proxy for W1 input importance
            # w1_pair_scores: [num_experts, moe_intermediate_size] — proxy for W2 input importance
            w2_scores = layer_metrics.get("w2_channel_scores", None)
            w1_scores = layer_metrics.get("w1_pair_scores", None)
            
            w1_input_absmax = torch.zeros(config.hidden_size)
            w2_input_absmax = torch.zeros(config.moe_intermediate_size)
            
            if w2_scores is not None and expert_idx < w2_scores.shape[0]:
                w1_input_absmax = w2_scores[expert_idx].float().abs()
            
            if w1_scores is not None and expert_idx < w1_scores.shape[0]:
                w2_input_absmax = w1_scores[expert_idx].float().abs()
            
            stats[layer_idx][expert_idx] = {
                "w1_input_absmax": w1_input_absmax,
                "w2_input_absmax": w2_input_absmax,
            }
    
    print(f"✓ Loaded stats for {config.num_hidden_layers} layers × {config.num_experts} experts")
    return stats


def get_fp8_input_indices(
    absmax: torch.Tensor,
    fp8_count: int,
    fp8_fraction: float,
) -> torch.Tensor:
    """Get indices of top-K input channels to keep in FP8."""
    K = absmax.numel()
    if fp8_count > 0:
        S = min(fp8_count, K)
    elif fp8_fraction > 0:
        S = max(1, int(K * fp8_fraction))
    else:
        return torch.tensor([], dtype=torch.long)
    
    _, top_indices = absmax.topk(S, largest=True)
    return top_indices.sort().values  # sort for contiguous memory access


def input_mixed_linear(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    fp8_input_indices: torch.Tensor,
) -> torch.Tensor:
    """Mixed-precision linear with input channel splitting.
    
    Y = X[:, fp8_cols] @ W[:, fp8_cols].T  (FP8)
      + X[:, fp4_cols] @ W[:, fp4_cols].T  (NVFP4)
    """
    S = fp8_input_indices.numel()
    K = weight.shape[1]
    
    if S == 0:
        return exact_eval.nvfp4_linear(input_tensor, weight)
    if S >= K:
        return exact_eval.fp8_linear(input_tensor, weight)
    
    input_2d, prefix_shape = exact_eval.flatten_for_linear(input_tensor)
    
    # Create FP4 indices (complement of FP8 indices)
    all_indices = torch.arange(K, device=weight.device)
    fp8_mask = torch.zeros(K, dtype=torch.bool, device=weight.device)
    fp8_mask[fp8_input_indices.to(weight.device)] = True
    fp4_indices = all_indices[~fp8_mask]
    
    # Snap FP4 count to multiple of 16 (NVFP4 group size requirement)
    fp4_count = fp4_indices.numel()
    if fp4_count % 16 != 0:
        # Move some channels from FP4 to FP8 to satisfy alignment
        deficit = fp4_count % 16
        # Move the least important FP4 channels to FP8
        fp8_input_indices_dev = fp8_input_indices.to(weight.device)
        moved = fp4_indices[-deficit:]
        fp8_input_indices_dev = torch.cat([fp8_input_indices_dev, moved])
        fp4_indices = fp4_indices[:-deficit]
    else:
        fp8_input_indices_dev = fp8_input_indices.to(weight.device)
    
    # Extract sub-matrices
    input_fp8 = input_2d[:, fp8_input_indices_dev].contiguous()
    weight_fp8 = weight[:, fp8_input_indices_dev].contiguous()
    
    input_fp4 = input_2d[:, fp4_indices].contiguous()
    weight_fp4 = weight[:, fp4_indices].contiguous()
    
    # Two GEMMs
    out = torch.zeros(
        (input_2d.shape[0], weight.shape[0]),
        dtype=input_2d.dtype,
        device=input_2d.device,
    )
    
    if fp8_input_indices_dev.numel() > 0:
        out += exact_eval.fp8_linear(input_fp8, weight_fp8)
    
    if fp4_indices.numel() > 0:
        out += exact_eval.nvfp4_linear(input_fp4, weight_fp4)
    
    return exact_eval.restore_linear_shape(out, prefix_shape)


def input_mix_moe_forward(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config: Any,
    mix_config: InputMixConfig,
    activation_stats: dict[int, dict[int, dict[str, torch.Tensor]]],
    layer_idx: int,
) -> torch.Tensor:
    """MoE forward with input channel mixed precision."""
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
    
    layer_stats = activation_stats.get(layer_idx, {})
    
    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        
        expert_stats = layer_stats.get(expert_idx, {})
        
        # W1 forward (gate_up_proj): input channels = hidden_size
        if mix_config.apply_to_w1 and (mix_config.fp8_input_count > 0 or mix_config.fp8_input_fraction > 0):
            w1_absmax = expert_stats.get("w1_input_absmax", torch.zeros(config.hidden_size))
            w1_fp8_idx = get_fp8_input_indices(
                w1_absmax, mix_config.fp8_input_count, mix_config.fp8_input_fraction
            )
            gate_up = input_mixed_linear(current_state, gate_up_proj[expert_idx], w1_fp8_idx)
        else:
            gate_up = exact_eval.nvfp4_linear(current_state, gate_up_proj[expert_idx])
        
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        
        # W2 forward (down_proj): input channels = moe_intermediate_size
        if mix_config.apply_to_w2 and (mix_config.fp8_input_count > 0 or mix_config.fp8_input_fraction > 0):
            w2_absmax = expert_stats.get("w2_input_absmax", torch.zeros(config.moe_intermediate_size))
            w2_fp8_idx = get_fp8_input_indices(
                w2_absmax, mix_config.fp8_input_count, mix_config.fp8_input_fraction
            )
            current_hidden = input_mixed_linear(hidden, down_proj[expert_idx], w2_fp8_idx)
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


def evaluate_input_mix_ppl(
    eval_ids: torch.Tensor,
    nsamples: int,
    seqlen: int,
    config: Any,
    weight_map: dict[str, str],
    snapshot_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    mix_config: InputMixConfig,
    activation_stats: dict[int, dict[int, dict[str, torch.Tensor]]],
    layer_batch_size: int,
) -> float:
    """Evaluate PPL with input channel mixed precision."""
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
                
                if mix_config.fp8_input_count == 0 and mix_config.fp8_input_fraction == 0.0:
                    # Baseline
                    if mix_config.label == "uniform_bf16":
                        moe_out = exact_eval.moe_forward_exact(hidden_states, moe_tensors, config, "bf16", "moe_only")
                    else:
                        moe_out = exact_eval.moe_forward_exact(hidden_states, moe_tensors, config, "nvfp4", "moe_only")
                else:
                    moe_out = input_mix_moe_forward(
                        hidden_states, moe_tensors, config, mix_config, activation_stats, layer_idx
                    )
                
                hidden_states = residual + moe_out
                hidden_bank[batch_start:batch_end].copy_(hidden_states.cpu())
                del hidden_states, residual, moe_out
            
            release_tensors(layer_tensors)
            print(f"  [{mix_config.label}] layer {layer_idx + 1}/{config.num_hidden_layers}", flush=True)
        
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
    parser.add_argument("--configs", type=str, default=None)
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
    
    # Load activation stats
    activation_stats = load_activation_stats(
        CALIBRATION_CACHE_PATH, METRIC_CACHE_PATH, args.model_id, config
    )
    
    # Filter configs
    configs_to_run = ALL_CONFIGS
    if args.configs:
        requested = set(args.configs.split(","))
        configs_to_run = [c for c in ALL_CONFIGS if c.label in requested]
    
    # Load existing results
    output_path = args.output
    results: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        try:
            with open(output_path) as f:
                existing = json.load(f)
            results = existing.get("results", {})
            print(f"[resume] loaded {len(results)} existing results")
        except Exception:
            pass
    
    for mix_config in configs_to_run:
        if mix_config.label in results:
            print(f"[skip] {mix_config.label} (PPL={results[mix_config.label].get('ppl', '?')})")
            continue
        
        print(f"\n=== {mix_config.label} ===", flush=True)
        print(f"    {mix_config.description}", flush=True)
        start_time = time.time()
        
        ppl = evaluate_input_mix_ppl(
            eval_ids, nsamples, seqlen, config, weight_map, snapshot_dir,
            device, dtype, mix_config, activation_stats, args.layer_batch_size,
        )
        elapsed = time.time() - start_time
        
        results[mix_config.label] = {
            "description": mix_config.description,
            "ppl": round(ppl, 5),
            "time_s": round(elapsed, 1),
            "nsamples": nsamples,
            "fp8_input_count": mix_config.fp8_input_count,
            "fp8_input_fraction": mix_config.fp8_input_fraction,
            "apply_to_w1": mix_config.apply_to_w1,
            "apply_to_w2": mix_config.apply_to_w2,
        }
        print(f"  -> PPL={ppl:.5f} ({elapsed:.0f}s)", flush=True)
        
        payload = {
            "metadata": {
                "model": args.model_id,
                "nsamples": nsamples,
                "seqlen": seqlen,
                "experiment": "input_channel_mixed_precision",
                "key_idea": "Split input channels into FP8 (outliers) + NVFP4 (normal), sum partial GEMMs",
            },
            "results": results,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Saved -> {output_path}", flush=True)
    
    print("\n=== SUMMARY ===")
    for label, res in sorted(results.items(), key=lambda x: x[1].get("ppl", 99)):
        print(f"  {label:45s}: PPL={res.get('ppl', '?'):.5f}")


if __name__ == "__main__":
    main()

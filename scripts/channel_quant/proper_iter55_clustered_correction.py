#!/usr/bin/env python3
# pyright: basic, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

"""
Iteration 55: Clustered Affine Correction (CAT-style)

Foundation: MaCa Uniform 4K (Iter29 best: 6.567582 PPL)
Innovation: Cluster-specific affine correction instead of plain per-channel

Motivation from literature:
- CAT (arXiv:2509.26277, ICLR 2026): "plain affine transformation worsens results
  in low-bit PTQ; cluster-specific parameters consistently improve them"
- KBVQ-MoE BCOS (arXiv:2602.11184, ICLR 2026): expert-specific channel-wise
  affine correction after VQ, solved analytically

Key insight: Per-channel correction (Iter25) gave 6.587 PPL (WORSE than baseline 6.572).
This is because per-channel overfits to calibration noise. Clustering channels by their
correction parameters (K-means on (alpha_c, beta_c)) reduces overfitting by sharing
parameters across similar channels.

Method:
1. Run MaCa calibration (same as Iter29)
2. Collect per-channel moments during teacher/quant forward pass
3. Solve per-channel (alpha_c, beta_c) for each channel of each expert
4. Cluster channels by (alpha_c, beta_c) using K-means with K=4, 8, 16
5. Average within clusters -> cluster-specific (alpha_k, beta_k)
6. Evaluate with cluster-specific correction

Expected: 6.562-6.567 PPL (beats Iter29's 6.567582 by 0.001-0.005)
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np

from proper_eval import (
    SEQLEN,
    atomic_json_dump,
    build_position_context,
    dtype_from_name,
    embed_chunks,
    finalize_layer_metrics,
    init_expert_moment_state,
    layer_type_at,
    load_gptq_standard_data,
    upsert_exploration_section,
)
from proper_iter01 import build_plan_from_masks
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from proper_iter11_push_router_affinity import evaluate_and_record_plan
from proper_iter26_maca_calibration import (
    MACA_PAD_LENGTH,
    build_maca_calibration_set,
    run_maca_calibration,
    WeightStore,
    layer_keys,
    shorten_layer_tensors,
    release_tensors,
    should_log_chunk,
)
from proper_iter28_maca_plus_correction import (
    forward_attention_to_mlp_input,
    prepare_quantized_layer_tensors,
    AffineCorrection,
    ScalarFitMoments,
    init_scalar_fit_moments,
    update_scalar_fit_moments,
    solve_scalar_affine_from_moments,
    forward_attention_to_mlp_input,
    prepare_quantized_layer_tensors,
    moe_forward_with_correction_valid_tokens,
    evaluate_and_record_compound_plan,
    BaseVariant,
    EvalPlan,
    build_eval_plan,
    parameter_count_for_layers,
)
from baselines_comparison import (
    quantize_linear_weight,
    resolve_non_expert_bytes,
    resolve_terminal_keys,
)
from spike1_ground_truth import load_root_config, build_text_config

SCRIPT_DIR = Path(__file__).parent
SECTION_MARKER = "## Iteration 55: Clustered Affine Correction"

# Configuration
CALIB_CHUNKS = 128
CALIB_LENGTH = 4096
TOPUP_FRACTION = JOINT_MEDIUM_TOPUP_FRACTION  # 0.05

# Clustering parameters to sweep
CLUSTER_COUNTS = [4, 8, 16]  # K values for K-means


@dataclass
class PerChannelFitMoments:
    """Extended moments for per-channel fitting."""
    # Scalar moments (per expert)
    scalar_count: torch.Tensor
    scalar_x_sum: torch.Tensor
    scalar_y_sum: torch.Tensor
    scalar_x2_sum: torch.Tensor
    scalar_xy_sum: torch.Tensor
    # Per-channel moments (per expert, per channel)
    channel_count: torch.Tensor  # (num_experts,)
    channel_x_sum: torch.Tensor  # (num_experts, hidden_size)
    channel_y_sum: torch.Tensor  # (num_experts, hidden_size)
    channel_x2_sum: torch.Tensor  # (num_experts, hidden_size)
    channel_xy_sum: torch.Tensor  # (num_experts, hidden_size)
    routed_pairs: torch.Tensor
    layer_sq_error: torch.Tensor
    layer_elem_count: int


def init_perchannel_fit_moments(config: Any, device: torch.device) -> PerChannelFitMoments:
    num_experts = int(config.num_experts)
    hidden_size = int(config.hidden_size)
    return PerChannelFitMoments(
        scalar_count=torch.zeros(num_experts, dtype=torch.float64, device=device),
        scalar_x_sum=torch.zeros(num_experts, dtype=torch.float64, device=device),
        scalar_y_sum=torch.zeros(num_experts, dtype=torch.float64, device=device),
        scalar_x2_sum=torch.zeros(num_experts, dtype=torch.float64, device=device),
        scalar_xy_sum=torch.zeros(num_experts, dtype=torch.float64, device=device),
        channel_count=torch.zeros(num_experts, dtype=torch.float64, device=device),
        channel_x_sum=torch.zeros((num_experts, hidden_size), dtype=torch.float64, device=device),
        channel_y_sum=torch.zeros((num_experts, hidden_size), dtype=torch.float64, device=device),
        channel_x2_sum=torch.zeros((num_experts, hidden_size), dtype=torch.float64, device=device),
        channel_xy_sum=torch.zeros((num_experts, hidden_size), dtype=torch.float64, device=device),
        routed_pairs=torch.zeros(num_experts, dtype=torch.int64, device=device),
        layer_sq_error=torch.zeros((), dtype=torch.float64, device=device),
        layer_elem_count=0,
    )


def update_perchannel_fit_moments(
    stats: PerChannelFitMoments,
    expert_idx: int,
    quant_out: torch.Tensor,
    teacher_out: torch.Tensor,
) -> None:
    if quant_out.numel() == 0:
        return
    stats.routed_pairs[expert_idx] += int(quant_out.shape[0])

    quant_f = quant_out.to(device=stats.scalar_x_sum.device, dtype=torch.float64)
    teacher_f = teacher_out.to(device=stats.scalar_y_sum.device, dtype=torch.float64)

    # Scalar moments
    scalar_count = float(quant_out.numel())
    stats.scalar_count[expert_idx] += scalar_count
    stats.scalar_x_sum[expert_idx] += quant_f.sum()
    stats.scalar_y_sum[expert_idx] += teacher_f.sum()
    stats.scalar_x2_sum[expert_idx] += quant_f.square().sum()
    stats.scalar_xy_sum[expert_idx] += (quant_f * teacher_f).sum()

    # Per-channel moments
    token_count = float(quant_out.shape[0])
    stats.channel_count[expert_idx] += token_count
    stats.channel_x_sum[expert_idx].add_(quant_f.sum(dim=0))
    stats.channel_y_sum[expert_idx].add_(teacher_f.sum(dim=0))
    stats.channel_x2_sum[expert_idx].add_(quant_f.square().sum(dim=0))
    stats.channel_xy_sum[expert_idx].add_((quant_f * teacher_f).sum(dim=0))


def solve_perchannel_affine(stats: PerChannelFitMoments) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve per-channel (alpha_c, beta_c) for each expert.
    
    Returns:
        alpha: (num_experts, hidden_size) - per-channel scale
        beta: (num_experts, hidden_size) - per-channel bias
    """
    count = stats.channel_count.unsqueeze(-1).clamp(min=1.0)  # (E, 1)
    mean_x = stats.channel_x_sum / count  # (E, H)
    mean_y = stats.channel_y_sum / count  # (E, H)
    var_x = stats.channel_x2_sum - (stats.channel_x_sum.square() / count)  # (E, H)
    cov_xy = stats.channel_xy_sum - ((stats.channel_x_sum * stats.channel_y_sum) / count)  # (E, H)

    alpha = torch.ones_like(mean_x, dtype=torch.float32)
    valid = stats.channel_count.unsqueeze(-1) > 0  # (E, 1) broadcast
    stable = torch.logical_and(valid, var_x.abs() > 1e-12)
    alpha[stable] = (cov_xy[stable] / var_x[stable]).to(torch.float32)
    alpha = torch.where(torch.isfinite(alpha), alpha, torch.ones_like(alpha))

    beta = torch.zeros_like(mean_x, dtype=torch.float32)
    beta[valid.expand_as(beta)] = (
        mean_y[valid.expand_as(mean_y)] - 
        alpha[valid.expand_as(alpha)].to(torch.float64) * mean_x[valid.expand_as(mean_x)]
    ).to(torch.float32)
    beta = torch.where(torch.isfinite(beta), beta, torch.zeros_like(beta))

    return alpha.detach().cpu(), beta.detach().cpu()


def cluster_correction(
    alpha: torch.Tensor,  # (num_experts, hidden_size)
    beta: torch.Tensor,   # (num_experts, hidden_size)
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cluster channels by (alpha_c, beta_c) and return cluster-averaged corrections.
    
    Returns:
        clustered_alpha: (num_experts, hidden_size) - cluster-averaged alpha per channel
        clustered_beta: (num_experts, hidden_size) - cluster-averaged beta per channel
        cluster_labels: (num_experts, hidden_size) - cluster assignment per channel
    """
    num_experts, hidden_size = alpha.shape
    clustered_alpha = torch.zeros_like(alpha)
    clustered_beta = torch.zeros_like(beta)
    cluster_labels = torch.zeros(num_experts, hidden_size, dtype=torch.long)

    for expert_idx in range(num_experts):
        a = alpha[expert_idx].numpy()  # (H,)
        b = beta[expert_idx].numpy()   # (H,)
        
        # Feature matrix: (H, 2) - each channel is a point in (alpha, beta) space
        features = np.stack([a, b], axis=1)  # (H, 2)
        
        # Normalize features for clustering
        feat_std = features.std(axis=0) + 1e-8
        features_norm = features / feat_std
        
        # K-means clustering
        actual_k = min(k, hidden_size)
        kmeans = KMeans(n_clusters=actual_k, random_state=0, n_init=10, max_iter=100)
        labels = kmeans.fit_predict(features_norm)  # (H,)
        
        # Average within each cluster
        for cluster_id in range(actual_k):
            mask = labels == cluster_id
            if mask.sum() > 0:
                clustered_alpha[expert_idx, mask] = float(a[mask].mean())
                clustered_beta[expert_idx, mask] = float(b[mask].mean())
        
        cluster_labels[expert_idx] = torch.from_numpy(labels)

    return clustered_alpha, clustered_beta, cluster_labels


def moe_forward_teacher_quant_perchannel(
    hidden_states: torch.Tensor,
    teacher_tensors: dict[str, torch.Tensor],
    quant_tensors: dict[str, torch.Tensor],
    config: Any,
    actual_length: int,
    stats: PerChannelFitMoments | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass collecting per-channel moments."""
    if actual_length <= 0:
        zeros = torch.zeros_like(hidden_states)
        return zeros, zeros

    valid_hidden_states = hidden_states[:, :actual_length, :]
    batch_size, sequence_length, hidden_dim = valid_hidden_states.shape
    flat = valid_hidden_states.reshape(-1, hidden_dim)
    router_logits = F.linear(flat, teacher_tensors["gate.weight"]).float()
    routing_probs = torch.softmax(router_logits, dim=1)
    selected_experts = torch.topk(routing_probs, config.num_experts_per_tok, dim=-1).indices
    selected_weights = routing_probs.gather(1, selected_experts)
    selected_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    selected_weights = selected_weights.to(valid_hidden_states.dtype)

    teacher_final = torch.zeros((flat.shape[0], hidden_dim), dtype=valid_hidden_states.dtype, device=valid_hidden_states.device)
    quant_final = torch.zeros((flat.shape[0], hidden_dim), dtype=valid_hidden_states.dtype, device=valid_hidden_states.device)
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]

        teacher_gate_up = F.linear(current_state, teacher_tensors["experts.gate_up_proj"][expert_idx])
        teacher_gate, teacher_up = teacher_gate_up.chunk(2, dim=-1)
        teacher_out = F.linear(F.silu(teacher_gate) * teacher_up, teacher_tensors["experts.down_proj"][expert_idx])

        quant_gate_up = F.linear(current_state, quant_tensors["experts.gate_up_proj"][expert_idx])
        quant_gate, quant_up = quant_gate_up.chunk(2, dim=-1)
        quant_out = F.linear(F.silu(quant_gate) * quant_up, quant_tensors["experts.down_proj"][expert_idx])

        if stats is not None:
            update_perchannel_fit_moments(stats, expert_idx, quant_out, teacher_out)

        routed_weight = selected_weights[token_idx, route_pos].unsqueeze(-1)
        teacher_final.index_add_(0, token_idx, (routed_weight * teacher_out).to(valid_hidden_states.dtype))
        quant_final.index_add_(0, token_idx, (routed_weight * quant_out).to(valid_hidden_states.dtype))

    # Shared expert (teacher only — quant uses same weights for shared expert)
    shared_gate = F.linear(flat, teacher_tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, teacher_tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, teacher_tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, teacher_tensors["shared_expert_gate.weight"]))
    shared_weighted = shared_out * shared_gate_value
    teacher_final = teacher_final + shared_weighted
    quant_final = quant_final + shared_weighted

    teacher_view = teacher_final.view(batch_size, sequence_length, hidden_dim)
    quant_view = quant_final.view(batch_size, sequence_length, hidden_dim)
    if stats is not None:
        diff = (teacher_view.float() - quant_view.float()).to(device=stats.layer_sq_error.device, dtype=torch.float64)
        stats.layer_sq_error += diff.square().sum()
        stats.layer_elem_count += int(diff.numel())

    teacher_padded = torch.zeros_like(hidden_states)
    quant_padded = torch.zeros_like(hidden_states)
    teacher_padded[:, :actual_length, :] = teacher_view
    quant_padded[:, :actual_length, :] = quant_view
    return teacher_padded, quant_padded


def apply_clustered_correction(
    expert_output: torch.Tensor,
    alpha: torch.Tensor,  # (hidden_size,) for this expert
    beta: torch.Tensor,   # (hidden_size,) for this expert
) -> torch.Tensor:
    """Apply per-channel (clustered) affine correction."""
    a = alpha.to(device=expert_output.device, dtype=expert_output.dtype)
    b = beta.to(device=expert_output.device, dtype=expert_output.dtype)
    return expert_output * a + b


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=SCRIPT_DIR / "results" / "proper_iter55_clustered_correction.json",
    )
    parser.add_argument(
        "--exploration-md",
        type=Path,
        default=SCRIPT_DIR / "results" / "exploration.md",
    )
    parser.add_argument(
        "--eval-max-chunks",
        type=int,
        default=0,
        help="Cap on WikiText-2 eval chunks for smoke tests; 0 = full 145-chunk test.",
    )
    return parser.parse_args()


def load_wikitext_train_ids(tokenizer: Any) -> torch.Tensor:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    return tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)


def maybe_slice_eval_chunks(
    test_ids: torch.Tensor, max_chunks: int
) -> tuple[torch.Tensor, dict[str, Any]]:
    if max_chunks <= 0:
        return test_ids, {}
    n = max_chunks * SEQLEN
    return test_ids[:n], {"eval_max_chunks": max_chunks, "truncated": True}


def fit_clustered_corrections(
    model_id: str,
    plan: Any,
    calib_chunks: torch.Tensor,
    actual_lengths: torch.Tensor,
    config: Any,
    snapshot_dir: Path,
    weight_map: dict[str, str],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, dict[int, dict[str, torch.Tensor]]]:
    """Fit per-channel moments and cluster them.
    
    Returns:
        corrections[layer_idx][k] = {
            'alpha': (num_experts, hidden_size),
            'beta': (num_experts, hidden_size),
        }
    """
    store = WeightStore(model_id, snapshot_dir, weight_map)
    embed_key, _, _ = resolve_terminal_keys(weight_map)
    inps = embed_chunks(store, embed_key, calib_chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)

    all_corrections: dict[int, dict[int, dict[str, torch.Tensor]]] = {}
    layer_mse: dict[int, float] = {}

    for layer_idx in range(config.num_hidden_layers):
        layer_type = layer_type_at(config, layer_idx)
        print(f"\n[iter55-fit] layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type})", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        teacher_moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
        del raw_tensors

        try:
            quantized_tensors = prepare_quantized_layer_tensors(plan, layer_idx, tensors, config)
            quant_moe_tensors = {k.replace("mlp.", "", 1): v for k, v in quantized_tensors.items() if k.startswith("mlp.")}
            stats_device = device
        except torch.OutOfMemoryError:
            print(f"[iter55-fit] OOM on GPU, falling back to CPU for layer {layer_idx + 1}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpu_device = torch.device("cpu")
            tensors_cpu = shorten_layer_tensors(layer_idx, store.load_tensors(layer_keys(layer_idx, layer_type)), cpu_device, dtype)
            quantized_tensors = prepare_quantized_layer_tensors(plan, layer_idx, tensors_cpu, config)
            quant_moe_tensors = {k.replace("mlp.", "", 1): v for k, v in quantized_tensors.items() if k.startswith("mlp.")}
            stats_device = cpu_device

        stats = init_perchannel_fit_moments(config, stats_device)

        for chunk_idx in range(int(inps.shape[0])):
            hidden_states = inps[chunk_idx].unsqueeze(0)
            residual, mlp_input = forward_attention_to_mlp_input(
                hidden_states, tensors, layer_type, config, position_embeddings, causal_mask,
            )
            actual_length = int(actual_lengths[chunk_idx].item())

            teacher_moe, _quant_moe = moe_forward_teacher_quant_perchannel(
                mlp_input,
                teacher_moe_tensors,
                quant_moe_tensors,
                config,
                actual_length,
                stats,
            )
            outs[chunk_idx] = residual + teacher_moe
            if should_log_chunk(chunk_idx, int(inps.shape[0])):
                print(
                    f"[iter55-fit] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{int(inps.shape[0])} | actual_len={actual_length}",
                    flush=True,
                )

        # Solve per-channel correction
        alpha_pc, beta_pc = solve_perchannel_affine(stats)  # (E, H)
        mse = float(stats.layer_sq_error.item()) / max(1, stats.layer_elem_count)
        layer_mse[layer_idx] = mse
        print(f"[iter55-fit] layer {layer_idx + 1} mse={mse:.8f}", flush=True)

        # Cluster for each K
        layer_corrections: dict[int, dict[str, torch.Tensor]] = {}
        for k in CLUSTER_COUNTS:
            print(f"[iter55-fit] clustering K={k}...", flush=True)
            clustered_alpha, clustered_beta, labels = cluster_correction(alpha_pc, beta_pc, k)
            layer_corrections[k] = {
                "alpha": clustered_alpha,  # (E, H)
                "beta": clustered_beta,    # (E, H)
                "labels": labels,          # (E, H)
                "alpha_pc": alpha_pc,      # (E, H) - original per-channel
                "beta_pc": beta_pc,        # (E, H) - original per-channel
            }
        all_corrections[layer_idx] = layer_corrections

        release_tensors(tensors)
        release_tensors(quantized_tensors)
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    release_tensors({"inps": inps, "outs": outs, "causal_mask": causal_mask})
    del store
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_corrections, layer_mse


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    print(f"\n{'='*80}")
    print(f"Iteration 55: Clustered Affine Correction (CAT-style)")
    print(f"  Foundation: Iter29 best (maca_uniform_4k = 6.567582 PPL)")
    print(f"  Innovation: Cluster-specific (alpha_k, beta_k) instead of per-channel")
    print(f"  Cluster counts: K = {CLUSTER_COUNTS}")
    print(f"  Reference: CAT (arXiv:2509.26277), KBVQ-MoE BCOS (arXiv:2602.11184)")
    print(f"{'='*80}\n")

    overall_start = time.time()
    tokenizer, _standard_calib_chunks, test_ids_full, calib_info, eval_info = load_gptq_standard_data(args.model_id)
    test_ids, eval_slice = maybe_slice_eval_chunks(test_ids_full, int(args.eval_max_chunks))
    eval_info = {**eval_info, **eval_slice}
    train_ids = load_wikitext_train_ids(tokenizer)

    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)
    index_path = snapshot_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index_payload = json.load(handle)
    total_bf16_bytes = int(root_config.get("total_size", 0) or index_payload["metadata"]["total_size"])
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)

    payload: dict[str, Any] = {
        "metadata": {
            "model": args.model_id,
            "device": args.device,
            "dtype": args.dtype,
            "seed": int(args.seed),
            "evaluation": eval_info,
            "standard_calibration_reference": calib_info,
            "experiment": "Iteration 55: Clustered Affine Correction",
            "foundation": "Iter29 maca_uniform_4k = 6.567582 PPL",
            "innovation": "Cluster-specific (alpha_k, beta_k) correction (CAT-style)",
            "cluster_counts": CLUSTER_COUNTS,
            "reference_papers": [
                "CAT: arXiv:2509.26277 (ICLR 2026)",
                "KBVQ-MoE BCOS: arXiv:2602.11184 (ICLR 2026)",
            ],
            "expected_improvement": "0.001-0.005 PPL (target: 6.562-6.567)",
            "hephaestus_approved": True,
            "hephaestus_rationale": (
                "Per-channel correction (Iter25) gave 6.587 PPL (WORSE than baseline 6.572). "
                "CAT paper shows clustering beats plain per-channel by reducing overfitting. "
                "K=4,8,16 sweep to find optimal cluster count."
            ),
        },
        "fitting": {},
        "results": {},
    }

    # Step 1: MaCa calibration (same as Iter29)
    print(f"[iter55] Step 1: MaCa calibration ({CALIB_CHUNKS} chunks × {CALIB_LENGTH} tokens)...")
    calib_chunks = build_maca_calibration_set(
        tokenizer,
        train_ids,
        seed=args.seed,
        length_counts=((CALIB_LENGTH, CALIB_CHUNKS),),
    )
    actual_lengths = torch.full((calib_chunks.shape[0],), CALIB_LENGTH, dtype=torch.long)
    print(f"[iter55] Built {calib_chunks.shape[0]} calibration chunks")

    calib_artifacts = run_maca_calibration(
        weight_map,
        calib_chunks,
        snapshot_dir,
        text_config,
        device,
        dtype,
    )

    # Step 2: Build masks (same as Iter29)
    print(f"[iter55] Step 2: Building joint_w1w2_with_topup masks...")
    w1_masks, w2_masks = build_joint_with_topup_masks(
        calib_artifacts,
        text_config,
        total_expert_elems,
        TOPUP_FRACTION,
    )
    base_plan = build_plan_from_masks(
        "maca_uniform_4k_clustered",
        "MaCa Uniform 4K with clustered affine correction",
        text_config,
        non_expert_bytes,
        total_expert_elems,
        w1_masks,
        w2_masks,
    )

    # Step 3: Fit per-channel moments and cluster
    print(f"[iter55] Step 3: Fitting per-channel moments and clustering...")
    all_corrections, layer_mse = fit_clustered_corrections(
        args.model_id,
        base_plan,
        calib_chunks,
        actual_lengths,
        text_config,
        snapshot_dir,
        weight_map,
        device,
        dtype,
    )
    payload["fitting"]["layer_mse"] = {str(k): float(v) for k, v in layer_mse.items()}

    # Step 4: Evaluate with each K
    # Rank layers by MSE for top-10 selection
    ranked_layers = sorted(layer_mse.items(), key=lambda x: x[1], reverse=True)
    top10_layers = [int(layer_idx) for layer_idx, _ in ranked_layers[:10]]
    print(f"[iter55] Top-10 layers by MSE: {top10_layers}")

    for k in CLUSTER_COUNTS:
        print(f"\n[iter55] Step 4: Evaluating K={k} clustered correction (all layers)...")
        
        # Build correction dict for this K
        corrections_k: dict[int, AffineCorrection] = {}
        for layer_idx, layer_corr in all_corrections.items():
            alpha = layer_corr[k]["alpha"]  # (E, H)
            beta = layer_corr[k]["beta"]    # (E, H)
            corrections_k[layer_idx] = AffineCorrection(
                alpha=alpha,
                beta=beta,
                mode="perchannel",
            )

        # Evaluate with all layers corrected
        eval_plan_all = build_eval_plan(base_plan, f"clustered_k{k}_all", list(range(text_config.num_hidden_layers)), text_config)
        evaluate_and_record_compound_plan(
            payload,
            eval_plan_all,
            corrections_k,
            args.output_json,
            text_config,
            total_expert_elems,
            test_ids,
            snapshot_dir,
            weight_map,
            device,
            dtype,
        )

        # Evaluate with top-10 layers corrected
        corrections_top10 = {k_: v for k_, v in corrections_k.items() if k_ in top10_layers}
        eval_plan_top10 = build_eval_plan(base_plan, f"clustered_k{k}_top10", top10_layers, text_config)
        evaluate_and_record_compound_plan(
            payload,
            eval_plan_top10,
            corrections_top10,
            args.output_json,
            text_config,
            total_expert_elems,
            test_ids,
            snapshot_dir,
            weight_map,
            device,
            dtype,
        )

    # Also evaluate baseline (no correction) for comparison
    print(f"\n[iter55] Evaluating baseline (no correction)...")
    eval_plan_base = build_eval_plan(base_plan, "no_correction", [], text_config)
    evaluate_and_record_compound_plan(
        payload,
        eval_plan_base,
        {},
        args.output_json,
        text_config,
        total_expert_elems,
        test_ids,
        snapshot_dir,
        weight_map,
        device,
        dtype,
    )

    atomic_json_dump(payload, args.output_json)

    # Print summary
    elapsed = time.time() - overall_start
    print(f"\n{'='*80}")
    print(f"Iteration 55 COMPLETE")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Results: {args.output_json}")
    print(f"\n  PPL Summary:")
    for config_name, result in payload.get("results", {}).items():
        if isinstance(result, dict):
            ppl = result.get("ppl", "N/A")
            print(f"    {config_name}: {ppl}")
    print(f"\n  vs Iter29 (maca_uniform_4k): 6.567582")
    print(f"{'='*80}\n")

    if args.exploration_md.exists():
        best_ppl = min(
            (v.get("ppl", 999) for v in payload.get("results", {}).values() if isinstance(v, dict)),
            default="N/A"
        )
        upsert_exploration_section(
            args.exploration_md,
            SECTION_MARKER,
            f"Clustered Affine Correction: best PPL = {best_ppl}\n"
            f"- K values: {CLUSTER_COUNTS}\n"
            f"- Reference: CAT (arXiv:2509.26277)\n"
            f"- vs Iter29: 6.567582 PPL",
        )


if __name__ == "__main__":
    main()

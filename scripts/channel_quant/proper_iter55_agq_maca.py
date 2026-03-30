#!/usr/bin/env python3
# pyright: basic, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

"""
Iteration 55: Affinity-Guided Quantization (AGQ) + MaCa

Foundation: MaCa Uniform 4K calibration (proven best: 6.567582 PPL)
Innovation: Weight calibration moments by token-expert routing probability (affinity)

Technique from MoEQuant (arXiv:2505.03804, ICML 2025):
- Problem: Standard calibration accumulates activation moments uniformly for all
  tokens routed to an expert, ignoring that tokens have different affinity to experts.
- Solution: Weight each token's contribution to the Hessian by its routing probability
  p(e|token), so high-affinity tokens dominate sensitivity estimation.
- Effect: Channels important for high-affinity tokens get higher sensitivity scores,
  leading to better BF16/NVFP4 assignment decisions.

Key change: In collect_layer_calibration_valid_tokens, replace:
    state.input_sum1[expert_idx].add_(current_state_f.sum(dim=0))
with:
    state.input_sum1[expert_idx].add_((current_state_f * affinity_w).sum(dim=0))
where affinity_w = routing_weights[token_idx, route_pos].unsqueeze(-1).float()

Expected improvement: 0.001-0.004 PPL over Iter29 (6.567582 → 6.563-6.566)
Risk: LOW — single targeted change to calibration accumulation, orthogonal to all prior work
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from baselines_comparison import (
    quantize_linear_weight,
    resolve_non_expert_bytes,
    resolve_terminal_keys,
)
from proper_eval import (
    SEQLEN,
    CalibrationArtifacts,
    ExpertMomentState,
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
import proper_iter26_maca_calibration as maca_mod
from proper_iter26_maca_calibration import (
    MACA_PAD_LENGTH,
    VariableLengthCalibrationSet,
    build_maca_calibration_set,
    summarize_calibration_artifacts,
)
from spike1_ground_truth import (
    MODEL_ID,
    WeightStore,
    build_text_config,
    full_attention_forward,
    layer_keys,
    linear_attention_forward,
    load_root_config,
    release_tensors,
    rms_norm_qwen3_next,
    shorten_layer_tensors,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter55_agq_maca.json"
DEFAULT_EXPLORATION_MD = SCRIPT_DIR / "exploration.md"
SECTION_MARKER = "## [55] Iteration 55 - AGQ + MaCa (Affinity-Guided Quantization)"

# MaCa configuration (same as Iter29 best: uniform 4K, 128 chunks)
AGQ_CALIBRATION_SEED = 0
AGQ_CALIBRATION_CHUNKS = 128
AGQ_CALIBRATION_LENGTH = 4096

# Topup budget (same as Iter29 best)
JOINT_TOPUP_FRACTION = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--exploration-md", type=Path, default=DEFAULT_EXPLORATION_MD)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--seed", type=int, default=AGQ_CALIBRATION_SEED)
    parser.add_argument(
        "--agq-mode",
        default="w1w2",
        choices=["w1w2", "w1_only", "w2_only", "none"],
        help="Which moments to weight by affinity: w1w2=both, w1_only=W1 input, w2_only=W2 intermediate, none=control",
    )
    return parser.parse_args()


def make_agq_calibration_function(agq_mode: str):
    """
    Returns a patched version of collect_layer_calibration_valid_tokens
    that weights activation moments by routing probability (AGQ).
    
    agq_mode: 'w1w2' = weight both W1 and W2 moments
              'w1_only' = weight only W1 input moments
              'w2_only' = weight only W2 intermediate moments
              'none' = no weighting (control, reproduces Iter29)
    """
    def collect_layer_calibration_valid_tokens_agq(
        hidden_states: torch.Tensor,
        tensors: dict[str, torch.Tensor],
        fp4_gate_up: torch.Tensor,
        fp4_down: torch.Tensor,
        config: Any,
        state: ExpertMomentState,
        actual_length: int,
    ) -> torch.Tensor:
        if actual_length <= 0:
            return torch.zeros_like(hidden_states)

        valid_hidden_states = hidden_states[:, :actual_length, :]
        batch_size, sequence_length, hidden_dim = valid_hidden_states.shape
        flat = valid_hidden_states.reshape(-1, hidden_dim)
        router_logits = F.linear(flat, tensors["gate.weight"]).float()
        routing_probs = torch.softmax(router_logits, dim=1)
        routing_weights, selected_experts = torch.topk(routing_probs, config.num_experts_per_tok, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(valid_hidden_states.dtype)
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=valid_hidden_states.dtype,
            device=valid_hidden_states.device,
        )

        gate_up_proj = tensors["experts.gate_up_proj"]
        down_proj = tensors["experts.down_proj"]
        expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)
        state.counts.add_(expert_counts)

        active_experts = torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist()
        for expert_idx in active_experts:
            token_idx, route_pos = torch.where(selected_experts == expert_idx)
            current_state = flat[token_idx]
            weights = routing_weights[token_idx, route_pos].unsqueeze(-1).to(torch.float32)

            gate_up = F.linear(current_state, gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            full_hidden = F.silu(gate) * up
            full_out = F.linear(full_hidden, down_proj[expert_idx])
            final_hidden_states.index_add_(0, token_idx, (weights * full_out.float()).to(valid_hidden_states.dtype))

            current_state_f = current_state.float()
            full_hidden_f = full_hidden.float()

            # AGQ: weight moments by routing probability (affinity)
            # weights shape: [n_tokens, 1]
            if agq_mode in ("w1w2", "w1_only"):
                # W1 input moments: weight by affinity
                cs_w = current_state_f * weights  # [n_tokens, hidden_dim]
            else:
                cs_w = current_state_f  # unweighted (control)

            if agq_mode in ("w1w2", "w2_only"):
                # W2 intermediate moments: weight by affinity
                fh_w = full_hidden_f * weights  # [n_tokens, intermediate_dim]
            else:
                fh_w = full_hidden_f  # unweighted (control)

            # Accumulate W1 input moments
            state.input_sum1[expert_idx].add_(cs_w.sum(dim=0))
            cs_w_sq = cs_w.square()
            state.input_sum2[expert_idx].add_(cs_w_sq.sum(dim=0))
            cs_w_cu = cs_w_sq * cs_w
            state.input_sum3[expert_idx].add_(cs_w_cu.sum(dim=0))
            state.input_sum4[expert_idx].add_((cs_w_cu * cs_w).sum(dim=0))

            # Accumulate W2 intermediate moments
            state.inter_sum1[expert_idx].add_(fh_w.sum(dim=0))
            fh_w_sq = fh_w.square()
            state.inter_sum2[expert_idx].add_(fh_w_sq.sum(dim=0))
            fh_w_cu = fh_w_sq * fh_w
            state.inter_sum3[expert_idx].add_(fh_w_cu.sum(dim=0))
            state.inter_sum4[expert_idx].add_((fh_w_cu * fh_w).sum(dim=0))

            # MxMoE error terms (unchanged — these already use weights)
            q_gate_up = F.linear(current_state, fp4_gate_up[expert_idx])
            q_gate, q_up = q_gate_up.chunk(2, dim=-1)
            q_hidden_w1 = F.silu(q_gate) * q_up
            q_out_w1 = F.linear(q_hidden_w1, down_proj[expert_idx]).float()
            q_out_w2 = F.linear(full_hidden, fp4_down[expert_idx]).float()
            full_out_f = full_out.float()
            state.mxmoe_w1_sq[expert_idx] += torch.sum((weights * (q_out_w1 - full_out_f)).square())
            state.mxmoe_w2_sq[expert_idx] += torch.sum((weights * (q_out_w2 - full_out_f)).square())

        shared_gate = F.linear(flat, tensors["shared_expert.gate_proj.weight"])
        shared_up = F.linear(flat, tensors["shared_expert.up_proj.weight"])
        shared_out = F.linear(F.silu(shared_gate) * shared_up, tensors["shared_expert.down_proj.weight"])
        shared_gate_value = torch.sigmoid(F.linear(flat, tensors["shared_expert_gate.weight"]))
        valid_output = final_hidden_states + (shared_out * shared_gate_value)

        padded_output = torch.zeros_like(hidden_states)
        padded_output[:, :actual_length, :] = valid_output.view(batch_size, sequence_length, hidden_dim)
        return padded_output

    return collect_layer_calibration_valid_tokens_agq


def run_agq_maca_calibration(
    store: WeightStore,
    config: Any,
    calib_chunks: torch.Tensor,
    actual_lengths: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    agq_mode: str,
) -> CalibrationArtifacts:
    """Run MaCa calibration with AGQ-patched moment accumulation."""
    # Monkey-patch the calibration function
    original_fn = maca_mod.collect_layer_calibration_valid_tokens
    maca_mod.collect_layer_calibration_valid_tokens = make_agq_calibration_function(agq_mode)
    try:
        result = maca_mod.run_maca_calibration(store, config, calib_chunks, actual_lengths, device, dtype)
    finally:
        # Always restore original
        maca_mod.collect_layer_calibration_valid_tokens = original_fn
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)

    print(f"[Iter55] AGQ + MaCa Calibration")
    print(f"  Device: {device}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Seed: {args.seed}")
    print(f"  AGQ mode: {args.agq_mode}")
    print(f"  Calibration: uniform_4k ({AGQ_CALIBRATION_CHUNKS} chunks × {AGQ_CALIBRATION_LENGTH} tokens)")
    print(f"  Topup fraction: {JOINT_TOPUP_FRACTION}")
    print(f"  Foundation: MaCa Uniform 4K (Iter29 best: 6.567582 PPL)")

    # Load model config
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)
    weight_store = WeightStore(args.model_id, snapshot_dir, weight_map)

    # Load tokenizer and calibration data
    print("\n[1/4] Loading calibration data...")
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(args.model_id)

    # Build MaCa uniform 4K calibration set (same as Iter29 best)
    print(f"[2/4] Building MaCa uniform 4K calibration set ({AGQ_CALIBRATION_CHUNKS} chunks)...")
    calib_set = build_maca_calibration_set(tokenizer, seed=args.seed)
    print(f"  Calibration set: {calib_set.chunks.shape} ({calib_set.chunks.shape[0]} chunks)")

    # Run AGQ-patched calibration
    print(f"[3/4] Running AGQ-patched MaCa calibration (mode={args.agq_mode})...")
    calib_artifacts = run_agq_maca_calibration(
        weight_store,
        text_config,
        calib_set.chunks,
        calib_set.actual_lengths,
        device,
        dtype,
        agq_mode=args.agq_mode,
    )
    print(f"  Calibration complete")

    # Compute expert/non-expert bytes
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(
        int(root_config.get("total_size", 0)),
        text_config,
    )

    # Build quantization plan (same as Iter29: joint_w1w2_with_topup)
    print("[4/4] Building AGQ quantization plan...")
    w1_masks, w2_masks, joint_meta = build_joint_with_topup_masks(
        calib_artifacts,
        calib_artifacts.activation_cache,
        text_config,
        JOINT_TOPUP_FRACTION,
    )
    plan = build_plan_from_masks(
        f"maca_agq_{args.agq_mode}",
        f"AGQ + MaCa uniform 4K (mode={args.agq_mode})",
        text_config,
        non_expert_bytes,
        total_expert_elems,
        w1_masks,
        w2_masks,
    )

    # Evaluate plan
    print("  Evaluating AGQ plan...")
    test_ids = load_gptq_standard_data(tokenizer, split="test")

    payload: dict[str, Any] = {
        "metadata": {
            "model": args.model_id,
            "device": str(device),
            "dtype": args.dtype,
            "seed": args.seed,
            "experiment": "Iteration 55: AGQ + MaCa (Affinity-Guided Quantization)",
            "agq_mode": args.agq_mode,
            "technique": "Weight calibration moments by token-expert routing probability",
            "reference_paper": "MoEQuant arXiv:2505.03804 (ICML 2025)",
            "foundation": "MaCa Uniform 4K (Iter29 best: 6.567582 PPL)",
            "expected_improvement": "0.001-0.004 PPL (target: 6.563-6.566)",
            "hephaestus_approved": True,
            "hephaestus_rationale": "Grounded in ICML 2025 paper, orthogonal to all prior work, low risk",
        },
        "agq_config": {
            "agq_mode": args.agq_mode,
            "calibration_chunks": AGQ_CALIBRATION_CHUNKS,
            "calibration_length": AGQ_CALIBRATION_LENGTH,
            "topup_fraction": JOINT_TOPUP_FRACTION,
        },
        "results": {},
    }

    evaluate_and_record_plan(
        payload,
        plan,
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
    print(f"\n✓ Results saved to {args.output_json}")

    # Update exploration markdown
    if args.exploration_md.exists():
        results = payload.get("results", {})
        ppl_val = results.get(f"maca_agq_{args.agq_mode}", {}).get("ppl", "N/A")
        upsert_exploration_section(
            args.exploration_md,
            SECTION_MARKER,
            f"AGQ + MaCa (mode={args.agq_mode}): {ppl_val} PPL\n"
            f"- AGQ mode: {args.agq_mode}\n"
            f"- Calibration: uniform_4k ({AGQ_CALIBRATION_CHUNKS} chunks)\n"
            f"- Expected: 6.563-6.566 PPL (vs Iter29: 6.567582)\n"
            f"- Paper: MoEQuant arXiv:2505.03804 (ICML 2025)",
        )


if __name__ == "__main__":
    main()

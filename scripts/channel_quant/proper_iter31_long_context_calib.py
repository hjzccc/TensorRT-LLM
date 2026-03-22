#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false
"""Iteration 31 — Long-context calibration sweep.

Iter29 showed that uniform-4096 (PPL 6.5676) beats all mixed-length MaCa variants.
The key insight: MACA_PAD_LENGTH=4096 in iter26/29 silently truncated any 8k+ chunks
to 4096 tokens. This iteration tests truly longer calibration sequences by raising the
pad length to 8192 and 16384.

Configs tested:
  uniform_8k   — 128 × 8192-token chunks (pad=8192)
  uniform_16k  — 128 × 16384-token chunks (pad=16384)
  uniform_4k   — 128 × 4096-token chunks (pad=4096) — control, should match iter29 best

All configs rebuild joint_w1w2_with_topup masks from their own calibration statistics
and evaluate on the full WikiText-2 test set (145 chunks × 2048 tokens).
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from baselines_comparison import LayerMetricBundle, quantize_linear_weight, resolve_non_expert_bytes, resolve_terminal_keys
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
)
from proper_iter01 import build_plan_from_masks
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from proper_iter11_push_router_affinity import evaluate_and_record_plan
from proper_iter25_learned_correction import maybe_slice_eval_chunks, resolve_requested_plans
from proper_iter26_maca_calibration import (
    VariableLengthCalibrationSet,
    load_wikitext_train_ids,
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
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter31_long_context_calib.json"
TOTAL_CALIBRATION_CHUNKS = 128


@dataclass(frozen=True)
class LongContextConfig:
    name: str
    chunk_length: int  # actual token length per chunk (= pad length for this config)
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--configs",
        default="",
        help="Comma-separated subset of configs to run. Default runs all three.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-max-chunks",
        type=int,
        default=0,
        help="Optional cap on WikiText-2 evaluation chunks for smoke tests; 0 means full 145-chunk test.",
    )
    return parser.parse_args()


def build_sweep_configs() -> list[LongContextConfig]:
    return [
        LongContextConfig(
            name="uniform_4k",
            chunk_length=4096,
            description="Control: 128 × 4096-token chunks. Should reproduce iter29 maca_uniform_4k at PPL 6.5676.",
        ),
        LongContextConfig(
            name="uniform_8k",
            chunk_length=8192,
            description="128 × 8192-token chunks with pad_length=8192. Tests whether truly longer context improves sensitivity estimation beyond 4096.",
        ),
        LongContextConfig(
            name="uniform_16k",
            chunk_length=16384,
            description="128 × 16384-token chunks with pad_length=16384. Pushes to 1/2 of the model's 32k context window.",
        ),
    ]


def should_log_chunk(chunk_idx: int, total_chunks: int) -> bool:
    return chunk_idx == 0 or (chunk_idx + 1) % 16 == 0 or chunk_idx + 1 == total_chunks


def build_long_context_calibration_set(
    tokenizer: Any,
    train_ids: torch.Tensor,
    seed: int,
    cfg: LongContextConfig,
) -> VariableLengthCalibrationSet:
    """Build a calibration set of TOTAL_CALIBRATION_CHUNKS chunks, each exactly cfg.chunk_length tokens."""
    rng = random.Random(f"iter31:{seed}:{cfg.name}")
    chunk_length = cfg.chunk_length
    max_start = int(train_ids.shape[1]) - chunk_length - 1
    if max_start < 0:
        raise RuntimeError(
            f"WikiText-2 train ({train_ids.shape[1]} tokens) is too short for chunk_length={chunk_length}"
        )

    padded_chunks: list[torch.Tensor] = []
    actual_lengths: list[int] = []
    for _ in range(TOTAL_CALIBRATION_CHUNKS):
        start = rng.randint(0, max_start)
        sample = train_ids[:, start : start + chunk_length]
        # No padding needed — each chunk is exactly chunk_length tokens
        padded_chunks.append(sample)
        actual_lengths.append(chunk_length)

    chunks = torch.cat(padded_chunks, dim=0).contiguous()
    actual_lengths_tensor = torch.as_tensor(actual_lengths, dtype=torch.long)
    actual_token_total = int(actual_lengths_tensor.sum().item())
    padded_token_total = int(chunks.numel())
    metadata = {
        "dataset": "wikitext/wikitext-2-raw-v1",
        "split": "train",
        "tokenizer": tokenizer.name_or_path,
        "base_seed": int(seed),
        "sampling_seed": f"iter31:{seed}:{cfg.name}",
        "config_name": cfg.name,
        "description": cfg.description,
        "chunk_length": int(chunk_length),
        "samples": int(chunks.shape[0]),
        "sample_shape": [int(chunks.shape[0]), int(chunks.shape[1])],
        "padding_value": 0,
        "actual_token_total": actual_token_total,
        "padded_token_total": padded_token_total,
        "real_token_fraction": 1.0,
        "train_tokens": int(train_ids.shape[1]),
        "standard_reference_tokens": int(SEQLEN * chunks.shape[0]),
    }
    return VariableLengthCalibrationSet(chunks=chunks, actual_lengths=actual_lengths_tensor, metadata=metadata)


@torch.inference_mode()
def run_long_context_calibration(
    store: WeightStore,
    config: Any,
    calib_chunks: torch.Tensor,
    actual_lengths: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> CalibrationArtifacts:
    """Run calibration with variable-length chunks, accumulating only valid tokens."""
    chunk_length = int(actual_lengths[0].item())
    print(f"\n=== Calibration (long-context uniform, chunk_length={chunk_length}) ===", flush=True)
    embed_key, _, _ = resolve_terminal_keys(store.weight_map)
    inps = embed_chunks(store, embed_key, calib_chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)

    routing_counts: dict[int, torch.Tensor] = {}
    activation_cache: dict[int, LayerMetricBundle] = {}
    mxmoe_w1_deltas: dict[int, torch.Tensor] = {}
    mxmoe_w2_deltas: dict[int, torch.Tensor] = {}
    mc_moe_scores: dict[int, torch.Tensor] = {}

    for layer_idx in range(config.num_hidden_layers):
        layer_type = layer_type_at(config, layer_idx)
        print(f"[long-calib] load layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type})", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors

        gate_up_proj = tensors["mlp.experts.gate_up_proj"]
        down_proj = tensors["mlp.experts.down_proj"]
        fp4_gate_up = torch.stack(
            [quantize_linear_weight(gate_up_proj[expert_idx], "fp4") for expert_idx in range(config.num_experts)],
            dim=0,
        )
        fp4_down = torch.stack(
            [quantize_linear_weight(down_proj[expert_idx], "fp4") for expert_idx in range(config.num_experts)],
            dim=0,
        )
        state = init_expert_moment_state(config, device)

        for chunk_idx in range(inps.shape[0]):
            actual_length = int(actual_lengths[chunk_idx].item())
            hidden_states = inps[chunk_idx].unsqueeze(0)
            residual = hidden_states
            hidden_norm = rms_norm_qwen3_next(hidden_states, tensors["input_layernorm.weight"], config.rms_norm_eps)
            if layer_type == "full_attention":
                attn_tensors = {k.replace("self_attn.", "", 1): v for k, v in tensors.items() if k.startswith("self_attn.")}
                mixed = full_attention_forward(hidden_norm, attn_tensors, config, position_embeddings, causal_mask)
            else:
                attn_tensors = {k.replace("linear_attn.", "", 1): v for k, v in tensors.items() if k.startswith("linear_attn.")}
                mixed = linear_attention_forward(hidden_norm, attn_tensors, config)
            hidden_states = residual + mixed

            residual = hidden_states
            mlp_input = rms_norm_qwen3_next(hidden_states, tensors["post_attention_layernorm.weight"], config.rms_norm_eps)
            moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}

            # Process all tokens (no padding — actual_length == chunk_length for all chunks)
            batch_size, sequence_length, hidden_dim = mlp_input.shape
            flat = mlp_input.reshape(-1, hidden_dim)
            router_logits = F.linear(flat, moe_tensors["gate.weight"]).float()
            routing_probs = torch.softmax(router_logits, dim=1)
            routing_weights, selected_experts = torch.topk(routing_probs, config.num_experts_per_tok, dim=-1)
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights.to(mlp_input.dtype)
            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim),
                dtype=mlp_input.dtype,
                device=mlp_input.device,
            )

            expert_gate_up = moe_tensors["experts.gate_up_proj"]
            expert_down = moe_tensors["experts.down_proj"]
            expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)
            state.counts.add_(expert_counts)

            active_experts = torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist()
            for expert_idx in active_experts:
                token_idx, route_pos = torch.where(selected_experts == expert_idx)
                current_state = flat[token_idx]
                weights = routing_weights[token_idx, route_pos].unsqueeze(-1).to(torch.float32)

                gate_up = F.linear(current_state, expert_gate_up[expert_idx])
                gate, up = gate_up.chunk(2, dim=-1)
                full_hidden = F.silu(gate) * up
                full_out = F.linear(full_hidden, expert_down[expert_idx])
                final_hidden_states.index_add_(0, token_idx, (weights * full_out.float()).to(mlp_input.dtype))

                current_state_f = current_state.float()
                full_hidden_f = full_hidden.float()
                state.input_sum1[expert_idx].add_(current_state_f.sum(dim=0))
                squared = current_state_f.square()
                state.input_sum2[expert_idx].add_(squared.sum(dim=0))
                cubed = squared * current_state_f
                state.input_sum3[expert_idx].add_(cubed.sum(dim=0))
                state.input_sum4[expert_idx].add_((cubed * current_state_f).sum(dim=0))
                state.inter_sum1[expert_idx].add_(full_hidden_f.sum(dim=0))
                inter_squared = full_hidden_f.square()
                state.inter_sum2[expert_idx].add_(inter_squared.sum(dim=0))
                inter_cubed = inter_squared * full_hidden_f
                state.inter_sum3[expert_idx].add_(inter_cubed.sum(dim=0))
                state.inter_sum4[expert_idx].add_((inter_cubed * full_hidden_f).sum(dim=0))

                q_gate_up = F.linear(current_state, fp4_gate_up[expert_idx])
                q_gate, q_up = q_gate_up.chunk(2, dim=-1)
                q_hidden_w1 = F.silu(q_gate) * q_up
                q_out_w1 = F.linear(q_hidden_w1, expert_down[expert_idx]).float()
                q_out_w2 = F.linear(full_hidden, fp4_down[expert_idx]).float()
                full_out_f = full_out.float()
                state.mxmoe_w1_sq[expert_idx] += torch.sum((weights * (q_out_w1 - full_out_f)).square())
                state.mxmoe_w2_sq[expert_idx] += torch.sum((weights * (q_out_w2 - full_out_f)).square())

            shared_gate = F.linear(flat, moe_tensors["shared_expert.gate_proj.weight"])
            shared_up = F.linear(flat, moe_tensors["shared_expert.up_proj.weight"])
            shared_out = F.linear(F.silu(shared_gate) * shared_up, moe_tensors["shared_expert.down_proj.weight"])
            shared_gate_value = torch.sigmoid(F.linear(flat, moe_tensors["shared_expert_gate.weight"]))
            moe_output = final_hidden_states + (shared_out * shared_gate_value)
            outs[chunk_idx] = residual + moe_output.view(batch_size, sequence_length, hidden_dim)

            if should_log_chunk(chunk_idx, int(inps.shape[0])):
                print(
                    f"[long-calib] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{inps.shape[0]} | len={actual_length}",
                    flush=True,
                )

        bundle, layer_w1, layer_w2, layer_mc = finalize_layer_metrics(
            gate_up_proj, down_proj, fp4_gate_up, fp4_down, state, config
        )
        routing_counts[layer_idx] = bundle.routing_counts
        activation_cache[layer_idx] = bundle
        mxmoe_w1_deltas[layer_idx] = layer_w1
        mxmoe_w2_deltas[layer_idx] = layer_w2
        mc_moe_scores[layer_idx] = layer_mc

        release_tensors({"fp4_gate_up": fp4_gate_up, "fp4_down": fp4_down})
        release_tensors(tensors)
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    release_tensors({"inps": inps, "outs": outs, "causal_mask": causal_mask})
    return CalibrationArtifacts(
        routing_counts=routing_counts,
        activation_cache=activation_cache,
        mxmoe_w1_deltas=mxmoe_w1_deltas,
        mxmoe_w2_deltas=mxmoe_w2_deltas,
        mc_moe_scores=mc_moe_scores,
    )


def build_joint_plan_for_config(
    cfg: LongContextConfig,
    calibration: CalibrationArtifacts,
    config: Any,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> tuple[Any, dict[str, Any]]:
    w1_masks, w2_masks, joint_meta = build_joint_with_topup_masks(
        calibration,
        calibration.activation_cache,
        config,
        JOINT_MEDIUM_TOPUP_FRACTION,
    )
    plan = build_plan_from_masks(
        cfg.name,
        cfg.description,
        config,
        non_expert_bytes,
        total_expert_elems,
        w1_masks,
        w2_masks,
    )
    return plan, {
        **joint_meta,
        "calibration_variant": cfg.name,
        "chunk_length": int(cfg.chunk_length),
        "source_base": "joint_w1w2_with_topup",
        "joint_topup_fraction": float(JOINT_MEDIUM_TOPUP_FRACTION),
    }


def print_results_table(payload: dict[str, Any], configs: list[LongContextConfig]) -> None:
    results = payload.get("results", {})
    control_row = results.get("uniform_4k")
    control_ppl = float(control_row["ppl"]) if isinstance(control_row, dict) and "ppl" in control_row else None

    ordered_names = [cfg.name for cfg in configs if cfg.name in results]
    print("\n" + "=" * 120, flush=True)
    print("proper_iter31_long_context_calib | Long-context calibration sweep | WikiText-2 full eval | GPTQ-standard", flush=True)
    print("=" * 120, flush=True)
    print(
        f"{'Config':<20} {'ChunkLen':>10} {'PPL':>10} {'Delta vs 4k':>14} {'Memory GB':>12} {'Eval chunks':>12} {'Time s':>10}",
        flush=True,
    )
    print("-" * 120, flush=True)
    for name in ordered_names:
        row = results[name]
        cfg = next((c for c in configs if c.name == name), None)
        chunk_len = cfg.chunk_length if cfg is not None else 0
        delta_text = "n/a" if control_ppl is None else f"{float(row['ppl']) - control_ppl:+.4f}"
        print(
            f"{name:<20} {chunk_len:>10d} {float(row['ppl']):>10.4f} {delta_text:>14} {float(row['memory_gb']):>12.3f} {int(row.get('eval_chunks', 0)):>12d} {float(row['time_s']):>10.1f}",
            flush=True,
        )
    print("-" * 120, flush=True)
    best_row = min(
        ((name, results[name]) for name in ordered_names if "ppl" in results[name]),
        key=lambda x: float(x[1]["ppl"]),
        default=None,
    )
    if best_row is not None:
        print(f"best: {best_row[0]} | ppl={float(best_row[1]['ppl']):.6f}", flush=True)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    configs = build_sweep_configs()
    requested_names = resolve_requested_plans(args.configs)
    if requested_names is not None:
        all_names = {cfg.name for cfg in configs}
        missing = sorted(requested_names.difference(all_names))
        if missing:
            raise ValueError(f"Unknown configs requested: {', '.join(missing)}")
        selected_configs = [cfg for cfg in configs if cfg.name in requested_names]
    else:
        selected_configs = configs

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

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
            "experiment": "Iteration 31 long-context calibration sweep",
            "hypothesis": (
                "iter29 showed uniform-4096 (PPL 6.5676) beats all mixed-length MaCa variants. "
                "However, iter29's maca_with_8k was silently truncated to 4096 tokens because MACA_PAD_LENGTH=4096. "
                "This iteration tests truly longer calibration by using chunk_length as the pad length. "
                "If longer context improves Hessian quality, uniform_8k should beat uniform_4k."
            ),
            "protocol": {
                "reference": str(SCRIPT_DIR / "proper_eval.py"),
                "full_test_join": True,
                "seqlen": SEQLEN,
                "non_overlapping": True,
                "loss_accumulation": "loss.float() * seqlen",
                "calibration_total_chunks": TOTAL_CALIBRATION_CHUNKS,
                "no_padding": True,
            },
            "configs": [
                {"name": cfg.name, "chunk_length": cfg.chunk_length, "description": cfg.description}
                for cfg in configs
            ],
            "requested_configs": sorted(requested_names) if requested_names is not None else [cfg.name for cfg in configs],
        },
        "calibration_sets": {},
        "results": {},
    }

    if args.output_json.exists():
        try:
            with args.output_json.open("r", encoding="utf-8") as fh:
                existing = json.load(fh)
            if isinstance(existing, dict):
                payload["metadata"] = {**existing.get("metadata", {}), **payload["metadata"]}
                payload["calibration_sets"] = {**existing.get("calibration_sets", {}), **payload["calibration_sets"]}
                payload["results"] = {**existing.get("results", {}), **payload["results"]}
                print(f"[resume] loaded existing results from {args.output_json}", flush=True)
        except Exception:
            print(f"[resume] ignoring unreadable existing results at {args.output_json}", flush=True)
    atomic_json_dump(args.output_json, payload)

    store = WeightStore(args.model_id, snapshot_dir, weight_map)

    for cfg in selected_configs:
        existing_row = payload.get("results", {}).get(cfg.name)
        if isinstance(existing_row, dict) and "ppl" in existing_row:
            print(f"[skip] {cfg.name} already present in {args.output_json}", flush=True)
            continue

        print(f"\n=== Build {cfg.name} calibration set (chunk_length={cfg.chunk_length}) ===", flush=True)
        calibration_set = build_long_context_calibration_set(tokenizer, train_ids, int(args.seed), cfg)
        payload["calibration_sets"][cfg.name] = calibration_set.metadata
        atomic_json_dump(args.output_json, payload)

        calibration = run_long_context_calibration(
            store, text_config, calibration_set.chunks, calibration_set.actual_lengths, device, dtype
        )
        payload["calibration_sets"][cfg.name]["summary"] = summarize_calibration_artifacts(calibration)
        atomic_json_dump(args.output_json, payload)

        plan, extras = build_joint_plan_for_config(
            cfg, calibration, text_config, non_expert_bytes, total_expert_elems
        )
        evaluate_and_record_plan(
            payload,
            plan,
            extras,
            args.output_json,
            text_config,
            total_expert_elems,
            test_ids,
            snapshot_dir,
            weight_map,
            device,
            dtype,
        )

    payload["metadata"]["elapsed_s"] = round(time.time() - overall_start, 1)
    atomic_json_dump(args.output_json, payload)
    print_results_table(payload, configs)
    print(f"\nSaved results -> {args.output_json}", flush=True)

    del store
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

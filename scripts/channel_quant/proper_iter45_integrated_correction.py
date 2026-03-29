#!/usr/bin/env python3
# pyright: basic, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from baselines_comparison import LayerMetricBundle, quantize_linear_weight, resolve_non_expert_bytes, resolve_terminal_keys
from proper_eval import CalibrationArtifacts, atomic_json_dump, build_position_context, dtype_from_name, embed_chunks, finalize_layer_metrics, init_expert_moment_state, layer_type_at, load_gptq_standard_data
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from proper_iter25_learned_correction import AffineCorrection, apply_affine_correction, evaluate_plan_with_corrections, filter_top_layers, forward_attention_to_mlp_input, maybe_slice_eval_chunks, resolve_requested_plans, should_log_chunk
from proper_iter26_maca_calibration import MACA_CHUNKS_PER_LENGTH, MACA_LENGTHS, VariableLengthCalibrationSet, build_maca_calibration_set, summarize_calibration_artifacts
from proper_iter28_maca_plus_correction import BaseVariant, build_maca_joint_plan, evaluate_and_record_compound_plan, init_scalar_fit_moments, moe_forward_teacher_quant_valid_tokens, slice_maca_set, solve_scalar_affine_from_moments, summarize_scalar_fit
from spike1_ground_truth import MODEL_ID, WeightStore, build_text_config, layer_keys, load_root_config, release_tensors, shorten_layer_tensors

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter45_integrated_correction.json"
PLAN_ALL = "maca_joint_integrated_correction_all"
PLAN_TOP10 = "maca_joint_integrated_correction_top10"
PLAN_NONE = "maca_joint_integrated_no_correction"
FIT_KEY = "maca_integrated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--plans",
        default=f"{PLAN_ALL},{PLAN_TOP10},{PLAN_NONE}",
        help="Comma-separated subset of plan names to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--maca-chunks-per-length",
        type=int,
        default=MACA_CHUNKS_PER_LENGTH,
        help="Number of MaCa chunks to keep per calibration length; useful for smoke tests.",
    )
    parser.add_argument(
        "--eval-max-chunks",
        type=int,
        default=0,
        help="Optional cap on evaluation chunks for smoke tests; 0 means full WikiText-2 test.",
    )
    return parser.parse_args()


def build_full_fp4_quant_moe_tensors(teacher_moe_tensors: dict[str, torch.Tensor], config: Any) -> dict[str, torch.Tensor]:
    quant_moe_tensors = dict(teacher_moe_tensors)
    quant_moe_tensors["experts.gate_up_proj"] = torch.stack(
        [quantize_linear_weight(teacher_moe_tensors["experts.gate_up_proj"][expert_idx], "fp4") for expert_idx in range(config.num_experts)],
        dim=0,
    )
    quant_moe_tensors["experts.down_proj"] = torch.stack(
        [quantize_linear_weight(teacher_moe_tensors["experts.down_proj"][expert_idx], "fp4") for expert_idx in range(config.num_experts)],
        dim=0,
    )
    return quant_moe_tensors


@torch.inference_mode()
def fit_integrated_scalar_corrections(
    model_id: str,
    calibration_set: VariableLengthCalibrationSet,
    config: Any,
    snapshot_dir: Path,
    weight_map: dict[str, str],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[int, AffineCorrection], list[int], dict[str, Any]]:
    store = WeightStore(model_id, snapshot_dir, weight_map)
    embed_key, _norm_key, _lm_head_key = resolve_terminal_keys(weight_map)
    calib_chunks = calibration_set.chunks
    actual_lengths = calibration_set.actual_lengths

    inps = embed_chunks(store, embed_key, calib_chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)

    corrections: dict[int, AffineCorrection] = {}
    layer_rows: list[dict[str, Any]] = []
    fitting_layers: dict[str, Any] = {}

    for layer_idx in range(config.num_hidden_layers):
        layer_type = layer_type_at(config, layer_idx)
        print(f"\n=== Fit integrated correction layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type}) ===", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors

        teacher_moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
        quant_moe_tensors = build_full_fp4_quant_moe_tensors(teacher_moe_tensors, config)
        stats = init_scalar_fit_moments(config, device)

        for chunk_idx in range(int(inps.shape[0])):
            hidden_states = inps[chunk_idx].unsqueeze(0)
            residual, mlp_input = forward_attention_to_mlp_input(
                hidden_states,
                tensors,
                layer_type,
                config,
                position_embeddings,
                causal_mask,
            )
            actual_length = int(actual_lengths[chunk_idx].item())
            teacher_moe, _quant_moe = moe_forward_teacher_quant_valid_tokens(
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
                    f"[fit:integrated] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{int(inps.shape[0])} | actual_len={actual_length}",
                    flush=True,
                )

        correction = solve_scalar_affine_from_moments(stats)
        corrections[layer_idx] = correction
        summary = summarize_scalar_fit(correction, stats)
        layer_rows.append({"layer": int(layer_idx), "teacher_quant_mse": summary["teacher_quant_mse"]})
        fitting_layers[str(layer_idx)] = {
            "layer": int(layer_idx),
            "teacher_quant_mse": summary["teacher_quant_mse"],
            "active_experts": summary["active_experts"],
            "routed_pairs": summary["routed_pairs"],
            "scalar": summary,
            "fit_backend": "gpu_fp4_full",
        }
        print(
            f"[fit:integrated] layer {layer_idx + 1}/{config.num_hidden_layers} -> mse={summary['teacher_quant_mse']:.8f} | active={summary['active_experts']} | beta_max={summary['beta_abs_max']:.6f}",
            flush=True,
        )

        release_tensors(tensors)
        release_tensors(quant_moe_tensors)
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    release_tensors({"inps": inps, "outs": outs, "causal_mask": causal_mask})
    del store
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ranked_layers = sorted(layer_rows, key=lambda row: (float(row["teacher_quant_mse"]), -int(row["layer"])), reverse=True)
    top10_layers = [int(row["layer"]) for row in ranked_layers[: min(10, len(ranked_layers))]]
    summary_payload = {
        "variant": FIT_KEY,
        "calibration": calibration_set.metadata,
        "fit_quantization": "full_fp4_per_layer",
        "top10_layers_by_local_mse": top10_layers,
        "layer_teacher_quant_mse": ranked_layers,
        "layers": fitting_layers,
    }
    return corrections, top10_layers, summary_payload


def collect_layer_calibration_valid_tokens_with_correction(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    fp4_gate_up: torch.Tensor,
    fp4_down: torch.Tensor,
    config: Any,
    state: Any,
    actual_length: int,
    correction: AffineCorrection | None,
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
    final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=valid_hidden_states.dtype, device=valid_hidden_states.device)

    gate_up_proj = tensors["experts.gate_up_proj"]
    down_proj = tensors["experts.down_proj"]
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)
    state.counts.add_(expert_counts)

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
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
        q_out_w1 = F.linear(q_hidden_w1, down_proj[expert_idx])
        q_out_w2 = F.linear(full_hidden, fp4_down[expert_idx])
        q_out_w1 = apply_affine_correction(q_out_w1, correction, expert_idx).float()
        q_out_w2 = apply_affine_correction(q_out_w2, correction, expert_idx).float()
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


@torch.inference_mode()
def run_maca_calibration_with_correction(
    store: WeightStore,
    config: Any,
    calib_chunks: torch.Tensor,
    actual_lengths: torch.Tensor,
    corrections: dict[int, AffineCorrection],
    device: torch.device,
    dtype: torch.dtype,
) -> CalibrationArtifacts:
    print("\n=== Calibration (MaCa multi-scale + integrated scalar correction) ===", flush=True)
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
        print(f"[maca-int] load layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type})", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors

        gate_up_proj = tensors["mlp.experts.gate_up_proj"]
        down_proj = tensors["mlp.experts.down_proj"]
        fp4_gate_up = torch.stack([quantize_linear_weight(gate_up_proj[expert_idx], "fp4") for expert_idx in range(config.num_experts)], dim=0)
        fp4_down = torch.stack([quantize_linear_weight(down_proj[expert_idx], "fp4") for expert_idx in range(config.num_experts)], dim=0)
        state = init_expert_moment_state(config, device)
        correction = corrections.get(layer_idx)

        for chunk_idx in range(inps.shape[0]):
            actual_length = int(actual_lengths[chunk_idx].item())
            hidden_states = inps[chunk_idx].unsqueeze(0)
            residual, mlp_input = forward_attention_to_mlp_input(
                hidden_states,
                tensors,
                layer_type,
                config,
                position_embeddings,
                causal_mask,
            )
            moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
            moe_out = collect_layer_calibration_valid_tokens_with_correction(
                mlp_input,
                moe_tensors,
                fp4_gate_up,
                fp4_down,
                config,
                state,
                actual_length,
                correction,
            )
            outs[chunk_idx] = residual + moe_out
            if should_log_chunk(chunk_idx, int(inps.shape[0])):
                print(
                    f"[maca-int] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{inps.shape[0]} | actual_len={actual_length}",
                    flush=True,
                )

        bundle, layer_w1, layer_w2, layer_mc = finalize_layer_metrics(gate_up_proj, down_proj, fp4_gate_up, fp4_down, state, config)
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


def print_results_table(payload: dict[str, Any]) -> None:
    results = payload.get("results", {})
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    print("\n" + "=" * 170, flush=True)
    print("proper_iter45_integrated_correction | MaCa integrated correction | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 170, flush=True)
    print(
        f"{'Config':<42} {'PPL':>10} {'Memory GB':>12} {'Params':>12} {'Eval chunks':>12} {'Time s':>10}",
        flush=True,
    )
    print("-" * 170, flush=True)
    for name, row in ordered:
        print(
            f"{name:<42} {float(row['ppl']):>10.4f} {float(row['memory_gb']):>12.3f} {int(row.get('correction_parameter_count', 0)):>12d} {int(row.get('eval_chunks', 0)):>12d} {float(row['time_s']):>10.1f}",
            flush=True,
        )
    print("-" * 170, flush=True)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    requested_plans = resolve_requested_plans(args.plans)
    all_plan_names = {PLAN_ALL, PLAN_TOP10, PLAN_NONE}
    if requested_plans is not None:
        missing = sorted(requested_plans - all_plan_names)
        if missing:
            raise ValueError(f"Unknown plans requested: {', '.join(missing)}")

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    overall_start = time.time()
    tokenizer, _calib_chunks_full, test_ids_full, calib_info, eval_info = load_gptq_standard_data(args.model_id)
    test_ids, eval_slice = maybe_slice_eval_chunks(test_ids_full, int(args.eval_max_chunks))
    eval_info = {**eval_info, **eval_slice}

    full_maca_set = build_maca_calibration_set(tokenizer, args.seed)
    maca_fit_set = slice_maca_set(full_maca_set, int(args.maca_chunks_per_length))

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
            "experiment": "Iteration 45 Integrated Correction",
            "hephaestus_approved": True,
            "hephaestus_rationale": "Estimate correction factors during MaCa calibration, use corrected calibration to rebuild masks, then apply correction during quantization.",
            "standard_calibration_reference": calib_info,
            "maca_calibration": maca_fit_set.metadata,
            "requested_plans": sorted(requested_plans) if requested_plans is not None else sorted(all_plan_names),
        },
        "fitting": {},
        "results": {},
    }
    if args.output_json.exists():
        try:
            with args.output_json.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if isinstance(existing, dict):
                payload["metadata"] = {**existing.get("metadata", {}), **payload["metadata"]}
                payload["fitting"] = {**existing.get("fitting", {}), **payload["fitting"]}
                payload["results"] = {**existing.get("results", {}), **payload["results"]}
                print(f"[resume] loaded existing results from {args.output_json}", flush=True)
        except Exception:
            print(f"[resume] ignoring unreadable existing results at {args.output_json}", flush=True)
    atomic_json_dump(args.output_json, payload)

    corrections, top10_layers, fit_payload = fit_integrated_scalar_corrections(
        args.model_id,
        maca_fit_set,
        text_config,
        snapshot_dir,
        weight_map,
        device,
        dtype,
    )
    payload["fitting"][FIT_KEY] = fit_payload
    atomic_json_dump(args.output_json, payload)

    store = WeightStore(args.model_id, snapshot_dir, weight_map)
    corrected_calibration = run_maca_calibration_with_correction(
        store,
        text_config,
        maca_fit_set.chunks,
        maca_fit_set.actual_lengths,
        corrections,
        device,
        dtype,
    )
    payload["metadata"]["integrated_calibration_summary"] = summarize_calibration_artifacts(corrected_calibration)
    atomic_json_dump(args.output_json, payload)

    maca_plan, maca_extras = build_maca_joint_plan(corrected_calibration, text_config, non_expert_bytes, total_expert_elems)
    base_variant = BaseVariant(
        key=FIT_KEY,
        plan=maca_plan,
        extras={
            **maca_extras,
            "base_variant": FIT_KEY,
            "integration": "scalar_affine_during_calibration",
            "fit_variant": FIT_KEY,
        },
        calibration_set=maca_fit_set,
    )

    evaluation_specs: list[tuple[str, dict[int, AffineCorrection], dict[str, Any]]] = [
        (
            PLAN_ALL,
            corrections,
            {
                **base_variant.extras,
                "correction_type": "scalar_affine",
                "correction_scope": "all_layers",
                "correction_layers": list(range(text_config.num_hidden_layers)),
                "top10_layers_by_local_mse": top10_layers,
            },
        ),
        (
            PLAN_TOP10,
            filter_top_layers(corrections, top10_layers),
            {
                **base_variant.extras,
                "correction_type": "scalar_affine",
                "correction_scope": "top10_local_mse_layers",
                "correction_layers": top10_layers,
                "top10_layers_by_local_mse": top10_layers,
            },
        ),
        (
            PLAN_NONE,
            {},
            {
                **base_variant.extras,
                "correction_type": "none",
                "correction_scope": "none",
                "correction_layers": [],
                "top10_layers_by_local_mse": top10_layers,
            },
        ),
    ]

    for eval_name, eval_corrections, extras in evaluation_specs:
        if requested_plans is not None and eval_name not in requested_plans:
            continue
        evaluate_and_record_compound_plan(
            args.model_id,
            payload,
            eval_name,
            base_variant,
            eval_corrections,
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
    print_results_table(payload)
    print(f"\nSaved results -> {args.output_json}", flush=True)

    del store
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

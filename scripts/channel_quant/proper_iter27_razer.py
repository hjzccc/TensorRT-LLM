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
import torch.nn as nn
import torch.nn.functional as F

from baselines_comparison import (
    estimate_baseline_memory_gb,
    quantize_down_proj_with_mask as quantize_down_proj_with_mask_standard,
    quantize_gate_up_with_pair_mask as quantize_gate_up_with_pair_mask_standard,
    quantize_linear_weight as quantize_linear_weight_standard,
    resolve_non_expert_bytes,
    resolve_terminal_keys,
)
from proper_eval import (
    LOGITS_CHUNK_TOKENS,
    SEQLEN,
    EvalPlan,
    atomic_json_dump,
    build_position_context,
    dtype_from_name,
    embed_chunks,
    layer_type_at,
    load_gptq_standard_data,
    run_calibration,
)
from proper_iter01 import build_mxmoe_topup_masks, build_plan_from_masks, total_channel_fraction, total_pair_fraction
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from spike1_ground_truth import (
    EPS,
    MODEL_ID,
    WeightStore,
    build_text_config,
    full_attention_forward,
    layer_keys,
    linear_attention_forward,
    load_root_config,
    move_tensor,
    quantize_to_fp8,
    release_tensors,
    rms_norm_qwen3_next,
    round_to_e2m1_grid,
    shorten_layer_tensors,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "proper_iter27_razer.json"
DEFAULT_EXPLORATION_MD = SCRIPT_DIR / "exploration.md"
SECTION_MARKER = "## [30] Iteration 27 - RaZeR NVFP4"
MXMOE_TOPUP_FRACTION = 0.05
SUCCESS_TARGETS = {
    "joint_w1w2_with_topup": 6.5725,
    "mxmoe_block_plus_channel_topup": 6.5763,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--exploration-md", type=Path, default=DEFAULT_EXPLORATION_MD)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16"])
    parser.add_argument(
        "--plans",
        default="",
        help="Comma-separated subset of plan names to evaluate; default runs all Iteration 27 plans.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_requested_plans(raw_value: str) -> set[str] | None:
    names = [part.strip() for part in raw_value.split(",") if part.strip()]
    return None if not names else set(names)


def load_reference_rows(output_json: Path) -> dict[str, dict[str, float]]:
    references: dict[str, dict[str, float]] = {}
    for path in sorted(RESULTS_DIR.glob("proper*.json")):
        if path == output_json or not path.exists():
            continue
        try:
            payload = load_json(path)
        except Exception:
            continue
        for name, row in payload.get("results", {}).items():
            if isinstance(row, dict) and "ppl" in row and "memory_gb" in row:
                references[str(name)] = {
                    "ppl": float(row["ppl"]),
                    "memory_gb": float(row["memory_gb"]),
                }
    return references


def upsert_exploration_section(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if SECTION_MARKER in existing:
        before, _marker, _rest = existing.partition(SECTION_MARKER)
        updated = before.rstrip() + "\n\n" + content.strip() + "\n"
    else:
        prefix = existing.rstrip()
        updated = prefix + "\n\n" + content.strip() + "\n" if prefix else content.strip() + "\n"
    path.write_text(updated, encoding="utf-8")


def summarize_calibration(routing_counts: dict[int, torch.Tensor]) -> dict[str, Any]:
    return {
        "active_experts_per_layer": {
            str(layer_idx): int((counts > 0).sum().item()) for layer_idx, counts in routing_counts.items()
        },
        "routing_top5_per_layer": {
            str(layer_idx): [
                {"expert_idx": int(expert_idx), "count": int(count)}
                for count, expert_idx in zip(
                    torch.topk(counts, k=min(5, counts.numel())).values.tolist(),
                    torch.topk(counts, k=min(5, counts.numel())).indices.tolist(),
                )
            ]
            for layer_idx, counts in routing_counts.items()
        },
    }


def quantize_to_nvfp4_razer(weight: torch.Tensor, groupsize: int = 16) -> torch.Tensor:
    if groupsize <= 0:
        raise ValueError(f"groupsize must be positive, got {groupsize}")
    if weight.ndim == 1:
        weight = weight.unsqueeze(-1)
        squeeze = True
    else:
        squeeze = False
    if weight.shape[0] % groupsize != 0:
        raise ValueError(f"Expected K dimension multiple of {groupsize}, got {tuple(weight.shape)}")

    weight_f = weight.to(torch.float32)
    blocks = weight_f.reshape(weight.shape[0] // groupsize, groupsize, weight.shape[1])
    absmax = blocks.abs().amax(dim=1, keepdim=True)
    scale = absmax / 6.0
    normalized = blocks / (scale + EPS)
    fp4_quant = round_to_e2m1_grid(normalized).to(torch.float32)

    pos_special = torch.full_like(normalized, 5.0)
    neg_special = torch.full_like(normalized, -5.0)

    pos_candidate = torch.where((normalized - pos_special).abs() < (normalized - fp4_quant).abs(), pos_special, fp4_quant)
    neg_candidate = torch.where((normalized - neg_special).abs() < (normalized - fp4_quant).abs(), neg_special, fp4_quant)

    pos_error = (normalized - pos_candidate).square().sum(dim=1, keepdim=True)
    neg_error = (normalized - neg_candidate).square().sum(dim=1, keepdim=True)
    chosen = torch.where(pos_error <= neg_error, pos_candidate, neg_candidate)

    dequantized = (chosen * scale).reshape_as(weight_f).to(torch.bfloat16)
    return dequantized.squeeze(-1) if squeeze else dequantized


def quantize_linear_weight_razer(weight: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "bf16":
        return weight
    weight_t = weight.transpose(0, 1).contiguous()
    if mode == "fp8":
        quantized = quantize_to_fp8(weight_t)
    elif mode == "fp4":
        quantized = quantize_to_nvfp4_razer(weight_t)
    else:
        raise ValueError(f"Unsupported quantization mode: {mode}")
    return quantized.transpose(0, 1).contiguous()


def quantize_gate_up_with_pair_mask_razer(weight: torch.Tensor, fp8_pair_mask: torch.Tensor) -> torch.Tensor:
    if int(fp8_pair_mask.numel() * 2) != int(weight.shape[0]):
        raise ValueError(f"Expected pair mask length {weight.shape[0] // 2}, got {fp8_pair_mask.numel()}")
    if not bool(fp8_pair_mask.any()):
        return quantize_linear_weight_razer(weight, "fp4")
    if bool(fp8_pair_mask.all()):
        return quantize_linear_weight_standard(weight, "fp8")

    weight_t = weight.transpose(0, 1).contiguous()
    fp4_t = quantize_to_nvfp4_razer(weight_t)
    fp8_t = quantize_to_fp8(weight_t)
    row_mask = torch.cat([fp8_pair_mask, fp8_pair_mask], dim=0).to(device=weight_t.device, dtype=torch.bool)
    mixed_t = torch.where(row_mask.view(1, -1), fp8_t, fp4_t)
    return mixed_t.transpose(0, 1).contiguous()


def quantize_down_proj_with_mask_razer(weight: torch.Tensor, fp8_mask: torch.Tensor) -> torch.Tensor:
    if not bool(fp8_mask.any()):
        return quantize_linear_weight_razer(weight, "fp4")
    if bool(fp8_mask.all()):
        return quantize_linear_weight_standard(weight, "fp8")

    weight_t = weight.transpose(0, 1).contiguous()
    fp4_t = quantize_to_nvfp4_razer(weight_t)
    fp8_t = quantize_to_fp8(weight_t)
    column_mask = fp8_mask.to(device=weight_t.device, dtype=torch.bool).view(1, -1)
    mixed_t = torch.where(column_mask, fp8_t, fp4_t)
    return mixed_t.transpose(0, 1).contiguous()


def plan_uses_razer(plan: EvalPlan) -> bool:
    return plan.name.startswith("razer_")


def quantize_linear_weight_for_plan(weight: torch.Tensor, mode: str, use_razer: bool) -> torch.Tensor:
    if mode == "fp4" and use_razer:
        return quantize_linear_weight_razer(weight, mode)
    return quantize_linear_weight_standard(weight, mode)


def prepare_layer_tensors_for_plan(plan: EvalPlan, layer_idx: int, tensors: dict[str, torch.Tensor], config: Any) -> None:
    if plan.mode == "bf16":
        return

    use_razer = plan_uses_razer(plan)
    gate_up_proj = tensors["mlp.experts.gate_up_proj"]
    down_proj = tensors["mlp.experts.down_proj"]
    prepared_gate_up: list[torch.Tensor] = []
    prepared_down: list[torch.Tensor] = []

    for expert_idx in range(config.num_experts):
        gate_up_weight = gate_up_proj[expert_idx]
        down_weight = down_proj[expert_idx]
        if plan.mode == "uniform_fp4":
            quant_gate_up = quantize_linear_weight_for_plan(gate_up_weight, "fp4", use_razer)
            quant_down = quantize_linear_weight_for_plan(down_weight, "fp4", use_razer)
        elif plan.mode == "uniform_fp8":
            quant_gate_up = quantize_linear_weight_standard(gate_up_weight, "fp8")
            quant_down = quantize_linear_weight_standard(down_weight, "fp8")
        elif plan.mode == "per_expert":
            layer_promoted = set() if plan.promoted_experts is None else (plan.promoted_experts.get(layer_idx) or set())
            precision = "fp8" if expert_idx in layer_promoted else "fp4"
            quant_gate_up = quantize_linear_weight_for_plan(gate_up_weight, precision, use_razer)
            quant_down = quantize_linear_weight_for_plan(down_weight, precision, use_razer)
        elif plan.mode == "per_block":
            w1_mode = "fp8" if plan.w1_projection_fp8 is not None and bool(plan.w1_projection_fp8[layer_idx][expert_idx]) else "fp4"
            w2_mode = "fp8" if plan.w2_projection_fp8 is not None and bool(plan.w2_projection_fp8[layer_idx][expert_idx]) else "fp4"
            quant_gate_up = quantize_linear_weight_for_plan(gate_up_weight, w1_mode, use_razer)
            quant_down = quantize_linear_weight_for_plan(down_weight, w2_mode, use_razer)
        elif plan.mode == "per_channel":
            if plan.w1_pair_masks is None or plan.w2_channel_masks is None:
                raise ValueError(f"Plan {plan.name} missing per-channel masks")
            w1_mask = plan.w1_pair_masks[layer_idx][expert_idx].to(device=gate_up_weight.device)
            w2_mask = plan.w2_channel_masks[layer_idx][expert_idx].to(device=down_weight.device)
            if use_razer:
                quant_gate_up = quantize_gate_up_with_pair_mask_razer(gate_up_weight, w1_mask)
                quant_down = quantize_down_proj_with_mask_razer(down_weight, w2_mask)
            else:
                quant_gate_up = quantize_gate_up_with_pair_mask_standard(gate_up_weight, w1_mask)
                quant_down = quantize_down_proj_with_mask_standard(down_weight, w2_mask)
        else:
            raise ValueError(f"Unsupported plan mode: {plan.mode}")
        prepared_gate_up.append(quant_gate_up)
        prepared_down.append(quant_down)

    tensors["mlp.experts.gate_up_proj"] = torch.stack(prepared_gate_up, dim=0)
    tensors["mlp.experts.down_proj"] = torch.stack(prepared_down, dim=0)
    del gate_up_proj, down_proj, prepared_gate_up, prepared_down
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def moe_forward_eval(hidden_states: torch.Tensor, tensors: dict[str, torch.Tensor], config: Any) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    router_logits = F.linear(flat, tensors["gate.weight"]).float()
    routing_weights = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(routing_weights, config.num_experts_per_tok, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
    expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=config.num_experts)

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]
        gate_up = F.linear(current_state, tensors["experts.gate_up_proj"][expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        current_hidden = F.linear(F.silu(gate) * up, tensors["experts.down_proj"][expert_idx])
        final_hidden_states.index_add_(0, token_idx, (routing_weights[token_idx, route_pos].unsqueeze(-1) * current_hidden).to(hidden_states.dtype))

    shared_gate = F.linear(flat, tensors["shared_expert.gate_proj.weight"])
    shared_up = F.linear(flat, tensors["shared_expert.up_proj.weight"])
    shared_out = F.linear(F.silu(shared_gate) * shared_up, tensors["shared_expert.down_proj.weight"])
    shared_gate_value = torch.sigmoid(F.linear(flat, tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_out * shared_gate_value
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


@torch.inference_mode()
def evaluate_plan(
    model_id: str,
    plan: EvalPlan,
    test_ids: torch.Tensor,
    config: Any,
    weight_map: dict[str, str],
    snapshot_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float, int]:
    embed_key, norm_key, lm_head_key = resolve_terminal_keys(weight_map)
    nsamples = int(test_ids.numel() // SEQLEN)
    eval_chunks = test_ids[:, : nsamples * SEQLEN].view(nsamples, SEQLEN)
    store = WeightStore(model_id, snapshot_dir, weight_map)

    root_tensors = store.load_tensors([norm_key, lm_head_key])
    final_norm = move_tensor(root_tensors[norm_key], device, dtype)
    lm_head = move_tensor(root_tensors[lm_head_key], device, dtype)
    del root_tensors

    inps = embed_chunks(store, embed_key, eval_chunks, device, dtype)
    outs = torch.zeros_like(inps)
    causal_mask, position_embeddings = build_position_context(config, inps[0:1], device)

    for layer_idx in range(config.num_hidden_layers):
        layer_type = layer_type_at(config, layer_idx)
        print(f"[{plan.name}] load layer {layer_idx + 1}/{config.num_hidden_layers} ({layer_type})", flush=True)
        raw_tensors = store.load_tensors(layer_keys(layer_idx, layer_type))
        tensors = shorten_layer_tensors(layer_idx, raw_tensors, device, dtype)
        del raw_tensors
        prepare_layer_tensors_for_plan(plan, layer_idx, tensors, config)

        for chunk_idx in range(nsamples):
            hidden_states = inps[chunk_idx].unsqueeze(0)
            residual = hidden_states
            hidden_norm = rms_norm_qwen3_next(hidden_states, tensors["input_layernorm.weight"], config.rms_norm_eps)
            if layer_type == "full_attention":
                attn_tensors = {k.replace("self_attn.", ""): v for k, v in tensors.items() if k.startswith("self_attn.")}
                mixed = full_attention_forward(hidden_norm, attn_tensors, config, position_embeddings, causal_mask)
            else:
                attn_tensors = {k.replace("linear_attn.", ""): v for k, v in tensors.items() if k.startswith("linear_attn.")}
                mixed = linear_attention_forward(hidden_norm, attn_tensors, config)
            hidden_states = residual + mixed

            residual = hidden_states
            mlp_input = rms_norm_qwen3_next(hidden_states, tensors["post_attention_layernorm.weight"], config.rms_norm_eps)
            moe_tensors = {k.replace("mlp.", "", 1): v for k, v in tensors.items() if k.startswith("mlp.")}
            moe_out = moe_forward_eval(mlp_input, moe_tensors, config)
            outs[chunk_idx] = residual + moe_out
            print(f"[{plan.name}] layer {layer_idx + 1}/{config.num_hidden_layers} chunk {chunk_idx + 1}/{nsamples}", flush=True)

        release_tensors(tensors)
        inps, outs = outs, inps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_ids_device = test_ids.to(device=device, non_blocking=True)
    loss_fct = nn.CrossEntropyLoss(reduction="sum")
    nlls: list[torch.Tensor] = []
    for chunk_idx in range(nsamples):
        hidden_states = inps[chunk_idx].unsqueeze(0)
        hidden_states = rms_norm_qwen3_next(hidden_states, final_norm, config.rms_norm_eps)
        shift_labels = test_ids_device[:, chunk_idx * SEQLEN : (chunk_idx + 1) * SEQLEN][:, 1:]
        shift_tokens = int(shift_labels.numel())
        chunk_loss_sum = torch.zeros((), dtype=torch.float64, device=hidden_states.device)
        for start in range(0, shift_tokens, LOGITS_CHUNK_TOKENS):
            end = min(start + LOGITS_CHUNK_TOKENS, shift_tokens)
            logits = F.linear(hidden_states[:, start:end, :].float(), lm_head.float())
            loss = loss_fct(logits.reshape(-1, logits.size(-1)), shift_labels[:, start:end].reshape(-1))
            chunk_loss_sum = chunk_loss_sum + loss.to(torch.float64)
            del logits, loss
        nlls.append((chunk_loss_sum / float(max(shift_tokens, 1))).to(torch.float32) * SEQLEN)
        print(f"[{plan.name}] logits chunk {chunk_idx + 1}/{nsamples}", flush=True)

    mean_nll = torch.stack(nlls).sum() / (nsamples * SEQLEN)
    ppl = torch.exp(mean_nll)

    release_tensors(
        {
            "inps": inps,
            "outs": outs,
            "final_norm": final_norm,
            "lm_head": lm_head,
            "causal_mask": causal_mask,
            "test_ids_device": test_ids_device,
        }
    )
    del store
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(ppl.item()), float(mean_nll.item()), nsamples


def evaluate_and_record_plan(
    output_payload: dict[str, Any],
    model_id: str,
    plan: EvalPlan,
    extras: dict[str, Any],
    output_json: Path,
    config: Any,
    total_expert_elems: int,
    test_ids: torch.Tensor,
    snapshot_dir: Path,
    weight_map: dict[str, str],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    result_rows = output_payload.setdefault("results", {})
    if plan.name in result_rows:
        print(f"[skip] {plan.name} already present", flush=True)
        return result_rows[plan.name]

    print(f"\n=== Eval: {plan.name} ===", flush=True)
    start_time = time.time()
    ppl, nll, nsamples = evaluate_plan(model_id, plan, test_ids, config, weight_map, snapshot_dir, device, dtype)
    elapsed = time.time() - start_time
    row = {
        "description": plan.description,
        "mode": plan.mode,
        "ppl": round(ppl, 6),
        "nll": round(nll, 6),
        "memory_gb": round(plan.memory_gb, 3),
        "fp8_weights": int(plan.fp8_weights),
        "fp8_fraction": round(float(plan.fp8_weights) / float(max(total_expert_elems, 1)), 6),
        "w1_pair_fraction": round(total_pair_fraction(config, plan.w1_pair_masks or {}), 6),
        "w2_channel_fraction": round(total_channel_fraction(config, plan.w2_channel_masks or {}), 6),
        "eval_chunks": int(nsamples),
        "seqlen": SEQLEN,
        "time_s": round(elapsed, 1),
        **extras,
    }
    result_rows[plan.name] = row
    atomic_json_dump(output_json, output_payload)
    print(
        f"[{plan.name}] done -> PPL={ppl:.4f} | NLL={nll:.6f} | memory={plan.memory_gb:.3f} GB | fp8={row['fp8_fraction']:.4f} | time={elapsed:.1f}s",
        flush=True,
    )
    return row


def build_iteration_plans(
    calibration: Any,
    config: Any,
    total_bf16_bytes: int,
    non_expert_bytes: int,
    total_expert_elems: int,
) -> list[tuple[EvalPlan, dict[str, Any]]]:
    joint_w1_masks, joint_w2_masks, joint_meta = build_joint_with_topup_masks(
        calibration,
        calibration.activation_cache,
        config,
        JOINT_MEDIUM_TOPUP_FRACTION,
    )
    mxmoe_w1_masks, mxmoe_w2_masks, mxmoe_meta = build_mxmoe_topup_masks(
        calibration,
        config,
        total_expert_elems,
        MXMOE_TOPUP_FRACTION,
    )

    plans: list[tuple[EvalPlan, dict[str, Any]]] = [
        (
            build_plan_from_masks(
                "razer_joint_w1w2_topup",
                "Reuse the current best joint three-tier plus 8% within-expert topup mask, but quantize every FP4 channel with the RaZeR NVFP4 element assignment.",
                config,
                non_expert_bytes,
                total_expert_elems,
                joint_w1_masks,
                joint_w2_masks,
            ),
            {
                **joint_meta,
                "source_base": "joint_w1w2_with_topup",
                "fp4_quantization": "razer_nvfp4_e4m3_plusminus5",
            },
        ),
        (
            build_plan_from_masks(
                "razer_mxmoe_topup_5pct",
                "Reuse the MxMoE block-plus-channel topup mask, but quantize every remaining FP4 channel with the RaZeR NVFP4 element assignment.",
                config,
                non_expert_bytes,
                total_expert_elems,
                mxmoe_w1_masks,
                mxmoe_w2_masks,
            ),
            {
                **mxmoe_meta,
                "source_base": "mxmoe_block_plus_channel_topup",
                "fp4_quantization": "razer_nvfp4_e4m3_plusminus5",
            },
        ),
        (
            EvalPlan(
                name="razer_uniform_fp4",
                description="Quantize all expert weights with uniform RaZeR NVFP4 and evaluate the dequantized BF16 weights with F.linear.",
                mode="uniform_fp4",
                memory_gb=estimate_baseline_memory_gb(total_bf16_bytes, non_expert_bytes, total_expert_elems, "fp4"),
                fp8_weights=0,
            ),
            {
                "source_base": "uniform_fp4",
                "fp4_quantization": "razer_nvfp4_e4m3_plusminus5",
            },
        ),
        (
            build_plan_from_masks(
                "standard_joint_w1w2_topup",
                "Rebuild the current best joint three-tier plus 8% within-expert topup mask with the unchanged standard NVFP4 path as a control.",
                config,
                non_expert_bytes,
                total_expert_elems,
                joint_w1_masks,
                joint_w2_masks,
            ),
            {
                **joint_meta,
                "source_base": "joint_w1w2_with_topup",
                "fp4_quantization": "standard_nvfp4",
                "expected_reference_ppl": SUCCESS_TARGETS["joint_w1w2_with_topup"],
            },
        ),
    ]
    return plans


def format_result_table(results: dict[str, Any]) -> str:
    lines = [
        "| Config | PPL | Memory (GB) | FP8 frac | W1 frac | W2 frac |",
        "|--------|-----|-------------|----------|---------|---------|",
    ]
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    for name, row in ordered:
        lines.append(
            f"| {name} | {float(row['ppl']):.4f} | {float(row['memory_gb']):.3f} | {float(row['fp8_fraction']):.4f} | {float(row['w1_pair_fraction']):.4f} | {float(row['w2_channel_fraction']):.4f} |"
        )
    return "\n".join(lines)


def build_insight_text(results: dict[str, Any], references: dict[str, dict[str, float]]) -> str:
    notes: list[str] = []
    standard_joint = results.get("standard_joint_w1w2_topup")
    razer_joint = results.get("razer_joint_w1w2_topup")
    razer_mxmoe = results.get("razer_mxmoe_topup_5pct")

    if standard_joint is not None and razer_joint is not None:
        delta = float(razer_joint["ppl"]) - float(standard_joint["ppl"])
        if delta < 0.0:
            notes.append(f"RaZeR improves the joint mask by {abs(delta):.4f} PPL versus the rebuilt standard control.")
        elif delta > 0.0:
            notes.append(f"RaZeR trails the rebuilt standard joint control by {delta:.4f} PPL.")
        else:
            notes.append("RaZeR exactly matches the rebuilt standard joint control.")

    historical_joint = references.get("joint_w1w2_with_topup")
    if historical_joint is not None and standard_joint is not None:
        drift = float(standard_joint["ppl"]) - float(historical_joint["ppl"])
        notes.append(f"The rebuilt standard joint control differs from the historical 6.5725 target by {drift:+.4f} PPL.")

    historical_mxmoe = references.get("mxmoe_block_plus_channel_topup")
    if historical_mxmoe is not None and razer_mxmoe is not None:
        delta = float(razer_mxmoe["ppl"]) - float(historical_mxmoe["ppl"])
        notes.append(f"The RaZeR MxMoE row moves {delta:+.4f} PPL relative to the historical MxMoE topup baseline.")

    if not notes:
        winner_name, winner = min(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
        return f"`{winner_name}` is currently the best row at {float(winner['ppl']):.4f}."
    return " ".join(notes)


def render_exploration_section(results: dict[str, Any], eval_info: dict[str, Any], references: dict[str, dict[str, float]]) -> str:
    table = format_result_table(results)
    return (
        f"{SECTION_MARKER}\n"
        f"**Eval**: Full WikiText-2 test ({eval_info['total_tokens']} tokens, {eval_info['nsamples']} chunks of {SEQLEN}), BF16 F.linear, FP32 loss.\n"
        f"**Approach**: Reused the standard GPTQ-style calibration and the existing strongest per-channel masks, then swapped only the FP4 expert-weight assignment so each 16-value NVFP4 block may repurpose the reclaimed special code as either +5 or -5 before dequantizing back to BF16.\n"
        f"**Result**:\n{table}\n"
        f"**Insight**: {build_insight_text(results, references)}\n"
        f"**Next**: If the joint RaZeR row wins, repeat the same swap on the next-best routed masks before trying any new calibration or routing changes."
    )


def print_results_table(results: dict[str, Any]) -> None:
    ordered = sorted(results.items(), key=lambda item: (float(item[1]["ppl"]), float(item[1]["memory_gb"]), item[0]))
    print("\n" + "=" * 182, flush=True)
    print("proper_iter27_razer | RaZeR NVFP4 swap on strongest masks | full WikiText-2 | GPTQ-standard eval", flush=True)
    print("=" * 182, flush=True)
    print(
        f"{'Config':<32} {'PPL':>10} {'Memory GB':>12} {'FP8 frac':>10} {'W1 frac':>10} {'W2 frac':>10} {'Time s':>10}",
        flush=True,
    )
    print("-" * 182, flush=True)
    for name, row in ordered:
        print(
            f"{name:<32} {float(row['ppl']):>10.4f} {float(row['memory_gb']):>12.3f} {float(row['fp8_fraction']):>10.4f} {float(row['w1_pair_fraction']):>10.4f} {float(row['w2_channel_fraction']):>10.4f} {float(row['time_s']):>10.1f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    requested_plans = resolve_requested_plans(args.plans)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    overall_start = time.time()
    tokenizer, calib_chunks, test_ids, calib_info, eval_info = load_gptq_standard_data(args.model_id)
    del tokenizer

    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)
    if text_config.layer_types is None:
        raise RuntimeError("Qwen3NextConfig.layer_types is required for proper evaluation")
    index_path = snapshot_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index_payload = json.load(handle)
    total_bf16_bytes = int(root_config.get("total_size", 0) or index_payload["metadata"]["total_size"])
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)

    output_payload: dict[str, Any] = {
        "metadata": {
            "model": args.model_id,
            "device": args.device,
            "dtype": args.dtype,
            "calibration": calib_info,
            "evaluation": eval_info,
            "quantization": "simulated quantization: FP8 uses the unchanged standard path; FP4 uses standard NVFP4 or blockwise RaZeR NVFP4 (+5/-5 special replacement) before BF16 F.linear; FP32 logits and loss",
            "protocol": {
                "reference": "/home/jerry/Documents/fork_new/MC-MoE/eval_ppl_utils.py",
                "full_test_join": True,
                "seqlen": SEQLEN,
                "non_overlapping": True,
                "loss_accumulation": "loss.float() * seqlen",
            },
            "experiment": "Iteration 27 RaZeR NVFP4 swap on strongest masks",
            "success_targets": SUCCESS_TARGETS,
            "razer_variant": {
                "format": "E4M3-compatible reassignment with unchanged NVFP4 block scales",
                "groupsize": 16,
                "special_values": [-5.0, 5.0],
                "selection_rule": "per block, choose the sign whose per-element best-of{standard FP4, special value} reconstruction gives lower squared error",
            },
        },
        "results": {},
    }
    if args.output_json.exists():
        try:
            existing = load_json(args.output_json)
            if isinstance(existing, dict):
                merged_metadata = dict(output_payload["metadata"])
                merged_metadata.update(existing.get("metadata", {}))
                output_payload.update(existing)
                output_payload["metadata"] = merged_metadata
                output_payload.setdefault("results", {})
                print(f"[resume] loaded existing results from {args.output_json}", flush=True)
        except Exception:
            print(f"[resume] ignoring unreadable existing results at {args.output_json}", flush=True)
    atomic_json_dump(args.output_json, output_payload)

    store = WeightStore(args.model_id, snapshot_dir, weight_map)
    calibration = run_calibration(store, text_config, calib_chunks, device, dtype)
    output_payload["metadata"]["calibration_summary"] = summarize_calibration(calibration.routing_counts)
    atomic_json_dump(args.output_json, output_payload)

    plans = build_iteration_plans(calibration, text_config, total_bf16_bytes, non_expert_bytes, total_expert_elems)
    if requested_plans is not None:
        plans = [(plan, extras) for plan, extras in plans if plan.name in requested_plans]
        missing = sorted(requested_plans.difference({plan.name for plan, _extras in plans}))
        if missing:
            raise ValueError(f"Unknown plan names requested: {', '.join(missing)}")

    for idx, (plan, extras) in enumerate(plans, start=1):
        print(f"\n=== Eval {idx}/{len(plans)}: {plan.name} ===", flush=True)
        evaluate_and_record_plan(
            output_payload,
            args.model_id,
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

    references = load_reference_rows(args.output_json)
    section = render_exploration_section(output_payload["results"], eval_info, references)
    upsert_exploration_section(args.exploration_md, section)
    output_payload["metadata"]["references"] = references
    output_payload["metadata"]["runtime_seconds"] = round(time.time() - overall_start, 3)
    atomic_json_dump(args.output_json, output_payload)
    print_results_table(output_payload["results"])
    print(f"\nSaved results -> {args.output_json}", flush=True)
    print(f"Updated exploration -> {args.exploration_md}", flush=True)


if __name__ == "__main__":
    main()

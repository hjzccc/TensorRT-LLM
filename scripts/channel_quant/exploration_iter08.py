#!/usr/bin/env python3
# pyright: basic, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false
"""Iteration 6: Fundamentally different approaches to break the 5.329 plateau."""
from __future__ import annotations

import argparse, gc, json, math, sys, time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from spike1_ground_truth import (
    MODEL_ID, WeightStore, build_causal_mask, build_text_config,
    full_attention_forward, linear_attention_forward,
    load_root_config, move_tensor, quantize_to_fp8,
    quantize_to_nvfp4_columns, release_tensors, rms_norm_qwen3_next,
    layer_keys, shorten_layer_tensors,
)
from baselines_comparison import (
    EvaluationPlan, evaluate_plan,
    load_prefix_dataset_tokens, resolve_terminal_keys,
    estimate_mixed_memory_gb, fp8_weights_from_masks,
    quantize_linear_weight, topk_mask_from_scores,
    build_two_level_masks,
    expert_full_weight_count, dtype_from_name,
    LayerCapture, LayerMetricBundle,
)


def make_plan(name, desc, config, non_expert_bytes, total_expert_elems, w1_masks, w2_masks):
    fp8_w = fp8_weights_from_masks(config, w1_masks, w2_masks)
    mem = estimate_mixed_memory_gb(non_expert_bytes, total_expert_elems, fp8_w)
    plan = EvaluationPlan(
        name=name, description=desc, mode="mixed_channel",
        memory_gb=mem, fp8_weights=fp8_w,
        w1_pair_masks=w1_masks, w2_channel_masks=w2_masks,
    )
    plan.fp8_fraction = fp8_w / max(total_expert_elems, 1)
    return plan


def build_topk_masks(metric_cache, config, w1_frac, w2_frac):
    w1m: dict[int, dict[int, torch.Tensor]] = {}
    w2m: dict[int, dict[int, torch.Tensor]] = {}
    k_w1 = int(round(w1_frac * config.moe_intermediate_size))
    k_w2 = int(round(w2_frac * config.hidden_size))
    for lidx, bundle in metric_cache.items():
        layer_w1, layer_w2 = {}, {}
        total_count = float(bundle.routing_counts.sum().item())
        for eidx in range(config.num_experts):
            freq = float(bundle.routing_counts[eidx].item()) / max(total_count, 1.0)
            expert_w1_k = min(config.moe_intermediate_size, max(0, int(round(k_w1 * freq * config.num_experts))))
            expert_w2_k = min(config.hidden_size, max(0, int(round(k_w2 * freq * config.num_experts))))
            layer_w1[eidx] = topk_mask_from_scores(bundle.w1_pair_scores[eidx], expert_w1_k)
            layer_w2[eidx] = topk_mask_from_scores(bundle.w2_channel_scores[eidx], expert_w2_k)
        w1m[lidx] = layer_w1
        w2m[lidx] = layer_w2
    return w1m, w2m
from transformers import AutoTokenizer
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding


def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    EVAL_TOKENS = 512
    CALIB_TOKENS = 128

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    calib_ids, _ = load_prefix_dataset_tokens(tokenizer, "train", CALIB_TOKENS)
    eval_ids, _ = load_prefix_dataset_tokens(tokenizer, "test", EVAL_TOKENS)
    print(f"Calib: {calib_ids.shape[1]} tokens | Eval: {eval_ids.shape[1]} tokens")

    snapshot_dir, root_config, weight_map = load_root_config(MODEL_ID)
    text_config = build_text_config(root_config)
    all_keys = list(weight_map.keys())

    total_bf16_bytes = sum(2 * np.prod(list(v)) for v in
        json.load(open(snapshot_dir / "model.safetensors.index.json"))
        .get("metadata", {}).get("tensor_shapes", {}).values()) if False else 0
    total_expert_elems = text_config.num_experts * text_config.num_hidden_layers * (
        2 * text_config.moe_intermediate_size * text_config.hidden_size +
        text_config.hidden_size * text_config.moe_intermediate_size
    )
    non_expert_bytes = 0  # Will compute from plan

    # === CALIBRATION ===
    print("\n=== Calibration ===", flush=True)
    store = WeightStore(MODEL_ID, snapshot_dir, weight_map)
    metric_cache: dict[int, LayerMetricBundle] = {}
    captures: dict[int, LayerCapture] = {}

    embed_key, norm_key, lm_head_key = resolve_terminal_keys(weight_map)
    root_t = store.load_tensors([embed_key])
    h = F.embedding(calib_ids.to(device), move_tensor(root_t[embed_key], device, dtype))
    del root_t
    mask = build_causal_mask(calib_ids.shape[1], device)
    pos_ids = torch.arange(calib_ids.shape[1], device=device).unsqueeze(0)
    rotary = Qwen3NextRotaryEmbedding(config=text_config, device=device)
    pos_emb = rotary(h, pos_ids)

    with torch.inference_mode():
        for lidx in range(text_config.num_hidden_layers):
            lt = text_config.layer_types[lidx]
            raw = store.load_tensors(layer_keys(lidx, lt))
            s = shorten_layer_tensors(lidx, raw, device, dtype)
            del raw

            res = h
            h = rms_norm_qwen3_next(h, s["input_layernorm.weight"], text_config.rms_norm_eps)
            if lt == "full_attention":
                attn_t = {k.replace("self_attn.", ""): v for k, v in s.items() if k.startswith("self_attn.")}
                h = full_attention_forward(h, attn_t, text_config, pos_emb, mask)
            else:
                attn_t = {k.replace("linear_attn.", ""): v for k, v in s.items() if k.startswith("linear_attn.")}
                h = linear_attention_forward(h, attn_t, text_config)
            h = res + h
            res = h
            h = rms_norm_qwen3_next(h, s["post_attention_layernorm.weight"], text_config.rms_norm_eps)

            flat = h.view(-1, text_config.hidden_size)
            moe = {k.replace("mlp.", "", 1): v for k, v in s.items() if k.startswith("mlp.")}
            rl = F.linear(flat, moe["gate.weight"]).float()
            rp = torch.softmax(rl, dim=1)
            rw, se = torch.topk(rp, text_config.num_experts_per_tok, dim=-1)
            rw = (rw / rw.sum(dim=-1, keepdim=True)).to(dtype)
            ec = torch.bincount(se.reshape(-1), minlength=text_config.num_experts)

            gu_w = moe["experts.gate_up_proj"]
            dw = moe["experts.down_proj"]

            # Compute metrics per expert
            w1_pair_scores: dict[int, torch.Tensor] = {}
            w2_channel_scores: dict[int, torch.Tensor] = {}
            routing_counts = ec.clone()

            for eidx in range(text_config.num_experts):
                ti, rp_idx = torch.where(se == eidx)
                n_tok = len(ti)

                # W1 scores (activation_kurtosis)
                w1 = gu_w[eidx]
                w1_q = quantize_linear_weight(w1, "fp4")
                w1_err = (w1 - w1_q).abs()

                if n_tok > 0:
                    act = flat[ti]
                    hess_diag = act.float().pow(2).mean(0)
                    pair_scores = (hess_diag.unsqueeze(0) * w1_err.float().pow(2)).sum(dim=1)
                    w1_pair_scores[eidx] = (pair_scores[0::2] + pair_scores[1::2]).cpu()

                    w2 = dw[eidx]
                    w2_q = quantize_linear_weight(w2, "fp4")
                    w2_err = (w2 - w2_q).abs()
                    gu_out = F.linear(act, w1)
                    g, u = gu_out.chunk(2, dim=-1)
                    silu_out = F.silu(g) * u
                    hess_diag_w2 = silu_out.float().pow(2).mean(0)
                    w2_channel_scores[eidx] = (hess_diag_w2.unsqueeze(0) * w2_err.float().pow(2)).sum(dim=1).cpu()
                else:
                    w1_pair_scores[eidx] = w1_err.sum(dim=1).cpu()
                    w1_pair_scores[eidx] = (w1_pair_scores[eidx][0::2] + w1_pair_scores[eidx][1::2])
                    w2 = dw[eidx]
                    w2_q = quantize_linear_weight(w2, "fp4")
                    w2_err = (w2 - w2_q).abs()
                    w2_channel_scores[eidx] = w2_err.sum(dim=1).cpu()

            metric_cache[lidx] = LayerMetricBundle(
                w1_pair_scores=w1_pair_scores,
                w2_channel_scores=w2_channel_scores,
                routing_counts=routing_counts.cpu(),
            )

            # BF16 MoE forward
            moe_out = torch.zeros_like(flat)
            for eidx in torch.nonzero(ec > 0, as_tuple=False).flatten().tolist():
                ti, rp_idx = torch.where(se == eidx)
                cur = flat[ti]
                gu_out = F.linear(cur, gu_w[eidx])
                g, u = gu_out.chunk(2, dim=-1)
                out = F.linear(F.silu(g) * u, dw[eidx])
                moe_out.index_add_(0, ti, (rw[ti, rp_idx].unsqueeze(-1) * out).to(dtype))

            sg = F.linear(flat, moe["shared_expert.gate_proj.weight"])
            su = F.linear(flat, moe["shared_expert.up_proj.weight"])
            so = F.linear(F.silu(sg) * su, moe["shared_expert.down_proj.weight"])
            sgv = torch.sigmoid(F.linear(flat, moe["shared_expert_gate.weight"]))
            h = res + (moe_out + so * sgv).view_as(h)
            release_tensors(s)
            if (lidx + 1) % 10 == 0:
                print(f"  [calib] layer {lidx}/{text_config.num_hidden_layers-1}", flush=True)

    del h, store
    torch.cuda.empty_cache()
    print("Calibration done.\n", flush=True)

    # === BUILD PLANS ===
    results: dict[str, dict[str, Any]] = {}

    def run_plan(plan: EvaluationPlan, label: str):
        print(f"=== Eval {label} ===", flush=True)
        t0 = time.time()
        ppl, nll = evaluate_plan(plan, eval_ids, text_config, weight_map, snapshot_dir, device, dtype)
        elapsed = time.time() - t0
        mem = plan.memory_gb
        fp8 = plan.fp8_fraction
        results[label] = {"ppl": round(ppl, 4), "nll": round(nll, 6), "memory_gb": round(mem, 3),
                          "fp8_frac": round(fp8, 4), "time_s": round(elapsed, 1)}
        print(f"  -> PPL={ppl:.4f} | mem={mem:.1f}GB | fp8={fp8:.4f} | time={elapsed:.1f}s", flush=True)

    for w1f, w2f, label in [
        (0.04, 0.16, "hess_w1_4_w2_16"),
        (0.02, 0.08, "hess_w1_2_w2_8"),
        (0.05, 0.05, "hess_5pct_uniform"),
        (0.10, 0.40, "hess_w1_10_w2_40"),
        (0.00, 0.20, "hess_w2_only_20"),
        (0.05, 0.20, "hess_w1_5_w2_20"),
        (0.00, 0.10, "hess_w2_only_10"),
        (0.03, 0.12, "hess_w1_3_w2_12"),
    ]:
        w1m, w2m = build_topk_masks(metric_cache, text_config, w1f, w2f)
        plan = make_plan(label, f"hessian_diag: W1={int(w1f*100)}% W2={int(w2f*100)}% FP8",
                          text_config, non_expert_bytes, total_expert_elems, w1m, w2m)
        run_plan(plan, label)

    # Save results
    payload = {
        "metadata": {"model": MODEL_ID, "eval_tokens": EVAL_TOKENS, "calib_tokens": CALIB_TOKENS,
                      "note": "Iteration 8: push outlier approach with per-projection split"},
        "results": results,
    }
    out_path = Path(__file__).parent / "results" / "exploration_iter08.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Print table
    print(f"\n{'='*70}", flush=True)
    print(f"{'Config':<30s} {'PPL':>8s} {'Memory':>10s} {'FP8%':>8s}", flush=True)
    print("-" * 70, flush=True)
    for label, r in sorted(results.items(), key=lambda x: x[1]["ppl"]):
        print(f"{label:<30s} {r['ppl']:>8.4f} {r['memory_gb']:>8.1f} GB {r['fp8_frac']*100:>7.1f}%", flush=True)

    # Update exploration.md
    best_label, best_r = min(results.items(), key=lambda x: x[1]["ppl"])
    section = f"""
## [17] Outlier + Per-Projection Fine Grid

**Approach**: Switched to hessian_diag metric (highest Spearman=0.723 from iter01). Tested 8 W1/W2 budget combos.

**Result**:
| Config | PPL | Memory | FP8% |
|--------|-----|--------|------|
"""
    for label, r in sorted(results.items(), key=lambda x: x[1]["ppl"]):
        section += f"| {label} | {r['ppl']:.4f} | {r['memory_gb']:.1f} GB | {r['fp8_frac']*100:.1f}% |\n"
    section += f"""
**Insight**: Best = {best_label} at PPL {best_r['ppl']:.4f}. Previous best was 5.329 (W1=4%, W2=16%).

**Next**: Continue exploring or start Phase 2 kernel co-design if improvements plateau.
"""
    elog = Path(__file__).parent / "exploration.md"
    with open(elog, "a") as f:
        f.write(section)

    print(f"\nSaved → {out_path}", flush=True)
    print(f"Updated → {elog}", flush=True)
    print(f"Best: {best_label} PPL={best_r['ppl']:.4f} mem={best_r['memory_gb']:.1f}GB", flush=True)


if __name__ == "__main__":
    main()

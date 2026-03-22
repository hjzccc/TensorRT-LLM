#!/usr/bin/env python3
"""Re-run perplexity with REAL FP8/FP4 computation via torch._scaled_mm."""
# pyright: reportMissingImports=false,reportAny=false,reportExplicitAny=false
import sys, json, time, math
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

import spike5_perplexity as s5
from spike1_ground_truth import (
    MODEL_ID, WeightStore, build_causal_mask, build_text_config,
    full_attention_forward, linear_attention_forward,
    load_root_config, move_tensor, quantize_to_fp8,
    quantize_to_nvfp4_columns, release_tensors, rms_norm_qwen3_next,
    layer_keys, shorten_layer_tensors,
)
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def fp8_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    w2d = w.contiguous()
    sx = (x2d.abs().amax().clamp(min=1e-12) / E4M3_MAX).float()
    sw = (w2d.abs().amax().clamp(min=1e-12) / E4M3_MAX).float()
    x8 = (x2d.float() / sx).to(torch.float8_e4m3fn)
    w8 = (w2d.float() / sw).to(torch.float8_e4m3fn)
    out = torch._scaled_mm(x8, w8.t(), scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16)
    return out.view(*x.shape[:-1], w.shape[0])


def quant_compute(x: torch.Tensor, w: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "bf16":
        return F.linear(x, w)
    wt = w.transpose(0, 1).contiguous()
    wq = (quantize_to_fp8(wt) if mode == "fp8" else quantize_to_nvfp4_columns(wt))
    wq = wq.transpose(0, 1).contiguous()
    return fp8_linear(x, wq)


def moe_forward_real(hidden_states, tensors, config, qcfg, promoted):
    B, S, H = hidden_states.shape
    flat = hidden_states.view(-1, H)
    rl = F.linear(flat, tensors["gate.weight"]).float()
    rw = torch.softmax(rl, dim=1)
    rw, se = torch.topk(rw, config.num_experts_per_tok, dim=-1)
    rw = (rw / rw.sum(dim=-1, keepdim=True)).to(hidden_states.dtype)
    final = torch.zeros(B * S, H, dtype=hidden_states.dtype, device=hidden_states.device)
    gu_w = tensors["experts.gate_up_proj"]
    dw = tensors["experts.down_proj"]
    ec = torch.bincount(se.reshape(-1), minlength=config.num_experts)
    for eidx in torch.nonzero(ec > 0, as_tuple=False).flatten().tolist():
        ti, rp = torch.where(se == eidx)
        cur = flat[ti]
        prec = qcfg.mode
        if prec == "mixed":
            prec = "fp8" if promoted and eidx in promoted else "fp4"
        gu = quant_compute(cur, gu_w[eidx], prec)
        g, u = gu.chunk(2, dim=-1)
        out = quant_compute(F.silu(g) * u, dw[eidx], prec)
        final.index_add_(0, ti, (rw[ti, rp].unsqueeze(-1) * out).to(hidden_states.dtype))
    sg = F.linear(flat, tensors["shared_expert.gate_proj.weight"])
    su = F.linear(flat, tensors["shared_expert.up_proj.weight"])
    so = F.linear(F.silu(sg) * su, tensors["shared_expert.down_proj.weight"])
    sgv = torch.sigmoid(F.linear(flat, tensors["shared_expert_gate.weight"]))
    final = final + so * sgv
    return final.view(B, S, H), rw


def run_one(label, qcfg, fp8_frac, eval_ids, text_config, weight_map, snapshot_dir,
            all_keys, embed_key, norm_key, lm_head_key, layer_counts, device, dtype):
    promoted = s5.resolve_promoted_experts(layer_counts, fp8_frac) if fp8_frac else None
    store = WeightStore(MODEL_ID, snapshot_dir, weight_map)
    rt = store.load_tensors([embed_key, norm_key, lm_head_key])
    ew = move_tensor(rt[embed_key], device, dtype)
    nw = move_tensor(rt[norm_key], device, dtype)
    hw = move_tensor(rt[lm_head_key], device, dtype)
    del rt
    with torch.inference_mode():
        h = F.embedding(eval_ids.to(device), ew)
        seq_len = eval_ids.shape[1]
        mask = build_causal_mask(seq_len, device)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        rotary = Qwen3NextRotaryEmbedding(config=text_config, device=device)
        pos_emb = rotary(h, pos_ids)
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
            moe = {k.replace("mlp.", "", 1): v for k, v in s.items() if k.startswith("mlp.")}
            lp = promoted.get(lidx) if promoted else None
            moe_out, _ = moe_forward_real(h, moe, text_config, qcfg, lp)
            h = res + moe_out
            release_tensors(s)
            if (lidx + 1) % 10 == 0:
                print(f"  [{label}] layer {lidx}/{text_config.num_hidden_layers-1}", flush=True)
        h = rms_norm_qwen3_next(h, nw, text_config.rms_norm_eps)
        logits = F.linear(h.float(), hw.float())
        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            eval_ids[:, 1:].to(device).view(-1),
        )
        ppl = math.exp(loss.item())
    del ew, nw, hw, store
    torch.cuda.empty_cache()
    return ppl, loss.item()


def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    EVAL_TOKENS = 512

    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    test_text = "\n\n".join(t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
    eval_ids = tokenizer.encode(test_text, return_tensors="pt")[:, :EVAL_TOKENS]
    train_text = "\n\n".join(t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"] if t.strip())
    calib_ids = tokenizer.encode(train_text, return_tensors="pt")[:, :128]
    print(f"Eval: {eval_ids.shape[1]} tokens | Calib: {calib_ids.shape[1]} tokens")

    snapshot_dir, root_config, weight_map = load_root_config(MODEL_ID)
    text_config = build_text_config(root_config)
    all_keys = list(weight_map.keys())
    embed_key = next(k for k in all_keys if "language_model.embed_tokens" in k)
    norm_key = "model.language_model.norm.weight"
    lm_head_key = "lm_head.weight"

    # Calibration pass for routing counts
    print("\n=== Calibration ===")
    layer_counts = {}
    store = WeightStore(MODEL_ID, snapshot_dir, weight_map)
    with torch.inference_mode():
        rt = store.load_tensors([embed_key])
        h = F.embedding(calib_ids.to(device), move_tensor(rt[embed_key], device, dtype))
        del rt
        csl = calib_ids.shape[1]
        mask = build_causal_mask(csl, device)
        pos_ids = torch.arange(csl, device=device).unsqueeze(0)
        rotary = Qwen3NextRotaryEmbedding(config=text_config, device=device)
        pos_emb = rotary(h, pos_ids)
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
            flat = h.view(-1, 2048)
            moe = {k.replace("mlp.", "", 1): v for k, v in s.items() if k.startswith("mlp.")}
            rl = F.linear(flat, moe["gate.weight"]).float()
            _, se = torch.topk(torch.softmax(rl, dim=1), text_config.num_experts_per_tok, dim=-1)
            layer_counts[lidx] = torch.bincount(se.reshape(-1), minlength=256)

            rw_val = torch.softmax(rl, dim=1)
            rw_val, se_val = torch.topk(rw_val, text_config.num_experts_per_tok, dim=-1)
            rw_val = (rw_val / rw_val.sum(dim=-1, keepdim=True)).to(dtype)
            gu_w, dw = moe["experts.gate_up_proj"], moe["experts.down_proj"]
            ec = torch.bincount(se_val.reshape(-1), minlength=256)
            moe_out = torch.zeros_like(flat)
            for eidx in torch.nonzero(ec > 0, as_tuple=False).flatten().tolist():
                ti, rp = torch.where(se_val == eidx)
                cur = flat[ti]
                gu = F.linear(cur, gu_w[eidx])
                g, u = gu.chunk(2, dim=-1)
                out = F.linear(F.silu(g) * u, dw[eidx])
                moe_out.index_add_(0, ti, (rw_val[ti, rp].unsqueeze(-1) * out).to(dtype))
            sg = F.linear(flat, moe["shared_expert.gate_proj.weight"])
            su = F.linear(flat, moe["shared_expert.up_proj.weight"])
            so = F.linear(F.silu(sg) * su, moe["shared_expert.down_proj.weight"])
            sgv = torch.sigmoid(F.linear(flat, moe["shared_expert_gate.weight"]))
            h = res + (moe_out + so * sgv).view_as(h)
            release_tensors(s)
            if (lidx + 1) % 10 == 0:
                print(f"  [calib] layer {lidx}/{text_config.num_hidden_layers-1}")
    del h, store
    torch.cuda.empty_cache()
    print("Calibration done.\n")

    configs = [
        ("bf16", s5.QuantConfig("bf16", "", "bf16"), None),
        ("uniform_fp4", s5.QuantConfig("fp4", "", "fp4"), None),
        ("uniform_fp8", s5.QuantConfig("fp8", "", "fp8"), None),
        ("mixed_25pct", s5.QuantConfig("m25", "", "mixed", 0.25), 0.25),
        ("mixed_50pct", s5.QuantConfig("m50", "", "mixed", 0.50), 0.50),
    ]

    results = {}
    for ci, (label, qcfg, fp8_frac) in enumerate(configs):
        print(f"=== Config {ci+1}/{len(configs)}: {label} ===")
        t0 = time.time()
        ppl, nll = run_one(
            label, qcfg, fp8_frac, eval_ids, text_config, weight_map, snapshot_dir,
            all_keys, embed_key, norm_key, lm_head_key, layer_counts, device, dtype,
        )
        elapsed = time.time() - t0
        results[label] = {"ppl": round(ppl, 4), "nll": round(nll, 6), "time_s": round(elapsed, 1)}
        print(f"  → PPL={ppl:.4f} ({elapsed:.0f}s)\n")

    ew = 256 * 40 * (2048*1024 + 512*2048)
    results["bf16"]["memory_gb"] = round(ew * 2.0 / 1e9, 1)
    results["uniform_fp4"]["memory_gb"] = round(ew * 0.5 / 1e9, 1)
    results["uniform_fp8"]["memory_gb"] = round(ew * 1.0 / 1e9, 1)
    results["mixed_25pct"]["memory_gb"] = round(ew * 0.625 / 1e9, 1)
    results["mixed_50pct"]["memory_gb"] = round(ew * 0.75 / 1e9, 1)

    output = {
        "metadata": {
            "model": MODEL_ID,
            "eval_tokens": int(eval_ids.shape[1]),
            "calib_tokens": 128,
            "computation": "REAL FP8 matmul via torch._scaled_mm (both weight and activation in FP8 E4M3)",
        },
        "results": results,
    }
    out_path = Path(__file__).parent / "results" / "real_precision_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"{'='*55}")
    print(f"RESULTS (real FP8 computation, {eval_ids.shape[1]} tokens)")
    print(f"{'='*55}")
    print(f"{'Method':<20s} {'PPL':>8s} {'Memory':>10s}")
    print("-" * 40)
    for label, r in sorted(results.items(), key=lambda x: x[1]["ppl"]):
        print(f"{label:<20s} {r['ppl']:>8.4f} {r.get('memory_gb','?'):>8} GB")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()

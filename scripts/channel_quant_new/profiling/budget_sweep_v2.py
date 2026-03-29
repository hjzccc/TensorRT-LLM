#!/usr/bin/env python3
"""Budget sweep: explicit budget parameter, no monkeypatching."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding
sys.path.insert(0, "/workspace/channel_quant_new")
sys.path.insert(0, "/workspace/channel_quant")
import tensorrt_llm._torch.auto_deploy.custom_ops
import exact_docker_eval as ee
from spike1_ground_truth import (
    build_text_config, layer_keys, load_root_config,
    move_tensor, release_tensors, rms_norm_qwen3_next, shorten_layer_tensors,
)

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
CAL_PATH = "/code/tensorrt_llm/scripts/channel_quant_new/profiling/calibration_slim.json"
OUT_DIR = "/code/tensorrt_llm/scripts/channel_quant_new/profiling"
W1_N, W2_N = 1024, 2048
W1_GRAN, W2_GRAN = 16, 32
NUM_LAYERS = 40
BF16_PPL, NVFP4_PPL = 6.5896, 6.8431
GAP = NVFP4_PPL - BF16_PPL
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

def snap(n, total, gran):
    if n <= 0: return 0
    if n >= total: return total
    return max(gran, ((n + gran // 2) // gran) * gran)

def depth_fraction(layer_idx, budget):
    fracs = np.linspace(budget * 0.33, budget * 1.67, NUM_LAYERS)
    return float(fracs[layer_idx] * budget / fracs.mean())

def build_masks(cal_layer, gate_up, down, layer_idx, budget, num_experts):
    bf16_frac = depth_fraction(layer_idx, budget)
    cal_experts = {e["expert"]: e for e in cal_layer.get("experts", [])}
    w1_tiers, w2_tiers = {}, {}
    for ei in range(num_experts):
        for scores_key, w, n_ch, gran, out in [
            ("w1", gate_up[ei], W1_N, W1_GRAN, w1_tiers),
            ("w2", down[ei], W2_N, W2_GRAN, w2_tiers),
        ]:
            cal_e = cal_experts.get(ei)
            if cal_e:
                scores = torch.tensor(cal_e[scores_key]["act_weighted"], dtype=torch.float32)
            else:
                scores = w.float().abs().mean(dim=-1).cpu()
            n_bf16 = snap(int(round(bf16_frac * n_ch)), n_ch, gran)
            tiers = torch.zeros(n_ch, dtype=torch.long)
            _, idx = scores.sort(descending=True)
            if n_bf16 > 0:
                tiers[idx[:n_bf16]] = 2
            wmag = w.float().abs().mean(dim=-1).cpu()
            nv = (tiers == 0).nonzero(as_tuple=True)[0]
            if nv.numel() > 0 and (wmag[nv] == 0).any():
                tiers[nv[wmag[nv] == 0]] = 2
            out[ei] = tiers
    return w1_tiers, w2_tiers

def tier_linear(x, w, tiers):
    n = w.shape[0]; dev = x.device
    out = torch.zeros(x.shape[0], n, dtype=x.dtype, device=dev)
    bf = (tiers == 2); nv = (tiers == 0)
    if nv.any():
        i = nv.nonzero(as_tuple=True)[0].to(dev)
        full_nvfp4 = ee.nvfp4_linear(x, w)
        out.index_copy_(1, i, full_nvfp4.index_select(1, i))
    if bf.any():
        i = bf.nonzero(as_tuple=True)[0].to(dev)
        out.index_copy_(1, i, ee.bf16_linear(x, w[i]))
    return out

def evaluate(eval_ids, ns, sl, cfg, wm, sd, dev, dt, cal, budget, lbs, label):
    store = ee.WeightStore(MODEL_ID, sd, wm)
    rt = store.load_tensors(["model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight"])
    ew = move_tensor(rt["model.language_model.embed_tokens.weight"], dev, dt)
    nw = move_tensor(rt["model.language_model.norm.weight"], dev, dt)
    lw = move_tensor(rt["lm_head.weight"], dev, dt); del rt
    chunks = eval_ids[:, :ns*sl].view(ns, sl).contiguous()
    hb = torch.empty((ns, sl, cfg.hidden_size), dtype=dt, device="cpu")
    for i in range(ns): hb[i].copy_(F.embedding(chunks[i:i+1].to(dev), ew).squeeze(0).cpu())
    pid = torch.arange(sl, device=dev).unsqueeze(0)
    rot = Qwen3NextRotaryEmbedding(config=cfg, device=dev)
    ri = torch.empty((1, sl, cfg.hidden_size), device=dev, dtype=dt)
    pe = rot(ri, pid); del ri
    nlls = []
    with torch.inference_mode():
        for li in range(cfg.num_hidden_layers):
            lt = cfg.layer_types[li]
            raw = store.load_tensors(layer_keys(li, lt))
            ld = shorten_layer_tensors(li, raw, dev, dt); del raw
            mt = {k.replace("mlp.","",1): v for k,v in ld.items() if k.startswith("mlp.")}
            w1t, w2t = build_masks(cal.get(str(li), {}), mt["experts.gate_up_proj"], mt["experts.down_proj"], li, budget, cfg.num_experts)
            for si in range(ns):
                h = hb[si:si+1].to(dev); res = h
                h = rms_norm_qwen3_next(h, ld["input_layernorm.weight"], cfg.rms_norm_eps)
                if lt == "full_attention":
                    at = {k.replace("self_attn.",""):v for k,v in ld.items() if k.startswith("self_attn.")}
                    h = ee.full_attention_forward_exact(h, at, cfg, pe, ee.build_causal_mask(sl, dev), mode="bf16", quantized=False)
                else:
                    at = {k.replace("linear_attn.",""):v for k,v in ld.items() if k.startswith("linear_attn.")}
                    h = ee.linear_attention_forward_exact(h, at, cfg, "bf16", "moe_only")
                h = res + h; res = h
                h = rms_norm_qwen3_next(h, ld["post_attention_layernorm.weight"], cfg.rms_norm_eps)
                flat = h.view(-1, cfg.hidden_size)
                rl = ee.bf16_linear(flat, mt["gate.weight"]).float()
                rw = torch.softmax(rl, dim=1)
                rw, se = torch.topk(rw, cfg.num_experts_per_tok, dim=-1)
                rw = rw / rw.sum(dim=-1, keepdim=True); rw = rw.to(dt)
                fhs = torch.zeros_like(flat)
                ec = torch.bincount(se.reshape(-1), minlength=cfg.num_experts)
                for ei in torch.nonzero(ec > 0, as_tuple=False).flatten().tolist():
                    ti, rp = torch.where(se == ei); cs = flat[ti]
                    gu = tier_linear(cs, mt["experts.gate_up_proj"][ei], w1t.get(ei, torch.zeros(W1_N, dtype=torch.long)))
                    g, u = gu.chunk(2, dim=-1); mid = F.silu(g) * u
                    ch = tier_linear(mid, mt["experts.down_proj"][ei], w2t.get(ei, torch.zeros(W2_N, dtype=torch.long)))
                    fhs.index_add_(0, ti, (ch * rw[ti, rp].unsqueeze(-1)).to(dt))
                shared = ee.bf16_linear(flat, mt["shared_expert.gate_proj.weight"])
                shared = F.silu(shared) * ee.bf16_linear(flat, mt["shared_expert.up_proj.weight"])
                shared = ee.bf16_linear(shared, mt["shared_expert.down_proj.weight"])
                sg = torch.sigmoid(ee.bf16_linear(flat, mt["shared_expert_gate.weight"]))
                h = res + (fhs + sg * shared).view_as(res)
                hb[si:si+1].copy_(h.cpu()); del h, res, fhs
            release_tensors(ld)
            if (li+1) % 10 == 0: print(f"  [{label}] layer {li+1}/{cfg.num_hidden_layers}", flush=True)
        for si in range(ns):
            torch.cuda.empty_cache()
            c = chunks[si:si+1].to(dev); h = hb[si:si+1].to(dev)
            h = rms_norm_qwen3_next(h, nw, cfg.rms_norm_eps)
            logits = F.linear(h, lw)
            sl2 = logits[:, :-1, :].contiguous().float()
            nlls.append(F.cross_entropy(sl2.view(-1, logits.size(-1)), c[:, 1:].view(-1)).float() * sl)
            del logits, sl2, h
    ppl = torch.exp(torch.stack(nlls).sum() / (ns * sl)).item()
    del ew, nw, lw, store; torch.cuda.empty_cache()
    return ppl

def main():
    dev = torch.device("cuda"); dt = torch.float16
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    eval_ids, mx, sl = ee.load_eval_data(tok, 2048)
    ns = min(145, mx)
    sd, rc, wm = load_root_config(MODEL_ID)
    cfg = build_text_config(rc)
    cal = None  # loaded per-layer
    print(f"Calibration: {cal["metadata"]["source"]}", flush=True)
    print(f"Eval: {ns} chunks from WikiText-2 TEST", flush=True)
    results = {}
    for budget in BUDGETS:
        label = f"act_weighted_{int(budget*100)}pct"
        print(f"\n=== {label} ({budget:.0%} BF16) ===", flush=True)
        t0 = time.time()
        ppl = evaluate(eval_ids, ns, sl, cfg, wm, sd, dev, dt, cal, budget, 1, label)
        el = time.time() - t0
        rec = (NVFP4_PPL - ppl) / GAP * 100
        results[label] = {"budget": budget, "ppl": round(ppl, 4), "recovery": round(rec, 1), "time_s": round(el, 1)}
        print(f"  -> PPL={ppl:.4f}  recovery={rec:.1f}%  ({el:.0f}s)", flush=True)
    out = Path(OUT_DIR) / "budget_sweep_v2.json"
    with out.open("w") as f:
        json.dump({"metadata": {"metric": "act_weighted", "allocation": "depth_ramp", "cal": "wikitext2_train_128x2048", "eval": "wikitext2_test_145x2048"}, "results": results}, f, indent=2)
    print(f"\nSaved -> {out}")
    for n, r in sorted(results.items(), key=lambda x: x[1]["budget"]):
        print(f"  {r['budget']:>5.0%}  PPL={r['ppl']:.4f}  recovery={r['recovery']:.1f}%")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Exploration: depth-aware and routing-aware BF16+NVFP4 allocation strategies.

Tests multiple allocation strategies at the same ~15% total BF16 budget:
  1. uniform     — current baseline: 15% BF16 for every expert in every layer
  2. depth_ramp  — linear ramp: 5% at layer 0 → 25% at layer 39
  3. routing     — per-expert: hot experts get more BF16 (proportional to sqrt(routing_weight))
  4. combined    — depth × routing: both effects together
  5. w2_heavy    — give W2 more BF16 than W1 in deep layers (W1=10%, W2=20% at layer 39)
  6. depth_step  — step function: layers 0-19 get 10%, layers 20-39 get 20%

All strategies constrained to the same total BF16 channel count for fair comparison.
Uses weight-magnitude metric for channel selection within each expert.
Evaluates on full 145-chunk WikiText-2 test set with exact TRT-LLM NVFP4 kernels.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

sys.path.insert(0, "/workspace/channel_quant_new")
sys.path.insert(0, "/workspace/channel_quant")
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
import exact_docker_eval as ee
from spike1_ground_truth import (
    build_text_config, layer_keys, load_root_config,
    move_tensor, release_tensors, rms_norm_qwen3_next, shorten_layer_tensors,
)

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
PROFILING_PATH = "/code/tensorrt_llm/scripts/channel_quant_new/profiling/error_profile.json"
OUTPUT_DIR = "/code/tensorrt_llm/scripts/channel_quant_new/profiling"
W1_N, W2_N = 1024, 2048
W1_GRAN, W2_GRAN = 16, 32
NUM_LAYERS = 40
NUM_EXPERTS = 256
TARGET_BF16_FRACTION = 0.15


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nsamples", type=int, default=145)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--layer-batch-size", type=int, default=1)
    return p.parse_args()


def snap(n, total, gran):
    if n <= 0: return 0
    if n >= total: return total
    return max(gran, ((n + gran // 2) // gran) * gran)


def load_profiling_data():
    with open(PROFILING_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Allocation strategies: each returns dict[layer_idx] -> dict[expert_idx] -> (w1_bf16_frac, w2_bf16_frac)
# ---------------------------------------------------------------------------

def allocate_uniform(profiling_data):
    alloc = {}
    for li in range(NUM_LAYERS):
        alloc[li] = {}
        for e in profiling_data[str(li)]["experts"]:
            ei = e["expert"]
            alloc[li][ei] = (TARGET_BF16_FRACTION, TARGET_BF16_FRACTION)
    return alloc


def allocate_depth_ramp(profiling_data):
    """Linear ramp: early layers get less BF16, deep layers get more."""
    alloc = {}
    fracs = np.linspace(0.05, 0.25, NUM_LAYERS)
    # Normalize to maintain overall 15% average
    fracs = fracs * (TARGET_BF16_FRACTION / fracs.mean())
    for li in range(NUM_LAYERS):
        alloc[li] = {}
        for e in profiling_data[str(li)]["experts"]:
            alloc[li][e["expert"]] = (float(fracs[li]), float(fracs[li]))
    return alloc


def allocate_depth_step(profiling_data):
    """Step: first half 10%, second half 20%."""
    alloc = {}
    for li in range(NUM_LAYERS):
        frac = 0.10 if li < NUM_LAYERS // 2 else 0.20
        alloc[li] = {}
        for e in profiling_data[str(li)]["experts"]:
            alloc[li][e["expert"]] = (frac, frac)
    return alloc


def allocate_routing(profiling_data):
    """Per-expert: BF16 proportional to sqrt(routing_weight_sum), clipped [5%, 30%]."""
    alloc = {}
    for li in range(NUM_LAYERS):
        experts = profiling_data[str(li)]["experts"]
        rw_values = {e["expert"]: e["routing_weight_sum"] for e in experts}
        sqrt_rw = {ei: math.sqrt(rw + 1e-8) for ei, rw in rw_values.items()}
        total_sqrt = sum(sqrt_rw.values())

        # Scale so mean fraction = TARGET_BF16_FRACTION
        n_experts = len(experts)
        raw_fracs = {ei: (v / total_sqrt) * n_experts * TARGET_BF16_FRACTION for ei, v in sqrt_rw.items()}

        # Clip to [0.05, 0.30]
        clipped = {ei: max(0.05, min(0.30, f)) for ei, f in raw_fracs.items()}
        # Renormalize to maintain target average
        clip_mean = sum(clipped.values()) / len(clipped)
        scale = TARGET_BF16_FRACTION / clip_mean if clip_mean > 0 else 1.0
        final = {ei: max(0.05, min(0.30, f * scale)) for ei, f in clipped.items()}

        alloc[li] = {ei: (final[ei], final[ei]) for ei in final}
    return alloc


def allocate_combined(profiling_data):
    """Depth × routing: both effects together. b(l,e) ∝ sqrt(depth_weight × routing_weight)."""
    alloc = {}
    depth_weights = np.linspace(0.5, 1.5, NUM_LAYERS)

    for li in range(NUM_LAYERS):
        experts = profiling_data[str(li)]["experts"]
        rw_values = {e["expert"]: e["routing_weight_sum"] for e in experts}

        scores = {}
        for ei, rw in rw_values.items():
            scores[ei] = math.sqrt(depth_weights[li] * (rw + 1e-8))

        total_score = sum(scores.values())
        n_experts = len(experts)
        raw_fracs = {ei: (s / total_score) * n_experts * TARGET_BF16_FRACTION for ei, s in scores.items()}

        clipped = {ei: max(0.05, min(0.30, f)) for ei, f in raw_fracs.items()}
        clip_mean = sum(clipped.values()) / len(clipped)
        scale = TARGET_BF16_FRACTION / clip_mean if clip_mean > 0 else 1.0
        final = {ei: max(0.05, min(0.30, f * scale)) for ei, f in clipped.items()}

        alloc[li] = {ei: (final[ei], final[ei]) for ei in final}
    return alloc


def allocate_w2_heavy(profiling_data):
    """Give W2 more BF16 than W1, especially in deep layers.
    W1: 10% uniform. W2: ramp 10% → 25% by depth. Same total budget as 15% uniform."""
    alloc = {}
    w2_fracs = np.linspace(0.10, 0.25, NUM_LAYERS)
    w1_frac = 0.10
    # Adjust W1 to keep total budget = 15%: total = (w1*1024 + w2*2048) / (1024+2048) = 15%
    # w1*1024 + w2*2048 = 0.15 * 3072 = 460.8 channels on average
    # w1 = (460.8 - w2*2048) / 1024 ... but simpler: just set W1=10% and W2 varies
    for li in range(NUM_LAYERS):
        alloc[li] = {}
        for e in profiling_data[str(li)]["experts"]:
            alloc[li][e["expert"]] = (w1_frac, float(w2_fracs[li]))
    return alloc


def allocate_error_proportional(profiling_data):
    """BF16 ∝ sqrt(routing_weight × total_mae), capturing actual MoE error contribution."""
    alloc = {}
    depth_weights = np.linspace(0.5, 1.5, NUM_LAYERS)

    for li in range(NUM_LAYERS):
        experts = profiling_data[str(li)]["experts"]
        scores = {}
        for e in experts:
            ei = e["expert"]
            rw = e["routing_weight_sum"]
            w1_mae = e["w1"]["total_mae"] if e["w1"] else 0
            w2_mae = e["w2"]["total_mae"] if e["w2"] else 0
            error_contribution = rw * (w1_mae + w2_mae)
            scores[ei] = math.sqrt(error_contribution + 1e-12)

        total_score = sum(scores.values())
        n_experts = len(experts)
        raw_fracs = {ei: (s / total_score) * n_experts * TARGET_BF16_FRACTION for ei, s in scores.items()}
        clipped = {ei: max(0.05, min(0.30, f)) for ei, f in raw_fracs.items()}
        clip_mean = sum(clipped.values()) / len(clipped)
        scale = TARGET_BF16_FRACTION / clip_mean if clip_mean > 0 else 1.0
        final = {ei: max(0.05, min(0.30, f * scale)) for ei, f in clipped.items()}

        alloc[li] = {ei: (final[ei], final[ei]) for ei in final}
    return alloc



def allocate_routing_threshold(profiling_data, threshold_percentile=50):
    """Threshold-based concentration: experts above threshold get high BF16, below get low."""
    alloc = {}
    for li in range(NUM_LAYERS):
        experts = profiling_data[str(li)]["experts"]
        rw_values = [e["routing_weight_sum"] for e in experts]
        threshold = np.percentile(rw_values, threshold_percentile)
        
        alloc[li] = {}
        for e in experts:
            ei = e["expert"]
            rw = e["routing_weight_sum"]
            # High-traffic experts: 25%, low-traffic: 8%
            frac = 0.25 if rw >= threshold else 0.08
            alloc[li][ei] = (frac, frac)
    
    # Renormalize to maintain target average
    total_frac = sum(sum(v.values()) for v in alloc.values()) / (NUM_LAYERS * NUM_EXPERTS)
    scale = TARGET_BF16_FRACTION / total_frac if total_frac > 0 else 1.0
    for li in alloc:
        for ei in alloc[li]:
            f = alloc[li][ei][0] * scale
            alloc[li][ei] = (max(0.05, min(0.30, f)), max(0.05, min(0.30, f)))
    
    return alloc


def allocate_routing_topk(profiling_data, k_fraction=0.25):
    """Top-K concentration: top K% of experts get high BF16, rest get low."""
    alloc = {}
    for li in range(NUM_LAYERS):
        experts = profiling_data[str(li)]["experts"]
        n_experts = len(experts)
        k = max(1, int(n_experts * k_fraction))
        
        # Sort by routing weight
        sorted_experts = sorted(experts, key=lambda e: e["routing_weight_sum"], reverse=True)
        top_k_set = {e["expert"] for e in sorted_experts[:k]}
        
        alloc[li] = {}
        for e in experts:
            ei = e["expert"]
            # Top-K experts: 28%, rest: 5%
            frac = 0.28 if ei in top_k_set else 0.05
            alloc[li][ei] = (frac, frac)
    
    # Renormalize to maintain target average
    total_frac = sum(sum(v.values()) for v in alloc.values()) / (NUM_LAYERS * NUM_EXPERTS)
    scale = TARGET_BF16_FRACTION / total_frac if total_frac > 0 else 1.0
    for li in alloc:
        for ei in alloc[li]:
            f = alloc[li][ei][0] * scale
            alloc[li][ei] = (max(0.05, min(0.30, f)), max(0.05, min(0.30, f)))
    
    return alloc


def allocate_routing_exponential(profiling_data):
    """Exponential concentration: BF16 ∝ exp(routing_weight_sum), more aggressive than sqrt."""
    alloc = {}
    for li in range(NUM_LAYERS):
        experts = profiling_data[str(li)]["experts"]
        rw_values = {e["expert"]: e["routing_weight_sum"] for e in experts}
        
        # Exponential scaling: exp(rw / max_rw) - 1
        max_rw = max(rw_values.values()) if rw_values else 1.0
        exp_rw = {ei: math.exp((rw / max_rw) * 2) - 1 for ei, rw in rw_values.items()}
        total_exp = sum(exp_rw.values())
        
        n_experts = len(experts)
        raw_fracs = {ei: (v / total_exp) * n_experts * TARGET_BF16_FRACTION for ei, v in exp_rw.items()}
        
        clipped = {ei: max(0.05, min(0.30, f)) for ei, f in raw_fracs.items()}
        clip_mean = sum(clipped.values()) / len(clipped)
        scale = TARGET_BF16_FRACTION / clip_mean if clip_mean > 0 else 1.0
        final = {ei: max(0.05, min(0.30, f * scale)) for ei, f in clipped.items()}
        
        alloc[li] = {ei: (final[ei], final[ei]) for ei in final}
    return alloc


ALL_STRATEGIES = [
    ("uniform", allocate_uniform),
    ("depth_ramp", allocate_depth_ramp),
    ("depth_step", allocate_depth_step),
    ("routing", allocate_routing),
    ("combined", allocate_combined),
    ("w2_heavy", allocate_w2_heavy),
    ("error_proportional", allocate_error_proportional),
    ("routing_threshold", allocate_routing_threshold),
    ("routing_topk", allocate_routing_topk),
    ("routing_exponential", allocate_routing_exponential),
]


# ---------------------------------------------------------------------------
# Mask building with per-expert BF16 fractions
# ---------------------------------------------------------------------------

def build_masks_for_layer(
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    expert_alloc: dict[int, tuple[float, float]],
    num_experts: int,
):
    w1_tiers = {}
    w2_tiers = {}
    for expert_idx in range(num_experts):
        if expert_idx not in expert_alloc:
            w1_tiers[expert_idx] = torch.zeros(W1_N, dtype=torch.long)
            w2_tiers[expert_idx] = torch.zeros(W2_N, dtype=torch.long)
            continue

        w1_bf16_frac, w2_bf16_frac = expert_alloc[expert_idx]

        for w, n_ch, gran, bf16_frac, tiers_out in [
            (gate_up_weights[expert_idx], W1_N, W1_GRAN, w1_bf16_frac, w1_tiers),
            (down_weights[expert_idx], W2_N, W2_GRAN, w2_bf16_frac, w2_tiers),
        ]:
            sensitivity = w.float().abs().mean(dim=-1).cpu()
            n_bf16 = snap(int(round(bf16_frac * n_ch)), n_ch, gran) if bf16_frac > 0 else 0
            tiers = torch.zeros(n_ch, dtype=torch.long)
            _, sorted_idx = sensitivity.sort(descending=True)
            if n_bf16 > 0:
                tiers[sorted_idx[:n_bf16]] = 2

            # Promote zero-weight NVFP4 channels to BF16
            nvfp4_ch = (tiers == 0).nonzero(as_tuple=True)[0]
            if nvfp4_ch.numel() > 0:
                zero_mask = sensitivity[nvfp4_ch] == 0
                if zero_mask.any():
                    tiers[nvfp4_ch[zero_mask]] = 2

            tiers_out[expert_idx] = tiers
    return w1_tiers, w2_tiers


def three_tier_linear(x, w, tiers):
    n_out = w.shape[0]
    device = x.device
    bf16_mask = tiers == 2
    nvfp4_mask = tiers == 0
    output = torch.zeros(x.shape[0], n_out, dtype=x.dtype, device=device)

    if bf16_mask.any():
        idx = bf16_mask.nonzero(as_tuple=True)[0].to(device)
        output.index_copy_(1, idx, ee.bf16_linear(x, w[idx]))

    if nvfp4_mask.any():
        idx = nvfp4_mask.nonzero(as_tuple=True)[0].to(device)
        w_sub = w[idx]
        n = w_sub.shape[0]
        pad = (32 - n % 32) % 32
        if pad:
            w_sub = F.pad(w_sub, (0, 0, 0, pad))
        o = ee.nvfp4_linear(x, w_sub)
        if pad:
            o = o[:, :n]
        output.index_copy_(1, idx, o)
    return output


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_strategy(
    eval_ids, nsamples, seqlen, config, weight_map, snapshot_dir,
    device, dtype, allocation, label, layer_batch_size,
):
    store = ee.WeightStore(MODEL_ID, snapshot_dir, weight_map)
    ek = "model.language_model.embed_tokens.weight"
    nk = "model.language_model.norm.weight"
    lk = "lm_head.weight"
    rt = store.load_tensors([ek, nk, lk])
    ew = move_tensor(rt[ek], device, dtype)
    nw = move_tensor(rt[nk], device, dtype)
    lw = move_tensor(rt[lk], device, dtype)
    del rt

    chunks = eval_ids[:, :nsamples * seqlen].view(nsamples, seqlen).contiguous()
    hb = torch.empty((nsamples, seqlen, config.hidden_size), dtype=dtype, device="cpu")
    for i in range(nsamples):
        hb[i].copy_(F.embedding(chunks[i:i+1].to(device), ew).squeeze(0).cpu())

    pid = torch.arange(seqlen, device=device).unsqueeze(0)
    rot = Qwen3NextRotaryEmbedding(config=config, device=device)
    ri = torch.empty((1, seqlen, config.hidden_size), device=device, dtype=dtype)
    pe = rot(ri, pid); del ri

    nlls = []
    with torch.inference_mode():
        for li in range(config.num_hidden_layers):
            lt = config.layer_types[li]
            raw = store.load_tensors(layer_keys(li, lt))
            ld = shorten_layer_tensors(li, raw, device, dtype); del raw

            mt = {k.replace("mlp.", "", 1): v for k, v in ld.items() if k.startswith("mlp.")}
            w1_tiers, w2_tiers = build_masks_for_layer(
                mt["experts.gate_up_proj"], mt["experts.down_proj"],
                allocation.get(li, {}), config.num_experts,
            )

            for si in range(nsamples):
                h = hb[si:si+1].to(device)
                res = h
                h = rms_norm_qwen3_next(h, ld["input_layernorm.weight"], config.rms_norm_eps)
                if lt == "full_attention":
                    at = {k.replace("self_attn.", ""): v for k, v in ld.items() if k.startswith("self_attn.")}
                    h = ee.full_attention_forward_exact(h, at, config, pe, ee.build_causal_mask(seqlen, device), mode="bf16", quantized=False)
                else:
                    at = {k.replace("linear_attn.", ""): v for k, v in ld.items() if k.startswith("linear_attn.")}
                    h = ee.linear_attention_forward_exact(h, at, config, "bf16", "moe_only")
                h = res + h; res = h
                h = rms_norm_qwen3_next(h, ld["post_attention_layernorm.weight"], config.rms_norm_eps)

                flat = h.view(-1, config.hidden_size)
                rl = ee.bf16_linear(flat, mt["gate.weight"]).float()
                rw = torch.softmax(rl, dim=1)
                rw, se = torch.topk(rw, config.num_experts_per_tok, dim=-1)
                rw = rw / rw.sum(dim=-1, keepdim=True)
                rw = rw.to(dtype)
                fhs = torch.zeros_like(flat)
                ec = torch.bincount(se.reshape(-1), minlength=config.num_experts)

                for ei in torch.nonzero(ec > 0, as_tuple=False).flatten().tolist():
                    ti, rp = torch.where(se == ei)
                    cs = flat[ti]
                    gu = three_tier_linear(cs, mt["experts.gate_up_proj"][ei], w1_tiers.get(ei, torch.zeros(W1_N, dtype=torch.long)))
                    g, u = gu.chunk(2, dim=-1)
                    mid = F.silu(g) * u
                    ch = three_tier_linear(mid, mt["experts.down_proj"][ei], w2_tiers.get(ei, torch.zeros(W2_N, dtype=torch.long)))
                    ch = ch * rw[ti, rp].unsqueeze(-1)
                    fhs.index_add_(0, ti, ch.to(dtype))

                shared = ee.bf16_linear(flat, mt["shared_expert.gate_proj.weight"])
                shared = F.silu(shared) * ee.bf16_linear(flat, mt["shared_expert.up_proj.weight"])
                shared = ee.bf16_linear(shared, mt["shared_expert.down_proj.weight"])
                sg = torch.sigmoid(ee.bf16_linear(flat, mt["shared_expert_gate.weight"]))
                fhs = fhs + sg * shared
                h = res + fhs.view_as(res)
                hb[si:si+1].copy_(h.cpu())
                del h, res, fhs

            release_tensors(ld)
            if (li + 1) % 10 == 0:
                print(f"  [{label}] layer {li+1}/{config.num_hidden_layers}", flush=True)

        for si in range(nsamples):
            torch.cuda.empty_cache()
            c = chunks[si:si+1].to(device)
            h = hb[si:si+1].to(device)
            h = rms_norm_qwen3_next(h, nw, config.rms_norm_eps)
            logits = F.linear(h, lw)
            sl = logits[:, :-1, :].contiguous().float()
            lab = c[:, 1:]
            loss = F.cross_entropy(sl.view(-1, logits.size(-1)), lab.view(-1))
            nlls.append(loss.float() * seqlen)
            del logits, sl, h

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()
    del ew, nw, lw, store; torch.cuda.empty_cache()
    return ppl


def main():
    args = parse_args()
    device = torch.device("cuda"); dtype = torch.float16
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    eval_ids, max_ns, seqlen = ee.load_eval_data(tok, args.seqlen)
    nsamples = min(args.nsamples, max_ns)
    snapshot_dir, root_config, weight_map = load_root_config(MODEL_ID)
    config = build_text_config(root_config)

    profiling_data = load_profiling_data()
    print(f"Eval: {nsamples} chunks of {seqlen} tokens ({nsamples * seqlen} total)", flush=True)
    print(f"Profiling data: {profiling_data['metadata']}", flush=True)

    results = {}
    for strategy_name, strategy_fn in ALL_STRATEGIES:
        print(f"\n=== {strategy_name} ===", flush=True)
        allocation = strategy_fn(profiling_data)

        # Log allocation stats
        all_fracs = []
        for li in range(NUM_LAYERS):
            for ei, (w1f, w2f) in allocation.get(li, {}).items():
                all_fracs.append((w1f + w2f) / 2)
        mean_frac = np.mean(all_fracs) if all_fracs else 0
        print(f"  Mean BF16 fraction: {mean_frac:.3f} ({len(all_fracs)} expert-layers)", flush=True)

        t0 = time.time()
        ppl = evaluate_strategy(
            eval_ids, nsamples, seqlen, config, weight_map, snapshot_dir,
            device, dtype, allocation, strategy_name, args.layer_batch_size,
        )
        elapsed = time.time() - t0
        results[strategy_name] = {
            "ppl": round(ppl, 4),
            "time_s": round(elapsed, 1),
            "mean_bf16_fraction": round(mean_frac, 4),
        }
        print(f"  -> PPL={ppl:.4f} ({elapsed:.0f}s)", flush=True)

    out_path = Path(OUTPUT_DIR) / "allocation_experiments.json"
    with out_path.open("w") as f:
        json.dump({"metadata": {"nsamples": nsamples, "seqlen": seqlen, "target_bf16": TARGET_BF16_FRACTION}, "results": results}, f, indent=2)
    print(f"\nSaved -> {out_path}", flush=True)
    print(f"\n{'Strategy':<25s} {'PPL':>10s} {'BF16%':>8s}")
    print("-" * 45)
    for name, r in sorted(results.items(), key=lambda x: x[1]["ppl"]):
        print(f"{name:<25s} {r['ppl']:>10.4f} {r['mean_bf16_fraction']:>7.1%}")


if __name__ == "__main__":
    main()

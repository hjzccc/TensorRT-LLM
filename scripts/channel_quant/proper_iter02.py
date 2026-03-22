#!/usr/bin/env python3
"""Iteration 2: Sweep topup percentage on MxMoE block + channel topup approach."""
import sys, json, time, gc, math
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from proper_iter01 import (
    run_calibration, build_mxmoe_topup_masks, 
    EvalPlan, evaluate_plan, resolve_terminal_keys,
    MODEL_ID, SEQLEN,
)
from proper_eval import (
    load_gptq_standard_data, run_calibration as run_calibration_pe,
)
from spike1_ground_truth import WeightStore, load_root_config, build_text_config

def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16

    tokenizer, calib_ids, test_ids, calib_info, eval_info = load_gptq_standard_data(MODEL_ID)
    snapshot_dir, root_config, weight_map = load_root_config(MODEL_ID)
    text_config = build_text_config(root_config)
    total_expert_elems = text_config.num_experts * text_config.num_hidden_layers * (
        2 * text_config.moe_intermediate_size * text_config.hidden_size +
        text_config.hidden_size * text_config.moe_intermediate_size
    )

    cache_path = Path(__file__).parent / "results" / "proper_iter01_calibration_cache.pt"
    if cache_path.exists():
        print(f"Loading cached calibration from {cache_path}", flush=True)
        raw_cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        base = raw_cache["base_calibration"]
        
        class _AttrDict:
            def __init__(self, d):
                for k, v in d.items():
                    setattr(self, k, v)
        
        act_cache = {}
        for lidx, bundle_dict in base["activation_cache"].items():
            act_cache[lidx] = _AttrDict(bundle_dict)
        
        calibration = _AttrDict({
            "routing_counts": base["routing_counts"],
            "activation_cache": act_cache,
            "mxmoe_w1_deltas": base["mxmoe_w1_deltas"],
            "mxmoe_w2_deltas": base["mxmoe_w2_deltas"],
            "mc_moe_scores": base["mc_moe_scores"],
        })
    else:
        print("Running calibration...", flush=True)
        calibration = run_calibration(text_config, weight_map, snapshot_dir, device, dtype)

    results = {}
    for topup_pct in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        label = f"topup_{int(topup_pct*100):02d}pct"
        print(f"\n=== {label} ===", flush=True)
        w1_masks, w2_masks, _meta = build_mxmoe_topup_masks(
            calibration, text_config, total_expert_elems, topup_pct
        )
        from proper_eval import fp8_weights_from_masks, estimate_mixed_memory_gb
        fp8_w = fp8_weights_from_masks(text_config, w1_masks, w2_masks)
        mem = estimate_mixed_memory_gb(0, total_expert_elems, fp8_w)
        plan = EvalPlan(
            name=label,
            description=f"MxMoE block + {int(topup_pct*100)}% channel topup",
            mode="per_channel",
            memory_gb=mem,
            fp8_weights=fp8_w,
            w1_pair_masks=w1_masks,
            w2_channel_masks=w2_masks,
        )
        frac = fp8_w / max(total_expert_elems, 1)
        t0 = time.time()
        ppl, nll, nsamples = evaluate_plan(plan, test_ids, text_config, weight_map, snapshot_dir, device, dtype)
        elapsed = time.time() - t0
        frac = fp8_w / max(total_expert_elems, 1)
        results[label] = {
            "ppl": round(ppl, 4), "nll": round(nll, 6),
            "memory_gb": round(mem, 3),
            "fp8_fraction": round(frac, 4),
            "topup_pct": topup_pct,
            "time_s": round(elapsed, 1),
        }
        print(f"  -> PPL={ppl:.4f} | mem={mem:.1f}GB | fp8={frac:.4f} | time={elapsed:.0f}s", flush=True)

    out_path = Path(__file__).parent / "results" / "proper_iter02.json"
    payload = {
        "metadata": {"model": MODEL_ID, "seqlen": SEQLEN, "experiment": "MxMoE block + channel topup sweep"},
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'Config':<20s} {'PPL':>8s} {'Memory':>10s} {'FP8%':>8s}")
    print("-" * 48)
    for label, r in sorted(results.items(), key=lambda x: x[1]["ppl"]):
        print(f"{label:<20s} {r['ppl']:>8.4f} {r['memory_gb']:>8.1f} GB {r['fp8_fraction']*100:>7.1f}%")
    print(f"\nSaved -> {out_path}")

if __name__ == "__main__":
    main()

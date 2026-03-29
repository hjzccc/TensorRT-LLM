#!/usr/bin/env python3
"""Budget sweep: test 10%, 15%, 20%, 25%, 30% BF16 with depth_ramp allocation."""
import sys
sys.path.insert(0, "/code/tensorrt_llm/scripts/channel_quant_new/profiling")
from run_allocation_experiments import *

BUDGET_LEVELS = [0.10, 0.15, 0.20, 0.25, 0.30]

def allocate_depth_ramp_budget(profiling_data, target_frac):
    alloc = {}
    fracs = np.linspace(target_frac * 0.33, target_frac * 1.67, NUM_LAYERS)
    fracs = fracs * (target_frac / fracs.mean())
    for li in range(NUM_LAYERS):
        alloc[li] = {}
        for e in profiling_data[str(li)]["experts"]:
            alloc[li][e["expert"]] = (float(fracs[li]), float(fracs[li]))
    return alloc

def main():
    args = parse_args()
    device = torch.device("cuda"); dtype = torch.float16
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    eval_ids, max_ns, seqlen = ee.load_eval_data(tok, args.seqlen)
    nsamples = min(args.nsamples, max_ns)
    snapshot_dir, root_config, weight_map = load_root_config(MODEL_ID)
    config = build_text_config(root_config)
    profiling_data = load_profiling_data()
    print(f"Eval: {nsamples} chunks of {seqlen} tokens", flush=True)

    results = {}
    for budget in BUDGET_LEVELS:
        label = f"depth_ramp_{int(budget*100)}pct"
        print(f"\n=== {label} ({budget:.0%} BF16 with depth ramp) ===", flush=True)
        allocation = allocate_depth_ramp_budget(profiling_data, budget)
        t0 = time.time()
        ppl = evaluate_strategy(
            eval_ids, nsamples, seqlen, config, weight_map, snapshot_dir,
            device, dtype, allocation, label, args.layer_batch_size,
        )
        elapsed = time.time() - t0
        gap_recovery = (6.8431 - ppl) / (6.8431 - 6.5896) * 100
        results[label] = {"budget": budget, "ppl": round(ppl, 4), "time_s": round(elapsed, 1), "gap_recovery_pct": round(gap_recovery, 1)}
        print(f"  -> PPL={ppl:.4f} gap_recovery={gap_recovery:.1f}% ({elapsed:.0f}s)", flush=True)

    out = Path(OUTPUT_DIR) / "budget_sweep.json"
    with out.open("w") as f:
        json.dump({"metadata": {"nsamples": nsamples, "seqlen": seqlen, "allocation": "depth_ramp"}, "results": results}, f, indent=2)
    print(f"\nSaved -> {out}")
    print(f"\n{Budget:>8s} {PPL:>10s} {Recovery:>10s}")
    print("-" * 30)
    for name, r in sorted(results.items(), key=lambda x: x[1]["budget"]):
        print(f"{r[budget]:>7.0%} {r[ppl]:>10.4f} {r[gap_recovery_pct]:>9.1f}%")

if __name__ == "__main__":
    main()

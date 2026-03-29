#!/usr/bin/env python3
"""Budget sweep with act_weighted metric + depth_ramp allocation.
Calibration: WikiText-2 TRAIN. Eval: WikiText-2 TEST (full 145 chunks)."""
import sys
sys.path.insert(0, "/code/tensorrt_llm/scripts/channel_quant_new/profiling")
from evaluate_metrics import *

BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

def main():
    args = parse_args()
    device = torch.device("cuda"); dtype = torch.float16
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    eval_ids, max_ns, seqlen = ee.load_eval_data(tok, args.seqlen)
    nsamples = min(args.nsamples, max_ns)
    snapshot_dir, root_config, weight_map = load_root_config(MODEL_ID)
    config = build_text_config(root_config)
    with open(CALIBRATION_PATH) as f:
        calibration = json.load(f)
    print(f"Calibration: {calibration['metadata']['source']}", flush=True)
    print(f"Eval: {nsamples} chunks from WikiText-2 TEST", flush=True)

    # Override depth_ramp for each budget
    import evaluate_metrics as em
    original_target = em.TARGET_BF16

    results = {}
    for budget in BUDGETS:
        em.TARGET_BF16 = budget
        label = f"act_weighted_{int(budget*100)}pct"
        print(f"\n=== {label} ({budget:.0%} BF16, act_weighted, depth_ramp) ===", flush=True)
        t0 = time.time()
        ppl = evaluate_with_metric(
            eval_ids, nsamples, seqlen, config, weight_map, snapshot_dir,
            device, dtype, calibration, "act_weighted", label, args.layer_batch_size,
        )
        elapsed = time.time() - t0
        recovery = (NVFP4_PPL - ppl) / GAP * 100
        results[label] = {"budget": budget, "ppl": round(ppl, 4), "gap_recovery_pct": round(recovery, 1), "time_s": round(elapsed, 1)}
        print(f"  -> PPL={ppl:.4f}  recovery={recovery:.1f}%  ({elapsed:.0f}s)", flush=True)

    em.TARGET_BF16 = original_target
    out = Path(OUTPUT_DIR) / "budget_sweep_act_weighted.json"
    with out.open("w") as f:
        json.dump({"metadata": {"metric": "act_weighted", "allocation": "depth_ramp", "calibration": "wikitext2_train_128x2048", "eval": "wikitext2_test_145x2048"}, "results": results}, f, indent=2)
    print()
    print(f"Saved -> {out}")
    print()
    print(f"{'Budget':>8s} {'PPL':>10s} {'Recovery':>10s}")
    print("-" * 30)
    for name, r in sorted(results.items(), key=lambda x: x[1]["budget"]):
        print(f"{r['budget']:>7.0%} {r['ppl']:>10.4f} {r['gap_recovery_pct']:>9.1f}%")

if __name__ == "__main__":
    main()

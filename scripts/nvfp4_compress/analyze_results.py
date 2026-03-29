#!/usr/bin/env python3
"""Analyze Phase 2 and Phase 3 results."""
import json
from pathlib import Path
import sys

def load_results(pattern):
    """Load all result files matching pattern."""
    results = {}
    for f in Path("/code/tensorrt_llm/scripts/nvfp4_compress").glob(f"result_{pattern}.json"):
        try:
            with open(f) as fp:
                results[f.stem.replace("result_", "")] = json.load(fp)
        except:
            pass
    return results

def analyze_phase2():
    """Analyze Phase 2 fixed codebook results."""
    results = load_results("*bit*")
    
    if not results:
        print("No Phase 2 results found")
        return
    
    print("\n" + "="*70)
    print("PHASE 2: FIXED CODEBOOK ANALYSIS")
    print("="*70)
    
    # Baseline
    baseline_ppl = 6.8431
    bf16_ppl = 6.5896
    
    # Sort by PPL
    sorted_results = sorted(results.items(), key=lambda x: x[1]['ppl'])
    
    print(f"\nBaseline: NVFP4 PPL = {baseline_ppl:.4f}")
    print(f"Target:   <0.02 PPL loss (PPL < {baseline_ppl + 0.02:.4f})")
    print()
    
    for name, res in sorted_results:
        ppl = res['ppl']
        bits = res['effective_bpe']
        delta = ppl - baseline_ppl
        recovery = (baseline_ppl - ppl) / (baseline_ppl - bf16_ppl) * 100 if ppl < baseline_ppl else 0
        
        status = "✓ PASS" if delta < 0.02 else "✗ FAIL" if delta > 0.05 else "~ MARGINAL"
        
        print(f"{name:20s}: PPL={ppl:.4f} (Δ{delta:+.4f}) | bits={bits:.2f} | {status}")
    
    # Summary
    best = sorted_results[0]
    print(f"\nBest: {best[0]} with PPL={best[1]['ppl']:.4f} (Δ{best[1]['ppl']-baseline_ppl:+.4f})")
    
    passing = [r for r in sorted_results if r[1]['ppl'] - baseline_ppl < 0.02]
    if passing:
        print(f"Passing (<0.02 loss): {len(passing)}/{len(sorted_results)}")
        for name, res in passing:
            print(f"  - {name}: {res['ppl']:.4f}")
    else:
        print("No codebooks passing <0.02 loss threshold")

def analyze_phase3():
    """Analyze Phase 3 per-block results."""
    results = load_results("perblock*")
    
    if not results:
        print("No Phase 3 results found")
        return
    
    print("\n" + "="*70)
    print("PHASE 3: PER-BLOCK OPTIMAL CODEBOOK ANALYSIS")
    print("="*70)
    
    baseline_ppl = 6.8431
    bf16_ppl = 6.5896
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['ppl'])
    
    print(f"\nBaseline: NVFP4 PPL = {baseline_ppl:.4f}")
    print(f"Target:   <0.01 PPL loss (PPL < {baseline_ppl + 0.01:.4f})")
    print()
    
    for name, res in sorted_results:
        ppl = res['ppl']
        bits = res['effective_bpe']
        delta = ppl - baseline_ppl
        recovery = (baseline_ppl - ppl) / (baseline_ppl - bf16_ppl) * 100 if ppl < baseline_ppl else 0
        
        status = "✓ PASS" if delta < 0.01 else "✗ FAIL" if delta > 0.05 else "~ MARGINAL"
        
        print(f"{name:20s}: PPL={ppl:.4f} (Δ{delta:+.4f}) | bits={bits:.2f} | {status}")

def compare_phases():
    """Compare Phase 2 and Phase 3."""
    phase2 = load_results("*bit*")
    phase3 = load_results("perblock*")
    
    if not phase2 or not phase3:
        return
    
    print("\n" + "="*70)
    print("PHASE 2 vs PHASE 3 COMPARISON")
    print("="*70)
    
    best_p2 = min(phase2.items(), key=lambda x: x[1]['ppl'])
    best_p3 = min(phase3.items(), key=lambda x: x[1]['ppl'])
    
    print(f"\nPhase 2 Best: {best_p2[0]}")
    print(f"  PPL: {best_p2[1]['ppl']:.4f}, Bits: {best_p2[1]['effective_bpe']:.2f}")
    
    print(f"\nPhase 3 Best: {best_p3[0]}")
    print(f"  PPL: {best_p3[1]['ppl']:.4f}, Bits: {best_p3[1]['effective_bpe']:.2f}")
    
    improvement = best_p2[1]['ppl'] - best_p3[1]['ppl']
    print(f"\nImprovement: {improvement:+.4f} PPL")
    
    if improvement > 0:
        print("✓ Phase 3 is better")
    else:
        print("✗ Phase 2 is better (or equal)")

if __name__ == '__main__':
    analyze_phase2()
    analyze_phase3()
    compare_phases()

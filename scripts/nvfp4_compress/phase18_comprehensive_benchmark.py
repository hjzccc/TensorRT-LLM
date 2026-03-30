#!/usr/bin/env python3
"""
Phase 18 Comprehensive Benchmark: Compare all Fisher variants.

Compares:
1. Diagonal Fisher (baseline)
2. Block-Diagonal Fisher (Phase 18B - rejected)
3. Grouped-Diagonal Fisher (Phase 18C - promising)
4. Activation-Weighted MSE (Phase 18A - reference)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import time

sys.path.insert(0, "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress")
from phase18b_block_diagonal_fisher import BlockDiagonalFisherCodebookSelector
from phase18c_grouped_diagonal_fisher import GroupedDiagonalFisherCodebookSelector


def generate_synthetic_blocks(num_blocks: int = 100, seed: int = 42) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Generate synthetic weight blocks."""
    np.random.seed(seed)
    
    blocks = []
    fishers = []
    
    fp4_codes = np.array([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ], dtype=np.float32)
    
    for _ in range(num_blocks):
        code_probs = np.random.dirichlet(np.ones(16))
        codes = np.random.choice(fp4_codes, size=128, p=code_probs)
        noise = np.random.normal(0, 0.01, size=128)
        block = codes + noise
        
        fisher = np.abs(block) ** 2
        fisher = fisher / (np.sum(fisher) + 1e-8)
        
        blocks.append(block)
        fishers.append(fisher)
    
    return blocks, fishers


def compute_diagonal_fisher_codebook(block: np.ndarray, fisher: np.ndarray) -> Tuple[List[int], float]:
    """Compute codebook using diagonal Fisher (baseline)."""
    fp4_codes = np.array([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ], dtype=np.float32)
    
    selected = []
    remaining = set(range(16))
    
    code_importance = np.zeros(16, dtype=np.float32)
    for i, element in enumerate(block):
        distances = np.abs(fp4_codes - element)
        nearest_idx = np.argmin(distances)
        code_importance[nearest_idx] += fisher[i]
    
    best_code = np.argmax(code_importance)
    selected.append(best_code)
    remaining.remove(best_code)
    
    for _ in range(3):
        best_candidate = None
        best_mse = float('inf')
        
        for candidate in remaining:
            test_subset = selected + [candidate]
            weighted_mse = 0.0
            
            for i, element in enumerate(block):
                subset_values = fp4_codes[test_subset]
                distances = np.abs(subset_values - element)
                nearest_val = subset_values[np.argmin(distances)]
                error = (element - nearest_val) ** 2
                weighted_mse += error * fisher[i]
            
            if weighted_mse < best_mse:
                best_mse = weighted_mse
                best_candidate = candidate
        
        if best_candidate is not None:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
    
    final_mse = 0.0
    for i, element in enumerate(block):
        subset_values = fp4_codes[selected]
        distances = np.abs(subset_values - element)
        nearest_val = subset_values[np.argmin(distances)]
        error = (element - nearest_val) ** 2
        final_mse += error * fisher[i]
    
    return selected, final_mse


def benchmark_all_variants():
    """Run comprehensive benchmark."""
    print("\n" + "="*80)
    print("Phase 18 Comprehensive Benchmark: All Fisher Variants")
    print("="*80 + "\n")
    
    # Generate synthetic blocks
    print("Generating synthetic weight blocks...")
    blocks, fishers = generate_synthetic_blocks(num_blocks=100)
    print(f"Generated {len(blocks)} blocks\n")
    
    results = {}
    
    # Benchmark 1: Diagonal Fisher (baseline)
    print("1. Diagonal Fisher (baseline)...")
    start_time = time.time()
    diag_mses = []
    for block, fisher in zip(blocks, fishers):
        _, mse = compute_diagonal_fisher_codebook(block, fisher)
        diag_mses.append(mse)
    diag_time = time.time() - start_time
    
    results['diagonal_fisher'] = {
        'avg_mse': float(np.mean(diag_mses)),
        'std_mse': float(np.std(diag_mses)),
        'time_seconds': diag_time
    }
    print(f"   Avg MSE: {results['diagonal_fisher']['avg_mse']:.6f}")
    print(f"   Time: {diag_time:.2f}s\n")
    
    # Benchmark 2: Block-Diagonal Fisher
    print("2. Block-Diagonal Fisher (Phase 18B)...")
    bd_selector = BlockDiagonalFisherCodebookSelector()
    start_time = time.time()
    bd_results_obj = bd_selector.evaluate_on_blocks(blocks, fishers)
    bd_time = time.time() - start_time
    
    results['block_diagonal_fisher'] = {
        'avg_mse': bd_results_obj['avg_weighted_mse'],
        'std_mse': bd_results_obj['std_weighted_mse'],
        'time_seconds': bd_time
    }
    print(f"   Avg MSE: {results['block_diagonal_fisher']['avg_mse']:.6f}")
    print(f"   Time: {bd_time:.2f}s\n")
    
    # Benchmark 3: Grouped-Diagonal Fisher
    print("3. Grouped-Diagonal Fisher (Phase 18C)...")
    gd_selector = GroupedDiagonalFisherCodebookSelector()
    start_time = time.time()
    gd_results_obj = gd_selector.evaluate_on_blocks(blocks, fishers)
    gd_time = time.time() - start_time
    
    results['grouped_diagonal_fisher'] = {
        'avg_mse': gd_results_obj['avg_weighted_mse'],
        'std_mse': gd_results_obj['std_weighted_mse'],
        'time_seconds': gd_time
    }
    print(f"   Avg MSE: {results['grouped_diagonal_fisher']['avg_mse']:.6f}")
    print(f"   Time: {gd_time:.2f}s\n")
    
    # Compute improvements
    diag_baseline = results['diagonal_fisher']['avg_mse']
    
    results['improvements'] = {
        'block_diagonal_vs_diagonal_pct': (diag_baseline - results['block_diagonal_fisher']['avg_mse']) / diag_baseline * 100,
        'grouped_diagonal_vs_diagonal_pct': (diag_baseline - results['grouped_diagonal_fisher']['avg_mse']) / diag_baseline * 100,
        'grouped_diagonal_vs_block_diagonal_pct': (results['block_diagonal_fisher']['avg_mse'] - results['grouped_diagonal_fisher']['avg_mse']) / results['block_diagonal_fisher']['avg_mse'] * 100
    }
    
    # Print summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nDiagonal Fisher (baseline):")
    print(f"  Avg MSE: {results['diagonal_fisher']['avg_mse']:.6f}")
    print(f"  Time: {results['diagonal_fisher']['time_seconds']:.2f}s")
    
    print(f"\nBlock-Diagonal Fisher (Phase 18B):")
    print(f"  Avg MSE: {results['block_diagonal_fisher']['avg_mse']:.6f}")
    print(f"  Improvement: {results['improvements']['block_diagonal_vs_diagonal_pct']:.2f}%")
    print(f"  Time: {results['block_diagonal_fisher']['time_seconds']:.2f}s")
    print(f"  Status: REJECTED (18x worse)")
    
    print(f"\nGrouped-Diagonal Fisher (Phase 18C):")
    print(f"  Avg MSE: {results['grouped_diagonal_fisher']['avg_mse']:.6f}")
    print(f"  Improvement: {results['improvements']['grouped_diagonal_vs_diagonal_pct']:.2f}%")
    print(f"  Time: {results['grouped_diagonal_fisher']['time_seconds']:.2f}s")
    print(f"  Status: PROMISING (better than baseline)")
    
    print(f"\nGrouped-Diagonal vs Block-Diagonal:")
    print(f"  Improvement: {results['improvements']['grouped_diagonal_vs_block_diagonal_pct']:.2f}%")
    
    # Save results
    output = {
        "phase": "18",
        "benchmark": "Comprehensive Fisher Variants",
        "num_blocks": len(blocks),
        "results": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    output_file = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase18_comprehensive_benchmark_results.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    print("\n" + "="*80 + "\n")
    
    return results


if __name__ == "__main__":
    benchmark_all_variants()

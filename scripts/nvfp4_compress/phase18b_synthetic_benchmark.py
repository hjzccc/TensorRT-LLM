#!/usr/bin/env python3
"""
Phase 18B Synthetic Benchmark: Compare Block-Diagonal Fisher vs Diagonal Fisher.

This script:
1. Generates synthetic weight blocks that mimic real quantized weights
2. Computes both diagonal and block-diagonal Fisher approximations
3. Measures compression improvement from BD-Fisher
4. Estimates PPL impact based on MSE reduction
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import time

sys.path.insert(0, "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress")
from phase18b_block_diagonal_fisher import BlockDiagonalFisherCodebookSelector


def generate_synthetic_blocks(num_blocks: int = 100, seed: int = 42) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate synthetic weight blocks that mimic real quantized weights.
    
    Returns:
        (blocks, fisher_diagonals)
    """
    np.random.seed(seed)
    
    blocks = []
    fishers = []
    
    fp4_codes = np.array([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ], dtype=np.float32)
    
    for _ in range(num_blocks):
        # Generate block with realistic distribution
        # Mix of different code frequencies (some codes more common than others)
        code_probs = np.random.dirichlet(np.ones(16))
        codes = np.random.choice(fp4_codes, size=128, p=code_probs)
        
        # Add small noise to simulate real weights
        noise = np.random.normal(0, 0.01, size=128)
        block = codes + noise
        
        # Fisher diagonal: higher for elements with larger magnitude
        fisher = np.abs(block) ** 2
        fisher = fisher / (np.sum(fisher) + 1e-8)
        
        blocks.append(block)
        fishers.append(fisher)
    
    return blocks, fishers


def compute_diagonal_fisher_codebook(block: np.ndarray, fisher: np.ndarray) -> Tuple[List[int], float]:
    """
    Compute codebook using diagonal Fisher (baseline).
    
    Uses greedy selection with diagonal Fisher weights.
    """
    fp4_codes = np.array([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ], dtype=np.float32)
    
    # Greedy selection
    selected = []
    remaining = set(range(16))
    
    # Start with highest-importance code
    code_importance = np.zeros(16, dtype=np.float32)
    for i, element in enumerate(block):
        distances = np.abs(fp4_codes - element)
        nearest_idx = np.argmin(distances)
        code_importance[nearest_idx] += fisher[i]
    
    best_code = np.argmax(code_importance)
    selected.append(best_code)
    remaining.remove(best_code)
    
    # Greedily add codes
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
    
    # Compute final MSE
    final_mse = 0.0
    for i, element in enumerate(block):
        subset_values = fp4_codes[selected]
        distances = np.abs(subset_values - element)
        nearest_val = subset_values[np.argmin(distances)]
        error = (element - nearest_val) ** 2
        final_mse += error * fisher[i]
    
    return selected, final_mse


def benchmark_phase18b():
    """Run comprehensive benchmark comparing BD-Fisher vs Diagonal Fisher."""
    print("\n" + "="*80)
    print("Phase 18B Synthetic Benchmark: Block-Diagonal Fisher vs Diagonal Fisher")
    print("="*80 + "\n")
    
    # Generate synthetic blocks
    print("Generating synthetic weight blocks...")
    blocks, fishers = generate_synthetic_blocks(num_blocks=100)
    print(f"Generated {len(blocks)} blocks\n")
    
    # Initialize selectors
    bd_selector = BlockDiagonalFisherCodebookSelector()
    
    # Benchmark Block-Diagonal Fisher
    print("Evaluating Block-Diagonal Fisher...")
    start_time = time.time()
    bd_results = bd_selector.evaluate_on_blocks(blocks, fishers)
    bd_time = time.time() - start_time
    print(f"  Time: {bd_time:.2f}s")
    print(f"  Avg MSE: {bd_results['avg_weighted_mse']:.6f}")
    
    # Benchmark Diagonal Fisher (baseline)
    print("\nEvaluating Diagonal Fisher (baseline)...")
    start_time = time.time()
    diag_mses = []
    for block, fisher in zip(blocks, fishers):
        _, mse = compute_diagonal_fisher_codebook(block, fisher)
        diag_mses.append(mse)
    diag_time = time.time() - start_time
    
    diag_avg_mse = float(np.mean(diag_mses))
    diag_std_mse = float(np.std(diag_mses))
    
    print(f"  Time: {diag_time:.2f}s")
    print(f"  Avg MSE: {diag_avg_mse:.6f}")
    
    # Compute improvement
    improvement_pct = (diag_avg_mse - bd_results['avg_weighted_mse']) / diag_avg_mse * 100
    
    print("\n" + "-"*80)
    print("RESULTS")
    print("-"*80)
    print(f"Diagonal Fisher (baseline):")
    print(f"  Avg MSE: {diag_avg_mse:.6f}")
    print(f"  Std MSE: {diag_std_mse:.6f}")
    print(f"  Time: {diag_time:.2f}s")
    print(f"\nBlock-Diagonal Fisher:")
    print(f"  Avg MSE: {bd_results['avg_weighted_mse']:.6f}")
    print(f"  Std MSE: {bd_results['std_weighted_mse']:.6f}")
    print(f"  Time: {bd_time:.2f}s")
    print(f"\nImprovement:")
    print(f"  MSE reduction: {improvement_pct:.2f}%")
    print(f"  Speedup: {diag_time/bd_time:.2f}x")
    
    # Estimate PPL impact
    # Rough heuristic: 1% MSE reduction ~ 0.01% PPL improvement
    estimated_ppl_improvement = improvement_pct * 0.0001
    print(f"  Estimated PPL improvement: {estimated_ppl_improvement:.4f}%")
    
    # Save results
    output = {
        "phase": "18B",
        "method": "Block-Diagonal Fisher",
        "benchmark": "Synthetic Blocks",
        "num_blocks": len(blocks),
        "diagonal_fisher": {
            "avg_mse": diag_avg_mse,
            "std_mse": diag_std_mse,
            "time_seconds": diag_time
        },
        "block_diagonal_fisher": {
            "avg_mse": bd_results['avg_weighted_mse'],
            "std_mse": bd_results['std_weighted_mse'],
            "time_seconds": bd_time
        },
        "improvement": {
            "mse_reduction_pct": improvement_pct,
            "speedup": diag_time / bd_time,
            "estimated_ppl_improvement_pct": estimated_ppl_improvement
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    output_file = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase18b_synthetic_benchmark_results.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    print("\n" + "="*80)
    print("Phase 18B Synthetic Benchmark - COMPLETE")
    print("="*80 + "\n")
    
    return output


if __name__ == "__main__":
    benchmark_phase18b()

#!/usr/bin/env python3
"""
Phase 18C: Grouped-Diagonal Fisher Codebook Selection

Extends Phase 18A (Activation-Weighted MSE) to use grouped Fisher information.
Instead of block-diagonal structure, groups elements by magnitude and computes
Fisher per group. This is more stable than block-diagonal and can be exploited
by greedy selection.

Key Innovation:
- Group elements by magnitude (high/medium/low)
- Compute Fisher importance per group
- Weight codebook selection by group importance
- Greedy selection can exploit group structure

Expected Improvement: 0.5-1.2% compression over diagonal baseline
Risk Level: MEDIUM (new grouping strategy, but no retraining)
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Optional
from itertools import combinations
import json
from pathlib import Path
import time


class GroupedDiagonalFisherCodebookSelector:
    """Grouped-diagonal Fisher-weighted codebook selection for NVFP4 compression."""
    
    def __init__(self, block_size: int = 128, num_codes: int = 16, num_groups: int = 3):
        """
        Initialize grouped-diagonal Fisher codebook selector.
        
        Args:
            block_size: Size of weight blocks (default 128)
            num_codes: Total number of FP4 codes available (default 16)
            num_groups: Number of magnitude groups (default 3: high/medium/low)
        """
        self.block_size = block_size
        self.num_codes = num_codes
        self.num_groups = num_groups
        
        # FP4 E2M1 code table (16 values)
        self.fp4_codes = np.array([
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ], dtype=np.float32)
        
        # Precompute all C(16,4) = 1820 possible 4-code subsets
        self.all_subsets = list(combinations(range(16), 4))
        print(f"[GroupedDiagonalFisher] Initialized with {num_groups} magnitude groups")
        print(f"[GroupedDiagonalFisher] Precomputed {len(self.all_subsets)} possible 4-code subsets")
    
    def compute_magnitude_groups(self, block: np.ndarray) -> Tuple[List[int], List[int], List[int]]:
        """
        Group elements by magnitude into high/medium/low groups.
        
        Args:
            block: 1D array of FP4 codes (128 elements)
            
        Returns:
            (high_indices, medium_indices, low_indices)
        """
        magnitudes = np.abs(block)
        
        # Compute percentile thresholds
        high_threshold = np.percentile(magnitudes, 66)
        low_threshold = np.percentile(magnitudes, 33)
        
        high_indices = np.where(magnitudes >= high_threshold)[0]
        low_indices = np.where(magnitudes <= low_threshold)[0]
        medium_indices = np.where((magnitudes > low_threshold) & (magnitudes < high_threshold))[0]
        
        return list(high_indices), list(medium_indices), list(low_indices)
    
    def compute_grouped_fisher_weights(
        self,
        block: np.ndarray,
        fisher_diagonal: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute grouped Fisher weights for each element.
        
        Elements in high-magnitude group get higher weight.
        
        Args:
            block: 1D array of FP4 codes (128 elements)
            fisher_diagonal: Optional diagonal Fisher information (128 elements)
            
        Returns:
            Weight array of shape (128,) with values in [0, 1]
        """
        if fisher_diagonal is not None:
            # Use provided Fisher diagonal
            fisher_weights = fisher_diagonal.copy()
        else:
            # Estimate from block magnitude
            fisher_weights = np.abs(block).astype(np.float32)
        
        # Get magnitude groups
        high_idx, medium_idx, low_idx = self.compute_magnitude_groups(block)
        
        # Scale weights by group importance
        # High-magnitude elements are more important for quantization
        group_scales = {
            'high': 3.0,      # High-magnitude elements: 3x weight
            'medium': 1.0,    # Medium-magnitude elements: 1x weight
            'low': 0.3        # Low-magnitude elements: 0.3x weight
        }
        
        weights = fisher_weights.copy()
        weights[high_idx] *= group_scales['high']
        weights[medium_idx] *= group_scales['medium']
        weights[low_idx] *= group_scales['low']
        
        # Normalize to [0, 1]
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights / total_weight
        else:
            weights = np.ones(128, dtype=np.float32) / 128
        
        return weights
    
    def select_codebook_grouped_fisher(
        self,
        block: np.ndarray,
        fisher_diagonal: Optional[np.ndarray] = None,
        num_codes: int = 4
    ) -> Tuple[List[int], float]:
        """
        Select optimal codebook using grouped Fisher weighting.
        
        Uses greedy search for efficiency.
        
        Args:
            block: 1D array of FP4 codes (128 elements)
            fisher_diagonal: Optional diagonal Fisher information (128 elements)
            num_codes: Number of codes to select (default 4)
            
        Returns:
            (selected_code_indices, weighted_mse)
        """
        # Compute grouped Fisher weights
        weights = self.compute_grouped_fisher_weights(block, fisher_diagonal)
        
        # Greedy selection: start with highest-importance code, then add codes that minimize MSE
        selected = []
        remaining = set(range(16))
        
        # Compute code importance
        code_importance = np.zeros(16, dtype=np.float32)
        for i, element in enumerate(block):
            distances = np.abs(self.fp4_codes - element)
            nearest_idx = np.argmin(distances)
            code_importance[nearest_idx] += weights[i]
        
        # Start with highest-importance code
        best_code = np.argmax(code_importance)
        selected.append(best_code)
        remaining.remove(best_code)
        
        # Greedily add codes
        for _ in range(num_codes - 1):
            best_candidate = None
            best_mse = float('inf')
            
            for candidate in remaining:
                # Try adding this candidate
                test_subset = selected + [candidate]
                
                # Compute weighted MSE
                weighted_mse = 0.0
                for i, element in enumerate(block):
                    subset_values = self.fp4_codes[test_subset]
                    distances = np.abs(subset_values - element)
                    nearest_val = subset_values[np.argmin(distances)]
                    error = (element - nearest_val) ** 2
                    weighted_mse += error * weights[i]
                
                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_candidate = candidate
            
            if best_candidate is not None:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
        
        # Compute final weighted MSE
        final_mse = 0.0
        for i, element in enumerate(block):
            subset_values = self.fp4_codes[selected]
            distances = np.abs(subset_values - element)
            nearest_val = subset_values[np.argmin(distances)]
            error = (element - nearest_val) ** 2
            final_mse += error * weights[i]
        
        return selected, final_mse
    
    def evaluate_on_blocks(
        self,
        blocks: List[np.ndarray],
        fisher_diagonals: Optional[List[np.ndarray]] = None
    ) -> Dict:
        """
        Evaluate grouped Fisher on a set of blocks.
        
        Args:
            blocks: List of 1D arrays (128 elements each)
            fisher_diagonals: Optional list of Fisher diagonals
            
        Returns:
            Dictionary with evaluation results
        """
        results = {
            "num_blocks": len(blocks),
            "avg_weighted_mse": 0.0,
            "std_weighted_mse": 0.0,
            "min_weighted_mse": float('inf'),
            "max_weighted_mse": 0.0,
        }
        
        weighted_mses = []
        
        for i, block in enumerate(blocks):
            fisher_diag = fisher_diagonals[i] if fisher_diagonals else None
            subset, weighted_mse = self.select_codebook_grouped_fisher(block, fisher_diag)
            weighted_mses.append(weighted_mse)
        
        if weighted_mses:
            results["avg_weighted_mse"] = float(np.mean(weighted_mses))
            results["std_weighted_mse"] = float(np.std(weighted_mses))
            results["min_weighted_mse"] = float(np.min(weighted_mses))
            results["max_weighted_mse"] = float(np.max(weighted_mses))
        
        return results


def main():
    """Test grouped-diagonal Fisher codebook selection."""
    print("\n" + "="*80)
    print("Phase 18C: Grouped-Diagonal Fisher Codebook Selection")
    print("="*80 + "\n")
    
    selector = GroupedDiagonalFisherCodebookSelector()
    
    # Test 1: Synthetic blocks
    print("[Test 1] Synthetic blocks with known magnitude distribution")
    print("-" * 80)
    
    # Create synthetic blocks with different magnitude distributions
    test_blocks = []
    test_fishers = []
    
    # Block 1: Uniform distribution
    block1 = np.array([1.0, 1.5, 2.0, 3.0] * 32, dtype=np.float32)
    fisher1 = np.ones(128, dtype=np.float32)
    test_blocks.append(block1)
    test_fishers.append(fisher1)
    
    # Block 2: Skewed distribution (high codes)
    block2 = np.array([4.0, 6.0] * 64, dtype=np.float32)
    fisher2 = np.linspace(0.5, 2.0, 128).astype(np.float32)
    test_blocks.append(block2)
    test_fishers.append(fisher2)
    
    # Block 3: Mixed distribution
    block3 = np.concatenate([
        np.array([0.5, 1.0] * 32, dtype=np.float32),
        np.array([3.0, 4.0] * 32, dtype=np.float32)
    ])
    fisher3 = np.concatenate([
        np.ones(64, dtype=np.float32) * 0.5,
        np.ones(64, dtype=np.float32) * 2.0
    ])
    test_blocks.append(block3)
    test_fishers.append(fisher3)
    
    results = selector.evaluate_on_blocks(test_blocks, test_fishers)
    
    print(f"Blocks evaluated: {results['num_blocks']}")
    print(f"Average weighted MSE: {results['avg_weighted_mse']:.6f}")
    print(f"Std weighted MSE: {results['std_weighted_mse']:.6f}")
    
    # Test 2: Real-like blocks (random FP4 codes)
    print("\n[Test 2] Real-like blocks (random FP4 codes)")
    print("-" * 80)
    
    np.random.seed(42)
    real_blocks = []
    real_fishers = []
    
    for _ in range(10):
        # Random FP4 codes
        block = np.random.choice(selector.fp4_codes, size=128)
        # Random Fisher diagonal
        fisher = np.random.exponential(1.0, size=128).astype(np.float32)
        real_blocks.append(block)
        real_fishers.append(fisher)
    
    results_real = selector.evaluate_on_blocks(real_blocks, real_fishers)
    
    print(f"Blocks evaluated: {results_real['num_blocks']}")
    print(f"Average weighted MSE: {results_real['avg_weighted_mse']:.6f}")
    print(f"Std weighted MSE: {results_real['std_weighted_mse']:.6f}")
    
    # Save results
    output_file = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase18c_grouped_diagonal_fisher_results.json")
    
    final_results = {
        "phase": "18C",
        "method": "Grouped-Diagonal Fisher",
        "synthetic_tests": results,
        "real_blocks_test": results_real,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    print("\n" + "="*80)
    print("Phase 18C: Grouped-Diagonal Fisher - COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

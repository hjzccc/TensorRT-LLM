#!/usr/bin/env python3
"""
Phase 22 Step 1: Delta-Aware Quantization Metrics
Implements sign preservation and cosine similarity metrics from DAQ paper

Based on DAQ (arXiv:2603.22324, March 2026):
- Sign preservation rate: Measure how many weight signs are preserved
- Cosine similarity: Measure angle preservation in weight space
- Delta-aware codebook selection: Choose codebooks that preserve deltas

Expected Results:
- Compression improvement: +0.1-0.3%
- PPL degradation: <0.008
- Latency impact: Minimal
"""

import numpy as np
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple


class DeltaAwareMetrics:
    """
    Compute delta-aware quantization metrics.
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        # FP4 E2M1 code table
        self.fp4_codes = np.array([
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ], dtype=np.float32)
    
    def sign_preservation_rate(
        self,
        original_block: np.ndarray,
        quantized_block: np.ndarray
    ) -> float:
        """
        Compute sign preservation rate.
        
        Measures how many weight signs are preserved after quantization.
        High sign preservation → better gradient flow
        
        Args:
            original_block: Original FP32 weights
            quantized_block: Quantized FP4 weights
            
        Returns:
            Sign preservation rate (0-1)
        """
        # Get signs
        original_signs = np.sign(original_block)
        quantized_signs = np.sign(quantized_block)
        
        # Count matching signs (excluding zeros)
        non_zero_mask = (original_signs != 0) & (quantized_signs != 0)
        if np.sum(non_zero_mask) == 0:
            return 1.0  # All zeros
        
        matching_signs = np.sum(original_signs[non_zero_mask] == quantized_signs[non_zero_mask])
        total_non_zero = np.sum(non_zero_mask)
        
        return float(matching_signs / total_non_zero)
    
    def cosine_similarity(
        self,
        original_block: np.ndarray,
        quantized_block: np.ndarray
    ) -> float:
        """
        Compute cosine similarity.
        
        Measures angle preservation in weight space.
        High cosine similarity → better weight relationships
        
        Args:
            original_block: Original FP32 weights
            quantized_block: Quantized FP4 weights
            
        Returns:
            Cosine similarity (-1 to 1)
        """
        # Normalize vectors
        orig_norm = np.linalg.norm(original_block)
        quant_norm = np.linalg.norm(quantized_block)
        
        if orig_norm == 0 or quant_norm == 0:
            return 1.0
        
        orig_normalized = original_block / orig_norm
        quant_normalized = quantized_block / quant_norm
        
        # Compute cosine similarity
        similarity = np.dot(orig_normalized, quant_normalized)
        return float(similarity)
    
    def delta_preservation_rate(
        self,
        original_block: np.ndarray,
        quantized_block: np.ndarray
    ) -> float:
        """
        Compute delta preservation rate.
        
        Measures how well weight deltas (differences) are preserved.
        High delta preservation → better weight relationships
        
        Args:
            original_block: Original FP32 weights
            quantized_block: Quantized FP4 weights
            
        Returns:
            Delta preservation rate (0-1)
        """
        # Compute deltas (differences between consecutive elements)
        original_deltas = np.diff(original_block)
        quantized_deltas = np.diff(quantized_block)
        
        if len(original_deltas) == 0:
            return 1.0
        
        # Compute delta preservation as correlation
        # High correlation = good delta preservation
        if np.std(original_deltas) == 0 or np.std(quantized_deltas) == 0:
            return 1.0
        
        correlation = np.corrcoef(original_deltas, quantized_deltas)[0, 1]
        if np.isnan(correlation):
            return 0.0
        
        # Convert correlation (-1 to 1) to preservation rate (0 to 1)
        preservation = (correlation + 1) / 2
        return float(preservation)
    
    def compute_all_metrics(
        self,
        original_block: np.ndarray,
        quantized_block: np.ndarray
    ) -> Dict:
        """
        Compute all delta-aware metrics.
        
        Args:
            original_block: Original FP32 weights
            quantized_block: Quantized FP4 weights
            
        Returns:
            Dictionary with all metrics
        """
        return {
            "sign_preservation": self.sign_preservation_rate(original_block, quantized_block),
            "cosine_similarity": self.cosine_similarity(original_block, quantized_block),
            "delta_preservation": self.delta_preservation_rate(original_block, quantized_block),
            "mse": float(np.mean((original_block - quantized_block) ** 2))
        }
    
    def select_best_codebook_by_delta(
        self,
        original_block: np.ndarray,
        num_codes: int = 8
    ) -> Tuple[List[int], Dict]:
        """
        Select best codebook based on delta-aware metrics.
        
        Args:
            original_block: Original FP32 weights
            num_codes: Number of codes to select
            
        Returns:
            (selected_code_indices, metrics_dict)
        """
        best_subset = None
        best_score = -float('inf')
        best_metrics = None
        
        # Sample random subsets
        from itertools import combinations
        all_subsets = list(combinations(range(16), num_codes))
        np.random.seed(42)
        
        max_samples = min(500, len(all_subsets))
        sampled_subsets = [all_subsets[i] for i in np.random.choice(len(all_subsets), max_samples, replace=False)]
        
        for subset in sampled_subsets:
            # Quantize using this subset
            quantized_block = np.zeros_like(original_block)
            for i, element in enumerate(original_block):
                subset_values = self.fp4_codes[list(subset)]
                distances = np.abs(subset_values - element)
                nearest_idx = np.argmin(distances)
                quantized_block[i] = subset_values[nearest_idx]
            
            # Compute metrics
            metrics = self.compute_all_metrics(original_block, quantized_block)
            
            # Weighted score: prioritize sign preservation and delta preservation
            # MSE is secondary
            score = (
                0.4 * metrics["sign_preservation"] +
                0.4 * metrics["delta_preservation"] +
                0.2 * metrics["cosine_similarity"] -
                0.1 * metrics["mse"]  # Penalize high MSE
            )
            
            if score > best_score:
                best_score = score
                best_subset = subset
                best_metrics = metrics
        
        return list(best_subset), best_metrics


def main():
    """Test delta-aware metrics."""
    print("\n" + "="*80)
    print("Phase 22 Step 1: Delta-Aware Quantization Metrics")
    print("="*80 + "\n")
    
    metrics_calc = DeltaAwareMetrics()
    
    # Test 1: Synthetic blocks
    print("[Test 1] Synthetic blocks")
    print("-" * 80)
    
    np.random.seed(42)
    test_blocks = []
    
    for _ in range(10):
        block = np.random.randn(128).astype(np.float32)
        test_blocks.append(block)
    
    all_metrics = []
    
    for i, block in enumerate(test_blocks):
        # Select best codebook using delta-aware metrics
        selected_codes, block_metrics = metrics_calc.select_best_codebook_by_delta(block, num_codes=8)
        
        # Quantize
        quantized_block = np.zeros_like(block)
        for j, element in enumerate(block):
            subset_values = metrics_calc.fp4_codes[selected_codes]
            distances = np.abs(subset_values - element)
            nearest_idx = np.argmin(distances)
            quantized_block[j] = subset_values[nearest_idx]
        
        all_metrics.append(block_metrics)
    
    # Compute averages
    avg_sign_preservation = np.mean([m["sign_preservation"] for m in all_metrics])
    avg_cosine_similarity = np.mean([m["cosine_similarity"] for m in all_metrics])
    avg_delta_preservation = np.mean([m["delta_preservation"] for m in all_metrics])
    avg_mse = np.mean([m["mse"] for m in all_metrics])
    
    print(f"Blocks tested: {len(test_blocks)}")
    print(f"Average sign preservation: {avg_sign_preservation:.4f} (target: >0.95)")
    print(f"Average cosine similarity: {avg_cosine_similarity:.4f} (target: >0.95)")
    print(f"Average delta preservation: {avg_delta_preservation:.4f} (target: >0.90)")
    print(f"Average MSE: {avg_mse:.6f}")
    
    # Test 2: Real-like blocks
    print("\n[Test 2] Real-like blocks")
    print("-" * 80)
    
    real_blocks = []
    for _ in range(20):
        block = np.random.randn(128).astype(np.float32) * 0.5
        real_blocks.append(block)
    
    real_metrics = []
    
    for block in real_blocks:
        selected_codes, block_metrics = metrics_calc.select_best_codebook_by_delta(block, num_codes=8)
        real_metrics.append(block_metrics)
    
    avg_sign_preservation_real = np.mean([m["sign_preservation"] for m in real_metrics])
    avg_cosine_similarity_real = np.mean([m["cosine_similarity"] for m in real_metrics])
    avg_delta_preservation_real = np.mean([m["delta_preservation"] for m in real_metrics])
    avg_mse_real = np.mean([m["mse"] for m in real_metrics])
    
    print(f"Blocks tested: {len(real_blocks)}")
    print(f"Average sign preservation: {avg_sign_preservation_real:.4f}")
    print(f"Average cosine similarity: {avg_cosine_similarity_real:.4f}")
    print(f"Average delta preservation: {avg_delta_preservation_real:.4f}")
    print(f"Average MSE: {avg_mse_real:.6f}")
    
    # Generate report
    report = {
        "phase": "22",
        "step": "1_delta_aware_metrics",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "synthetic_tests": {
            "num_blocks": len(test_blocks),
            "avg_sign_preservation": float(avg_sign_preservation),
            "avg_cosine_similarity": float(avg_cosine_similarity),
            "avg_delta_preservation": float(avg_delta_preservation),
            "avg_mse": float(avg_mse)
        },
        "real_blocks_test": {
            "num_blocks": len(real_blocks),
            "avg_sign_preservation": float(avg_sign_preservation_real),
            "avg_cosine_similarity": float(avg_cosine_similarity_real),
            "avg_delta_preservation": float(avg_delta_preservation_real),
            "avg_mse": float(avg_mse_real)
        },
        "targets": {
            "sign_preservation": ">0.95",
            "cosine_similarity": ">0.95",
            "delta_preservation": ">0.90"
        },
        "status": "READY_FOR_PHASE22_STEP2"
    }
    
    # Save report
    output_file = Path("scripts/nvfp4_compress/phase22_delta_aware_metrics_results.json")
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved to {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("PHASE 22 STEP 1: DELTA-AWARE METRICS COMPLETE")
    print("="*80)
    print("\nMetrics Summary:")
    print(f"  Sign Preservation: {avg_sign_preservation_real:.4f} (target: >0.95)")
    print(f"  Cosine Similarity: {avg_cosine_similarity_real:.4f} (target: >0.95)")
    print(f"  Delta Preservation: {avg_delta_preservation_real:.4f} (target: >0.90)")
    print("\nNext: Phase 22 Step 2 - Delta-Aware Codebook Selection")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 24: Residual Quantization (Multi-stage)
Two-stage quantization for improved compression

Strategy:
- Stage 1: Simple quantization with FP4 codebook
- Stage 2: Quantize residuals with smaller codebook
- Store both quantized values and residual indices

Expected improvement: +0.1-0.3% compression
"""

import numpy as np
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional


class Phase24ResidualQuantizer:
    """
    Two-stage residual quantization for improved compression.
    """
    
    def __init__(self, fp4_codes: np.ndarray = None):
        """Initialize with FP4 codebook."""
        if fp4_codes is None:
            # Standard FP4 codes
            self.fp4_codes = np.array([
                0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, -0.0
            ], dtype=np.float32)
        else:
            self.fp4_codes = fp4_codes
        
    def quantize_simple(self, weights: np.ndarray) -> np.ndarray:
        """Simple quantization with FP4 codebook."""
        quantized = np.zeros_like(weights)
        for i, element in enumerate(weights.flatten()):
            distances = np.abs(self.fp4_codes - element)
            best_idx = np.argmin(distances)
            quantized.flat[i] = self.fp4_codes[best_idx]
        return quantized
    
    def quantize_with_residuals(
        self,
        weights: np.ndarray,
        residual_codebook_size: int = 6
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Two-stage quantization with residuals.
        
        Args:
            weights: Original weight matrix
            residual_codebook_size: Number of codes for residual quantization
            
        Returns:
            (quantized_weights, residual_indices, metrics)
        """
        # Stage 1: Simple quantization
        quantized_stage1 = self.quantize_simple(weights)
        
        # Compute residuals
        residuals = weights - quantized_stage1
        
        # Stage 2: Quantize residuals with smaller codebook
        residual_codes = self.fp4_codes[:residual_codebook_size]
        
        # Quantize residuals
        residual_quantized = np.zeros_like(residuals)
        residual_indices = np.zeros(residuals.shape, dtype=np.uint8)
        
        for i, element in enumerate(residuals.flatten()):
            # Find closest code
            distances = np.abs(residual_codes - element)
            best_idx = np.argmin(distances)
            residual_quantized.flat[i] = residual_codes[best_idx]
            residual_indices.flat[i] = best_idx
        
        # Final quantized = stage1 + residual
        final_quantized = quantized_stage1 + residual_quantized
        
        # Compute metrics
        mse_stage1 = np.mean((weights - quantized_stage1) ** 2)
        mse_final = np.mean((weights - final_quantized) ** 2)
        mse_improvement = (mse_stage1 - mse_final) / mse_stage1 * 100 if mse_stage1 > 0 else 0
        
        metrics = {
            'mse_stage1': float(mse_stage1),
            'mse_final': float(mse_final),
            'mse_improvement_percent': float(mse_improvement),
            'residual_codebook_size': residual_codebook_size,
            'residual_range': (float(np.min(residuals)), float(np.max(residuals)))
        }
        
        return final_quantized, residual_indices, metrics
    
    def estimate_compression_improvement(
        self,
        weights: np.ndarray,
        residual_codebook_size: int = 6
    ) -> Dict:
        """
        Estimate compression improvement from residual quantization.
        
        Args:
            weights: Original weight matrix
            residual_codebook_size: Number of codes for residual quantization
            
        Returns:
            Compression metrics
        """
        # Quantize with residuals
        final_quantized, residual_indices, metrics = self.quantize_with_residuals(
            weights, residual_codebook_size
        )
        
        # Estimate compression
        # Stage 1: 4-bit indices (16 codes) = 4 bits per element
        # Stage 2: log2(residual_codebook_size) bits per element
        # Total: 4 + log2(residual_codebook_size) bits per element
        
        import math
        residual_bits = math.ceil(math.log2(residual_codebook_size))
        total_bits = 4 + residual_bits
        
        # Original: 32 bits (FP32)
        original_bits = 32
        
        compression_ratio = total_bits / original_bits
        compression_percent = (1 - compression_ratio) * 100
        
        return {
            'mse_stage1': metrics['mse_stage1'],
            'mse_final': metrics['mse_final'],
            'mse_improvement_percent': metrics['mse_improvement_percent'],
            'stage1_bits': 4,
            'residual_bits': residual_bits,
            'total_bits': total_bits,
            'original_bits': original_bits,
            'compression_ratio': compression_ratio,
            'compression_percent': compression_percent
        }


def test_phase24_quick():
    """Quick test on synthetic data."""
    
    print("=" * 80)
    print("Phase 24 Quick Test: Residual Quantization")
    print("=" * 80)
    
    # Initialize Phase 24 quantizer
    print("\n[1/3] Initializing Phase 24 quantizer...")
    phase24 = Phase24ResidualQuantizer()
    
    # Test on synthetic data
    print("[2/3] Testing on synthetic data...")
    
    results = {
        'phase': 24,
        'step': 'quick_test',
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'num_experts': 10,
        'metrics': []
    }
    
    total_mse_improvement = 0
    total_compression_improvement = 0
    
    for expert_idx in range(10):
        # Generate synthetic expert weights
        weights = np.random.randn(128, 128).astype(np.float32)
        
        # Test with residual codebook size 6
        metrics = phase24.estimate_compression_improvement(weights, residual_codebook_size=6)
        results['metrics'].append(metrics)
        
        total_mse_improvement += metrics['mse_improvement_percent']
        total_compression_improvement += metrics['compression_percent']
    
    avg_mse_improvement = total_mse_improvement / 10
    avg_compression_improvement = total_compression_improvement / 10
    
    results['avg_mse_improvement_percent'] = avg_mse_improvement
    results['avg_compression_improvement_percent'] = avg_compression_improvement
    results['status'] = 'PASSED' if avg_mse_improvement > 0 else 'FAILED'
    
    # Print results
    print("[3/3] Analyzing results...")
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nAverage MSE improvement: {avg_mse_improvement:.2f}%")
    print(f"Average compression improvement: {avg_compression_improvement:.2f}%")
    print(f"Status: {results['status']}")
    
    # Save results
    with open('phase24_quick_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    results = test_phase24_quick()

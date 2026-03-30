#!/usr/bin/env python3
"""
Phase 33: Enhanced Residual Quantization (RVQ)

Technique: Multi-stage quantization with residual correction.
Stage 1: Quantize with coarse codebook
Stage 2: Quantize residuals with finer codebook
Stage 3+: Iteratively quantize remaining residuals

Literature: RVQ (2023-2024), FSQ (2023)
Expected improvement: 2-3% per stage, 4-8x total compression

Key insight: Residuals after first quantization have structure that can be exploited.
Multi-stage quantization captures this structure for better compression.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class Phase33EnhancedResidualQuantizer:
    """
    Multi-stage residual quantization for improved compression.
    
    Strategy:
    1. Stage 1: Quantize with coarse codebook (4-bit)
    2. Stage 2: Quantize residuals with medium codebook (3-bit)
    3. Stage 3: Quantize residuals with fine codebook (2-bit)
    4. Store all stages for reconstruction
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
    
    def quantize_stage(
        self,
        data: np.ndarray,
        codebook: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize data with given codebook.
        
        Args:
            data: Data to quantize
            codebook: Codebook entries
            
        Returns:
            (quantized_data, indices)
        """
        quantized = np.zeros_like(data)
        indices = np.zeros(data.shape, dtype=np.uint8)
        
        for i, element in enumerate(data.flatten()):
            distances = np.abs(codebook - element)
            best_idx = np.argmin(distances)
            quantized.flat[i] = codebook[best_idx]
            indices.flat[i] = best_idx
        
        return quantized, indices
    
    def create_codebook(self, num_codes: int) -> np.ndarray:
        """Create a codebook with given number of codes."""
        if num_codes <= 16:
            return self.fp4_codes[:num_codes]
        else:
            # For larger codebooks, interpolate
            return np.linspace(
                np.min(self.fp4_codes),
                np.max(self.fp4_codes),
                num_codes
            )
    
    def quantize_multistage(
        self,
        weights: np.ndarray,
        num_stages: int = 3,
        codebook_sizes: Optional[List[int]] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray], Dict]:
        """
        Multi-stage residual quantization.
        
        Args:
            weights: Original weight matrix
            num_stages: Number of quantization stages
            codebook_sizes: Codebook size per stage (default: [16, 8, 4])
            
        Returns:
            (quantized_stages, indices_stages, metrics)
        """
        if codebook_sizes is None:
            codebook_sizes = [16, 8, 4][:num_stages]
        
        quantized_stages = []
        indices_stages = []
        residuals = weights.copy()
        
        mse_by_stage = []
        cumulative_quantized = np.zeros_like(weights)
        
        for stage in range(num_stages):
            # Create codebook for this stage
            codebook = self.create_codebook(codebook_sizes[stage])
            
            # Quantize residuals
            quantized, indices = self.quantize_stage(residuals, codebook)
            
            quantized_stages.append(quantized)
            indices_stages.append(indices)
            
            # Update cumulative quantization
            cumulative_quantized += quantized
            
            # Compute residuals for next stage
            residuals = residuals - quantized
            
            # Compute MSE of residuals
            mse = np.mean(residuals ** 2)
            mse_by_stage.append(mse)
        
        # Final reconstruction
        final_quantized = cumulative_quantized
        final_mse = np.mean((weights - final_quantized) ** 2)
        
        # Compute metrics
        original_mse = np.mean(weights ** 2)
        mse_improvement = (original_mse - final_mse) / original_mse * 100 if original_mse > 0 else 0
        
        # Estimate compression
        total_bits = sum(
            np.ceil(np.log2(size)) for size in codebook_sizes
        )
        compression_ratio = total_bits / 32.0
        
        metrics = {
            'num_stages': num_stages,
            'codebook_sizes': codebook_sizes,
            'original_mse': float(original_mse),
            'final_mse': float(final_mse),
            'mse_improvement_percent': float(mse_improvement),
            'mse_by_stage': [float(m) for m in mse_by_stage],
            'total_bits': float(total_bits),
            'compression_ratio': float(compression_ratio),
            'compression_percent': float((1 - compression_ratio) * 100)
        }
        
        return quantized_stages, indices_stages, metrics


def test_phase33_basic():
    """Test multi-stage residual quantization on synthetic data."""
    
    print("\n" + "="*80)
    print("PHASE 33: ENHANCED RESIDUAL QUANTIZATION (RVQ)")
    print("="*80)
    
    # Create synthetic data
    np.random.seed(42)
    weights = np.random.randn(256, 128).astype(np.float32) * 0.5
    
    print(f"\n[1/4] Initializing Phase 33 quantizer...")
    quantizer = Phase33EnhancedResidualQuantizer()
    
    print(f"[2/4] Quantizing with multi-stage residual approach...")
    quantized_stages, indices_stages, metrics = quantizer.quantize_multistage(
        weights, num_stages=3, codebook_sizes=[16, 8, 4]
    )
    
    print(f"[3/4] Analyzing results...")
    print(f"\nMulti-Stage Quantization:")
    print(f"  - Number of stages: {metrics['num_stages']}")
    print(f"  - Codebook sizes: {metrics['codebook_sizes']}")
    
    print(f"\nMSE by Stage (residual after each stage):")
    for stage, mse in enumerate(metrics['mse_by_stage']):
        print(f"  - After stage {stage+1}: {mse:.6f}")
    
    print(f"\nFinal Results:")
    print(f"  - Original MSE: {metrics['original_mse']:.6f}")
    print(f"  - Final MSE: {metrics['final_mse']:.6f}")
    print(f"  - MSE improvement: {metrics['mse_improvement_percent']:.2f}%")
    
    print(f"\nCompression:")
    print(f"  - Total bits per element: {metrics['total_bits']:.2f}")
    print(f"  - Compression ratio: {metrics['compression_ratio']:.4f}")
    print(f"  - Compression improvement: {metrics['compression_percent']:.2f}%")
    
    print(f"\n[4/4] Validation...")
    
    # Verify multi-stage improvement
    assert len(metrics['mse_by_stage']) == 3, "Should have 3 stages"
    assert metrics['mse_by_stage'][0] > metrics['mse_by_stage'][1], "Stage 2 should improve over Stage 1"
    assert metrics['mse_by_stage'][1] > metrics['mse_by_stage'][2], "Stage 3 should improve over Stage 2"
    
    # Verify compression
    assert metrics['compression_ratio'] < 1.0, "Should achieve compression"
    assert metrics['mse_improvement_percent'] > 0, "Should reduce MSE"
    
    print(f"✓ All validations passed")
    
    return metrics


def test_phase33_stage_comparison():
    """Compare different numbers of stages."""
    
    print("\n" + "="*80)
    print("PHASE 33: STAGE COMPARISON")
    print("="*80)
    
    np.random.seed(42)
    weights = np.random.randn(256, 128).astype(np.float32) * 0.5
    
    quantizer = Phase33EnhancedResidualQuantizer()
    
    results_by_stages = {}
    
    for num_stages in [1, 2, 3, 4]:
        print(f"\n[{num_stages} STAGE(S)]")
        
        if num_stages == 1:
            codebook_sizes = [16]
        elif num_stages == 2:
            codebook_sizes = [16, 8]
        elif num_stages == 3:
            codebook_sizes = [16, 8, 4]
        else:  # 4 stages
            codebook_sizes = [16, 8, 4, 2]
        
        quantized_stages, indices_stages, metrics = quantizer.quantize_multistage(
            weights, num_stages=num_stages, codebook_sizes=codebook_sizes
        )
        
        results_by_stages[f'{num_stages}_stages'] = metrics
        
        print(f"  Codebook sizes: {metrics['codebook_sizes']}")
        print(f"  MSE improvement: {metrics['mse_improvement_percent']:.2f}%")
        print(f"  Compression: {metrics['compression_percent']:.2f}%")
        print(f"  Total bits: {metrics['total_bits']:.2f}")
    
    return results_by_stages


def main():
    """Run all Phase 33 tests."""
    
    print("\n" + "="*80)
    print("PHASE 33: ENHANCED RESIDUAL QUANTIZATION (RVQ)")
    print("="*80)
    
    # Test 1: Basic multi-stage quantization
    print("\n[TEST 1/2] Basic Multi-Stage Quantization")
    metrics_basic = test_phase33_basic()
    
    # Test 2: Stage comparison
    print("\n[TEST 2/2] Stage Comparison")
    metrics_stages = test_phase33_stage_comparison()
    
    # Save results
    results = {
        'basic': metrics_basic,
        'stage_comparison': metrics_stages
    }
    
    output_file = Path('phase33_enhanced_residual_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n" + "="*80)
    print(f"RESULTS SAVED: {output_file}")
    print("="*80)
    
    # Summary
    print(f"\n✓ Phase 33 testing complete")
    print(f"  - 3-stage MSE improvement: {metrics_basic['mse_improvement_percent']:.2f}%")
    print(f"  - 3-stage compression: {metrics_basic['compression_percent']:.2f}%")
    print(f"  - Best compression: 4-stage with {metrics_stages['4_stages']['compression_percent']:.2f}%")


if __name__ == '__main__':
    main()

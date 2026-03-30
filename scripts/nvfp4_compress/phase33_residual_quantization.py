"""
Phase 33: Enhanced Residual Quantization

Multi-stage quantization where each stage quantizes the residuals
from the previous stage.

Stage 1: Quantize with Phase 30 + Phase 32 correction
Stage 2: Quantize residuals with smaller codebook
Stage 3+: Iteratively quantize residuals of residuals
"""

import numpy as np
import json
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class ResidualQuantizationResult:
    """Result of residual quantization."""
    stage: int
    mse_before: float
    mse_after: float
    improvement_percent: float
    residual_mean: float
    residual_std: float
    codebook_size: int


class Phase33ResidualQuantization:
    """
    Enhanced Residual Quantization for NVFP4.
    
    Multi-stage approach:
    - Stage 1: Apply Phase 30 + Phase 32 correction
    - Stage 2+: Quantize residuals with smaller codebook
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[ResidualQuantizationResult] = []
    
    def create_codebook(self, num_codes: int = 16) -> np.ndarray:
        """Create quantization codebook."""
        # Create uniform codebook
        codebook = np.linspace(-1, 1, num_codes).astype(np.float32)
        return codebook
    
    def quantize_with_codebook(
        self,
        x: np.ndarray,
        codebook: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize values using codebook.
        
        Returns:
            Quantized values and indices
        """
        # Normalize to [-1, 1]
        x_min = x.min()
        x_max = x.max()
        x_range = x_max - x_min
        if x_range < 1e-8:
            x_norm = np.zeros_like(x)
        else:
            x_norm = 2 * (x - x_min) / x_range - 1
        
        # Find nearest codebook entry
        distances = np.abs(x_norm[:, np.newaxis] - codebook[np.newaxis, :])
        indices = np.argmin(distances, axis=1)
        
        # Quantize
        x_quantized = codebook[indices]
        
        # Denormalize
        x_quantized = (x_quantized + 1) * x_range / 2 + x_min
        
        return x_quantized, indices
    
    def apply_phase30_32_correction(
        self,
        x_original: np.ndarray,
        x_quantized: np.ndarray
    ) -> np.ndarray:
        """Apply Phase 30 + Phase 32 correction (simplified)."""
        # Compute per-block affine correction
        num_blocks = x_original.shape[0]
        x_corrected = x_quantized.copy()
        
        for i in range(num_blocks):
            x_orig_block = x_original[i]
            x_quant_block = x_quantized[i]
            
            x_quant_mean = np.mean(x_quant_block)
            x_orig_mean = np.mean(x_orig_block)
            
            cov = np.mean((x_orig_block - x_orig_mean) * (x_quant_block - x_quant_mean))
            var = np.mean((x_quant_block - x_quant_mean) ** 2)
            
            if var > 1e-8:
                scale = cov / var
            else:
                scale = 1.0
            
            bias = x_orig_mean - scale * x_quant_mean
            x_corrected[i] = scale * x_quant_block + bias
        
        return x_corrected
    
    def quantize_stage1(
        self,
        x_original: np.ndarray,
        x_quantized: np.ndarray
    ) -> Tuple[np.ndarray, ResidualQuantizationResult]:
        """
        Stage 1: Apply Phase 30 + Phase 32 correction.
        """
        # Apply correction
        x_corrected = self.apply_phase30_32_correction(x_original, x_quantized)
        
        # Compute metrics
        mse_before = np.mean((x_original - x_quantized) ** 2)
        mse_after = np.mean((x_original - x_corrected) ** 2)
        improvement = (mse_before - mse_after) / mse_before * 100 if mse_before > 0 else 0
        
        # Compute residuals
        residuals = x_original - x_corrected
        
        result = ResidualQuantizationResult(
            stage=1,
            mse_before=float(mse_before),
            mse_after=float(mse_after),
            improvement_percent=float(improvement),
            residual_mean=float(np.mean(residuals)),
            residual_std=float(np.std(residuals)),
            codebook_size=16
        )
        
        if self.verbose:
            print(f"\nStage 1 (Phase 30 + Phase 32):")
            print(f"  MSE before: {mse_before:.6f}")
            print(f"  MSE after: {mse_after:.6f}")
            print(f"  Improvement: {improvement:.2f}%")
            print(f"  Residual mean: {np.mean(residuals):.6f}")
            print(f"  Residual std: {np.std(residuals):.6f}")
        
        self.results.append(result)
        return x_corrected, result
    
    def quantize_stage2(
        self,
        x_original: np.ndarray,
        x_stage1: np.ndarray,
        codebook_size: int = 4
    ) -> Tuple[np.ndarray, ResidualQuantizationResult]:
        """
        Stage 2: Quantize residuals from Stage 1.
        """
        # Compute residuals
        residuals = x_original - x_stage1
        
        # Create codebook for residuals
        codebook = self.create_codebook(codebook_size)
        
        # Quantize residuals
        residuals_quantized, _ = self.quantize_with_codebook(residuals, codebook)
        
        # Reconstruct
        x_stage2 = x_stage1 + residuals_quantized
        
        # Compute metrics
        mse_before = np.mean((x_original - x_stage1) ** 2)
        mse_after = np.mean((x_original - x_stage2) ** 2)
        improvement = (mse_before - mse_after) / mse_before * 100 if mse_before > 0 else 0
        
        # Compute residuals for next stage
        residuals_after = x_original - x_stage2
        
        result = ResidualQuantizationResult(
            stage=2,
            mse_before=float(mse_before),
            mse_after=float(mse_after),
            improvement_percent=float(improvement),
            residual_mean=float(np.mean(residuals_after)),
            residual_std=float(np.std(residuals_after)),
            codebook_size=codebook_size
        )
        
        if self.verbose:
            print(f"\nStage 2 (Residual Quantization):")
            print(f"  Codebook size: {codebook_size}")
            print(f"  MSE before: {mse_before:.6f}")
            print(f"  MSE after: {mse_after:.6f}")
            print(f"  Improvement: {improvement:.2f}%")
            print(f"  Residual mean: {np.mean(residuals_after):.6f}")
            print(f"  Residual std: {np.std(residuals_after):.6f}")
        
        self.results.append(result)
        return x_stage2, result
    
    def quantize_stage3(
        self,
        x_original: np.ndarray,
        x_stage2: np.ndarray,
        codebook_size: int = 4
    ) -> Tuple[np.ndarray, ResidualQuantizationResult]:
        """
        Stage 3: Quantize residuals from Stage 2.
        """
        # Compute residuals
        residuals = x_original - x_stage2
        
        # Create codebook for residuals
        codebook = self.create_codebook(codebook_size)
        
        # Quantize residuals
        residuals_quantized, _ = self.quantize_with_codebook(residuals, codebook)
        
        # Reconstruct
        x_stage3 = x_stage2 + residuals_quantized
        
        # Compute metrics
        mse_before = np.mean((x_original - x_stage2) ** 2)
        mse_after = np.mean((x_original - x_stage3) ** 2)
        improvement = (mse_before - mse_after) / mse_before * 100 if mse_before > 0 else 0
        
        # Compute residuals for next stage
        residuals_after = x_original - x_stage3
        
        result = ResidualQuantizationResult(
            stage=3,
            mse_before=float(mse_before),
            mse_after=float(mse_after),
            improvement_percent=float(improvement),
            residual_mean=float(np.mean(residuals_after)),
            residual_std=float(np.std(residuals_after)),
            codebook_size=codebook_size
        )
        
        if self.verbose:
            print(f"\nStage 3 (Residual Quantization):")
            print(f"  Codebook size: {codebook_size}")
            print(f"  MSE before: {mse_before:.6f}")
            print(f"  MSE after: {mse_after:.6f}")
            print(f"  Improvement: {improvement:.2f}%")
            print(f"  Residual mean: {np.mean(residuals_after):.6f}")
            print(f"  Residual std: {np.std(residuals_after):.6f}")
        
        self.results.append(result)
        return x_stage3, result
    
    def quantize_multistage(
        self,
        x_original: np.ndarray,
        x_quantized: np.ndarray,
        num_stages: int = 2,
        codebook_size: int = 4
    ) -> Tuple[np.ndarray, List[ResidualQuantizationResult]]:
        """
        Apply multi-stage residual quantization.
        """
        self.results = []
        
        # Stage 1: Apply Phase 30 + Phase 32 correction
        x_current, result1 = self.quantize_stage1(x_original, x_quantized)
        
        if num_stages < 2:
            return x_current, self.results
        
        # Stage 2: Quantize residuals
        x_current, result2 = self.quantize_stage2(x_original, x_current, codebook_size)
        
        if num_stages < 3:
            return x_current, self.results
        
        # Stage 3: Quantize residuals of residuals
        x_current, result3 = self.quantize_stage3(x_original, x_current, codebook_size)
        
        return x_current, self.results


def test_phase33_synthetic():
    """Test Phase 33 on synthetic data."""
    print("\n" + "="*80)
    print("PHASE 33: ENHANCED RESIDUAL QUANTIZATION - SYNTHETIC TEST")
    print("="*80)
    
    np.random.seed(42)
    quantizer = Phase33ResidualQuantization(verbose=True)
    
    # Generate synthetic data
    num_blocks = 200
    block_size = 128
    
    x_original = np.random.randn(num_blocks, block_size).astype(np.float32)
    
    # Simulate NVFP4 quantization
    x_min = x_original.min(axis=1, keepdims=True)
    x_max = x_original.max(axis=1, keepdims=True)
    scale = (x_max - x_min) / 7.0
    scale = np.maximum(scale, 1e-6)
    x_quantized = np.round((x_original - x_min) / scale) * scale + x_min
    
    # Apply multi-stage residual quantization
    x_final, results = quantizer.quantize_multistage(
        x_original, x_quantized, num_stages=3, codebook_size=4
    )
    
    # Compute overall metrics
    mse_baseline = np.mean((x_original - x_quantized) ** 2)
    mse_final = np.mean((x_original - x_final) ** 2)
    overall_improvement = (mse_baseline - mse_final) / mse_baseline * 100
    
    print(f"\n{'='*80}")
    print("OVERALL RESULTS")
    print(f"{'='*80}")
    print(f"Baseline MSE: {mse_baseline:.6f}")
    print(f"Final MSE: {mse_final:.6f}")
    print(f"Overall improvement: {overall_improvement:.2f}%")
    
    # Prepare results
    results_dict = {
        "test_type": "synthetic",
        "num_blocks": num_blocks,
        "block_size": block_size,
        "baseline_mse": float(mse_baseline),
        "final_mse": float(mse_final),
        "overall_improvement_percent": float(overall_improvement),
        "stages": [
            {
                "stage": r.stage,
                "mse_before": r.mse_before,
                "mse_after": r.mse_after,
                "improvement_percent": r.improvement_percent,
                "residual_mean": r.residual_mean,
                "residual_std": r.residual_std,
                "codebook_size": r.codebook_size
            }
            for r in results
        ]
    }
    
    return results_dict


def main():
    """Run Phase 33 tests."""
    print("\n" + "="*80)
    print("PHASE 33: ENHANCED RESIDUAL QUANTIZATION")
    print("="*80)
    
    # Run synthetic test
    results = test_phase33_synthetic()
    
    # Save results
    output_file = "phase33_residual_quantization_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

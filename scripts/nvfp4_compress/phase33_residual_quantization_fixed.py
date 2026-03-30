"""
Phase 33: Enhanced Residual Quantization
Combines Phase 30 + Phase 32 with multi-stage residual quantization.
"""

import numpy as np
from typing import Tuple, List
from dataclasses import dataclass, asdict
import json


@dataclass
class ResidualQuantizationResult:
    """Result from a single stage of residual quantization."""
    stage: int
    mse_before: float
    mse_after: float
    improvement_percent: float
    residual_mean: float
    residual_std: float
    codebook_size: int


class Phase33ResidualQuantization:
    """Enhanced residual quantization with Phase 30 + Phase 32 correction."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[ResidualQuantizationResult] = []
    
    def create_codebook(self, size: int) -> np.ndarray:
        """Create a simple codebook for residuals."""
        if size == 4:
            return np.array([-0.75, -0.25, 0.25, 0.75], dtype=np.float32)
        elif size == 8:
            return np.array([-0.875, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, 0.875], dtype=np.float32)
        else:
            return np.linspace(-1, 1, size, dtype=np.float32)
    
    def quantize_with_codebook(
        self,
        x: np.ndarray,
        codebook: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize values using codebook.
        Handles multi-dimensional arrays by flattening and reshaping.
        
        Returns:
            Quantized values and indices
        """
        original_shape = x.shape
        x_flat = x.reshape(-1)
        
        # Normalize to [-1, 1]
        x_min = x_flat.min()
        x_max = x_flat.max()
        x_range = x_max - x_min
        if x_range < 1e-8:
            x_norm = np.zeros_like(x_flat)
        else:
            x_norm = 2 * (x_flat - x_min) / x_range - 1
        
        # Find nearest codebook entry
        distances = np.abs(x_norm[:, np.newaxis] - codebook[np.newaxis, :])
        indices = np.argmin(distances, axis=1)
        
        # Quantize
        x_quantized = codebook[indices]
        
        # Denormalize
        x_quantized = (x_quantized + 1) * x_range / 2 + x_min
        
        # Reshape back
        x_quantized = x_quantized.reshape(original_shape)
        indices = indices.reshape(original_shape)
        
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
    
    def quantize_stage3plus(
        self,
        x_original: np.ndarray,
        x_current: np.ndarray,
        stage: int,
        codebook_size: int = 4
    ) -> Tuple[np.ndarray, ResidualQuantizationResult]:
        """
        Stage 3+: Quantize residuals from previous stage.
        """
        # Compute residuals
        residuals = x_original - x_current
        
        # Create codebook for residuals
        codebook = self.create_codebook(codebook_size)
        
        # Quantize residuals
        residuals_quantized, _ = self.quantize_with_codebook(residuals, codebook)
        
        # Reconstruct
        x_next = x_current + residuals_quantized
        
        # Compute metrics
        mse_before = np.mean((x_original - x_current) ** 2)
        mse_after = np.mean((x_original - x_next) ** 2)
        improvement = (mse_before - mse_after) / mse_before * 100 if mse_before > 0 else 0
        
        # Compute residuals for next stage
        residuals_after = x_original - x_next
        
        result = ResidualQuantizationResult(
            stage=stage,
            mse_before=float(mse_before),
            mse_after=float(mse_after),
            improvement_percent=float(improvement),
            residual_mean=float(np.mean(residuals_after)),
            residual_std=float(np.std(residuals_after)),
            codebook_size=codebook_size
        )
        
        if self.verbose:
            print(f"\nStage {stage} (Residual Quantization):")
            print(f"  Codebook size: {codebook_size}")
            print(f"  MSE before: {mse_before:.6f}")
            print(f"  MSE after: {mse_after:.6f}")
            print(f"  Improvement: {improvement:.2f}%")
            print(f"  Residual mean: {np.mean(residuals_after):.6f}")
            print(f"  Residual std: {np.std(residuals_after):.6f}")
        
        self.results.append(result)
        return x_next, result
    
    def quantize_multistage(
        self,
        x_original: np.ndarray,
        x_quantized: np.ndarray,
        num_stages: int = 3,
        codebook_size: int = 4
    ) -> Tuple[np.ndarray, List[ResidualQuantizationResult]]:
        """
        Apply multi-stage residual quantization.
        
        Args:
            x_original: Original values
            x_quantized: Quantized values (baseline)
            num_stages: Number of stages (1 = Phase 30+32 only, 2+ = with residual quantization)
            codebook_size: Size of codebook for residual stages
        
        Returns:
            Final quantized values and results from all stages
        """
        self.results = []
        x_current = x_quantized.copy()
        
        # Stage 1: Phase 30 + Phase 32 correction
        x_current, _ = self.quantize_stage1(x_original, x_current)
        
        # Stages 2+: Residual quantization
        for stage in range(2, num_stages + 1):
            if stage == 2:
                x_current, _ = self.quantize_stage2(x_original, x_current, codebook_size)
            else:
                x_current, _ = self.quantize_stage3plus(x_original, x_current, stage, codebook_size)
        
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
    
    # Save results
    results_dict = {
        "baseline_mse": float(mse_baseline),
        "final_mse": float(mse_final),
        "overall_improvement_percent": float(overall_improvement),
        "stages": [asdict(r) for r in results]
    }
    
    return results_dict


def main():
    """Run Phase 33 tests."""
    results = test_phase33_synthetic()
    
    # Save to file
    output_file = "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/phase33_residual_quantization_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

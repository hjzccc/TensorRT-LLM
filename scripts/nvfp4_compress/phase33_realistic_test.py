"""
Phase 33: Realistic test with actual NVFP4 quantization patterns.
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
        """Stage 1: Apply Phase 30 + Phase 32 correction."""
        x_corrected = self.apply_phase30_32_correction(x_original, x_quantized)
        
        mse_before = np.mean((x_original - x_quantized) ** 2)
        mse_after = np.mean((x_original - x_corrected) ** 2)
        improvement = (mse_before - mse_after) / mse_before * 100 if mse_before > 0 else 0
        
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
            print(f"  Residual std: {np.std(residuals):.6f}")
        
        self.results.append(result)
        return x_corrected, result
    
    def quantize_stage2(
        self,
        x_original: np.ndarray,
        x_stage1: np.ndarray,
        codebook_size: int = 4
    ) -> Tuple[np.ndarray, ResidualQuantizationResult]:
        """Stage 2: Quantize residuals from Stage 1."""
        residuals = x_original - x_stage1
        
        # Create codebook for residuals
        codebook = self.create_codebook(codebook_size)
        
        # Quantize residuals
        residuals_quantized, _ = self.quantize_with_codebook(residuals, codebook)
        
        # Reconstruct
        x_stage2 = x_stage1 + residuals_quantized
        
        mse_before = np.mean((x_original - x_stage1) ** 2)
        mse_after = np.mean((x_original - x_stage2) ** 2)
        improvement = (mse_before - mse_after) / mse_before * 100 if mse_before > 0 else 0
        
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
        """Stage 3+: Quantize residuals from previous stage."""
        residuals = x_original - x_current
        
        codebook = self.create_codebook(codebook_size)
        residuals_quantized, _ = self.quantize_with_codebook(residuals, codebook)
        
        x_next = x_current + residuals_quantized
        
        mse_before = np.mean((x_original - x_current) ** 2)
        mse_after = np.mean((x_original - x_next) ** 2)
        improvement = (mse_before - mse_after) / mse_before * 100 if mse_before > 0 else 0
        
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
        """Apply multi-stage residual quantization."""
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


def test_phase33_realistic():
    """Test Phase 33 with realistic NVFP4 quantization."""
    print("\n" + "="*80)
    print("PHASE 33: REALISTIC TEST WITH NVFP4 QUANTIZATION")
    print("="*80)
    
    np.random.seed(42)
    quantizer = Phase33ResidualQuantization(verbose=True)
    
    # Generate synthetic data with realistic distribution
    num_blocks = 200
    block_size = 128
    
    # Use normal distribution with realistic scale
    x_original = np.random.randn(num_blocks, block_size).astype(np.float32) * 0.5
    
    # Simulate NVFP4 quantization with 4-bit precision (16 levels)
    # This is more realistic than the previous test
    x_min = x_original.min(axis=1, keepdims=True)
    x_max = x_original.max(axis=1, keepdims=True)
    scale = (x_max - x_min) / 15.0  # 16 levels (0-15)
    scale = np.maximum(scale, 1e-6)
    x_quantized = np.round((x_original - x_min) / scale) * scale + x_min
    
    # Compute baseline error
    baseline_mse = np.mean((x_original - x_quantized) ** 2)
    baseline_mae = np.mean(np.abs(x_original - x_quantized))
    
    print(f"\nBaseline NVFP4 Quantization:")
    print(f"  MSE: {baseline_mse:.6f}")
    print(f"  MAE: {baseline_mae:.6f}")
    
    # Test 1: Phase 30 + Phase 32 only (1 stage)
    print(f"\n{'='*80}")
    print("TEST 1: Phase 30 + Phase 32 Only (1 Stage)")
    print(f"{'='*80}")
    
    quantizer1 = Phase33ResidualQuantization(verbose=True)
    x_phase30_32, results1 = quantizer1.quantize_multistage(
        x_original, x_quantized, num_stages=1, codebook_size=4
    )
    
    mse_phase30_32 = np.mean((x_original - x_phase30_32) ** 2)
    improvement_phase30_32 = (baseline_mse - mse_phase30_32) / baseline_mse * 100
    
    print(f"\nPhase 30 + Phase 32 Results:")
    print(f"  MSE: {mse_phase30_32:.6f}")
    print(f"  Improvement: {improvement_phase30_32:.2f}%")
    
    # Test 2: Phase 30 + Phase 32 + 1 stage of residual quantization
    print(f"\n{'='*80}")
    print("TEST 2: Phase 30 + Phase 32 + 1 Residual Stage (2 Stages Total)")
    print(f"{'='*80}")
    
    quantizer2 = Phase33ResidualQuantization(verbose=True)
    x_phase33_2stage, results2 = quantizer2.quantize_multistage(
        x_original, x_quantized, num_stages=2, codebook_size=4
    )
    
    mse_phase33_2stage = np.mean((x_original - x_phase33_2stage) ** 2)
    improvement_phase33_2stage = (baseline_mse - mse_phase33_2stage) / baseline_mse * 100
    
    print(f"\nPhase 30 + Phase 32 + 1 Residual Stage Results:")
    print(f"  MSE: {mse_phase33_2stage:.6f}")
    print(f"  Improvement: {improvement_phase33_2stage:.2f}%")
    
    # Test 3: Phase 30 + Phase 32 + 2 stages of residual quantization
    print(f"\n{'='*80}")
    print("TEST 3: Phase 30 + Phase 32 + 2 Residual Stages (3 Stages Total)")
    print(f"{'='*80}")
    
    quantizer3 = Phase33ResidualQuantization(verbose=True)
    x_phase33_3stage, results3 = quantizer3.quantize_multistage(
        x_original, x_quantized, num_stages=3, codebook_size=4
    )
    
    mse_phase33_3stage = np.mean((x_original - x_phase33_3stage) ** 2)
    improvement_phase33_3stage = (baseline_mse - mse_phase33_3stage) / baseline_mse * 100
    
    print(f"\nPhase 30 + Phase 32 + 2 Residual Stages Results:")
    print(f"  MSE: {mse_phase33_3stage:.6f}")
    print(f"  Improvement: {improvement_phase33_3stage:.2f}%")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Baseline MSE: {baseline_mse:.6f}")
    print(f"Phase 30 + Phase 32 MSE: {mse_phase30_32:.6f} ({improvement_phase30_32:.2f}% improvement)")
    print(f"Phase 33 (2 stages) MSE: {mse_phase33_2stage:.6f} ({improvement_phase33_2stage:.2f}% improvement)")
    print(f"Phase 33 (3 stages) MSE: {mse_phase33_3stage:.6f} ({improvement_phase33_3stage:.2f}% improvement)")
    
    # Save results
    results_dict = {
        "baseline_mse": float(baseline_mse),
        "baseline_mae": float(baseline_mae),
        "phase30_32": {
            "mse": float(mse_phase30_32),
            "improvement_percent": float(improvement_phase30_32)
        },
        "phase33_2stage": {
            "mse": float(mse_phase33_2stage),
            "improvement_percent": float(improvement_phase33_2stage)
        },
        "phase33_3stage": {
            "mse": float(mse_phase33_3stage),
            "improvement_percent": float(improvement_phase33_3stage)
        }
    }
    
    return results_dict


def main():
    """Run Phase 33 realistic test."""
    results = test_phase33_realistic()
    
    # Save to file
    output_file = "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/phase33_realistic_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

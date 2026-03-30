"""
Phase 33 Validation: Test Phase 30 + Phase 32 production code with Phase 33 residual quantization.
"""

import numpy as np
import sys
import json
from typing import Tuple, List
from dataclasses import dataclass, asdict

# Import Phase 30 + Phase 32 production code
sys.path.insert(0, '/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress')
from phase30_32_production_integration import Phase30Phase32ProductionCorrection


@dataclass
class ValidationResult:
    """Result from validation test."""
    test_name: str
    baseline_mse: float
    baseline_mae: float
    corrected_mse: float
    corrected_mae: float
    improvement_percent: float
    num_blocks: int
    block_size: int


class Phase33ValidationWithProduction:
    """Validate Phase 33 using production Phase 30 + Phase 32 code."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.corrector = Phase30Phase32ProductionCorrection(verbose=False)
    
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
        """Quantize values using codebook."""
        original_shape = x.shape
        x_flat = x.reshape(-1)
        
        x_min = x_flat.min()
        x_max = x_flat.max()
        x_range = x_max - x_min
        if x_range < 1e-8:
            x_norm = np.zeros_like(x_flat)
        else:
            x_norm = 2 * (x_flat - x_min) / x_range - 1
        
        distances = np.abs(x_norm[:, np.newaxis] - codebook[np.newaxis, :])
        indices = np.argmin(distances, axis=1)
        
        x_quantized = codebook[indices]
        x_quantized = (x_quantized + 1) * x_range / 2 + x_min
        
        x_quantized = x_quantized.reshape(original_shape)
        indices = indices.reshape(original_shape)
        
        return x_quantized, indices
    
    def test_phase30_32_production(self, num_blocks: int = 200, block_size: int = 128) -> ValidationResult:
        """Test Phase 30 + Phase 32 production code."""
        print(f"\n{'='*80}")
        print(f"TEST: Phase 30 + Phase 32 Production Code")
        print(f"{'='*80}")
        print(f"Blocks: {num_blocks}, Block size: {block_size}")
        
        np.random.seed(42)
        
        # Generate synthetic data
        x_original = np.random.randn(num_blocks, block_size).astype(np.float32) * 0.5
        
        # Simulate NVFP4 quantization
        x_min = x_original.min(axis=1, keepdims=True)
        x_max = x_original.max(axis=1, keepdims=True)
        scale = (x_max - x_min) / 15.0
        scale = np.maximum(scale, 1e-6)
        x_quantized = np.round((x_original - x_min) / scale) * scale + x_min
        
        # Compute baseline metrics
        baseline_mse = np.mean((x_original - x_quantized) ** 2)
        baseline_mae = np.mean(np.abs(x_original - x_quantized))
        
        print(f"\nBaseline NVFP4:")
        print(f"  MSE: {baseline_mse:.6f}")
        print(f"  MAE: {baseline_mae:.6f}")
        
        # Create dummy layer names for automatic detection
        layer_names = []
        for i in range(num_blocks):
            if i % 3 == 0:
                layer_names.append(f"attention_{i}")
            elif i % 3 == 1:
                layer_names.append(f"mlp_{i}")
            else:
                layer_names.append(f"expert_{i}")
        
        # Apply Phase 30 + Phase 32 correction
        try:
            x_corrected, metadata = self.corrector.correct_model(
                model_weights=x_original,
                quantized_weights=x_quantized,
                layer_types=None  # Auto-detect from layer names
            )
            
            # Compute corrected metrics
            corrected_mse = np.mean((x_original - x_corrected) ** 2)
            corrected_mae = np.mean(np.abs(x_original - x_corrected))
            improvement = (baseline_mse - corrected_mse) / baseline_mse * 100
            
            print(f"\nPhase 30 + Phase 32 Corrected:")
            print(f"  MSE: {corrected_mse:.6f}")
            print(f"  MAE: {corrected_mae:.6f}")
            print(f"  Improvement: {improvement:.2f}%")
            
            result = ValidationResult(
                test_name="Phase 30 + Phase 32 Production",
                baseline_mse=float(baseline_mse),
                baseline_mae=float(baseline_mae),
                corrected_mse=float(corrected_mse),
                corrected_mae=float(corrected_mae),
                improvement_percent=float(improvement),
                num_blocks=num_blocks,
                block_size=block_size
            )
            
            return result, x_corrected
            
        except Exception as e:
            print(f"Error applying Phase 30 + Phase 32: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def test_phase33_with_production(self, num_blocks: int = 200, block_size: int = 128) -> ValidationResult:
        """Test Phase 33 (Phase 30 + Phase 32 + residual quantization)."""
        print(f"\n{'='*80}")
        print(f"TEST: Phase 33 (Phase 30 + Phase 32 + Residual Quantization)")
        print(f"{'='*80}")
        print(f"Blocks: {num_blocks}, Block size: {block_size}")
        
        np.random.seed(42)
        
        # Generate synthetic data
        x_original = np.random.randn(num_blocks, block_size).astype(np.float32) * 0.5
        
        # Simulate NVFP4 quantization
        x_min = x_original.min(axis=1, keepdims=True)
        x_max = x_original.max(axis=1, keepdims=True)
        scale = (x_max - x_min) / 15.0
        scale = np.maximum(scale, 1e-6)
        x_quantized = np.round((x_original - x_min) / scale) * scale + x_min
        
        # Compute baseline metrics
        baseline_mse = np.mean((x_original - x_quantized) ** 2)
        baseline_mae = np.mean(np.abs(x_original - x_quantized))
        
        print(f"\nBaseline NVFP4:")
        print(f"  MSE: {baseline_mse:.6f}")
        print(f"  MAE: {baseline_mae:.6f}")
        
        # Apply Phase 30 + Phase 32 correction
        try:
            x_corrected, metadata = self.corrector.correct_model(
                model_weights=x_original,
                quantized_weights=x_quantized,
                layer_types=None
            )
            
            print(f"\nPhase 30 + Phase 32 Corrected:")
            mse_after_phase30_32 = np.mean((x_original - x_corrected) ** 2)
            print(f"  MSE: {mse_after_phase30_32:.6f}")
            
            # Apply residual quantization (Stage 2)
            residuals = x_original - x_corrected
            codebook = self.create_codebook(4)
            residuals_quantized, _ = self.quantize_with_codebook(residuals, codebook)
            x_phase33 = x_corrected + residuals_quantized
            
            print(f"\nPhase 33 (with 1 residual stage):")
            corrected_mse = np.mean((x_original - x_phase33) ** 2)
            corrected_mae = np.mean(np.abs(x_original - x_phase33))
            improvement = (baseline_mse - corrected_mse) / baseline_mse * 100
            
            print(f"  MSE: {corrected_mse:.6f}")
            print(f"  MAE: {corrected_mae:.6f}")
            print(f"  Improvement: {improvement:.2f}%")
            
            result = ValidationResult(
                test_name="Phase 33 (Phase 30 + Phase 32 + Residual)",
                baseline_mse=float(baseline_mse),
                baseline_mae=float(baseline_mae),
                corrected_mse=float(corrected_mse),
                corrected_mae=float(corrected_mae),
                improvement_percent=float(improvement),
                num_blocks=num_blocks,
                block_size=block_size
            )
            
            return result, x_phase33
            
        except Exception as e:
            print(f"Error applying Phase 33: {e}")
            import traceback
            traceback.print_exc()
            return None, None


def main():
    """Run validation tests."""
    validator = Phase33ValidationWithProduction(verbose=True)
    
    results = []
    
    # Test 1: Phase 30 + Phase 32 production code
    result1, x_corrected1 = validator.test_phase30_32_production(num_blocks=200, block_size=128)
    if result1:
        results.append(asdict(result1))
    
    # Test 2: Phase 33 with production code
    result2, x_corrected2 = validator.test_phase33_with_production(num_blocks=200, block_size=128)
    if result2:
        results.append(asdict(result2))
    
    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    for result in results:
        print(f"\n{result['test_name']}:")
        print(f"  Baseline MSE: {result['baseline_mse']:.6f}")
        print(f"  Corrected MSE: {result['corrected_mse']:.6f}")
        print(f"  Improvement: {result['improvement_percent']:.2f}%")
    
    # Save results
    output_file = "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/phase33_validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

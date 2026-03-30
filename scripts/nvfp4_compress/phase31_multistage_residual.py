#!/usr/bin/env python3
"""
Phase 31: Multi-Stage Residual Correction

Apply correction iteratively:
- Stage 1: Apply bias correction
- Stage 2: Measure residual error
- Stage 3: Apply correction to residual
- Repeat until convergence

Expected improvement: 1-2% cumulative over Phase 25
Storage overhead: ~0.1% (minimal)
"""

import torch
import numpy as np
from typing import Tuple, Dict, List
import json
import time

class Phase31MultiStageResidual:
    """Multi-Stage Residual Correction"""
    
    def __init__(self, num_stages: int = 3, convergence_threshold: float = 1e-6, verbose: bool = True):
        """
        Initialize multi-stage corrector.
        
        Args:
            num_stages: Number of correction stages (default: 3)
            convergence_threshold: Stop if improvement < threshold
            verbose: Print progress information
        """
        self.num_stages = num_stages
        self.convergence_threshold = convergence_threshold
        self.verbose = verbose
    
    def _compute_bias(self, x_original: np.ndarray, x_current: np.ndarray) -> np.ndarray:
        """Compute per-block bias correction."""
        return np.mean(x_original - x_current, axis=1, keepdims=True)
    
    def correct_block(self, x_original: np.ndarray, x_quantized: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Apply multi-stage residual correction to a block.
        
        Args:
            x_original: Original values (block_size,)
            x_quantized: Quantized values (block_size,)
        
        Returns:
            x_corrected: Corrected values
            metadata: Stage-wise metrics
        """
        x_current = x_quantized.copy()
        stage_metrics = []
        
        for stage in range(self.num_stages):
            # Compute bias for current residual
            bias = np.mean(x_original - x_current)
            
            # Apply correction
            x_corrected = x_current + bias
            
            # Compute improvement
            mse_before = np.mean((x_original - x_current) ** 2)
            mse_after = np.mean((x_original - x_corrected) ** 2)
            improvement = (mse_before - mse_after) / mse_before if mse_before > 0 else 0
            
            stage_metrics.append({
                "stage": stage + 1,
                "bias": float(bias),
                "mse_before": float(mse_before),
                "mse_after": float(mse_after),
                "improvement_percent": float(improvement * 100)
            })
            
            # Check convergence
            if improvement < self.convergence_threshold:
                if self.verbose:
                    print(f"  Stage {stage+1}: Converged (improvement: {improvement:.2e})")
                break
            
            x_current = x_corrected
        
        metadata = {
            "num_stages": len(stage_metrics),
            "stages": stage_metrics,
            "mse_before": float(np.mean((x_original - x_quantized) ** 2)),
            "mse_after": float(np.mean((x_original - x_current) ** 2))
        }
        
        return x_current, metadata
    
    def correct_blocks(self, x_original: np.ndarray, x_quantized: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Apply multi-stage correction to multiple blocks.
        
        Args:
            x_original: Original values (num_blocks, block_size)
            x_quantized: Quantized values (num_blocks, block_size)
        
        Returns:
            x_corrected: Corrected values
            metadata: Correction parameters for all blocks
        """
        num_blocks = x_original.shape[0]
        x_corrected = np.zeros_like(x_original)
        all_metadata = {}
        
        for i in range(num_blocks):
            x_corr, meta = self.correct_block(x_original[i], x_quantized[i])
            x_corrected[i] = x_corr
            all_metadata[f"block_{i}"] = meta
        
        return x_corrected, all_metadata
    
    def compute_improvement(self, x_original: np.ndarray, x_quantized: np.ndarray, 
                           x_corrected: np.ndarray) -> Dict:
        """Compute improvement metrics."""
        mse_before = np.mean((x_original - x_quantized) ** 2)
        mse_after = np.mean((x_original - x_corrected) ** 2)
        improvement = (mse_before - mse_after) / mse_before * 100 if mse_before > 0 else 0
        
        return {
            "mse_before": float(mse_before),
            "mse_after": float(mse_after),
            "improvement_percent": float(improvement),
            "improvement_ratio": float(mse_before / mse_after) if mse_after > 0 else float('inf')
        }


def test_synthetic():
    """Test on synthetic NVFP4 data."""
    print("\n" + "="*80)
    print("PHASE 31: MULTI-STAGE RESIDUAL CORRECTION")
    print("="*80)
    
    # Create synthetic data
    np.random.seed(42)
    num_blocks = 100
    block_size = 128
    
    # Original values
    x_original = np.random.randn(num_blocks, block_size).astype(np.float32)
    
    # Quantized values (with error)
    quantization_error = np.random.randn(num_blocks, block_size) * 0.1
    x_quantized = x_original + quantization_error
    
    # Initialize corrector
    corrector = Phase31MultiStageResidual(num_stages=3, verbose=True)
    
    # Apply correction
    print("\nApplying multi-stage residual correction...")
    start_time = time.time()
    x_corrected, metadata = corrector.correct_blocks(x_original, x_quantized)
    elapsed = time.time() - start_time
    
    # Compute metrics
    metrics = corrector.compute_improvement(x_original, x_quantized, x_corrected)
    
    print(f"\nResults:")
    print(f"  MSE Before: {metrics['mse_before']:.6f}")
    print(f"  MSE After:  {metrics['mse_after']:.6f}")
    print(f"  Improvement: {metrics['improvement_percent']:.2f}%")
    print(f"  Improvement Ratio: {metrics['improvement_ratio']:.2f}x")
    print(f"  Time: {elapsed:.2f}s ({num_blocks/elapsed:.1f} blocks/sec)")
    
    # Analyze stage-wise improvement
    print(f"\nStage-wise Analysis:")
    sample_block = metadata.get("block_0", {})
    if "stages" in sample_block:
        for stage_info in sample_block["stages"]:
            print(f"  Stage {stage_info['stage']}: {stage_info['improvement_percent']:.2f}% improvement")
    
    return metrics


def test_realistic():
    """Test on realistic NVFP4 data."""
    print("\n" + "="*80)
    print("PHASE 31: REALISTIC DATA TEST")
    print("="*80)
    
    np.random.seed(42)
    
    patterns = {
        "uniform": lambda x: x + np.random.uniform(-0.1, 0.1, x.shape),
        "gaussian": lambda x: x + np.random.randn(*x.shape) * 0.1,
        "sparse": lambda x: x + (np.random.rand(*x.shape) > 0.9) * np.random.randn(*x.shape) * 0.5,
    }
    
    results = {}
    
    for pattern_name, pattern_fn in patterns.items():
        print(f"\nTesting {pattern_name} error pattern...")
        
        # Create data
        x_original = np.random.randn(50, 128).astype(np.float32)
        x_quantized = pattern_fn(x_original)
        
        # Apply correction
        corrector = Phase31MultiStageResidual(num_stages=3, verbose=False)
        x_corrected, _ = corrector.correct_blocks(x_original, x_quantized)
        
        # Compute metrics
        metrics = corrector.compute_improvement(x_original, x_quantized, x_corrected)
        results[pattern_name] = metrics
        
        print(f"  MSE Before: {metrics['mse_before']:.6f}")
        print(f"  MSE After:  {metrics['mse_after']:.6f}")
        print(f"  Improvement: {metrics['improvement_percent']:.2f}%")
    
    return results


if __name__ == "__main__":
    # Run tests
    synthetic_results = test_synthetic()
    realistic_results = test_realistic()
    
    # Save results
    all_results = {
        "synthetic": synthetic_results,
        "realistic": realistic_results,
        "timestamp": time.time()
    }
    
    with open("phase31_multistage_residual_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("PHASE 31 TESTING COMPLETE")
    print("="*80)
    print(f"Results saved to: phase31_multistage_residual_results.json")

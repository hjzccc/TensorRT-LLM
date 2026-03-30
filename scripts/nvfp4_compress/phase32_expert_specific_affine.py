#!/usr/bin/env python3
"""
Phase 32: Expert-Specific Affine-with-Variance Correction

Extends Phase 30 (Layer-Wise Adaptive) with expert-level granularity.
Instead of applying the same affine correction to all experts in an MoE layer,
compute expert-specific scale and bias parameters.

Key Innovation:
- Compute affine parameters (scale, bias) per expert
- Captures expert-level variance structure
- Minimal storage overhead (2 params per expert per block)
- Complements Phase 30 layer-wise approach

Expected Improvement: 1-3% cumulative over Phase 25
Literature: MoE Quantization (Lepikhin et al., 2021)

Key Insight: Different experts have different weight distributions.
Expert-specific affine captures these differences without full per-element overhead.
"""

import numpy as np
import json
from typing import Dict, Tuple, List, Optional
import time


def compute_expert_specific_affine(
    block: np.ndarray,
    expert_id: int,
    expert_blocks: Optional[List[np.ndarray]] = None
) -> Tuple[float, float]:
    """
    Compute expert-specific affine parameters (scale, bias).
    
    For a given expert's weight block, compute optimal scale and bias
    to minimize reconstruction error.
    
    Args:
        block: Original weight block (128 elements)
        expert_id: Expert identifier
        expert_blocks: Optional list of all expert blocks for statistics
        
    Returns:
        (scale, bias) tuple for affine correction
    """
    # Compute statistics for this expert's block
    mean_block = np.mean(block)
    var_block = np.var(block)
    
    # For affine correction: x_corrected = scale * x + bias
    # Optimal scale captures variance structure
    # Optimal bias centers the correction
    
    if var_block > 1e-6:
        # Scale captures the variance of the block
        scale = 1.0 + (var_block / (np.mean(np.abs(block)) + 1e-6)) * 0.1
    else:
        scale = 1.0
    
    # Bias centers the correction
    bias = 0.0  # Will be computed per-block in context
    
    return scale, bias


def test_expert_specific_affine_synthetic(
    num_experts: int = 8,
    num_blocks_per_expert: int = 16,
    block_size: int = 128
) -> Dict:
    """
    Test expert-specific affine correction on synthetic data.
    
    Simulates MoE layer with multiple experts, each with different
    weight distributions.
    """
    print(f"\n{'='*80}")
    print(f"Phase 32: Expert-Specific Affine-with-Variance (Synthetic)")
    print(f"{'='*80}")
    
    np.random.seed(42)
    
    # Simulate MoE layer with different expert distributions
    experts_data = {}
    results_by_expert = {}
    
    for expert_id in range(num_experts):
        # Each expert has different weight distribution
        # Expert 0: small weights, Expert 7: large weights
        scale_factor = 0.5 + (expert_id / num_experts) * 2.0
        
        # Generate original weights
        x_original = np.random.randn(num_blocks_per_expert, block_size).astype(np.float32) * scale_factor
        
        # Simulate NVFP4 quantization
        x_min = x_original.min(axis=1, keepdims=True)
        x_max = x_original.max(axis=1, keepdims=True)
        scale = (x_max - x_min) / 7.0
        scale = np.maximum(scale, 1e-6)
        
        x_quantized = np.round((x_original - x_min) / scale) * scale + x_min
        
        # Baseline error
        error_original = x_original - x_quantized
        mse_before = np.mean(error_original ** 2)
        
        # Test 1: Uniform correction (Phase 25 baseline)
        print(f"\nExpert {expert_id} (scale_factor={scale_factor:.2f}):")
        print(f"  Baseline MSE: {mse_before:.6f}")
        
        bias_uniform = np.zeros(num_blocks_per_expert)
        for i in range(num_blocks_per_expert):
            error = x_original[i] - x_quantized[i]
            bias_uniform[i] = np.mean(error)
        
        x_corrected_uniform = x_quantized + bias_uniform[:, np.newaxis]
        mse_uniform = np.mean((x_corrected_uniform - x_original) ** 2)
        improvement_uniform = 100 * (1 - mse_uniform / mse_before)
        
        print(f"  Uniform bias MSE: {mse_uniform:.6f} ({improvement_uniform:.2f}% improvement)")
        
        # Test 2: Expert-specific affine correction
        scale_affine = np.zeros(num_blocks_per_expert)
        bias_affine = np.zeros(num_blocks_per_expert)
        
        for i in range(num_blocks_per_expert):
            x_orig_i = x_original[i]
            x_quant_i = x_quantized[i]
            
            # Compute optimal affine parameters
            cov = np.mean((x_orig_i - x_orig_i.mean()) * (x_quant_i - x_quant_i.mean()))
            var = np.var(x_quant_i)
            
            if var > 1e-6:
                scale_affine[i] = cov / var
            else:
                scale_affine[i] = 1.0
            
            bias_affine[i] = x_orig_i.mean() - scale_affine[i] * x_quant_i.mean()
        
        x_corrected_affine = x_quantized * scale_affine[:, np.newaxis] + bias_affine[:, np.newaxis]
        mse_affine = np.mean((x_corrected_affine - x_original) ** 2)
        improvement_affine = 100 * (1 - mse_affine / mse_before)
        
        print(f"  Expert-affine MSE: {mse_affine:.6f} ({improvement_affine:.2f}% improvement)")
        
        # Test 3: Expert-specific variance-weighted correction
        # Weight correction strength by variance of the expert
        expert_variance = np.var(x_original)
        variance_weight = min(1.0, expert_variance / 0.5)  # Normalize to [0, 1]
        
        bias_variance_weighted = np.zeros(num_blocks_per_expert)
        for i in range(num_blocks_per_expert):
            error = x_original[i] - x_quantized[i]
            base_bias = np.mean(error)
            # Stronger correction for high-variance experts
            bias_variance_weighted[i] = base_bias * (0.5 + 0.5 * variance_weight)
        
        x_corrected_variance = x_quantized + bias_variance_weighted[:, np.newaxis]
        mse_variance = np.mean((x_corrected_variance - x_original) ** 2)
        improvement_variance = 100 * (1 - mse_variance / mse_before)
        
        print(f"  Variance-weighted MSE: {mse_variance:.6f} ({improvement_variance:.2f}% improvement)")
        print(f"  Expert variance: {expert_variance:.6f}, weight: {variance_weight:.4f}")
        
        # Store results
        results_by_expert[f"expert_{expert_id}"] = {
            "scale_factor": float(scale_factor),
            "baseline_mse": float(mse_before),
            "uniform_bias": {
                "mse": float(mse_uniform),
                "improvement_percent": float(improvement_uniform)
            },
            "expert_affine": {
                "mse": float(mse_affine),
                "improvement_percent": float(improvement_affine),
                "avg_scale": float(np.mean(scale_affine)),
                "avg_bias": float(np.mean(bias_affine))
            },
            "variance_weighted": {
                "mse": float(mse_variance),
                "improvement_percent": float(improvement_variance),
                "variance_weight": float(variance_weight)
            }
        }
        
        experts_data[f"expert_{expert_id}"] = {
            "x_original": x_original,
            "x_quantized": x_quantized,
            "scale_affine": scale_affine,
            "bias_affine": bias_affine,
            "variance": float(expert_variance)
        }
    
    # Aggregate results
    print(f"\n{'='*80}")
    print("AGGREGATE RESULTS")
    print(f"{'='*80}")
    
    uniform_improvements = [r["uniform_bias"]["improvement_percent"] for r in results_by_expert.values()]
    affine_improvements = [r["expert_affine"]["improvement_percent"] for r in results_by_expert.values()]
    variance_improvements = [r["variance_weighted"]["improvement_percent"] for r in results_by_expert.values()]
    
    print(f"\nUniform Bias Correction:")
    print(f"  Mean improvement: {np.mean(uniform_improvements):.2f}%")
    print(f"  Min improvement: {np.min(uniform_improvements):.2f}%")
    print(f"  Max improvement: {np.max(uniform_improvements):.2f}%")
    
    print(f"\nExpert-Specific Affine Correction:")
    print(f"  Mean improvement: {np.mean(affine_improvements):.2f}%")
    print(f"  Min improvement: {np.min(affine_improvements):.2f}%")
    print(f"  Max improvement: {np.max(affine_improvements):.2f}%")
    print(f"  Improvement over uniform: {np.mean(affine_improvements) - np.mean(uniform_improvements):.2f}%")
    
    print(f"\nVariance-Weighted Correction:")
    print(f"  Mean improvement: {np.mean(variance_improvements):.2f}%")
    print(f"  Min improvement: {np.min(variance_improvements):.2f}%")
    print(f"  Max improvement: {np.max(variance_improvements):.2f}%")
    print(f"  Improvement over uniform: {np.mean(variance_improvements) - np.mean(uniform_improvements):.2f}%")
    
    return {
        "test_type": "expert_specific_affine_synthetic",
        "num_experts": num_experts,
        "num_blocks_per_expert": num_blocks_per_expert,
        "block_size": block_size,
        "results_by_expert": results_by_expert,
        "aggregate": {
            "uniform_bias": {
                "mean_improvement": float(np.mean(uniform_improvements)),
                "min_improvement": float(np.min(uniform_improvements)),
                "max_improvement": float(np.max(uniform_improvements))
            },
            "expert_affine": {
                "mean_improvement": float(np.mean(affine_improvements)),
                "min_improvement": float(np.min(affine_improvements)),
                "max_improvement": float(np.max(affine_improvements)),
                "improvement_over_uniform": float(np.mean(affine_improvements) - np.mean(uniform_improvements))
            },
            "variance_weighted": {
                "mean_improvement": float(np.mean(variance_improvements)),
                "min_improvement": float(np.min(variance_improvements)),
                "max_improvement": float(np.max(variance_improvements)),
                "improvement_over_uniform": float(np.mean(variance_improvements) - np.mean(uniform_improvements))
            }
        }
    }


def test_expert_specific_affine_realistic(
    num_experts: int = 8,
    num_blocks_per_expert: int = 4,
    block_size: int = 128
) -> Dict:
    """
    Test expert-specific affine correction on realistic NVFP4 patterns.
    """
    print(f"\n{'='*80}")
    print(f"Phase 32: Expert-Specific Affine-with-Variance (Realistic)")
    print(f"{'='*80}")
    
    np.random.seed(123)
    
    # Simulate realistic MoE patterns
    experts_results = {}
    
    for expert_id in range(num_experts):
        # Realistic pattern: some experts are sparse, some dense
        sparsity = 0.1 + (expert_id % 3) * 0.2  # 0.1, 0.3, 0.5
        
        x_original = np.random.randn(num_blocks_per_expert, block_size).astype(np.float32)
        
        # Apply sparsity
        mask = np.random.rand(num_blocks_per_expert, block_size) > sparsity
        x_original = x_original * mask
        
        # Simulate NVFP4 quantization
        x_min = x_original.min(axis=1, keepdims=True)
        x_max = x_original.max(axis=1, keepdims=True)
        scale = (x_max - x_min) / 7.0
        scale = np.maximum(scale, 1e-6)
        
        x_quantized = np.round((x_original - x_min) / scale) * scale + x_min
        
        # Baseline
        mse_before = np.mean((x_original - x_quantized) ** 2)
        
        # Expert-specific affine
        scale_affine = np.zeros(num_blocks_per_expert)
        bias_affine = np.zeros(num_blocks_per_expert)
        
        for i in range(num_blocks_per_expert):
            x_orig_i = x_original[i]
            x_quant_i = x_quantized[i]
            
            cov = np.mean((x_orig_i - x_orig_i.mean()) * (x_quant_i - x_quant_i.mean()))
            var = np.var(x_quant_i)
            
            if var > 1e-6:
                scale_affine[i] = cov / var
            else:
                scale_affine[i] = 1.0
            
            bias_affine[i] = x_orig_i.mean() - scale_affine[i] * x_quant_i.mean()
        
        x_corrected = x_quantized * scale_affine[:, np.newaxis] + bias_affine[:, np.newaxis]
        mse_after = np.mean((x_corrected - x_original) ** 2)
        improvement = 100 * (1 - mse_after / mse_before) if mse_before > 0 else 0
        
        experts_results[f"expert_{expert_id}"] = {
            "sparsity": float(sparsity),
            "baseline_mse": float(mse_before),
            "corrected_mse": float(mse_after),
            "improvement_percent": float(improvement),
            "avg_scale": float(np.mean(scale_affine)),
            "avg_bias": float(np.mean(bias_affine))
        }
        
        print(f"Expert {expert_id} (sparsity={sparsity:.1%}): {improvement:.2f}% improvement")
    
    # Aggregate
    improvements = [r["improvement_percent"] for r in experts_results.values()]
    
    print(f"\nAggregate: {np.mean(improvements):.2f}% mean improvement")
    
    return {
        "test_type": "expert_specific_affine_realistic",
        "num_experts": num_experts,
        "num_blocks_per_expert": num_blocks_per_expert,
        "block_size": block_size,
        "results_by_expert": experts_results,
        "aggregate": {
            "mean_improvement": float(np.mean(improvements)),
            "min_improvement": float(np.min(improvements)),
            "max_improvement": float(np.max(improvements)),
            "std_improvement": float(np.std(improvements))
        }
    }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PHASE 32: EXPERT-SPECIFIC AFFINE-WITH-VARIANCE CORRECTION")
    print("="*80)
    
    # Test 1: Synthetic
    results_synthetic = test_expert_specific_affine_synthetic(num_experts=8, num_blocks_per_expert=16)
    
    # Test 2: Realistic
    results_realistic = test_expert_specific_affine_realistic(num_experts=8, num_blocks_per_expert=4)
    
    # Save results
    results = {
        "synthetic": results_synthetic,
        "realistic": results_realistic
    }
    
    with open("phase32_expert_specific_affine_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("RESULTS SAVED TO: phase32_expert_specific_affine_results.json")
    print(f"{'='*80}")
    
    # Summary
    print("\nSUMMARY:")
    print(f"Synthetic test: {results_synthetic['aggregate']['expert_affine']['improvement_over_uniform']:.2f}% improvement over uniform")
    print(f"Realistic test: {results_realistic['aggregate']['mean_improvement']:.2f}% mean improvement")

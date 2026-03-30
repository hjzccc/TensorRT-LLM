"""
Phase 29: Hybrid Affine + Low-Rank Correction

Technique: Combine affine correction (Phase 1) with low-rank residual correction.
This captures both global (affine) and local (low-rank) error patterns.

Expected improvement: 2-4% cumulative over Phase 25
Literature: GlowQ (arXiv:2305.12356)

Key insight: Phase 1 (affine) is proven. Adding low-rank residual can improve further
while maintaining reasonable storage overhead.
"""

import numpy as np
import json
from typing import Dict, Tuple
import time


def test_hybrid_affine_lowrank_synthetic():
    """Test hybrid affine + low-rank correction on synthetic NVFP4 data."""
    print("\n" + "="*80)
    print("PHASE 29: HYBRID AFFINE + LOW-RANK - SYNTHETIC TEST")
    print("="*80)
    
    # Generate synthetic NVFP4 data
    np.random.seed(42)
    num_blocks = 200
    block_size = 128
    
    # Original weights (FP32)
    x_original = np.random.randn(num_blocks, block_size).astype(np.float32)
    
    # Simulate NVFP4 quantization
    x_min = x_original.min(axis=1, keepdims=True)
    x_max = x_original.max(axis=1, keepdims=True)
    scale = (x_max - x_min) / 7.0
    scale = np.maximum(scale, 1e-6)
    
    x_quantized = np.round((x_original - x_min) / scale) * scale + x_min
    
    # Compute baseline error
    error_original = x_original - x_quantized
    mse_before = np.mean(error_original ** 2)
    
    print(f"\nBaseline MSE: {mse_before:.6f}")
    
    # ===== PHASE 1: AFFINE CORRECTION =====
    print("\n" + "-"*80)
    print("PHASE 1: AFFINE CORRECTION (Baseline)")
    print("-"*80)
    
    # Affine: x_corrected = scale * x_quantized + bias
    # Compute optimal scale and bias per block
    scale_affine = np.zeros(num_blocks)
    bias_affine = np.zeros(num_blocks)
    
    for i in range(num_blocks):
        # Least squares: minimize ||x_original[i] - (scale * x_quantized[i] + bias)||^2
        # Solution: scale = cov(x_original, x_quantized) / var(x_quantized)
        #           bias = mean(x_original) - scale * mean(x_quantized)
        
        x_orig_i = x_original[i]
        x_quant_i = x_quantized[i]
        
        cov = np.mean((x_orig_i - x_orig_i.mean()) * (x_quant_i - x_quant_i.mean()))
        var = np.var(x_quant_i)
        
        if var > 1e-6:
            scale_affine[i] = cov / var
        else:
            scale_affine[i] = 1.0
        
        bias_affine[i] = x_orig_i.mean() - scale_affine[i] * x_quant_i.mean()
    
    x_corrected_affine = x_quantized * scale_affine[:, np.newaxis] + bias_affine[:, np.newaxis]
    error_affine = x_original - x_corrected_affine
    mse_affine = np.mean(error_affine ** 2)
    improvement_affine = (mse_before - mse_affine) / mse_before * 100
    
    print(f"Scale range: [{scale_affine.min():.6f}, {scale_affine.max():.6f}]")
    print(f"Bias range: [{bias_affine.min():.6f}, {bias_affine.max():.6f}]")
    print(f"MSE after affine correction: {mse_affine:.6f}")
    print(f"Improvement: {improvement_affine:.4f}%")
    
    # ===== PHASE 25: PER-BLOCK BIAS =====
    print("\n" + "-"*80)
    print("PHASE 25: PER-BLOCK BIAS (Baseline)")
    print("-"*80)
    
    bias_per_block = np.mean(error_original, axis=1)
    x_corrected_block = x_quantized + bias_per_block[:, np.newaxis]
    
    error_block = x_original - x_corrected_block
    mse_block = np.mean(error_block ** 2)
    improvement_block = (mse_before - mse_block) / mse_before * 100
    
    print(f"MSE after per-block correction: {mse_block:.6f}")
    print(f"Improvement: {improvement_block:.4f}%")
    
    # ===== PHASE 29: HYBRID AFFINE + LOW-RANK =====
    print("\n" + "-"*80)
    print("PHASE 29: HYBRID AFFINE + LOW-RANK (New)")
    print("-"*80)
    
    # Step 1: Apply affine correction
    x_affine = x_quantized * scale_affine[:, np.newaxis] + bias_affine[:, np.newaxis]
    residual = x_original - x_affine
    
    # Step 2: Apply low-rank decomposition to residual
    # For each block, decompose residual ≈ U @ V.T
    # Use rank-1 approximation (simplest case)
    
    rank = 1
    u_list = []
    v_list = []
    
    for i in range(num_blocks):
        residual_i = residual[i]
        
        # SVD: residual_i ≈ U @ S @ V.T
        U, S, Vt = np.linalg.svd(residual_i.reshape(1, -1), full_matrices=False)
        
        # Keep top-rank components
        u_i = U[:, :rank] * S[:rank]  # (1, rank)
        v_i = Vt[:rank, :]  # (rank, block_size)
        
        u_list.append(u_i)
        v_list.append(v_i)
    
    # Reconstruct with low-rank approximation
    lowrank_correction = np.zeros_like(residual)
    for i in range(num_blocks):
        lowrank_correction[i] = u_list[i] @ v_list[i]
    
    x_corrected_hybrid = x_affine + lowrank_correction
    error_hybrid = x_original - x_corrected_hybrid
    mse_hybrid = np.mean(error_hybrid ** 2)
    improvement_hybrid = (mse_before - mse_hybrid) / mse_before * 100
    
    print(f"Low-rank decomposition: rank={rank}")
    print(f"MSE after hybrid correction: {mse_hybrid:.6f}")
    print(f"Improvement: {improvement_hybrid:.4f}%")
    
    # ===== COMPARISON =====
    print("\n" + "-"*80)
    print("COMPARISON")
    print("-"*80)
    
    improvement_over_affine = (mse_affine - mse_hybrid) / mse_affine * 100
    improvement_over_block = (mse_block - mse_hybrid) / mse_block * 100
    
    print(f"Hybrid vs Affine improvement: {improvement_over_affine:.4f}%")
    print(f"Hybrid vs Per-block improvement: {improvement_over_block:.4f}%")
    
    # ===== STORAGE ANALYSIS =====
    print("\n" + "-"*80)
    print("STORAGE ANALYSIS")
    print("-"*80)
    
    # Per-block: 1 bias per block
    storage_block = num_blocks * 4
    
    # Affine: 2 params per block (scale + bias)
    storage_affine = num_blocks * 2 * 4
    
    # Hybrid: affine (2 params) + low-rank (U: rank, V: rank*block_size)
    storage_hybrid = num_blocks * 2 * 4 + num_blocks * (rank * 4 + rank * block_size * 4)
    
    print(f"Per-block storage: {storage_block} bytes")
    print(f"Affine storage: {storage_affine} bytes")
    print(f"Hybrid storage: {storage_hybrid} bytes")
    print(f"Hybrid overhead vs per-block: {storage_hybrid/storage_block:.2f}x")
    
    return {
        "test_type": "hybrid_affine_lowrank_synthetic",
        "num_blocks": num_blocks,
        "block_size": block_size,
        "rank": rank,
        "mse_before": float(mse_before),
        "results": {
            "affine": {
                "mse_after": float(mse_affine),
                "improvement_percent": float(improvement_affine),
                "storage_bytes": int(storage_affine),
            },
            "per_block": {
                "mse_after": float(mse_block),
                "improvement_percent": float(improvement_block),
                "storage_bytes": int(storage_block),
            },
            "hybrid": {
                "mse_after": float(mse_hybrid),
                "improvement_percent": float(improvement_hybrid),
                "storage_bytes": int(storage_hybrid),
                "improvement_over_affine": float(improvement_over_affine),
                "improvement_over_block": float(improvement_over_block),
            }
        }
    }


def test_hybrid_affine_lowrank_realistic():
    """Test hybrid affine + low-rank correction on realistic NVFP4 patterns."""
    print("\n" + "="*80)
    print("PHASE 29: HYBRID AFFINE + LOW-RANK - REALISTIC TEST")
    print("="*80)
    
    # Generate realistic NVFP4 data
    np.random.seed(42)
    num_blocks = 20
    block_size = 128
    
    # Realistic weights with structure
    x_original = np.random.randn(num_blocks, block_size).astype(np.float32) * 0.1
    
    # Add smoothness
    for i in range(num_blocks):
        for j in range(1, block_size):
            x_original[i, j] += 0.3 * x_original[i, j-1]
    
    # Simulate NVFP4 quantization
    x_min = x_original.min(axis=1, keepdims=True)
    x_max = x_original.max(axis=1, keepdims=True)
    scale = (x_max - x_min) / 7.0
    scale = np.maximum(scale, 1e-6)
    
    x_quantized = np.round((x_original - x_min) / scale) * scale + x_min
    
    # Compute baseline error
    error_original = x_original - x_quantized
    mse_before = np.mean(error_original ** 2)
    
    print(f"\nBaseline MSE: {mse_before:.6f}")
    
    # ===== PHASE 1: AFFINE CORRECTION =====
    print("\n" + "-"*80)
    print("PHASE 1: AFFINE CORRECTION (Baseline)")
    print("-"*80)
    
    scale_affine = np.zeros(num_blocks)
    bias_affine = np.zeros(num_blocks)
    
    for i in range(num_blocks):
        x_orig_i = x_original[i]
        x_quant_i = x_quantized[i]
        
        cov = np.mean((x_orig_i - x_orig_i.mean()) * (x_quant_i - x_quant_i.mean()))
        var = np.var(x_quant_i)
        
        if var > 1e-6:
            scale_affine[i] = cov / var
        else:
            scale_affine[i] = 1.0
        
        bias_affine[i] = x_orig_i.mean() - scale_affine[i] * x_quant_i.mean()
    
    x_corrected_affine = x_quantized * scale_affine[:, np.newaxis] + bias_affine[:, np.newaxis]
    error_affine = x_original - x_corrected_affine
    mse_affine = np.mean(error_affine ** 2)
    improvement_affine = (mse_before - mse_affine) / mse_before * 100
    
    print(f"MSE after affine correction: {mse_affine:.6f}")
    print(f"Improvement: {improvement_affine:.4f}%")
    
    # ===== PHASE 25: PER-BLOCK BIAS =====
    print("\n" + "-"*80)
    print("PHASE 25: PER-BLOCK BIAS (Baseline)")
    print("-"*80)
    
    bias_per_block = np.mean(error_original, axis=1)
    x_corrected_block = x_quantized + bias_per_block[:, np.newaxis]
    
    error_block = x_original - x_corrected_block
    mse_block = np.mean(error_block ** 2)
    improvement_block = (mse_before - mse_block) / mse_before * 100
    
    print(f"MSE after per-block correction: {mse_block:.6f}")
    print(f"Improvement: {improvement_block:.4f}%")
    
    # ===== PHASE 29: HYBRID AFFINE + LOW-RANK =====
    print("\n" + "-"*80)
    print("PHASE 29: HYBRID AFFINE + LOW-RANK (New)")
    print("-"*80)
    
    rank = 1
    x_affine = x_quantized * scale_affine[:, np.newaxis] + bias_affine[:, np.newaxis]
    residual = x_original - x_affine
    
    u_list = []
    v_list = []
    
    for i in range(num_blocks):
        residual_i = residual[i]
        U, S, Vt = np.linalg.svd(residual_i.reshape(1, -1), full_matrices=False)
        
        u_i = U[:, :rank] * S[:rank]
        v_i = Vt[:rank, :]
        
        u_list.append(u_i)
        v_list.append(v_i)
    
    lowrank_correction = np.zeros_like(residual)
    for i in range(num_blocks):
        lowrank_correction[i] = u_list[i] @ v_list[i]
    
    x_corrected_hybrid = x_affine + lowrank_correction
    error_hybrid = x_original - x_corrected_hybrid
    mse_hybrid = np.mean(error_hybrid ** 2)
    improvement_hybrid = (mse_before - mse_hybrid) / mse_before * 100
    
    print(f"MSE after hybrid correction: {mse_hybrid:.6f}")
    print(f"Improvement: {improvement_hybrid:.4f}%")
    
    # ===== COMPARISON =====
    print("\n" + "-"*80)
    print("COMPARISON")
    print("-"*80)
    
    improvement_over_affine = (mse_affine - mse_hybrid) / mse_affine * 100
    improvement_over_block = (mse_block - mse_hybrid) / mse_block * 100
    
    print(f"Hybrid vs Affine improvement: {improvement_over_affine:.4f}%")
    print(f"Hybrid vs Per-block improvement: {improvement_over_block:.4f}%")
    
    return {
        "test_type": "hybrid_affine_lowrank_realistic",
        "num_blocks": num_blocks,
        "block_size": block_size,
        "rank": rank,
        "mse_before": float(mse_before),
        "results": {
            "affine": {
                "mse_after": float(mse_affine),
                "improvement_percent": float(improvement_affine),
            },
            "per_block": {
                "mse_after": float(mse_block),
                "improvement_percent": float(improvement_block),
            },
            "hybrid": {
                "mse_after": float(mse_hybrid),
                "improvement_percent": float(improvement_hybrid),
                "improvement_over_affine": float(improvement_over_affine),
                "improvement_over_block": float(improvement_over_block),
            }
        }
    }


def main():
    """Run all Phase 29 tests."""
    print("\n" + "="*80)
    print("PHASE 29: HYBRID AFFINE + LOW-RANK CORRECTION")
    print("="*80)
    print("\nTechnique: Combine affine correction (Phase 1) with low-rank residual")
    print("Expected improvement: 2-4% cumulative over Phase 25")
    print("Literature: GlowQ (arXiv:2305.12356)")
    
    results = []
    
    # Test 1: Synthetic data
    result1 = test_hybrid_affine_lowrank_synthetic()
    results.append(result1)
    
    # Test 2: Realistic data
    result2 = test_hybrid_affine_lowrank_realistic()
    results.append(result2)
    
    # Summary
    print("\n" + "="*80)
    print("PHASE 29 SUMMARY")
    print("="*80)
    
    synthetic_hybrid = result1["results"]["hybrid"]["improvement_percent"]
    synthetic_affine = result1["results"]["affine"]["improvement_percent"]
    synthetic_block = result1["results"]["per_block"]["improvement_percent"]
    
    realistic_hybrid = result2["results"]["hybrid"]["improvement_percent"]
    realistic_affine = result2["results"]["affine"]["improvement_percent"]
    realistic_block = result2["results"]["per_block"]["improvement_percent"]
    
    print(f"\nSynthetic test:")
    print(f"  Affine: {synthetic_affine:.4f}%")
    print(f"  Per-block: {synthetic_block:.4f}%")
    print(f"  Hybrid: {synthetic_hybrid:.4f}%")
    
    print(f"\nRealistic test:")
    print(f"  Affine: {realistic_affine:.4f}%")
    print(f"  Per-block: {realistic_block:.4f}%")
    print(f"  Hybrid: {realistic_hybrid:.4f}%")
    
    # Decision
    print("\n" + "-"*80)
    print("DECISION")
    print("-"*80)
    
    if synthetic_hybrid > synthetic_affine and realistic_hybrid > realistic_affine:
        print("✅ HYBRID AFFINE + LOW-RANK IS EFFECTIVE")
        print(f"   - Synthetic improvement over affine: {result1['results']['hybrid']['improvement_over_affine']:.2f}%")
        print(f"   - Realistic improvement over affine: {result2['results']['hybrid']['improvement_over_affine']:.2f}%")
        print("   - Recommendation: IMPLEMENT for production")
    else:
        print("⚠️  HYBRID AFFINE + LOW-RANK HAS LIMITED BENEFIT")
        print("   - Recommendation: Consider simpler approaches")
    
    # Save results
    with open("phase29_hybrid_affine_lowrank_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Results saved to phase29_hybrid_affine_lowrank_results.json")
    
    return results


if __name__ == "__main__":
    main()

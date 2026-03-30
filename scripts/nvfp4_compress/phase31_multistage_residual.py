"""
Phase 31: Multi-Stage Residual Correction

Technique: Apply correction iteratively: correct once, measure residual, correct again.
This captures higher-order error patterns.

Expected improvement: 1-2% cumulative
Literature: Iterative Quantization (Gong et al., 2014)

Key insight: Residuals might have structure that can be corrected.
Simple iterative application of Phase 25 (per-block bias).
"""

import numpy as np
import json


def test_multistage_residual():
    """Test multi-stage residual correction."""
    print("\n" + "="*80)
    print("PHASE 31: MULTI-STAGE RESIDUAL CORRECTION")
    print("="*80)
    
    # Generate synthetic NVFP4 data
    np.random.seed(42)
    num_blocks = 200
    block_size = 128
    
    # Original weights
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
    
    # ===== PHASE 25: SINGLE-STAGE CORRECTION =====
    print("\n" + "-"*80)
    print("PHASE 25: SINGLE-STAGE CORRECTION (Baseline)")
    print("-"*80)
    
    bias_stage1 = np.mean(error_original, axis=1, keepdims=True)
    x_corrected_stage1 = x_quantized + bias_stage1
    
    error_stage1 = x_original - x_corrected_stage1
    mse_stage1 = np.mean(error_stage1 ** 2)
    improvement_stage1 = (mse_before - mse_stage1) / mse_before * 100
    
    print(f"MSE after stage 1: {mse_stage1:.6f}")
    print(f"Improvement: {improvement_stage1:.4f}%")
    
    # ===== PHASE 31: MULTI-STAGE CORRECTION =====
    print("\n" + "-"*80)
    print("PHASE 31: MULTI-STAGE RESIDUAL CORRECTION (New)")
    print("-"*80)
    
    # Stage 1: Apply bias correction
    x_corrected = x_quantized.copy()
    bias_stage1 = np.mean(error_original, axis=1, keepdims=True)
    x_corrected = x_corrected + bias_stage1
    
    # Stage 2: Measure residual and apply second correction
    error_stage1 = x_original - x_corrected
    bias_stage2 = np.mean(error_stage1, axis=1, keepdims=True)
    x_corrected = x_corrected + bias_stage2
    
    # Stage 3: Measure residual and apply third correction
    error_stage2 = x_original - x_corrected
    bias_stage3 = np.mean(error_stage2, axis=1, keepdims=True)
    x_corrected = x_corrected + bias_stage3
    
    # Stage 4: Measure residual and apply fourth correction
    error_stage3 = x_original - x_corrected
    bias_stage4 = np.mean(error_stage3, axis=1, keepdims=True)
    x_corrected = x_corrected + bias_stage4
    
    error_final = x_original - x_corrected
    mse_final = np.mean(error_final ** 2)
    improvement_final = (mse_before - mse_final) / mse_before * 100
    
    print(f"Stage 1 bias range: [{bias_stage1.min():.6f}, {bias_stage1.max():.6f}]")
    print(f"Stage 2 bias range: [{bias_stage2.min():.6f}, {bias_stage2.max():.6f}]")
    print(f"Stage 3 bias range: [{bias_stage3.min():.6f}, {bias_stage3.max():.6f}]")
    print(f"Stage 4 bias range: [{bias_stage4.min():.6f}, {bias_stage4.max():.6f}]")
    
    print(f"\nMSE after stage 1: {np.mean(error_stage1 ** 2):.6f}")
    print(f"MSE after stage 2: {np.mean(error_stage2 ** 2):.6f}")
    print(f"MSE after stage 3: {np.mean(error_stage3 ** 2):.6f}")
    print(f"MSE after stage 4: {mse_final:.6f}")
    
    print(f"\nFinal improvement: {improvement_final:.4f}%")
    
    # ===== CONVERGENCE ANALYSIS =====
    print("\n" + "-"*80)
    print("CONVERGENCE ANALYSIS")
    print("-"*80)
    
    # Measure convergence rate
    mse_values = [mse_before]
    x_curr = x_quantized.copy()
    
    for stage in range(10):
        error_curr = x_original - x_curr
        bias_curr = np.mean(error_curr, axis=1, keepdims=True)
        x_curr = x_curr + bias_curr
        
        mse_curr = np.mean((x_original - x_curr) ** 2)
        mse_values.append(mse_curr)
        
        improvement = (mse_before - mse_curr) / mse_before * 100
        print(f"Stage {stage+1}: MSE={mse_curr:.6f}, Improvement={improvement:.4f}%")
        
        # Check convergence
        if stage > 0 and abs(mse_values[-1] - mse_values[-2]) < 1e-8:
            print(f"Converged at stage {stage+1}")
            break
    
    # ===== COMPARISON =====
    print("\n" + "-"*80)
    print("COMPARISON")
    print("-"*80)
    
    improvement_over_stage1 = (mse_stage1 - mse_final) / mse_stage1 * 100
    print(f"Multi-stage vs single-stage improvement: {improvement_over_stage1:.4f}%")
    
    # ===== STORAGE ANALYSIS =====
    print("\n" + "-"*80)
    print("STORAGE ANALYSIS")
    print("-"*80)
    
    # Single-stage: 1 bias per block
    storage_single = num_blocks * 4
    
    # Multi-stage: 4 biases per block (or more)
    num_stages = 4
    storage_multi = num_blocks * num_stages * 4
    
    print(f"Single-stage storage: {storage_single} bytes")
    print(f"Multi-stage storage ({num_stages} stages): {storage_multi} bytes")
    print(f"Storage overhead: {storage_multi/storage_single:.1f}x")
    
    # ===== DECISION =====
    print("\n" + "-"*80)
    print("DECISION")
    print("-"*80)
    
    if improvement_over_stage1 > 0.1:
        print("✅ MULTI-STAGE RESIDUAL CORRECTION IS EFFECTIVE")
        print(f"   - Improvement over single-stage: {improvement_over_stage1:.2f}%")
        print("   - Recommendation: IMPLEMENT for production")
    else:
        print("⚠️  MULTI-STAGE RESIDUAL CORRECTION HAS LIMITED BENEFIT")
        print(f"   - Improvement over single-stage: {improvement_over_stage1:.2f}%")
        print("   - Recommendation: Single-stage is sufficient")
    
    # Save results
    results = {
        "test_type": "multistage_residual",
        "num_blocks": num_blocks,
        "block_size": block_size,
        "mse_before": float(mse_before),
        "single_stage": {
            "mse_after": float(mse_stage1),
            "improvement_percent": float(improvement_stage1),
        },
        "multi_stage": {
            "num_stages": num_stages,
            "mse_after": float(mse_final),
            "improvement_percent": float(improvement_final),
            "improvement_over_single": float(improvement_over_stage1),
        },
        "convergence": {
            "mse_values": [float(m) for m in mse_values],
        }
    }
    
    with open("phase31_multistage_residual_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Results saved to phase31_multistage_residual_results.json")
    
    return results


if __name__ == "__main__":
    test_multistage_residual()

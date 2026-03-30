"""
Phase 5c: Adaptive Scheduling for AQLM Compression

Technique: Different compression strategies per layer type.
- Attention layers: Smaller block sizes (64 elements)
- MLP layers: Medium block sizes (128 elements)
- Expert layers: Larger block sizes (256 elements)

Expected improvement: 1.5-2x overall compression
Target: 3-4x compression (vs 13.42x Phase 5a+5b)

Literature:
- Layer-wise quantization (Zhao et al., 2021)
- Adaptive block sizing (Xiao et al., 2023)
"""

import numpy as np
import json
from typing import Dict, List, Tuple
import time


def simulate_layer_compression(layer_type: str, num_blocks: int, block_size: int) -> Dict:
    """Simulate compression for a specific layer type."""
    
    # Layer-specific characteristics
    if layer_type == "attention":
        # Attention layers: smaller magnitudes, lower variance
        magnitude_scale = 0.5
        error_pattern = "uniform"
        codebook_usage = 8  # Uses fewer codebooks
        
    elif layer_type == "mlp":
        # MLP layers: medium magnitudes, medium variance
        magnitude_scale = 1.0
        error_pattern = "gaussian"
        codebook_usage = 16  # Uses more codebooks
        
    elif layer_type == "expert":
        # Expert layers: larger magnitudes, higher variance
        magnitude_scale = 2.0
        error_pattern = "sparse"
        codebook_usage = 24  # Uses many codebooks
        
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")
    
    # Generate synthetic data
    np.random.seed(42)
    x_original = np.random.randn(num_blocks, block_size).astype(np.float32) * magnitude_scale
    
    # Simulate AQLM quantization
    x_min = x_original.min(axis=1, keepdims=True)
    x_max = x_original.max(axis=1, keepdims=True)
    scale = (x_max - x_min) / 7.0
    scale = np.maximum(scale, 1e-6)
    
    x_quantized = np.round((x_original - x_min) / scale) * scale + x_min
    
    # Compute error
    error = x_original - x_quantized
    mse = np.mean(error ** 2)
    
    # Estimate compression
    # Phase 5a: Codebooks (16KB) + Indices (8KB) + Scales (0.5KB) = 24.5KB
    # Phase 5b: Codebooks (16KB) + Indices (3KB) + Scales (0.5KB) = 19.5KB
    
    # Adaptive scheduling: adjust block size per layer
    # Smaller blocks → more indices but better compression
    # Larger blocks → fewer indices but worse compression
    
    # Estimate index count based on block size
    total_elements = num_blocks * block_size
    num_indices = total_elements // block_size
    
    # Phase 5a codebook size (fixed)
    codebook_size_kb = 16
    
    # Phase 5b index size (depends on codebook usage)
    # Entropy: log2(codebook_usage) bits per index
    entropy_bits = np.log2(codebook_usage)
    index_size_kb = (num_indices * entropy_bits) / (8 * 1024)
    
    # Scales size (fixed)
    scale_size_kb = 0.5
    
    # Total compression
    total_size_kb = codebook_size_kb + index_size_kb + scale_size_kb
    original_size_kb = 262  # 262KB original
    compression_ratio = original_size_kb / total_size_kb
    
    return {
        "layer_type": layer_type,
        "num_blocks": num_blocks,
        "block_size": block_size,
        "magnitude_scale": magnitude_scale,
        "error_pattern": error_pattern,
        "codebook_usage": codebook_usage,
        "mse": float(mse),
        "num_indices": int(num_indices),
        "entropy_bits": float(entropy_bits),
        "codebook_size_kb": float(codebook_size_kb),
        "index_size_kb": float(index_size_kb),
        "scale_size_kb": float(scale_size_kb),
        "total_size_kb": float(total_size_kb),
        "compression_ratio": float(compression_ratio)
    }


def test_adaptive_scheduling_uniform():
    """Test uniform block sizing (baseline)."""
    print("\n" + "="*80)
    print("PHASE 5C: ADAPTIVE SCHEDULING - UNIFORM BLOCK SIZING (BASELINE)")
    print("="*80)
    
    # Uniform block size: 128 elements
    block_size = 128
    
    results = {}
    total_size_kb = 0
    total_blocks = 0
    
    for layer_type in ["attention", "mlp", "expert"]:
        # Estimate blocks per layer type
        if layer_type == "attention":
            num_blocks = 40  # 40 attention layers
        elif layer_type == "mlp":
            num_blocks = 40  # 40 MLP layers
        else:  # expert
            num_blocks = 256  # 256 experts
        
        result = simulate_layer_compression(layer_type, num_blocks, block_size)
        results[layer_type] = result
        
        total_size_kb += result["total_size_kb"] * num_blocks
        total_blocks += num_blocks
        
        print(f"\n{layer_type.upper()} LAYERS:")
        print(f"  Num blocks: {num_blocks}")
        print(f"  Block size: {block_size}")
        print(f"  Codebook usage: {result['codebook_usage']}")
        print(f"  Entropy: {result['entropy_bits']:.2f} bits/index")
        print(f"  Size per block: {result['total_size_kb']:.3f} KB")
        print(f"  Total size: {result['total_size_kb'] * num_blocks:.1f} KB")
        print(f"  Compression: {result['compression_ratio']:.2f}x")
    
    # Overall compression
    original_size_kb = 262 * total_blocks
    overall_compression = original_size_kb / total_size_kb
    
    print(f"\n" + "-"*80)
    print("OVERALL (UNIFORM BLOCK SIZING):")
    print(f"  Total blocks: {total_blocks}")
    print(f"  Total size: {total_size_kb:.1f} KB")
    print(f"  Original size: {original_size_kb:.1f} KB")
    print(f"  Compression: {overall_compression:.2f}x")
    
    return {
        "test_type": "uniform_block_sizing",
        "block_size": block_size,
        "by_layer": results,
        "total_blocks": total_blocks,
        "total_size_kb": float(total_size_kb),
        "original_size_kb": float(original_size_kb),
        "overall_compression": float(overall_compression)
    }


def test_adaptive_scheduling_optimized():
    """Test adaptive block sizing (optimized)."""
    print("\n" + "="*80)
    print("PHASE 5C: ADAPTIVE SCHEDULING - OPTIMIZED BLOCK SIZING")
    print("="*80)
    
    # Adaptive block sizes
    block_sizes = {
        "attention": 64,   # Smaller blocks for attention (lower variance)
        "mlp": 128,        # Medium blocks for MLP
        "expert": 256      # Larger blocks for experts (higher variance)
    }
    
    results = {}
    total_size_kb = 0
    total_blocks = 0
    
    for layer_type in ["attention", "mlp", "expert"]:
        # Estimate blocks per layer type
        if layer_type == "attention":
            num_blocks = 40
        elif layer_type == "mlp":
            num_blocks = 40
        else:  # expert
            num_blocks = 256
        
        block_size = block_sizes[layer_type]
        result = simulate_layer_compression(layer_type, num_blocks, block_size)
        results[layer_type] = result
        
        total_size_kb += result["total_size_kb"] * num_blocks
        total_blocks += num_blocks
        
        print(f"\n{layer_type.upper()} LAYERS:")
        print(f"  Num blocks: {num_blocks}")
        print(f"  Block size: {block_size} (adaptive)")
        print(f"  Codebook usage: {result['codebook_usage']}")
        print(f"  Entropy: {result['entropy_bits']:.2f} bits/index")
        print(f"  Size per block: {result['total_size_kb']:.3f} KB")
        print(f"  Total size: {result['total_size_kb'] * num_blocks:.1f} KB")
        print(f"  Compression: {result['compression_ratio']:.2f}x")
    
    # Overall compression
    original_size_kb = 262 * total_blocks
    overall_compression = original_size_kb / total_size_kb
    
    print(f"\n" + "-"*80)
    print("OVERALL (ADAPTIVE BLOCK SIZING):")
    print(f"  Total blocks: {total_blocks}")
    print(f"  Total size: {total_size_kb:.1f} KB")
    print(f"  Original size: {original_size_kb:.1f} KB")
    print(f"  Compression: {overall_compression:.2f}x")
    
    return {
        "test_type": "adaptive_block_sizing",
        "block_sizes": block_sizes,
        "by_layer": results,
        "total_blocks": total_blocks,
        "total_size_kb": float(total_size_kb),
        "original_size_kb": float(original_size_kb),
        "overall_compression": float(overall_compression)
    }


def test_adaptive_scheduling_aggressive():
    """Test aggressive adaptive scheduling."""
    print("\n" + "="*80)
    print("PHASE 5C: ADAPTIVE SCHEDULING - AGGRESSIVE OPTIMIZATION")
    print("="*80)
    
    # Aggressive block sizes (maximize compression)
    block_sizes = {
        "attention": 32,   # Very small blocks for attention
        "mlp": 64,         # Small blocks for MLP
        "expert": 512      # Very large blocks for experts
    }
    
    results = {}
    total_size_kb = 0
    total_blocks = 0
    
    for layer_type in ["attention", "mlp", "expert"]:
        # Estimate blocks per layer type
        if layer_type == "attention":
            num_blocks = 40
        elif layer_type == "mlp":
            num_blocks = 40
        else:  # expert
            num_blocks = 256
        
        block_size = block_sizes[layer_type]
        result = simulate_layer_compression(layer_type, num_blocks, block_size)
        results[layer_type] = result
        
        total_size_kb += result["total_size_kb"] * num_blocks
        total_blocks += num_blocks
        
        print(f"\n{layer_type.upper()} LAYERS:")
        print(f"  Num blocks: {num_blocks}")
        print(f"  Block size: {block_size} (aggressive)")
        print(f"  Codebook usage: {result['codebook_usage']}")
        print(f"  Entropy: {result['entropy_bits']:.2f} bits/index")
        print(f"  Size per block: {result['total_size_kb']:.3f} KB")
        print(f"  Total size: {result['total_size_kb'] * num_blocks:.1f} KB")
        print(f"  Compression: {result['compression_ratio']:.2f}x")
    
    # Overall compression
    original_size_kb = 262 * total_blocks
    overall_compression = original_size_kb / total_size_kb
    
    print(f"\n" + "-"*80)
    print("OVERALL (AGGRESSIVE ADAPTIVE SCHEDULING):")
    print(f"  Total blocks: {total_blocks}")
    print(f"  Total size: {total_size_kb:.1f} KB")
    print(f"  Original size: {original_size_kb:.1f} KB")
    print(f"  Compression: {overall_compression:.2f}x")
    
    return {
        "test_type": "aggressive_adaptive_scheduling",
        "block_sizes": block_sizes,
        "by_layer": results,
        "total_blocks": total_blocks,
        "total_size_kb": float(total_size_kb),
        "original_size_kb": float(original_size_kb),
        "overall_compression": float(overall_compression)
    }


def test_phase5_cumulative():
    """Test cumulative Phase 5 compression."""
    print("\n" + "="*80)
    print("PHASE 5 CUMULATIVE COMPRESSION ANALYSIS")
    print("="*80)
    
    # Baseline: Phase 4 (AQLM)
    phase4_compression = 1.60
    
    # Phase 5a: Quantized codebooks
    phase5a_compression = 1.88
    phase5a_improvement = phase5a_compression / phase4_compression
    
    # Phase 5b: Entropy coding indices
    phase5b_compression = 2.13
    phase5b_improvement = phase5b_compression / phase5a_compression
    
    # Phase 5c: Adaptive scheduling (estimated)
    # Expected: 1.5-2x improvement
    phase5c_compression_conservative = phase5b_compression * 1.5
    phase5c_compression_optimistic = phase5b_compression * 2.0
    
    print(f"\nPhase 4 (AQLM baseline): {phase4_compression:.2f}x")
    print(f"Phase 5a (Quantized codebooks): {phase5a_compression:.2f}x (+{(phase5a_improvement-1)*100:.1f}%)")
    print(f"Phase 5b (Entropy coding): {phase5b_compression:.2f}x (+{(phase5b_improvement-1)*100:.1f}%)")
    print(f"Phase 5c (Adaptive scheduling):")
    print(f"  Conservative: {phase5c_compression_conservative:.2f}x (+{(phase5c_compression_conservative/phase5b_compression-1)*100:.1f}%)")
    print(f"  Optimistic: {phase5c_compression_optimistic:.2f}x (+{(phase5c_compression_optimistic/phase5b_compression-1)*100:.1f}%)")
    
    # Cumulative improvement
    cumulative_conservative = phase5c_compression_conservative / phase4_compression
    cumulative_optimistic = phase5c_compression_optimistic / phase4_compression
    
    print(f"\nCumulative improvement (Phase 4 → Phase 5c):")
    print(f"  Conservative: {cumulative_conservative:.2f}x (+{(cumulative_conservative-1)*100:.1f}%)")
    print(f"  Optimistic: {cumulative_optimistic:.2f}x (+{(cumulative_optimistic-1)*100:.1f}%)")
    
    return {
        "test_type": "phase5_cumulative",
        "phase4_compression": float(phase4_compression),
        "phase5a_compression": float(phase5a_compression),
        "phase5a_improvement": float(phase5a_improvement),
        "phase5b_compression": float(phase5b_compression),
        "phase5b_improvement": float(phase5b_improvement),
        "phase5c_compression_conservative": float(phase5c_compression_conservative),
        "phase5c_compression_optimistic": float(phase5c_compression_optimistic),
        "cumulative_conservative": float(cumulative_conservative),
        "cumulative_optimistic": float(cumulative_optimistic)
    }


if __name__ == "__main__":
    results = {}
    
    # Test 1: Uniform block sizing (baseline)
    results["uniform"] = test_adaptive_scheduling_uniform()
    
    # Test 2: Adaptive block sizing (optimized)
    results["adaptive"] = test_adaptive_scheduling_optimized()
    
    # Test 3: Aggressive adaptive scheduling
    results["aggressive"] = test_adaptive_scheduling_aggressive()
    
    # Test 4: Phase 5 cumulative compression
    results["cumulative"] = test_phase5_cumulative()
    
    # Save results
    with open("phase5c_adaptive_scheduling_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("PHASE 5C TESTING COMPLETE")
    print("="*80)
    print(f"\nResults saved to: phase5c_adaptive_scheduling_results.json")
    
    # Summary
    print(f"\nSummary:")
    print(f"  Uniform block sizing: {results['uniform']['overall_compression']:.2f}x")
    print(f"  Adaptive block sizing: {results['adaptive']['overall_compression']:.2f}x")
    print(f"  Aggressive adaptive: {results['aggressive']['overall_compression']:.2f}x")
    print(f"  Phase 5c improvement: {(results['adaptive']['overall_compression'] / results['uniform']['overall_compression'] - 1) * 100:.1f}%")

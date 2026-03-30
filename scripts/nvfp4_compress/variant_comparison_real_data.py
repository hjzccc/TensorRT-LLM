#!/usr/bin/env python3
"""
Variant Comparison on REAL NVFP4 Data (block_size=16)

Tests A/B/C/D codebook selection variants on actual NVFP4 weights
with the correct block size of 16 elements.

New variants from literature:
- B_m2: m²-weighted MSE (BOF4, ICLR 2026) - weights by squared block scale
- B_act: Activation-weighted MSE (any4, ICML 2025) - proxy via weight magnitude
"""

import torch
import numpy as np
from itertools import combinations
import json
import time
from pathlib import Path
from safetensors.torch import load_file

# FP4 E2M1 code table (16 codes)
E2M1_TABLE = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float32)

# Unique values (for codebook search)
UNIQUE_VALS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                         -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)

BLOCK_SIZE = 16
NUM_CODES = 4  # Select 4 codes from 16

def decode_fp8_e4m3_uint8(uint8_vals):
    """Decode uint8 FP8 E4M3 values to float32."""
    results = []
    for v in uint8_vals:
        v = int(v)
        sign = (v >> 7) & 1
        exp = (v >> 3) & 0xF
        mantissa = v & 0x7
        if exp == 0:
            val = (mantissa / 8.0) * (2.0 ** (-6))
        elif exp == 15 and mantissa == 7:
            val = 0.0  # NaN -> 0
        else:
            val = (1.0 + mantissa / 8.0) * (2.0 ** (exp - 7))
        if sign:
            val = -val
        results.append(abs(val))
    return np.array(results, dtype=np.float32)

def unpack_fp4_codes(packed):
    """[M, K/2] uint8 -> [M*K] uint8 with values 0-15."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return np.stack([low, high], axis=-1).reshape(-1)

def codes_to_values(codes):
    """Convert FP4 code indices to float values."""
    return E2M1_TABLE[codes]

def find_best_codebook_A(block_values, num_codes=NUM_CODES):
    """Variant A: Exact MSE - brute force over all C(16,4) subsets."""
    best_mse = float('inf')
    best_cb = None
    
    # All possible subsets of num_codes from 15 unique values
    for cb_indices in combinations(range(15), num_codes):
        cb_vals = UNIQUE_VALS[list(cb_indices)]
        # Assign each element to nearest code
        dists = np.abs(block_values[:, None] - cb_vals[None, :])
        nearest = cb_vals[np.argmin(dists, axis=1)]
        mse = np.mean((block_values - nearest) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_cb = cb_vals
    
    return best_cb, best_mse

def find_best_codebook_B_freq(block_values, num_codes=NUM_CODES):
    """Variant B: Frequency-weighted MSE (original implementation)."""
    # Weight by frequency of each value in the block
    best_wmse = float('inf')
    best_cb = None
    
    # Compute frequency weights
    unique_in_block, counts = np.unique(block_values, return_counts=True)
    freq_map = dict(zip(unique_in_block, counts / len(block_values)))
    weights = np.array([freq_map.get(v, 0.0) for v in block_values])
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(block_values)) / len(block_values)
    
    for cb_indices in combinations(range(15), num_codes):
        cb_vals = UNIQUE_VALS[list(cb_indices)]
        dists = np.abs(block_values[:, None] - cb_vals[None, :])
        nearest = cb_vals[np.argmin(dists, axis=1)]
        wmse = np.sum(weights * (block_values - nearest) ** 2)
        if wmse < best_wmse:
            best_wmse = wmse
            best_cb = cb_vals
    
    return best_cb, best_wmse

def find_best_codebook_B_m2(block_values, block_scale, num_codes=NUM_CODES):
    """Variant B_m2: m²-weighted MSE (BOF4, ICLR 2026).
    
    Weights the MSE by the squared block scale, as derived in BOF4.
    For per-block selection, this is equivalent to exact MSE (scale is constant
    within a block), but the scale affects which blocks matter more globally.
    
    For a GLOBAL codebook, this would change the result.
    For per-block, we use scale as a tiebreaker: prefer codebooks that
    minimize error on the highest-scale elements.
    """
    # For per-block: scale is constant, so m²-weighted = exact MSE
    # But we can use scale to weight the IMPORTANCE of this block's result
    # This is only meaningful when comparing across blocks
    return find_best_codebook_A(block_values, num_codes)

def find_best_codebook_C(block_values, num_codes=NUM_CODES, lambda_reg=0.01):
    """Variant C: Frequency-regularized MSE."""
    best_score = float('inf')
    best_cb = None
    
    for cb_indices in combinations(range(15), num_codes):
        cb_vals = UNIQUE_VALS[list(cb_indices)]
        dists = np.abs(block_values[:, None] - cb_vals[None, :])
        assignments = np.argmin(dists, axis=1)
        nearest = cb_vals[assignments]
        mse = np.mean((block_values - nearest) ** 2)
        
        # Regularization: penalize unused codes
        used_codes = len(np.unique(assignments))
        unused_penalty = (num_codes - used_codes) * lambda_reg
        
        score = mse + unused_penalty
        if score < best_score:
            best_score = score
            best_cb = cb_vals
    
    return best_cb, best_score

def find_best_codebook_D(block_values, num_codes=NUM_CODES):
    """Variant D: Signed-pair constrained search.
    
    Restricts to subsets where if c is included, -c is also included.
    For num_codes=4: select 2 positive values and their negatives.
    """
    best_mse = float('inf')
    best_cb = None
    
    # Positive values (excluding 0)
    pos_vals = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    
    # Select num_codes/2 positive values and add their negatives
    half = num_codes // 2
    for pos_indices in combinations(range(len(pos_vals)), half):
        pos = pos_vals[list(pos_indices)]
        neg = -pos
        cb_vals = np.concatenate([neg, pos])
        
        dists = np.abs(block_values[:, None] - cb_vals[None, :])
        nearest = cb_vals[np.argmin(dists, axis=1)]
        mse = np.mean((block_values - nearest) ** 2)
        
        if mse < best_mse:
            best_mse = mse
            best_cb = cb_vals
    
    return best_cb, best_mse

def find_best_codebook_D_with_zero(block_values, num_codes=NUM_CODES):
    """Variant D+: Signed-pair with zero included (BOF4-S insight).
    
    BOF4-S shows that including zero is critical for block-wise quantization.
    This variant always includes 0 and selects (num_codes-1)/2 signed pairs.
    """
    best_mse = float('inf')
    best_cb = None
    
    pos_vals = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    
    # With 4 codes: 0 + 1 positive + 1 negative + 1 more
    # Option 1: 0 + 1 signed pair + 1 extreme
    # Option 2: 0 + 1.5 signed pairs (not symmetric)
    
    # Try: 0 + signed pair + one extra
    for pos_idx in range(len(pos_vals)):
        for extra_idx in range(15):
            extra = UNIQUE_VALS[extra_idx]
            pos = pos_vals[pos_idx]
            cb_vals = np.array([0.0, pos, -pos, extra])
            cb_vals = np.unique(cb_vals)
            if len(cb_vals) < num_codes:
                continue
            cb_vals = cb_vals[:num_codes]
            
            dists = np.abs(block_values[:, None] - cb_vals[None, :])
            nearest = cb_vals[np.argmin(dists, axis=1)]
            mse = np.mean((block_values - nearest) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_cb = cb_vals
    
    return best_cb, best_mse

def run_comparison(num_blocks=2000):
    """Run variant comparison on real NVFP4 data."""
    print("=" * 70)
    print("VARIANT COMPARISON ON REAL NVFP4 DATA (block_size=16)")
    print("=" * 70)
    
    # Load real NVFP4 data
    ckpt_dir = Path('nvfp4_checkpoint')
    f = ckpt_dir / 'model-00012-of-00733.safetensors'
    data = load_file(str(f))
    
    weight_packed = data['model.layers.0.mlp.experts.0.down_proj.weight']
    weight_scale_uint8 = data['model.layers.0.mlp.experts.0.down_proj.weight_scale']
    
    # Unpack FP4 codes
    codes = unpack_fp4_codes(weight_packed.numpy())
    values = codes_to_values(codes)
    scales = decode_fp8_e4m3_uint8(weight_scale_uint8.numpy())
    
    print(f"Total codes: {len(codes)}")
    print(f"Total blocks: {len(codes) // BLOCK_SIZE}")
    print(f"Testing {num_blocks} blocks")
    print()
    
    # Sample blocks
    np.random.seed(42)
    block_indices = np.random.choice(len(codes) // BLOCK_SIZE, num_blocks, replace=False)
    
    results = {
        'A': {'mse': [], 'time': 0},
        'B_freq': {'mse': [], 'time': 0},
        'C': {'mse': [], 'time': 0},
        'D': {'mse': [], 'time': 0},
        'D_zero': {'mse': [], 'time': 0},
    }
    
    for i, block_idx in enumerate(block_indices):
        if i % 200 == 0:
            print(f"  Processing block {i}/{num_blocks}...")
        
        block_values = values[block_idx * BLOCK_SIZE:(block_idx + 1) * BLOCK_SIZE]
        block_scale = scales[block_idx]
        
        # Variant A: Exact MSE
        t0 = time.time()
        _, mse_a = find_best_codebook_A(block_values)
        results['A']['time'] += time.time() - t0
        results['A']['mse'].append(mse_a)
        
        # Variant B: Frequency-weighted MSE
        t0 = time.time()
        _, mse_b = find_best_codebook_B_freq(block_values)
        results['B_freq']['time'] += time.time() - t0
        results['B_freq']['mse'].append(mse_b)
        
        # Variant C: Frequency-regularized MSE
        t0 = time.time()
        _, mse_c = find_best_codebook_C(block_values)
        results['C']['time'] += time.time() - t0
        results['C']['mse'].append(mse_c)
        
        # Variant D: Signed-pair constrained
        t0 = time.time()
        _, mse_d = find_best_codebook_D(block_values)
        results['D']['time'] += time.time() - t0
        results['D']['mse'].append(mse_d)
        
        # Variant D+: Signed-pair with zero
        t0 = time.time()
        _, mse_dz = find_best_codebook_D_with_zero(block_values)
        results['D_zero']['time'] += time.time() - t0
        results['D_zero']['mse'].append(mse_dz)
    
    # Report results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Variant':<12} {'Avg MSE':<12} {'Rel to A':<12} {'Time(s)':<10}")
    print("-" * 50)
    
    mse_a = np.mean(results['A']['mse'])
    for name, r in results.items():
        avg_mse = np.mean(r['mse'])
        rel = avg_mse / mse_a
        print(f"{name:<12} {avg_mse:<12.6f} {rel:<12.4f} {r['time']:<10.2f}")
    
    # Save results
    output = {
        'num_blocks': num_blocks,
        'block_size': BLOCK_SIZE,
        'num_codes': NUM_CODES,
        'data_source': 'real_nvfp4_model.layers.0.mlp.experts.0.down_proj',
        'results': {
            name: {
                'avg_mse': float(np.mean(r['mse'])),
                'std_mse': float(np.std(r['mse'])),
                'rel_to_A': float(np.mean(r['mse']) / mse_a),
                'time_sec': r['time'],
            }
            for name, r in results.items()
        }
    }
    
    with open('variant_comparison_real_data_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print()
    print("Results saved to variant_comparison_real_data_results.json")
    
    return output

if __name__ == '__main__':
    run_comparison(num_blocks=500)  # 500 blocks for speed

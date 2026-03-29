#!/usr/bin/env python3
"""Fast Shannon entropy measurement of NVFP4 codes on Qwen3.5-35B-A3B."""
import glob
import math
import sys
import time
from collections import Counter

import numpy as np
import torch
from safetensors import safe_open

FP4_MAX = 6.0
FP8_E4M3_MAX = 448.0
FP4_GLOBAL_SCALE_MAX = FP8_E4M3_MAX * FP4_MAX
BLOCK_SIZE = 16

POSITIVE_FP4 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
BOUNDARIES = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)


def quantize_expert_fast(weight_np):
    """Quantize one expert weight matrix [out, in] to NVFP4 codes.
    
    Returns code histogram (16,) and scale histogram dict.
    Uses searchsorted for O(n log k) instead of O(n*k).
    """
    rows, cols = weight_np.shape
    amax = np.abs(weight_np).max()
    if amax == 0:
        return np.zeros(16, dtype=np.int64), {}, 0, 0

    global_scale = FP4_GLOBAL_SCALE_MAX / amax
    scaled = weight_np * global_scale

    pad = (BLOCK_SIZE - cols % BLOCK_SIZE) % BLOCK_SIZE
    if pad > 0:
        scaled = np.pad(scaled, ((0, 0), (0, pad)))
    
    padded_cols = scaled.shape[1]
    blocks = scaled.reshape(rows, padded_cols // BLOCK_SIZE, BLOCK_SIZE)
    
    block_amax = np.abs(blocks).max(axis=2)
    raw_scales = block_amax / FP4_MAX
    
    fp8_scales = raw_scales.astype(np.float16).astype(np.float32)
    safe_scales = np.where(fp8_scales == 0, 1.0, fp8_scales)
    
    normalized = blocks / safe_scales[:, :, np.newaxis]
    
    signs = np.sign(normalized)
    abs_vals = np.abs(normalized)
    
    pos_codes = np.searchsorted(BOUNDARIES, abs_vals.ravel()).reshape(abs_vals.shape)
    
    full_codes = np.where(signs >= 0, pos_codes + 8, 7 - pos_codes)
    
    if pad > 0:
        full_codes = full_codes.reshape(rows, -1)[:, :cols]
    
    code_hist = np.zeros(16, dtype=np.int64)
    for c in range(16):
        code_hist[c] = np.sum(full_codes == c)
    
    n_elements = rows * cols
    n_blocks = rows * (padded_cols // BLOCK_SIZE)
    
    scale_hist = {}
    unique_s, counts_s = np.unique(np.round(fp8_scales.ravel(), 6), return_counts=True)
    for v, cnt in zip(unique_s, counts_s):
        scale_hist[round(float(v), 6)] = int(cnt)
    
    return code_hist, scale_hist, n_elements, n_blocks


def shannon_entropy(counts_arr_or_dict, total):
    H = 0.0
    if isinstance(counts_arr_or_dict, dict):
        vals = counts_arr_or_dict.values()
    else:
        vals = counts_arr_or_dict
    for c in vals:
        if c > 0:
            p = c / total
            H -= p * math.log2(p)
    return H


def main():
    files = sorted(glob.glob(
        '/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/*/model.safetensors-*.safetensors'
    ))
    print(f'Found {len(files)} safetensor files', flush=True)

    total_fp4 = np.zeros(16, dtype=np.int64)
    total_elem = 0
    total_blk = 0
    all_scale_counts = Counter()
    layer_fp4 = {}
    
    t0 = time.time()
    
    for fi, fpath in enumerate(files):
        fname = fpath.split('/')[-1]
        print(f'\n[{fi+1}/{len(files)}] {fname}', flush=True)
        
        with safe_open(fpath, framework='pt', device='cpu') as f:
            for key in f.keys():
                if 'experts' not in key:
                    continue
                if 'gate_up_proj' not in key and 'down_proj' not in key:
                    continue
                if 'mtp.' in key or 'shared_expert' in key:
                    continue
                
                layer = int(key.split('.')[3])
                proj = 'w1' if 'gate_up' in key else 'w2'
                packed = f.get_tensor(key).float().numpy()
                n_exp = packed.shape[0]
                
                print(f'  L{layer:02d} {proj} [{n_exp}x{packed.shape[1]}x{packed.shape[2]}]', end='', flush=True)
                
                lkey = (layer, proj)
                if lkey not in layer_fp4:
                    layer_fp4[lkey] = np.zeros(16, dtype=np.int64)
                
                batch_t = time.time()
                for ei in range(n_exp):
                    hist, shist, ne, nb = quantize_expert_fast(packed[ei])
                    total_fp4 += hist
                    layer_fp4[lkey] += hist
                    total_elem += ne
                    total_blk += nb
                    for v, c in shist.items():
                        all_scale_counts[v] += c
                
                elapsed = time.time() - t0
                batch_time = time.time() - batch_t
                print(f'  {batch_time:.0f}s (total {elapsed:.0f}s, {total_elem/1e9:.2f}B elem)', flush=True)

    elapsed = time.time() - t0
    
    print(f'\n{"="*60}')
    print(f'COMPLETED in {elapsed:.0f}s')
    print(f'Total FP4 elements: {total_elem:,}')
    print(f'Total FP8 blocks:   {total_blk:,}')
    
    labels = ['-6','-4','-3','-2','-1.5','-1','-0.5','-0',
              '+0','+0.5','+1','+1.5','+2','+3','+4','+6']
    
    print(f'\n{"="*60}')
    print('FP4 CODE DISTRIBUTION')
    print(f'{"="*60}')
    total = total_fp4.sum()
    for i in range(16):
        pct = 100 * total_fp4[i] / total
        bar = '#' * int(pct * 2)
        print(f'  {i:2d} ({labels[i]:>5s}): {total_fp4[i]:>14,} ({pct:6.2f}%) {bar}')
    
    H_fp4 = shannon_entropy(total_fp4, total)
    print(f'\nShannon entropy (FP4):  {H_fp4:.4f} bits  (max 4.0)')
    print(f'Compression potential:  {(1 - H_fp4/4)*100:.1f}%')
    
    H_fp8 = shannon_entropy(all_scale_counts, total_blk)
    print(f'\nShannon entropy (FP8):  {H_fp8:.4f} bits  (max 8.0)')
    print(f'Unique FP8 values:      {len(all_scale_counts)}')
    
    print(f'\nTop 15 FP8 scale values:')
    for val, cnt in sorted(all_scale_counts.items(), key=lambda x: -x[1])[:15]:
        print(f'  {val:10.4f}: {cnt:>12,} ({100*cnt/total_blk:5.2f}%)')
    
    eff = H_fp4 + H_fp8 / BLOCK_SIZE
    print(f'\n{"="*60}')
    print(f'LOSSLESS COMPRESSION SUMMARY')
    print(f'{"="*60}')
    print(f'FP4 code entropy:      {H_fp4:.4f} bits/element')
    print(f'FP8 scale entropy:     {H_fp8:.4f} bits/scale  ({H_fp8/BLOCK_SIZE:.4f} bits/element)')
    print(f'Effective bits/elem:   {eff:.4f}')
    print(f'NVFP4 uncompressed:    4.5000 bits/element')
    print(f'Lossless savings:      {(1 - eff/4.5)*100:.1f}%')
    print(f'vs BF16 compression:   {16/eff:.1f}x')
    
    print(f'\n{"="*60}')
    print(f'PER-LAYER ENTROPY')
    print(f'{"="*60}')
    for lkey in sorted(layer_fp4.keys()):
        layer, proj = lkey
        lt = layer_fp4[lkey].sum()
        H = shannon_entropy(layer_fp4[lkey], lt)
        print(f'  L{layer:02d} {proj}: {H:.4f} bits ({(1-H/4)*100:.1f}% savings)')
    
    print(f'\nZERO ANALYSIS:')
    neg_zero = int(total_fp4[7])
    pos_zero = int(total_fp4[8])
    print(f'  -0 (code 7): {neg_zero:,} ({100*neg_zero/total:.2f}%)')
    print(f'  +0 (code 8): {pos_zero:,} ({100*pos_zero/total:.2f}%)')
    print(f'  RaZeR candidates: {min(neg_zero, pos_zero):,}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Full code entropy analysis on ALL experts, ALL layers, block sizes 16 and 32.

Measures:
1. Marginal entropy per element (should match our prior 3.977)
2. Pairwise mutual information between adjacent elements within blocks
3. Conditional entropy: H(x_i | x_{i-1}) for consecutive elements
4. Position-wise entropy: does position within block affect code distribution?
"""
import sys
import time
import json
import math
from pathlib import Path
from collections import Counter

import numpy as np
from safetensors import safe_open

FP4_UNIQUE = np.array([-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
BOUNDARIES = np.array([-5, -3.5, -2.5, -1.75, -1.25, -0.75, -0.25, 0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=np.float32)
N_CODES = 15

MODEL_DIR = "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots"


def quantize_to_codes(block):
    amax = np.abs(block).max()
    if amax == 0:
        return np.zeros(len(block), dtype=np.int8)
    scale = np.float16(amax / 6.0).astype(np.float32)
    if scale == 0:
        scale = 1.0
    norm = block / scale
    return np.clip(np.searchsorted(BOUNDARIES, norm), 0, N_CODES - 1).astype(np.int8)


def shannon_entropy(counter, total):
    H = 0.0
    for c in counter.values():
        if c > 0:
            p = c / total
            H -= p * math.log2(p)
    return H


def main():
    model_dir = Path(MODEL_DIR)
    snapshot = list(model_dir.iterdir())[0]
    files = sorted(snapshot.glob("*.safetensors"))

    print(f"Processing ALL experts, ALL layers", flush=True)
    t0 = time.time()

    marginal_counts = Counter()
    position_counts_16 = [Counter() for _ in range(16)]
    position_counts_32 = [Counter() for _ in range(32)]
    pair_counts = Counter()
    total_elements = 0
    total_blocks_16 = 0
    total_blocks_32 = 0
    total_pairs = 0

    for fi, fpath in enumerate(files):
        fname = fpath.name
        print(f"\n[{fi+1}/{len(files)}] {fname}", flush=True)

        with safe_open(str(fpath), framework='pt', device='cpu') as f:
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

                print(f"  L{layer:02d} {proj} [{n_exp}x{packed.shape[1]}x{packed.shape[2]}]", end='', flush=True)
                batch_t = time.time()

                for ei in range(n_exp):
                    weight = packed[ei]
                    rows, cols = weight.shape

                    pad32 = (32 - cols % 32) % 32
                    if pad32 > 0:
                        weight = np.pad(weight, ((0, 0), (0, pad32)))

                    padded_cols = weight.shape[1]

                    blocks_16 = weight.reshape(-1, 16)
                    for bi in range(blocks_16.shape[0]):
                        codes = quantize_to_codes(blocks_16[bi])

                        for pos in range(16):
                            c = int(codes[pos])
                            marginal_counts[c] += 1
                            position_counts_16[pos][c] += 1

                        for j in range(15):
                            pair_counts[(int(codes[j]), int(codes[j+1]))] += 1
                            total_pairs += 1

                        total_blocks_16 += 1

                    blocks_32 = weight.reshape(-1, 32)
                    for bi in range(blocks_32.shape[0]):
                        codes32 = np.concatenate([
                            quantize_to_codes(blocks_32[bi][:16]),
                            quantize_to_codes(blocks_32[bi][16:])
                        ])
                        for pos in range(32):
                            position_counts_32[pos][int(codes32[pos])] += 1

                        pair_counts[(int(codes32[15]), int(codes32[16]))] += 1
                        total_pairs += 1
                        total_blocks_32 += 1

                    total_elements += rows * cols

                elapsed = time.time() - batch_t
                print(f"  {elapsed:.0f}s (total {total_elements/1e9:.2f}B)", flush=True)

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"COMPLETED in {elapsed:.0f}s")
    print(f"Total elements: {total_elements:,}")
    print(f"Total block-16: {total_blocks_16:,}")
    print(f"Total block-32: {total_blocks_32:,}")
    print(f"Total adjacent pairs: {total_pairs:,}")

    H_marginal = shannon_entropy(marginal_counts, sum(marginal_counts.values()))
    print(f"\n{'='*60}")
    print(f"MARGINAL ENTROPY: {H_marginal:.4f} bits (max {math.log2(N_CODES):.4f})")

    print(f"\n{'='*60}")
    print(f"POSITION-WISE ENTROPY (block-16)")
    for pos in range(16):
        total_pos = sum(position_counts_16[pos].values())
        H_pos = shannon_entropy(position_counts_16[pos], total_pos)
        print(f"  Position {pos:2d}: H={H_pos:.4f} bits")

    print(f"\nPOSITION-WISE ENTROPY (block-32)")
    for pos in range(32):
        total_pos = sum(position_counts_32[pos].values())
        H_pos = shannon_entropy(position_counts_32[pos], total_pos)
        print(f"  Position {pos:2d}: H={H_pos:.4f} bits")

    H_pair = shannon_entropy(pair_counts, total_pairs)
    H_conditional = H_pair - H_marginal
    MI = 2 * H_marginal - H_pair

    print(f"\n{'='*60}")
    print(f"PAIRWISE ANALYSIS (adjacent elements)")
    print(f"  H(x_i): {H_marginal:.4f} bits")
    print(f"  H(x_i, x_{{i+1}}): {H_pair:.4f} bits")
    print(f"  H(x_{{i+1}} | x_i): {H_conditional:.4f} bits")
    print(f"  MI(x_i; x_{{i+1}}): {MI:.4f} bits")
    print(f"  Correlation ratio: {MI/H_marginal*100:.2f}%")

    if MI > 0.1:
        print(f"  SIGNIFICANT correlation — {MI:.3f} bits of mutual info per adjacent pair")
    elif MI > 0.01:
        print(f"  WEAK correlation — {MI:.3f} bits")
    else:
        print(f"  NEGLIGIBLE correlation — elements are effectively independent")

    print(f"\n  Theoretical compression limit using pairwise context:")
    print(f"    Independent: {H_marginal:.4f} bits/elem")
    print(f"    With pairwise context: {H_conditional:.4f} bits/elem")
    print(f"    Savings: {(1-H_conditional/H_marginal)*100:.2f}%")

    results = {
        'total_elements': total_elements,
        'total_blocks_16': total_blocks_16,
        'total_blocks_32': total_blocks_32,
        'marginal_entropy': H_marginal,
        'pair_joint_entropy': H_pair,
        'conditional_entropy': H_conditional,
        'mutual_information': MI,
        'position_entropy_16': [shannon_entropy(position_counts_16[p], sum(position_counts_16[p].values())) for p in range(16)],
        'position_entropy_32': [shannon_entropy(position_counts_32[p], sum(position_counts_32[p].values())) for p in range(32)],
    }

    out_path = "/code/tensorrt_llm/scripts/nvfp4_compress/code_entropy_full.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()

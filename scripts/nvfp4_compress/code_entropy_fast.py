#!/usr/bin/env python3
"""Fast vectorized code entropy on ALL experts, ALL layers, block sizes 16 and 32.
No Python loops over blocks — fully numpy vectorized.
Uses multiprocessing across 16 workers for parallel tensor processing.
"""
import sys
import time
import json
import math
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from safetensors import safe_open

N_CODES = 15
FP4_UNIQUE = np.array([-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
BOUNDARIES = np.array([-5, -3.5, -2.5, -1.75, -1.25, -0.75, -0.25, 0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=np.float32)

MODEL_DIR = "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots"


def quantize_block_vectorized(weight_2d, block_size=16):
    """Vectorized: quantize entire weight matrix to FP4 codes in blocks."""
    rows, cols = weight_2d.shape
    pad = (block_size - cols % block_size) % block_size
    if pad > 0:
        weight_2d = np.pad(weight_2d, ((0, 0), (0, pad)))
    blocks = weight_2d.reshape(-1, block_size)
    amax = np.abs(blocks).max(axis=1, keepdims=True)
    amax[amax == 0] = 1.0
    scales = np.float16(amax / 6.0).astype(np.float32)
    scales[scales == 0] = 1.0
    normalized = blocks / scales
    codes = np.searchsorted(BOUNDARIES, normalized.ravel()).reshape(blocks.shape)
    codes = np.clip(codes, 0, N_CODES - 1).astype(np.int8)
    return codes, rows * cols


def process_one_expert(args):
    """Process one expert weight: return marginal counts, position counts, pair counts."""
    fpath, key, expert_idx = args

    with safe_open(fpath, framework='pt', device='cpu') as f:
        weight = f.get_tensor(key)[expert_idx].float().numpy()

    rows, cols = weight.shape

    codes_16, n_elem = quantize_block_vectorized(weight, block_size=16)
    n_blocks_16 = codes_16.shape[0]

    marginal = np.zeros(N_CODES, dtype=np.int64)
    for c in range(N_CODES):
        marginal[c] = np.sum(codes_16 == c)

    pos_counts_16 = np.zeros((16, N_CODES), dtype=np.int64)
    for pos in range(16):
        col = codes_16[:, pos]
        for c in range(N_CODES):
            pos_counts_16[pos, c] = np.sum(col == c)

    left = codes_16[:, :-1].ravel()
    right = codes_16[:, 1:].ravel()
    pair_flat = left.astype(np.int32) * N_CODES + right.astype(np.int32)
    pair_hist = np.bincount(pair_flat, minlength=N_CODES * N_CODES)

    pad32 = (32 - cols % 32) % 32
    if pad32 > 0:
        weight32 = np.pad(weight, ((0, 0), (0, pad32)))
    else:
        weight32 = weight
    blocks_32_raw = weight32.reshape(-1, 32)
    half1 = blocks_32_raw[:, :16]
    half2 = blocks_32_raw[:, 16:]
    codes_h1, _ = quantize_block_vectorized(half1.reshape(-1, half1.shape[1]), 16)
    codes_h2, _ = quantize_block_vectorized(half2.reshape(-1, half2.shape[1]), 16)
    cross_left = codes_h1[:, -1].ravel()
    cross_right = codes_h2[:, 0].ravel()
    cross_flat = cross_left.astype(np.int32) * N_CODES + cross_right.astype(np.int32)
    cross_hist = np.bincount(cross_flat, minlength=N_CODES * N_CODES)

    pos_counts_32 = np.zeros((32, N_CODES), dtype=np.int64)
    codes_32 = np.concatenate([codes_h1, codes_h2], axis=1)
    for pos in range(32):
        col = codes_32[:, pos]
        for c in range(N_CODES):
            pos_counts_32[pos, c] = np.sum(col == c)

    return {
        'marginal': marginal,
        'pos16': pos_counts_16,
        'pos32': pos_counts_32,
        'pair_hist': pair_hist,
        'cross_hist': cross_hist,
        'n_elem': n_elem,
        'n_blocks_16': n_blocks_16,
        'n_blocks_32': codes_32.shape[0],
        'n_pairs_within': len(left),
        'n_pairs_cross': len(cross_left),
    }


def shannon_entropy_from_counts(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def main():
    model_dir = Path(MODEL_DIR)
    snapshot = list(model_dir.iterdir())[0]
    files = sorted(snapshot.glob("*.safetensors"))

    print("Building task list for ALL experts...", flush=True)
    tasks = []
    for fpath in files:
        with safe_open(str(fpath), framework='pt', device='cpu') as f:
            for key in f.keys():
                if 'experts' not in key:
                    continue
                if 'gate_up_proj' not in key and 'down_proj' not in key:
                    continue
                if 'mtp.' in key or 'shared_expert' in key:
                    continue
                n_exp = f.get_tensor(key).shape[0]
                for ei in range(n_exp):
                    tasks.append((str(fpath), key, ei))

    print(f"Total tasks: {len(tasks)} experts", flush=True)
    t0 = time.time()

    agg_marginal = np.zeros(N_CODES, dtype=np.int64)
    agg_pos16 = np.zeros((16, N_CODES), dtype=np.int64)
    agg_pos32 = np.zeros((32, N_CODES), dtype=np.int64)
    agg_pair = np.zeros(N_CODES * N_CODES, dtype=np.int64)
    agg_cross = np.zeros(N_CODES * N_CODES, dtype=np.int64)
    total_elem = 0
    total_b16 = 0
    total_b32 = 0
    total_pairs_within = 0
    total_pairs_cross = 0
    done = 0

    N_WORKERS = 16
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(process_one_expert, t): t for t in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            agg_marginal += r['marginal']
            agg_pos16 += r['pos16']
            agg_pos32 += r['pos32']
            agg_pair += r['pair_hist']
            agg_cross += r['cross_hist']
            total_elem += r['n_elem']
            total_b16 += r['n_blocks_16']
            total_b32 += r['n_blocks_32']
            total_pairs_within += r['n_pairs_within']
            total_pairs_cross += r['n_pairs_cross']
            done += 1
            if done % 500 == 0:
                elapsed = time.time() - t0
                print(f"  {done}/{len(tasks)} experts ({total_elem/1e9:.1f}B elem, {elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"\nCOMPLETED in {elapsed:.0f}s")
    print(f"Total elements: {total_elem:,}")
    print(f"Total block-16: {total_b16:,}")
    print(f"Total block-32: {total_b32:,}")

    H_marginal = shannon_entropy_from_counts(agg_marginal)

    print(f"\n{'='*60}")
    print(f"MARGINAL ENTROPY: {H_marginal:.4f} bits (max {math.log2(N_CODES):.4f})")
    print(f"Code distribution:")
    total_m = agg_marginal.sum()
    for i in range(N_CODES):
        pct = 100 * agg_marginal[i] / total_m
        print(f"  Code {i:2d} ({FP4_UNIQUE[i]:+5.1f}): {agg_marginal[i]:>14,} ({pct:5.2f}%)")

    print(f"\n{'='*60}")
    print(f"POSITION-WISE ENTROPY (block-16)")
    for pos in range(16):
        H = shannon_entropy_from_counts(agg_pos16[pos])
        print(f"  Pos {pos:2d}: {H:.4f} bits")

    print(f"\nPOSITION-WISE ENTROPY (block-32)")
    for pos in range(32):
        H = shannon_entropy_from_counts(agg_pos32[pos])
        print(f"  Pos {pos:2d}: {H:.4f} bits")

    pair_matrix = agg_pair.reshape(N_CODES, N_CODES)
    total_within = total_pairs_within
    H_pair_within = shannon_entropy_from_counts(agg_pair)
    H_cond_within = H_pair_within - H_marginal
    MI_within = 2 * H_marginal - H_pair_within

    cross_matrix = agg_cross.reshape(N_CODES, N_CODES)
    total_cross = total_pairs_cross
    H_pair_cross = shannon_entropy_from_counts(agg_cross)
    H_cond_cross = H_pair_cross - H_marginal
    MI_cross = 2 * H_marginal - H_pair_cross

    print(f"\n{'='*60}")
    print(f"PAIRWISE: WITHIN block-16 (adjacent elements, same block scale)")
    print(f"  H(x_i): {H_marginal:.4f}")
    print(f"  H(x_i, x_{{i+1}}): {H_pair_within:.4f}")
    print(f"  H(x_{{i+1}} | x_i): {H_cond_within:.4f}")
    print(f"  MI: {MI_within:.4f} bits ({MI_within/H_marginal*100:.2f}% of marginal)")
    print(f"  Pairs: {total_within:,}")

    print(f"\n{'='*60}")
    print(f"PAIRWISE: ACROSS block-16 boundary (last elem block N, first elem block N+1)")
    print(f"  H(x_i): {H_marginal:.4f}")
    print(f"  H(x_i, x_{{i+1}}): {H_pair_cross:.4f}")
    print(f"  H(x_{{i+1}} | x_i): {H_cond_cross:.4f}")
    print(f"  MI: {MI_cross:.4f} bits ({MI_cross/H_marginal*100:.2f}% of marginal)")
    print(f"  Pairs: {total_cross:,}")

    print(f"\n{'='*60}")
    print(f"COMPRESSION LIMITS")
    print(f"  Independent:         {H_marginal:.4f} bits/elem")
    print(f"  Within-block context: {H_cond_within:.4f} bits/elem (saves {(1-H_cond_within/H_marginal)*100:.2f}%)")
    print(f"  Cross-block context:  {H_cond_cross:.4f} bits/elem (saves {(1-H_cond_cross/H_marginal)*100:.2f}%)")

    results = {
        'total_elements': int(total_elem),
        'marginal_entropy': H_marginal,
        'pair_joint_within': H_pair_within,
        'conditional_within': H_cond_within,
        'mi_within': MI_within,
        'pair_joint_cross': H_pair_cross,
        'conditional_cross': H_cond_cross,
        'mi_cross': MI_cross,
        'position_entropy_16': [float(shannon_entropy_from_counts(agg_pos16[p])) for p in range(16)],
        'position_entropy_32': [float(shannon_entropy_from_counts(agg_pos32[p])) for p in range(32)],
        'marginal_distribution': agg_marginal.tolist(),
    }

    out_path = "/code/tensorrt_llm/scripts/nvfp4_compress/code_entropy_full.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()

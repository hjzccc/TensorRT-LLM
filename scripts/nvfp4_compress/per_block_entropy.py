#!/usr/bin/env python3
"""Per-block Shannon entropy: for each NVFP4 block of 16 (and 32), compute the
entropy of the code distribution WITHIN that block. Fully vectorized, 16 workers.
Stores: histogram of per-block entropies, unique-code-count distribution, and summary stats.
"""
import sys
import time
import json
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from safetensors import safe_open

N_CODES = 15
BOUNDARIES = np.array([-5, -3.5, -2.5, -1.75, -1.25, -0.75, -0.25, 0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=np.float32)
MODEL_DIR = "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots"


def process_expert(args):
    fpath, key, expert_idx = args

    with safe_open(fpath, framework='pt', device='cpu') as f:
        weight = f.get_tensor(key)[expert_idx].float().numpy()

    rows, cols = weight.shape
    results = {}

    for bs in [16, 32]:
        pad = (bs - cols % bs) % bs
        if pad > 0:
            w = np.pad(weight, ((0, 0), (0, pad)))
        else:
            w = weight

        if bs == 32:
            blocks_raw = w.reshape(-1, 32)
            half1 = blocks_raw[:, :16]
            half2 = blocks_raw[:, 16:]

            h1_blocks = half1.reshape(-1, 16)
            amax1 = np.abs(h1_blocks).max(axis=1, keepdims=True)
            amax1[amax1 == 0] = 1.0
            s1 = np.float16(amax1 / 6.0).astype(np.float32)
            s1[s1 == 0] = 1.0
            n1 = h1_blocks / s1
            c1 = np.clip(np.searchsorted(BOUNDARIES, n1.ravel()).reshape(n1.shape), 0, N_CODES - 1).astype(np.int8)

            h2_blocks = half2.reshape(-1, 16)
            amax2 = np.abs(h2_blocks).max(axis=1, keepdims=True)
            amax2[amax2 == 0] = 1.0
            s2 = np.float16(amax2 / 6.0).astype(np.float32)
            s2[s2 == 0] = 1.0
            n2 = h2_blocks / s2
            c2 = np.clip(np.searchsorted(BOUNDARIES, n2.ravel()).reshape(n2.shape), 0, N_CODES - 1).astype(np.int8)

            codes = np.concatenate([c1, c2], axis=1)
        else:
            blocks = w.reshape(-1, bs)
            amax = np.abs(blocks).max(axis=1, keepdims=True)
            amax[amax == 0] = 1.0
            scales = np.float16(amax / 6.0).astype(np.float32)
            scales[scales == 0] = 1.0
            normalized = blocks / scales
            codes = np.clip(np.searchsorted(BOUNDARIES, normalized.ravel()).reshape(blocks.shape), 0, N_CODES - 1).astype(np.int8)

        n_blocks = codes.shape[0]

        onehot = np.zeros((n_blocks, N_CODES), dtype=np.int32)
        for c in range(N_CODES):
            onehot[:, c] = (codes == c).sum(axis=1)

        freqs = onehot.astype(np.float64) / bs
        freqs[freqs == 0] = 1.0
        log_freqs = np.log2(freqs)
        freqs[onehot == 0] = 0.0
        log_freqs[onehot == 0] = 0.0
        entropies = -(freqs * log_freqs).sum(axis=1)

        unique_counts = (onehot > 0).sum(axis=1)

        entropy_hist, entropy_bins = np.histogram(entropies, bins=100, range=(0.0, 4.0))
        unique_hist = np.bincount(unique_counts, minlength=N_CODES + 1)

        results[bs] = {
            'n_blocks': n_blocks,
            'entropy_sum': float(entropies.sum()),
            'entropy_sq_sum': float((entropies ** 2).sum()),
            'entropy_min': float(entropies.min()),
            'entropy_max': float(entropies.max()),
            'entropy_hist': entropy_hist.tolist(),
            'entropy_bins': entropy_bins.tolist(),
            'unique_hist': unique_hist.tolist(),
            'unique_sum': int(unique_counts.sum()),
        }

    return results


def main():
    model_dir = Path(MODEL_DIR)
    snapshot = list(model_dir.iterdir())[0]
    files = sorted(snapshot.glob("*.safetensors"))

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

    print(f"Total: {len(tasks)} experts", flush=True)
    t0 = time.time()

    agg = {}
    for bs in [16, 32]:
        agg[bs] = {
            'n_blocks': 0,
            'entropy_sum': 0.0,
            'entropy_sq_sum': 0.0,
            'entropy_min': 999.0,
            'entropy_max': 0.0,
            'entropy_hist': np.zeros(100, dtype=np.int64),
            'unique_hist': np.zeros(N_CODES + 1, dtype=np.int64),
            'unique_sum': 0,
        }

    done = 0
    with ProcessPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(process_expert, t): t for t in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            for bs in [16, 32]:
                a = agg[bs]
                rb = r[bs]
                a['n_blocks'] += rb['n_blocks']
                a['entropy_sum'] += rb['entropy_sum']
                a['entropy_sq_sum'] += rb['entropy_sq_sum']
                a['entropy_min'] = min(a['entropy_min'], rb['entropy_min'])
                a['entropy_max'] = max(a['entropy_max'], rb['entropy_max'])
                a['entropy_hist'] += np.array(rb['entropy_hist'], dtype=np.int64)
                a['unique_hist'] += np.array(rb['unique_hist'], dtype=np.int64)
                a['unique_sum'] += rb['unique_sum']
            done += 1
            if done % 1000 == 0:
                elapsed = time.time() - t0
                print(f"  {done}/{len(tasks)} experts ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.0f}s\n", flush=True)

    results = {}
    for bs in [16, 32]:
        a = agg[bs]
        n = a['n_blocks']
        mean_ent = a['entropy_sum'] / n
        std_ent = math.sqrt(a['entropy_sq_sum'] / n - mean_ent ** 2)
        mean_unique = a['unique_sum'] / n

        print(f"{'='*60}")
        print(f"BLOCK SIZE {bs}")
        print(f"{'='*60}")
        print(f"Total blocks: {n:,}")
        print(f"\nPer-block Shannon entropy (bits/element):")
        print(f"  Mean:   {mean_ent:.4f}")
        print(f"  Std:    {std_ent:.4f}")
        print(f"  Min:    {a['entropy_min']:.4f}")
        print(f"  Max:    {a['entropy_max']:.4f}")
        print(f"  vs global marginal: 3.857 bits")
        print(f"  Reduction: {(1 - mean_ent / 3.857) * 100:.1f}%")

        print(f"\nUnique codes per block:")
        print(f"  Mean: {mean_unique:.2f}")
        print(f"  Distribution:")
        uh = a['unique_hist']
        for k in range(1, N_CODES + 1):
            if uh[k] > 0:
                pct = 100 * uh[k] / n
                bits_needed = math.ceil(math.log2(max(k, 2)))
                print(f"    {k:2d} unique codes: {uh[k]:>12,} blocks ({pct:5.1f}%) → {bits_needed} bits/elem with fixed codebook")

        print(f"\nEntropy histogram (bits/elem → fraction of blocks):")
        bins = np.array(a['entropy_hist'])
        bin_edges = np.linspace(0, 4, 101)
        for i in range(0, 100, 5):
            chunk = bins[i:i+5].sum()
            if chunk > 0:
                lo = bin_edges[i]
                hi = bin_edges[i+5]
                pct = 100 * chunk / n
                bar = '#' * int(pct)
                print(f"    [{lo:.1f}-{hi:.1f}): {pct:5.1f}% {bar}")

        results[f'block_{bs}'] = {
            'n_blocks': n,
            'mean_entropy': mean_ent,
            'std_entropy': std_ent,
            'min_entropy': a['entropy_min'],
            'max_entropy': a['entropy_max'],
            'mean_unique_codes': mean_unique,
            'entropy_histogram': a['entropy_hist'].tolist(),
            'entropy_bin_edges': np.linspace(0, 4, 101).tolist(),
            'unique_code_histogram': a['unique_hist'].tolist(),
        }

    out_path = "/code/tensorrt_llm/scripts/nvfp4_compress/per_block_entropy.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()

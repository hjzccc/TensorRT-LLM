#!/usr/bin/env python3
"""Full block entropy: ALL layers, ALL experts, block sizes 16 and 32."""
import sys
import time
import math
import json
from collections import Counter
from pathlib import Path

import numpy as np
from safetensors import safe_open

BLOCK_16 = 16
FP4_UNIQUE = np.array([-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
BOUNDARIES = np.array([-5, -3.5, -2.5, -1.75, -1.25, -0.75, -0.25, 0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=np.float32)

MODEL_DIR = "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots"


def quantize_block(block):
    amax = np.abs(block).max()
    if amax == 0:
        return np.zeros(len(block), dtype=np.int8)
    scale = np.float16(amax / 6.0).astype(np.float32)
    if scale == 0:
        scale = 1.0
    norm = block / scale
    return np.clip(np.searchsorted(BOUNDARIES, norm), 0, 14).astype(np.int8)


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

    print(f"Processing ALL experts across ALL layers", flush=True)
    t0 = time.time()

    elem_counts = Counter()
    block16_counts = Counter()
    block32_counts = Counter()
    total_elements = 0
    total_blocks_16 = 0
    total_blocks_32 = 0

    for fi, fpath in enumerate(files):
        fname = fpath.name
        print(f"[{fi+1}/{len(files)}] {fname}", flush=True)
        with safe_open(str(fpath), framework='pt', device='cpu') as f:
            for key in f.keys():
                if 'experts' not in key:
                    continue
                if 'gate_up_proj' not in key and 'down_proj' not in key:
                    continue
                if 'mtp.' in key or 'shared_expert' in key:
                    continue

                packed = f.get_tensor(key).float().numpy()
                n_exp = packed.shape[0]
                layer = key.split('.')[3]
                proj = 'w1' if 'gate_up' in key else 'w2'
                print(f"  L{layer} {proj} [{n_exp}x{packed.shape[1]}x{packed.shape[2]}]", end='', flush=True)

                batch_t = time.time()
                for ei in range(n_exp):
                    weight = packed[ei]
                    rows, cols = weight.shape
                    pad32 = (32 - cols % 32) % 32
                    if pad32 > 0:
                        weight = np.pad(weight, ((0, 0), (0, pad32)))

                    blocks_16 = weight.reshape(-1, BLOCK_16)

                    for bi in range(blocks_16.shape[0]):
                        codes = quantize_block(blocks_16[bi])

                        for c in codes:
                            elem_counts[int(c)] += 1
                        total_elements += BLOCK_16

                        block16_counts[tuple(codes.tolist())] += 1
                        total_blocks_16 += 1

                    blocks_32 = weight.reshape(-1, 32)
                    for bi in range(blocks_32.shape[0]):
                        top = quantize_block(blocks_32[bi][:16])
                        bot = quantize_block(blocks_32[bi][16:])
                        combined = tuple(np.concatenate([top, bot]).tolist())
                        block32_counts[combined] += 1
                        total_blocks_32 += 1

                dt = time.time() - batch_t
                elapsed = time.time() - t0
                print(f"  {dt:.0f}s (total {elapsed:.0f}s, {total_elements/1e9:.2f}B elem)", flush=True)

    elapsed = time.time() - t0

    H_elem = shannon_entropy(elem_counts, total_elements)
    H_block16 = shannon_entropy(block16_counts, total_blocks_16)
    H_block32 = shannon_entropy(block32_counts, total_blocks_32)

    unique_16 = len(block16_counts)
    unique_32 = len(block32_counts)
    repeated_16 = sum(1 for c in block16_counts.values() if c > 1)
    repeated_32 = sum(1 for c in block32_counts.values() if c > 1)

    print(f"\n{'='*70}")
    print(f"FULL ENTROPY ANALYSIS — ALL LAYERS, ALL EXPERTS")
    print(f"{'='*70}")
    print(f"Total elements:    {total_elements:,}")
    print(f"Total blocks (16): {total_blocks_16:,}")
    print(f"Total blocks (32): {total_blocks_32:,}")
    print(f"Time: {elapsed:.0f}s")

    print(f"\n--- Element-level ---")
    print(f"Marginal entropy: {H_elem:.4f} bits (max 3.907 for 15 symbols)")

    print(f"\n--- Block-16 ---")
    print(f"Unique patterns:  {unique_16:,} out of {total_blocks_16:,} blocks")
    print(f"Repeated patterns: {repeated_16:,} ({100*repeated_16/unique_16:.1f}% of unique)")
    print(f"Unique/Total ratio: {unique_16/total_blocks_16:.6f}")
    print(f"Joint entropy:    {H_block16:.4f} bits per block")
    print(f"Per-element:      {H_block16/16:.4f} bits")
    print(f"If independent:   {H_elem*16:.4f} bits per block")
    print(f"Mutual info:      {H_elem*16 - H_block16:.4f} bits per block ({(H_elem*16 - H_block16)/16:.4f}/elem)")

    print(f"\n--- Block-32 (two consecutive NVFP4 blocks) ---")
    print(f"Unique patterns:  {unique_32:,} out of {total_blocks_32:,} blocks")
    print(f"Repeated patterns: {repeated_32:,} ({100*repeated_32/unique_32:.1f}% of unique)")
    print(f"Unique/Total ratio: {unique_32/total_blocks_32:.6f}")
    print(f"Joint entropy:    {H_block32:.4f} bits per block")
    print(f"Per-element:      {H_block32/32:.4f} bits")
    print(f"If independent:   {H_elem*32:.4f} bits per block")
    print(f"Mutual info:      {H_elem*32 - H_block32:.4f} bits per block ({(H_elem*32 - H_block32)/32:.4f}/elem)")

    print(f"\n--- Top 10 most common block-16 patterns ---")
    for pattern, count in block16_counts.most_common(10):
        fp4_vals = [FP4_UNIQUE[i] for i in pattern]
        print(f"  count={count:>6d}  codes={list(pattern)}  values={fp4_vals}")

    results = {
        'total_elements': total_elements,
        'total_blocks_16': total_blocks_16,
        'total_blocks_32': total_blocks_32,
        'elem_entropy': H_elem,
        'block16_joint_entropy': H_block16,
        'block16_per_element': H_block16 / 16,
        'block16_unique': unique_16,
        'block16_repeated': repeated_16,
        'block32_joint_entropy': H_block32,
        'block32_per_element': H_block32 / 32,
        'block32_unique': unique_32,
        'block32_repeated': repeated_32,
    }

    out_path = "/code/tensorrt_llm/scripts/channel_quant_new/profiling/block_entropy_full.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()

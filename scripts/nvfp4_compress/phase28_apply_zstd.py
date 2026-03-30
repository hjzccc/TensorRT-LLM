#!/usr/bin/env python3
"""
Phase 28: Apply lossless zstd compression to compressed NVFP4 checkpoint.

Compresses weight_indices, weight_codebook_entries, and weight_scale tensors
using zstd, achieving ~20% additional size reduction with zero accuracy impact.

Usage:
    python phase28_apply_zstd.py [--input DIR] [--output DIR] [--level N]
"""

import os, sys, json, time, shutil, argparse
import torch
import numpy as np
import zstandard as zstd
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

COMPRESS_SUFFIXES = ('.weight_indices', '.weight_codebook_entries', '.weight_scale')
ZSTD_LEVEL = 3


def compress_shard(input_path: str, output_path: str, level: int = ZSTD_LEVEL) -> dict:
    """Compress a single shard. Returns size stats."""
    cctx = zstd.ZstdCompressor(level=level)
    
    tensors_out = {}
    orig_bytes = 0
    comp_bytes = 0
    n_compressed = 0
    n_passthrough = 0
    
    with safe_open(input_path, framework='pt') as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            
            should_compress = (
                tensor.dtype == torch.uint8 and
                any(key.endswith(s) for s in COMPRESS_SUFFIXES)
            )
            
            if should_compress:
                raw = tensor.numpy().tobytes()
                compressed = cctx.compress(raw)
                
                orig_bytes += len(raw)
                comp_bytes += len(compressed)
                n_compressed += 1
                
                # Store compressed data as uint8 tensor
                comp_tensor = torch.frombuffer(bytearray(compressed), dtype=torch.uint8)
                tensors_out[key + '.__zstd__'] = comp_tensor
                # Store metadata: [is_uint8, *shape]
                tensors_out[key + '.__meta__'] = torch.tensor(
                    [1] + list(tensor.shape), dtype=torch.int64
                )
            else:
                tensors_out[key] = tensor
                n_passthrough += 1
    
    save_file(tensors_out, output_path)
    return {'orig_bytes': orig_bytes, 'comp_bytes': comp_bytes,
            'n_compressed': n_compressed, 'n_passthrough': n_passthrough}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='scripts/nvfp4_compress/compressed_2b075b_zero_fixed_weighted_abs')
    parser.add_argument('--output', default='scripts/nvfp4_compress/compressed_2b075b_zstd3')
    parser.add_argument('--level', type=int, default=ZSTD_LEVEL)
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    shards = sorted(input_dir.glob('model-*.safetensors'))
    print(f'Phase 28: Compressing {len(shards)} shards with zstd-{args.level}')
    print(f'  Input:  {input_dir}')
    print(f'  Output: {output_dir}')
    print()
    
    total = {'orig_bytes': 0, 'comp_bytes': 0, 'n_compressed': 0, 'n_passthrough': 0}
    t0 = time.time()
    
    for i, shard in enumerate(shards):
        size = shard.stat().st_size
        out_shard = output_dir / shard.name
        
        if size < 1000:
            shutil.copy2(shard, out_shard)
            continue
        
        stats = compress_shard(str(shard), str(out_shard), args.level)
        for k in total:
            total[k] += stats[k]
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(shards) - i - 1) / rate
            savings = (1 - total['comp_bytes'] / max(total['orig_bytes'], 1)) * 100
            print(f'  [{i+1}/{len(shards)}] {elapsed:.0f}s, ETA {eta:.0f}s, savings so far: {savings:.1f}%')
    
    # Copy non-shard files
    for f in input_dir.iterdir():
        if not (f.name.startswith('model-') and f.name.endswith('.safetensors')):
            dest = output_dir / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
    
    # Update manifest
    manifest_path = output_dir / 'compression_manifest.json'
    if manifest_path.exists():
        with open(manifest_path) as mf:
            manifest = json.load(mf)
        manifest['zstd_compressed'] = True
        manifest['zstd_level'] = args.level
        with open(manifest_path, 'w') as mf:
            json.dump(manifest, mf, indent=2)
    
    elapsed = time.time() - t0
    savings_pct = (1 - total['comp_bytes'] / max(total['orig_bytes'], 1)) * 100
    
    # Compute actual output size
    out_size = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file())
    in_size = sum(f.stat().st_size for f in input_dir.rglob('*') if f.is_file())
    
    results = {
        'phase': 28,
        'method': 'lossless_zstd_compression',
        'zstd_level': args.level,
        'elapsed_seconds': elapsed,
        'compressible_orig_gb': total['orig_bytes'] / 1024**3,
        'compressible_comp_gb': total['comp_bytes'] / 1024**3,
        'compressible_savings_pct': savings_pct,
        'input_total_gb': in_size / 1024**3,
        'output_total_gb': out_size / 1024**3,
        'overall_savings_gb': (in_size - out_size) / 1024**3,
        'overall_savings_pct': (1 - out_size / in_size) * 100,
        'compression_ratio_vs_nvfp4': 21.28 / (out_size / 1024**3),
        'n_tensors_compressed': total['n_compressed'],
        'n_tensors_passthrough': total['n_passthrough'],
    }
    
    print()
    print('=' * 60)
    print('PHASE 28 RESULTS:')
    print(f'  Compressible data: {results["compressible_orig_gb"]:.2f} GB -> {results["compressible_comp_gb"]:.2f} GB ({savings_pct:.1f}% savings)')
    print(f'  Total checkpoint: {results["input_total_gb"]:.2f} GB -> {results["output_total_gb"]:.2f} GB')
    print(f'  Overall savings: {results["overall_savings_gb"]:.2f} GB ({results["overall_savings_pct"]:.1f}%)')
    print(f'  Compression ratio vs NVFP4: {results["compression_ratio_vs_nvfp4"]:.2f}x')
    print(f'  Time: {elapsed:.0f}s')
    print()
    
    results_path = 'scripts/nvfp4_compress/phase28_zstd_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to: {results_path}')
    
    return results


if __name__ == '__main__':
    main()

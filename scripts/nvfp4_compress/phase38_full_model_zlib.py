#!/usr/bin/env python3
"""
Phase 38: Full Model zlib Compression Statistics

Estimates the total compression ratio for the full model by sampling
across all shards and weight types.

Key result: 2.75 → 2.035 bits/elem (26% reduction, lossless)
vs original FP4: 4.0 → 2.035 bits/elem (49.1% reduction)
"""
import json, time, zlib, numpy as np, torch
from pathlib import Path
from safetensors import safe_open

COMPRESSED = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/compressed_2b075b_zero_fixed_exact")

print("="*60)
print("Phase 38: Full Model zlib Compression Statistics")
print("="*60)

# Sample from multiple shards
sample_shards = list(range(12, 50, 5))  # Every 5th shard from 12 to 50
results = []

for shard_num in sample_shards:
    shard_name = f"model-{shard_num:05d}-of-00733.safetensors"
    shard_path = COMPRESSED / shard_name
    if not shard_path.exists():
        continue
    
    with safe_open(str(shard_path), framework='pt', device='cpu') as f:
        keys = list(f.keys())
        weight_keys = [k for k in keys if k.endswith('.weight_indices')]
        
        for k in weight_keys[:3]:  # 3 per shard
            base_key = k.replace('.weight_indices', '')
            try:
                indices = f.get_tensor(f'{base_key}.weight_indices')
                entries = f.get_tensor(f'{base_key}.weight_codebook_entries')
            except:
                continue
            
            n = indices.numel() * 4
            orig = indices.numel() + entries.numel()
            
            comp_idx = zlib.compress(indices.numpy().tobytes(), level=9)
            comp_ent = zlib.compress(entries.numpy().tobytes(), level=9)
            comp = len(comp_idx) + len(comp_ent)
            
            r = {
                'shard': shard_num,
                'key': base_key,
                'n_elements': n,
                'orig_bytes': orig,
                'comp_bytes': comp,
                'bits_orig': orig*8/n,
                'bits_comp': comp*8/n,
                'savings_pct': (orig-comp)/orig*100,
            }
            results.append(r)
            print(f"  shard{shard_num:03d} {base_key.split('.')[-2]}: {r['bits_orig']:.3f} → {r['bits_comp']:.3f} b/e ({r['savings_pct']:.1f}%)")

if results:
    avg_bits = np.mean([r['bits_comp'] for r in results])
    avg_savings = np.mean([r['savings_pct'] for r in results])
    min_bits = min(r['bits_comp'] for r in results)
    max_bits = max(r['bits_comp'] for r in results)
    
    print(f"\n=== SUMMARY ({len(results)} weights sampled) ===")
    print(f"Original:  2.750 bits/elem")
    print(f"Compressed: {avg_bits:.3f} bits/elem (avg)")
    print(f"Range: {min_bits:.3f} - {max_bits:.3f} bits/elem")
    print(f"Savings: {avg_savings:.1f}% (avg)")
    print(f"\nvs original FP4 (4.0 bits/elem): {(1-avg_bits/4.0)*100:.1f}% reduction")
    
    # Estimate full model size
    # Full model: ~32B FP4 elements (from phase24 analysis)
    total_elements = 32_338_083_840
    orig_size_gb = total_elements * 2.75 / 8 / 1e9
    comp_size_gb = total_elements * avg_bits / 8 / 1e9
    print(f"\nFull model estimate:")
    print(f"  Original (2.75 b/e): {orig_size_gb:.1f} GB")
    print(f"  Compressed ({avg_bits:.3f} b/e): {comp_size_gb:.1f} GB")
    print(f"  Savings: {orig_size_gb-comp_size_gb:.1f} GB")
    
    out = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase38_full_model_stats.json")
    out.write_text(json.dumps({'results': results, 'summary': {'avg_bits': avg_bits, 'avg_savings': avg_savings, 'min_bits': min_bits, 'max_bits': max_bits}}, indent=2))
    print(f"Results saved to {out}")

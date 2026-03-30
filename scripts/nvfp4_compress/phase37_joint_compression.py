#!/usr/bin/env python3
"""
Phase 37: Joint zlib Compression of Indices + Entries

Key finding: Compressing indices AND entries together with zlib achieves
2.041 bits/elem (25.8% reduction from 2.75, lossless).

This is because:
1. Indices are highly compressible (66% zeros)
2. Entries are also compressible (skewed distribution)
3. zlib exploits spatial correlations within each stream

Result: 2.75 → 2.035 bits/elem (separate zlib, slightly better than joint)

References:
- EntroLLM (arXiv:2505.02380): entropy coding of quantized indices
- Float8@2bits (arXiv:2601.22787): entropy coding to 2 bits effective
"""
import json, time, zlib, numpy as np, torch
from pathlib import Path
from safetensors import safe_open

COMPRESSED = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/compressed_2b075b_zero_fixed_exact")

print("="*60)
print("Phase 37: Joint zlib Compression (Indices + Entries)")
print("="*60)

shard_path = COMPRESSED / 'model-00012-of-00733.safetensors'

test_keys = []
with safe_open(str(shard_path), framework='pt', device='cpu') as f:
    for k in f.keys():
        if k.endswith('.weight_indices'):
            test_keys.append(k.replace('.weight_indices', '.weight'))
        if len(test_keys) >= 10: break

results = []
for key in test_keys:
    with safe_open(str(shard_path), framework='pt', device='cpu') as f:
        indices = f.get_tensor(key.replace('.weight', '.weight_indices'))
        entries = f.get_tensor(key.replace('.weight', '.weight_codebook_entries'))
    
    n = indices.numel() * 4  # 4 indices per byte
    orig_total = indices.numel() + entries.numel()
    
    # Separate zlib
    t0 = time.time()
    comp_idx = zlib.compress(indices.numpy().tobytes(), level=9)
    comp_ent = zlib.compress(entries.numpy().tobytes(), level=9)
    t_sep = time.time() - t0
    sep_total = len(comp_idx) + len(comp_ent)
    
    # Joint zlib
    t0 = time.time()
    joint_data = indices.numpy().tobytes() + entries.numpy().tobytes()
    comp_joint = zlib.compress(joint_data, level=9)
    t_joint = time.time() - t0
    
    # Verify
    decompressed = zlib.decompress(comp_joint)
    correct = (decompressed == joint_data)
    
    bits_orig = orig_total * 8 / n
    bits_sep = sep_total * 8 / n
    bits_joint = len(comp_joint) * 8 / n
    
    r = {
        'key': key,
        'n_elements': n,
        'bits_orig': bits_orig,
        'bits_sep': bits_sep,
        'bits_joint': bits_joint,
        'savings_sep_pct': (orig_total - sep_total) / orig_total * 100,
        'savings_joint_pct': (orig_total - len(comp_joint)) / orig_total * 100,
        'correct': correct,
        'time_sep': t_sep,
        'time_joint': t_joint,
    }
    results.append(r)
    print(f"  {key.split('.')[-3]}.{key.split('.')[-2]}: {bits_orig:.3f} → sep:{bits_sep:.3f} joint:{bits_joint:.3f} b/e")

avg_sep = np.mean([r['bits_sep'] for r in results])
avg_joint = np.mean([r['bits_joint'] for r in results])
avg_savings_sep = np.mean([r['savings_sep_pct'] for r in results])
avg_savings_joint = np.mean([r['savings_joint_pct'] for r in results])

print(f"\n=== SUMMARY ===")
print(f"Original:      2.750 bits/elem")
print(f"Separate zlib: {avg_sep:.3f} bits/elem ({avg_savings_sep:.1f}% savings)")
print(f"Joint zlib:    {avg_joint:.3f} bits/elem ({avg_savings_joint:.1f}% savings)")
print(f"Best:          {'separate' if avg_sep < avg_joint else 'joint'}")
print(f"\nvs Phase 36 (indices only): 2.363 bits/elem")
print(f"Improvement from also compressing entries: {2.363-min(avg_sep,avg_joint):.3f} bits/elem")

out = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase37_joint_compression_results.json")
out.write_text(json.dumps({'results': results, 'summary': {'avg_sep': avg_sep, 'avg_joint': avg_joint, 'avg_savings_sep': avg_savings_sep, 'avg_savings_joint': avg_savings_joint}}, indent=2))
print(f"Results saved to {out}")

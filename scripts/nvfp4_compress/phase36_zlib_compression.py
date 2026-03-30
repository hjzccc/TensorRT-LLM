#!/usr/bin/env python3
"""
Phase 36: zlib Compression of Indices (Best Entropy Coding)

zlib achieves 2.224 bits/elem (vs Huffman 2.302, entropy limit 2.208).
This is because zlib exploits both:
1. Symbol frequency skew (Huffman component)
2. Spatial correlations between adjacent indices (LZ77 component)

Result: 2.75 → 2.224 bits/elem (19.1% reduction, lossless)

References:
- EntroLLM (arXiv:2505.02380): entropy coding of quantized indices
- zlib: Deutsch & Gailly 1996 (LZ77 + Huffman)
"""
import json, time, zlib, numpy as np, torch
from pathlib import Path
from safetensors import safe_open

COMPRESSED = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/compressed_2b075b_zero_fixed_exact")

def decode_indices(indices_tensor):
    flat = indices_tensor.flatten().numpy().astype(np.uint8)
    idx2 = np.empty(len(flat)*4, dtype=np.uint8)
    idx2[0::4] = flat & 0x03; idx2[1::4] = (flat >> 2) & 0x03
    idx2[2::4] = (flat >> 4) & 0x03; idx2[3::4] = (flat >> 6) & 0x03
    return idx2

print("="*60)
print("Phase 36: zlib Compression of Indices")
print("="*60)

shard_path = COMPRESSED / 'model-00012-of-00733.safetensors'

test_keys = []
with safe_open(str(shard_path), framework='pt', device='cpu') as f:
    for k in f.keys():
        if k.endswith('.weight_indices'):
            test_keys.append(k.replace('.weight_indices', '.weight'))
        if len(test_keys) >= 8: break

results = []
for key in test_keys:
    with safe_open(str(shard_path), framework='pt', device='cpu') as f:
        indices = f.get_tensor(key.replace('.weight', '.weight_indices'))
        entries = f.get_tensor(key.replace('.weight', '.weight_codebook_entries'))
    
    idx2 = decode_indices(indices)
    n = len(idx2)
    
    # Compress with zlib
    t0 = time.time()
    compressed = zlib.compress(indices.numpy().tobytes(), level=9)
    t_compress = time.time() - t0
    
    # Decompress (verify)
    t0 = time.time()
    decompressed = zlib.decompress(compressed)
    t_decompress = time.time() - t0
    correct = (decompressed == indices.numpy().tobytes())
    
    orig_idx = indices.numel()
    comp_idx = len(compressed)
    entries_bytes = entries.numel()
    
    orig_total = orig_idx + entries_bytes
    comp_total = comp_idx + entries_bytes
    
    bits_orig = orig_total * 8 / n
    bits_comp = comp_total * 8 / n
    
    r = {
        'key': key,
        'n_elements': n,
        'orig_total': orig_total,
        'comp_total': comp_total,
        'bits_orig': bits_orig,
        'bits_comp': bits_comp,
        'savings_pct': (orig_total - comp_total) / orig_total * 100,
        'compress_time': t_compress,
        'decompress_time': t_decompress,
        'correct': correct,
    }
    results.append(r)
    print(f"  {key.split('.')[-3]}.{key.split('.')[-2]}: {bits_orig:.3f} → {bits_comp:.3f} b/e ({r['savings_pct']:.1f}%) [{t_compress:.3f}s/{t_decompress:.3f}s]")

avg_bits = np.mean([r['bits_comp'] for r in results])
avg_savings = np.mean([r['savings_pct'] for r in results])
print(f"\n=== SUMMARY ===")
print(f"Average: {avg_bits:.3f} bits/elem ({avg_savings:.1f}% savings)")
print(f"Improvement: 2.75 → {avg_bits:.3f} bits/elem")
print(f"vs Huffman (Phase 32): 2.375 bits/elem")
print(f"vs Entropy limit: 2.208 bits/elem")
print(f"zlib is {'better' if avg_bits < 2.375 else 'worse'} than Huffman")

out = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase36_zlib_results.json")
out.write_text(json.dumps({'results': results, 'summary': {'avg_bits': avg_bits, 'avg_savings': avg_savings}}, indent=2))
print(f"Results saved to {out}")

#!/usr/bin/env python3
"""
Phase 35: RLE + Entropy Coding of Indices

Key finding: RLE achieves 1.357 bits/elem for indices (better than Huffman 1.458).
Combined with codebook entries (0.75 bits/elem): 2.107 bits/elem total.
This is a 23.4% reduction from current 2.75 bits/elem, with ZERO MSE cost.

RLE scheme:
- Encode (zero_run_length, non_zero_index) pairs
- zero_run_length: Elias gamma coding (variable length)
- non_zero_index: 2 bits (values 1, 2, 3)

References:
- EntroLLM (arXiv:2505.02380): entropy coding of quantized indices
- Float8@2bits (arXiv:2601.22787): entropy coding to 2 bits effective
"""
import json, time, numpy as np, torch
from pathlib import Path
from safetensors import safe_open
from collections import Counter

COMPRESSED = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/compressed_2b075b_zero_fixed_exact")

def decode_indices(indices_tensor):
    """Decode 2-bit packed indices."""
    flat = indices_tensor.flatten().numpy().astype(np.uint8)
    idx2 = np.empty(len(flat)*4, dtype=np.uint8)
    idx2[0::4] = flat & 0x03; idx2[1::4] = (flat >> 2) & 0x03
    idx2[2::4] = (flat >> 4) & 0x03; idx2[3::4] = (flat >> 6) & 0x03
    return idx2

def elias_gamma_encode(n):
    """Encode positive integer n using Elias gamma coding."""
    if n == 0: return '0'
    k = int(np.floor(np.log2(n)))
    return '0' * k + bin(n)[2:]  # k zeros + binary representation

def elias_gamma_decode(bits, pos):
    """Decode Elias gamma from bits starting at pos. Returns (value, new_pos)."""
    k = 0
    while pos < len(bits) and bits[pos] == '0':
        k += 1; pos += 1
    if pos + k > len(bits): return None, pos
    val = int(bits[pos:pos+k+1], 2)
    return val, pos + k + 1

def rle_encode(indices):
    """RLE encode index stream. Returns bit string."""
    bits = []
    i = 0
    n = len(indices)
    
    while i < n:
        # Count zero run
        zero_run = 0
        while i < n and indices[i] == 0:
            zero_run += 1; i += 1
        
        # Encode zero run (Elias gamma of zero_run+1 to handle 0)
        bits.append(elias_gamma_encode(zero_run + 1))
        
        if i < n:
            # Encode non-zero value (2 bits: values 1,2,3)
            bits.append(f'{indices[i]:02b}')
            i += 1
    
    return ''.join(bits)

def rle_decode(bits, n):
    """Decode RLE bit string to n indices."""
    result = []
    pos = 0
    
    while len(result) < n and pos < len(bits):
        # Decode zero run
        zero_run_plus1, pos = elias_gamma_decode(bits, pos)
        if zero_run_plus1 is None: break
        zero_run = zero_run_plus1 - 1
        result.extend([0] * zero_run)
        
        if len(result) >= n: break
        
        # Decode non-zero value
        if pos + 2 <= len(bits):
            val = int(bits[pos:pos+2], 2)
            result.append(val)
            pos += 2
    
    return result[:n]

def pack_bits(bit_string):
    pad = (8 - len(bit_string) % 8) % 8
    bit_string += '0' * pad
    return bytes(int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8)), pad

def unpack_bits(data, pad):
    bits = ''.join(f'{b:08b}' for b in data)
    return bits[:len(bits)-pad] if pad else bits

print("="*60)
print("Phase 35: RLE + Entropy Coding of Indices")
print("="*60)

shard_path = COMPRESSED / 'model-00012-of-00733.safetensors'

# Test on multiple weights
test_keys = []
with safe_open(str(shard_path), framework='pt', device='cpu') as f:
    for k in f.keys():
        if k.endswith('.weight_indices'):
            test_keys.append(k.replace('.weight_indices', '.weight'))
        if len(test_keys) >= 5: break

results = []
for key in test_keys:
    with safe_open(str(shard_path), framework='pt', device='cpu') as f:
        indices = f.get_tensor(key.replace('.weight', '.weight_indices'))
        entries = f.get_tensor(key.replace('.weight', '.weight_codebook_entries'))
    
    idx2 = decode_indices(indices)
    n = len(idx2)
    
    # RLE encode
    t0 = time.time()
    encoded = rle_encode(idx2)
    packed, pad = pack_bits(encoded)
    t_encode = time.time() - t0
    
    # RLE decode (verify)
    t0 = time.time()
    decoded_bits = unpack_bits(packed, pad)
    decoded = rle_decode(decoded_bits, n)
    t_decode = time.time() - t0
    
    correct = (decoded == idx2.tolist())
    
    # Sizes
    orig_idx_bytes = indices.numel()
    rle_bytes = len(packed)
    entries_bytes = entries.numel()
    
    orig_total = orig_idx_bytes + entries_bytes
    rle_total = rle_bytes + entries_bytes
    
    bits_orig = orig_total * 8 / n
    bits_rle = rle_total * 8 / n
    
    r = {
        'key': key,
        'n_elements': n,
        'orig_idx_bytes': orig_idx_bytes,
        'rle_bytes': rle_bytes,
        'entries_bytes': entries_bytes,
        'bits_orig': bits_orig,
        'bits_rle': bits_rle,
        'savings_pct': (orig_total - rle_total) / orig_total * 100,
        'encode_time': t_encode,
        'decode_time': t_decode,
        'correct': correct,
    }
    results.append(r)
    print(f"\n{key.split('.')[-3]}.{key.split('.')[-2]}:")
    print(f"  Original: {orig_total:,} bytes = {bits_orig:.3f} bits/elem")
    print(f"  RLE:      {rle_total:,} bytes = {bits_rle:.3f} bits/elem")
    print(f"  Savings:  {r['savings_pct']:.1f}%")
    print(f"  Encode: {t_encode:.3f}s, Decode: {t_decode:.3f}s, Correct: {correct}")

avg_bits = np.mean([r['bits_rle'] for r in results])
avg_savings = np.mean([r['savings_pct'] for r in results])
print(f"\n=== SUMMARY ===")
print(f"Average: {avg_bits:.3f} bits/elem ({avg_savings:.1f}% savings)")
print(f"Improvement: 2.75 → {avg_bits:.3f} bits/elem")
print(f"vs Huffman: 2.375 bits/elem")
print(f"RLE is {'better' if avg_bits < 2.375 else 'worse'} than Huffman")

out = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase35_rle_results.json")
out.write_text(json.dumps({'results': results, 'summary': {'avg_bits': avg_bits, 'avg_savings': avg_savings}}, indent=2))
print(f"\nResults saved to {out}")

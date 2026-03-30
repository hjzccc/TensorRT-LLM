#!/usr/bin/env python3
"""
Phase 32: Entropy Coding of 2-bit Indices

Key finding: Index distribution is highly skewed (66% index 0),
giving entropy of 1.458 bits/symbol vs 2.0 bits stored.
Entropy coding saves 0.54 bits/elem → 2.75 → 2.21 bits/elem (19.7% reduction).

This is lossless compression of the index stream - zero MSE cost.

References:
- EntroLLM (arXiv:2505.02380): entropy coding of LLM quantization indices
- Float8@2bits (arXiv:2601.22787): entropy coding to 2 bits effective
- ANS (Asymmetric Numeral Systems): Duda 2009, used in zstd/brotli

Implementation: Huffman coding (simple, fast, near-optimal for 4 symbols)
"""
import json, time, numpy as np, torch
from pathlib import Path
from collections import Counter
import heapq
from safetensors import safe_open

COMPRESSED = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/compressed_2b075b_zero_fixed_exact")

# ─── Huffman coding ───────────────────────────────────────────────────────────

class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol; self.freq = freq
        self.left = self.right = None
    def __lt__(self, other): return self.freq < other.freq

def build_huffman(freqs):
    """Build Huffman tree from frequency dict."""
    heap = [HuffmanNode(s, f) for s, f in freqs.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        l = heapq.heappop(heap); r = heapq.heappop(heap)
        n = HuffmanNode(None, l.freq + r.freq)
        n.left = l; n.right = r
        heapq.heappush(heap, n)
    return heap[0]

def get_codes(node, prefix="", codes=None):
    if codes is None: codes = {}
    if node.symbol is not None:
        codes[node.symbol] = prefix or "0"
    else:
        get_codes(node.left, prefix+"0", codes)
        get_codes(node.right, prefix+"1", codes)
    return codes

def huffman_encode(symbols, codes):
    """Encode symbols using Huffman codes. Returns bit string."""
    return "".join(codes[s] for s in symbols)

def huffman_decode(bits, root, n):
    """Decode n symbols from bit string."""
    result = []; node = root
    for b in bits:
        node = node.left if b == '0' else node.right
        if node.symbol is not None:
            result.append(node.symbol)
            node = root
            if len(result) == n: break
    return result

def pack_bits(bit_string):
    """Pack bit string into bytes."""
    # Pad to multiple of 8
    pad = (8 - len(bit_string) % 8) % 8
    bit_string += '0' * pad
    return bytes(int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8)), pad

def unpack_bits(data, pad):
    """Unpack bytes to bit string."""
    bits = ''.join(f'{b:08b}' for b in data)
    return bits[:len(bits)-pad] if pad else bits

# ─── Analysis ─────────────────────────────────────────────────────────────────

def analyze_weight(key, shard_path):
    """Analyze entropy of indices for one weight."""
    with safe_open(str(shard_path), framework='pt', device='cpu') as f:
        indices = f.get_tensor(key.replace('.weight', '.weight_indices'))
    
    flat = indices.flatten().numpy().astype(np.uint8)
    idx2 = np.empty(len(flat)*4, dtype=np.uint8)
    idx2[0::4] = flat & 0x03; idx2[1::4] = (flat >> 2) & 0x03
    idx2[2::4] = (flat >> 4) & 0x03; idx2[3::4] = (flat >> 6) & 0x03
    
    counts = np.bincount(idx2, minlength=4)
    total = len(idx2)
    probs = counts / total
    entropy = -np.sum(probs[probs>0] * np.log2(probs[probs>0]))
    
    return {
        'n_elements': total,
        'counts': counts.tolist(),
        'probs': probs.tolist(),
        'entropy': float(entropy),
        'current_bits': 2.0,
        'savings': float(2.0 - entropy),
    }

def test_huffman_on_weight(key, shard_path):
    """Test Huffman coding on one weight's indices."""
    with safe_open(str(shard_path), framework='pt', device='cpu') as f:
        indices = f.get_tensor(key.replace('.weight', '.weight_indices'))
        entries = f.get_tensor(key.replace('.weight', '.weight_codebook_entries'))
    
    flat = indices.flatten().numpy().astype(np.uint8)
    idx2 = np.empty(len(flat)*4, dtype=np.uint8)
    idx2[0::4] = flat & 0x03; idx2[1::4] = (flat >> 2) & 0x03
    idx2[2::4] = (flat >> 4) & 0x03; idx2[3::4] = (flat >> 6) & 0x03
    
    n = len(idx2)
    counts = Counter(idx2.tolist())
    
    # Build Huffman tree
    tree = build_huffman(counts)
    codes = get_codes(tree)
    
    # Encode
    t0 = time.time()
    encoded = huffman_encode(idx2.tolist(), codes)
    packed, pad = pack_bits(encoded)
    t_encode = time.time() - t0
    
    # Decode (verify)
    t0 = time.time()
    decoded_bits = unpack_bits(packed, pad)
    decoded = huffman_decode(decoded_bits, tree, n)
    t_decode = time.time() - t0
    
    # Verify
    correct = all(a == b for a, b in zip(idx2.tolist(), decoded))
    
    # Compute sizes
    orig_bytes = len(flat)  # 2 bits/elem packed
    huffman_bytes = len(packed)
    entries_bytes = entries.numel()
    
    orig_total = orig_bytes + entries_bytes
    huffman_total = huffman_bytes + entries_bytes
    
    bits_per_elem_orig = (orig_total * 8) / n
    bits_per_elem_huffman = (huffman_total * 8) / n
    
    return {
        'n_elements': n,
        'codes': {str(k): v for k, v in codes.items()},
        'orig_index_bytes': orig_bytes,
        'huffman_index_bytes': huffman_bytes,
        'entries_bytes': entries_bytes,
        'orig_total_bytes': orig_total,
        'huffman_total_bytes': huffman_total,
        'bits_per_elem_orig': bits_per_elem_orig,
        'bits_per_elem_huffman': bits_per_elem_huffman,
        'savings_bits_per_elem': bits_per_elem_orig - bits_per_elem_huffman,
        'savings_pct': (orig_total - huffman_total) / orig_total * 100,
        'encode_time': t_encode,
        'decode_time': t_decode,
        'correct': correct,
    }

# ─── Main ─────────────────────────────────────────────────────────────────────

print("="*60)
print("Phase 32: Entropy Coding of 2-bit Indices")
print("="*60)

shard_path = COMPRESSED / 'model-00012-of-00733.safetensors'
key = 'model.layers.0.mlp.experts.0.down_proj.weight'

print(f"\nAnalyzing: {key}")
analysis = analyze_weight(key, shard_path)
print(f"  Elements: {analysis['n_elements']:,}")
print(f"  Distribution: {[f'{p*100:.1f}%' for p in analysis['probs']]}")
print(f"  Entropy: {analysis['entropy']:.4f} bits/symbol")
print(f"  Savings: {analysis['savings']:.4f} bits/elem")

print(f"\nTesting Huffman coding...")
result = test_huffman_on_weight(key, shard_path)
print(f"  Huffman codes: {result['codes']}")
print(f"  Original: {result['orig_total_bytes']:,} bytes = {result['bits_per_elem_orig']:.3f} bits/elem")
print(f"  Huffman:  {result['huffman_total_bytes']:,} bytes = {result['bits_per_elem_huffman']:.3f} bits/elem")
print(f"  Savings:  {result['savings_bits_per_elem']:.3f} bits/elem ({result['savings_pct']:.1f}%)")
print(f"  Encode time: {result['encode_time']:.3f}s")
print(f"  Decode time: {result['decode_time']:.3f}s")
print(f"  Correct: {result['correct']}")

# Test on multiple weights
print(f"\nTesting on 5 weights...")
test_keys = []
with safe_open(str(shard_path), framework='pt', device='cpu') as f:
    for k in f.keys():
        if k.endswith('.weight_indices'):
            test_keys.append(k.replace('.weight_indices', '.weight'))
        if len(test_keys) >= 5: break

all_results = []
for k in test_keys:
    r = test_huffman_on_weight(k, shard_path)
    all_results.append(r)
    print(f"  {k.split('.')[-3]}.{k.split('.')[-2]}: {r['bits_per_elem_orig']:.3f} → {r['bits_per_elem_huffman']:.3f} bits/elem ({r['savings_pct']:.1f}% savings)")

avg_savings = np.mean([r['savings_pct'] for r in all_results])
avg_bits = np.mean([r['bits_per_elem_huffman'] for r in all_results])
print(f"\nAverage: {avg_bits:.3f} bits/elem ({avg_savings:.1f}% savings)")
print(f"Improvement: 2.75 → {avg_bits:.3f} bits/elem")

# Save results
results = {
    'phase': 32,
    'technique': 'huffman_entropy_coding_indices',
    'reference': 'EntroLLM (arXiv:2505.02380)',
    'analysis': analysis,
    'single_weight_test': result,
    'multi_weight_results': all_results,
    'summary': {
        'avg_bits_per_elem': avg_bits,
        'avg_savings_pct': avg_savings,
        'original_bits_per_elem': 2.75,
        'improvement': 2.75 - avg_bits,
    }
}
out = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase32_entropy_coding_results.json")
out.write_text(json.dumps(results, indent=2))
print(f"\nResults saved to {out}")

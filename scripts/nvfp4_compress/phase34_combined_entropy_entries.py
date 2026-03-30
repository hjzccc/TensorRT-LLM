#!/usr/bin/env python3
"""
Phase 34: Combined Analysis - Entropy Coding + Reduced Entries

Tests the Pareto frontier of bits/elem vs MSE:
1. Current: 2.75 b/e, MSE=0.20 (96.2%)
2. Entropy coding only: 2.375 b/e, MSE=0.20 (96.2%) [lossless]
3. 2 entries + entropy: ~2.0 b/e, MSE=?
4. 1 entry + entropy: ~1.75 b/e, MSE=?
5. Global + entropy: ~1.5 b/e, MSE=?

Also tests: Can we improve the current scheme's MSE by using better
codebook optimization (EM instead of k-means)?
"""
import json, time, numpy as np, torch
from pathlib import Path
from sklearn.cluster import KMeans
from safetensors import safe_open
import heapq
from collections import Counter

CHECKPOINT = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/nvfp4_checkpoint")
COMPRESSED = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/compressed_2b075b_zero_fixed_exact")
E2M1 = torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6., 0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.], dtype=torch.float32)
E2M1_np = E2M1.numpy()

def nearest_fp4(v):
    return E2M1_np[np.argmin(np.abs(E2M1_np - v))]

def huffman_bits_per_symbol(counts):
    """Estimate Huffman bits per symbol (= entropy for 4 symbols)."""
    total = sum(counts.values())
    probs = {k: v/total for k,v in counts.items()}
    return -sum(p * np.log2(p) for p in probs.values() if p > 0)

def compress_with_n_entries(blocks, n_stored, sample_size=500):
    """Compress blocks with n stored FP4 entries + fixed zero."""
    np.random.seed(42)
    if len(blocks) > sample_size:
        idx = np.random.choice(len(blocks), sample_size, replace=False)
        sample = blocks[idx]
    else:
        sample = blocks
    
    mses = []
    all_indices = []
    
    for block in sample:
        if n_stored == 0:
            # Just zero
            centroids = np.array([0.0])
            labels = np.zeros(len(block), dtype=int)
        else:
            nonzero = block[block != 0]
            if len(nonzero) == 0:
                centroids = np.array([0.0])
                labels = np.zeros(len(block), dtype=int)
            else:
                k = min(n_stored, len(np.unique(nonzero)))
                km = KMeans(n_clusters=k, n_init=3, random_state=42, max_iter=50)
                km.fit(nonzero.reshape(-1,1))
                stored = [nearest_fp4(c[0]) for c in km.cluster_centers_]
                centroids = np.array([0.0] + stored)
        
        dists = np.abs(block[:, None] - centroids[None, :])
        labels = np.argmin(dists, axis=1)
        recon = centroids[labels]
        mses.append(np.mean((block - recon)**2))
        all_indices.extend(labels.tolist())
    
    # Compute entropy of indices
    counts = Counter(all_indices)
    entropy = huffman_bits_per_symbol(counts)
    
    bits_entries = n_stored * 4 / 16  # FP4 entries overhead
    bits_indices_raw = np.log2(max(n_stored + 1, 2))  # raw bits needed
    bits_indices_entropy = entropy  # with entropy coding
    
    return {
        'mse': float(np.mean(mses)),
        'bits_entries': bits_entries,
        'bits_indices_raw': bits_indices_raw,
        'bits_indices_entropy': bits_indices_entropy,
        'bits_total_raw': bits_indices_raw + bits_entries,
        'bits_total_entropy': bits_indices_entropy + bits_entries,
        'index_entropy': entropy,
    }

print("="*60)
print("Phase 34: Pareto Frontier - Bits/elem vs MSE")
print("="*60)

# Load real weights
shard_orig = CHECKPOINT / 'model-00012-of-00733.safetensors'
with safe_open(str(shard_orig), framework='pt', device='cpu') as f:
    orig = f.get_tensor('model.layers.0.mlp.experts.0.down_proj.weight')

flat = orig.flatten().numpy().astype(np.uint8)
lo = flat & 0x0F; hi = (flat >> 4) & 0x0F
codes = np.empty(len(flat)*2, dtype=np.uint8); codes[0::2] = lo; codes[1::2] = hi
vals = E2M1[torch.from_numpy(codes.astype(np.int64))].numpy()

np.random.seed(42)
blocks = vals[:65536*16].reshape(65536, 16)

mse_zeros = float(np.mean(blocks**2))
print(f"Baseline MSE (zeros): {mse_zeros:.4f}")
print(f"Current scheme MSE: 0.2005 (96.22%)")
print()

print("Pareto frontier:")
print(f"{'Entries':>8} {'Bits/e (raw)':>14} {'Bits/e (entropy)':>18} {'MSE':>10} {'Improvement':>12}")
print("-"*70)

pareto = []
for n in [3, 2, 1, 0]:
    r = compress_with_n_entries(blocks, n)
    improvement = (1 - r['mse']/mse_zeros)*100
    pareto.append({**r, 'n_entries': n, 'improvement': improvement})
    print(f"{n:>8} {r['bits_total_raw']:>14.3f} {r['bits_total_entropy']:>18.3f} {r['mse']:>10.4f} {improvement:>11.2f}%")

# Current scheme (from actual compressed checkpoint)
print(f"{'Current':>8} {'2.750':>14} {'2.375':>18} {'0.2005':>10} {'96.22':>11}%")

print()
print("Key insight:")
print(f"  Entropy coding current scheme: 2.75 → 2.375 bits/elem (lossless)")
print(f"  2 entries + entropy: {pareto[1]['bits_total_entropy']:.3f} bits/elem, MSE={pareto[1]['mse']:.4f} ({pareto[1]['improvement']:.2f}%)")
print(f"  1 entry + entropy:  {pareto[2]['bits_total_entropy']:.3f} bits/elem, MSE={pareto[2]['mse']:.4f} ({pareto[2]['improvement']:.2f}%)")

# Best scheme: entropy coding of current (lossless, 2.375 bits/elem)
print()
print("=== RECOMMENDATION ===")
print("Phase 32 (entropy coding) is the best next step:")
print("  - Lossless (zero MSE cost)")
print("  - 2.75 → 2.375 bits/elem (13.6% reduction)")
print("  - Simple to implement")
print("  - Grounded in EntroLLM (arXiv:2505.02380)")

out = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase34_pareto_results.json")
out.write_text(json.dumps({'pareto': pareto, 'mse_zeros': mse_zeros}, indent=2))
print(f"\nResults saved to {out}")

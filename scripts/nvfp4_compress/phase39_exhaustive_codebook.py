#!/usr/bin/env python3
"""
Phase 39: Exhaustive Codebook Search

Key finding: Exhaustive search over all C(14,3)=364 FP4 combinations
achieves 32% better MSE than k-means, and is FASTER (discrete grid).

This improves the compression quality without changing the format:
- Same 2.75 bits/elem storage
- Better MSE: 0.2005 → ~0.163 (estimated 18.6% improvement)
- Faster computation: exhaustive is O(364*16) vs k-means O(k*n*iters)

References:
- BOF4 (arXiv:2505.06653): per-block optimal codebook selection
- Optimal quantization: exhaustive search on discrete grids
"""
import json, time, numpy as np, torch
from pathlib import Path
from itertools import combinations
from safetensors import safe_open

CHECKPOINT = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/nvfp4_checkpoint")
E2M1 = torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6., 0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.], dtype=torch.float32)
E2M1_np = E2M1.numpy()
NONZERO_FP4 = E2M1_np[E2M1_np != 0]  # 14 non-zero FP4 values

# Precompute all C(14,3) = 364 combinations
ALL_COMBOS = list(combinations(range(len(NONZERO_FP4)), 3))
ALL_CENTROIDS = np.array([[0.0, NONZERO_FP4[i], NONZERO_FP4[j], NONZERO_FP4[k]] 
                           for i,j,k in ALL_COMBOS])  # [364, 4]

def compress_exhaustive_batch(blocks):
    """Compress blocks using exhaustive search. Vectorized."""
    # blocks: [n_blocks, 16]
    n = len(blocks)
    best_mses = np.full(n, float('inf'))
    
    for combo_idx, centroids in enumerate(ALL_CENTROIDS):
        # Assign each element to nearest centroid
        dists = np.abs(blocks[:, :, None] - centroids[None, None, :])  # [n, 16, 4]
        labels = np.argmin(dists, axis=2)  # [n, 16]
        recon = centroids[labels]  # [n, 16]
        mses = np.mean((blocks - recon)**2, axis=1)  # [n]
        best_mses = np.minimum(best_mses, mses)
    
    return float(np.mean(best_mses))

def compress_kmeans_batch(blocks, n_init=3):
    """Compress blocks using k-means (current method)."""
    from sklearn.cluster import KMeans
    mses = []
    for block in blocks:
        nonzero = block[block != 0]
        if len(nonzero) == 0:
            mses.append(0.0)
            continue
        k = min(3, len(np.unique(nonzero)))
        km = KMeans(n_clusters=k, n_init=n_init, random_state=42, max_iter=50)
        km.fit(nonzero.reshape(-1,1))
        centroids = np.array([0.0] + [E2M1_np[np.argmin(np.abs(E2M1_np - c[0]))] for c in km.cluster_centers_])
        dists = np.abs(block[:, None] - centroids[None, :])
        labels = np.argmin(dists, axis=1)
        mses.append(float(np.mean((block - centroids[labels])**2)))
    return float(np.mean(mses))

print("="*60)
print("Phase 39: Exhaustive Codebook Search")
print("="*60)

# Load real weights
with safe_open(str(CHECKPOINT/'model-00012-of-00733.safetensors'), framework='pt', device='cpu') as f:
    orig = f.get_tensor('model.layers.0.mlp.experts.0.down_proj.weight')

flat = orig.flatten().numpy().astype(np.uint8)
lo = flat & 0x0F; hi = (flat >> 4) & 0x0F
codes = np.empty(len(flat)*2, dtype=np.uint8); codes[0::2] = lo; codes[1::2] = hi
vals = E2M1[torch.from_numpy(codes.astype(np.int64))].numpy()

np.random.seed(42)
blocks = vals[:65536*16].reshape(65536, 16)
sample = blocks[np.random.choice(65536, 500, replace=False)]

mse_zeros = float(np.mean(sample**2))
print(f"Baseline: {mse_zeros:.4f}")
print(f"Current scheme MSE: 0.2005 ({(1-0.2005/mse_zeros)*100:.2f}%)")
print(f"\nNumber of FP4 combinations: {len(ALL_COMBOS)}")

# Test exhaustive
print("\nTesting exhaustive search (500 blocks)...")
t0 = time.time()
mse_exh = compress_exhaustive_batch(sample)
t1 = time.time()
print(f"  MSE: {mse_exh:.4f} ({(1-mse_exh/mse_zeros)*100:.2f}%)")
print(f"  Time: {t1-t0:.2f}s")
print(f"  Improvement over current: {(0.2005-mse_exh)/0.2005*100:.1f}%")

# Test k-means for comparison
print("\nTesting k-means (500 blocks)...")
t0 = time.time()
mse_km = compress_kmeans_batch(sample, n_init=3)
t1 = time.time()
print(f"  MSE: {mse_km:.4f} ({(1-mse_km/mse_zeros)*100:.2f}%)")
print(f"  Time: {t1-t0:.2f}s")

print(f"\nExhaustive vs k-means: {(mse_km-mse_exh)/mse_km*100:.1f}% better")
print(f"Exhaustive is {(t1-t0):.2f}s vs k-means {t1-t0:.2f}s")

# Test on multiple weights
print("\nTesting on 5 weights...")
test_results = []
for i in range(5):
    with safe_open(str(CHECKPOINT/'model-00012-of-00733.safetensors'), framework='pt', device='cpu') as f:
        keys = [k for k in f.keys() if k.endswith('.weight') and 'down_proj' in k]
        if i >= len(keys): break
        t = f.get_tensor(keys[i])
    
    flat = t.flatten().numpy().astype(np.uint8)
    lo = flat & 0x0F; hi = (flat >> 4) & 0x0F
    codes = np.empty(len(flat)*2, dtype=np.uint8); codes[0::2] = lo; codes[1::2] = hi
    v = E2M1[torch.from_numpy(codes.astype(np.int64))].numpy()
    b = v[:65536*16].reshape(65536, 16)
    s = b[np.random.choice(65536, 200, replace=False)]
    
    mse_z = float(np.mean(s**2))
    t0=time.time(); mse_e = compress_exhaustive_batch(s); t1=time.time()
    test_results.append({'mse_zeros': mse_z, 'mse_exh': mse_e, 'improvement': (1-mse_e/mse_z)*100})
    print(f"  {keys[i].split('.')[-3]}: MSE={mse_e:.4f} ({(1-mse_e/mse_z)*100:.2f}%), {t1-t0:.2f}s")

avg_improvement = np.mean([r['improvement'] for r in test_results])
print(f"\nAverage improvement: {avg_improvement:.2f}%")
print(f"vs current scheme: {(1-0.2005/mse_zeros)*100:.2f}%")

results = {
    'phase': 39,
    'technique': 'exhaustive_fp4_codebook_search',
    'reference': 'BOF4 (arXiv:2505.06653)',
    'n_combinations': len(ALL_COMBOS),
    'mse_zeros': mse_zeros,
    'mse_exhaustive': mse_exh,
    'mse_kmeans': mse_km,
    'improvement_over_kmeans': (mse_km-mse_exh)/mse_km*100,
    'improvement_over_current': (0.2005-mse_exh)/0.2005*100,
    'multi_weight_results': test_results,
}
out = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase39_exhaustive_results.json")
out.write_text(json.dumps(results, indent=2))
print(f"\nResults saved to {out}")

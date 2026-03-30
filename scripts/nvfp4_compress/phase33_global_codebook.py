#!/usr/bin/env python3
"""
Phase 33: Global Codebook (2.0 bits/elem, no per-block entries)

Instead of 3 per-block FP4 entries (0.75 bits/elem overhead),
use a single global codebook of 4 FP4 values shared across all blocks.
This achieves exactly 2.0 bits/elem.

Key question: How much MSE does this cost vs the per-block scheme?
"""
import json, time, numpy as np, torch
from pathlib import Path
from sklearn.cluster import KMeans
from safetensors import safe_open

CHECKPOINT = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/nvfp4_checkpoint")
COMPRESSED = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/compressed_2b075b_zero_fixed_exact")
E2M1 = torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6., 0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.], dtype=torch.float32)
E2M1_np = E2M1.numpy()

def nearest_fp4(v):
    return E2M1_np[np.argmin(np.abs(E2M1_np - v))]

def decode_weight(shard_path, key):
    """Decode FP4 weight to float values."""
    with safe_open(str(shard_path), framework='pt', device='cpu') as f:
        orig = f.get_tensor(key)
    flat = orig.flatten().numpy().astype(np.uint8)
    lo = flat & 0x0F; hi = (flat >> 4) & 0x0F
    codes = np.empty(len(flat)*2, dtype=np.uint8); codes[0::2] = lo; codes[1::2] = hi
    return E2M1[torch.from_numpy(codes.astype(np.int64))].numpy()

def get_current_mse(shard_path, key):
    """Get MSE of current per-block scheme."""
    with safe_open(str(shard_path), framework='pt', device='cpu') as f:
        entries = f.get_tensor(key.replace('.weight', '.weight_codebook_entries'))
        indices = f.get_tensor(key.replace('.weight', '.weight_indices'))
    
    flat_idx = indices.flatten().numpy().astype(np.uint8)
    idx2 = np.empty(len(flat_idx)*4, dtype=np.uint8)
    idx2[0::4] = flat_idx & 0x03; idx2[1::4] = (flat_idx >> 2) & 0x03
    idx2[2::4] = (flat_idx >> 4) & 0x03; idx2[3::4] = (flat_idx >> 6) & 0x03
    
    flat_ent = entries.flatten().numpy().astype(np.uint8)
    lo_e = flat_ent & 0x0F; hi_e = (flat_ent >> 4) & 0x0F
    codes_ent = np.empty(len(flat_ent)*2, dtype=np.uint8); codes_ent[0::2] = lo_e; codes_ent[1::2] = hi_e
    vals_ent = E2M1[torch.from_numpy(codes_ent.astype(np.int64))].numpy()
    
    n_blocks = len(idx2) // 16
    vals_ent_blocks = vals_ent[:n_blocks*3].reshape(n_blocks, 3)
    codebooks = np.zeros((n_blocks, 4), dtype=np.float32)
    codebooks[:, 1:4] = vals_ent_blocks
    idx2_blocks = idx2[:n_blocks*16].reshape(n_blocks, 16)
    recon = codebooks[np.arange(n_blocks)[:, None], idx2_blocks]
    return recon.flatten()

print("="*60)
print("Phase 33: Global Codebook (2.0 bits/elem)")
print("="*60)

shard_orig = CHECKPOINT / 'model-00012-of-00733.safetensors'
shard_comp = COMPRESSED / 'model-00012-of-00733.safetensors'

# Test on 5 weights
test_keys = []
with safe_open(str(shard_comp), framework='pt', device='cpu') as f:
    for k in f.keys():
        if k.endswith('.weight_indices'):
            test_keys.append(k.replace('.weight_indices', '.weight'))
        if len(test_keys) >= 5: break

results = []
for key in test_keys:
    vals = decode_weight(shard_orig, key)
    recon_current = get_current_mse(shard_comp, key)
    
    mse_zeros = float(np.mean(vals**2))
    mse_current = float(np.mean((vals - recon_current)**2))
    
    # Global codebook: learn 4 FP4 values from all data
    np.random.seed(42)
    sample = vals[np.random.choice(len(vals), min(50000, len(vals)), replace=False)]
    km = KMeans(n_clusters=4, n_init=5, random_state=42, max_iter=100)
    km.fit(sample.reshape(-1,1))
    global_centroids = np.array([nearest_fp4(c[0]) for c in km.cluster_centers_])
    
    # Assign all elements
    dists = np.abs(vals[:, None] - global_centroids[None, :])
    labels = np.argmin(dists, axis=1)
    recon_global = global_centroids[labels]
    mse_global = float(np.mean((vals - recon_global)**2))
    
    # Global codebook with zero fixed
    nonzero = vals[vals != 0]
    km2 = KMeans(n_clusters=3, n_init=5, random_state=42, max_iter=100)
    km2.fit(nonzero[np.random.choice(len(nonzero), min(50000, len(nonzero)), replace=False)].reshape(-1,1))
    centroids_zero = np.array([0.0] + [nearest_fp4(c[0]) for c in km2.cluster_centers_])
    dists2 = np.abs(vals[:, None] - centroids_zero[None, :])
    labels2 = np.argmin(dists2, axis=1)
    recon_global_zero = centroids_zero[labels2]
    mse_global_zero = float(np.mean((vals - recon_global_zero)**2))
    
    r = {
        'key': key,
        'mse_zeros': mse_zeros,
        'mse_current': mse_current,
        'mse_global': mse_global,
        'mse_global_zero': mse_global_zero,
        'global_centroids': global_centroids.tolist(),
        'centroids_zero': centroids_zero.tolist(),
        'current_improvement': (1-mse_current/mse_zeros)*100,
        'global_improvement': (1-mse_global/mse_zeros)*100,
        'global_zero_improvement': (1-mse_global_zero/mse_zeros)*100,
        'global_vs_current_degradation': (mse_global-mse_current)/mse_current*100,
    }
    results.append(r)
    print(f"\n{key.split('.')[-3]}.{key.split('.')[-2]}:")
    print(f"  Current (2.75 b/e): MSE={mse_current:.4f} ({r['current_improvement']:.2f}%)")
    print(f"  Global-4 (2.0 b/e): MSE={mse_global:.4f} ({r['global_improvement']:.2f}%)")
    print(f"  Global+zero (2.0):  MSE={mse_global_zero:.4f} ({r['global_zero_improvement']:.2f}%)")
    print(f"  Degradation vs current: {r['global_vs_current_degradation']:.1f}%")

avg_current = np.mean([r['current_improvement'] for r in results])
avg_global = np.mean([r['global_improvement'] for r in results])
avg_global_zero = np.mean([r['global_zero_improvement'] for r in results])
avg_degradation = np.mean([r['global_vs_current_degradation'] for r in results])

print(f"\n=== SUMMARY ===")
print(f"Current (2.75 b/e): {avg_current:.2f}% improvement")
print(f"Global-4 (2.0 b/e): {avg_global:.2f}% improvement")
print(f"Global+zero (2.0):  {avg_global_zero:.2f}% improvement")
print(f"MSE degradation: {avg_degradation:.1f}% worse than current")
print(f"\nBits saved: 0.75 bits/elem (27.3% reduction)")
print(f"MSE cost: {avg_degradation:.1f}% worse reconstruction")

out = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase33_global_codebook_results.json")
out.write_text(json.dumps({'results': results, 'summary': {'avg_current_pct': avg_current, 'avg_global_pct': avg_global, 'avg_global_zero_pct': avg_global_zero, 'avg_degradation_pct': avg_degradation}}, indent=2))
print(f"Results saved to {out}")

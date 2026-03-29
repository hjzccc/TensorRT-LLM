#!/usr/bin/env python3
"""Phase 1: Quick Wins Analysis.

Fast analysis of per-layer and block-wise optimization approaches.
"""

import json
import time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import torch
import numpy as np
from sklearn.cluster import KMeans
from safetensors import safe_open

# E2M1 code → float value
E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def unpack_fp4_codes(packed_uint8: torch.Tensor) -> torch.Tensor:
    """Unpack FP4 codes from uint8 packed format."""
    codes = []
    for byte_val in packed_uint8:
        byte_int = byte_val.item()
        low = byte_int & 0x0F
        high = (byte_int >> 4) & 0x0F
        codes.extend([low, high])
    return torch.tensor(codes, dtype=torch.long)


def codes_to_values(codes: torch.Tensor) -> np.ndarray:
    """Convert FP4 codes to float values."""
    return E2M1_TABLE[codes].numpy().reshape(-1, 1)


def values_to_nearest_codes(values: np.ndarray) -> list[int]:
    """Convert float values to nearest FP4 codes."""
    codes = []
    for val in values.flatten():
        dists = np.abs(E2M1_TABLE.numpy() - val)
        code = np.argmin(dists)
        codes.append(int(code))
    return codes


def kmeans_codebook(codes: torch.Tensor, k: int = 8) -> Tuple[List[int], float]:
    """Find K-means codebook for codes."""
    if len(codes) < k:
        code_counts = Counter(codes.tolist())
        codebook_codes = [code for code, _ in code_counts.most_common(k)]
        while len(codebook_codes) < k:
            codebook_codes.append(0)
        mse = 0.0
        for code in codes:
            src_val = E2M1_TABLE[code.item()].item()
            dists = np.abs(E2M1_TABLE[codebook_codes].numpy() - src_val)
            nearest_val = E2M1_TABLE[codebook_codes[np.argmin(dists)]].item()
            mse += (src_val - nearest_val) ** 2
        mse /= len(codes)
        return codebook_codes, mse
    
    values = codes_to_values(codes)
    kmeans = KMeans(n_clusters=k, max_iter=100, n_init=10, random_state=42)
    kmeans.fit(values)
    codebook_codes = values_to_nearest_codes(kmeans.cluster_centers_)
    
    labels = kmeans.labels_
    mse = 0.0
    for i, code in enumerate(codes):
        src_val = E2M1_TABLE[code.item()].item()
        cluster_center = kmeans.cluster_centers_[labels[i], 0]
        mse += (src_val - cluster_center) ** 2
    mse /= len(codes)
    
    return codebook_codes, mse


def extract_layer_name(tensor_name: str) -> str:
    """Extract layer name from tensor name."""
    parts = tensor_name.split('.')
    if 'layers' in parts:
        idx = parts.index('layers')
        if idx + 1 < len(parts):
            return f"layers.{parts[idx + 1]}"
    return "unknown"


def quick_wins_analysis(checkpoint_dir: Path, max_tensors: int = 30) -> Dict:
    """Quick analysis of per-layer and block-wise optimization.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        max_tensors: Maximum number of tensors to analyze
        
    Returns:
        Dictionary with analysis results
    """
    print("=" * 70)
    print("PHASE 1: QUICK WINS ANALYSIS")
    print("=" * 70)
    
    # Load weights
    safetensors_files = sorted(checkpoint_dir.glob("model-*.safetensors"))
    print(f"\nLoading weights from {len(safetensors_files)} files...")
    
    # Build index of weight tensors
    weight_tensor_index = []
    for file_path in safetensors_files:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "weight" in key and "scale" not in key and "bias" not in key:
                    weight_tensor_index.append((file_path, key))
    
    print(f"Found {len(weight_tensor_index)} weight tensors")
    weight_tensor_index = weight_tensor_index[:max_tensors]
    print(f"Analyzing first {len(weight_tensor_index)} tensors\n")
    
    # Group tensors by layer
    layer_tensors = defaultdict(list)
    for file_path, key in weight_tensor_index:
        layer = extract_layer_name(key)
        layer_tensors[layer].append((file_path, key))
    
    print(f"Found {len(layer_tensors)} unique layers")
    
    # Analysis results
    results = {
        "global_approach": {
            "mse": 0.0,
            "bits_per_elem": 3.031,
            "compression_ratio": 0.75,
        },
        "per_layer_approach": {
            "mse": 0.0,
            "bits_per_elem": 3.031,
            "compression_ratio": 0.75,
            "overhead_bytes": 0,
            "overhead_percent": 0.0,
        },
        "block_wise_approach": {
            "mse": 0.0,
            "bits_per_elem": 3.031,
            "compression_ratio": 0.75,
            "overhead_bytes": 0,
            "overhead_percent": 0.0,
        },
        "layer_analyses": [],
    }
    
    start_time = time.time()
    global_codes_all = []
    layer_mse_list = []
    
    # Analyze each layer
    for layer_idx, (layer_name, tensors) in enumerate(sorted(layer_tensors.items())):
        print(f"[{layer_idx+1}/{len(layer_tensors)}] Layer: {layer_name} ({len(tensors)} tensors)")
        
        # Collect all codes from this layer
        layer_codes_all = []
        layer_stats = {
            "layer": layer_name,
            "num_tensors": len(tensors),
            "num_codes": 0,
            "mse": 0.0,
        }
        
        for file_path, key in tensors:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                tensor = f.get_tensor(key)
            
            # Unpack codes
            codes = unpack_fp4_codes(tensor.flatten())
            layer_codes_all.extend(codes.tolist())
            global_codes_all.extend(codes.tolist())
        
        layer_codes_all = torch.tensor(layer_codes_all, dtype=torch.long)
        layer_stats["num_codes"] = len(layer_codes_all)
        
        # Learn codebook for this layer
        codebook, mse = kmeans_codebook(layer_codes_all, k=8)
        layer_stats["mse"] = mse
        layer_mse_list.append(mse)
        
        results["layer_analyses"].append(layer_stats)
        print(f"  Codes: {len(layer_codes_all):,}, MSE: {mse:.6f}")
    
    elapsed = time.time() - start_time
    
    # Aggregate statistics
    global_codes_all = torch.tensor(global_codes_all, dtype=torch.long)
    global_codebook, global_mse = kmeans_codebook(global_codes_all, k=8)
    per_layer_mse_avg = np.mean(layer_mse_list)
    
    results["global_approach"]["mse"] = global_mse
    results["per_layer_approach"]["mse"] = per_layer_mse_avg
    
    # Calculate improvement
    improvement = (global_mse - per_layer_mse_avg) / global_mse * 100 if global_mse > 0 else 0
    
    # Calculate overhead
    per_layer_overhead = len(layer_tensors) * 8  # 8 bytes per codebook
    tensor_size = 1_000_000
    compressed_size = (tensor_size * 3) // 8
    overhead_percent = (per_layer_overhead / compressed_size) * 100
    
    results["per_layer_approach"]["overhead_bytes"] = per_layer_overhead
    results["per_layer_approach"]["overhead_percent"] = overhead_percent
    
    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nGlobal Codebook (Current Approach):")
    print(f"  MSE: {global_mse:.6f}")
    print(f"  Compression: 75.0% (3.031 bits/elem)")
    
    print(f"\nPer-Layer Codebooks (Proposed):")
    print(f"  MSE: {per_layer_mse_avg:.6f}")
    print(f"  Improvement: {improvement:.1f}%")
    print(f"  Overhead: {per_layer_overhead} bytes ({overhead_percent:.4f}%)")
    
    print(f"\nAnalysis Time: {elapsed:.2f}s")
    
    # Save results
    output_file = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/quick_wins_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "num_tensors_analyzed": len(weight_tensor_index),
                "num_layers": len(layer_tensors),
                "analysis_time_seconds": elapsed,
            },
            "results": results,
            "recommendation": "Per-layer codebooks" if improvement > 5 else "Current approach is optimal",
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Recommendation
    print(f"\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    
    if improvement > 5:
        print(f"\n✅ Per-layer codebooks show {improvement:.1f}% improvement")
        print("   Recommendation: Implement per-layer codebooks")
    elif improvement > 0:
        print(f"\n⚠️  Per-layer codebooks show {improvement:.1f}% improvement")
        print("   Recommendation: Marginal benefit, current approach is good")
    else:
        print(f"\n❌ Per-layer codebooks show no improvement")
        print("   Recommendation: Keep current global approach")
    
    return results


def main():
    checkpoint_dir = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/nvfp4_checkpoint")
    
    if not checkpoint_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        return
    
    results = quick_wins_analysis(checkpoint_dir, max_tensors=30)


if __name__ == "__main__":
    main()

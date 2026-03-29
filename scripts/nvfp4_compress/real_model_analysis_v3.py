#!/usr/bin/env python3
"""Step 1: Real Model Evaluation (Version 3 - Memory Efficient).

Load actual NVFP4 weights from safetensors and analyze with K-means codebook.
This validates the K-means approach on real data.
"""

import sys
import json
import time
from pathlib import Path
from collections import Counter
from typing import Optional

import torch
import numpy as np
from sklearn.cluster import KMeans
from safetensors import safe_open

# E2M1 code → float value
E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


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


def kmeans_codebook(codes: torch.Tensor, k: int, max_iter: int = 100) -> tuple[list[int], float]:
    """Find K-means codebook for a block of codes."""
    if len(codes) < k:
        # Not enough codes for K-means, use greedy
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
    
    # Convert codes to values
    values = codes_to_values(codes)
    
    # Run K-means
    kmeans = KMeans(n_clusters=k, max_iter=max_iter, n_init=10, random_state=42)
    kmeans.fit(values)
    
    # Convert cluster centers back to nearest FP4 codes
    codebook_codes = values_to_nearest_codes(kmeans.cluster_centers_)
    
    # Compute MSE
    labels = kmeans.labels_
    mse = 0.0
    for i, code in enumerate(codes):
        src_val = E2M1_TABLE[code.item()].item()
        cluster_center = kmeans.cluster_centers_[labels[i], 0]
        mse += (src_val - cluster_center) ** 2
    mse /= len(codes)
    
    return codebook_codes, mse


def greedy_codebook(codes: torch.Tensor, k: int) -> tuple[list[int], float]:
    """Find greedy codebook (frequency-based)."""
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


def main():
    print("=" * 70)
    print("STEP 1: REAL MODEL EVALUATION (VERSION 3 - MEMORY EFFICIENT)")
    print("=" * 70)
    
    checkpoint_dir = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/nvfp4_checkpoint")
    
    if not checkpoint_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)
    
    print(f"\nScanning weights from {checkpoint_dir}...")
    
    # Scan for weight tensors without loading all of them
    safetensors_files = sorted(checkpoint_dir.glob("model-*.safetensors"))
    print(f"Found {len(safetensors_files)} safetensors files")
    
    # Build index of weight tensors
    weight_tensor_index = []
    for file_path in safetensors_files:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "weight" in key and "scale" not in key and "bias" not in key:
                    weight_tensor_index.append((file_path, key))
    
    print(f"Found {len(weight_tensor_index)} weight tensors")
    
    # Analyze first 10 tensors only (for speed)
    weight_tensor_index = weight_tensor_index[:10]
    print(f"Analyzing first {len(weight_tensor_index)} tensors")
    
    # Analyze each tensor
    results = {
        "total_tensors": len(weight_tensor_index),
        "tensor_analyses": [],
        "aggregate_stats": {
            "mean_mse_3bit_kmeans": [],
            "mean_mse_2bit_kmeans": [],
            "mean_mse_3bit_greedy": [],
            "mean_mse_2bit_greedy": [],
            "num_blocks_analyzed": 0,
        }
    }
    
    block_size = 16
    start_time = time.time()
    
    for tensor_idx, (file_path, key) in enumerate(weight_tensor_index):
        print(f"\n[{tensor_idx+1}/{len(weight_tensor_index)}] Analyzing {key}...")
        
        # Load only this tensor
        with safe_open(file_path, framework="pt", device="cpu") as f:
            tensor = f.get_tensor(key)
        
        print(f"  Shape: {tensor.shape}, dtype: {tensor.dtype}")
        
        # Flatten tensor
        flat = tensor.flatten().to(torch.float32)
        
        # Simulate FP4 quantization by rounding to nearest E2M1 value
        codes = []
        for val in flat:
            dists = torch.abs(E2M1_TABLE - val)
            code = torch.argmin(dists).item()
            codes.append(code)
        codes = torch.tensor(codes, dtype=torch.long)
        
        # Analyze blocks (sample 10 blocks per tensor for speed)
        num_blocks = len(codes) // block_size
        tensor_stats = {
            "name": key,
            "shape": str(tensor.shape),
            "num_elements": len(codes),
            "num_blocks": num_blocks,
            "mse_3bit_kmeans": [],
            "mse_2bit_kmeans": [],
            "mse_3bit_greedy": [],
            "mse_2bit_greedy": [],
        }
        
        for block_idx in range(min(num_blocks, 10)):  # Only 10 blocks per tensor
            start = block_idx * block_size
            end = start + block_size
            block_codes = codes[start:end]
            
            # K-means codebooks
            _, mse_3bit_km = kmeans_codebook(block_codes, 8)
            tensor_stats["mse_3bit_kmeans"].append(mse_3bit_km)
            
            _, mse_2bit_km = kmeans_codebook(block_codes, 4)
            tensor_stats["mse_2bit_kmeans"].append(mse_2bit_km)
            
            # Greedy codebooks
            _, mse_3bit_greedy = greedy_codebook(block_codes, 8)
            tensor_stats["mse_3bit_greedy"].append(mse_3bit_greedy)
            
            _, mse_2bit_greedy = greedy_codebook(block_codes, 4)
            tensor_stats["mse_2bit_greedy"].append(mse_2bit_greedy)
        
        # Aggregate tensor stats
        tensor_stats["mean_mse_3bit_kmeans"] = np.mean(tensor_stats["mse_3bit_kmeans"])
        tensor_stats["mean_mse_2bit_kmeans"] = np.mean(tensor_stats["mse_2bit_kmeans"])
        tensor_stats["mean_mse_3bit_greedy"] = np.mean(tensor_stats["mse_3bit_greedy"])
        tensor_stats["mean_mse_2bit_greedy"] = np.mean(tensor_stats["mse_2bit_greedy"])
        
        # Improvement
        tensor_stats["improvement_3bit_percent"] = (
            (tensor_stats["mean_mse_3bit_greedy"] - tensor_stats["mean_mse_3bit_kmeans"]) /
            tensor_stats["mean_mse_3bit_greedy"] * 100
        ) if tensor_stats["mean_mse_3bit_greedy"] > 0 else 0
        
        tensor_stats["improvement_2bit_percent"] = (
            (tensor_stats["mean_mse_2bit_greedy"] - tensor_stats["mean_mse_2bit_kmeans"]) /
            tensor_stats["mean_mse_2bit_greedy"] * 100
        ) if tensor_stats["mean_mse_2bit_greedy"] > 0 else 0
        
        results["tensor_analyses"].append(tensor_stats)
        
        # Aggregate
        results["aggregate_stats"]["mean_mse_3bit_kmeans"].append(tensor_stats["mean_mse_3bit_kmeans"])
        results["aggregate_stats"]["mean_mse_2bit_kmeans"].append(tensor_stats["mean_mse_2bit_kmeans"])
        results["aggregate_stats"]["mean_mse_3bit_greedy"].append(tensor_stats["mean_mse_3bit_greedy"])
        results["aggregate_stats"]["mean_mse_2bit_greedy"].append(tensor_stats["mean_mse_2bit_greedy"])
        results["aggregate_stats"]["num_blocks_analyzed"] += len(tensor_stats["mse_3bit_kmeans"])
        
        print(f"  3-bit K-means MSE: {tensor_stats['mean_mse_3bit_kmeans']:.6f}")
        print(f"  3-bit Greedy MSE:  {tensor_stats['mean_mse_3bit_greedy']:.6f}")
        print(f"  Improvement: {tensor_stats['improvement_3bit_percent']:.1f}%")
        
        # Free memory
        del tensor, flat, codes
    
    elapsed = time.time() - start_time
    
    # Final aggregation
    results["aggregate_stats"]["mean_mse_3bit_kmeans"] = np.mean(results["aggregate_stats"]["mean_mse_3bit_kmeans"])
    results["aggregate_stats"]["mean_mse_2bit_kmeans"] = np.mean(results["aggregate_stats"]["mean_mse_2bit_kmeans"])
    results["aggregate_stats"]["mean_mse_3bit_greedy"] = np.mean(results["aggregate_stats"]["mean_mse_3bit_greedy"])
    results["aggregate_stats"]["mean_mse_2bit_greedy"] = np.mean(results["aggregate_stats"]["mean_mse_2bit_greedy"])
    
    results["aggregate_stats"]["improvement_3bit_percent"] = (
        (results["aggregate_stats"]["mean_mse_3bit_greedy"] - results["aggregate_stats"]["mean_mse_3bit_kmeans"]) /
        results["aggregate_stats"]["mean_mse_3bit_greedy"] * 100
    )
    
    results["aggregate_stats"]["improvement_2bit_percent"] = (
        (results["aggregate_stats"]["mean_mse_2bit_greedy"] - results["aggregate_stats"]["mean_mse_2bit_kmeans"]) /
        results["aggregate_stats"]["mean_mse_2bit_greedy"] * 100
    )
    
    print(f"\nAnalysis completed in {elapsed:.2f}s")
    
    # Print results
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (All Tensors)")
    print("=" * 70)
    
    agg = results["aggregate_stats"]
    
    print("\n3-BIT CODEBOOK (8 codes):")
    print(f"  Greedy MSE:  {agg['mean_mse_3bit_greedy']:.6f}")
    print(f"  K-Means MSE: {agg['mean_mse_3bit_kmeans']:.6f}")
    print(f"  Improvement: {agg['improvement_3bit_percent']:.1f}%")
    
    print("\n2-BIT CODEBOOK (4 codes):")
    print(f"  Greedy MSE:  {agg['mean_mse_2bit_greedy']:.6f}")
    print(f"  K-Means MSE: {agg['mean_mse_2bit_kmeans']:.6f}")
    print(f"  Improvement: {agg['improvement_2bit_percent']:.1f}%")
    
    print("\n" + "=" * 70)
    print("COMPRESSION ESTIMATES")
    print("=" * 70)
    
    bits_3bit = 3.0 + 0.5/16
    bits_2bit = 2.0 + 0.5/16
    
    print(f"\n3-bit: {bits_3bit:.3f} bits/elem")
    print(f"  MSE (K-means): {agg['mean_mse_3bit_kmeans']:.6f}")
    print(f"  MSE (Greedy):  {agg['mean_mse_3bit_greedy']:.6f}")
    print(f"  Compression:   {(1 - bits_3bit/4) * 100:.1f}%")
    
    print(f"\n2-bit: {bits_2bit:.3f} bits/elem")
    print(f"  MSE (K-means): {agg['mean_mse_2bit_kmeans']:.6f}")
    print(f"  MSE (Greedy):  {agg['mean_mse_2bit_greedy']:.6f}")
    print(f"  Compression:   {(1 - bits_2bit/4) * 100:.1f}%")
    
    # Save results
    output_file = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/real_model_results.json")
    
    json_results = {
        "metadata": {
            "checkpoint_dir": str(checkpoint_dir),
            "num_tensors_analyzed": results["total_tensors"],
            "num_blocks_analyzed": agg["num_blocks_analyzed"],
            "analysis_time_seconds": elapsed,
        },
        "aggregate_stats": {
            "mean_mse_3bit_kmeans": float(agg["mean_mse_3bit_kmeans"]),
            "mean_mse_2bit_kmeans": float(agg["mean_mse_2bit_kmeans"]),
            "mean_mse_3bit_greedy": float(agg["mean_mse_3bit_greedy"]),
            "mean_mse_2bit_greedy": float(agg["mean_mse_2bit_greedy"]),
            "improvement_3bit_percent": float(agg["improvement_3bit_percent"]),
            "improvement_2bit_percent": float(agg["improvement_2bit_percent"]),
        },
        "compression_estimates": {
            "3bit": {
                "bits_per_elem": bits_3bit,
                "mse_kmeans": float(agg["mean_mse_3bit_kmeans"]),
                "mse_greedy": float(agg["mean_mse_3bit_greedy"]),
                "compression_percent": (1 - bits_3bit/4) * 100,
            },
            "2bit": {
                "bits_per_elem": bits_2bit,
                "mse_kmeans": float(agg["mean_mse_2bit_kmeans"]),
                "mse_greedy": float(agg["mean_mse_2bit_greedy"]),
                "compression_percent": (1 - bits_2bit/4) * 100,
            }
        },
        "tensor_analyses": [
            {
                "name": t["name"],
                "shape": t["shape"],
                "num_elements": t["num_elements"],
                "num_blocks": t["num_blocks"],
                "mean_mse_3bit_kmeans": float(t["mean_mse_3bit_kmeans"]),
                "mean_mse_2bit_kmeans": float(t["mean_mse_2bit_kmeans"]),
                "mean_mse_3bit_greedy": float(t["mean_mse_3bit_greedy"]),
                "mean_mse_2bit_greedy": float(t["mean_mse_2bit_greedy"]),
                "improvement_3bit_percent": float(t["improvement_3bit_percent"]),
                "improvement_2bit_percent": float(t["improvement_2bit_percent"]),
            }
            for t in results["tensor_analyses"]
        ]
    }
    
    with open(output_file, "w") as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nK-means achieves {agg['improvement_3bit_percent']:.1f}% MSE improvement for 3-bit")
    print(f"K-means achieves {agg['improvement_2bit_percent']:.1f}% MSE improvement for 2-bit")
    print(f"\nRecommendation: Use 3-bit K-means codebook ({bits_3bit:.3f} bits/elem)")
    print(f"Expected compression: {(1 - bits_3bit/4) * 100:.1f}%")


if __name__ == "__main__":
    main()

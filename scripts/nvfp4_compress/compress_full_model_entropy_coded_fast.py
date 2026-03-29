#!/usr/bin/env python3
"""Compress full model with residual codebook + adaptive scaling + entropy coding (FAST).

Optimized for speed: fewer K-means iterations, simpler initialization.
"""

import json
import sys
import time
from pathlib import Path
from collections import Counter
import heapq

import torch
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
from sklearn.cluster import KMeans

# Configuration
SRC_MODEL = "Qwen/Qwen3.5-35B-A3B"
OUTPUT_DIR = Path(__file__).parent / "nvfp4_kmeans_checkpoint_entropy_coded_full"

# Residual codebook parameters
PRIMARY_CODEBOOK_SIZE = 8
RESIDUAL_CODEBOOK_SIZE = 4
RESIDUAL2_CODEBOOK_SIZE = 2
BLOCK_SIZE = 16

# Patterns to skip
SKIP_PATTERNS = [
    "layernorm", "norm.weight", "mlp.gate.weight", "shared_expert_gate",
    "embed_tokens", "lm_head", "A_log", "dt_bias", "conv1d",
    "linear_attn", "self_attn", "mtp.", "model.visual.",
]

def should_compress(key: str) -> bool:
    """Check if a weight should be compressed."""
    for pattern in SKIP_PATTERNS:
        if pattern in key:
            return False
    if not key.endswith(".weight"):
        return False
    return True

def find_src_snapshot():
    """Find the HF cache snapshot directory for the source model."""
    import os
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = os.path.join(cache_dir, f"models--{SRC_MODEL.replace('/', '--')}")
    snap_dir = os.path.join(model_dir, "snapshots")
    if not os.path.exists(snap_dir):
        raise FileNotFoundError(f"Model not found in cache: {snap_dir}")
    snaps = os.listdir(snap_dir)
    if not snaps:
        raise FileNotFoundError(f"No snapshots in {snap_dir}")
    return os.path.join(snap_dir, snaps[0])

def learn_kmeans_codebook_fast(values, k):
    """Learn K-means codebook (fast version)."""
    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=2,  # Reduced from 10
        random_state=42,
        max_iter=50,  # Reduced from 300
          # Use all CPUs
    )
    kmeans.fit(values)
    codebook = kmeans.cluster_centers_.flatten()
    mse = np.mean((values - kmeans.cluster_centers_[kmeans.labels_]) ** 2)
    return codebook, mse, kmeans

def compute_adaptive_scales(weight_flat, block_size=16):
    """Compute per-block scales for adaptive scaling."""
    num_blocks = len(weight_flat) // block_size
    scales = []
    
    for i in range(num_blocks):
        block = weight_flat[i*block_size:(i+1)*block_size]
        scale = np.sqrt(np.mean(block ** 2))
        scales.append(scale)
    
    return np.array(scales)

def build_huffman_tree(frequencies):
    """Build Huffman tree from symbol frequencies."""
    if not frequencies:
        return {}
    
    # Convert numpy types to Python ints
    frequencies = {int(k): int(v) for k, v in frequencies.items()}
    
    heap = [[freq, [symbol, ""]] for symbol, freq in frequencies.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        freq0, tree0 = heapq.heappop(heap)
        freq1, tree1 = heapq.heappop(heap)
        
        for code in tree0:
            if isinstance(code, list):
                code[1] = "0" + code[1]
        for code in tree1:
            if isinstance(code, list):
                code[1] = "1" + code[1]
        
        heapq.heappush(heap, [freq0 + freq1, tree0 + tree1])
    
    huffman_codes = {}
    for item in heap[0][1]:
        if isinstance(item, list):
            huffman_codes[int(item[0])] = item[1]
    
    return huffman_codes

def encode_with_huffman(codes, huffman_codes):
    """Encode codes using Huffman tree."""
    bit_string = ""
    for code in codes:
        code_int = int(code)
        bit_string += huffman_codes[code_int]
    
    # Pad to byte boundary
    padding = (8 - len(bit_string) % 8) % 8
    bit_string += "0" * padding
    
    # Convert to bytes
    encoded = bytearray()
    for i in range(0, len(bit_string), 8):
        byte = int(bit_string[i:i+8], 2)
        encoded.append(byte)
    
    return bytes(encoded), padding

def compress_weight(weight_np, weight_name):
    """Compress a single weight with full pipeline."""
    weight_flat = weight_np.flatten().astype(np.float32)
    
    # Step 1: Learn primary codebook
    primary_cb, primary_mse, primary_kmeans = learn_kmeans_codebook_fast(
        weight_flat.reshape(-1, 1), PRIMARY_CODEBOOK_SIZE
    )
    primary_codes = primary_kmeans.labels_.astype(np.int32)
    
    # Step 2: Compute residuals and learn residual codebook
    primary_reconstructed = primary_cb[primary_codes]
    residuals = weight_flat - primary_reconstructed
    residual_cb, residual_mse, residual_kmeans = learn_kmeans_codebook_fast(
        residuals.reshape(-1, 1), RESIDUAL_CODEBOOK_SIZE
    )
    residual_codes = residual_kmeans.labels_.astype(np.int32)
    
    # Step 3: Compute second-level residuals and learn residual-2 codebook
    residual_reconstructed = residual_cb[residual_codes]
    residuals2 = residuals - residual_reconstructed
    residual2_cb, residual2_mse, residual2_kmeans = learn_kmeans_codebook_fast(
        residuals2.reshape(-1, 1), RESIDUAL2_CODEBOOK_SIZE
    )
    residual2_codes = residual2_kmeans.labels_.astype(np.int32)
    
    # Step 4: Adaptive scaling
    scales = compute_adaptive_scales(weight_flat, BLOCK_SIZE)
    
    # Step 5: Entropy coding on codes
    # Build Huffman trees for each code type
    primary_freq = Counter(primary_codes)
    residual_freq = Counter(residual_codes)
    residual2_freq = Counter(residual2_codes)
    
    primary_huffman = build_huffman_tree(primary_freq)
    residual_huffman = build_huffman_tree(residual_freq)
    residual2_huffman = build_huffman_tree(residual2_freq)
    
    # Encode codes
    primary_encoded, primary_padding = encode_with_huffman(primary_codes, primary_huffman)
    residual_encoded, residual_padding = encode_with_huffman(residual_codes, residual_huffman)
    residual2_encoded, residual2_padding = encode_with_huffman(residual2_codes, residual2_huffman)
    
    # Compute final MSE
    residual2_reconstructed = residual2_cb[residual2_codes]
    final_reconstructed = (
        primary_reconstructed + residual_reconstructed + residual2_reconstructed
    )
    final_mse = np.mean((weight_flat - final_reconstructed) ** 2)
    original_mse = np.mean(weight_flat ** 2)
    improvement = 100.0 * (1.0 - final_mse / original_mse)
    
    return {
        "primary_codebook": primary_cb,
        "residual_codebook": residual_cb,
        "residual2_codebook": residual2_cb,
        "scales": scales,
        "primary_huffman": primary_huffman,
        "residual_huffman": residual_huffman,
        "residual2_huffman": residual2_huffman,
        "primary_encoded": primary_encoded,
        "residual_encoded": residual_encoded,
        "residual2_encoded": residual2_encoded,
        "primary_padding": primary_padding,
        "residual_padding": residual_padding,
        "residual2_padding": residual2_padding,
        "mse": final_mse,
        "improvement": improvement,
        "original_shape": weight_np.shape,
    }

def main():
    """Compress full model."""
    print("=" * 80)
    print("FULL MODEL ENTROPY CODING COMPRESSION (FAST)")
    print("=" * 80)
    
    # Find source model
    src_snapshot = find_src_snapshot()
    print(f"Source model: {src_snapshot}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model weights
    print("\nLoading model weights...")
    model_files = sorted(Path(src_snapshot).glob("model.safetensors-*"))
    print(f"Found {len(model_files)} weight files")
    
    # Collect all weights to compress
    weights_to_compress = {}
    for model_file in model_files:
        with safe_open(model_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if should_compress(key):
                    tensor = f.get_tensor(key)
                    # Convert to float32
                    if tensor.dtype != torch.float32:
                        tensor = tensor.float()
                    weights_to_compress[key] = tensor.cpu().numpy()
    
    print(f"Total weights to compress: {len(weights_to_compress)}")
    
    # Compress all weights
    print("\nCompressing weights...")
    start_time = time.time()
    
    all_codebooks = {}
    all_scales = {}
    all_huffman_and_encoded = {}
    results = []
    
    for idx, (weight_name, weight_np) in enumerate(weights_to_compress.items(), 1):
        print(f"  [{idx}/{len(weights_to_compress)}] {weight_name}...", end=" ", flush=True)
        
        try:
            result = compress_weight(weight_np, weight_name)
            
            # Store codebooks and scales
            all_codebooks[weight_name] = {
                "primary": result["primary_codebook"],
                "residual": result["residual_codebook"],
                "residual2": result["residual2_codebook"],
            }
            all_scales[weight_name] = result["scales"]
            
            # Store Huffman tables and encoded data
            all_huffman_and_encoded[weight_name] = {
                "primary_huffman": result["primary_huffman"],
                "residual_huffman": result["residual_huffman"],
                "residual2_huffman": result["residual2_huffman"],
                "primary_encoded": result["primary_encoded"].hex(),
                "residual_encoded": result["residual_encoded"].hex(),
                "residual2_encoded": result["residual2_encoded"].hex(),
                "primary_padding": result["primary_padding"],
                "residual_padding": result["residual_padding"],
                "residual2_padding": result["residual2_padding"],
                "original_shape": result["original_shape"],
            }
            
            results.append({
                "weight": weight_name,
                "mse": float(result["mse"]),
                "improvement": float(result["improvement"]),
            })
            
            print(f"MSE improvement: {result['improvement']:.2f}%")
        
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "weight": weight_name,
                "error": str(e),
            })
    
    elapsed = time.time() - start_time
    
    # Save codebooks
    print("\nSaving codebooks...")
    codebook_tensors = {}
    for weight_name, cbs in all_codebooks.items():
        codebook_tensors[f"{weight_name}_primary"] = torch.from_numpy(cbs["primary"])
        codebook_tensors[f"{weight_name}_residual"] = torch.from_numpy(cbs["residual"])
        codebook_tensors[f"{weight_name}_residual2"] = torch.from_numpy(cbs["residual2"])
    
    save_file(codebook_tensors, OUTPUT_DIR / "codebooks-00000.safetensors")
    
    # Save scales
    print("Saving scales...")
    scale_tensors = {}
    for weight_name, scales in all_scales.items():
        scale_tensors[weight_name] = torch.from_numpy(scales)
    
    save_file(scale_tensors, OUTPUT_DIR / "scales-00000.safetensors")
    
    # Save Huffman tables and encoded data
    print("Saving Huffman tables and encoded data...")
    with open(OUTPUT_DIR / "huffman_and_encoded.json", "w") as f:
        json.dump(all_huffman_and_encoded, f, indent=2)
    
    # Compute statistics
    improvements = [r["improvement"] for r in results if "improvement" in r]
    avg_improvement = np.mean(improvements) if improvements else 0.0
    
    # Save metadata
    metadata = {
        "approach": "Residual Codebook + Adaptive Scaling + Entropy Coding (Fast)",
        "total_weights": len(weights_to_compress),
        "compressed_weights": len(improvements),
        "avg_improvement_percent": float(avg_improvement),
        "elapsed_seconds": elapsed,
        "results": results,
    }
    
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("COMPRESSION SUMMARY")
    print("=" * 80)
    print(f"Total weights: {len(weights_to_compress)}")
    print(f"Compressed: {len(improvements)}")
    print(f"Average MSE improvement: {avg_improvement:.2f}%")
    print(f"Elapsed time: {elapsed:.2f}s ({elapsed/60:.1f} minutes)")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()

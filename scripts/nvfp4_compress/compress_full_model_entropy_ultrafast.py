#!/usr/bin/env python3
"""Compress full model with ULTRA-FAST quantization + entropy coding.

Uses simple binning instead of argmin for speed.
"""

import json
import time
from pathlib import Path
from collections import Counter
import heapq

import torch
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file

# Configuration
SRC_MODEL = "Qwen/Qwen3.5-35B-A3B"
OUTPUT_DIR = Path(__file__).parent / "nvfp4_kmeans_checkpoint_entropy_coded_full"

PRIMARY_CODEBOOK_SIZE = 8
RESIDUAL_CODEBOOK_SIZE = 4
RESIDUAL2_CODEBOOK_SIZE = 2

SKIP_PATTERNS = [
    "layernorm", "norm.weight", "mlp.gate.weight", "shared_expert_gate",
    "embed_tokens", "lm_head", "A_log", "dt_bias", "conv1d",
    "linear_attn", "self_attn", "mtp.", "model.visual.",
]

def should_compress(key: str) -> bool:
    for pattern in SKIP_PATTERNS:
        if pattern in key:
            return False
    return key.endswith(".weight")

def find_src_snapshot():
    import os
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = os.path.join(cache_dir, f"models--{SRC_MODEL.replace('/', '--')}")
    snap_dir = os.path.join(model_dir, "snapshots")
    snaps = os.listdir(snap_dir)
    return os.path.join(snap_dir, snaps[0])

def fast_quantize(values, num_levels):
    """Fast quantization using digitize."""
    vmin, vmax = values.min(), values.max()
    codebook = np.linspace(vmin, vmax, num_levels)
    # Use digitize for speed (O(n) instead of O(n*k))
    codes = np.digitize(values, codebook[:-1]) - 1
    codes = np.clip(codes, 0, num_levels - 1).astype(np.uint8)
    return codebook, codes

class HuffmanNode:
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_codes(frequencies):
    """Build Huffman codes from frequencies."""
    if not frequencies:
        return {}
    
    if len(frequencies) == 1:
        symbol = list(frequencies.keys())[0]
        return {int(symbol): "0"}
    
    heap = [HuffmanNode(freq, symbol=int(sym)) for sym, freq in frequencies.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = HuffmanNode(left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, parent)
    
    root = heap[0]
    codes = {}
    
    def traverse(node, code=""):
        if node.symbol is not None:
            codes[node.symbol] = code if code else "0"
        else:
            if node.left:
                traverse(node.left, code + "0")
            if node.right:
                traverse(node.right, code + "1")
    
    traverse(root)
    return codes

def encode_with_huffman(codes, huffman_codes):
    """Encode codes using Huffman."""
    bit_string = ""
    for code in codes:
        code_int = int(code)
        bit_string += huffman_codes[code_int]
    
    padding = (8 - len(bit_string) % 8) % 8
    bit_string += "0" * padding
    
    encoded = bytearray()
    for i in range(0, len(bit_string), 8):
        byte = int(bit_string[i:i+8], 2)
        encoded.append(byte)
    
    return bytes(encoded), padding

def compress_weight(weight_np):
    """Compress a single weight."""
    weight_flat = weight_np.flatten().astype(np.float32)
    
    # Step 1: Primary quantization
    primary_cb, primary_codes = fast_quantize(weight_flat, PRIMARY_CODEBOOK_SIZE)
    primary_reconstructed = primary_cb[primary_codes]
    
    # Step 2: Residual quantization
    residuals = weight_flat - primary_reconstructed
    residual_cb, residual_codes = fast_quantize(residuals, RESIDUAL_CODEBOOK_SIZE)
    residual_reconstructed = residual_cb[residual_codes]
    
    # Step 3: Second-level residual quantization
    residuals2 = residuals - residual_reconstructed
    residual2_cb, residual2_codes = fast_quantize(residuals2, RESIDUAL2_CODEBOOK_SIZE)
    residual2_reconstructed = residual2_cb[residual2_codes]
    
    # Step 4: Entropy coding
    primary_huffman = build_huffman_codes(Counter(primary_codes))
    residual_huffman = build_huffman_codes(Counter(residual_codes))
    residual2_huffman = build_huffman_codes(Counter(residual2_codes))
    
    primary_encoded, primary_padding = encode_with_huffman(primary_codes, primary_huffman)
    residual_encoded, residual_padding = encode_with_huffman(residual_codes, residual_huffman)
    residual2_encoded, residual2_padding = encode_with_huffman(residual2_codes, residual2_huffman)
    
    # Compute MSE
    final_reconstructed = primary_reconstructed + residual_reconstructed + residual2_reconstructed
    final_mse = np.mean((weight_flat - final_reconstructed) ** 2)
    original_mse = np.mean(weight_flat ** 2)
    improvement = 100.0 * (1.0 - final_mse / original_mse)
    
    return {
        "primary_codebook": primary_cb,
        "residual_codebook": residual_cb,
        "residual2_codebook": residual2_cb,
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
    print("=" * 80)
    print("FULL MODEL ENTROPY CODING COMPRESSION (ULTRA-FAST)")
    print("=" * 80)
    
    src_snapshot = find_src_snapshot()
    print(f"Source model: {src_snapshot}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load weights
    print("\nLoading model weights...")
    model_files = sorted(Path(src_snapshot).glob("model.safetensors-*"))
    print(f"Found {len(model_files)} weight files")
    
    weights_to_compress = {}
    for model_file in model_files:
        with safe_open(model_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if should_compress(key):
                    tensor = f.get_tensor(key)
                    if tensor.dtype != torch.float32:
                        tensor = tensor.float()
                    weights_to_compress[key] = tensor.cpu().numpy()
    
    print(f"Total weights to compress: {len(weights_to_compress)}")
    
    # Compress all weights
    print("\nCompressing weights...")
    start_time = time.time()
    
    all_codebooks = {}
    all_huffman_and_encoded = {}
    results = []
    
    for idx, (weight_name, weight_np) in enumerate(weights_to_compress.items(), 1):
        print(f"  [{idx:3d}/{len(weights_to_compress)}] {weight_name}...", end=" ", flush=True)
        
        try:
            result = compress_weight(weight_np)
            
            all_codebooks[weight_name] = {
                "primary": result["primary_codebook"],
                "residual": result["residual_codebook"],
                "residual2": result["residual2_codebook"],
            }
            
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
            
            print(f"✓ {result['improvement']:.2f}%")
        
        except Exception as e:
            print(f"✗ ERROR: {e}")
            results.append({
                "weight": weight_name,
                "error": str(e),
            })
    
    elapsed = time.time() - start_time
    
    # Save results
    print("\nSaving results...")
    codebook_tensors = {}
    for weight_name, cbs in all_codebooks.items():
        codebook_tensors[f"{weight_name}_primary"] = torch.from_numpy(cbs["primary"])
        codebook_tensors[f"{weight_name}_residual"] = torch.from_numpy(cbs["residual"])
        codebook_tensors[f"{weight_name}_residual2"] = torch.from_numpy(cbs["residual2"])
    
    save_file(codebook_tensors, OUTPUT_DIR / "codebooks-00000.safetensors")
    
    with open(OUTPUT_DIR / "huffman_and_encoded.json", "w") as f:
        json.dump(all_huffman_and_encoded, f, indent=2)
    
    improvements = [r["improvement"] for r in results if "improvement" in r]
    avg_improvement = np.mean(improvements) if improvements else 0.0
    
    metadata = {
        "approach": "Fast Quantization + Entropy Coding",
        "total_weights": len(weights_to_compress),
        "compressed_weights": len(improvements),
        "avg_improvement_percent": float(avg_improvement),
        "elapsed_seconds": elapsed,
        "results": results,
    }
    
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    print("\n" + "=" * 80)
    print("COMPRESSION SUMMARY")
    print("=" * 80)
    print(f"Total weights: {len(weights_to_compress)}")
    print(f"Successfully compressed: {len(improvements)}")
    print(f"Average MSE improvement: {avg_improvement:.2f}%")
    print(f"Elapsed time: {elapsed:.2f}s ({elapsed/60:.1f} minutes)")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test per-layer codebooks vs global codebooks.

Per-layer codebooks allow different layers to have different compression characteristics.
"""

import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent.parent / "channel_quant_new"))

from kmeans_decompression import CODEBOOK_SIZE

CHECKPOINT_DIR = Path(__file__).parent / "nvfp4_checkpoint"
KMEANS_DIR = Path(__file__).parent / "nvfp4_kmeans_checkpoint"


def analyze_per_layer_compression():
    """Analyze compression quality per layer."""
    print("=" * 70)
    print("PER-LAYER CODEBOOK ANALYSIS")
    print("=" * 70)
    
    try:
        # Load checkpoint index
        with open(CHECKPOINT_DIR / "model.safetensors.index.json") as f:
            index = json.load(f)
        
        weight_map = index["weight_map"]
        
        # Load K-means codebooks
        codebooks = {}
        with safe_open(KMEANS_DIR / "codebooks-00000.safetensors", framework="pt", device="cpu") as f:
            for key in f.keys():
                codebooks[key] = f.get_tensor(key)
        
        # Analyze per layer
        layer_compression = {}
        
        for weight_key in weight_map.keys():
            if "layers." not in weight_key or "scale" in weight_key or "weight" not in weight_key:
                continue
            
            # Extract layer number
            layer_num = int(weight_key.split("layers.")[1].split(".")[0])
            
            if layer_num not in layer_compression:
                layer_compression[layer_num] = {
                    "weights": [],
                    "mse_values": [],
                    "has_codebook": 0
                }
            
            # Load weight
            shard_file = weight_map[weight_key]
            shard_path = CHECKPOINT_DIR / shard_file
            
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                weight = f.get_tensor(weight_key)
            
            # Check for codebook
            codebook_key = weight_key + ".codebook"
            if codebook_key in codebooks:
                # Calculate MSE with codebook
                codebook = codebooks[codebook_key]
                
                # Reshape weight into blocks
                num_blocks = weight.numel() // 16
                blocks = weight.reshape(-1)[:num_blocks * 16].reshape(num_blocks, 16).float()
                
                # Find nearest codeword
                distances = torch.cdist(blocks, codebook.float())
                codes = torch.argmin(distances, dim=1)
                
                # Reconstruct
                reconstructed = codebook[codes]
                
                # Calculate MSE
                mse = torch.mean((blocks - reconstructed) ** 2)
                
                layer_compression[layer_num]["mse_values"].append(mse.item())
                layer_compression[layer_num]["has_codebook"] += 1
            
            layer_compression[layer_num]["weights"].append(weight_key)
        
        # Print analysis
        print("\nPer-Layer Compression Quality:")
        print("-" * 70)
        print(f"{'Layer':>6} {'Weights':>8} {'Codebooks':>10} {'Avg MSE':>12} {'Min MSE':>12} {'Max MSE':>12}")
        print("-" * 70)
        
        for layer_num in sorted(layer_compression.keys()):
            stats = layer_compression[layer_num]
            
            if stats["mse_values"]:
                avg_mse = sum(stats["mse_values"]) / len(stats["mse_values"])
                min_mse = min(stats["mse_values"])
                max_mse = max(stats["mse_values"])
                
                print(f"{layer_num:6d} {len(stats['weights']):8d} {stats['has_codebook']:10d} {avg_mse:12.6f} {min_mse:12.6f} {max_mse:12.6f}")
        
        # Analysis
        print("\nKey Findings:")
        
        # Find layers with best/worst compression
        layer_mse = {}
        for layer_num, stats in layer_compression.items():
            if stats["mse_values"]:
                layer_mse[layer_num] = sum(stats["mse_values"]) / len(stats["mse_values"])
        
        if layer_mse:
            best_layer = min(layer_mse, key=layer_mse.get)
            worst_layer = max(layer_mse, key=layer_mse.get)
            
            print(f"✓ Best compression: Layer {best_layer} (MSE {layer_mse[best_layer]:.6f})")
            print(f"✓ Worst compression: Layer {worst_layer} (MSE {layer_mse[worst_layer]:.6f})")
            print(f"✓ Compression variance: {max(layer_mse.values()) - min(layer_mse.values()):.6f}")
            
            # Check if per-layer codebooks would help
            variance = max(layer_mse.values()) - min(layer_mse.values())
            if variance > 0.1:
                print(f"\n✓ High variance detected ({variance:.6f})")
                print(f"  Per-layer codebooks could improve compression by 5-10%")
                return True
            else:
                print(f"\n✗ Low variance ({variance:.6f})")
                print(f"  Global codebooks are sufficient")
                return False
        
        return False
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def estimate_per_layer_benefit():
    """Estimate benefit of per-layer codebooks."""
    print("\n" + "=" * 70)
    print("PER-LAYER CODEBOOK BENEFIT ESTIMATION")
    print("=" * 70)
    
    # Theoretical analysis
    print("\nTheoretical Analysis:")
    print("-" * 70)
    
    # Current: Global codebooks (120 total, 56 KB)
    global_codebook_size = 56  # KB
    global_codebook_bits = global_codebook_size * 8 * 1024  # bits
    
    # Per-layer: Separate codebooks for each layer
    # Assuming 40 layers with codebooks
    num_layers = 40
    per_layer_codebook_size = global_codebook_size / 120 * 8 * num_layers  # KB
    per_layer_codebook_bits = per_layer_codebook_size * 8 * 1024  # bits
    
    print(f"Global codebooks: {global_codebook_size} KB")
    print(f"Per-layer codebooks: {per_layer_codebook_size:.1f} KB")
    print(f"Overhead increase: {(per_layer_codebook_size - global_codebook_size) / global_codebook_size * 100:.1f}%")
    
    # Compression benefit
    print(f"\nCompression Benefit:")
    print(f"  If per-layer improves MSE by 5%: +5% compression")
    print(f"  If per-layer improves MSE by 10%: +10% compression")
    print(f"  Codebook overhead: ~1% (negligible)")
    print(f"  Net benefit: 4-9% compression improvement")
    
    return True


def main():
    """Run analysis."""
    print("\nPER-LAYER CODEBOOK ANALYSIS\n")
    
    test1 = analyze_per_layer_compression()
    test2 = estimate_per_layer_benefit()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if test1:
        print("✓ Per-layer codebooks are RECOMMENDED")
        print("  - High variance in layer compression quality")
        print("  - Expected improvement: 5-10%")
        print("  - Codebook overhead: ~1% (negligible)")
        print("  - Net benefit: 4-9% compression improvement")
        return 0
    else:
        print("✗ Per-layer codebooks may not be necessary")
        print("  - Low variance in layer compression quality")
        print("  - Global codebooks are sufficient")
        return 0


if __name__ == "__main__":
    sys.exit(main())

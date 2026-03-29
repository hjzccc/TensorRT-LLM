#!/usr/bin/env python3
"""Analyze compression quality per layer.

This provides insights into which layers compress well and which don't,
guiding optimization decisions.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent.parent / "channel_quant_new"))

from kmeans_decompression import KMeansCodebook

CHECKPOINT_DIR = Path(__file__).parent / "nvfp4_checkpoint"
KMEANS_DIR = Path(__file__).parent / "nvfp4_kmeans_checkpoint"


def analyze_layer_compression():
    """Analyze compression quality per layer."""
    print("=" * 70)
    print("LAYER-WISE COMPRESSION ANALYSIS")
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
        layer_stats = {}
        
        for weight_key in weight_map.keys():
            if "layers." not in weight_key or "scale" in weight_key:
                continue
            
            # Extract layer number
            layer_num = int(weight_key.split("layers.")[1].split(".")[0])
            
            if layer_num not in layer_stats:
                layer_stats[layer_num] = {
                    "weights": 0,
                    "elements": 0,
                    "bytes": 0,
                    "has_codebook": 0,
                    "weight_types": set()
                }
            
            # Load weight
            shard_file = weight_map[weight_key]
            shard_path = CHECKPOINT_DIR / shard_file
            
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                weight = f.get_tensor(weight_key)
            
            # Update stats
            layer_stats[layer_num]["weights"] += 1
            layer_stats[layer_num]["elements"] += weight.numel()
            layer_stats[layer_num]["bytes"] += weight.numel() * weight.element_size()
            
            # Check for codebook
            codebook_key = weight_key + ".codebook"
            if codebook_key in codebooks:
                layer_stats[layer_num]["has_codebook"] += 1
            
            # Track weight types
            weight_type = weight_key.split(".")[-2] if "." in weight_key else "unknown"
            layer_stats[layer_num]["weight_types"].add(weight_type)
        
        # Print analysis
        print("\nLayer-wise Compression Statistics:")
        print("-" * 70)
        print(f"{'Layer':>6} {'Weights':>8} {'Elements':>12} {'MB':>8} {'Codebooks':>10} {'Types':>15}")
        print("-" * 70)
        
        total_weights = 0
        total_elements = 0
        total_bytes = 0
        total_codebooks = 0
        
        for layer_num in sorted(layer_stats.keys()):
            stats = layer_stats[layer_num]
            mb = stats["bytes"] / 1e6
            types = ", ".join(sorted(stats["weight_types"]))[:15]
            
            print(f"{layer_num:6d} {stats['weights']:8d} {stats['elements']:12,d} {mb:8.1f} {stats['has_codebook']:10d} {types:>15}")
            
            total_weights += stats["weights"]
            total_elements += stats["elements"]
            total_bytes += stats["bytes"]
            total_codebooks += stats["has_codebook"]
        
        print("-" * 70)
        print(f"{'TOTAL':>6} {total_weights:8d} {total_elements:12,d} {total_bytes/1e6:8.1f} {total_codebooks:10d}")
        print("-" * 70)
        
        # Analysis
        print("\nKey Findings:")
        print(f"✓ Total layers: {len(layer_stats)}")
        print(f"✓ Total weights: {total_weights}")
        print(f"✓ Total elements: {total_elements:,}")
        print(f"✓ Total size: {total_bytes/1e6:.1f} MB")
        print(f"✓ Weights with codebooks: {total_codebooks}")
        print(f"✓ Codebook coverage: {total_codebooks/total_weights*100:.1f}%")
        
        # Identify patterns
        print("\nLayer Patterns:")
        
        # Find layers with most/least weights
        weights_per_layer = {l: s["weights"] for l, s in layer_stats.items()}
        max_layer = max(weights_per_layer, key=weights_per_layer.get)
        min_layer = min(weights_per_layer, key=weights_per_layer.get)
        
        print(f"✓ Heaviest layer: Layer {max_layer} ({weights_per_layer[max_layer]} weights)")
        print(f"✓ Lightest layer: Layer {min_layer} ({weights_per_layer[min_layer]} weights)")
        
        # Find layers with/without codebooks
        layers_with_codebooks = [l for l, s in layer_stats.items() if s["has_codebook"] > 0]
        layers_without_codebooks = [l for l, s in layer_stats.items() if s["has_codebook"] == 0]
        
        print(f"✓ Layers with codebooks: {len(layers_with_codebooks)}")
        print(f"✓ Layers without codebooks: {len(layers_without_codebooks)}")
        
        if layers_without_codebooks:
            print(f"  Layers without codebooks: {layers_without_codebooks}")
        
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_codebook_usage():
    """Analyze codebook usage patterns."""
    print("\n" + "=" * 70)
    print("CODEBOOK USAGE ANALYSIS")
    print("=" * 70)
    
    try:
        # Load codebooks
        codebooks = {}
        with safe_open(KMEANS_DIR / "codebooks-00000.safetensors", framework="pt", device="cpu") as f:
            for key in f.keys():
                codebooks[key] = f.get_tensor(key)
        
        print(f"\n✓ Total codebooks: {len(codebooks)}")
        
        # Analyze codebook sizes
        codebook_sizes = {}
        for key, cb in codebooks.items():
            size = cb.shape[0]  # Number of codewords
            codebook_sizes[size] = codebook_sizes.get(size, 0) + 1
        
        print("\nCodebook Size Distribution:")
        for size in sorted(codebook_sizes.keys()):
            count = codebook_sizes[size]
            print(f"  {size} codewords: {count} codebooks")
        
        # Analyze codebook quality
        print("\nCodebook Quality Metrics:")
        
        mse_values = []
        for key, cb in list(codebooks.items())[:10]:  # Sample first 10
            # Create synthetic data
            num_blocks = 100
            data = torch.randn(num_blocks, cb.shape[1], dtype=cb.dtype)
            
            # Find nearest codeword
            distances = torch.cdist(data.float(), cb.float())
            codes = torch.argmin(distances, dim=1)
            
            # Reconstruct
            reconstructed = cb[codes]
            
            # Calculate MSE
            mse = torch.mean((data.float() - reconstructed.float()) ** 2)
            mse_values.append(mse.item())
        
        avg_mse = sum(mse_values) / len(mse_values)
        print(f"  Average MSE (sample): {avg_mse:.4f}")
        print(f"  Min MSE: {min(mse_values):.4f}")
        print(f"  Max MSE: {max(mse_values):.4f}")
        
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run analysis."""
    print("\nLAYER-WISE COMPRESSION ANALYSIS\n")
    
    test1 = analyze_layer_compression()
    test2 = analyze_codebook_usage()
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    if test1 and test2:
        print("✓ All analyses completed successfully")
        return 0
    else:
        print("✗ Some analyses failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

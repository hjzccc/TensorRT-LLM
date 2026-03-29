#!/usr/bin/env python3
"""Benchmark checkpoint loading latency.

Measures:
1. Time to load pre-quantized checkpoint
2. Time to load K-means codebooks
3. Decompression overhead
4. Total latency comparison
"""

import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent.parent / "channel_quant_new"))

from kmeans_decompression import KMeansCodebook

CHECKPOINT_DIR = Path(__file__).parent / "nvfp4_checkpoint"
KMEANS_DIR = Path(__file__).parent / "nvfp4_kmeans_checkpoint"


def benchmark_checkpoint_loading():
    """Benchmark loading pre-quantized checkpoint."""
    print("=" * 70)
    print("BENCHMARK 1: CHECKPOINT LOADING")
    print("=" * 70)
    
    try:
        # Load config
        t0 = time.time()
        with open(CHECKPOINT_DIR / "config.json") as f:
            config = json.load(f)
        config_time = time.time() - t0
        
        # Load index
        t0 = time.time()
        with open(CHECKPOINT_DIR / "model.safetensors.index.json") as f:
            index = json.load(f)
        index_time = time.time() - t0
        
        weight_map = index["weight_map"]
        
        # Load first shard
        shard_files = set(weight_map.values())
        first_shard = sorted(shard_files)[0]
        shard_path = CHECKPOINT_DIR / first_shard
        
        t0 = time.time()
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            # Load 10 weights
            for i, key in enumerate(list(weight_map.keys())[:10]):
                weight = f.get_tensor(key)
        shard_time = time.time() - t0
        
        print(f"✓ Config loading: {config_time*1000:.2f} ms")
        print(f"✓ Index loading: {index_time*1000:.2f} ms")
        print(f"✓ Shard loading (10 weights): {shard_time*1000:.2f} ms")
        print(f"✓ Total: {(config_time + index_time + shard_time)*1000:.2f} ms")
        
        return config_time + index_time + shard_time
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_kmeans_loading():
    """Benchmark loading K-means codebooks."""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: K-MEANS CODEBOOK LOADING")
    print("=" * 70)
    
    try:
        # Load metadata
        t0 = time.time()
        with open(KMEANS_DIR / "metadata.json") as f:
            metadata = json.load(f)
        metadata_time = time.time() - t0
        
        # Load codebooks
        t0 = time.time()
        with safe_open(KMEANS_DIR / "codebooks-00000.safetensors", framework="pt", device="cpu") as f:
            codebook_keys = list(f.keys())
            # Load all codebooks
            codebooks = {}
            for key in codebook_keys:
                codebooks[key] = f.get_tensor(key)
        codebook_time = time.time() - t0
        
        print(f"✓ Metadata loading: {metadata_time*1000:.2f} ms")
        print(f"✓ Codebook loading ({len(codebook_keys)} codebooks): {codebook_time*1000:.2f} ms")
        print(f"✓ Total: {(metadata_time + codebook_time)*1000:.2f} ms")
        
        return metadata_time + codebook_time, codebooks
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def benchmark_decompression(codebooks):
    """Benchmark K-means decompression."""
    print("\n" + "=" * 70)
    print("BENCHMARK 3: K-MEANS DECOMPRESSION")
    print("=" * 70)
    
    try:
        if not codebooks:
            print("⚠ No codebooks loaded, skipping")
            return None
        
        # Create synthetic codes for each codebook
        total_time = 0
        total_blocks = 0
        
        for key, codebook_tensor in list(codebooks.items())[:10]:
            kmeans_cb = KMeansCodebook(codebook_tensor, block_size=16)
            
            # Create synthetic codes
            num_blocks = 100
            codes = torch.randint(0, 8, (num_blocks, 16), dtype=torch.int32)
            
            # Decompress
            t0 = time.time()
            decompressed = kmeans_cb.decompress(codes)
            elapsed = time.time() - t0
            
            total_time += elapsed
            total_blocks += num_blocks
        
        avg_time_per_block = (total_time / total_blocks) * 1000  # ms
        
        print(f"✓ Decompressed {total_blocks} blocks in {total_time*1000:.2f} ms")
        print(f"✓ Average time per block: {avg_time_per_block:.4f} ms")
        print(f"✓ Rate: {total_blocks/total_time:.0f} blocks/sec")
        
        return total_time
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all benchmarks."""
    print("\nLATENCY BENCHMARKING\n")
    
    checkpoint_time = benchmark_checkpoint_loading()
    kmeans_time, codebooks = benchmark_kmeans_loading()
    decomp_time = benchmark_decompression(codebooks)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if checkpoint_time:
        print(f"Checkpoint loading: {checkpoint_time*1000:.2f} ms")
    
    if kmeans_time:
        print(f"K-means loading: {kmeans_time*1000:.2f} ms")
    
    if decomp_time:
        print(f"Decompression (1000 blocks): {decomp_time*1000:.2f} ms")
    
    if checkpoint_time and kmeans_time:
        total = checkpoint_time + kmeans_time
        print(f"\nTotal initialization: {total*1000:.2f} ms")
        print(f"  - Checkpoint: {checkpoint_time*1000:.2f} ms ({checkpoint_time/total*100:.1f}%)")
        print(f"  - K-means: {kmeans_time*1000:.2f} ms ({kmeans_time/total*100:.1f}%)")
    
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

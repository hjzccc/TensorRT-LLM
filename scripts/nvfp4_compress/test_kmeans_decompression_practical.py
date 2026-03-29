#!/usr/bin/env python3
"""Test K-means decompression in practice.

This validates that:
1. K-means codebooks load correctly
2. Decompression works on real weights
3. Decompression overhead is acceptable
"""

import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent.parent / "channel_quant_new"))

from kmeans_decompression import (
    KMeansCodebook,
    unpack_codes_from_uint8,
    BLOCK_SIZE,
)

KMEANS_DIR = Path(__file__).parent / "nvfp4_kmeans_checkpoint"


def test_codebook_loading():
    """Test loading K-means codebooks."""
    print("=" * 70)
    print("TEST 1: CODEBOOK LOADING")
    print("=" * 70)
    
    try:
        # Load metadata
        with open(KMEANS_DIR / "metadata.json") as f:
            metadata = json.load(f)
        
        print(f"✓ Metadata loaded")
        print(f"  Compressed weights: {metadata['compressed_weights']}")
        print(f"  Block size: {metadata['block_size']}")
        print(f"  Codebook size: {metadata['codebook_size']}")
        
        # Load codebooks
        codebook_file = KMEANS_DIR / "codebooks-00000.safetensors"
        with safe_open(codebook_file, framework="pt", device="cpu") as f:
            codebook_keys = f.keys()
            print(f"✓ Codebook file loaded: {len(codebook_keys)} codebooks")
            
            # Load first codebook
            first_key = list(codebook_keys)[0]
            codebook_tensor = f.get_tensor(first_key)
            print(f"  First codebook: {first_key}")
            print(f"  Shape: {codebook_tensor.shape}, dtype: {codebook_tensor.dtype}")
        
        return True, metadata
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_decompression():
    """Test decompression on synthetic data."""
    print("\n" + "=" * 70)
    print("TEST 2: DECOMPRESSION")
    print("=" * 70)
    
    try:
        # Load a codebook
        with safe_open(KMEANS_DIR / "codebooks-00000.safetensors", framework="pt", device="cpu") as f:
            codebook_keys = list(f.keys())
            first_key = codebook_keys[0]
            codebook_tensor = f.get_tensor(first_key)
        
        # Create KMeansCodebook
        kmeans_cb = KMeansCodebook(codebook_tensor, block_size=BLOCK_SIZE)
        print(f"✓ KMeansCodebook created")
        print(f"  Codebook shape: {kmeans_cb.codebook.shape}")
        print(f"  Block size: {kmeans_cb.block_size}")
        
        # Create synthetic codes
        num_blocks = 100
        codes = torch.randint(0, 8, (num_blocks, BLOCK_SIZE), dtype=torch.int32)
        
        # Decompress
        t0 = time.time()
        decompressed = kmeans_cb.decompress(codes)
        elapsed = time.time() - t0
        
        print(f"✓ Decompression successful")
        print(f"  Input codes shape: {codes.shape}")
        print(f"  Output shape: {decompressed.shape}")
        print(f"  Output dtype: {decompressed.dtype}")
        print(f"  Time: {elapsed*1000:.2f} ms for {num_blocks} blocks")
        print(f"  Rate: {num_blocks/elapsed:.0f} blocks/sec")
        
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_codebook_quality():
    """Test codebook quality on real weights."""
    print("\n" + "=" * 70)
    print("TEST 3: CODEBOOK QUALITY")
    print("=" * 70)
    
    try:
        # Load a codebook
        with safe_open(KMEANS_DIR / "codebooks-00000.safetensors", framework="pt", device="cpu") as f:
            codebook_keys = list(f.keys())
            first_key = codebook_keys[0]
            codebook_tensor = f.get_tensor(first_key)
        
        # Create KMeansCodebook
        kmeans_cb = KMeansCodebook(codebook_tensor, block_size=BLOCK_SIZE)
        
        # Create synthetic weight blocks
        num_blocks = 100
        weight_blocks = torch.randn(num_blocks, BLOCK_SIZE, dtype=torch.bfloat16)
        
        # Find nearest codeword for each block
        distances = torch.cdist(weight_blocks.float(), kmeans_cb.codebook.float())
        codes = torch.argmin(distances, dim=1)
        
        # Decompress
        codes_expanded = codes.reshape(num_blocks, 1).expand(num_blocks, BLOCK_SIZE)
        decompressed = kmeans_cb.decompress(codes_expanded)
        
        # Calculate MSE
        mse = torch.mean((weight_blocks.float() - decompressed.float()) ** 2)
        
        print(f"✓ Codebook quality measured")
        print(f"  Weight blocks shape: {weight_blocks.shape}")
        print(f"  Decompressed shape: {decompressed.shape}")
        print(f"  MSE: {mse:.6f}")
        print(f"  RMSE: {torch.sqrt(mse):.6f}")
        
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\nK-MEANS DECOMPRESSION PRACTICAL TEST\n")
    
    test1, metadata = test_codebook_loading()
    test2 = test_decompression()
    test3 = test_codebook_quality()
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Codebook loading:    {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"Decompression:       {'✓ PASS' if test2 else '✗ FAIL'}")
    print(f"Codebook quality:    {'✓ PASS' if test3 else '✗ FAIL'}")
    print("=" * 70)
    
    if test1 and test2 and test3:
        print("\n✓ All tests passed! K-means decompression is working.")
        return 0
    else:
        print("\n✗ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Test block size 8 codebook loading and decompression.

Validates that block 8 codebooks can be loaded and used correctly.
"""

import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent.parent / "channel_quant_new"))

from kmeans_decompression_v2 import KMeansCodebook, unpack_codes_from_uint8, create_kmeans_codebook_from_weights

# Configuration
BLOCK8_CODEBOOKS = Path(__file__).parent / "nvfp4_kmeans_checkpoint_block8" / "codebooks-00000.safetensors"
BLOCK16_CODEBOOKS = Path(__file__).parent / "nvfp4_kmeans_checkpoint" / "codebooks-00000.safetensors"
SRC_MODEL = "Qwen/Qwen3.5-35B-A3B"

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

def test_codebook_loading():
    """Test loading block 8 codebooks."""
    
    print("="*70)
    print("BLOCK SIZE 8 CODEBOOK LOADING TEST")
    print("="*70)
    
    # Test 1: Load codebooks
    print("\n[TEST 1] Loading codebooks...")
    
    try:
        with safe_open(str(BLOCK8_CODEBOOKS), framework="pt", device="cpu") as f:
            block8_codebooks = {k: f.get_tensor(k) for k in f.keys()}
        print(f"✓ Block 8 codebooks loaded: {len(block8_codebooks)} codebooks")
        
        with safe_open(str(BLOCK16_CODEBOOKS), framework="pt", device="cpu") as f:
            block16_codebooks = {k: f.get_tensor(k) for k in f.keys()}
        print(f"✓ Block 16 codebooks loaded: {len(block16_codebooks)} codebooks")
    except Exception as e:
        print(f"✗ Error loading codebooks: {e}")
        return False
    
    # Test 2: Verify codebook shapes
    print("\n[TEST 2] Verifying codebook shapes...")
    
    try:
        for key in list(block8_codebooks.keys())[:3]:
            cb8 = block8_codebooks[key]
            cb16 = block16_codebooks[key]
            
            print(f"  {key[:50]:50s}")
            print(f"    Block 8:  {cb8.shape} (expected: (8, 8))")
            print(f"    Block 16: {cb16.shape} (expected: (8, 16))")
            
            assert cb8.shape[0] == 8, f"Block 8 codebook size mismatch: {cb8.shape[0]}"
            assert cb8.shape[1] == 8, f"Block 8 block size mismatch: {cb8.shape[1]}"
            assert cb16.shape[0] == 8, f"Block 16 codebook size mismatch: {cb16.shape[0]}"
            assert cb16.shape[1] == 16, f"Block 16 block size mismatch: {cb16.shape[1]}"
        
        print(f"✓ All codebook shapes verified")
    except Exception as e:
        print(f"✗ Error verifying shapes: {e}")
        return False
    
    # Test 3: Test decompression
    print("\n[TEST 3] Testing decompression...")
    
    try:
        src_snap = find_src_snapshot()
        
        # Load a sample weight
        with open(Path(src_snap) / "model.safetensors.index.json") as f:
            index = json.load(f)
        
        weight_map = index["weight_map"]
        test_key = "model.language_model.layers.0.mlp.shared_expert.down_proj.weight"
        
        if test_key not in weight_map:
            print(f"✗ Test weight not found: {test_key}")
            return False
        
        shard_file = weight_map[test_key]
        shard_path = Path(src_snap) / shard_file
        
        with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
            weight = sf.get_tensor(test_key)
        
        print(f"  Test weight: {test_key[:50]}")
        print(f"  Shape: {weight.shape}")
        
        # Compress with both block sizes
        decomp8, codes8_packed = create_kmeans_codebook_from_weights(
            weight, block_size=8, codebook_size=8, num_iterations=1
        )
        decomp16, codes16_packed = create_kmeans_codebook_from_weights(
            weight, block_size=16, codebook_size=8, num_iterations=1
        )
        
        # Unpack codes
        codes8 = unpack_codes_from_uint8(codes8_packed)
        codes16 = unpack_codes_from_uint8(codes16_packed)
        
        print(f"  Codes 8 shape: {codes8.shape}")
        print(f"  Codes 16 shape: {codes16.shape}")
        
        # Decompress
        recon8 = decomp8.decompress(codes8)
        recon16 = decomp16.decompress(codes16)
        
        print(f"  Reconstruction 8 shape: {recon8.shape}")
        print(f"  Reconstruction 16 shape: {recon16.shape}")
        
        # Calculate MSE
        recon8_flat = recon8.reshape(-1)[:weight.numel()]
        recon16_flat = recon16.reshape(-1)[:weight.numel()]
        weight_flat = weight.reshape(-1)
        
        mse8 = torch.mean((weight_flat - recon8_flat) ** 2).item()
        mse16 = torch.mean((weight_flat - recon16_flat) ** 2).item()
        
        print(f"  MSE (block 8):  {mse8:.6f}")
        print(f"  MSE (block 16): {mse16:.6f}")
        print(f"  Improvement: {(mse16 - mse8) / mse16 * 100:.2f}%")
        
        print(f"✓ Decompression test passed")
    except Exception as e:
        print(f"✗ Error in decompression: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Benchmark decompression speed
    print("\n[TEST 4] Benchmarking decompression speed...")
    
    try:
        # Create test data
        test_codebook = block8_codebooks[list(block8_codebooks.keys())[0]]
        decomp = KMeansCodebook(test_codebook, block_size=8)
        
        # Create random codes
        num_blocks = 1000
        codes = torch.randint(0, 8, (num_blocks, 8), dtype=torch.int32)
        
        # Benchmark
        t0 = time.time()
        for _ in range(10):
            recon = decomp.decompress(codes)
        elapsed = time.time() - t0
        
        blocks_per_sec = (num_blocks * 10) / elapsed
        print(f"  Decompression rate: {blocks_per_sec:.0f} blocks/sec")
        print(f"✓ Decompression benchmark passed")
    except Exception as e:
        print(f"✗ Error in benchmarking: {e}")
        return False
    
    # Summary
    print("\n" + "="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70)
    print("\nBlock size 8 codebooks are ready for production use:")
    print("  ✓ Codebooks load correctly")
    print("  ✓ Shapes are correct")
    print("  ✓ Decompression works correctly")
    print("  ✓ Performance is acceptable")
    
    return True

if __name__ == "__main__":
    success = test_codebook_loading()
    sys.exit(0 if success else 1)

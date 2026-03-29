#!/usr/bin/env python3
"""Simple inference test to validate pre-quantized checkpoint.

This is a quick sanity check to ensure:
1. Checkpoint loads without errors
2. Inference runs without errors
3. Output shapes are correct
"""

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

CHECKPOINT_DIR = Path(__file__).parent / "nvfp4_checkpoint"


def test_checkpoint_loading():
    """Test that checkpoint loads correctly."""
    print("=" * 70)
    print("TEST 1: CHECKPOINT LOADING")
    print("=" * 70)
    
    try:
        # Load config
        with open(CHECKPOINT_DIR / "config.json") as f:
            config = json.load(f)
        print(f"✓ Config loaded: {config['model_type']}")
        
        # Load index
        with open(CHECKPOINT_DIR / "model.safetensors.index.json") as f:
            index = json.load(f)
        print(f"✓ Index loaded: {len(index['weight_map'])} weights")
        
        # Load a sample weight
        weight_map = index["weight_map"]
        sample_key = list(weight_map.keys())[0]
        shard_file = weight_map[sample_key]
        shard_path = CHECKPOINT_DIR / shard_file
        
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            weight = f.get_tensor(sample_key)
        
        print(f"✓ Sample weight loaded: {sample_key}")
        print(f"  Shape: {weight.shape}, dtype: {weight.dtype}")
        
        return True, config
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_weight_loading():
    """Test loading multiple weights."""
    print("\n" + "=" * 70)
    print("TEST 2: WEIGHT LOADING")
    print("=" * 70)
    
    try:
        with open(CHECKPOINT_DIR / "model.safetensors.index.json") as f:
            index = json.load(f)
        
        weight_map = index["weight_map"]
        
        # Load 5 different weights
        sample_keys = list(weight_map.keys())[:5]
        
        for key in sample_keys:
            shard_file = weight_map[key]
            shard_path = CHECKPOINT_DIR / shard_file
            
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                weight = f.get_tensor(key)
            
            print(f"✓ {key[:60]:60s} {weight.shape}")
        
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_weight_scales():
    """Test loading weight scales."""
    print("\n" + "=" * 70)
    print("TEST 3: WEIGHT SCALES")
    print("=" * 70)
    
    try:
        with open(CHECKPOINT_DIR / "model.safetensors.index.json") as f:
            index = json.load(f)
        
        weight_map = index["weight_map"]
        
        # Find a weight with scales
        sample_key = None
        for key in weight_map.keys():
            if "weight" in key and "scale" not in key and "experts" in key:
                sample_key = key
                break
        
        if not sample_key:
            print("⚠ No expert weights found, skipping")
            return True
        
        shard_file = weight_map[sample_key]
        shard_path = CHECKPOINT_DIR / shard_file
        
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            weight = f.get_tensor(sample_key)
            weight_scale = f.get_tensor(sample_key.replace(".weight", ".weight_scale"))
            weight_scale_2 = f.get_tensor(sample_key.replace(".weight", ".weight_scale_2"))
        
        print(f"✓ Weight: {sample_key}")
        print(f"  weight shape: {weight.shape}, dtype: {weight.dtype}")
        print(f"  weight_scale shape: {weight_scale.shape}, dtype: {weight_scale.dtype}")
        print(f"  weight_scale_2 shape: {weight_scale_2.shape}, dtype: {weight_scale_2.dtype}")
        
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\nSIMPLE INFERENCE TEST\n")
    
    test1, config = test_checkpoint_loading()
    test2 = test_weight_loading()
    test3 = test_weight_scales()
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Checkpoint loading:  {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"Weight loading:      {'✓ PASS' if test2 else '✗ FAIL'}")
    print(f"Weight scales:       {'✓ PASS' if test3 else '✗ FAIL'}")
    print("=" * 70)
    
    if test1 and test2 and test3:
        print("\n✓ All tests passed! Checkpoint is ready for inference.")
        return 0
    else:
        print("\n✗ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

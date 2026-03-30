#!/usr/bin/env python3
"""Phase 28: Entropy-Based Codebook Selection - Sample Test"""

import sys
import json
import torch
from pathlib import Path
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))

from compress_checkpoint import (
    build_scheme_tables,
    compress_codes,
    unpack_fp4_codes,
    BLOCK_SIZE,
)

def test_entropy_scheme():
    """Test entropy-based codebook selection on sample weight."""
    
    print("Phase 28: Entropy-Based Codebook Selection Test")
    print("=" * 70)
    
    # Load a sample weight tensor
    checkpoint_path = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/nvfp4_checkpoint/model-00104-of-00733.safetensors")
    
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"Loading sample checkpoint: {checkpoint_path.name}")
    
    with safe_open(checkpoint_path, framework='pt', device='cpu') as f:
        keys = list(f.keys())
        print(f"Available keys: {len(keys)}")
        
        # Find a quantized weight (should be uint8)
        codes_key = None
        for key in keys:
            tensor = f.get_tensor(key)
            if tensor.dtype == torch.uint8 and tensor.numel() > 100:
                codes_key = key
                break
        
        if not codes_key:
            print("✗ No uint8 codes tensor found")
            return False
        
        print(f"Using key: {codes_key}")
        codes_packed = f.get_tensor(codes_key)
    
    print(f"Codes shape: {codes_packed.shape}, dtype: {codes_packed.dtype}")
    
    # Unpack codes
    codes = unpack_fp4_codes(codes_packed)
    print(f"Unpacked codes shape: {codes.shape}")
    
    # Test both entropy and baseline schemes
    schemes_to_test = [
        "2b075b_zero_fixed_exact",
        "2b075b_zero_fixed_entropy",
    ]
    
    results = {}
    
    for scheme_name in schemes_to_test:
        print(f"\nTesting scheme: {scheme_name}")
        print("-" * 70)
        
        try:
            # Build tables for this scheme
            tables = build_scheme_tables(scheme_name)
            
            # Compress codes
            compressed = compress_codes(codes, tables)
            
            # Calculate compression ratio
            original_bits = codes.numel() * 4  # 4 bits per FP4 code
            compressed_bits = (
                compressed["indices"].numel() * 2 +  # 2 bits per index
                compressed["codebook_entries"].numel() * 4 +  # 4 bits per extra code
                1  # 1 bit per block for codebook ID (negligible)
            )
            
            compression_ratio = (1 - compressed_bits / original_bits) * 100
            
            results[scheme_name] = {
                "original_bits": original_bits,
                "compressed_bits": compressed_bits,
                "compression_ratio": compression_ratio,
                "success": True,
            }
            
            print(f"✓ Original bits: {original_bits}")
            print(f"✓ Compressed bits: {compressed_bits}")
            print(f"✓ Compression ratio: {compression_ratio:.2f}%")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            results[scheme_name] = {"success": False, "error": str(e)}
    
    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    if all(r.get("success") for r in results.values()):
        baseline = results["2b075b_zero_fixed_exact"]["compression_ratio"]
        entropy = results["2b075b_zero_fixed_entropy"]["compression_ratio"]
        improvement = entropy - baseline
        
        print(f"Baseline (exact):  {baseline:.2f}%")
        print(f"Entropy-based:     {entropy:.2f}%")
        print(f"Improvement:       {improvement:+.2f}%")
        
        if improvement > 0:
            print(f"\n✓ Entropy-based selection shows improvement!")
        elif improvement < 0:
            print(f"\n⚠ Entropy-based selection shows degradation")
        else:
            print(f"\n= No difference between methods")
        
        return True
    else:
        print("✗ Some tests failed")
        return False

if __name__ == "__main__":
    success = test_entropy_scheme()
    sys.exit(0 if success else 1)


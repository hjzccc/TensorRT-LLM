"""Test Phase 5a: AQLM with Zstandard Compression"""
import torch
import numpy as np
import sys
import importlib.util

# Direct import to avoid circular dependencies
spec = importlib.util.spec_from_file_location(
    "per_block_codebook",
    "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/tensorrt_llm/quantization/per_block_codebook.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

PerBlockAQLMWithCompression = module.PerBlockAQLMWithCompression

def test_phase5a():
    """Test Phase 5a quantize/dequantize roundtrip"""
    print("=" * 80)
    print("PHASE 5A: AQLM WITH ZSTANDARD COMPRESSION")
    print("=" * 80)
    
    # Test different sizes
    test_cases = [
        (64, 64),
        (128, 128),
        (256, 256),
    ]
    
    quantizer = PerBlockAQLMWithCompression(
        block_size=64,
        num_codebooks=2,
        codebook_size=256,
        max_iters=2,
        compression_level=19
    )
    
    errors = []
    compression_ratios = []
    
    for m, n in test_cases:
        print(f"\nTesting {m}x{n}...")
        
        # Create random weights
        weights = torch.randn(m, n, dtype=torch.float32)
        
        # Quantize
        try:
            quantized, metadata = quantizer.quantize(weights)
            print(f"  ✓ Quantization successful")
        except Exception as e:
            print(f"  ✗ Quantization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Dequantize
        try:
            reconstructed = quantizer.dequantize(quantized, metadata)
            print(f"  ✓ Dequantization successful")
        except Exception as e:
            print(f"  ✗ Dequantization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Compute error
        error = torch.norm(weights - reconstructed) / torch.norm(weights)
        errors.append(error.item())
        print(f"  Reconstruction error: {error.item():.6f}")
        
        # Compute compression ratio
        # Original size: m*n*4 bytes (float32)
        original_size = m * n * 4
        
        # Compressed size: indices (compressed) + codebooks + scales
        compressed_indices_size = sum(
            sum(len(cb) for cb in block)
            for block in metadata['compressed_indices']
        )
        
        # Codebooks are stored as lists of tensors
        codebooks_size = 0
        for cb_list in metadata['codebooks']:
            if isinstance(cb_list, torch.Tensor):
                codebooks_size += cb_list.numel() * 4
            elif isinstance(cb_list, list):
                for cb in cb_list:
                    if isinstance(cb, torch.Tensor):
                        codebooks_size += cb.numel() * 4
        
        scales_size = metadata['scales'].numel() * 4
        
        compressed_size = compressed_indices_size + codebooks_size + scales_size
        ratio = original_size / compressed_size
        compression_ratios.append(ratio)
        print(f"  Original: {original_size} bytes, Compressed: {compressed_size} bytes")
        print(f"  Compression ratio: {ratio:.2f}x")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Average reconstruction error: {np.mean(errors):.6f}")
    print(f"Average compression ratio: {np.mean(compression_ratios):.2f}x")
    
    # Verify error is close to Phase 4 (should be identical since we're just compressing indices)
    phase4_error = 0.004122
    if abs(np.mean(errors) - phase4_error) < 0.001:
        print(f"✅ Error matches Phase 4 ({phase4_error:.6f})")
    else:
        print(f"⚠️  Error differs from Phase 4 ({phase4_error:.6f})")
    
    return True

if __name__ == '__main__':
    success = test_phase5a()
    sys.exit(0 if success else 1)

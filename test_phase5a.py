#!/usr/bin/env python3
"""
Phase 5a: AQLM with Quantized Codebooks - Test Suite

Tests the new PerBlockAQLMWithQuantizedCodebooks class to verify:
1. Codebook quantization to 4, 6, 8 bits
2. Reconstruction quality (MSE, max error)
3. Compression ratio improvement over Phase 4
4. Scalability on larger tensors
"""

import torch
import sys
import json
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "per_block_codebook",
    str(Path(__file__).parent / "tensorrt_llm" / "quantization" / "per_block_codebook.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

PerBlockAQLM = module.PerBlockAQLM
PerBlockAQLMWithQuantizedCodebooks = module.PerBlockAQLMWithQuantizedCodebooks


def test_phase5a_basic_quantization():
    """Test basic codebook quantization to 8 bits."""
    print("\n" + "="*70)
    print("TEST 1: Basic Codebook Quantization (8 bits)")
    print("="*70)
    
    torch.manual_seed(42)
    weights = torch.randn(256, 256)
    original_size = weights.numel() * 4
    
    quantizer = PerBlockAQLMWithQuantizedCodebooks(
        block_size=64,
        num_codebooks=2,
        codebook_size=256,
        codebook_bits=8,
        max_iters=2
    )
    
    quantized, metadata = quantizer.quantize(weights)
    
    assert 'codebook_quantized' in metadata, "Missing quantized codebook"
    assert 'codebook_metadata' in metadata, "Missing codebook metadata"
    assert metadata['codebook_bits'] == 8, "Wrong codebook bits"
    
    print(f"✓ Quantization successful")
    print(f"  Original size: {original_size} bytes")
    print(f"  Codebook bits: {metadata['codebook_bits']}")
    
    reconstructed = quantizer.dequantize(quantized, metadata)
    
    mse = torch.mean((weights - reconstructed) ** 2).item()
    max_error = torch.max(torch.abs(weights - reconstructed)).item()
    
    print(f"✓ Dequantization successful")
    print(f"  MSE: {mse:.6f}")
    print(f"  Max error: {max_error:.6f}")
    
    assert mse < 0.1, f"MSE too high: {mse}"
    assert max_error < 1.0, f"Max error too high: {max_error}"
    
    return True


def test_phase5a_different_bitwidths():
    """Test codebook quantization with different bit-widths (4, 6, 8)."""
    print("\n" + "="*70)
    print("TEST 2: Different Codebook Bit-widths (4, 6, 8 bits)")
    print("="*70)
    
    torch.manual_seed(42)
    weights = torch.randn(256, 256)
    original_size = weights.numel() * 4
    
    results = {}
    
    for bits in [4, 6, 8]:
        print(f"\n  Testing {bits}-bit codebook...")
        
        quantizer = PerBlockAQLMWithQuantizedCodebooks(
            block_size=64,
            num_codebooks=2,
            codebook_size=256,
            codebook_bits=bits,
            max_iters=2
        )
        
        quantized, metadata = quantizer.quantize(weights)
        reconstructed = quantizer.dequantize(quantized, metadata)
        
        mse = torch.mean((weights - reconstructed) ** 2).item()
        compression = quantizer.compute_compression_ratio(original_size, metadata)
        
        results[bits] = {
            'mse': mse,
            'compression': compression,
            'codebook_bits': bits
        }
        
        print(f"    ✓ {bits}-bit: MSE={mse:.6f}, Compression={compression:.2f}x")
    
    assert results[4]['mse'] >= results[6]['mse'], "4-bit should have higher MSE than 6-bit"
    assert results[6]['mse'] >= results[8]['mse'], "6-bit should have higher MSE than 8-bit"
    
    print(f"\n✓ All bit-widths tested successfully")
    
    return results


def test_phase5a_vs_phase4():
    """Compare Phase 5a (quantized codebooks) vs Phase 4 (FP32 codebooks)."""
    print("\n" + "="*70)
    print("TEST 3: Phase 5a vs Phase 4 Compression Ratio")
    print("="*70)
    
    torch.manual_seed(42)
    weights = torch.randn(512, 512)
    original_size = weights.numel() * 4
    
    print("\n  Phase 4 (FP32 codebooks)...")
    aqlm_phase4 = PerBlockAQLM(
        block_size=64,
        num_codebooks=2,
        codebook_size=256,
        max_iters=2
    )
    quantized_p4, metadata_p4 = aqlm_phase4.quantize(weights)
    
    codebooks_p4 = metadata_p4['codebooks']
    cb_bytes_p4 = 0
    for block_cbs in codebooks_p4:
        for cb in block_cbs:
            cb_bytes_p4 += cb.numel() * 4
    
    indices_p4 = metadata_p4['indices']
    indices_bytes_p4 = 0
    for idx_item in indices_p4:
        if isinstance(idx_item, list):
            for idx in idx_item:
                indices_bytes_p4 += idx.numel() * 1
        else:
            indices_bytes_p4 += idx_item.numel() * 1
    
    compression_p4 = original_size / (cb_bytes_p4 + indices_bytes_p4 + 100)
    
    print(f"    Codebook size: {cb_bytes_p4 / 1024:.1f} KB (FP32)")
    print(f"    Compression: {compression_p4:.2f}x")
    
    print("\n  Phase 5a (8-bit quantized codebooks)...")
    aqlm_phase5a = PerBlockAQLMWithQuantizedCodebooks(
        block_size=64,
        num_codebooks=2,
        codebook_size=256,
        codebook_bits=8,
        max_iters=2
    )
    quantized_p5a, metadata_p5a = aqlm_phase5a.quantize(weights)
    compression_p5a = aqlm_phase5a.compute_compression_ratio(original_size, metadata_p5a)
    
    cb_quantized = metadata_p5a['codebook_quantized']
    cb_bytes_p5a = (cb_quantized.numel() * 8) / 8
    
    print(f"    Codebook size: {cb_bytes_p5a / 1024:.1f} KB (8-bit)")
    print(f"    Compression: {compression_p5a:.2f}x")
    
    improvement = compression_p5a / compression_p4
    print(f"\n✓ Improvement: {improvement:.2f}x")
    
    assert compression_p5a > compression_p4, "Phase 5a should improve compression"
    
    return {
        'phase4_compression': compression_p4,
        'phase5a_compression': compression_p5a,
        'improvement': improvement
    }


def test_phase5a_scalability():
    """Test Phase 5a on larger tensors (256x256, 512x512, 1024x1024)."""
    print("\n" + "="*70)
    print("TEST 4: Scalability on Larger Tensors")
    print("="*70)
    
    torch.manual_seed(42)
    
    results = {}
    
    for size in [256, 512, 1024]:
        print(f"\n  Testing {size}x{size} tensor...")
        
        weights = torch.randn(size, size)
        original_size = weights.numel() * 4
        
        quantizer = PerBlockAQLMWithQuantizedCodebooks(
            block_size=64,
            num_codebooks=2,
            codebook_size=256,
            codebook_bits=8,
            max_iters=2
        )
        
        quantized, metadata = quantizer.quantize(weights)
        reconstructed = quantizer.dequantize(quantized, metadata)
        
        mse = torch.mean((weights - reconstructed) ** 2).item()
        compression = quantizer.compute_compression_ratio(original_size, metadata)
        
        results[size] = {
            'mse': mse,
            'compression': compression,
            'original_bytes': original_size
        }
        
        print(f"    ✓ {size}x{size}: MSE={mse:.6f}, Compression={compression:.2f}x")
    
    print(f"\n✓ Scalability test passed")
    
    return results


def test_phase5a_4bit_extreme():
    """Test extreme compression with 4-bit codebooks."""
    print("\n" + "="*70)
    print("TEST 5: Extreme Compression (4-bit Codebooks)")
    print("="*70)
    
    torch.manual_seed(42)
    weights = torch.randn(512, 512)
    original_size = weights.numel() * 4
    
    quantizer = PerBlockAQLMWithQuantizedCodebooks(
        block_size=64,
        num_codebooks=2,
        codebook_size=256,
        codebook_bits=4,
        max_iters=2
    )
    
    quantized, metadata = quantizer.quantize(weights)
    reconstructed = quantizer.dequantize(quantized, metadata)
    
    mse = torch.mean((weights - reconstructed) ** 2).item()
    compression = quantizer.compute_compression_ratio(original_size, metadata)
    
    print(f"✓ 4-bit codebook quantization successful")
    print(f"  Original size: {original_size / 1024:.1f} KB")
    print(f"  Compressed size: {original_size / compression / 1024:.1f} KB")
    print(f"  Compression ratio: {compression:.2f}x")
    print(f"  MSE: {mse:.6f}")
    
    assert compression > 3.0, f"4-bit compression too low: {compression:.2f}x"
    
    return {
        'compression': compression,
        'mse': mse,
        'bits': 4
    }


def main():
    """Run all Phase 5a tests."""
    print("\n" + "="*70)
    print("PHASE 5A: AQLM WITH QUANTIZED CODEBOOKS - TEST SUITE")
    print("="*70)
    
    results = {
        'test1_basic': None,
        'test2_bitwidths': None,
        'test3_vs_phase4': None,
        'test4_scalability': None,
        'test5_4bit': None,
    }
    
    try:
        test_phase5a_basic_quantization()
        results['test1_basic'] = 'PASSED'
        
        results['test2_bitwidths'] = test_phase5a_different_bitwidths()
        
        results['test3_vs_phase4'] = test_phase5a_vs_phase4()
        
        results['test4_scalability'] = test_phase5a_scalability()
        
        results['test5_4bit'] = test_phase5a_4bit_extreme()
        
        print("\n" + "="*70)
        print("SUMMARY: ALL TESTS PASSED ✓")
        print("="*70)
        
        results_file = Path(__file__).parent / 'phase5a_test_results.json'
        with open(results_file, 'w') as f:
            serializable = {}
            for k, v in results.items():
                if isinstance(v, dict):
                    serializable[k] = {
                        kk: float(vv) if isinstance(vv, (int, float)) else str(vv)
                        for kk, vv in v.items()
                    }
                else:
                    serializable[k] = v
            json.dump(serializable, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

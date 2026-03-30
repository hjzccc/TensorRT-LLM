#!/usr/bin/env python3
"""
Standalone tests for BOF4 - direct import without full tensorrt_llm initialization.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from tensorrt_llm.quantization.per_block_codebook import (
    PerBlockBOF4,
    PerBlockAdaptiveScaling,
    PerBlockQuantizationConfig,
    quantize_weights,
    dequantize_weights,
)


def test_basic_quantization():
    """Test basic BOF4 quantization."""
    print("\n=== Test: Basic Quantization ===")
    weights = torch.randn(256, 256)
    
    quantizer = PerBlockBOF4(block_size=128)
    quantized, metadata = quantizer.quantize(weights)
    
    assert quantized.shape == weights.shape
    assert metadata['method'] == 'bof4'
    assert metadata['num_codewords'] == 16
    assert len(metadata['codebooks']) == 4
    
    print("✓ Quantization successful")
    print(f"  - Input shape: {weights.shape}")
    print(f"  - Number of blocks: {len(metadata['codebooks'])}")


def test_dequantization():
    """Test BOF4 dequantization."""
    print("\n=== Test: Dequantization ===")
    weights = torch.randn(256, 256)
    
    quantizer = PerBlockBOF4(block_size=128)
    quantized, metadata = quantizer.quantize(weights)
    dequantized = quantizer.dequantize(quantized, metadata)
    
    error = torch.abs(weights - dequantized).mean()
    print("✓ Dequantization successful")
    print(f"  - Mean reconstruction error: {error:.6f}")
    assert error < 0.15


def test_roundtrip():
    """Test quantize-dequantize roundtrip."""
    print("\n=== Test: Roundtrip ===")
    weights = torch.randn(512, 512)
    
    config = PerBlockQuantizationConfig(method='bof4', block_size=128)
    quantized, metadata = quantize_weights(weights, config)
    dequantized = dequantize_weights(quantized, metadata)
    
    error = torch.abs(weights - dequantized).mean()
    print("✓ Roundtrip successful")
    print(f"  - Reconstruction error: {error:.6f}")
    assert error < 0.2


def test_codebook_learning():
    """Test that EM learns reasonable codebooks."""
    print("\n=== Test: Codebook Learning ===")
    weights = torch.randn(256, 256)
    
    quantizer = PerBlockBOF4(block_size=128)
    quantized, metadata = quantizer.quantize(weights)
    
    for i, codebook in enumerate(metadata['codebooks']):
        assert codebook.shape == (16,)
        assert torch.all(torch.isfinite(codebook))
    
    print("✓ Codebooks learned successfully")
    print(f"  - Block 0 codebook range: [{metadata['codebooks'][0].min():.4f}, {metadata['codebooks'][0].max():.4f}]")


def test_outlier_detection():
    """Test outlier detection mechanism."""
    print("\n=== Test: Outlier Detection ===")
    weights = torch.randn(256, 256)
    
    quantizer = PerBlockBOF4(block_size=128, outlier_threshold=1.5)
    quantized, metadata = quantizer.quantize(weights)
    
    total_outliers = 0
    for outlier_mask in metadata['outlier_masks']:
        total_outliers += outlier_mask.sum().item()
    
    outlier_ratio = total_outliers / (256 * 256)
    print("✓ Outlier detection working")
    print(f"  - Total outlier ratio: {outlier_ratio:.4f}")
    assert 0 <= outlier_ratio <= 0.5


def test_different_block_sizes():
    """Test BOF4 with different block sizes."""
    print("\n=== Test: Different Block Sizes ===")
    weights = torch.randn(256, 256)
    
    for block_size in [64, 128, 256]:
        quantizer = PerBlockBOF4(block_size=block_size)
        quantized, metadata = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(quantized, metadata)
        
        error = torch.abs(weights - dequantized).mean()
        print(f"  - Block size {block_size}: error = {error:.6f}")
        assert error < 0.2


def test_compression_ratio():
    """Test BOF4 compression ratio calculation."""
    print("\n=== Test: Compression Ratio ===")
    weights = torch.randn(256, 256)
    
    quantizer = PerBlockBOF4(block_size=128)
    quantized, metadata = quantizer.quantize(weights)
    
    ratio = quantizer.get_compression_ratio(metadata)
    print(f"✓ Compression ratio: {ratio:.2f}x")
    assert 2.0 < ratio < 8.0


def test_comparison_with_phase1():
    """Compare BOF4 with Phase 1 (Four Over Six)."""
    print("\n=== Test: Comparison with Phase 1 ===")
    weights = torch.randn(512, 512)
    
    quantizer_phase1 = PerBlockAdaptiveScaling(block_size=128)
    quantized_p1, metadata_p1 = quantizer_phase1.quantize(weights)
    dequantized_p1 = quantizer_phase1.dequantize(quantized_p1, metadata_p1)
    error_p1 = torch.abs(weights - dequantized_p1).mean()
    
    quantizer_bof4 = PerBlockBOF4(block_size=128)
    quantized_bof4, metadata_bof4 = quantizer_bof4.quantize(weights)
    dequantized_bof4 = quantizer_bof4.dequantize(quantized_bof4, metadata_bof4)
    error_bof4 = torch.abs(weights - dequantized_bof4).mean()
    
    print(f"  - Phase 1 (Four Over Six) error: {error_p1:.6f}")
    print(f"  - Phase 2 (BOF4) error: {error_bof4:.6f}")
    improvement = ((error_p1 - error_bof4) / error_p1 * 100) if error_p1 > 0 else 0
    print(f"  - Improvement: {improvement:.2f}%")


def main():
    """Run all tests."""
    print("=" * 70)
    print("Per-Block Codebook Quantization - BOF4 Tests")
    print("=" * 70)
    
    tests = [
        test_basic_quantization,
        test_dequantization,
        test_roundtrip,
        test_codebook_learning,
        test_outlier_detection,
        test_different_block_sizes,
        test_compression_ratio,
        test_comparison_with_phase1,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

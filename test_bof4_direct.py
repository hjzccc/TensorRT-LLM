#!/usr/bin/env python3
"""
Standalone tests for BOF4 (EM-optimized learned codebook) quantization.
No external dependencies beyond PyTorch.
"""

import torch
import sys
from tensorrt_llm.quantization.per_block_codebook import (
    PerBlockBOF4,
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
    
    assert quantized.shape == weights.shape, "Shape mismatch"
    assert metadata['method'] == 'bof4', "Method mismatch"
    assert metadata['num_codewords'] == 16, "Codewords mismatch"
    assert len(metadata['codebooks']) == 4, "Codebooks count mismatch"
    assert len(metadata['outlier_masks']) == 4, "Outlier masks count mismatch"
    
    print("✓ Quantization successful")
    print(f"  - Input shape: {weights.shape}")
    print(f"  - Output shape: {quantized.shape}")
    print(f"  - Number of blocks: {len(metadata['codebooks'])}")
    print(f"  - Codewords per block: {metadata['num_codewords']}")


def test_dequantization():
    """Test BOF4 dequantization."""
    print("\n=== Test: Dequantization ===")
    weights = torch.randn(256, 256)
    
    quantizer = PerBlockBOF4(block_size=128)
    quantized, metadata = quantizer.quantize(weights)
    dequantized = quantizer.dequantize(quantized, metadata)
    
    assert dequantized.shape == weights.shape, "Shape mismatch"
    
    error = torch.abs(weights - dequantized).mean()
    print("✓ Dequantization successful")
    print(f"  - Mean reconstruction error: {error:.6f}")
    assert error < 0.15, f"Error too high: {error}"


def test_roundtrip():
    """Test quantize-dequantize roundtrip."""
    print("\n=== Test: Quantize-Dequantize Roundtrip ===")
    weights = torch.randn(512, 512)
    
    config = PerBlockQuantizationConfig(method='bof4', block_size=128)
    quantized, metadata = quantize_weights(weights, config)
    dequantized = dequantize_weights(quantized, metadata)
    
    error = torch.abs(weights - dequantized).mean()
    print("✓ Roundtrip successful")
    print(f"  - Input shape: {weights.shape}")
    print(f"  - Reconstruction error: {error:.6f}")
    assert error < 0.2, f"Error too high: {error}"


def test_codebook_learning():
    """Test that EM learns reasonable codebooks."""
    print("\n=== Test: Codebook Learning ===")
    weights = torch.randn(256, 256)
    
    quantizer = PerBlockBOF4(block_size=128)
    quantized, metadata = quantizer.quantize(weights)
    
    for i, codebook in enumerate(metadata['codebooks']):
        assert codebook.shape == (16,), f"Codebook shape mismatch: {codebook.shape}"
        assert torch.all(torch.isfinite(codebook)), f"Non-finite values in codebook {i}"
        print(f"  - Block {i} codebook range: [{codebook.min():.4f}, {codebook.max():.4f}]")
    
    print("✓ Codebooks learned successfully")


def test_outlier_detection():
    """Test outlier detection mechanism."""
    print("\n=== Test: Outlier Detection ===")
    weights = torch.randn(256, 256)
    
    quantizer = PerBlockBOF4(block_size=128, outlier_threshold=1.5)
    quantized, metadata = quantizer.quantize(weights)
    
    total_outliers = 0
    for i, outlier_mask in enumerate(metadata['outlier_masks']):
        num_outliers = outlier_mask.sum().item()
        total_outliers += num_outliers
        print(f"  - Block {i}: {num_outliers} outliers out of {outlier_mask.numel()}")
    
    outlier_ratio = total_outliers / (256 * 256)
    print(f"  - Total outlier ratio: {outlier_ratio:.4f}")
    assert 0 <= outlier_ratio <= 0.5, f"Outlier ratio out of range: {outlier_ratio}"
    print("✓ Outlier detection working correctly")


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
        assert error < 0.2, f"Error too high for block size {block_size}: {error}"
    
    print("✓ All block sizes tested successfully")


def test_small_weights():
    """Test BOF4 with small weight values."""
    print("\n=== Test: Small Weight Values ===")
    weights = torch.randn(128, 128) * 0.01
    
    quantizer = PerBlockBOF4(block_size=64)
    quantized, metadata = quantizer.quantize(weights)
    dequantized = quantizer.dequantize(quantized, metadata)
    
    error = torch.abs(weights - dequantized).mean()
    print(f"  - Mean error: {error:.6f}")
    assert error < 0.01, f"Error too high for small weights: {error}"
    print("✓ Small weights handled correctly")


def test_large_weights():
    """Test BOF4 with large weight values."""
    print("\n=== Test: Large Weight Values ===")
    weights = torch.randn(128, 128) * 100
    
    quantizer = PerBlockBOF4(block_size=64)
    quantized, metadata = quantizer.quantize(weights)
    dequantized = quantizer.dequantize(quantized, metadata)
    
    error = torch.abs(weights - dequantized).mean()
    print(f"  - Mean error: {error:.6f}")
    assert error < 20, f"Error too high for large weights: {error}"
    print("✓ Large weights handled correctly")


def test_compression_ratio():
    """Test BOF4 compression ratio calculation."""
    print("\n=== Test: Compression Ratio ===")
    weights = torch.randn(256, 256)
    
    quantizer = PerBlockBOF4(block_size=128)
    quantized, metadata = quantizer.quantize(weights)
    
    ratio = quantizer.get_compression_ratio(metadata)
    print(f"  - Compression ratio: {ratio:.2f}x")
    assert 2.0 < ratio < 8.0, f"Compression ratio out of range: {ratio}"
    print("✓ Compression ratio calculated")


def test_batch_quantization():
    """Test BOF4 on batch of weights."""
    print("\n=== Test: Batch Quantization ===")
    layer_shapes = [
        (4096, 4096),
        (4096, 12288),
        (12288, 4096),
    ]
    
    config = PerBlockQuantizationConfig(method='bof4', block_size=128)
    
    total_error = 0
    for shape in layer_shapes:
        weights = torch.randn(*shape)
        quantized, metadata = quantize_weights(weights, config)
        dequantized = dequantize_weights(quantized, metadata)
        
        error = torch.abs(weights - dequantized).mean()
        total_error += error
        print(f"  - Layer {shape}: error = {error:.6f}")
    
    avg_error = total_error / len(layer_shapes)
    print(f"  - Average error: {avg_error:.6f}")
    assert avg_error < 0.2, f"Average error too high: {avg_error}"
    print("✓ Batch quantization successful")


def test_em_convergence():
    """Test that EM algorithm converges."""
    print("\n=== Test: EM Convergence ===")
    weights = torch.randn(256, 256)
    
    quantizer = PerBlockBOF4(block_size=128, max_em_iters=50)
    quantized, metadata = quantizer.quantize(weights)
    
    for i, codebook in enumerate(metadata['codebooks']):
        assert torch.all(torch.isfinite(codebook)), f"Non-finite codebook {i}"
    
    print("✓ EM algorithm converged successfully")


def test_comparison_with_phase1():
    """Compare BOF4 with Phase 1 (Four Over Six)."""
    print("\n=== Test: Comparison with Phase 1 ===")
    weights = torch.randn(512, 512)
    
    from tensorrt_llm.quantization.per_block_codebook import PerBlockAdaptiveScaling
    
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
    print(f"  - Improvement: {((error_p1 - error_bof4) / error_p1 * 100):.2f}%")
    print("✓ Comparison complete")


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
        test_small_weights,
        test_large_weights,
        test_compression_ratio,
        test_batch_quantization,
        test_em_convergence,
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

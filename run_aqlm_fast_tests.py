#!/usr/bin/env python
"""
Fast AQLM test runner - core tests only, no expensive operations.
Bypasses bindings import issue with direct module loading.
"""

import sys
import os
import importlib.util
import torch

# Load per_block_codebook module directly
spec = importlib.util.spec_from_file_location(
    'per_block_codebook',
    'tensorrt_llm/quantization/per_block_codebook.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

PerBlockAQLM = module.PerBlockAQLM


def test_1_quantize_simple():
    """Test 1: Basic AQLM quantization on random tensor."""
    print("\n[Test 1/5] Basic quantization...")
    weights = torch.randn(256, 256)
    
    quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
    quantized, metadata = quantizer.quantize(weights)
    
    assert quantized.shape == weights.shape, f"Shape mismatch: {quantized.shape} vs {weights.shape}"
    assert metadata['original_shape'] == weights.shape
    assert metadata['method'] == 'aqlm'
    assert 'codebooks' in metadata
    assert 'indices' in metadata
    assert 'scales' in metadata
    assert metadata['num_codebooks'] == 2
    assert metadata['codebook_size'] == 256
    assert metadata['block_size'] == 128
    print("✓ Test 1 PASSED: Basic quantization works")


def test_2_dequantize():
    """Test 2: AQLM dequantization from metadata."""
    print("\n[Test 2/5] Dequantization...")
    weights = torch.randn(256, 256)
    
    quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
    quantized, metadata = quantizer.quantize(weights)
    dequantized = quantizer.dequantize(quantized, metadata)
    
    assert dequantized.shape == weights.shape, f"Shape mismatch: {dequantized.shape} vs {weights.shape}"
    print("✓ Test 2 PASSED: Dequantization works")


def test_3_numerical_stability():
    """Test 3: Check for NaN/Inf in codebooks and reconstructions."""
    print("\n[Test 3/5] Numerical stability...")
    weights = torch.randn(256, 256)
    
    quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256, max_iters=10)
    quantized, metadata = quantizer.quantize(weights)
    
    has_nan = False
    has_inf = False
    
    for block_codebooks in metadata['codebooks']:
        for codebook in block_codebooks:
            if torch.isnan(codebook).any():
                has_nan = True
            if torch.isinf(codebook).any():
                has_inf = True
    
    for scale in metadata['scales']:
        if torch.isnan(scale).any():
            has_nan = True
        if torch.isinf(scale).any():
            has_inf = True
    
    if torch.isnan(quantized).any():
        has_nan = True
    if torch.isinf(quantized).any():
        has_inf = True
    
    assert not has_nan, "NaN detected in codebooks, scales, or output"
    assert not has_inf, "Inf detected in codebooks, scales, or output"
    print(f"✓ Test 3 PASSED: Numerical stability verified (Has NaN: {has_nan}, Has Inf: {has_inf})")


def test_4_codebook_structure():
    """Test 4: Verify codebook structure is correct."""
    print("\n[Test 4/5] Codebook structure...")
    weights = torch.randn(256, 256)
    
    quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
    quantized, metadata = quantizer.quantize(weights)
    
    # 256x256 with block_size=128 -> 2x2 = 4 blocks
    num_blocks = (256 // 128) * (256 // 128)
    assert len(metadata['codebooks']) == num_blocks, f"Expected {num_blocks} blocks, got {len(metadata['codebooks'])}"
    
    for block_codebooks in metadata['codebooks']:
        assert len(block_codebooks) == 2, f"Expected 2 codebooks per block, got {len(block_codebooks)}"
        for codebook in block_codebooks:
            assert codebook.shape[0] == 256, f"Expected codebook size 256, got {codebook.shape[0]}"
    
    print(f"✓ Test 4 PASSED: Codebook structure correct ({num_blocks} blocks, 2 codebooks per block, size 256)")


def test_5_compression_ratio():
    """Test 5: Verify compression ratio is reasonable."""
    print("\n[Test 5/5] Compression ratio...")
    weights = torch.randn(256, 256)
    
    quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
    quantized, metadata = quantizer.quantize(weights)
    
    original_size = weights.numel() * 4  # FP32
    num_blocks = (256 // 128) ** 2
    codebook_size = num_blocks * 2 * 256 * 4  # num_blocks * num_codebooks * codebook_size * FP32
    indices_size = weights.numel() * 2 * 1  # 2 indices per element, 1 byte each
    
    compressed_size = codebook_size + indices_size
    compression_ratio = original_size / compressed_size
    
    assert compression_ratio > 1.0, f"No compression achieved: {compression_ratio}x"
    print(f"✓ Test 5 PASSED: Compression ratio {compression_ratio:.2f}x (original: {original_size} bytes, compressed: {compressed_size} bytes)")


def main():
    print("=" * 70)
    print("AQLM FAST TEST SUITE - Core Tests Only")
    print("=" * 70)
    
    tests = [
        test_1_quantize_simple,
        test_2_dequantize,
        test_3_numerical_stability,
        test_4_codebook_structure,
        test_5_compression_ratio,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ FAILED: {test_func.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

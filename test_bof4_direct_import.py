#!/usr/bin/env python3
"""
Direct import test for BOF4 implementation.
Bypasses full tensorrt_llm initialization to avoid nvtx import issues.
"""

import sys
import torch
import importlib.util

# Load per_block_codebook module directly
spec = importlib.util.spec_from_file_location(
    'per_block_codebook',
    'tensorrt_llm/quantization/per_block_codebook.py'
)
module = importlib.util.module_from_spec(spec)
sys.modules['per_block_codebook'] = module
spec.loader.exec_module(module)

# Extract classes
PerBlockBOF4 = module.PerBlockBOF4
PerBlockQuantizationConfig = module.PerBlockQuantizationConfig
quantize_weights = module.quantize_weights
dequantize_weights = module.dequantize_weights

print("=" * 70)
print("BOF4 DIRECT IMPORT TEST SUITE")
print("=" * 70)

test_results = []

# Test 1: Basic quantization
print("\n[Test 1] Basic quantization (shape/metadata validation)")
try:
    weights = torch.randn(256, 256)
    quantizer = PerBlockBOF4(block_size=128, num_codewords=16)
    quantized, metadata = quantizer.quantize(weights)
    
    assert quantized.shape == weights.shape, f"Shape mismatch: {quantized.shape} vs {weights.shape}"
    assert metadata['method'] == 'bof4', "Method should be 'bof4'"
    assert metadata['block_size'] == 128, "Block size should be 128"
    assert 'codebooks' in metadata, "Metadata missing 'codebooks'"
    assert 'outlier_masks' in metadata, "Metadata missing 'outlier_masks'"
    assert 'scales' in metadata, "Metadata missing 'scales'"
    
    print("✓ PASS: Shape and metadata correct")
    test_results.append(("Test 1: Basic quantization", True, None))
except Exception as e:
    print(f"✗ FAIL: {e}")
    test_results.append(("Test 1: Basic quantization", False, str(e)))

# Test 2: Dequantization
print("\n[Test 2] Dequantization (reconstruction error < 0.15)")
try:
    weights = torch.randn(256, 256)
    quantizer = PerBlockBOF4(block_size=128, num_codewords=16)
    quantized, metadata = quantizer.quantize(weights)
    dequantized = quantizer.dequantize(quantized, metadata)
    
    error = torch.abs(weights - dequantized).mean().item()
    print(f"  Reconstruction error: {error:.6f}")
    assert error < 0.15, f"Error too high: {error}"
    
    print("✓ PASS: Reconstruction error acceptable")
    test_results.append(("Test 2: Dequantization", True, None))
except Exception as e:
    print(f"✗ FAIL: {e}")
    test_results.append(("Test 2: Dequantization", False, str(e)))

# Test 3: Roundtrip
print("\n[Test 3] Quantize-dequantize roundtrip (error < 0.2)")
try:
    weights = torch.randn(512, 512)
    quantizer = PerBlockBOF4(block_size=128, num_codewords=16)
    quantized, metadata = quantizer.quantize(weights)
    dequantized = quantizer.dequantize(quantized, metadata)
    
    error = torch.abs(weights - dequantized).mean().item()
    print(f"  Roundtrip error: {error:.6f}")
    assert error < 0.2, f"Error too high: {error}"
    
    print("✓ PASS: Roundtrip error acceptable")
    test_results.append(("Test 3: Roundtrip", True, None))
except Exception as e:
    print(f"✗ FAIL: {e}")
    test_results.append(("Test 3: Roundtrip", False, str(e)))

# Test 4: Codebook learning
print("\n[Test 4] Codebook learning (EM learns valid codebooks)")
try:
    weights = torch.randn(256, 256)
    quantizer = PerBlockBOF4(block_size=128, num_codewords=16, max_em_iters=50)
    quantized, metadata = quantizer.quantize(weights)
    
    # Check codebooks exist and have correct shape
    codebooks = metadata['codebooks']
    assert len(codebooks) > 0, "No codebooks learned"
    
    for i, cb in enumerate(codebooks):
        assert cb.shape[0] == 16, f"Codebook {i} has wrong size: {cb.shape[0]}"
        assert torch.isfinite(cb).all(), f"Codebook {i} has non-finite values"
    
    print(f"  Learned {len(codebooks)} codebooks")
    print("✓ PASS: Codebooks learned successfully")
    test_results.append(("Test 4: Codebook learning", True, None))
except Exception as e:
    print(f"✗ FAIL: {e}")
    test_results.append(("Test 4: Codebook learning", False, str(e)))

# Test 5: Outlier detection
print("\n[Test 5] Outlier detection (0-50% outlier ratio expected)")
try:
    weights = torch.randn(256, 256)
    quantizer = PerBlockBOF4(block_size=128, num_codewords=16, outlier_threshold=2.0)
    quantized, metadata = quantizer.quantize(weights)
    
    outlier_masks = metadata['outlier_masks']
    total_outliers = sum(mask.sum().item() for mask in outlier_masks)
    total_elements = sum(mask.numel() for mask in outlier_masks)
    outlier_ratio = total_outliers / total_elements if total_elements > 0 else 0
    
    print(f"  Outlier ratio: {outlier_ratio:.2%}")
    assert 0 <= outlier_ratio <= 0.5, f"Outlier ratio out of range: {outlier_ratio}"
    
    print("✓ PASS: Outlier detection working")
    test_results.append(("Test 5: Outlier detection", True, None))
except Exception as e:
    print(f"✗ FAIL: {e}")
    test_results.append(("Test 5: Outlier detection", False, str(e)))

# Test 6: Different block sizes
print("\n[Test 6] Different block sizes (64, 128, 256)")
try:
    weights = torch.randn(512, 512)
    
    for block_size in [64, 128, 256]:
        quantizer = PerBlockBOF4(block_size=block_size, num_codewords=16)
        quantized, metadata = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(quantized, metadata)
        
        error = torch.abs(weights - dequantized).mean().item()
        print(f"  Block size {block_size}: error = {error:.6f}")
        assert error < 0.3, f"Error too high for block size {block_size}: {error}"
    
    print("✓ PASS: All block sizes work")
    test_results.append(("Test 6: Different block sizes", True, None))
except Exception as e:
    print(f"✗ FAIL: {e}")
    test_results.append(("Test 6: Different block sizes", False, str(e)))

# Test 7: Compression ratio
print("\n[Test 7] Compression ratio (2-8x range)")
try:
    weights = torch.randn(1024, 1024)
    quantizer = PerBlockBOF4(block_size=128, num_codewords=16)
    quantized, metadata = quantizer.quantize(weights)
    
    ratio = quantizer.compression_ratio(weights.shape, metadata)
    print(f"  Compression ratio: {ratio:.2f}x")
    assert 2 <= ratio <= 8, f"Compression ratio out of range: {ratio}"
    
    print("✓ PASS: Compression ratio in expected range")
    test_results.append(("Test 7: Compression ratio", True, None))
except Exception as e:
    print(f"✗ FAIL: {e}")
    test_results.append(("Test 7: Compression ratio", False, str(e)))

# Test 8: Comparison with Phase 1
print("\n[Test 8] Comparison with Phase 1 (BOF4 should be better or equal)")
try:
    weights = torch.randn(256, 256)
    
    # Phase 1: Four Over Six
    phase1_quantizer = module.PerBlockAdaptiveScaling(block_size=128)
    phase1_quantized, phase1_metadata = phase1_quantizer.quantize(weights)
    phase1_dequantized = phase1_quantizer.dequantize(phase1_quantized, phase1_metadata)
    phase1_error = torch.abs(weights - phase1_dequantized).mean().item()
    
    # Phase 2: BOF4
    phase2_quantizer = PerBlockBOF4(block_size=128, num_codewords=16)
    phase2_quantized, phase2_metadata = phase2_quantizer.quantize(weights)
    phase2_dequantized = phase2_quantizer.dequantize(phase2_quantized, phase2_metadata)
    phase2_error = torch.abs(weights - phase2_dequantized).mean().item()
    
    print(f"  Phase 1 error: {phase1_error:.6f}")
    print(f"  Phase 2 error: {phase2_error:.6f}")
    print(f"  Improvement: {(phase1_error - phase2_error) / phase1_error * 100:.2f}%")
    
    # BOF4 should be better or within 5% (due to randomness)
    assert phase2_error <= phase1_error * 1.05, f"Phase 2 worse than Phase 1: {phase2_error} vs {phase1_error}"
    
    print("✓ PASS: BOF4 comparable or better than Phase 1")
    test_results.append(("Test 8: Phase 1 comparison", True, None))
except Exception as e:
    print(f"✗ FAIL: {e}")
    test_results.append(("Test 8: Phase 1 comparison", False, str(e)))

# Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

passed = sum(1 for _, result, _ in test_results if result)
failed = sum(1 for _, result, _ in test_results if not result)

for test_name, result, error in test_results:
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"{status}: {test_name}")
    if error:
        print(f"       {error}")

print(f"\nTotal: {passed} passed, {failed} failed out of {len(test_results)} tests")
print("=" * 70)

sys.exit(0 if failed == 0 else 1)

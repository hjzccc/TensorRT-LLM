"""
Direct standalone tests for per-block codebook quantization.
Imports module directly without TensorRT-LLM package initialization.
"""

import torch
import sys
import os
import importlib.util

# Load the module directly
module_path = os.path.join(
    os.path.dirname(__file__),
    'tensorrt_llm/quantization/per_block_codebook.py'
)

spec = importlib.util.spec_from_file_location("per_block_codebook", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

PerBlockAdaptiveScaling = module.PerBlockAdaptiveScaling
PerBlockQuantizationConfig = module.PerBlockQuantizationConfig
quantize_weights = module.quantize_weights
dequantize_weights = module.dequantize_weights


def test_quantize_simple():
    """Test basic quantization."""
    print("\n=== Test: Basic Quantization ===")
    weights = torch.randn(256, 256)
    
    quantizer = PerBlockAdaptiveScaling(block_size=128)
    quantized, metadata = quantizer.quantize(weights)
    
    assert quantized.shape == weights.shape
    assert metadata['original_shape'] == weights.shape
    assert metadata['method'] == 'four_over_six'
    assert len(metadata['scales']) == 4  # 256/128 = 2, so 2x2 = 4 blocks
    
    print(f"✓ Quantization successful")
    print(f"  - Input shape: {weights.shape}")
    print(f"  - Output shape: {quantized.shape}")
    print(f"  - Number of blocks: {len(metadata['scales'])}")


def test_dequantize():
    """Test dequantization."""
    print("\n=== Test: Dequantization ===")
    weights = torch.randn(256, 256)
    
    quantizer = PerBlockAdaptiveScaling(block_size=128)
    quantized, metadata = quantizer.quantize(weights)
    dequantized = quantizer.dequantize(quantized, metadata)
    
    assert dequantized.shape == weights.shape
    
    error = torch.abs(weights - dequantized).mean()
    print(f"✓ Dequantization successful")
    print(f"  - Mean reconstruction error: {error:.6f}")
    assert error < 0.1


def test_roundtrip():
    """Test full quantize-dequantize roundtrip."""
    print("\n=== Test: Quantize-Dequantize Roundtrip ===")
    weights = torch.randn(512, 512)
    
    config = PerBlockQuantizationConfig(method='four_over_six', block_size=128)
    quantized, metadata = quantize_weights(weights, config)
    dequantized = dequantize_weights(quantized, metadata)
    
    error = torch.abs(weights - dequantized).mean()
    print(f"✓ Roundtrip successful")
    print(f"  - Input shape: {weights.shape}")
    print(f"  - Reconstruction error: {error:.6f}")
    assert error < 0.2


def test_compression_ratio():
    """Test compression ratio calculation."""
    print("\n=== Test: Compression Ratio ===")
    weights = torch.randn(256, 256)
    
    quantizer = PerBlockAdaptiveScaling(block_size=128)
    quantized, metadata = quantizer.quantize(weights)
    
    ratio = quantizer.get_compression_ratio(metadata)
    print(f"✓ Compression ratio calculated")
    print(f"  - Compression ratio: {ratio:.2f}x")
    print(f"  - Original: 32-bit float")
    print(f"  - Compressed: 4-bit weights + per-block scale")
    assert 7.0 < ratio < 9.0


def test_different_block_sizes():
    """Test with different block sizes."""
    print("\n=== Test: Different Block Sizes ===")
    weights = torch.randn(256, 256)
    
    for block_size in [64, 128, 256]:
        quantizer = PerBlockAdaptiveScaling(block_size=block_size)
        quantized, metadata = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(quantized, metadata)
        
        error = torch.abs(weights - dequantized).mean()
        print(f"  - Block size {block_size}: error = {error:.6f}")
        assert error < 0.2
    
    print(f"✓ All block sizes tested successfully")


def test_small_weights():
    """Test with small weight values."""
    print("\n=== Test: Small Weight Values ===")
    weights = torch.randn(128, 128) * 0.01
    
    quantizer = PerBlockAdaptiveScaling(block_size=64)
    quantized, metadata = quantizer.quantize(weights)
    dequantized = quantizer.dequantize(quantized, metadata)
    
    relative_error = torch.abs(weights - dequantized) / (torch.abs(weights) + 1e-8)
    print(f"✓ Small weights handled correctly")
    print(f"  - Max relative error: {relative_error.max():.4f}")
    assert relative_error.max() < 1.0


def test_large_weights():
    """Test with large weight values."""
    print("\n=== Test: Large Weight Values ===")
    weights = torch.randn(128, 128) * 100.0
    
    quantizer = PerBlockAdaptiveScaling(block_size=64)
    quantized, metadata = quantizer.quantize(weights)
    dequantized = quantizer.dequantize(quantized, metadata)
    
    error = torch.abs(weights - dequantized).mean()
    print(f"✓ Large weights handled correctly")
    print(f"  - Mean error: {error:.6f}")
    assert error < 10.0


def test_metadata_preservation():
    """Test that metadata is correctly preserved."""
    print("\n=== Test: Metadata Preservation ===")
    weights = torch.randn(256, 256)
    
    quantizer = PerBlockAdaptiveScaling(block_size=128)
    quantized, metadata = quantizer.quantize(weights)
    
    assert 'method' in metadata
    assert 'block_size' in metadata
    assert 'scales' in metadata
    assert 'original_shape' in metadata
    assert 'dtype' in metadata
    
    assert metadata['method'] == 'four_over_six'
    assert metadata['block_size'] == 128
    assert metadata['original_shape'] == (256, 256)
    
    print(f"✓ Metadata preserved correctly")
    print(f"  - Method: {metadata['method']}")
    print(f"  - Block size: {metadata['block_size']}")
    print(f"  - Original shape: {metadata['original_shape']}")
    print(f"  - Data type: {metadata['dtype']}")


def test_multiple_layers():
    """Test quantizing multiple weight matrices."""
    print("\n=== Test: Multiple Layers ===")
    weights_list = [
        torch.randn(256, 256),
        torch.randn(512, 512),
        torch.randn(1024, 1024),
    ]
    
    config = PerBlockQuantizationConfig(method='four_over_six', block_size=128)
    
    for i, weights in enumerate(weights_list):
        quantized, metadata = quantize_weights(weights, config)
        dequantized = dequantize_weights(quantized, metadata)
        
        error = torch.abs(weights - dequantized).mean()
        print(f"  - Layer {i+1} {weights.shape}: error = {error:.6f}")
        assert error < 0.2
    
    print(f"✓ All layers quantized successfully")


def test_batch_quantization():
    """Test quantizing batches of weights."""
    print("\n=== Test: Batch Quantization (Simulated Model) ===")
    layer_shapes = [
        (4096, 4096),   # Attention
        (4096, 12288),  # FFN up
        (12288, 4096),  # FFN down
    ]
    
    config = PerBlockQuantizationConfig(method='four_over_six', block_size=128)
    
    total_error = 0
    for shape in layer_shapes:
        weights = torch.randn(*shape)
        quantized, metadata = quantize_weights(weights, config)
        dequantized = dequantize_weights(quantized, metadata)
        
        error = torch.abs(weights - dequantized).mean()
        total_error += error
        print(f"  - Layer {shape}: error = {error:.6f}")
    
    avg_error = total_error / len(layer_shapes)
    print(f"✓ Batch quantization successful")
    print(f"  - Average error: {avg_error:.6f}")
    assert avg_error < 0.2


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("Per-Block Codebook Quantization - Standalone Tests")
    print("=" * 70)
    
    tests = [
        test_quantize_simple,
        test_dequantize,
        test_roundtrip,
        test_compression_ratio,
        test_different_block_sizes,
        test_small_weights,
        test_large_weights,
        test_metadata_preservation,
        test_multiple_layers,
        test_batch_quantization,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

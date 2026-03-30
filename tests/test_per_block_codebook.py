"""
Tests for per-block codebook quantization.
"""

import torch
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tensorrt_llm.quantization.per_block_codebook import (
    PerBlockAdaptiveScaling,
    PerBlockQuantizationConfig,
    quantize_weights,
    dequantize_weights,
)


class TestPerBlockAdaptiveScaling:
    """Test Four Over Six adaptive scaling quantization."""
    
    def test_quantize_simple(self):
        """Test basic quantization."""
        # Create simple weight tensor
        weights = torch.randn(256, 256)
        
        # Quantize
        quantizer = PerBlockAdaptiveScaling(block_size=128)
        quantized, metadata = quantizer.quantize(weights)
        
        # Check shapes
        assert quantized.shape == weights.shape
        assert metadata['original_shape'] == weights.shape
        assert metadata['method'] == 'four_over_six'
        
        # Check scales
        assert len(metadata['scales']) == 4  # 256/128 = 2, so 2x2 = 4 blocks
    
    def test_dequantize(self):
        """Test dequantization."""
        weights = torch.randn(256, 256)
        
        quantizer = PerBlockAdaptiveScaling(block_size=128)
        quantized, metadata = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(quantized, metadata)
        
        # Check shape
        assert dequantized.shape == weights.shape
        
        # Check reconstruction error is reasonable
        error = torch.abs(weights - dequantized).mean()
        print(f"Mean reconstruction error: {error:.6f}")
        assert error < 0.1  # Should be small for FP4
    
    def test_quantize_dequantize_roundtrip(self):
        """Test full quantize-dequantize roundtrip."""
        weights = torch.randn(512, 512)
        
        config = PerBlockQuantizationConfig(method='four_over_six', block_size=128)
        quantized, metadata = quantize_weights(weights, config)
        dequantized = dequantize_weights(quantized, metadata)
        
        # Check reconstruction
        error = torch.abs(weights - dequantized).mean()
        print(f"Roundtrip error: {error:.6f}")
        assert error < 0.2
    
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        weights = torch.randn(256, 256)
        
        quantizer = PerBlockAdaptiveScaling(block_size=128)
        quantized, metadata = quantizer.quantize(weights)
        
        ratio = quantizer.get_compression_ratio(metadata)
        print(f"Compression ratio: {ratio:.2f}x")
        
        # Should be around 8x (32-bit -> 4-bit)
        assert 7.0 < ratio < 9.0
    
    def test_different_block_sizes(self):
        """Test with different block sizes."""
        weights = torch.randn(256, 256)
        
        for block_size in [64, 128, 256]:
            quantizer = PerBlockAdaptiveScaling(block_size=block_size)
            quantized, metadata = quantizer.quantize(weights)
            dequantized = quantizer.dequantize(quantized, metadata)
            
            error = torch.abs(weights - dequantized).mean()
            print(f"Block size {block_size}: error = {error:.6f}")
            assert error < 0.2
    
    def test_small_weights(self):
        """Test with small weight values."""
        weights = torch.randn(128, 128) * 0.01  # Small values
        
        quantizer = PerBlockAdaptiveScaling(block_size=64)
        quantized, metadata = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(quantized, metadata)
        
        # Relative error should be reasonable
        relative_error = torch.abs(weights - dequantized) / (torch.abs(weights) + 1e-8)
        print(f"Max relative error: {relative_error.max():.4f}")
        assert relative_error.max() < 1.0  # Less than 100% relative error
    
    def test_large_weights(self):
        """Test with large weight values."""
        weights = torch.randn(128, 128) * 100.0  # Large values
        
        quantizer = PerBlockAdaptiveScaling(block_size=64)
        quantized, metadata = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(quantized, metadata)
        
        error = torch.abs(weights - dequantized).mean()
        print(f"Large weights error: {error:.6f}")
        assert error < 10.0  # Absolute error scales with magnitude
    
    def test_device_consistency(self):
        """Test consistency across devices."""
        weights = torch.randn(256, 256)
        
        quantizer = PerBlockAdaptiveScaling(block_size=128)
        
        # CPU
        quantized_cpu, metadata_cpu = quantizer.quantize(weights)
        dequantized_cpu = quantizer.dequantize(quantized_cpu, metadata_cpu)
        
        # Check results
        assert dequantized_cpu.device.type == 'cpu'
        error_cpu = torch.abs(weights - dequantized_cpu).mean()
        print(f"CPU error: {error_cpu:.6f}")
        assert error_cpu < 0.2
    
    def test_metadata_preservation(self):
        """Test that metadata is correctly preserved."""
        weights = torch.randn(256, 256)
        
        quantizer = PerBlockAdaptiveScaling(block_size=128)
        quantized, metadata = quantizer.quantize(weights)
        
        # Check metadata fields
        assert 'method' in metadata
        assert 'block_size' in metadata
        assert 'scales' in metadata
        assert 'original_shape' in metadata
        assert 'dtype' in metadata
        
        assert metadata['method'] == 'four_over_six'
        assert metadata['block_size'] == 128
        assert metadata['original_shape'] == (256, 256)
        assert metadata['dtype'] == weights.dtype


class TestPerBlockQuantizationConfig:
    """Test configuration class."""
    
    def test_create_quantizer(self):
        """Test quantizer creation from config."""
        config = PerBlockQuantizationConfig(
            method='four_over_six',
            block_size=128,
            bits=4
        )
        
        quantizer = config.create_quantizer()
        assert isinstance(quantizer, PerBlockAdaptiveScaling)
        assert quantizer.block_size == 128
    
    def test_invalid_method(self):
        """Test error on invalid method."""
        config = PerBlockQuantizationConfig(method='invalid_method')
        
        with pytest.raises(ValueError):
            config.create_quantizer()


class TestIntegration:
    """Integration tests."""
    
    def test_quantize_multiple_layers(self):
        """Test quantizing multiple weight matrices."""
        weights_list = [
            torch.randn(256, 256),
            torch.randn(512, 512),
            torch.randn(1024, 1024),
        ]
        
        config = PerBlockQuantizationConfig(method='four_over_six', block_size=128)
        
        for weights in weights_list:
            quantized, metadata = quantize_weights(weights, config)
            dequantized = dequantize_weights(quantized, metadata)
            
            error = torch.abs(weights - dequantized).mean()
            print(f"Shape {weights.shape}: error = {error:.6f}")
            assert error < 0.2
    
    def test_batch_quantization(self):
        """Test quantizing batches of weights."""
        # Simulate quantizing all weights in a small model
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
            print(f"Layer {shape}: error = {error:.6f}")
        
        avg_error = total_error / len(layer_shapes)
        print(f"Average error: {avg_error:.6f}")
        assert avg_error < 0.2


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])

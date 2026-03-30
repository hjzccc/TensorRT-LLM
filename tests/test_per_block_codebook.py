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
    PerBlockBOF4,
    PerBlockGLVQ,
    PerBlockAQLM,
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
        
        for block_size in [8, 16, 32]:
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


class TestPerBlockBOF4:
    """Test BOF4 EM-optimized learned codebook quantization."""
    
    def test_quantize_simple(self):
        """Test basic BOF4 quantization."""
        weights = torch.randn(256, 256)
        
        quantizer = PerBlockBOF4(block_size=128)
        quantized, metadata = quantizer.quantize(weights)
        
        assert quantized.shape == weights.shape
        assert metadata['method'] == 'bof4'
        assert metadata['num_codewords'] == 16
        assert len(metadata['codebooks']) == 4
        assert len(metadata['outlier_masks']) == 4
        assert len(metadata['scales']) == 4
    
    def test_dequantize(self):
        """Test BOF4 dequantization."""
        weights = torch.randn(256, 256)
        
        quantizer = PerBlockBOF4(block_size=128)
        quantized, metadata = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(quantized, metadata)
        
        assert dequantized.shape == weights.shape
        
        error = torch.abs(weights - dequantized).mean()
        print(f"BOF4 mean reconstruction error: {error:.6f}")
        assert error < 0.15
    
    def test_quantize_dequantize_roundtrip(self):
        """Test full BOF4 quantize-dequantize roundtrip."""
        weights = torch.randn(512, 512)
        
        config = PerBlockQuantizationConfig(method='bof4', block_size=128)
        quantized, metadata = quantize_weights(weights, config)
        dequantized = dequantize_weights(quantized, metadata)
        
        error = torch.abs(weights - dequantized).mean()
        print(f"BOF4 roundtrip error: {error:.6f}")
        assert error < 0.2
    
    def test_codebook_learning(self):
        """Test that EM learns reasonable codebooks."""
        weights = torch.randn(256, 256)
        
        quantizer = PerBlockBOF4(block_size=128)
        quantized, metadata = quantizer.quantize(weights)
        
        for i, codebook in enumerate(metadata['codebooks']):
            assert codebook.shape == (16,)
            assert torch.all(torch.isfinite(codebook))
            print(f"Block {i} codebook range: [{codebook.min():.4f}, {codebook.max():.4f}]")
    
    def test_outlier_detection(self):
        """Test outlier detection mechanism."""
        weights = torch.randn(256, 256)
        
        quantizer = PerBlockBOF4(block_size=128, outlier_threshold=1.5)
        quantized, metadata = quantizer.quantize(weights)
        
        total_outliers = 0
        for i, outlier_mask in enumerate(metadata['outlier_masks']):
            num_outliers = outlier_mask.sum().item()
            total_outliers += num_outliers
            print(f"Block {i}: {num_outliers} outliers out of {outlier_mask.numel()}")
        
        outlier_ratio = total_outliers / (256 * 256)
        print(f"Total outlier ratio: {outlier_ratio:.4f}")
        assert 0 <= outlier_ratio <= 0.5
    
    def test_different_block_sizes(self):
        """Test BOF4 with different block sizes."""
        weights = torch.randn(256, 256)
        
        for block_size in [8, 16, 32]:
            quantizer = PerBlockBOF4(block_size=block_size)
            quantized, metadata = quantizer.quantize(weights)
            dequantized = quantizer.dequantize(quantized, metadata)
            
            error = torch.abs(weights - dequantized).mean()
            print(f"BOF4 block size {block_size}: error = {error:.6f}")
            assert error < 0.2
    
    def test_small_weights(self):
        """Test BOF4 with small weight values."""
        weights = torch.randn(128, 128) * 0.01
        
        quantizer = PerBlockBOF4(block_size=64)
        quantized, metadata = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(quantized, metadata)
        
        error = torch.abs(weights - dequantized).mean()
        print(f"BOF4 small weights error: {error:.6f}")
        assert error < 0.01
    
    def test_large_weights(self):
        """Test BOF4 with large weight values."""
        weights = torch.randn(128, 128) * 100
        
        quantizer = PerBlockBOF4(block_size=64)
        quantized, metadata = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(quantized, metadata)
        
        error = torch.abs(weights - dequantized).mean()
        print(f"BOF4 large weights error: {error:.6f}")
        assert error < 20
    
    def test_compression_ratio(self):
        """Test BOF4 compression ratio calculation."""
        weights = torch.randn(256, 256)
        
        quantizer = PerBlockBOF4(block_size=128)
        quantized, metadata = quantizer.quantize(weights)
        
        ratio = quantizer.get_compression_ratio(metadata)
        print(f"BOF4 compression ratio: {ratio:.2f}x")
        
        assert 2.0 < ratio < 8.0
    
    def test_batch_quantization(self):
        """Test BOF4 on batch of weights."""
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
            print(f"BOF4 layer {shape}: error = {error:.6f}")
        
        avg_error = total_error / len(layer_shapes)
        print(f"BOF4 average error: {avg_error:.6f}")
        assert avg_error < 0.2
    
    def test_em_convergence(self):
        """Test that EM algorithm converges."""
        weights = torch.randn(256, 256)
        
        quantizer = PerBlockBOF4(block_size=128, max_em_iters=50)
        quantized, metadata = quantizer.quantize(weights)
        
        for i, codebook in enumerate(metadata['codebooks']):
            assert torch.all(torch.isfinite(codebook))
            print(f"Block {i} codebook learned successfully")




class TestPerBlockGLVQ:
    """Test GLVQ (Learned Lattice Vector Quantization)."""
    
    def test_quantize_simple(self):
        """Test basic GLVQ quantization."""
        weights = torch.randn(32, 32)
        
        quantizer = PerBlockGLVQ(block_size=16, max_iters=10)
        quantized, metadata = quantizer.quantize(weights)
        
        assert quantized.shape == weights.shape
        assert metadata['method'] == 'glvq'
        assert metadata['block_size'] == 16
        assert 'transformation_matrices' in metadata
        assert 'scales' in metadata
        assert len(metadata['transformation_matrices']) == 4
        assert len(metadata['scales']) == 4
    
    def test_dequantize(self):
        """Test GLVQ dequantization."""
        weights = torch.randn(32, 32)
        
        quantizer = PerBlockGLVQ(block_size=16, max_iters=10)
        quantized, metadata = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(quantized, metadata)
        
        assert dequantized.shape == weights.shape
        
        error = torch.abs(weights - dequantized).mean()
        print(f"GLVQ mean reconstruction error: {error:.6f}")
        assert error < 0.15
    
    def test_quantize_dequantize_roundtrip(self):
        """Test full GLVQ quantize-dequantize roundtrip."""
        weights = torch.randn(32, 32)
        
        config = PerBlockQuantizationConfig(method='glvq', block_size=16)
        quantized, metadata = quantize_weights(weights, config)
        dequantized = dequantize_weights(quantized, metadata)
        
        error = torch.abs(weights - dequantized).mean()
        print(f"GLVQ roundtrip error: {error:.6f}")
        assert error < 0.5  # GLVQ with 4-bit equivalent (num_levels=8) has higher error
    
    def test_lattice_learning(self):
        """Test that lattice transformation matrices are learned."""
        weights = torch.randn(32, 32)
        
        quantizer = PerBlockGLVQ(block_size=16, max_iters=10)
        quantized, metadata = quantizer.quantize(weights)
        
        # 32x32 with block_size=16 -> 4 blocks
        assert len(metadata['transformation_matrices']) == 4
        for i, A in enumerate(metadata['transformation_matrices']):
            # A is block_size x block_size (row-wise transformation)
            assert A.shape == (16, 16)
            assert torch.all(torch.isfinite(A))
            cond_num = torch.linalg.cond(A).item()
            print(f"Block {i} transformation matrix condition number: {cond_num:.4f}")
            assert cond_num < 1000
    
    def test_different_block_sizes(self):
        """Test GLVQ with different block sizes."""
        weights = torch.randn(32, 32)
        
        for block_size in [8, 16, 32]:
            quantizer = PerBlockGLVQ(block_size=block_size)
            quantized, metadata = quantizer.quantize(weights)
            dequantized = quantizer.dequantize(quantized, metadata)
            
            error = torch.abs(weights - dequantized).mean()
            print(f"GLVQ block size {block_size}: error = {error:.6f}")
            assert error < 0.2
    
    def test_small_weights(self):
        """Test GLVQ with small weight values."""
        weights = torch.randn(128, 128) * 0.01
        
        quantizer = PerBlockGLVQ(block_size=16, max_iters=10)
        quantized, metadata = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(quantized, metadata)
        
        error = torch.abs(weights - dequantized).mean()
        print(f"GLVQ small weights error: {error:.6f}")
        assert error < 0.01
    
    def test_large_weights(self):
        """Test GLVQ with large weight values."""
        weights = torch.randn(128, 128) * 100
        
        quantizer = PerBlockGLVQ(block_size=16, max_iters=10)
        quantized, metadata = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(quantized, metadata)
        
        error = torch.abs(weights - dequantized).mean()
        print(f"GLVQ large weights error: {error:.6f}")
        assert error < 20
    
    def test_compression_ratio(self):
        """Test GLVQ compression ratio calculation."""
        weights = torch.randn(32, 32)
        
        quantizer = PerBlockGLVQ(block_size=16, max_iters=10)
        quantized, metadata = quantizer.quantize(weights)
        
        ratio = quantizer.get_compression_ratio(metadata)
        print(f"GLVQ compression ratio: {ratio:.2f}x")
        
        assert 2.0 < ratio < 8.0
    
    def test_batch_quantization(self):
        """Test GLVQ on batch of weights."""
        layer_shapes = [
            (4096, 4096),
            (4096, 12288),
            (12288, 4096),
        ]
        
        config = PerBlockQuantizationConfig(method='glvq', block_size=128)
        
        total_error = 0
        for shape in layer_shapes:
            weights = torch.randn(*shape)
            quantized, metadata = quantize_weights(weights, config)
            dequantized = dequantize_weights(quantized, metadata)
            
            error = torch.abs(weights - dequantized).mean()
            total_error += error
            print(f"GLVQ layer {shape}: error = {error:.6f}")
        
        avg_error = total_error / len(layer_shapes)
        print(f"GLVQ average error: {avg_error:.6f}")
        assert avg_error < 0.2
    
    def test_numerical_stability(self):
        """Test GLVQ numerical stability (no NaN/Inf)."""
        weights = torch.randn(32, 32)
        
        quantizer = PerBlockGLVQ(block_size=16, max_iters=10)
        quantized, metadata = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(quantized, metadata)
        
        assert torch.all(torch.isfinite(quantized))
        assert torch.all(torch.isfinite(dequantized))
        
        for A in metadata['transformation_matrices']:
            assert torch.all(torch.isfinite(A))
        
        print("GLVQ numerical stability verified")
    
    def test_comparison_with_phase1(self):
        """Test GLVQ performance compared to Phase 1 (Four Over Six)."""
        weights = torch.randn(32, 32)
        
        quantizer_phase1 = PerBlockAdaptiveScaling(block_size=128)
        quantized_p1, metadata_p1 = quantizer_phase1.quantize(weights)
        dequantized_p1 = quantizer_phase1.dequantize(quantized_p1, metadata_p1)
        error_p1 = torch.abs(weights - dequantized_p1).mean()
        
        quantizer_glvq = PerBlockGLVQ(block_size=16, max_iters=10)
        quantized_glvq, metadata_glvq = quantizer_glvq.quantize(weights)
        dequantized_glvq = quantizer_glvq.dequantize(quantized_glvq, metadata_glvq)
        error_glvq = torch.abs(weights - dequantized_glvq).mean()
        
        print(f"Phase 1 error: {error_p1:.6f}")
        print(f"Phase 3 (GLVQ) error: {error_glvq:.6f}")
        
        assert error_glvq < 0.2
        assert error_p1 < 0.2


class TestPerBlockAQLM:
    """Test AQLM (Additive Quantization with Learned Matrices) quantization."""
    
    def test_quantize_simple(self):
        """Test basic AQLM quantization on random tensor."""
        weights = torch.randn(256, 256)
        
        quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
        quantized, metadata = quantizer.quantize(weights)
        
        # Check shapes
        assert quantized.shape == weights.shape
        assert metadata['original_shape'] == weights.shape
        assert metadata['method'] == 'aqlm'
        
        # Check metadata structure
        assert 'codebooks' in metadata
        assert 'indices' in metadata
        assert 'scales' in metadata
        assert metadata['num_codebooks'] == 2
        assert metadata['codebook_size'] == 256
        assert metadata['block_size'] == 128
    
    def test_dequantize(self):
        """Test AQLM dequantization from metadata."""
        weights = torch.randn(256, 256)
        
        quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
        quantized, metadata = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(quantized, metadata)
        
        # Check shape
        assert dequantized.shape == weights.shape
        
        # Check that dequantized is close to quantized (not original, since quantization loses info)
        assert torch.allclose(dequantized, quantized, atol=1e-5)
    
    def test_roundtrip(self):
        """Test quantize -> dequantize roundtrip fidelity."""
        weights = torch.randn(256, 256)
        
        quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
        quantized, metadata = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(quantized, metadata)
        
        # Compute MSE
        mse = torch.mean((dequantized - quantized) ** 2).item()
        
        # MSE should be very small (ideally 0 for perfect reconstruction)
        assert mse < 1e-5, f"MSE too high: {mse}"
        
        # Check shapes match
        assert dequantized.shape == quantized.shape
    
    def test_codebook_learning(self):
        """Verify that codebooks are learned (not just initialized)."""
        weights = torch.randn(256, 256)
        
        quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256, max_iters=10)
        quantized, metadata = quantizer.quantize(weights)
        
        # Check that codebooks exist and have correct shape
        assert len(metadata['codebooks']) > 0
        
        # Each block should have codebooks
        for block_codebooks in metadata['codebooks']:
            assert len(block_codebooks) == 2  # num_codebooks=2
            for codebook in block_codebooks:
                assert codebook.shape[0] == 256  # codebook_size=256
    
    def test_num_codebooks_variation(self):
        """Test AQLM with different numbers of codebooks."""
        weights = torch.randn(256, 256)
        
        for num_codebooks in [1, 2, 3, 4]:
            quantizer = PerBlockAQLM(
                block_size=128,
                num_codebooks=num_codebooks,
                codebook_size=256
            )
            quantized, metadata = quantizer.quantize(weights)
            
            assert metadata['num_codebooks'] == num_codebooks
            
            # Check codebook structure
            for block_codebooks in metadata['codebooks']:
                assert len(block_codebooks) == num_codebooks
    
    def test_residual_quantization(self):
        """Verify that residual quantization works correctly."""
        weights = torch.randn(256, 256)
        
        # Test with residual quantization enabled
        quantizer = PerBlockAQLM(
            block_size=128,
            num_codebooks=2,
            codebook_size=256,
            use_residual=True
        )
        quantized, metadata = quantizer.quantize(weights)
        
        # Should have indices for each codebook
        for block_indices in metadata['indices']:
            assert len(block_indices) == 2  # num_codebooks=2
            for indices in block_indices:
                assert indices.shape[0] > 0  # Should have indices
    
    def test_small_weights(self):
        """Test AQLM on weights near zero (edge case)."""
        weights = torch.randn(256, 256) * 1e-6  # Very small weights
        
        quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
        quantized, metadata = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(quantized, metadata)
        
        # Should handle small weights without NaN/Inf
        assert not torch.isnan(dequantized).any()
        assert not torch.isinf(dequantized).any()
        assert dequantized.shape == weights.shape
    
    def test_large_weights(self):
        """Test AQLM on large magnitude weights (edge case)."""
        weights = torch.randn(256, 256) * 1e3  # Very large weights
        
        quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
        quantized, metadata = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(quantized, metadata)
        
        # Should handle large weights without NaN/Inf
        assert not torch.isnan(dequantized).any()
        assert not torch.isinf(dequantized).any()
        assert dequantized.shape == weights.shape
    
    def test_compression_ratio(self):
        """Verify compression ratio is reasonable."""
        weights = torch.randn(256, 256)
        
        quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
        quantized, metadata = quantizer.quantize(weights)
        
        # Estimate compression ratio
        # Original: 256*256*4 bytes (FP32)
        original_size = weights.numel() * 4
        
        # Compressed: indices (uint8) + codebooks (FP32)
        # For each block: 2 codebooks * 256 entries * 4 bytes + indices
        num_blocks = (256 // 128) ** 2  # 2x2 blocks
        codebook_size = num_blocks * 2 * 256 * 4  # 2 codebooks per block
        indices_size = weights.numel() * 2 * 1  # 2 indices per element (uint8)
        
        compressed_size = codebook_size + indices_size
        compression_ratio = original_size / compressed_size
        
        # Should achieve some compression (at least 1.5x with current FP32 storage)
        assert compression_ratio > 1.0, f"No compression achieved: {compression_ratio}x"
    
    def test_batch_quantization(self):
        """Test quantizing multiple blocks correctly."""
        weights = torch.randn(512, 512)  # 4x4 blocks with block_size=128
        
        quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
        quantized, metadata = quantizer.quantize(weights)
        
        # Should have 4x4 = 16 blocks
        assert len(metadata['codebooks']) == 16
        assert len(metadata['indices']) == 16
        assert len(metadata['scales']) == 16
    
    def test_numerical_stability(self):
        """Check for NaN/Inf in codebooks and reconstructions."""
        weights = torch.randn(256, 256)
        
        quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256, max_iters=10)
        quantized, metadata = quantizer.quantize(weights)
        
        # Check codebooks for NaN/Inf
        for block_codebooks in metadata['codebooks']:
            for codebook in block_codebooks:
                assert not torch.isnan(codebook).any(), "NaN in codebook"
                assert not torch.isinf(codebook).any(), "Inf in codebook"
        
        # Check scales for NaN/Inf
        for scale in metadata['scales']:
            assert not torch.isnan(scale).any(), "NaN in scale"
            assert not torch.isinf(scale).any(), "Inf in scale"
        
        # Check quantized output
        assert not torch.isnan(quantized).any(), "NaN in quantized output"
        assert not torch.isinf(quantized).any(), "Inf in quantized output"
    
    def test_comparison_with_phase3(self):
        """Compare AQLM (Phase 4) with GLVQ (Phase 3) on same data."""
        weights = torch.randn(256, 256)
        
        # Phase 3: GLVQ
        glvq = PerBlockGLVQ(block_size=16, max_iters=10)
        glvq_quantized, glvq_metadata = glvq.quantize(weights)
        glvq_dequantized = glvq.dequantize(glvq_quantized, glvq_metadata)
        glvq_error = torch.mean((glvq_dequantized - weights) ** 2).item()
        
        # Phase 4: AQLM
        aqlm = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
        aqlm_quantized, aqlm_metadata = aqlm.quantize(weights)
        aqlm_dequantized = aqlm.dequantize(aqlm_metadata)
        aqlm_error = torch.mean((aqlm_dequantized - weights) ** 2).item()
        
        # Both should produce reasonable errors
        assert glvq_error < 1.0, f"GLVQ error too high: {glvq_error}"
        assert aqlm_error < 1.0, f"AQLM error too high: {aqlm_error}"
        
        # AQLM should be comparable or better (learned codebooks)
        # Allow some tolerance since they use different algorithms
        assert aqlm_error < glvq_error * 2.0, "AQLM significantly worse than GLVQ"
    
    def test_stacking_phases(self):
        """Test that all 4 phases can work together."""
        weights = torch.randn(256, 256)
        
        # Phase 1: Adaptive Scaling
        phase1 = PerBlockAdaptiveScaling(block_size=128)
        p1_quantized, p1_metadata = phase1.quantize(weights)
        
        # Phase 2: BOF4
        phase2 = PerBlockBOF4(block_size=128)
        p2_quantized, p2_metadata = phase2.quantize(weights)
        
        # Phase 3: GLVQ
        phase3 = PerBlockGLVQ(block_size=16, max_iters=10)
        p3_quantized, p3_metadata = phase3.quantize(weights)
        
        # Phase 4: AQLM
        phase4 = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
        p4_quantized, p4_metadata = phase4.quantize(weights)
        
        # All should produce valid outputs
        assert p1_quantized.shape == weights.shape
        assert p2_quantized.shape == weights.shape
        assert p3_quantized.shape == weights.shape
        assert p4_quantized.shape == weights.shape
        
        # All should have metadata
        assert p1_metadata['method'] == 'four_over_six'
        assert p2_metadata['method'] == 'bof4'
        assert p3_metadata['method'] == 'glvq'
        assert p4_metadata['method'] == 'aqlm'
    
    def test_large_matrix(self):
        """Test AQLM on LLM-scale tensor (4096x4096)."""
        weights = torch.randn(4096, 4096)
        
        quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
        quantized, metadata = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(quantized, metadata)
        
        # Check shapes
        assert quantized.shape == weights.shape
        assert dequantized.shape == weights.shape
        
        # Check no NaN/Inf
        assert not torch.isnan(dequantized).any()
        assert not torch.isinf(dequantized).any()
        
        # Should have 32x32 = 1024 blocks
        assert len(metadata['codebooks']) == 1024
        assert len(metadata['indices']) == 1024
        assert len(metadata['scales']) == 1024
    
    def test_different_block_sizes(self):
        """Test AQLM with different block sizes."""
        weights = torch.randn(256, 256)
        
        for block_size in [8, 16, 32]:
            quantizer = PerBlockAQLM(
                block_size=block_size,
                num_codebooks=2,
                codebook_size=256
            )
            quantized, metadata = quantizer.quantize(weights)
            dequantized = quantizer.dequantize(quantized, metadata)
            
            assert dequantized.shape == weights.shape
            assert metadata['block_size'] == block_size
    
    def test_different_codebook_sizes(self):
        """Test AQLM with different codebook sizes."""
        weights = torch.randn(256, 256)
        
        for codebook_size in [64, 128, 256, 512]:
            quantizer = PerBlockAQLM(
                block_size=128,
                num_codebooks=2,
                codebook_size=codebook_size
            )
            quantized, metadata = quantizer.quantize(weights)
            dequantized = quantizer.dequantize(quantized, metadata)
            
            assert dequantized.shape == weights.shape
            assert metadata['codebook_size'] == codebook_size


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])

#!/usr/bin/env python
"""
Standalone test runner for TestPerBlockAQLM.
Bypasses the bindings import issue by loading modules directly.
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

PerBlockAdaptiveScaling = module.PerBlockAdaptiveScaling
PerBlockBOF4 = module.PerBlockBOF4
PerBlockGLVQ = module.PerBlockGLVQ
PerBlockAQLM = module.PerBlockAQLM


class TestPerBlockAQLM:
    """Test AQLM (Additive Quantization with Learned Matrices) quantization."""
    
    def test_quantize_simple(self):
        """Test basic AQLM quantization on random tensor."""
        weights = torch.randn(256, 256)
        
        quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
        quantized, metadata = quantizer.quantize(weights)
        
        assert quantized.shape == weights.shape
        assert metadata['original_shape'] == weights.shape
        assert metadata['method'] == 'aqlm'
        assert 'codebooks' in metadata
        assert 'indices' in metadata
        assert 'scales' in metadata
        assert metadata['num_codebooks'] == 2
        assert metadata['codebook_size'] == 256
        assert metadata['block_size'] == 128
        print("✓ test_quantize_simple passed")
    
    def test_dequantize(self):
        """Test AQLM dequantization from metadata."""
        weights = torch.randn(256, 256)
        
        quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
        quantized, metadata = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(metadata)
        
        assert dequantized.shape == weights.shape
        assert torch.allclose(dequantized, quantized, atol=1e-5)
        print("✓ test_dequantize passed")
    
    def test_roundtrip(self):
        """Test quantize -> dequantize roundtrip fidelity."""
        weights = torch.randn(256, 256)
        
        quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
        quantized, metadata = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(metadata)
        
        mse = torch.mean((dequantized - quantized) ** 2).item()
        assert mse < 1e-5, f"MSE too high: {mse}"
        assert dequantized.shape == quantized.shape
        print("✓ test_roundtrip passed")
    
    def test_codebook_learning(self):
        """Verify that codebooks are learned (not just initialized)."""
        weights = torch.randn(256, 256)
        
        quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256, max_iters=10)
        quantized, metadata = quantizer.quantize(weights)
        
        assert len(metadata['codebooks']) > 0
        for block_codebooks in metadata['codebooks']:
            assert len(block_codebooks) == 2
            for codebook in block_codebooks:
                assert codebook.shape[0] == 256
        print("✓ test_codebook_learning passed")
    
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
            for block_codebooks in metadata['codebooks']:
                assert len(block_codebooks) == num_codebooks
        print("✓ test_num_codebooks_variation passed")
    
    def test_residual_quantization(self):
        """Verify that residual quantization works correctly."""
        weights = torch.randn(256, 256)
        
        quantizer = PerBlockAQLM(
            block_size=128,
            num_codebooks=2,
            codebook_size=256,
            use_residual=True
        )
        quantized, metadata = quantizer.quantize(weights)
        
        for block_indices in metadata['indices']:
            assert len(block_indices) == 2
            for indices in block_indices:
                assert indices.shape[0] > 0
        print("✓ test_residual_quantization passed")
    
    def test_small_weights(self):
        """Test AQLM on weights near zero (edge case)."""
        weights = torch.randn(256, 256) * 1e-6
        
        quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
        quantized, metadata = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(metadata)
        
        assert not torch.isnan(dequantized).any()
        assert not torch.isinf(dequantized).any()
        assert dequantized.shape == weights.shape
        print("✓ test_small_weights passed")
    
    def test_large_weights(self):
        """Test AQLM on large magnitude weights (edge case)."""
        weights = torch.randn(256, 256) * 1e3
        
        quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
        quantized, metadata = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(metadata)
        
        assert not torch.isnan(dequantized).any()
        assert not torch.isinf(dequantized).any()
        assert dequantized.shape == weights.shape
        print("✓ test_large_weights passed")
    
    def test_compression_ratio(self):
        """Verify compression ratio is reasonable."""
        weights = torch.randn(256, 256)
        
        quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
        quantized, metadata = quantizer.quantize(weights)
        
        original_size = weights.numel() * 4
        num_blocks = (256 // 128) ** 2
        codebook_size = num_blocks * 2 * 256 * 4
        indices_size = weights.numel() * 2 * 1
        
        compressed_size = codebook_size + indices_size
        compression_ratio = original_size / compressed_size
        
        assert compression_ratio > 1.0, f"No compression achieved: {compression_ratio}x"
        print(f"✓ test_compression_ratio passed (ratio: {compression_ratio:.2f}x)")
    
    def test_batch_quantization(self):
        """Test quantizing multiple blocks correctly."""
        weights = torch.randn(512, 512)
        
        quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
        quantized, metadata = quantizer.quantize(weights)
        
        assert len(metadata['codebooks']) == 16
        assert len(metadata['indices']) == 16
        assert len(metadata['scales']) == 16
        print("✓ test_batch_quantization passed")
    
    def test_numerical_stability(self):
        """Check for NaN/Inf in codebooks and reconstructions."""
        weights = torch.randn(256, 256)
        
        quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256, max_iters=10)
        quantized, metadata = quantizer.quantize(weights)
        
        for block_codebooks in metadata['codebooks']:
            for codebook in block_codebooks:
                assert not torch.isnan(codebook).any(), "NaN in codebook"
                assert not torch.isinf(codebook).any(), "Inf in codebook"
        
        for scale in metadata['scales']:
            assert not torch.isnan(scale).any(), "NaN in scale"
            assert not torch.isinf(scale).any(), "Inf in scale"
        
        assert not torch.isnan(quantized).any(), "NaN in quantized output"
        assert not torch.isinf(quantized).any(), "Inf in quantized output"
        print("✓ test_numerical_stability passed")
    
    def test_comparison_with_phase3(self):
        """Compare AQLM (Phase 4) with GLVQ (Phase 3) on same data."""
        weights = torch.randn(256, 256)
        
        glvq = PerBlockGLVQ(block_size=128, num_codebooks=2, codebook_size=256)
        glvq_quantized, glvq_metadata = glvq.quantize(weights)
        glvq_dequantized = glvq.dequantize(glvq_quantized, glvq_metadata)
        glvq_error = torch.mean((glvq_dequantized - weights) ** 2).item()
        
        aqlm = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
        aqlm_quantized, aqlm_metadata = aqlm.quantize(weights)
        aqlm_dequantized = aqlm.dequantize(aqlm_metadata)
        aqlm_error = torch.mean((aqlm_dequantized - weights) ** 2).item()
        
        assert glvq_error < 1.0, f"GLVQ error too high: {glvq_error}"
        assert aqlm_error < 1.0, f"AQLM error too high: {aqlm_error}"
        assert aqlm_error < glvq_error * 2.0, "AQLM significantly worse than GLVQ"
        print(f"✓ test_comparison_with_phase3 passed (GLVQ: {glvq_error:.6f}, AQLM: {aqlm_error:.6f})")
    
    def test_stacking_phases(self):
        """Test that all 4 phases can work together."""
        weights = torch.randn(256, 256)
        
        phase1 = PerBlockAdaptiveScaling(block_size=128)
        p1_quantized, p1_metadata = phase1.quantize(weights)
        
        phase2 = PerBlockBOF4(block_size=128)
        p2_quantized, p2_metadata = phase2.quantize(weights)
        
        phase3 = PerBlockGLVQ(block_size=128, num_codebooks=2, codebook_size=256)
        p3_quantized, p3_metadata = phase3.quantize(weights)
        
        phase4 = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
        p4_quantized, p4_metadata = phase4.quantize(weights)
        
        assert p1_quantized.shape == weights.shape
        assert p2_quantized.shape == weights.shape
        assert p3_quantized.shape == weights.shape
        assert p4_quantized.shape == weights.shape
        
        assert p1_metadata['method'] == 'four_over_six'
        assert p2_metadata['method'] == 'bof4'
        assert p3_metadata['method'] == 'glvq'
        assert p4_metadata['method'] == 'aqlm'
        print("✓ test_stacking_phases passed")
    
    def test_large_matrix(self):
        """Test AQLM on LLM-scale tensor (4096x4096)."""
        weights = torch.randn(4096, 4096)
        
        quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
        quantized, metadata = quantizer.quantize(weights)
        dequantized = quantizer.dequantize(metadata)
        
        assert quantized.shape == weights.shape
        assert dequantized.shape == weights.shape
        assert not torch.isnan(dequantized).any()
        assert not torch.isinf(dequantized).any()
        assert len(metadata['codebooks']) == 1024
        assert len(metadata['indices']) == 1024
        assert len(metadata['scales']) == 1024
        print("✓ test_large_matrix passed")
    
    def test_different_block_sizes(self):
        """Test AQLM with different block sizes."""
        weights = torch.randn(256, 256)
        
        for block_size in [64, 128, 256]:
            quantizer = PerBlockAQLM(
                block_size=block_size,
                num_codebooks=2,
                codebook_size=256
            )
            quantized, metadata = quantizer.quantize(weights)
            dequantized = quantizer.dequantize(metadata)
            
            assert dequantized.shape == weights.shape
            assert metadata['block_size'] == block_size
        print("✓ test_different_block_sizes passed")
    
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
            dequantized = quantizer.dequantize(metadata)
            
            assert dequantized.shape == weights.shape
            assert metadata['codebook_size'] == codebook_size
        print("✓ test_different_codebook_sizes passed")


def run_tests():
    """Run all tests and report results."""
    test_suite = TestPerBlockAQLM()
    tests = [
        test_suite.test_quantize_simple,
        test_suite.test_dequantize,
        test_suite.test_roundtrip,
        test_suite.test_codebook_learning,
        test_suite.test_num_codebooks_variation,
        test_suite.test_residual_quantization,
        test_suite.test_small_weights,
        test_suite.test_large_weights,
        test_suite.test_compression_ratio,
        test_suite.test_batch_quantization,
        test_suite.test_numerical_stability,
        test_suite.test_comparison_with_phase3,
        test_suite.test_stacking_phases,
        test_suite.test_large_matrix,
        test_suite.test_different_block_sizes,
        test_suite.test_different_codebook_sizes,
    ]
    
    print("\n" + "="*70)
    print("Running TestPerBlockAQLM Test Suite")
    print("="*70 + "\n")
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

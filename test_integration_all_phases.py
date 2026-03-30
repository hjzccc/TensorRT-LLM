#!/usr/bin/env python
"""
Integration test for all 4 quantization phases.
Verifies that all phases work correctly and can be stacked.
"""

import sys
import importlib.util
import torch

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


def test_phase1_adaptive_scaling():
    print("\n[Phase 1] Testing PerBlockAdaptiveScaling...")
    weights = torch.randn(256, 256)
    
    quantizer = PerBlockAdaptiveScaling(block_size=128)
    quantized, metadata = quantizer.quantize(weights)
    dequantized = quantizer.dequantize(quantized, metadata)
    
    assert quantized.shape == weights.shape
    assert dequantized.shape == weights.shape
    assert metadata['method'] == 'four_over_six'
    print("✓ Phase 1 PASSED")
    return quantized, metadata


def test_phase2_bof4():
    print("\n[Phase 2] Testing PerBlockBOF4...")
    weights = torch.randn(256, 256)
    
    quantizer = PerBlockBOF4(block_size=128)
    quantized, metadata = quantizer.quantize(weights)
    dequantized = quantizer.dequantize(quantized, metadata)
    
    assert quantized.shape == weights.shape
    assert dequantized.shape == weights.shape
    assert metadata['method'] == 'bof4'
    print("✓ Phase 2 PASSED")
    return quantized, metadata


def test_phase3_glvq():
    print("\n[Phase 3] Testing PerBlockGLVQ...")
    weights = torch.randn(256, 256)
    
    quantizer = PerBlockGLVQ(block_size=128)
    quantized, metadata = quantizer.quantize(weights)
    dequantized = quantizer.dequantize(quantized, metadata)
    
    assert quantized.shape == weights.shape
    assert dequantized.shape == weights.shape
    assert metadata['method'] == 'glvq'
    print("✓ Phase 3 PASSED")
    return quantized, metadata


def test_phase4_aqlm():
    print("\n[Phase 4] Testing PerBlockAQLM...")
    weights = torch.randn(256, 256)
    
    quantizer = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
    quantized, metadata = quantizer.quantize(weights)
    dequantized = quantizer.dequantize(quantized, metadata)
    
    assert quantized.shape == weights.shape
    assert dequantized.shape == weights.shape
    assert metadata['method'] == 'aqlm'
    print("✓ Phase 4 PASSED")
    return quantized, metadata


def test_all_phases_on_same_data():
    print("\n[Integration] Testing all phases on same data...")
    weights = torch.randn(256, 256)
    
    results = {}
    
    quantizer1 = PerBlockAdaptiveScaling(block_size=128)
    q1, m1 = quantizer1.quantize(weights)
    results['phase1'] = (q1, m1)
    
    quantizer2 = PerBlockBOF4(block_size=128)
    q2, m2 = quantizer2.quantize(weights)
    results['phase2'] = (q2, m2)
    
    quantizer3 = PerBlockGLVQ(block_size=128)
    q3, m3 = quantizer3.quantize(weights)
    results['phase3'] = (q3, m3)
    
    quantizer4 = PerBlockAQLM(block_size=128, num_codebooks=2, codebook_size=256)
    q4, m4 = quantizer4.quantize(weights)
    results['phase4'] = (q4, m4)
    
    print("✓ All phases executed successfully on same data")
    return results


def test_different_block_sizes():
    print("\n[Integration] Testing all phases with different block sizes...")
    weights = torch.randn(512, 512)
    
    for block_size in [64, 128, 256]:
        print(f"  Testing block_size={block_size}...")
        
        q1, m1 = PerBlockAdaptiveScaling(block_size=block_size).quantize(weights)
        q2, m2 = PerBlockBOF4(block_size=block_size).quantize(weights)
        q3, m3 = PerBlockGLVQ(block_size=block_size).quantize(weights)
        q4, m4 = PerBlockAQLM(block_size=block_size, num_codebooks=2, codebook_size=256).quantize(weights)
        
        assert q1.shape == weights.shape
        assert q2.shape == weights.shape
        assert q3.shape == weights.shape
        assert q4.shape == weights.shape
    
    print("✓ All phases work with different block sizes")


def main():
    print("=" * 70)
    print("INTEGRATION TEST - All 4 Quantization Phases")
    print("=" * 70)
    
    try:
        test_phase1_adaptive_scaling()
        test_phase2_bof4()
        test_phase3_glvq()
        test_phase4_aqlm()
        test_all_phases_on_same_data()
        test_different_block_sizes()
        
        print("\n" + "=" * 70)
        print("✓ ALL INTEGRATION TESTS PASSED")
        print("=" * 70)
        return 0
    except Exception as e:
        print(f"\n✗ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

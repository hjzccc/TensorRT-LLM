#!/usr/bin/env python
"""
Quick integration test - verifies all 4 phases work on small data.
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


def main():
    print("=" * 70)
    print("QUICK INTEGRATION TEST - All 4 Phases")
    print("=" * 70)
    
    weights = torch.randn(128, 128)
    
    try:
        print("\n[Phase 1] PerBlockAdaptiveScaling...")
        q1 = PerBlockAdaptiveScaling(block_size=64)
        quantized1, meta1 = q1.quantize(weights)
        dequantized1 = q1.dequantize(quantized1, meta1)
        assert quantized1.shape == weights.shape
        assert dequantized1.shape == weights.shape
        print("✓ Phase 1 works")
        
        print("\n[Phase 2] PerBlockBOF4...")
        q2 = PerBlockBOF4(block_size=64)
        quantized2, meta2 = q2.quantize(weights)
        dequantized2 = q2.dequantize(quantized2, meta2)
        assert quantized2.shape == weights.shape
        assert dequantized2.shape == weights.shape
        print("✓ Phase 2 works")
        
        print("\n[Phase 3] PerBlockGLVQ...")
        q3 = PerBlockGLVQ(block_size=64)
        quantized3, meta3 = q3.quantize(weights)
        dequantized3 = q3.dequantize(quantized3, meta3)
        assert quantized3.shape == weights.shape
        assert dequantized3.shape == weights.shape
        print("✓ Phase 3 works")
        
        print("\n[Phase 4] PerBlockAQLM...")
        q4 = PerBlockAQLM(block_size=64, num_codebooks=2, codebook_size=256)
        quantized4, meta4 = q4.quantize(weights)
        dequantized4 = q4.dequantize(quantized4, meta4)
        assert quantized4.shape == weights.shape
        assert dequantized4.shape == weights.shape
        print("✓ Phase 4 works")
        
        print("\n" + "=" * 70)
        print("✓ ALL PHASES WORKING")
        print("=" * 70)
        return 0
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

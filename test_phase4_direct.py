"""
Direct test of Phase 4 AQLM implementation without full TensorRT-LLM imports.
"""
import sys
import torch
import numpy as np

# Import only the quantization module directly
sys.path.insert(0, '/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile')

# Direct import to avoid __init__.py circular dependency
import importlib.util
spec = importlib.util.spec_from_file_location(
    "per_block_codebook",
    "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/tensorrt_llm/quantization/per_block_codebook.py"
)
per_block_codebook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(per_block_codebook)

PerBlockAQLM = per_block_codebook.PerBlockAQLM
PerBlockGLVQ = per_block_codebook.PerBlockGLVQ
PerBlockBOF4 = per_block_codebook.PerBlockBOF4
PerBlockAdaptiveScaling = per_block_codebook.PerBlockAdaptiveScaling

print("=" * 80)
print("PHASE 4 AQLM DIRECT TEST")
print("=" * 80)

# Test 1: Basic instantiation
print("\n[TEST 1] AQLM Instantiation")
try:
    aqlm = PerBlockAQLM(block_size=64, num_codebooks=2, codebook_size=256)
    print(f"✅ AQLM created: {aqlm}")
    print(f"   - block_size: {aqlm.block_size}")
    print(f"   - num_codebooks: {aqlm.num_codebooks}")
    print(f"   - codebook_size: {aqlm.codebook_size}")
except Exception as e:
    print(f"❌ AQLM instantiation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Simple quantization
print("\n[TEST 2] AQLM Quantization (64x64 block)")
try:
    weights = torch.randn(64, 64)
    quantized, metadata = aqlm.quantize(weights)
    print(f"✅ Quantization successful")
    print(f"   - Input shape: {weights.shape}")
    print(f"   - Quantized shape: {quantized.shape}")
    print(f"   - Metadata keys: {list(metadata.keys())}")
except Exception as e:
    print(f"❌ Quantization failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Dequantization
print("\n[TEST 3] AQLM Dequantization")
try:
    reconstructed = aqlm.dequantize(quantized, metadata)
    error = torch.norm(weights - reconstructed) / torch.norm(weights)
    print(f"✅ Dequantization successful")
    print(f"   - Reconstructed shape: {reconstructed.shape}")
    print(f"   - Relative error: {error:.6f}")
except Exception as e:
    print(f"❌ Dequantization failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Larger matrix
print("\n[TEST 4] AQLM on 256x256 matrix")
try:
    weights_large = torch.randn(256, 256)
    quantized_large, metadata_large = aqlm.quantize(weights_large)
    reconstructed_large = aqlm.dequantize(quantized_large, metadata_large)
    error_large = torch.norm(weights_large - reconstructed_large) / torch.norm(weights_large)
    print(f"✅ Large matrix quantization successful")
    print(f"   - Input shape: {weights_large.shape}")
    print(f"   - Relative error: {error_large:.6f}")
except Exception as e:
    print(f"❌ Large matrix test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Compare with Phase 3 (GLVQ)
print("\n[TEST 5] Compare AQLM vs GLVQ")
try:
    glvq = PerBlockGLVQ(block_size=64)
    weights_test = torch.randn(64, 64)
    
    # GLVQ
    glvq_quant, glvq_meta = glvq.quantize(weights_test)
    glvq_recon = glvq.dequantize(glvq_quant, glvq_meta)
    glvq_error = torch.norm(weights_test - glvq_recon) / torch.norm(weights_test)
    
    # AQLM
    aqlm_quant, aqlm_meta = aqlm.quantize(weights_test)
    aqlm_recon = aqlm.dequantize(aqlm_quant, aqlm_meta)
    aqlm_error = torch.norm(weights_test - aqlm_recon) / torch.norm(weights_test)
    
    print(f"✅ Comparison complete")
    print(f"   - GLVQ error: {glvq_error:.6f}")
    print(f"   - AQLM error: {aqlm_error:.6f}")
    print(f"   - AQLM better: {aqlm_error < glvq_error}")
except Exception as e:
    print(f"❌ Comparison failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)

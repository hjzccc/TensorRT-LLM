"""Simple AQLM test with debugging."""
import sys
import torch
import importlib.util

spec = importlib.util.spec_from_file_location(
    "per_block_codebook",
    "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/tensorrt_llm/quantization/per_block_codebook.py"
)
per_block_codebook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(per_block_codebook)

PerBlockAQLM = per_block_codebook.PerBlockAQLM

print("Creating AQLM with small block size...")
aqlm = PerBlockAQLM(block_size=32, num_codebooks=2, codebook_size=16, max_iters=2)
print(f"✅ AQLM created")

print("\nCreating small test weights (32x32)...")
weights = torch.randn(32, 32)
print(f"✅ Weights created: {weights.shape}")

print("\nStarting quantization...")
sys.stdout.flush()
try:
    quantized, metadata = aqlm.quantize(weights)
    print(f"✅ Quantization done: {quantized.shape}")
except Exception as e:
    print(f"❌ Quantization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStarting dequantization...")
sys.stdout.flush()
try:
    reconstructed = aqlm.dequantize(quantized, metadata)
    print(f"✅ Dequantization done: {reconstructed.shape}")
    error = torch.norm(weights - reconstructed) / torch.norm(weights)
    print(f"   Relative error: {error:.6f}")
except Exception as e:
    print(f"❌ Dequantization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ ALL TESTS PASSED")

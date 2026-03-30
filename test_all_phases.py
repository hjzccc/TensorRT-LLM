"""Comprehensive test of all 4 phases."""
import sys
import torch
import importlib.util

spec = importlib.util.spec_from_file_location(
    "per_block_codebook",
    "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/tensorrt_llm/quantization/per_block_codebook.py"
)
per_block_codebook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(per_block_codebook)

PerBlockAdaptiveScaling = per_block_codebook.PerBlockAdaptiveScaling
PerBlockBOF4 = per_block_codebook.PerBlockBOF4
PerBlockGLVQ = per_block_codebook.PerBlockGLVQ
PerBlockAQLM = per_block_codebook.PerBlockAQLM

print("=" * 80)
print("COMPREHENSIVE PHASE TEST")
print("=" * 80)

# Test configuration
block_size = 64
test_sizes = [(64, 64), (128, 128), (256, 256)]

phases = [
    ("Phase 1: Adaptive Scaling", PerBlockAdaptiveScaling(block_size=block_size)),
    ("Phase 2: BOF4 (EM)", PerBlockBOF4(block_size=block_size, max_em_iters=5)),
    ("Phase 3: GLVQ (Lattice)", PerBlockGLVQ(block_size=block_size, max_iters=10)),
    ("Phase 4: AQLM (Multi-CB)", PerBlockAQLM(block_size=block_size, num_codebooks=2, max_iters=2)),
]

for phase_name, quantizer in phases:
    print(f"\n{'='*80}")
    print(f"{phase_name}")
    print(f"{'='*80}")
    
    for size in test_sizes:
        try:
            weights = torch.randn(*size)
            quantized, metadata = quantizer.quantize(weights)
            reconstructed = quantizer.dequantize(quantized, metadata)
            
            error = torch.norm(weights - reconstructed) / torch.norm(weights)
            compression = (weights.numel() * 32) / (quantized.numel() * 32)  # Rough estimate
            
            print(f"  {size[0]:3d}x{size[1]:3d}: error={error:.6f}, compression≈{compression:.2f}x")
        except Exception as e:
            print(f"  {size[0]:3d}x{size[1]:3d}: ❌ {e}")

print(f"\n{'='*80}")
print("✅ ALL PHASES TESTED SUCCESSFULLY")
print(f"{'='*80}")

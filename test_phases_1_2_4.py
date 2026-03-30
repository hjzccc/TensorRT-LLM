"""Test Phases 1, 2, and 4 (skip slow Phase 3)."""
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
PerBlockAQLM = per_block_codebook.PerBlockAQLM

print("=" * 80)
print("PHASE VALIDATION TEST (1, 2, 4)")
print("=" * 80)

test_configs = [
    ("Phase 1: Adaptive Scaling", PerBlockAdaptiveScaling(block_size=64)),
    ("Phase 2: BOF4 (EM)", PerBlockBOF4(block_size=64, max_em_iters=5)),
    ("Phase 4: AQLM (Multi-CB)", PerBlockAQLM(block_size=64, num_codebooks=2, max_iters=3)),
]

test_sizes = [(64, 64), (128, 128), (256, 256)]

results = {}

for phase_name, quantizer in test_configs:
    print(f"\n{phase_name}")
    print("-" * 80)
    results[phase_name] = []
    
    for size in test_sizes:
        try:
            weights = torch.randn(*size)
            quantized, metadata = quantizer.quantize(weights)
            reconstructed = quantizer.dequantize(quantized, metadata)
            
            error = torch.norm(weights - reconstructed) / torch.norm(weights)
            results[phase_name].append((size, error))
            
            print(f"  {size[0]:3d}x{size[1]:3d}: error={error:.6f} ✅")
        except Exception as e:
            print(f"  {size[0]:3d}x{size[1]:3d}: {e} ❌")
            import traceback
            traceback.print_exc()

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

for phase_name, errors in results.items():
    if errors:
        avg_error = sum(e for _, e in errors) / len(errors)
        print(f"{phase_name}: avg_error={avg_error:.6f}")

print("\n✅ VALIDATION COMPLETE")

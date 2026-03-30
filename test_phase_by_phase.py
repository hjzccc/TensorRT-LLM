"""Test each phase individually."""
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

print("Testing Phase 1: Adaptive Scaling")
sys.stdout.flush()
try:
    p1 = PerBlockAdaptiveScaling(block_size=64)
    w = torch.randn(64, 64)
    q, m = p1.quantize(w)
    r = p1.dequantize(q, m)
    err = torch.norm(w - r) / torch.norm(w)
    print(f"✅ Phase 1: error={err:.6f}")
except Exception as e:
    print(f"❌ Phase 1: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting Phase 2: BOF4")
sys.stdout.flush()
try:
    p2 = PerBlockBOF4(block_size=64, max_em_iters=3)
    w = torch.randn(64, 64)
    q, m = p2.quantize(w)
    r = p2.dequantize(q, m)
    err = torch.norm(w - r) / torch.norm(w)
    print(f"✅ Phase 2: error={err:.6f}")
except Exception as e:
    print(f"❌ Phase 2: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting Phase 3: GLVQ (with timeout)")
sys.stdout.flush()
try:
    p3 = PerBlockGLVQ(block_size=64, max_iters=5)  # Reduce iterations
    w = torch.randn(64, 64)
    print("  Quantizing...")
    sys.stdout.flush()
    q, m = p3.quantize(w)
    print("  Dequantizing...")
    sys.stdout.flush()
    r = p3.dequantize(q, m)
    err = torch.norm(w - r) / torch.norm(w)
    print(f"✅ Phase 3: error={err:.6f}")
except Exception as e:
    print(f"❌ Phase 3: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting Phase 4: AQLM")
sys.stdout.flush()
try:
    p4 = PerBlockAQLM(block_size=64, num_codebooks=2, max_iters=2)
    w = torch.randn(64, 64)
    q, m = p4.quantize(w)
    r = p4.dequantize(q, m)
    err = torch.norm(w - r) / torch.norm(w)
    print(f"✅ Phase 4: error={err:.6f}")
except Exception as e:
    print(f"❌ Phase 4: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ PHASE TESTING COMPLETE")

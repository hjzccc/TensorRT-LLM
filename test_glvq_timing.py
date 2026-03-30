"""Test GLVQ timing with different configurations."""
import sys
import torch
import time
import importlib.util

spec = importlib.util.spec_from_file_location(
    "per_block_codebook",
    "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/tensorrt_llm/quantization/per_block_codebook.py"
)
per_block_codebook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(per_block_codebook)

PerBlockGLVQ = per_block_codebook.PerBlockGLVQ

configs = [
    (32, 5),   # block_size=32, max_iters=5
    (32, 10),  # block_size=32, max_iters=10
    (64, 2),   # block_size=64, max_iters=2
]

for block_size, max_iters in configs:
    print(f"\nTesting GLVQ: block_size={block_size}, max_iters={max_iters}")
    sys.stdout.flush()
    
    try:
        glvq = PerBlockGLVQ(block_size=block_size, max_iters=max_iters)
        w = torch.randn(block_size, block_size)
        
        start = time.time()
        q, m = glvq.quantize(w)
        elapsed = time.time() - start
        
        print(f"  ✅ Quantization: {elapsed:.2f}s")
        
        start = time.time()
        r = glvq.dequantize(q, m)
        elapsed = time.time() - start
        
        err = torch.norm(w - r) / torch.norm(w)
        print(f"  ✅ Dequantization: {elapsed:.2f}s, error={err:.6f}")
    except Exception as e:
        print(f"  ❌ {e}")
        import traceback
        traceback.print_exc()

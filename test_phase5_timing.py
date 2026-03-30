"""Analyze timing breakdown of AQLM"""
import torch
import numpy as np
import sys
import importlib.util
import time

spec = importlib.util.spec_from_file_location(
    "per_block_codebook",
    "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/tensorrt_llm/quantization/per_block_codebook.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

PerBlockAQLM = module.PerBlockAQLM

def test_timing():
    """Analyze timing breakdown"""
    print("=" * 80)
    print("TIMING ANALYSIS: AQLM EM ITERATIONS")
    print("=" * 80)
    
    test_shape = (128, 128)
    weights = torch.randn(*test_shape)
    
    # Test different iteration counts
    for max_iters in [1, 2, 3]:
        quantizer = PerBlockAQLM(
            block_size=64,
            num_codebooks=2,
            codebook_size=256,
            max_iters=max_iters
        )
        
        start = time.time()
        q, m = quantizer.quantize(weights)
        r = quantizer.dequantize(q, m)
        total_time = time.time() - start
        
        error = torch.norm(weights - r) / torch.norm(weights)
        
        print(f"\nmax_iters={max_iters}:")
        print(f"  Time: {total_time:.4f}s")
        print(f"  Error: {error.item():.6f}")
        print(f"  Time per iteration: {total_time / max_iters:.4f}s")

if __name__ == "__main__":
    test_timing()

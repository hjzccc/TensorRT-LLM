"""Test single LLM layer"""
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

def test_single_layer():
    """Test single LLM layer"""
    print("=" * 80)
    print("SINGLE LAYER TEST - PHASE 4")
    print("=" * 80)
    
    # Test on a single realistic layer
    shape = (1024, 1024)  # Smaller than full Llama layers
    
    quantizer = PerBlockAQLM(
        block_size=64,
        num_codebooks=2,
        codebook_size=256,
        max_iters=2
    )
    
    print(f"\nTesting on {shape} matrix:")
    
    weights = torch.randn(*shape)
    params = np.prod(shape)
    
    start = time.time()
    q, m = quantizer.quantize(weights)
    r = quantizer.dequantize(q, m)
    layer_time = time.time() - start
    
    error = torch.norm(weights - r) / torch.norm(weights)
    throughput = params / layer_time / 1e6  # M params/sec
    
    print(f"  Error: {error.item():.6f}")
    print(f"  Time: {layer_time:.2f}s")
    print(f"  Throughput: {throughput:.2f}M params/sec")
    print(f"  Estimated time for 4096x4096 layer: {(4096*4096) / throughput:.2f}s")
    print(f"  Estimated time for Llama-7B (6.7B params): {6.7e9 / throughput / 60:.1f} minutes")
    
    if error.item() < 0.01:
        print(f"\n✓ SUCCESS: Error {error.item():.6f} < 1%")
        return True
    else:
        print(f"\n✗ NEEDS WORK: Error {error.item():.6f} >= 1%")
        return False

if __name__ == "__main__":
    success = test_single_layer()
    sys.exit(0 if success else 1)

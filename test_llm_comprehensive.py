"""Comprehensive LLM layer testing for Phase 4"""
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

def test_llm_layers():
    """Test Phase 4 on realistic LLM layers"""
    print("=" * 80)
    print("COMPREHENSIVE LLM LAYER TESTING - PHASE 4")
    print("=" * 80)
    
    # Realistic LLM layer sizes (from Llama-7B)
    layers = {
        'attn_q': (4096, 4096),      # Query projection
        'attn_k': (4096, 4096),      # Key projection
        'attn_v': (4096, 4096),      # Value projection
        'attn_out': (4096, 4096),    # Output projection
        'ffn_up': (4096, 11008),     # FFN up projection
        'ffn_down': (11008, 4096),   # FFN down projection
    }
    
    quantizer = PerBlockAQLM(
        block_size=64,
        num_codebooks=2,
        codebook_size=256,
        max_iters=2
    )
    
    print(f"\n{'Layer':<15} {'Shape':<20} {'Error':<15} {'Time (s)':<15} {'Params':<15}")
    print("-" * 80)
    
    total_params = 0
    total_time = 0
    total_error_sum = 0
    
    for layer_name, shape in layers.items():
        weights = torch.randn(*shape)
        params = np.prod(shape)
        total_params += params
        
        start = time.time()
        q, m = quantizer.quantize(weights)
        r = quantizer.dequantize(q, m)
        layer_time = time.time() - start
        total_time += layer_time
        
        error = torch.norm(weights - r) / torch.norm(weights)
        total_error_sum += error.item()
        
        print(f"{layer_name:<15} {str(shape):<20} {error.item():<15.6f} {layer_time:<15.4f} {params:<15}")
    
    avg_error = total_error_sum / len(layers)
    throughput = total_params / total_time / 1e6  # M params/sec
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average error: {avg_error:.6f}")
    print(f"Throughput: {throughput:.2f}M params/sec")
    print(f"Estimated time for Llama-7B (6.7B params): {6.7e9 / throughput / 60:.1f} minutes")
    
    # Compression analysis
    print(f"\nCompression Analysis:")
    print(f"  Original size: {total_params * 4 / 1e9:.2f} GB (FP32)")
    print(f"  Codebook size: {2 * 256 * 4 / 1e6:.2f} MB (2 codebooks x 256 entries)")
    print(f"  Index size: {total_params * 16 / 8 / 1e9:.2f} GB (16 bits per param)")
    print(f"  Total compressed: {(2 * 256 * 4 + total_params * 16 / 8) / 1e9:.2f} GB")
    print(f"  Compression ratio: {total_params * 4 / ((2 * 256 * 4 + total_params * 16 / 8)):.2f}x")
    
    if avg_error < 0.01:
        print(f"\n✓ SUCCESS: Average error {avg_error:.6f} < 1%")
        return True
    else:
        print(f"\n✗ NEEDS WORK: Average error {avg_error:.6f} >= 1%")
        return False

if __name__ == "__main__":
    success = test_llm_layers()
    sys.exit(0 if success else 1)

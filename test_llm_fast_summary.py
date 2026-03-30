"""
Fast LLM test: Only test Phase 4 (Phase 5c is too slow for multiple layers).
"""
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
    """Test Phase 4 on realistic LLM layer sizes"""
    print("=" * 80)
    print("PHASE 4 AQLM: LLM LAYER QUANTIZATION TEST")
    print("=" * 80)
    
    # Realistic LLM layer sizes
    layers = {
        'attn_q': (256, 256),
        'attn_k': (256, 256),
        'attn_v': (256, 256),
        'attn_o': (256, 256),
        'ffn_up': (256, 1024),
        'ffn_down': (1024, 256),
    }
    
    quantizer = PerBlockAQLM(
        block_size=64,
        num_codebooks=2,
        codebook_size=256,
        max_iters=1
    )
    
    print("\nLayer-wise Quantization Results:")
    print("-" * 80)
    print(f"{'Layer':<20} {'Shape':<15} {'Error':<15} {'Time (s)':<15}")
    print("-" * 80)
    
    errors = []
    times = []
    total_params = 0
    
    for layer_name, shape in layers.items():
        weights = torch.randn(*shape)
        total_params += weights.numel()
        
        start = time.time()
        q, m = quantizer.quantize(weights)
        r = quantizer.dequantize(q, m)
        elapsed = time.time() - start
        
        error = torch.norm(weights - r) / torch.norm(weights)
        errors.append(error.item())
        times.append(elapsed)
        
        print(f"{layer_name:<20} {str(shape):<15} {error.item():<15.6f} {elapsed:<15.2f}")
    
    print("-" * 80)
    print(f"{'AVERAGE':<20} {'':<15} {np.mean(errors):<15.6f} {np.mean(times):<15.2f}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total parameters: {total_params:,}")
    print(f"Average error: {np.mean(errors):.6f}")
    print(f"Average time per layer: {np.mean(times):.2f}s")
    print(f"Total time: {sum(times):.2f}s")
    print(f"Throughput: {total_params / sum(times) / 1e6:.2f}M params/sec")
    
    # Compression analysis
    print("\n" + "=" * 80)
    print("COMPRESSION ANALYSIS")
    print("=" * 80)
    
    # Estimate compression ratio
    # Original: total_params * 4 bytes (float32)
    # Compressed: indices (2 bits per param) + codebooks (256*2*4 bytes per layer)
    original_size = total_params * 4
    compressed_size = total_params * 0.25 + len(layers) * 256 * 2 * 4  # 2 bits per param + codebooks
    ratio = original_size / compressed_size
    
    print(f"Original size: {original_size / 1e6:.2f} MB")
    print(f"Compressed size: {compressed_size / 1e6:.2f} MB")
    print(f"Compression ratio: {ratio:.2f}x")
    print(f"Bits per parameter: {np.log2(ratio) * 32:.2f} bits")
    
    return True

if __name__ == '__main__':
    try:
        success = test_llm_layers()
        print("\n✅ Test completed successfully")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

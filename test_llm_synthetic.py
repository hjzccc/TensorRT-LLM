"""Synthetic LLM testing - Phase 4 on realistic layer sizes"""
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

def test_synthetic_llm():
    """Test Phase 4 on synthetic LLM layers"""
    print("=" * 80)
    print("SYNTHETIC LLM TESTING - PHASE 4 (AQLM)")
    print("=" * 80)
    
    # Realistic layer sizes from Llama-7B
    # Using smaller subset for faster testing
    layers = {
        'attn_q': (4096, 4096),
        'attn_k': (4096, 4096),
        'attn_v': (4096, 4096),
        'attn_o': (4096, 4096),
        'ffn_gate': (4096, 11008),
        'ffn_up': (4096, 11008),
        'ffn_down': (11008, 4096),
    }
    
    total_params = sum(np.prod(shape) for shape in layers.values())
    print(f"\nTesting on {len(layers)} layers with {total_params:,} total parameters")
    print(f"({total_params/1e9:.2f}B params - representative of 1 transformer block)")
    
    quantizer = PerBlockAQLM(
        block_size=64,
        num_codebooks=2,
        codebook_size=256,
        max_iters=2
    )
    
    print(f"\n{'Layer':<15} {'Shape':<20} {'Error':<15} {'Time (s)':<15} {'Params':<15}")
    print("-" * 80)
    
    total_time = 0
    errors = []
    
    for layer_name, shape in layers.items():
        params = np.prod(shape)
        
        # Create synthetic weights with realistic distribution
        weights = torch.randn(*shape, dtype=torch.float32)
        
        start = time.time()
        try:
            q, m = quantizer.quantize(weights)
            r = quantizer.dequantize(q, m)
            layer_time = time.time() - start
            total_time += layer_time
            
            error = torch.norm(weights - r) / torch.norm(weights)
            errors.append(error.item())
            
            print(f"{layer_name:<15} {str(shape):<20} {error.item():<15.6f} {layer_time:<15.4f} {params:<15}")
        except Exception as e:
            print(f"{layer_name:<15} {str(shape):<20} ERROR: {str(e)[:30]}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if errors:
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        min_error = np.min(errors)
        
        print(f"\nError statistics:")
        print(f"  Average: {avg_error:.6f}")
        print(f"  Min: {min_error:.6f}")
        print(f"  Max: {max_error:.6f}")
        print(f"  Std: {np.std(errors):.6f}")
        
        print(f"\nPerformance:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Total time: {total_time:.2f}s")
        throughput = total_params / total_time / 1e6
        print(f"  Throughput: {throughput:.2f}M params/sec")
        
        # Extrapolate to full Llama-7B
        llama_7b_params = 6.7e9
        estimated_time = llama_7b_params / throughput
        print(f"\nExtrapolation to Llama-7B (6.7B params):")
        print(f"  Estimated time: {estimated_time/3600:.1f} hours ({estimated_time/60:.1f} minutes)")
        
        # Compression analysis
        print(f"\nCompression analysis:")
        print(f"  Original size: {total_params * 4 / 1e9:.2f} GB (FP32)")
        print(f"  Codebook size: {2 * 256 * 4 / 1e6:.2f} MB (2 codebooks x 256 entries)")
        print(f"  Index size: {total_params * 16 / 8 / 1e9:.2f} GB (16 bits per param)")
        print(f"  Total compressed: {(2 * 256 * 4 + total_params * 16 / 8) / 1e9:.2f} GB")
        compression_ratio = total_params * 4 / ((2 * 256 * 4 + total_params * 16 / 8))
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        
        # Success criteria
        print(f"\nSuccess criteria:")
        success = True
        if avg_error < 0.01:
            print(f"  ✓ Average error {avg_error:.6f} < 1%")
        else:
            print(f"  ✗ Average error {avg_error:.6f} >= 1%")
            success = False
        
        if max_error < 0.02:
            print(f"  ✓ Max error {max_error:.6f} < 2%")
        else:
            print(f"  ✗ Max error {max_error:.6f} >= 2%")
            success = False
        
        if throughput > 0.005:
            print(f"  ✓ Throughput {throughput:.2f}M params/sec > 0.005M/sec")
        else:
            print(f"  ✗ Throughput {throughput:.2f}M params/sec <= 0.005M/sec")
            success = False
        
        return success
    else:
        print("✗ No layers quantized successfully")
        return False

if __name__ == "__main__":
    success = test_synthetic_llm()
    sys.exit(0 if success else 1)

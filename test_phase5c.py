"""Test Phase 5c: AQLM with Learned Quantization Schedules"""
import torch
import numpy as np
import sys
import importlib.util

# Direct import to avoid circular dependencies
spec = importlib.util.spec_from_file_location(
    "per_block_codebook",
    "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/tensorrt_llm/quantization/per_block_codebook.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

PerBlockAQLM = module.PerBlockAQLM
PerBlockAQLMScheduled = module.PerBlockAQLMScheduled

def test_phase5c_single_layer():
    """Test Phase 5c on single layer"""
    print("=" * 80)
    print("PHASE 5C: AQLM WITH LEARNED QUANTIZATION SCHEDULES")
    print("=" * 80)
    
    # Test different weight distributions
    test_cases = [
        ("normal", torch.randn(256, 256)),
        ("sparse", torch.randn(256, 256) * (torch.rand(256, 256) > 0.7).float()),
        ("outliers", torch.randn(256, 256) + torch.randn(256, 256) * 5),
    ]
    
    phase4_quantizer = PerBlockAQLM(
        block_size=64,
        num_codebooks=2,
        codebook_size=256,
        max_iters=2
    )
    
    phase5c_quantizer = PerBlockAQLMScheduled(
        block_size=64,
        num_codebooks=2,
        base_codebook_size=256,
        max_iters=2
    )
    
    print("\nComparing Phase 4 (fixed) vs Phase 5c (scheduled):\n")
    
    for dist_name, weights in test_cases:
        print(f"Distribution: {dist_name}")
        print("-" * 80)
        
        # Phase 4 baseline
        quantized4, metadata4 = phase4_quantizer.quantize(weights)
        reconstructed4 = phase4_quantizer.dequantize(quantized4, metadata4)
        error4 = torch.norm(weights - reconstructed4) / torch.norm(weights)
        
        # Phase 5c scheduled
        quantized5c, metadata5c = phase5c_quantizer.quantize(weights, layer_name=dist_name)
        reconstructed5c = phase5c_quantizer.dequantize(quantized5c, metadata5c)
        error5c = torch.norm(weights - reconstructed5c) / torch.norm(weights)
        
        # Show statistics
        stats = metadata5c['weight_stats']
        print(f"  Weight stats:")
        print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        print(f"    Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
        print(f"    Kurtosis: {stats['kurtosis']:.4f}, Sparsity: {stats['sparsity']:.4f}")
        
        # Show codebook selection
        selected_size = metadata5c['selected_codebook_size']
        base_size = metadata5c['base_codebook_size']
        print(f"  Codebook size: {selected_size} (base: {base_size})")
        
        # Show errors
        print(f"  Phase 4 error: {error4.item():.6f}")
        print(f"  Phase 5c error: {error5c.item():.6f}")
        
        # Show improvement
        improvement = (error4.item() - error5c.item()) / error4.item() * 100
        print(f"  Improvement: {improvement:+.2f}%")
        print()

def test_phase5c_multi_layer():
    """Test Phase 5c on multiple layers"""
    print("=" * 80)
    print("MULTI-LAYER QUANTIZATION SCHEDULE")
    print("=" * 80)
    
    # Simulate LLM layers with different characteristics
    layers = {
        'embedding': torch.randn(4096, 768),  # Embedding layer
        'attn_q': torch.randn(768, 768),      # Attention query
        'attn_k': torch.randn(768, 768),      # Attention key
        'attn_v': torch.randn(768, 768),      # Attention value
        'ffn_up': torch.randn(768, 3072),     # FFN up-projection
        'ffn_down': torch.randn(3072, 768),   # FFN down-projection
    }
    
    quantizer = PerBlockAQLMScheduled(
        block_size=64,
        num_codebooks=2,
        base_codebook_size=256,
        max_iters=2
    )
    
    # Analyze schedule
    schedule = quantizer.analyze_layer_schedule(layers)
    
    print("\nLayer-wise Quantization Schedule:")
    print("-" * 80)
    print(f"{'Layer':<20} {'Codebook':<12} {'Bits/Param':<12} {'Kurtosis':<12}")
    print("-" * 80)
    
    total_bits = 0
    for layer_name, info in schedule.items():
        cb_size = info['codebook_size']
        bits = info['bits_per_param']
        kurtosis = info['stats']['kurtosis']
        print(f"{layer_name:<20} {cb_size:<12} {bits:<12.2f} {kurtosis:<12.2f}")
        total_bits += bits
    
    print("-" * 80)
    print(f"{'Average':<20} {'':<12} {total_bits/len(layers):<12.2f}")
    print()

if __name__ == '__main__':
    try:
        test_phase5c_single_layer()
        test_phase5c_multi_layer()
        print("✅ Phase 5c tests completed successfully")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

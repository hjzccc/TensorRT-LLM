"""
Fast test on simulated LLM weights (smaller scale).
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
PerBlockAQLMScheduled = module.PerBlockAQLMScheduled

def test_real_llm_fast():
    """Fast test on simulated LLM"""
    print("=" * 80)
    print("FAST LLM SIMULATION TEST")
    print("=" * 80)
    
    # Create smaller simulated LLM weights (for speed)
    weights_dict = {
        'embedding': torch.randn(4096, 256) * 0.1,
        'attn_q': torch.randn(256, 256) * 0.01,
        'attn_k': torch.randn(256, 256) * 0.01,
        'attn_v': torch.randn(256, 256) * 0.01,
        'ffn_up': torch.randn(256, 1024) * 0.02,
        'ffn_down': torch.randn(1024, 256) * 0.02,
    }
    
    total_params = sum(w.numel() for w in weights_dict.values())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Number of layers: {len(weights_dict)}")
    
    # Initialize quantizers
    phase4_quantizer = PerBlockAQLM(
        block_size=64,
        num_codebooks=2,
        codebook_size=256,
        max_iters=1  # Reduce iterations for speed
    )
    
    phase5c_quantizer = PerBlockAQLMScheduled(
        block_size=64,
        num_codebooks=2,
        base_codebook_size=256,
        max_iters=1
    )
    
    # Test Phase 4
    print("\n" + "=" * 80)
    print("PHASE 4: AQLM (BASELINE)")
    print("=" * 80)
    
    phase4_errors = {}
    phase4_start = time.time()
    
    for layer_name, weights in weights_dict.items():
        print(f"  {layer_name}...", end=" ", flush=True)
        quantized, metadata = phase4_quantizer.quantize(weights)
        reconstructed = phase4_quantizer.dequantize(quantized, metadata)
        error = torch.norm(weights - reconstructed) / torch.norm(weights)
        phase4_errors[layer_name] = error.item()
        print(f"error={error.item():.6f}")
    
    phase4_time = time.time() - phase4_start
    phase4_avg_error = np.mean(list(phase4_errors.values()))
    
    print(f"\nPhase 4 time: {phase4_time:.2f}s")
    print(f"Phase 4 avg error: {phase4_avg_error:.6f}")
    
    # Test Phase 5c
    print("\n" + "=" * 80)
    print("PHASE 5C: AQLM WITH LEARNED SCHEDULES")
    print("=" * 80)
    
    phase5c_errors = {}
    phase5c_start = time.time()
    
    for layer_name, weights in weights_dict.items():
        print(f"  {layer_name}...", end=" ", flush=True)
        quantized, metadata = phase5c_quantizer.quantize(weights, layer_name=layer_name)
        reconstructed = phase5c_quantizer.dequantize(quantized, metadata)
        error = torch.norm(weights - reconstructed) / torch.norm(weights)
        phase5c_errors[layer_name] = error.item()
        cb_size = metadata['selected_codebook_size']
        print(f"error={error.item():.6f}, cb_size={cb_size}")
    
    phase5c_time = time.time() - phase5c_start
    phase5c_avg_error = np.mean(list(phase5c_errors.values()))
    
    print(f"\nPhase 5c time: {phase5c_time:.2f}s")
    print(f"Phase 5c avg error: {phase5c_avg_error:.6f}")
    
    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    error_improvement = (phase4_avg_error - phase5c_avg_error) / phase4_avg_error * 100
    time_overhead = (phase5c_time - phase4_time) / phase4_time * 100
    
    print(f"Error improvement: {error_improvement:+.2f}%")
    print(f"Time overhead: {time_overhead:+.2f}%")
    
    # Per-layer analysis
    print("\n" + "=" * 80)
    print("PER-LAYER ANALYSIS")
    print("=" * 80)
    print(f"{'Layer':<20} {'Phase 4':<15} {'Phase 5c':<15} {'Improvement':<15}")
    print("-" * 80)
    
    for layer_name in sorted(weights_dict.keys()):
        e4 = phase4_errors[layer_name]
        e5c = phase5c_errors[layer_name]
        improvement = (e4 - e5c) / e4 * 100
        print(f"{layer_name:<20} {e4:<15.6f} {e5c:<15.6f} {improvement:<15.2f}%")
    
    print("\n✅ Test completed successfully")
    return True

if __name__ == '__main__':
    try:
        success = test_real_llm_fast()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

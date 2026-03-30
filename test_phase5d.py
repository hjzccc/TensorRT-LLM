"""Test Phase 5d: Faster EM Optimization"""
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
PerBlockAQLMFast = module.PerBlockAQLMFast

def test_phase5d():
    """Test Phase 5d speedup"""
    print("=" * 80)
    print("PHASE 5D: FASTER EM OPTIMIZATION")
    print("=" * 80)
    
    # Test on realistic LLM layers
    layers = {
        'attn_q': (256, 256),
        'attn_k': (256, 256),
        'ffn_up': (256, 1024),
    }
    
    # Phase 4 baseline (2 iterations)
    phase4_quantizer = PerBlockAQLM(
        block_size=64,
        num_codebooks=2,
        codebook_size=256,
        max_iters=2
    )
    
    # Phase 5d fast (1 iteration)
    phase5d_quantizer = PerBlockAQLMFast(
        block_size=64,
        num_codebooks=2,
        codebook_size=256,
        max_iters=1
    )
    
    print("\nComparing Phase 4 (2 iters) vs Phase 5d (1 iter):\n")
    print(f"{'Layer':<20} {'Phase 4':<20} {'Phase 5d':<20} {'Speedup':<15} {'Error Diff':<15}")
    print("-" * 90)
    
    phase4_times = []
    phase5d_times = []
    phase4_errors = []
    phase5d_errors = []
    
    for layer_name, shape in layers.items():
        weights = torch.randn(*shape)
        
        # Phase 4
        start = time.time()
        q4, m4 = phase4_quantizer.quantize(weights)
        r4 = phase4_quantizer.dequantize(q4, m4)
        time4 = time.time() - start
        error4 = torch.norm(weights - r4) / torch.norm(weights)
        
        # Phase 5d
        start = time.time()
        q5d, m5d = phase5d_quantizer.quantize(weights)
        r5d = phase5d_quantizer.dequantize(q5d, m5d)
        time5d = time.time() - start
        error5d = torch.norm(weights - r5d) / torch.norm(weights)
        
        phase4_times.append(time4)
        phase5d_times.append(time5d)
        phase4_errors.append(error4.item())
        phase5d_errors.append(error5d.item())
        
        speedup = time4 / time5d
        error_diff = (error5d.item() - error4.item()) / error4.item() * 100
        
        print(f"{layer_name:<20} {time4:<20.2f}s {time5d:<20.2f}s {speedup:<15.2f}x {error_diff:<15.2f}%")
    
    print("-" * 90)
    avg_speedup = np.mean(np.array(phase4_times) / np.array(phase5d_times))
    avg_error_diff = np.mean(np.array(phase5d_errors) / np.array(phase4_errors) - 1) * 100
    
    print(f"{'AVERAGE':<20} {np.mean(phase4_times):<20.2f}s {np.mean(phase5d_times):<20.2f}s {avg_speedup:<15.2f}x {avg_error_diff:<15.2f}%")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Phase 4 avg time: {np.mean(phase4_times):.2f}s")
    print(f"Phase 5d avg time: {np.mean(phase5d_times):.2f}s")
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Average error increase: {avg_error_diff:+.2f}%")
    
    # Estimate total speedup for full LLM
    total_params = sum(np.prod(shape) for shape in layers.values())
    phase4_total = np.mean(phase4_times) / np.prod(layers['attn_q']) * total_params
    phase5d_total = np.mean(phase5d_times) / np.prod(layers['attn_q']) * total_params
    
    print(f"\nEstimated time for {total_params:,} params:")
    print(f"  Phase 4: {phase4_total:.2f}s")
    print(f"  Phase 5d: {phase5d_total:.2f}s")
    print(f"  Speedup: {phase4_total / phase5d_total:.2f}x")
    
    if avg_error_diff < 5:
        print(f"\n✅ Phase 5d achieves {avg_speedup:.2f}x speedup with only {avg_error_diff:+.2f}% error increase")
    else:
        print(f"\n⚠️  Phase 5d achieves {avg_speedup:.2f}x speedup but {avg_error_diff:+.2f}% error increase")
    
    return True

if __name__ == '__main__':
    try:
        success = test_phase5d()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

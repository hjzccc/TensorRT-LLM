"""Test Phase 5e: Learned Codebook Initialization (Warm-Start)"""
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
PerBlockAQLMWarmStart = module.PerBlockAQLMWarmStart
PerBlockAQLMWarmStartOptimized = module.PerBlockAQLMWarmStartOptimized

def test_phase5e():
    """Test Phase 5e warm-start initialization"""
    print("=" * 80)
    print("PHASE 5E: LEARNED CODEBOOK INITIALIZATION (WARM-START)")
    print("=" * 80)
    
    # Test on realistic LLM layers
    layers = {
        'attn_q': (256, 256),
        'attn_k': (256, 256),
        'ffn_up': (256, 1024),
    }
    
    # Phase 4 baseline (2 iterations, no warm-start)
    phase4_quantizer = PerBlockAQLM(
        block_size=64,
        num_codebooks=2,
        codebook_size=256,
        max_iters=2
    )
    
    # Phase 5d fast (1 iteration, no warm-start) - problematic
    phase5d_quantizer = PerBlockAQLMFast(
        block_size=64,
        num_codebooks=2,
        codebook_size=256,
        max_iters=1
    )
    
    # Phase 5e warm-start (1 iteration with BOF4 initialization)
    phase5e_quantizer = PerBlockAQLMWarmStart(
        block_size=64,
        num_codebooks=2,
        codebook_size=256,
        max_iters=1
    )
    
    # Phase 5e optimized (1 full iteration with warm-start)
    phase5e_opt_quantizer = PerBlockAQLMWarmStartOptimized(
        block_size=64,
        num_codebooks=2,
        codebook_size=256,
        max_iters=2
    )
    
    print("\nComparing Phase 4 vs Phase 5d vs Phase 5e vs Phase 5e-Opt:\n")
    print(f"{'Layer':<15} {'Phase 4':<15} {'Phase 5d':<15} {'Phase 5e':<15} {'Phase 5e-Opt':<15}")
    print(f"{'':15} {'(2 iter)':<15} {'(1 iter)':<15} {'(1 iter WS)':<15} {'(1 iter WS)':<15}")
    print("-" * 75)
    
    phase4_times = []
    phase5d_times = []
    phase5e_times = []
    phase5e_opt_times = []
    
    phase4_errors = []
    phase5d_errors = []
    phase5e_errors = []
    phase5e_opt_errors = []
    
    for layer_name, shape in layers.items():
        weights = torch.randn(*shape)
        
        # Phase 4 (baseline)
        start = time.time()
        q4, m4 = phase4_quantizer.quantize(weights)
        r4 = phase4_quantizer.dequantize(q4, m4)
        time4 = time.time() - start
        error4 = torch.norm(weights - r4) / torch.norm(weights)
        
        # Phase 5d (problematic)
        start = time.time()
        q5d, m5d = phase5d_quantizer.quantize(weights)
        r5d = phase5d_quantizer.dequantize(q5d, m5d)
        time5d = time.time() - start
        error5d = torch.norm(weights - r5d) / torch.norm(weights)
        
        # Phase 5e (warm-start)
        start = time.time()
        q5e, m5e = phase5e_quantizer.quantize(weights)
        r5e = phase5e_quantizer.dequantize(q5e, m5e)
        time5e = time.time() - start
        error5e = torch.norm(weights - r5e) / torch.norm(weights)
        
        # Phase 5e optimized
        start = time.time()
        q5e_opt, m5e_opt = phase5e_opt_quantizer.quantize(weights)
        r5e_opt = phase5e_opt_quantizer.dequantize(q5e_opt, m5e_opt)
        time5e_opt = time.time() - start
        error5e_opt = torch.norm(weights - r5e_opt) / torch.norm(weights)
        
        phase4_times.append(time4)
        phase5d_times.append(time5d)
        phase5e_times.append(time5e)
        phase5e_opt_times.append(time5e_opt)
        
        phase4_errors.append(error4.item())
        phase5d_errors.append(error5d.item())
        phase5e_errors.append(error5e.item())
        phase5e_opt_errors.append(error5e_opt.item())
        
        print(f"{layer_name:<15} {error4.item():<15.6f} {error5d.item():<15.6f} {error5e.item():<15.6f} {error5e_opt.item():<15.6f}")
    
    # Summary statistics
    print("\n" + "=" * 75)
    print("SUMMARY STATISTICS")
    print("=" * 75)
    
    avg_error4 = np.mean(phase4_errors)
    avg_error5d = np.mean(phase5d_errors)
    avg_error5e = np.mean(phase5e_errors)
    avg_error5e_opt = np.mean(phase5e_opt_errors)
    
    avg_time4 = np.mean(phase4_times)
    avg_time5d = np.mean(phase5d_times)
    avg_time5e = np.mean(phase5e_times)
    avg_time5e_opt = np.mean(phase5e_opt_times)
    
    print(f"\nAverage Error:")
    print(f"  Phase 4 (baseline):     {avg_error4:.6f}")
    print(f"  Phase 5d (1 iter):      {avg_error5d:.6f} ({(avg_error5d/avg_error4 - 1)*100:+.2f}%)")
    print(f"  Phase 5e (warm-start):  {avg_error5e:.6f} ({(avg_error5e/avg_error4 - 1)*100:+.2f}%)")
    print(f"  Phase 5e-Opt:           {avg_error5e_opt:.6f} ({(avg_error5e_opt/avg_error4 - 1)*100:+.2f}%)")
    
    print(f"\nAverage Time (seconds):")
    print(f"  Phase 4 (baseline):     {avg_time4:.4f}s")
    print(f"  Phase 5d (1 iter):      {avg_time5d:.4f}s (speedup: {avg_time4/avg_time5d:.2f}x)")
    print(f"  Phase 5e (warm-start):  {avg_time5e:.4f}s (speedup: {avg_time4/avg_time5e:.2f}x)")
    print(f"  Phase 5e-Opt:           {avg_time5e_opt:.4f}s (speedup: {avg_time4/avg_time5e_opt:.2f}x)")
    
    print(f"\nKey Findings:")
    print(f"  Phase 5d Problem: +{(avg_error5d/avg_error4 - 1)*100:.2f}% error increase (unacceptable)")
    print(f"  Phase 5e Solution: {(avg_error5e/avg_error4 - 1)*100:+.2f}% error change with {avg_time4/avg_time5e:.2f}x speedup")
    print(f"  Phase 5e-Opt: {(avg_error5e_opt/avg_error4 - 1)*100:+.2f}% error change with {avg_time4/avg_time5e_opt:.2f}x speedup")
    
    # Determine success
    print("\n" + "=" * 75)
    if avg_error5e < avg_error4 * 1.01:  # Allow 1% error increase
        print("✓ PHASE 5E SUCCESS: Warm-start achieves <1% error increase with speedup!")
        return True
    else:
        print("✗ PHASE 5E NEEDS WORK: Error increase too high")
        return False

if __name__ == "__main__":
    success = test_phase5e()
    sys.exit(0 if success else 1)

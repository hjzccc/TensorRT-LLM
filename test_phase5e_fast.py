"""Test Phase 5e: Fast version with smaller matrices"""
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

def test_phase5e_fast():
    """Test Phase 5e with smaller matrices for speed"""
    print("=" * 80)
    print("PHASE 5E: LEARNED CODEBOOK INITIALIZATION (FAST TEST)")
    print("=" * 80)
    
    # Use smaller matrices for faster testing
    test_shape = (128, 128)  # Smaller than before
    
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
    
    # Phase 5e (1 iteration with improved init)
    phase5e_quantizer = PerBlockAQLMWarmStart(
        block_size=64,
        num_codebooks=2,
        codebook_size=256,
        max_iters=1
    )
    
    print(f"\nTesting on {test_shape} matrix:\n")
    print(f"{'Method':<25} {'Error':<15} {'Time (s)':<15} {'vs Phase 4':<15}")
    print("-" * 70)
    
    weights = torch.randn(*test_shape)
    
    # Phase 4
    start = time.time()
    q4, m4 = phase4_quantizer.quantize(weights)
    r4 = phase4_quantizer.dequantize(q4, m4)
    time4 = time.time() - start
    error4 = torch.norm(weights - r4) / torch.norm(weights)
    
    print(f"{'Phase 4 (2 iter)':<25} {error4.item():<15.6f} {time4:<15.4f} {'baseline':<15}")
    
    # Phase 5d
    start = time.time()
    q5d, m5d = phase5d_quantizer.quantize(weights)
    r5d = phase5d_quantizer.dequantize(q5d, m5d)
    time5d = time.time() - start
    error5d = torch.norm(weights - r5d) / torch.norm(weights)
    
    error_diff_5d = (error5d.item() / error4.item() - 1) * 100
    print(f"{'Phase 5d (1 iter)':<25} {error5d.item():<15.6f} {time5d:<15.4f} {error_diff_5d:+.2f}%")
    
    # Phase 5e
    start = time.time()
    q5e, m5e = phase5e_quantizer.quantize(weights)
    r5e = phase5e_quantizer.dequantize(q5e, m5e)
    time5e = time.time() - start
    error5e = torch.norm(weights - r5e) / torch.norm(weights)
    
    error_diff_5e = (error5e.item() / error4.item() - 1) * 100
    speedup_5e = time4 / time5e
    print(f"{'Phase 5e (1 iter WS)':<25} {error5e.item():<15.6f} {time5e:<15.4f} {error_diff_5e:+.2f}%")
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    print(f"\nPhase 5d Problem:")
    print(f"  Error increase: {error_diff_5d:+.2f}% (unacceptable)")
    print(f"  Speedup: {time4/time5d:.2f}x")
    
    print(f"\nPhase 5e Solution:")
    print(f"  Error change: {error_diff_5e:+.2f}%")
    print(f"  Speedup: {speedup_5e:.2f}x")
    
    if error_diff_5e < 1.0:  # Less than 1% error increase
        print(f"\n✓ PHASE 5E SUCCESS: Achieves {speedup_5e:.2f}x speedup with {error_diff_5e:+.2f}% error change!")
        return True
    else:
        print(f"\n✗ PHASE 5E NEEDS WORK: Error increase {error_diff_5e:+.2f}% is too high")
        return False

if __name__ == "__main__":
    success = test_phase5e_fast()
    sys.exit(0 if success else 1)

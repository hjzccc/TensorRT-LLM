"""Test Phase 5b: AQLM with Adaptive Block Sizes"""
import torch
import numpy as np
import sys
import importlib.util
import time

# Direct import to avoid circular dependencies
spec = importlib.util.spec_from_file_location(
    "per_block_codebook",
    "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/tensorrt_llm/quantization/per_block_codebook.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

PerBlockAQLM = module.PerBlockAQLM
PerBlockAQLMAdaptive = module.PerBlockAQLMAdaptive

def test_phase5b():
    """Test Phase 5b quantize/dequantize roundtrip"""
    print("=" * 80)
    print("PHASE 5B: AQLM WITH ADAPTIVE BLOCK SIZES")
    print("=" * 80)
    
    # Test different sizes
    test_cases = [
        (64, 64),
        (128, 128),
        (256, 256),
    ]
    
    # Phase 4 baseline
    phase4_quantizer = PerBlockAQLM(
        block_size=64,
        num_codebooks=2,
        codebook_size=256,
        max_iters=2
    )
    
    # Phase 5b adaptive
    phase5b_quantizer = PerBlockAQLMAdaptive(
        base_block_size=64,
        num_codebooks=2,
        codebook_size=256,
        max_iters=2,
        error_threshold_low=0.003,
        error_threshold_high=0.006
    )
    
    print("\nComparing Phase 4 (baseline) vs Phase 5b (adaptive):\n")
    
    phase4_errors = []
    phase5b_errors = []
    
    for m, n in test_cases:
        print(f"Testing {m}x{n}...")
        
        # Create random weights
        weights = torch.randn(m, n, dtype=torch.float32)
        
        # Phase 4 baseline
        try:
            quantized4, metadata4 = phase4_quantizer.quantize(weights)
            reconstructed4 = phase4_quantizer.dequantize(quantized4, metadata4)
            error4 = torch.norm(weights - reconstructed4) / torch.norm(weights)
            phase4_errors.append(error4.item())
            print(f"  Phase 4: error={error4.item():.6f}")
        except Exception as e:
            print(f"  Phase 4 failed: {e}")
            return False
        
        # Phase 5b adaptive
        try:
            print(f"  Phase 5b: analyzing block errors...", end=" ", flush=True)
            start_time = time.time()
            quantized5b, metadata5b = phase5b_quantizer.quantize(weights)
            analyze_time = time.time() - start_time
            print(f"({analyze_time:.2f}s)", flush=True)
            
            print(f"  Phase 5b: dequantizing...", end=" ", flush=True)
            start_time = time.time()
            reconstructed5b = phase5b_quantizer.dequantize(quantized5b, metadata5b)
            deq_time = time.time() - start_time
            print(f"({deq_time:.2f}s)", flush=True)
            
            error5b = torch.norm(weights - reconstructed5b) / torch.norm(weights)
            phase5b_errors.append(error5b.item())
            print(f"  Phase 5b: error={error5b.item():.6f}")
            
            # Show adaptive block size distribution
            adaptive_sizes = metadata5b['adaptive_sizes']
            size_32 = (adaptive_sizes == 32).sum().item()
            size_64 = (adaptive_sizes == 64).sum().item()
            size_128 = (adaptive_sizes == 128).sum().item()
            print(f"  Block sizes: 32x{size_32} | 64x{size_64} | 128x{size_128}")
            
            # Show improvement
            improvement = (error4.item() - error5b.item()) / error4.item() * 100
            print(f"  Improvement: {improvement:+.2f}%")
        except Exception as e:
            print(f"  Phase 5b failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Phase 4 (baseline) avg error: {np.mean(phase4_errors):.6f}")
    print(f"Phase 5b (adaptive) avg error: {np.mean(phase5b_errors):.6f}")
    
    improvement = (np.mean(phase4_errors) - np.mean(phase5b_errors)) / np.mean(phase4_errors) * 100
    print(f"Overall improvement: {improvement:+.2f}%")
    
    if improvement > 0:
        print(f"✅ Phase 5b shows {improvement:.2f}% accuracy improvement!")
    else:
        print(f"⚠️  Phase 5b shows {improvement:.2f}% accuracy change (expected: +5-15%)")
    
    return True

if __name__ == '__main__':
    success = test_phase5b()
    sys.exit(0 if success else 1)

"""
Summary test: Compare Phase 4 vs Phase 5c on realistic layer sizes.
Uses smaller matrices to avoid timeout.
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

def test_llm_layers():
    """Test on realistic LLM layer sizes"""
    print("=" * 80)
    print("LLM LAYER QUANTIZATION TEST")
    print("=" * 80)
    
    # Realistic LLM layer sizes (but smaller for speed)
    layers = {
        'attn_q': (256, 256),      # Attention query
        'attn_k': (256, 256),      # Attention key
        'attn_v': (256, 256),      # Attention value
        'attn_o': (256, 256),      # Attention output
        'ffn_up': (256, 1024),     # FFN up-projection
        'ffn_down': (1024, 256),   # FFN down-projection
    }
    
    phase4_quantizer = PerBlockAQLM(
        block_size=64,
        num_codebooks=2,
        codebook_size=256,
        max_iters=1
    )
    
    phase5c_quantizer = PerBlockAQLMScheduled(
        block_size=64,
        num_codebooks=2,
        base_codebook_size=256,
        max_iters=1
    )
    
    print("\nLayer-wise Quantization Results:")
    print("-" * 80)
    print(f"{'Layer':<20} {'Shape':<15} {'Phase 4':<15} {'Phase 5c':<15} {'Improvement':<15}")
    print("-" * 80)
    
    phase4_errors = []
    phase5c_errors = []
    
    for layer_name, shape in layers.items():
        weights = torch.randn(*shape)
        
        # Phase 4
        q4, m4 = phase4_quantizer.quantize(weights)
        r4 = phase4_quantizer.dequantize(q4, m4)
        e4 = torch.norm(weights - r4) / torch.norm(weights)
        phase4_errors.append(e4.item())
        
        # Phase 5c
        q5c, m5c = phase5c_quantizer.quantize(weights, layer_name=layer_name)
        r5c = phase5c_quantizer.dequantize(q5c, m5c)
        e5c = torch.norm(weights - r5c) / torch.norm(weights)
        phase5c_errors.append(e5c.item())
        
        improvement = (e4.item() - e5c.item()) / e4.item() * 100
        
        print(f"{layer_name:<20} {str(shape):<15} {e4.item():<15.6f} {e5c.item():<15.6f} {improvement:<15.2f}%")
    
    print("-" * 80)
    print(f"{'AVERAGE':<20} {'':<15} {np.mean(phase4_errors):<15.6f} {np.mean(phase5c_errors):<15.6f} {(np.mean(phase4_errors) - np.mean(phase5c_errors)) / np.mean(phase4_errors) * 100:<15.2f}%")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Phase 4 average error: {np.mean(phase4_errors):.6f}")
    print(f"Phase 5c average error: {np.mean(phase5c_errors):.6f}")
    print(f"Overall improvement: {(np.mean(phase4_errors) - np.mean(phase5c_errors)) / np.mean(phase4_errors) * 100:+.2f}%")
    
    # Codebook size distribution
    print("\n" + "=" * 80)
    print("CODEBOOK SIZE DISTRIBUTION (Phase 5c)")
    print("=" * 80)
    
    cb_sizes = {}
    for layer_name, shape in layers.items():
        weights = torch.randn(*shape)
        _, metadata = phase5c_quantizer.quantize(weights, layer_name=layer_name)
        cb_size = metadata['selected_codebook_size']
        cb_sizes[layer_name] = cb_size
        print(f"{layer_name:<20} codebook_size={cb_size}")
    
    print(f"\nAverage codebook size: {np.mean(list(cb_sizes.values())):.0f}")
    print(f"Compression improvement potential: {(256 - np.mean(list(cb_sizes.values()))) / 256 * 100:+.2f}%")
    
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

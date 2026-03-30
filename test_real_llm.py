"""Real LLM testing - Phase 4 on TinyLlama-1.1B"""
import torch
import numpy as np
import sys
import importlib.util
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

spec = importlib.util.spec_from_file_location(
    "per_block_codebook",
    "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/tensorrt_llm/quantization/per_block_codebook.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

PerBlockAQLM = module.PerBlockAQLM

def test_real_llm():
    """Test Phase 4 on real LLM"""
    print("=" * 80)
    print("REAL LLM TESTING - PHASE 4 (AQLM)")
    print("=" * 80)
    
    # Load TinyLlama-1.1B
    print("\nLoading TinyLlama-1.1B...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cuda"
        )
        print(f"✓ Loaded {model_name}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Get model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel info:")
    print(f"  Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"  Device: {next(model.parameters()).device}")
    
    # Extract and quantize a few layers
    print(f"\nQuantizing sample layers...")
    
    quantizer = PerBlockAQLM(
        block_size=64,
        num_codebooks=2,
        codebook_size=256,
        max_iters=2
    )
    
    # Get attention and FFN layers
    layers_to_test = []
    for i, layer in enumerate(model.model.layers[:2]):  # Test first 2 layers
        # Attention layers
        if hasattr(layer.self_attn, 'q_proj'):
            layers_to_test.append(('attn_q', layer.self_attn.q_proj.weight, i))
        if hasattr(layer.self_attn, 'k_proj'):
            layers_to_test.append(('attn_k', layer.self_attn.k_proj.weight, i))
        if hasattr(layer.self_attn, 'v_proj'):
            layers_to_test.append(('attn_v', layer.self_attn.v_proj.weight, i))
        if hasattr(layer.self_attn, 'o_proj'):
            layers_to_test.append(('attn_o', layer.self_attn.o_proj.weight, i))
        
        # FFN layers
        if hasattr(layer.mlp, 'gate_proj'):
            layers_to_test.append(('ffn_gate', layer.mlp.gate_proj.weight, i))
        if hasattr(layer.mlp, 'up_proj'):
            layers_to_test.append(('ffn_up', layer.mlp.up_proj.weight, i))
        if hasattr(layer.mlp, 'down_proj'):
            layers_to_test.append(('ffn_down', layer.mlp.down_proj.weight, i))
    
    print(f"\n{'Layer':<20} {'Shape':<20} {'Error':<15} {'Time (s)':<15}")
    print("-" * 70)
    
    total_params_tested = 0
    total_time = 0
    errors = []
    
    for layer_name, weight, layer_idx in layers_to_test[:6]:  # Test first 6 layers
        shape = weight.shape
        params = np.prod(shape)
        total_params_tested += params
        
        # Move to CPU for quantization
        weight_cpu = weight.cpu().float()
        
        start = time.time()
        try:
            q, m = quantizer.quantize(weight_cpu)
            r = quantizer.dequantize(q, m)
            layer_time = time.time() - start
            total_time += layer_time
            
            error = torch.norm(weight_cpu - r) / torch.norm(weight_cpu)
            errors.append(error.item())
            
            print(f"{layer_name:<20} {str(shape):<20} {error.item():<15.6f} {layer_time:<15.4f}")
        except Exception as e:
            print(f"{layer_name:<20} {str(shape):<20} ERROR: {str(e)[:30]}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if errors:
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        min_error = np.min(errors)
        
        print(f"\nError statistics:")
        print(f"  Average: {avg_error:.6f}")
        print(f"  Min: {min_error:.6f}")
        print(f"  Max: {max_error:.6f}")
        
        print(f"\nPerformance:")
        print(f"  Total parameters tested: {total_params_tested:,}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {total_params_tested / total_time / 1e6:.2f}M params/sec")
        
        # Extrapolate to full model
        estimated_time = total_params / (total_params_tested / total_time)
        print(f"\nExtrapolation to full model:")
        print(f"  Estimated time for {total_params/1e9:.2f}B params: {estimated_time/3600:.1f} hours")
        
        # Success criteria
        print(f"\nSuccess criteria:")
        if avg_error < 0.01:
            print(f"  ✓ Average error {avg_error:.6f} < 1%")
            return True
        else:
            print(f"  ✗ Average error {avg_error:.6f} >= 1%")
            return False
    else:
        print("✗ No layers quantized successfully")
        return False

if __name__ == "__main__":
    success = test_real_llm()
    sys.exit(0 if success else 1)

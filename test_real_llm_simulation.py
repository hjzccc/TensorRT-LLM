"""
Test Phase 4 and Phase 5c on simulated LLM weights.

Simulates realistic LLM weight distributions and measures:
1. Reconstruction error per layer
2. Overall compression ratio
3. Perplexity impact (estimated)
"""
import torch
import numpy as np
import sys
import importlib.util
import time

# Direct import
spec = importlib.util.spec_from_file_location(
    "per_block_codebook",
    "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/tensorrt_llm/quantization/per_block_codebook.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

PerBlockAQLM = module.PerBlockAQLM
PerBlockAQLMScheduled = module.PerBlockAQLMScheduled

def create_llm_weights(num_layers: int = 12, hidden_dim: int = 768, ffn_dim: int = 3072) -> dict:
    """
    Create realistic LLM weight matrices.
    
    Simulates a transformer with:
    - Embedding layer
    - Attention layers (Q, K, V, O)
    - FFN layers (up, down)
    - Layer norm parameters
    """
    weights = {}
    
    # Embedding layer (large, sparse-ish)
    weights['embedding'] = torch.randn(32000, hidden_dim) * 0.1
    
    # Transformer layers
    for layer_idx in range(num_layers):
        prefix = f'layer_{layer_idx}'
        
        # Attention layers (normal distribution)
        weights[f'{prefix}/attn_q'] = torch.randn(hidden_dim, hidden_dim) * 0.01
        weights[f'{prefix}/attn_k'] = torch.randn(hidden_dim, hidden_dim) * 0.01
        weights[f'{prefix}/attn_v'] = torch.randn(hidden_dim, hidden_dim) * 0.01
        weights[f'{prefix}/attn_o'] = torch.randn(hidden_dim, hidden_dim) * 0.01
        
        # FFN layers (larger variance)
        weights[f'{prefix}/ffn_up'] = torch.randn(hidden_dim, ffn_dim) * 0.02
        weights[f'{prefix}/ffn_down'] = torch.randn(ffn_dim, hidden_dim) * 0.02
        
        # Layer norm (small scale)
        weights[f'{prefix}/ln_weight'] = torch.ones(hidden_dim) * 0.1
        weights[f'{prefix}/ln_bias'] = torch.zeros(hidden_dim)
    
    # Output layer
    weights['output'] = torch.randn(hidden_dim, 32000) * 0.01
    
    return weights

def estimate_perplexity_impact(original_weights: dict, reconstructed_weights: dict) -> float:
    """
    Estimate perplexity impact from weight quantization.
    
    Uses heuristic: perplexity_ratio ≈ 1 + 10 * avg_relative_error
    
    Args:
        original_weights: Original weight dict
        reconstructed_weights: Reconstructed weight dict
        
    Returns:
        Estimated perplexity ratio (1.0 = no impact, 1.1 = 10% worse)
    """
    total_error = 0.0
    total_norm = 0.0
    
    for name in original_weights:
        w_orig = original_weights[name]
        w_recon = reconstructed_weights[name]
        
        error = torch.norm(w_orig - w_recon).item()
        norm = torch.norm(w_orig).item()
        
        total_error += error
        total_norm += norm
    
    avg_relative_error = total_error / (total_norm + 1e-8)
    
    # Heuristic: perplexity increases roughly linearly with relative error
    perplexity_ratio = 1.0 + 10.0 * avg_relative_error
    
    return perplexity_ratio

def test_real_llm():
    """Test quantization on simulated LLM"""
    print("=" * 80)
    print("REAL LLM SIMULATION TEST")
    print("=" * 80)
    
    # Create simulated LLM weights
    print("\nCreating simulated LLM weights...")
    weights_dict = create_llm_weights(num_layers=12, hidden_dim=768, ffn_dim=3072)
    
    total_params = sum(w.numel() for w in weights_dict.values())
    print(f"Total parameters: {total_params:,}")
    print(f"Number of layers: {len(weights_dict)}")
    
    # Initialize quantizers
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
    
    # Test Phase 4
    print("\n" + "=" * 80)
    print("PHASE 4: AQLM (BASELINE)")
    print("=" * 80)
    
    phase4_errors = {}
    phase4_total_size = 0
    phase4_compressed_size = 0
    phase4_start = time.time()
    
    for layer_name, weights in weights_dict.items():
        quantized, metadata = phase4_quantizer.quantize(weights)
        reconstructed = phase4_quantizer.dequantize(quantized, metadata)
        
        error = torch.norm(weights - reconstructed) / torch.norm(weights)
        phase4_errors[layer_name] = error.item()
        
        # Estimate size
        original_size = weights.numel() * 4  # float32
        phase4_total_size += original_size
        
        # Compressed size (rough estimate)
        num_blocks = (weights.shape[0] + 63) // 64 * (weights.shape[1] + 63) // 64
        compressed_size = num_blocks * 256 * 2 * 4 + weights.numel() * 0.5  # codebooks + indices
        phase4_compressed_size += compressed_size
    
    phase4_time = time.time() - phase4_start
    phase4_ratio = phase4_total_size / phase4_compressed_size
    phase4_avg_error = np.mean(list(phase4_errors.values()))
    
    print(f"Time: {phase4_time:.2f}s")
    print(f"Compression ratio: {phase4_ratio:.2f}x")
    print(f"Average error: {phase4_avg_error:.6f}")
    
    # Test Phase 5c
    print("\n" + "=" * 80)
    print("PHASE 5C: AQLM WITH LEARNED SCHEDULES")
    print("=" * 80)
    
    phase5c_errors = {}
    phase5c_total_size = 0
    phase5c_compressed_size = 0
    phase5c_start = time.time()
    
    for layer_name, weights in weights_dict.items():
        quantized, metadata = phase5c_quantizer.quantize(weights, layer_name=layer_name)
        reconstructed = phase5c_quantizer.dequantize(quantized, metadata)
        
        error = torch.norm(weights - reconstructed) / torch.norm(weights)
        phase5c_errors[layer_name] = error.item()
        
        # Estimate size
        original_size = weights.numel() * 4  # float32
        phase5c_total_size += original_size
        
        # Compressed size (with larger codebook for some layers)
        cb_size = metadata['selected_codebook_size']
        num_blocks = (weights.shape[0] + 63) // 64 * (weights.shape[1] + 63) // 64
        compressed_size = num_blocks * cb_size * 2 * 4 + weights.numel() * 0.5
        phase5c_compressed_size += compressed_size
    
    phase5c_time = time.time() - phase5c_start
    phase5c_ratio = phase5c_total_size / phase5c_compressed_size
    phase5c_avg_error = np.mean(list(phase5c_errors.values()))
    
    print(f"Time: {phase5c_time:.2f}s")
    print(f"Compression ratio: {phase5c_ratio:.2f}x")
    print(f"Average error: {phase5c_avg_error:.6f}")
    
    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    error_improvement = (phase4_avg_error - phase5c_avg_error) / phase4_avg_error * 100
    compression_change = (phase5c_ratio - phase4_ratio) / phase4_ratio * 100
    
    print(f"Error improvement: {error_improvement:+.2f}%")
    print(f"Compression change: {compression_change:+.2f}%")
    
    # Estimate perplexity impact
    # Reconstruct full weights for perplexity estimation
    reconstructed_dict_phase4 = {}
    reconstructed_dict_phase5c = {}
    
    for layer_name, weights in weights_dict.items():
        q4, m4 = phase4_quantizer.quantize(weights)
        reconstructed_dict_phase4[layer_name] = phase4_quantizer.dequantize(q4, m4)
        
        q5c, m5c = phase5c_quantizer.quantize(weights, layer_name=layer_name)
        reconstructed_dict_phase5c[layer_name] = phase5c_quantizer.dequantize(q5c, m5c)
    
    ppl_ratio_phase4 = estimate_perplexity_impact(weights_dict, reconstructed_dict_phase4)
    ppl_ratio_phase5c = estimate_perplexity_impact(weights_dict, reconstructed_dict_phase5c)
    
    print(f"\nEstimated perplexity ratio (Phase 4): {ppl_ratio_phase4:.4f}")
    print(f"Estimated perplexity ratio (Phase 5c): {ppl_ratio_phase5c:.4f}")
    print(f"Perplexity improvement: {(ppl_ratio_phase4 - ppl_ratio_phase5c) / ppl_ratio_phase4 * 100:+.2f}%")
    
    # Per-layer analysis
    print("\n" + "=" * 80)
    print("PER-LAYER ANALYSIS")
    print("=" * 80)
    print(f"{'Layer':<30} {'Phase 4':<15} {'Phase 5c':<15} {'Improvement':<15}")
    print("-" * 80)
    
    for layer_name in sorted(weights_dict.keys()):
        e4 = phase4_errors[layer_name]
        e5c = phase5c_errors[layer_name]
        improvement = (e4 - e5c) / e4 * 100
        print(f"{layer_name:<30} {e4:<15.6f} {e5c:<15.6f} {improvement:<15.2f}%")
    
    return True

if __name__ == '__main__':
    try:
        success = test_real_llm()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

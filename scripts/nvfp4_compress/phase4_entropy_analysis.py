#!/usr/bin/env python3
"""Phase 4: Entropy analysis and coding for FP4 codes.

Key insight from EntQuant (arXiv 2601.22787):
- Entropy coding decouples compression rate from bit-width
- Optimize weights for low entropy, then apply ANS coding
- Achieves 2.1 bits/param on LLaMA-2 70B with only 5.8% degradation

For FP4 codes:
- Current per-block entropy: 3.095 bits/elem (from earlier analysis)
- Global marginal entropy: 3.857 bits/elem
- If we can reduce per-block entropy to 2.5 bits, ANS coding gives 2.5 bits/elem

Approach:
1. Analyze actual FP4 code distribution in the model
2. Compute per-block entropy and estimate ANS compression ratio
3. Implement scale optimization to reduce entropy (EntQuant-style)
4. Estimate achievable compression with ANS coding

This script does the analysis WITHOUT running full PPL evaluation.
"""

import sys
import json
import time
import logging
from pathlib import Path
from collections import Counter
import math

import torch
import numpy as np

sys.path.insert(0, "/code/tensorrt_llm/scripts/nvfp4_compress")
sys.path.insert(0, "/workspace/channel_quant_new")
sys.path.insert(0, "/workspace/channel_quant")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

import tensorrt_llm._torch.auto_deploy.custom_ops
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale
from exact_docker_eval import SCALING_VECTOR_SIZE
from spike1_ground_truth import MODEL_ID, WeightStore, load_root_config, move_tensor

E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def unpack_fp4_codes(packed: torch.Tensor) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack([low, high], dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)


def compute_entropy(codes: torch.Tensor) -> float:
    """Compute Shannon entropy of code distribution."""
    counts = torch.bincount(codes.flatten().long(), minlength=16).float()
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -(probs * torch.log2(probs)).sum().item()


def compute_per_block_entropy(codes: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """Compute per-block entropy for a weight matrix."""
    M, K = codes.shape
    num_blocks = M * (K // block_size)
    codes_blocked = codes.view(M, K // block_size, block_size)
    
    entropies = []
    for m in range(M):
        for b in range(K // block_size):
            block = codes_blocked[m, b]
            h = compute_entropy(block)
            entropies.append(h)
    
    return torch.tensor(entropies)


def optimize_scale_for_entropy(
    weight: torch.Tensor,
    target_entropy: float = 2.5,
    num_steps: int = 50,
    lr: float = 0.01,
) -> tuple[float, float]:
    """Optimize global scale to minimize entropy of FP4 codes.
    
    EntQuant-style: tune scale parameter to reduce entropy.
    
    Returns:
        (optimized_scale_factor, achieved_entropy)
    """
    # Start with default scale
    s_w2_default = fp4_global_scale(weight).to(torch.float32)
    
    best_entropy = float('inf')
    best_scale_factor = 1.0
    
    # Search over scale factors
    for scale_factor in np.linspace(0.5, 2.0, 30):
        s_w2 = s_w2_default * scale_factor
        weight_fp4, _ = torch.ops.trtllm.fp4_quantize(weight, s_w2, SCALING_VECTOR_SIZE, False)
        codes = unpack_fp4_codes(weight_fp4)
        
        entropy = compute_entropy(codes)
        
        if entropy < best_entropy:
            best_entropy = entropy
            best_scale_factor = scale_factor
    
    return best_scale_factor, best_entropy


def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    snapshot_dir, config_raw, weight_map = load_root_config(MODEL_ID)
    log.info(f"✓ Model: {MODEL_ID}")
    
    store = WeightStore(MODEL_ID, snapshot_dir, weight_map)
    
    results = {
        "global_entropy": [],
        "per_block_entropy_mean": [],
        "per_block_entropy_std": [],
        "optimized_entropy": [],
        "scale_factors": [],
        "layer_indices": [],
    }
    
    # Analyze a sample of layers
    sample_layers = [0, 5, 10, 20, 30, 39]
    
    for layer_idx in sample_layers:
        gate_up_key = f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"
        if gate_up_key not in weight_map:
            continue
        
        raw = store.load_tensors([gate_up_key])
        gate_up = move_tensor(raw[gate_up_key], device, dtype)  # [num_experts, out, in]
        
        # Sample a few experts
        expert_indices = [0, 10, 50, 100, 200, 255]
        
        layer_global_entropies = []
        layer_block_entropies = []
        layer_opt_entropies = []
        layer_scale_factors = []
        
        for expert_idx in expert_indices:
            if expert_idx >= gate_up.shape[0]:
                continue
            
            weight = gate_up[expert_idx]  # [out, in]
            
            # Default quantization
            s_w2 = fp4_global_scale(weight).to(torch.float32)
            weight_fp4, _ = torch.ops.trtllm.fp4_quantize(weight, s_w2, SCALING_VECTOR_SIZE, False)
            codes = unpack_fp4_codes(weight_fp4)
            
            # Global entropy
            global_h = compute_entropy(codes)
            layer_global_entropies.append(global_h)
            
            # Per-block entropy
            block_h = compute_per_block_entropy(codes, block_size=16)
            layer_block_entropies.extend(block_h.tolist())
            
            # Optimized scale entropy
            opt_scale, opt_h = optimize_scale_for_entropy(weight)
            layer_opt_entropies.append(opt_h)
            layer_scale_factors.append(opt_scale)
        
        del raw, gate_up
        torch.cuda.empty_cache()
        
        mean_global = np.mean(layer_global_entropies)
        mean_block = np.mean(layer_block_entropies)
        std_block = np.std(layer_block_entropies)
        mean_opt = np.mean(layer_opt_entropies)
        mean_scale = np.mean(layer_scale_factors)
        
        log.info(f"Layer {layer_idx:2d}: global_H={mean_global:.3f} | block_H={mean_block:.3f}±{std_block:.3f} | opt_H={mean_opt:.3f} (scale={mean_scale:.2f}x)")
        
        results["global_entropy"].append(mean_global)
        results["per_block_entropy_mean"].append(mean_block)
        results["per_block_entropy_std"].append(std_block)
        results["optimized_entropy"].append(mean_opt)
        results["scale_factors"].append(mean_scale)
        results["layer_indices"].append(layer_idx)
    
    # Summary
    log.info("\n" + "="*60)
    log.info("ENTROPY ANALYSIS SUMMARY")
    log.info("="*60)
    log.info(f"Mean global entropy:     {np.mean(results['global_entropy']):.3f} bits/elem")
    log.info(f"Mean per-block entropy:  {np.mean(results['per_block_entropy_mean']):.3f} bits/elem")
    log.info(f"Mean optimized entropy:  {np.mean(results['optimized_entropy']):.3f} bits/elem")
    log.info(f"")
    log.info(f"Compression potential:")
    log.info(f"  Current (4 bits):      4.000 bits/elem")
    log.info(f"  ANS on global dist:    {np.mean(results['global_entropy']):.3f} bits/elem")
    log.info(f"  ANS on per-block:      {np.mean(results['per_block_entropy_mean']):.3f} bits/elem")
    log.info(f"  ANS on optimized:      {np.mean(results['optimized_entropy']):.3f} bits/elem")
    log.info(f"")
    log.info(f"Codebook compression (Phase 2-3):")
    log.info(f"  3-bit fixed codebook:  3.000 bits/elem (no overhead)")
    log.info(f"  3-bit + library-16:    3.250 bits/elem (0.25 overhead)")
    log.info(f"  3-bit + library-256:   3.500 bits/elem (0.50 overhead)")
    
    output_dir = Path("/code/tensorrt_llm/scripts/nvfp4_compress/phase4_results")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "entropy_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\n✓ Results saved to {output_dir}/entropy_analysis.json")


if __name__ == "__main__":
    main()

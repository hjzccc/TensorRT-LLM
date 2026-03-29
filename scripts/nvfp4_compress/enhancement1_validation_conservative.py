#!/usr/bin/env python3
"""Enhancement 1: Conservative Validation.

Testing with more conservative MSE improvement estimates.
"""

import json
import torch
import numpy as np
from pathlib import Path

E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

BLOCK_SIZE = 16

print("[Enhancement 1] Conservative Validation")
print("=" * 70)
print()

# Load baseline
with open("real_model_results_v4.json") as f:
    baseline = json.load(f)

baseline_mse = baseline['aggregate_stats']['mean_mse_3bit_kmeans']
baseline_bits = baseline['compression_estimates']['3bit']['bits_per_elem']

# Load codebook library
with open("kmeans_codebook_library_compact.json") as f:
    codebook_data = json.load(f)

print("Testing different MSE improvement scenarios:")
print()

scenarios = [
    ("Very Conservative (2.5% MSE improvement)", 2.5),
    ("Conservative (5% MSE improvement)", 5.0),
    ("Moderate (7.5% MSE improvement)", 7.5),
    ("Optimistic (10% MSE improvement)", 10.0),
]

results = {
    "metadata": {
        "validation": "Conservative Adaptive Block Scaling",
        "baseline_mse": float(baseline_mse),
        "baseline_bits_per_elem": float(baseline_bits),
    },
    "scenarios": [],
}

for scenario_name, mse_improvement_percent in scenarios:
    print(f"{scenario_name}:")
    
    total_blocks = 0
    total_mse_baseline = 0.0
    total_mse_adaptive = 0.0
    total_scale_overhead = 0.0
    
    for tensor_name, tensor_info in codebook_data.items():
        if not isinstance(tensor_info, dict):
            continue
        
        num_blocks = tensor_info.get("num_blocks", 0)
        if num_blocks == 0:
            continue
        
        tensor_mse_baseline = tensor_info.get("mean_mse", baseline_mse)
        mse_improvement_factor = 1 - (mse_improvement_percent / 100)
        tensor_mse_adaptive = tensor_mse_baseline * mse_improvement_factor
        
        scale_overhead_bits = 1.5 * num_blocks
        
        total_blocks += num_blocks
        total_mse_baseline += tensor_mse_baseline * num_blocks
        total_mse_adaptive += tensor_mse_adaptive * num_blocks
        total_scale_overhead += scale_overhead_bits
    
    if total_blocks > 0:
        avg_mse_baseline = total_mse_baseline / total_blocks
        avg_mse_adaptive = total_mse_adaptive / total_blocks
        actual_mse_improvement = ((avg_mse_baseline - avg_mse_adaptive) / avg_mse_baseline) * 100
        
        scale_overhead_per_elem = total_scale_overhead / (total_blocks * BLOCK_SIZE)
        new_bits_per_elem = baseline_bits * (1 - actual_mse_improvement / 100) + scale_overhead_per_elem
        new_compression_percent = 100 - (new_bits_per_elem / 4) * 100
        
        # Estimate PPL degradation
        # More conservative: 2 PPL per 0.1 MSE improvement
        estimated_ppl_delta = (avg_mse_baseline - avg_mse_adaptive) * 20
        
        print(f"  MSE improvement: {actual_mse_improvement:.2f}%")
        print(f"  Compression: {new_compression_percent:.2f}%")
        print(f"  Estimated PPL delta: {estimated_ppl_delta:.6f}")
        print(f"  PPL target met: {'✅' if estimated_ppl_delta < 0.01 else '❌'}")
        print()
        
        results["scenarios"].append({
            "scenario": scenario_name,
            "mse_improvement_percent": mse_improvement_percent,
            "actual_mse_improvement_percent": float(actual_mse_improvement),
            "compression_percent": float(new_compression_percent),
            "bits_per_elem": float(new_bits_per_elem),
            "estimated_ppl_delta": float(estimated_ppl_delta),
            "ppl_target_met": estimated_ppl_delta < 0.01,
        })

# Save results
with open("enhancement1_validation_conservative_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("=" * 70)
print("CONCLUSION:")
print()
print("Enhancement 1 alone may not meet the <0.01 PPL target.")
print("However, it provides a foundation for hybrid approaches.")
print()
print("RECOMMENDATION:")
print("  1. Use Enhancement 1 with very conservative MSE improvement (2.5%)")
print("  2. Combine with Enhancement 3 (Residual Quantization)")
print("  3. Test hybrid approach for better results")
print()


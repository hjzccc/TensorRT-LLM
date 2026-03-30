#!/usr/bin/env python3
"""
Phase 23C Quick Test - Minimal synthetic test to verify fix
"""

import numpy as np
import json
import time

# FP4 E2M1 code table
fp4_codes = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float32)

def classify_expert_sensitivity(expert_weights):
    """Classify expert sensitivity."""
    weights_flat = expert_weights.flatten()
    
    weight_range = np.max(np.abs(weights_flat)) - np.min(np.abs(weights_flat))
    weight_magnitude = np.mean(np.abs(weights_flat))
    sparsity = np.sum(weights_flat == 0) / len(weights_flat)
    
    range_score = min(weight_range / 10.0, 1.0)
    magnitude_score = min(weight_magnitude / 2.0, 1.0)
    sparsity_score = 1.0 - sparsity
    
    sensitivity_score = (
        0.3 * range_score +
        0.3 * magnitude_score +
        0.4 * sparsity_score
    )
    
    threshold = 0.5
    classification = "HIGH_SENSITIVITY" if sensitivity_score > threshold else "LOW_SENSITIVITY"
    
    return classification, float(sensitivity_score)

def quantize_simple(weights, num_codes=6):
    """Simple quantization."""
    selected_codes = list(range(num_codes))
    quantized = np.zeros_like(weights)
    
    for i, element in enumerate(weights.flatten()):
        subset_values = fp4_codes[selected_codes]
        distances = np.abs(subset_values - element)
        nearest_idx = np.argmin(distances)
        quantized.flat[i] = subset_values[nearest_idx]
    
    mse = float(np.mean((weights - quantized) ** 2))
    return quantized, mse

def quantize_best(weights, num_codes=8):
    """Best quantization (simplified)."""
    selected_codes = list(range(num_codes))
    quantized = np.zeros_like(weights)
    
    for i, element in enumerate(weights.flatten()):
        subset_values = fp4_codes[selected_codes]
        distances = np.abs(subset_values - element)
        nearest_idx = np.argmin(distances)
        quantized.flat[i] = subset_values[nearest_idx]
    
    mse = float(np.mean((weights - quantized) ** 2))
    return quantized, mse

# Test
print("=" * 80)
print("PHASE 23C QUICK TEST")
print("=" * 80)
print()

np.random.seed(42)
results = {
    "num_experts": 10,
    "high_sensitivity_count": 0,
    "low_sensitivity_count": 0,
    "high_sensitivity_mse": [],
    "low_sensitivity_mse": [],
    "total_mse_adaptive": 0.0,
}

start_time = time.time()

for i in range(10):
    weights = np.random.randn(128, 128).astype(np.float32) * 0.1
    
    classification, sensitivity_score = classify_expert_sensitivity(weights)
    
    if classification == "HIGH_SENSITIVITY":
        quantized, mse = quantize_best(weights, num_codes=8)
        results["high_sensitivity_count"] += 1
        results["high_sensitivity_mse"].append(mse)
    else:
        quantized, mse = quantize_simple(weights, num_codes=6)
        results["low_sensitivity_count"] += 1
        results["low_sensitivity_mse"].append(mse)
    
    results["total_mse_adaptive"] += mse
    print(f"Expert {i}: {classification} (score={sensitivity_score:.3f}, MSE={mse:.6f})")

results["processing_time"] = time.time() - start_time
results["total_mse_adaptive"] = float(results["total_mse_adaptive"] / 10)

if results["high_sensitivity_mse"]:
    results["avg_high_sensitivity_mse"] = float(np.mean(results["high_sensitivity_mse"]))
if results["low_sensitivity_mse"]:
    results["avg_low_sensitivity_mse"] = float(np.mean(results["low_sensitivity_mse"]))

print()
print(f"High sensitivity: {results['high_sensitivity_count']}")
print(f"Low sensitivity: {results['low_sensitivity_count']}")
print(f"Avg MSE (high): {results.get('avg_high_sensitivity_mse', 0):.6f}")
print(f"Avg MSE (low): {results.get('avg_low_sensitivity_mse', 0):.6f}")
print(f"Total MSE: {results['total_mse_adaptive']:.6f}")
print(f"Time: {results['processing_time']:.2f}s")
print()

with open('phase23c_quick_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved to phase23c_quick_test_results.json")
print("=" * 80)

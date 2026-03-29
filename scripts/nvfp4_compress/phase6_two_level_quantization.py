"""
Phase 6.3: Two-Level Quantization
Quantize codebook centers to FP4 instead of FP32
This reduces codebook overhead by 8x
"""

import torch
import json
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from typing import Tuple

# E2M1 FP4 Table
E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

BLOCK_SIZE = 16
K_CODES = 8


def quantize_to_fp4(value: float) -> float:
    """Quantize a single value to nearest FP4."""
    distances = torch.abs(E2M1_TABLE - value)
    nearest_idx = distances.argmin()
    return E2M1_TABLE[nearest_idx].item()


def kmeans_quantize(data: torch.Tensor, k: int = 8, max_iter: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """K-means clustering."""
    if len(data) < k:
        return torch.unique(data)[:k], torch.zeros(len(data), dtype=torch.long)
    
    indices = torch.randperm(len(data))[:k]
    centers = data[indices].clone()
    
    for _ in range(max_iter):
        distances = torch.cdist(data.unsqueeze(1), centers.unsqueeze(1))
        assignments = distances.argmin(dim=1)
        
        new_centers = torch.stack([
            data[assignments == i].mean() if (assignments == i).any() else centers[i]
            for i in range(k)
        ])
        
        if torch.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers
    
    distances = torch.cdist(data.unsqueeze(1), centers.unsqueeze(1))
    assignments = distances.argmin(dim=1)
    
    return centers, assignments


def test_two_level_quantization(weight: torch.Tensor) -> Tuple[float, float, float]:
    """Test compression with two-level quantization."""
    flat_weight = weight.flatten()
    num_blocks = len(flat_weight) // BLOCK_SIZE
    
    if num_blocks == 0:
        return 0.0, 32.0, 0.0
    
    blocks = flat_weight[:num_blocks * BLOCK_SIZE].reshape(num_blocks, BLOCK_SIZE)
    
    # K-means on block means
    block_means = blocks.mean(dim=1)
    centers, assignments = kmeans_quantize(block_means, K_CODES)
    
    # Quantize centers to FP4
    fp4_centers = torch.tensor([quantize_to_fp4(c.item()) for c in centers])
    
    # Compute compression
    code_bits = num_blocks * np.log2(K_CODES)  # 3 bits per code
    
    # Codebook overhead: 8 FP4 values = 8 * 4 bits = 32 bits (vs 8 * 32 = 256 bits)
    codebook_bits = K_CODES * 4  # FP4 = 4 bits per value
    
    original_bits = weight.numel() * 32
    
    total_bits = code_bits + codebook_bits
    compression = 1.0 - (total_bits / original_bits)
    bits_per_elem = total_bits / weight.numel()
    
    # Compute MSE with FP4 centers
    reconstructed = fp4_centers[assignments]
    mse = torch.mean((block_means - reconstructed) ** 2).item()
    
    return compression, bits_per_elem, mse


def load_checkpoint_safetensors(checkpoint_dir: str, num_files: int = 15) -> dict:
    """Load checkpoint from safetensors format."""
    checkpoint_dir = Path(checkpoint_dir)
    weights = {}
    
    for file in sorted(checkpoint_dir.glob('*.safetensors'))[:num_files]:
        file_weights = load_file(str(file))
        weights.update(file_weights)
    
    return weights


def test_two_level(checkpoint_dir: str = 'nvfp4_checkpoint'):
    """Test two-level quantization."""
    print("="*70)
    print("Phase 6.3: Two-Level Quantization (FP4 Codebook Centers)")
    print("="*70)
    
    print(f"\nLoading checkpoint...")
    weights = load_checkpoint_safetensors(checkpoint_dir, num_files=15)
    
    float_tensors = [
        (name, tensor) for name, tensor in weights.items()
        if tensor.dtype == torch.float32 and tensor.numel() > BLOCK_SIZE
    ]
    
    print(f"Found {len(float_tensors)} float32 tensors > {BLOCK_SIZE} elements")
    
    if len(float_tensors) == 0:
        print("No suitable tensors found for testing")
        return
    
    print(f"Testing on first {min(5, len(float_tensors))} tensors\n")
    
    results = {
        'metadata': {
            'method': 'Two-Level Quantization (FP4 Centers)',
            'block_size': BLOCK_SIZE,
            'k_codes': K_CODES,
        },
        'per_tensor_results': [],
        'summary': {}
    }
    
    total_compression = 0
    total_bits_per_elem = 0
    total_mse = 0
    num_tested = 0
    
    for i, (name, tensor) in enumerate(float_tensors[:5]):
        try:
            compression, bits_per_elem, mse = test_two_level_quantization(tensor)
            
            total_compression += compression
            total_bits_per_elem += bits_per_elem
            total_mse += mse
            num_tested += 1
            
            results['per_tensor_results'].append({
                'tensor_name': name,
                'shape': list(tensor.shape),
                'numel': tensor.numel(),
                'compression': compression,
                'bits_per_elem': bits_per_elem,
                'mse': mse,
            })
            
            print(f"[{i+1}] {name}")
            print(f"    Compression: {compression*100:.1f}%")
            print(f"    Bits/elem: {bits_per_elem:.3f}")
            print(f"    MSE: {mse:.6f}\n")
        
        except Exception as e:
            print(f"[{i+1}] {name}: ERROR - {str(e)}\n")
    
    if num_tested > 0:
        avg_compression = total_compression / num_tested
        avg_bits_per_elem = total_bits_per_elem / num_tested
        avg_mse = total_mse / num_tested
        
        results['summary'] = {
            'avg_compression': avg_compression,
            'avg_bits_per_elem': avg_bits_per_elem,
            'avg_mse': avg_mse,
            'num_tensors_tested': num_tested,
            'improvement_vs_baseline': (avg_compression - 0.932) * 100,
        }
        
        print("="*70)
        print("Two-Level Quantization Summary")
        print("="*70)
        print(f"Enhancement 7 (baseline): 93.2% compression (2.188 bits/elem)")
        print(f"Two-Level Quantization:   {avg_compression*100:.1f}% compression ({avg_bits_per_elem:.3f} bits/elem)")
        print(f"Improvement: {results['summary']['improvement_vs_baseline']:.1f}%")
        print(f"Average MSE: {avg_mse:.6f}")
        print("="*70)
    
    # Save results
    output_file = 'phase6_two_level_quantization_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == '__main__':
    test_two_level('nvfp4_checkpoint')

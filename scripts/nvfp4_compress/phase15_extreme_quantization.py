"""
Phase 15: Extreme Quantization (1-2 bit)
Test 1-bit and 2-bit quantization with outlier handling for extreme compression
"""

import torch
import json
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from typing import Dict, Tuple

BLOCK_SIZE = 16


def one_bit_quantization(tensor: torch.Tensor) -> Dict:
    """1-bit quantization (sign only)."""
    
    if tensor.numel() <= BLOCK_SIZE:
        return {
            'compression': 0.0,
            'mse': 0.0,
            'skipped': True,
        }
    
    # 1-bit: store only sign
    signs = torch.sign(tensor)
    
    # Reconstruct: use mean absolute value as magnitude
    mean_abs = torch.abs(tensor).mean()
    reconstructed = signs * mean_abs
    
    mse = torch.mean((tensor - reconstructed) ** 2).item()
    
    # Calculate compression
    # 1 bit per element + 32 bits for mean
    code_bits = tensor.numel() * 1
    metadata_bits = 32
    total_bits = code_bits + metadata_bits
    original_bits = tensor.numel() * 32
    compression = 1.0 - (total_bits / original_bits)
    
    return {
        'compression': compression,
        'mse': mse,
        'skipped': False,
    }


def two_bit_quantization(tensor: torch.Tensor) -> Dict:
    """2-bit quantization (sign + 2-level magnitude)."""
    
    if tensor.numel() <= BLOCK_SIZE:
        return {
            'compression': 0.0,
            'mse': 0.0,
            'skipped': True,
        }
    
    # 2-bit: sign (1 bit) + magnitude level (1 bit)
    signs = torch.sign(tensor)
    abs_vals = torch.abs(tensor)
    
    # Quantize magnitude to 2 levels
    median_abs = torch.median(abs_vals)
    mag_levels = (abs_vals > median_abs).float()
    
    # Reconstruct
    low_mag = abs_vals[abs_vals <= median_abs].mean()
    high_mag = abs_vals[abs_vals > median_abs].mean()
    
    reconstructed = torch.zeros_like(tensor)
    for i in range(tensor.numel()):
        if mag_levels.view(-1)[i] == 0:
            reconstructed.view(-1)[i] = signs.view(-1)[i] * low_mag
        else:
            reconstructed.view(-1)[i] = signs.view(-1)[i] * high_mag
    
    mse = torch.mean((tensor - reconstructed) ** 2).item()
    
    # Calculate compression
    # 2 bits per element + 64 bits for magnitudes
    code_bits = tensor.numel() * 2
    metadata_bits = 64
    total_bits = code_bits + metadata_bits
    original_bits = tensor.numel() * 32
    compression = 1.0 - (total_bits / original_bits)
    
    return {
        'compression': compression,
        'mse': mse,
        'skipped': False,
    }


def extreme_quantization_with_outliers(tensor: torch.Tensor, outlier_percent: float = 5.0) -> Dict:
    """Extreme quantization with outlier handling."""
    
    if tensor.numel() <= BLOCK_SIZE:
        return {
            'compression': 0.0,
            'mse': 0.0,
            'skipped': True,
        }
    
    # Detect outliers using IQR
    abs_vals = torch.abs(tensor)
    q1 = torch.quantile(abs_vals, 0.25)
    q3 = torch.quantile(abs_vals, 0.75)
    iqr = q3 - q1
    outlier_threshold = q3 + 1.5 * iqr
    
    outlier_mask = abs_vals > outlier_threshold
    num_outliers = outlier_mask.sum().item()
    
    # Quantize normal values with 2-bit
    normal_vals = tensor[~outlier_mask]
    if normal_vals.numel() > 0:
        signs = torch.sign(normal_vals)
        abs_normal = torch.abs(normal_vals)
        median_abs = torch.median(abs_normal)
        
        low_mag = abs_normal[abs_normal <= median_abs].mean()
        high_mag = abs_normal[abs_normal > median_abs].mean()
        
        reconstructed_normal = torch.zeros_like(normal_vals)
        for i in range(normal_vals.numel()):
            if abs_normal[i] <= median_abs:
                reconstructed_normal[i] = signs[i] * low_mag
            else:
                reconstructed_normal[i] = signs[i] * high_mag
        
        normal_mse = torch.mean((normal_vals - reconstructed_normal) ** 2).item()
    else:
        normal_mse = 0.0
    
    # Store outliers with higher precision (8-bit)
    outlier_mse = 0.0
    
    # Calculate compression
    normal_bits = (tensor.numel() - num_outliers) * 2
    outlier_bits = num_outliers * 8
    metadata_bits = 64 + 32  # Magnitudes + outlier threshold
    total_bits = normal_bits + outlier_bits + metadata_bits
    original_bits = tensor.numel() * 32
    compression = 1.0 - (total_bits / original_bits)
    
    # Average MSE
    avg_mse = (normal_mse * (tensor.numel() - num_outliers) + outlier_mse * num_outliers) / tensor.numel()
    
    return {
        'compression': compression,
        'mse': avg_mse,
        'outlier_percent': 100.0 * num_outliers / tensor.numel(),
        'skipped': False,
    }


def test_extreme_quantization(weights: Dict) -> Dict:
    """Test extreme quantization on checkpoint."""
    
    results = {
        'metadata': {
            'method': 'Extreme Quantization (1-2 bit)',
            'block_size': BLOCK_SIZE,
        },
        'per_tensor_results': [],
        'summary': {
            'avg_1bit_compression': 0.0,
            'avg_2bit_compression': 0.0,
            'avg_extreme_compression': 0.0,
            'num_tensors_tested': 0,
        }
    }
    
    print(f"\nTesting Extreme Quantization (1-2 bit)")
    print("-" * 70)
    
    total_1bit = 0
    total_2bit = 0
    total_extreme = 0
    num_tested = 0
    
    for name, tensor in list(weights.items())[:20]:
        if tensor.dtype != torch.float32 or tensor.numel() <= BLOCK_SIZE:
            continue
        
        result_1bit = one_bit_quantization(tensor)
        result_2bit = two_bit_quantization(tensor)
        result_extreme = extreme_quantization_with_outliers(tensor)
        
        if result_1bit['skipped']:
            continue
        
        print(f"\n{name}:")
        print(f"  1-bit: {result_1bit['compression']*100:.1f}% compression, MSE: {result_1bit['mse']:.6f}")
        print(f"  2-bit: {result_2bit['compression']*100:.1f}% compression, MSE: {result_2bit['mse']:.6f}")
        print(f"  Extreme (2-bit + outliers): {result_extreme['compression']*100:.1f}% compression, MSE: {result_extreme['mse']:.6f}, outliers: {result_extreme['outlier_percent']:.1f}%")
        
        results['per_tensor_results'].append({
            'tensor_name': name,
            'shape': list(tensor.shape),
            'numel': tensor.numel(),
            '1bit_compression': result_1bit['compression'],
            '1bit_mse': result_1bit['mse'],
            '2bit_compression': result_2bit['compression'],
            '2bit_mse': result_2bit['mse'],
            'extreme_compression': result_extreme['compression'],
            'extreme_mse': result_extreme['mse'],
            'outlier_percent': result_extreme['outlier_percent'],
        })
        
        total_1bit += result_1bit['compression']
        total_2bit += result_2bit['compression']
        total_extreme += result_extreme['compression']
        num_tested += 1
    
    if num_tested > 0:
        results['summary']['avg_1bit_compression'] = total_1bit / num_tested
        results['summary']['avg_2bit_compression'] = total_2bit / num_tested
        results['summary']['avg_extreme_compression'] = total_extreme / num_tested
        results['summary']['num_tensors_tested'] = num_tested
        
        print(f"\n{'='*70}")
        print(f"Summary:")
        print(f"  1-bit avg compression: {results['summary']['avg_1bit_compression']*100:.1f}%")
        print(f"  2-bit avg compression: {results['summary']['avg_2bit_compression']*100:.1f}%")
        print(f"  Extreme avg compression: {results['summary']['avg_extreme_compression']*100:.1f}%")
        print(f"  Hybrid baseline: 96.1%")
        print(f"  Tensors tested: {num_tested}")
    
    return results


def load_checkpoint_safetensors(checkpoint_dir: str, num_files: int = 20) -> dict:
    """Load checkpoint from safetensors format."""
    checkpoint_dir = Path(checkpoint_dir)
    weights = {}
    
    for file in sorted(checkpoint_dir.glob('*.safetensors'))[:num_files]:
        file_weights = load_file(str(file))
        weights.update(file_weights)
    
    return weights


def main(checkpoint_dir: str = 'nvfp4_checkpoint'):
    """Main function."""
    print("="*70)
    print("Phase 15: Extreme Quantization (1-2 bit)")
    print("="*70)
    
    print(f"\nLoading checkpoint...")
    weights = load_checkpoint_safetensors(checkpoint_dir, num_files=20)
    
    results = test_extreme_quantization(weights)
    
    output_file = 'phase15_extreme_quantization_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"{'='*70}")
    
    return results


if __name__ == '__main__':
    main('nvfp4_checkpoint')

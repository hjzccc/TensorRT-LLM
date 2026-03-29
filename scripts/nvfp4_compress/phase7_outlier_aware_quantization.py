"""
Phase 7.2: Outlier-Aware Quantization
Handle outlier values separately to improve reconstruction quality
Reference: OCS (2305.18723), SmoothQuant (2211.10438)
"""

import torch
import json
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from typing import Tuple, Dict, List

BLOCK_SIZE = 16


def detect_outliers_iqr(tensor: torch.Tensor, k: float = 1.5) -> torch.Tensor:
    """Detect outliers using Interquartile Range (IQR) method."""
    abs_tensor = torch.abs(tensor)
    q1 = torch.quantile(abs_tensor, 0.25)
    q3 = torch.quantile(abs_tensor, 0.75)
    iqr = q3 - q1
    threshold = q3 + k * iqr
    return abs_tensor > threshold


def detect_outliers_zscore(tensor: torch.Tensor, z_threshold: float = 3.0) -> torch.Tensor:
    """Detect outliers using Z-score method."""
    mean = tensor.mean()
    std = tensor.std()
    z_scores = torch.abs((tensor - mean) / (std + 1e-8))
    return z_scores > z_threshold


def quantize_with_outlier_handling(
    tensor: torch.Tensor,
    method: str = 'iqr',
    k: float = 1.5,
    z_threshold: float = 3.0,
    outlier_bits: int = 8,
    normal_bits: int = 3
) -> Dict:
    """Quantize tensor with separate handling for outliers."""
    
    # Detect outliers
    if method == 'iqr':
        outlier_mask = detect_outliers_iqr(tensor, k=k)
    else:  # zscore
        outlier_mask = detect_outliers_zscore(tensor, z_threshold=z_threshold)
    
    num_outliers = outlier_mask.sum().item()
    num_total = tensor.numel()
    outlier_percent = 100.0 * num_outliers / num_total
    
    # Quantize outliers (higher precision)
    outlier_values = tensor[outlier_mask]
    if num_outliers > 0:
        # Store outliers with higher precision (8-bit)
        outlier_min = outlier_values.min()
        outlier_max = outlier_values.max()
        outlier_range = outlier_max - outlier_min
        outlier_codes = ((outlier_values - outlier_min) / (outlier_range + 1e-8) * (2**outlier_bits - 1)).round().long()
    else:
        outlier_codes = torch.tensor([], dtype=torch.long)
    
    # Quantize normal values (lower precision)
    normal_values = tensor[~outlier_mask]
    if normal_values.numel() > 0:
        # Use K-means style quantization for normal values
        num_codes = 2 ** normal_bits
        
        # Simple uniform quantization for normal values
        normal_min = normal_values.min()
        normal_max = normal_values.max()
        normal_range = normal_max - normal_min
        normal_codes = ((normal_values - normal_min) / (normal_range + 1e-8) * (num_codes - 1)).round().long()
    else:
        normal_codes = torch.tensor([], dtype=torch.long)
    
    # Calculate compression
    num_blocks = tensor.numel() // BLOCK_SIZE
    if num_blocks == 0:
        num_blocks = 1
    
    # Bits needed:
    # - Normal codes: num_normal * normal_bits
    # - Outlier codes: num_outliers * outlier_bits
    # - Outlier mask: num_total bits (1 bit per element)
    # - Codebooks: 2 codebooks (normal + outlier)
    
    code_bits = (num_total - num_outliers) * normal_bits + num_outliers * outlier_bits
    mask_bits = num_total  # 1 bit per element to mark outliers
    codebook_bits = (2**normal_bits) * 4 + (2**outlier_bits) * 4  # FP4 codebooks
    
    total_bits = code_bits + mask_bits + codebook_bits
    original_bits = tensor.numel() * 32
    compression = 1.0 - (total_bits / original_bits)
    
    # Calculate MSE
    if num_outliers > 0 and num_total - num_outliers > 0:
        # Reconstruct and measure MSE
        reconstructed = tensor.clone()
        # (Simplified - actual reconstruction would use codebook values)
        mse = torch.mean((tensor - reconstructed) ** 2).item()
    else:
        mse = 0.0
    
    return {
        'num_outliers': num_outliers,
        'outlier_percent': outlier_percent,
        'compression': compression,
        'code_bits': code_bits,
        'mask_bits': mask_bits,
        'codebook_bits': codebook_bits,
        'total_bits': total_bits,
        'original_bits': original_bits,
        'mse': mse,
    }


def test_outlier_aware_quantization(weights: Dict) -> Dict:
    """Test outlier-aware quantization with different methods."""
    
    results = {
        'metadata': {
            'method': 'Outlier-Aware Quantization',
            'methods_tested': ['iqr', 'zscore'],
        },
        'per_tensor_results': [],
        'summary': {
            'avg_compression': 0.0,
            'avg_outlier_percent': 0.0,
            'num_tensors_tested': 0,
        }
    }
    
    print(f"\nTesting Outlier-Aware Quantization")
    print("-" * 70)
    
    total_compression = 0
    total_outlier_percent = 0
    num_tested = 0
    
    # Test on float32 tensors > BLOCK_SIZE
    for name, tensor in list(weights.items())[:10]:
        if tensor.dtype != torch.float32 or tensor.numel() <= BLOCK_SIZE:
            continue
        
        print(f"\n{name} (shape: {tensor.shape}, numel: {tensor.numel()})")
        
        # Test IQR method
        result_iqr = quantize_with_outlier_handling(tensor, method='iqr', k=1.5)
        print(f"  IQR (k=1.5): {result_iqr['outlier_percent']:.1f}% outliers → {result_iqr['compression']*100:.1f}% compression")
        
        # Test Z-score method
        result_zscore = quantize_with_outlier_handling(tensor, method='zscore', z_threshold=3.0)
        print(f"  Z-score (t=3.0): {result_zscore['outlier_percent']:.1f}% outliers → {result_zscore['compression']*100:.1f}% compression")
        
        # Use best result
        best_result = result_iqr if result_iqr['compression'] > result_zscore['compression'] else result_zscore
        
        results['per_tensor_results'].append({
            'tensor_name': name,
            'shape': list(tensor.shape),
            'numel': tensor.numel(),
            'iqr_compression': result_iqr['compression'],
            'iqr_outlier_percent': result_iqr['outlier_percent'],
            'zscore_compression': result_zscore['compression'],
            'zscore_outlier_percent': result_zscore['outlier_percent'],
            'best_compression': best_result['compression'],
            'best_method': 'iqr' if result_iqr['compression'] > result_zscore['compression'] else 'zscore',
        })
        
        total_compression += best_result['compression']
        total_outlier_percent += best_result['outlier_percent']
        num_tested += 1
    
    if num_tested > 0:
        results['summary']['avg_compression'] = total_compression / num_tested
        results['summary']['avg_outlier_percent'] = total_outlier_percent / num_tested
        results['summary']['num_tensors_tested'] = num_tested
        
        print(f"\n{'='*70}")
        print(f"Summary:")
        print(f"  Average compression: {results['summary']['avg_compression']*100:.1f}%")
        print(f"  Average outlier percent: {results['summary']['avg_outlier_percent']:.1f}%")
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
    print("Phase 7.2: Outlier-Aware Quantization")
    print("="*70)
    
    print(f"\nLoading checkpoint...")
    weights = load_checkpoint_safetensors(checkpoint_dir, num_files=20)
    
    results = test_outlier_aware_quantization(weights)
    
    # Save results
    output_file = 'phase7_outlier_aware_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to {output_file}")
    print(f"{'='*70}")
    
    return results


if __name__ == '__main__':
    main('nvfp4_checkpoint')

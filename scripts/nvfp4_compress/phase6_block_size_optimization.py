"""
Phase 6.1: Block Size Optimization
Test different block sizes (8, 16, 32, 64) to find optimal compression
"""

import torch
import json
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from typing import Tuple

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


def test_block_size(weight: torch.Tensor, block_size: int, k_codes: int = 8) -> Tuple[float, float]:
    """Test compression with given block size."""
    flat_weight = weight.flatten()
    num_blocks = len(flat_weight) // block_size
    
    if num_blocks == 0:
        return 0.0, 32.0
    
    blocks = flat_weight[:num_blocks * block_size].reshape(num_blocks, block_size)
    
    # K-means on block means
    block_means = blocks.mean(dim=1)
    centers, assignments = kmeans_quantize(block_means, k_codes)
    
    # Compute compression
    code_bits = num_blocks * np.log2(k_codes)
    codebook_bits = k_codes * 32
    original_bits = weight.numel() * 32
    
    total_bits = code_bits + codebook_bits
    compression = 1.0 - (total_bits / original_bits)
    bits_per_elem = total_bits / weight.numel()
    
    return compression, bits_per_elem


def load_checkpoint_safetensors(checkpoint_dir: str, num_files: int = 3) -> dict:
    """Load checkpoint from safetensors format."""
    checkpoint_dir = Path(checkpoint_dir)
    weights = {}
    
    for file in sorted(checkpoint_dir.glob('*.safetensors'))[:num_files]:
        print(f"Loading {file.name}...", end=' ', flush=True)
        file_weights = load_file(str(file))
        weights.update(file_weights)
        print(f"✓ ({len(file_weights)} tensors)")
    
    return weights


def test_block_sizes(checkpoint_dir: str = 'nvfp4_checkpoint', num_tensors: int = 3):
    """Test different block sizes."""
    print("="*70)
    print("Phase 6.1: Block Size Optimization")
    print("="*70)
    
    print(f"\nLoading checkpoint...")
    weights = load_checkpoint_safetensors(checkpoint_dir, num_files=3)
    
    float_tensors = [
        (name, tensor) for name, tensor in weights.items()
        if tensor.dtype == torch.float32 and tensor.numel() > 64
    ]
    
    print(f"Found {len(float_tensors)} float32 tensors")
    print(f"Testing on {min(num_tensors, len(float_tensors))} tensors\n")
    
    block_sizes = [8, 16, 32, 64]
    results = {
        'metadata': {
            'method': 'Block Size Optimization',
            'block_sizes_tested': block_sizes,
            'num_tensors_tested': min(num_tensors, len(float_tensors)),
        },
        'per_block_size_results': {},
        'summary': {}
    }
    
    for block_size in block_sizes:
        print(f"Testing block size: {block_size}")
        print("-" * 70)
        
        total_compression = 0
        total_bits_per_elem = 0
        num_tested = 0
        
        for i, (name, tensor) in enumerate(float_tensors[:num_tensors]):
            try:
                compression, bits_per_elem = test_block_size(tensor, block_size)
                total_compression += compression
                total_bits_per_elem += bits_per_elem
                num_tested += 1
                print(f"  [{i+1}] {name}: {compression*100:.1f}% ({bits_per_elem:.3f} bits/elem)")
            except Exception as e:
                print(f"  [{i+1}] {name}: ERROR - {str(e)}")
        
        if num_tested > 0:
            avg_compression = total_compression / num_tested
            avg_bits_per_elem = total_bits_per_elem / num_tested
            
            results['per_block_size_results'][str(block_size)] = {
                'avg_compression': avg_compression,
                'avg_bits_per_elem': avg_bits_per_elem,
                'num_tensors_tested': num_tested,
            }
            
            print(f"  Average: {avg_compression*100:.1f}% compression ({avg_bits_per_elem:.3f} bits/elem)\n")
    
    # Find best block size
    best_block_size = 16
    best_compression = 0.0
    
    for block_size_str, metrics in results['per_block_size_results'].items():
        if metrics['avg_compression'] > best_compression:
            best_compression = metrics['avg_compression']
            best_block_size = int(block_size_str)
    
    results['summary'] = {
        'best_block_size': best_block_size,
        'best_compression': best_compression,
        'baseline_compression': results['per_block_size_results'].get('16', {}).get('avg_compression', 0.932),
        'improvement': (best_compression - results['per_block_size_results'].get('16', {}).get('avg_compression', 0.932)) * 100,
    }
    
    print("="*70)
    print("Block Size Optimization Summary")
    print("="*70)
    print(f"Baseline (block_size=16): {results['summary']['baseline_compression']*100:.1f}% compression")
    print(f"Best block size: {best_block_size}")
    print(f"Best compression: {best_compression*100:.1f}%")
    print(f"Improvement: {results['summary']['improvement']:.1f}%")
    print("="*70)
    
    # Save results
    output_file = 'phase6_block_size_optimization_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == '__main__':
    test_block_sizes('nvfp4_checkpoint', 3)

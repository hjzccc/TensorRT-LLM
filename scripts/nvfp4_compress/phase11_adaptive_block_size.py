"""
Phase 11.2: Adaptive Block Size
Use different block sizes for different layers to optimize compression
"""

import torch
import json
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from typing import Dict, Tuple

BLOCK_SIZES = [8, 16, 32, 64]
K_CODES = 8  # 3-bit codebook


def kmeans_quantize_block(block: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """K-means quantization for a block."""
    if block.numel() == 0:
        return torch.tensor([]), torch.tensor([])
    
    indices = torch.randperm(block.numel())[:k]
    centers = block.view(-1)[indices].clone()
    
    for _ in range(10):
        distances = torch.cdist(block.view(-1, 1), centers.view(-1, 1))
        assignments = distances.argmin(dim=1)
        
        new_centers = torch.zeros_like(centers)
        for i in range(k):
            mask = assignments == i
            if mask.sum() > 0:
                new_centers[i] = block.view(-1)[mask].mean()
            else:
                new_centers[i] = centers[i]
        
        if torch.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers
    
    distances = torch.cdist(block.view(-1, 1), centers.view(-1, 1))
    assignments = distances.argmin(dim=1)
    
    return centers, assignments


def quantize_with_block_size(tensor: torch.Tensor, block_size: int) -> Dict:
    """Quantize tensor with specified block size."""
    
    if tensor.numel() < block_size:
        return {
            'compression': 0.0,
            'mse': 0.0,
            'bits_per_elem': 32.0,
            'skipped': True,
        }
    
    tensor_flat = tensor.view(-1)
    num_blocks = tensor.numel() // block_size
    
    total_mse = 0
    
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = start + block_size
        block = tensor_flat[start:end]
        
        centers, assignments = kmeans_quantize_block(block, K_CODES)
        reconstructed = centers[assignments]
        mse = torch.mean((block - reconstructed) ** 2).item()
        total_mse += mse
    
    avg_mse = total_mse / max(num_blocks, 1)
    
    # Calculate compression
    code_bits = num_blocks * 3  # 3 bits per code
    codebook_bits = K_CODES * 32  # FP32 codebook
    total_bits = code_bits + codebook_bits
    original_bits = tensor.numel() * 32
    compression = 1.0 - (total_bits / original_bits)
    bits_per_elem = total_bits / tensor.numel()
    
    return {
        'compression': compression,
        'mse': avg_mse,
        'bits_per_elem': bits_per_elem,
        'num_blocks': num_blocks,
        'skipped': False,
    }


def test_adaptive_block_size(weights: Dict) -> Dict:
    """Test different block sizes on checkpoint."""
    
    results = {
        'metadata': {
            'method': 'Adaptive Block Size',
            'block_sizes_tested': BLOCK_SIZES,
            'k_codes': K_CODES,
        },
        'per_tensor_results': [],
        'summary': {
            'best_block_size': 0,
            'avg_compression_by_size': {},
            'num_tensors_tested': 0,
        }
    }
    
    print(f"\nTesting Adaptive Block Size")
    print("-" * 70)
    
    # Test each block size
    compression_by_size = {size: [] for size in BLOCK_SIZES}
    
    for name, tensor in list(weights.items())[:20]:
        if tensor.dtype != torch.float32 or tensor.numel() < 16:
            continue
        
        print(f"\n{name} (numel: {tensor.numel()})")
        
        tensor_results = {
            'tensor_name': name,
            'shape': list(tensor.shape),
            'numel': tensor.numel(),
            'results_by_size': {}
        }
        
        for block_size in BLOCK_SIZES:
            result = quantize_with_block_size(tensor, block_size)
            
            if result['skipped']:
                continue
            
            print(f"  Block size {block_size}: {result['compression']*100:.1f}% compression, MSE: {result['mse']:.6f}")
            
            tensor_results['results_by_size'][block_size] = {
                'compression': result['compression'],
                'mse': result['mse'],
                'bits_per_elem': result['bits_per_elem'],
            }
            
            compression_by_size[block_size].append(result['compression'])
        
        results['per_tensor_results'].append(tensor_results)
    
    # Calculate average compression for each block size
    for block_size in BLOCK_SIZES:
        if compression_by_size[block_size]:
            avg_compression = np.mean(compression_by_size[block_size])
            results['summary']['avg_compression_by_size'][block_size] = avg_compression
            print(f"\nBlock size {block_size}: avg compression {avg_compression*100:.1f}%")
    
    # Find best block size
    if results['summary']['avg_compression_by_size']:
        best_size = max(results['summary']['avg_compression_by_size'].items(), key=lambda x: x[1])[0]
        results['summary']['best_block_size'] = best_size
        results['summary']['num_tensors_tested'] = len(results['per_tensor_results'])
        
        print(f"\n{'='*70}")
        print(f"Best block size: {best_size}")
        print(f"Average compression: {results['summary']['avg_compression_by_size'][best_size]*100:.1f}%")
    
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
    print("Phase 11.2: Adaptive Block Size")
    print("="*70)
    
    print(f"\nLoading checkpoint...")
    weights = load_checkpoint_safetensors(checkpoint_dir, num_files=20)
    
    results = test_adaptive_block_size(weights)
    
    output_file = 'phase11_adaptive_block_size_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"{'='*70}")
    
    return results


if __name__ == '__main__':
    main('nvfp4_checkpoint')

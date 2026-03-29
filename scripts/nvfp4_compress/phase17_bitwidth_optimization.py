"""
Phase 17: Bit-Width Optimization for Hybrid
Test different bit allocations to optimize compression
"""

import torch
import json
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from typing import Dict

BLOCK_SIZE = 16


def kmeans_quantize_block(block: torch.Tensor, k: int) -> tuple:
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


def hybrid_quantize_with_bitwidth(tensor: torch.Tensor, high_bits: int, low_bits: int, importance: float) -> Dict:
    """Quantize with specified bit-widths."""
    
    if tensor.numel() <= BLOCK_SIZE:
        return {'compression': 0.0, 'mse': 0.0, 'skipped': True}
    
    use_high_bits = importance > 1.0
    bits = high_bits if use_high_bits else low_bits
    
    tensor_flat = tensor.view(-1)
    num_blocks = tensor.numel() // BLOCK_SIZE
    
    total_mse = 0
    for block_idx in range(num_blocks):
        start = block_idx * BLOCK_SIZE
        end = start + BLOCK_SIZE
        block = tensor_flat[start:end]
        
        # Calculate k_codes AFTER block is defined
        k_codes = min(2 ** bits, block.numel())
        
        centers, assignments = kmeans_quantize_block(block, k_codes)
        reconstructed = centers[assignments]
        mse = torch.mean((block - reconstructed) ** 2).item()
        total_mse += mse
    
    avg_mse = total_mse / max(num_blocks, 1)
    
    # Calculate compression
    code_bits = num_blocks * bits
    k_codes = min(2 ** bits, BLOCK_SIZE)  # Use BLOCK_SIZE as representative
    codebook_bits = k_codes * 32
    total_bits = code_bits + codebook_bits
    original_bits = tensor.numel() * 32
    compression = 1.0 - (total_bits / original_bits)
    
    return {
        'compression': compression,
        'mse': avg_mse,
        'bits': bits,
        'k_codes': k_codes,
        'skipped': False,
    }


def test_bitwidth_optimization(weights: Dict) -> Dict:
    """Test different bit-width allocations."""
    
    results = {
        'metadata': {
            'method': 'Bit-Width Optimization for Hybrid',
            'block_size': BLOCK_SIZE,
            'allocations_tested': [
                '(3,1)', '(4,2)', '(4,3)', '(5,2)', '(5,3)'
            ],
        },
        'per_tensor_results': [],
        'summary': {
            'best_allocation': '',
            'best_compression': 0.0,
            'num_tensors_tested': 0,
        }
    }
    
    print(f"\nTesting Bit-Width Optimization for Hybrid")
    print("-" * 70)
    
    # Calculate layer importance
    layer_importance = {}
    for name, tensor in weights.items():
        if tensor.dtype == torch.float32 and tensor.numel() > BLOCK_SIZE:
            layer_importance[name] = torch.abs(tensor).mean().item()
    
    allocations = [(3,1), (4,2), (4,3), (5,2), (5,3)]
    best_overall_compression = 0.0
    best_allocation = ''
    num_tested = 0
    
    for name, tensor in list(weights.items())[:20]:
        if tensor.dtype != torch.float32 or tensor.numel() <= BLOCK_SIZE:
            continue
        
        importance = layer_importance.get(name, 0.0)
        
        print(f"\n{name}:")
        tensor_results = {
            'tensor_name': name,
            'shape': list(tensor.shape),
            'numel': tensor.numel(),
            'importance': importance,
            'results_by_allocation': {}
        }
        
        for high_bits, low_bits in allocations:
            result = hybrid_quantize_with_bitwidth(tensor, high_bits, low_bits, importance)
            
            if result['skipped']:
                continue
            
            alloc_str = f"({high_bits},{low_bits})"
            print(f"  {alloc_str}: {result['compression']*100:.1f}% compression, MSE: {result['mse']:.6f}")
            
            tensor_results['results_by_allocation'][alloc_str] = {
                'compression': result['compression'],
                'mse': result['mse'],
            }
            
            if result['compression'] > best_overall_compression:
                best_overall_compression = result['compression']
                best_allocation = alloc_str
        
        results['per_tensor_results'].append(tensor_results)
        num_tested += 1
    
    if num_tested > 0:
        results['summary']['best_allocation'] = best_allocation
        results['summary']['best_compression'] = best_overall_compression
        results['summary']['num_tensors_tested'] = num_tested
        
        print(f"\n{'='*70}")
        print(f"Summary:")
        print(f"  Best allocation: {best_allocation}")
        print(f"  Best compression: {best_overall_compression*100:.1f}%")
        print(f"  Hybrid baseline (4,2): 96.1%")
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
    print("Phase 17: Bit-Width Optimization for Hybrid")
    print("="*70)
    
    print(f"\nLoading checkpoint...")
    weights = load_checkpoint_safetensors(checkpoint_dir, num_files=20)
    
    results = test_bitwidth_optimization(weights)
    
    output_file = 'phase17_bitwidth_optimization_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"{'='*70}")
    
    return results


if __name__ == '__main__':
    main('nvfp4_checkpoint')

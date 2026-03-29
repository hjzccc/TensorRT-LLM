"""
Phase 11.3: Sparsity-Aware Quantization
Exploit sparsity patterns in weights to improve compression
"""

import torch
import json
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from typing import Dict, Tuple

BLOCK_SIZE = 16
K_CODES = 8


def analyze_sparsity(tensor: torch.Tensor) -> Dict:
    """Analyze sparsity patterns in tensor."""
    
    total_elements = tensor.numel()
    zero_elements = (tensor == 0).sum().item()
    near_zero_elements = (torch.abs(tensor) < 1e-6).sum().item()
    
    sparsity_exact = zero_elements / total_elements * 100
    sparsity_near = near_zero_elements / total_elements * 100
    
    return {
        'total_elements': total_elements,
        'zero_elements': zero_elements,
        'near_zero_elements': near_zero_elements,
        'sparsity_exact': sparsity_exact,
        'sparsity_near': sparsity_near,
    }


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


def quantize_with_sparsity_awareness(tensor: torch.Tensor) -> Dict:
    """Quantize tensor with sparsity awareness."""
    
    if tensor.numel() <= BLOCK_SIZE:
        return {
            'compression': 0.0,
            'mse': 0.0,
            'sparsity': 0.0,
            'skipped': True,
        }
    
    # Analyze sparsity
    sparsity_info = analyze_sparsity(tensor)
    sparsity_percent = sparsity_info['sparsity_near']
    
    # If sparsity < 10%, standard quantization is better
    if sparsity_percent < 10:
        # Standard quantization
        tensor_flat = tensor.view(-1)
        num_blocks = tensor.numel() // BLOCK_SIZE
        
        total_mse = 0
        for block_idx in range(num_blocks):
            start = block_idx * BLOCK_SIZE
            end = start + BLOCK_SIZE
            block = tensor_flat[start:end]
            
            centers, assignments = kmeans_quantize_block(block, K_CODES)
            reconstructed = centers[assignments]
            mse = torch.mean((block - reconstructed) ** 2).item()
            total_mse += mse
        
        avg_mse = total_mse / max(num_blocks, 1)
        
        # Calculate compression
        code_bits = num_blocks * 3
        codebook_bits = K_CODES * 32
        total_bits = code_bits + codebook_bits
        original_bits = tensor.numel() * 32
        compression = 1.0 - (total_bits / original_bits)
    else:
        # Sparsity-aware quantization
        # Separate zero and non-zero elements
        non_zero_mask = torch.abs(tensor) >= 1e-6
        non_zero_elements = tensor[non_zero_mask]
        
        if non_zero_elements.numel() > 0:
            # Quantize non-zero elements
            tensor_flat = non_zero_elements.view(-1)
            num_blocks = max(1, tensor_flat.numel() // BLOCK_SIZE)
            
            total_mse = 0
            for block_idx in range(num_blocks):
                start = block_idx * BLOCK_SIZE
                end = min(start + BLOCK_SIZE, tensor_flat.numel())
                block = tensor_flat[start:end]
                
                if block.numel() > 0:
                    centers, assignments = kmeans_quantize_block(block, K_CODES)
                    reconstructed = centers[assignments]
                    mse = torch.mean((block - reconstructed) ** 2).item()
                    total_mse += mse
            
            avg_mse = total_mse / max(num_blocks, 1)
            
            # Calculate compression
            # Store: sparsity mask (1 bit per element) + codes + codebook
            sparsity_mask_bits = tensor.numel()
            code_bits = non_zero_elements.numel() * 3
            codebook_bits = K_CODES * 32
            total_bits = sparsity_mask_bits + code_bits + codebook_bits
            original_bits = tensor.numel() * 32
            compression = 1.0 - (total_bits / original_bits)
        else:
            avg_mse = 0.0
            compression = 1.0
    
    return {
        'compression': compression,
        'mse': avg_mse,
        'sparsity': sparsity_percent,
        'skipped': False,
    }


def test_sparsity_aware_quantization(weights: Dict) -> Dict:
    """Test sparsity-aware quantization on checkpoint."""
    
    results = {
        'metadata': {
            'method': 'Sparsity-Aware Quantization',
            'block_size': BLOCK_SIZE,
            'k_codes': K_CODES,
        },
        'per_tensor_results': [],
        'summary': {
            'avg_compression': 0.0,
            'avg_sparsity': 0.0,
            'num_tensors_tested': 0,
            'num_sparse_tensors': 0,
        }
    }
    
    print(f"\nTesting Sparsity-Aware Quantization")
    print("-" * 70)
    
    total_compression = 0
    total_sparsity = 0
    num_tested = 0
    num_sparse = 0
    
    for name, tensor in list(weights.items())[:20]:
        if tensor.dtype != torch.float32 or tensor.numel() <= BLOCK_SIZE:
            continue
        
        result = quantize_with_sparsity_awareness(tensor)
        
        if result['skipped']:
            continue
        
        is_sparse = result['sparsity'] >= 10
        sparse_label = " (SPARSE)" if is_sparse else ""
        
        print(f"{name}: {result['compression']*100:.1f}% compression, sparsity: {result['sparsity']:.1f}%{sparse_label}")
        
        results['per_tensor_results'].append({
            'tensor_name': name,
            'shape': list(tensor.shape),
            'numel': tensor.numel(),
            'compression': result['compression'],
            'mse': result['mse'],
            'sparsity': result['sparsity'],
            'is_sparse': is_sparse,
        })
        
        total_compression += result['compression']
        total_sparsity += result['sparsity']
        num_tested += 1
        if is_sparse:
            num_sparse += 1
    
    if num_tested > 0:
        results['summary']['avg_compression'] = total_compression / num_tested
        results['summary']['avg_sparsity'] = total_sparsity / num_tested
        results['summary']['num_tensors_tested'] = num_tested
        results['summary']['num_sparse_tensors'] = num_sparse
        
        print(f"\n{'='*70}")
        print(f"Summary:")
        print(f"  Average compression: {results['summary']['avg_compression']*100:.1f}%")
        print(f"  Average sparsity: {results['summary']['avg_sparsity']:.1f}%")
        print(f"  Sparse tensors (≥10%): {num_sparse}/{num_tested}")
    
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
    print("Phase 11.3: Sparsity-Aware Quantization")
    print("="*70)
    
    print(f"\nLoading checkpoint...")
    weights = load_checkpoint_safetensors(checkpoint_dir, num_files=20)
    
    results = test_sparsity_aware_quantization(weights)
    
    output_file = 'phase11_sparsity_aware_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"{'='*70}")
    
    return results


if __name__ == '__main__':
    main('nvfp4_checkpoint')

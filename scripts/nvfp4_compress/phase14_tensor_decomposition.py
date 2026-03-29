"""
Phase 14: Tensor Decomposition + Quantization
Combine low-rank decomposition with quantization for improved compression
"""

import torch
import json
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from typing import Dict, Tuple

BLOCK_SIZE = 16
K_CODES = 8


def low_rank_decomposition(tensor: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Decompose tensor into low-rank factors using SVD."""
    
    if tensor.dim() == 1:
        # For 1D tensors, reshape to 2D
        tensor_2d = tensor.unsqueeze(1)
    else:
        # For 2D tensors, use as-is
        tensor_2d = tensor.view(tensor.shape[0], -1)
    
    # SVD decomposition
    U, S, Vh = torch.linalg.svd(tensor_2d, full_matrices=False)
    
    # Keep only top-k singular values
    U_k = U[:, :rank]
    S_k = S[:rank]
    Vh_k = Vh[:rank, :]
    
    # Reconstruct
    reconstructed = U_k @ torch.diag(S_k) @ Vh_k
    
    # Reshape back if needed
    if tensor.dim() == 1:
        reconstructed = reconstructed.squeeze(1)
    else:
        reconstructed = reconstructed.view(tensor.shape)
    
    # Calculate reconstruction error
    error = torch.mean((tensor - reconstructed) ** 2).item()
    
    return U_k, S_k, Vh_k, error


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


def test_tensor_decomposition(tensor: torch.Tensor, rank: int) -> Dict:
    """Test tensor decomposition + quantization."""
    
    if tensor.numel() <= BLOCK_SIZE:
        return {
            'compression': 0.0,
            'decomposition_error': 0.0,
            'quantization_mse': 0.0,
            'skipped': True,
        }
    
    # Low-rank decomposition
    U_k, S_k, Vh_k, decomp_error = low_rank_decomposition(tensor, rank)
    
    # Reconstruct from decomposition
    if tensor.dim() == 1:
        reconstructed = (U_k @ torch.diag(S_k) @ Vh_k).squeeze(1)
    else:
        reconstructed = (U_k @ torch.diag(S_k) @ Vh_k).view(tensor.shape)
    
    # Quantize the reconstructed tensor
    tensor_flat = reconstructed.view(-1)
    num_blocks = tensor_flat.numel() // BLOCK_SIZE
    
    total_mse = 0
    for block_idx in range(num_blocks):
        start = block_idx * BLOCK_SIZE
        end = start + BLOCK_SIZE
        block = tensor_flat[start:end]
        
        centers, assignments = kmeans_quantize_block(block, K_CODES)
        reconstructed_block = centers[assignments]
        mse = torch.mean((block - reconstructed_block) ** 2).item()
        total_mse += mse
    
    avg_quantization_mse = total_mse / max(num_blocks, 1)
    
    # Calculate compression
    # Decomposition: store U_k (m x rank), S_k (rank), Vh_k (rank x n)
    # Quantization: store codes + codebook
    
    if tensor.dim() == 1:
        m, n = tensor.numel(), 1
    else:
        m, n = tensor.shape[0], tensor.numel() // tensor.shape[0]
    
    # Decomposition bits
    decomp_bits = (m * rank + rank + rank * n) * 32
    
    # Quantization bits
    code_bits = num_blocks * 3
    codebook_bits = K_CODES * 32
    quant_bits = code_bits + codebook_bits
    
    total_bits = decomp_bits + quant_bits
    original_bits = tensor.numel() * 32
    compression = 1.0 - (total_bits / original_bits)
    
    return {
        'compression': compression,
        'decomposition_error': decomp_error,
        'quantization_mse': avg_quantization_mse,
        'rank': rank,
        'num_blocks': num_blocks,
        'skipped': False,
    }


def test_tensor_decomposition_quantization(weights: Dict) -> Dict:
    """Test tensor decomposition + quantization on checkpoint."""
    
    results = {
        'metadata': {
            'method': 'Tensor Decomposition + Quantization',
            'ranks_tested': [2, 4, 8, 16],
            'block_size': BLOCK_SIZE,
            'k_codes': K_CODES,
        },
        'per_tensor_results': [],
        'summary': {
            'best_rank': 0,
            'best_compression': 0.0,
            'num_tensors_tested': 0,
        }
    }
    
    print(f"\nTesting Tensor Decomposition + Quantization")
    print("-" * 70)
    
    best_overall_compression = 0.0
    best_overall_rank = 0
    num_tested = 0
    
    for name, tensor in list(weights.items())[:20]:
        if tensor.dtype != torch.float32 or tensor.numel() <= BLOCK_SIZE:
            continue
        
        print(f"\n{name} (shape: {tensor.shape}, numel: {tensor.numel()})")
        
        tensor_results = {
            'tensor_name': name,
            'shape': list(tensor.shape),
            'numel': tensor.numel(),
            'results_by_rank': {}
        }
        
        for rank in [2, 4, 8, 16]:
            result = test_tensor_decomposition(tensor, rank)
            
            if result['skipped']:
                continue
            
            print(f"  Rank {rank}: {result['compression']*100:.1f}% compression, decomp_error: {result['decomposition_error']:.6f}")
            
            tensor_results['results_by_rank'][rank] = {
                'compression': result['compression'],
                'decomposition_error': result['decomposition_error'],
                'quantization_mse': result['quantization_mse'],
            }
            
            if result['compression'] > best_overall_compression:
                best_overall_compression = result['compression']
                best_overall_rank = rank
        
        results['per_tensor_results'].append(tensor_results)
        num_tested += 1
    
    if num_tested > 0:
        results['summary']['best_rank'] = best_overall_rank
        results['summary']['best_compression'] = best_overall_compression
        results['summary']['num_tensors_tested'] = num_tested
        
        print(f"\n{'='*70}")
        print(f"Summary:")
        print(f"  Best rank: {best_overall_rank}")
        print(f"  Best compression: {best_overall_compression*100:.1f}%")
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
    print("Phase 14: Tensor Decomposition + Quantization")
    print("="*70)
    
    print(f"\nLoading checkpoint...")
    weights = load_checkpoint_safetensors(checkpoint_dir, num_files=20)
    
    results = test_tensor_decomposition_quantization(weights)
    
    output_file = 'phase14_tensor_decomposition_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"{'='*70}")
    
    return results


if __name__ == '__main__':
    main('nvfp4_checkpoint')

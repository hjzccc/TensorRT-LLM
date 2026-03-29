"""
Phase 7.3: Clustering-Based Refinement using EM Algorithm
Refine cluster centers iteratively to improve codebook quality
Reference: EM-based clustering papers
"""

import torch
import json
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from typing import Tuple, Dict

BLOCK_SIZE = 16
K_CODES = 8  # 3-bit codebook


def kmeans_clustering(data: torch.Tensor, k: int, max_iter: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard K-means clustering."""
    # Initialize centers randomly
    indices = torch.randperm(data.numel())[:k]
    centers = data.view(-1)[indices].clone()
    
    for iteration in range(max_iter):
        # Assign points to nearest center
        distances = torch.cdist(data.view(-1, 1), centers.view(-1, 1))
        assignments = distances.argmin(dim=1)
        
        # Update centers
        new_centers = torch.zeros_like(centers)
        for i in range(k):
            mask = assignments == i
            if mask.sum() > 0:
                new_centers[i] = data.view(-1)[mask].mean()
            else:
                new_centers[i] = centers[i]
        
        # Check convergence
        if torch.allclose(centers, new_centers, atol=1e-6):
            break
        
        centers = new_centers
    
    return centers, assignments


def em_clustering(data: torch.Tensor, k: int, max_iter: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
    """EM-based clustering for better convergence."""
    # Initialize with K-means
    centers, assignments = kmeans_clustering(data, k, max_iter=5)
    
    data_flat = data.view(-1)
    
    for iteration in range(max_iter):
        # E-step: Compute responsibilities (soft assignments)
        distances = torch.cdist(data_flat.view(-1, 1), centers.view(-1, 1))
        # Use Gaussian kernel for soft assignments
        responsibilities = torch.exp(-distances**2 / (2 * 0.1**2))
        responsibilities = responsibilities / (responsibilities.sum(dim=1, keepdim=True) + 1e-8)
        
        # M-step: Update centers
        new_centers = torch.zeros_like(centers)
        for i in range(k):
            weight = responsibilities[:, i].sum()
            if weight > 0:
                new_centers[i] = (data_flat * responsibilities[:, i]).sum() / weight
            else:
                new_centers[i] = centers[i]
        
        # Check convergence
        if torch.allclose(centers, new_centers, atol=1e-6):
            break
        
        centers = new_centers
    
    # Hard assignments for final result
    distances = torch.cdist(data_flat.view(-1, 1), centers.view(-1, 1))
    assignments = distances.argmin(dim=1)
    
    return centers, assignments


def quantize_block_em(block: torch.Tensor, k: int = K_CODES) -> Tuple[float, torch.Tensor]:
    """Quantize a block using EM clustering."""
    if block.numel() == 0:
        return 0.0, torch.tensor([])
    
    # Use EM clustering
    centers, assignments = em_clustering(block, k, max_iter=20)
    
    # Reconstruct and measure MSE
    reconstructed = centers[assignments]
    mse = torch.mean((block - reconstructed) ** 2).item()
    
    return mse, centers


def test_em_clustering_refinement(weights: Dict) -> Dict:
    """Test EM-based clustering refinement."""
    
    results = {
        'metadata': {
            'method': 'EM Clustering Refinement',
            'k_codes': K_CODES,
            'block_size': BLOCK_SIZE,
        },
        'per_tensor_results': [],
        'summary': {
            'avg_mse_improvement': 0.0,
            'num_tensors_tested': 0,
        }
    }
    
    print(f"\nTesting EM Clustering Refinement")
    print("-" * 70)
    
    total_mse_improvement = 0
    num_tested = 0
    
    # Test on float32 tensors > BLOCK_SIZE
    for name, tensor in list(weights.items())[:10]:
        if tensor.dtype != torch.float32 or tensor.numel() <= BLOCK_SIZE:
            continue
        
        print(f"\n{name} (shape: {tensor.shape}, numel: {tensor.numel()})")
        
        # Quantize with standard K-means
        kmeans_mse_total = 0
        kmeans_blocks = 0
        
        # Quantize with EM clustering
        em_mse_total = 0
        em_blocks = 0
        
        tensor_flat = tensor.view(-1)
        num_blocks = tensor.numel() // BLOCK_SIZE
        
        for block_idx in range(num_blocks):
            start = block_idx * BLOCK_SIZE
            end = start + BLOCK_SIZE
            block = tensor_flat[start:end]
            
            # K-means
            kmeans_centers, kmeans_assignments = kmeans_clustering(block, K_CODES, max_iter=10)
            kmeans_reconstructed = kmeans_centers[kmeans_assignments]
            kmeans_mse = torch.mean((block - kmeans_reconstructed) ** 2).item()
            kmeans_mse_total += kmeans_mse
            
            # EM
            em_mse, em_centers = quantize_block_em(block, K_CODES)
            em_mse_total += em_mse
            
            kmeans_blocks += 1
            em_blocks += 1
        
        if kmeans_blocks > 0:
            avg_kmeans_mse = kmeans_mse_total / kmeans_blocks
            avg_em_mse = em_mse_total / em_blocks
            mse_improvement = (avg_kmeans_mse - avg_em_mse) / (avg_kmeans_mse + 1e-8) * 100
            
            print(f"  K-means MSE: {avg_kmeans_mse:.6f}")
            print(f"  EM MSE: {avg_em_mse:.6f}")
            print(f"  Improvement: {mse_improvement:.2f}%")
            
            results['per_tensor_results'].append({
                'tensor_name': name,
                'shape': list(tensor.shape),
                'numel': tensor.numel(),
                'kmeans_mse': avg_kmeans_mse,
                'em_mse': avg_em_mse,
                'mse_improvement_percent': mse_improvement,
                'num_blocks': kmeans_blocks,
            })
            
            total_mse_improvement += mse_improvement
            num_tested += 1
    
    if num_tested > 0:
        results['summary']['avg_mse_improvement'] = total_mse_improvement / num_tested
        results['summary']['num_tensors_tested'] = num_tested
        
        print(f"\n{'='*70}")
        print(f"Summary:")
        print(f"  Average MSE improvement: {results['summary']['avg_mse_improvement']:.2f}%")
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
    print("Phase 7.3: EM Clustering Refinement")
    print("="*70)
    
    print(f"\nLoading checkpoint...")
    weights = load_checkpoint_safetensors(checkpoint_dir, num_files=20)
    
    results = test_em_clustering_refinement(weights)
    
    # Save results
    output_file = 'phase7_em_clustering_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to {output_file}")
    print(f"{'='*70}")
    
    return results


if __name__ == '__main__':
    main('nvfp4_checkpoint')

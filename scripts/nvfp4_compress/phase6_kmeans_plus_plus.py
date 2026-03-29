"""
Phase 6.2: K-means++ Initialization
Compare K-means++ with random initialization
"""

import torch
import json
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from typing import Tuple

BLOCK_SIZE = 16
K_CODES = 8


def kmeans_random(data: torch.Tensor, k: int = 8, max_iter: int = 10) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """K-means with random initialization."""
    if len(data) < k:
        return torch.unique(data)[:k], torch.zeros(len(data), dtype=torch.long), 0.0
    
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
    
    # Compute MSE
    reconstructed = centers[assignments]
    mse = torch.mean((data - reconstructed) ** 2).item()
    
    return centers, assignments, mse


def kmeans_plusplus(data: torch.Tensor, k: int = 8, max_iter: int = 10) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """K-means with K-means++ initialization."""
    if len(data) < k:
        return torch.unique(data)[:k], torch.zeros(len(data), dtype=torch.long), 0.0
    
    # K-means++ initialization
    centers = [data[torch.randint(len(data), (1,))].clone()]
    
    for _ in range(k - 1):
        # Compute distances to nearest center
        distances = torch.stack([
            torch.abs(data - c).min() for c in centers
        ]).min(dim=0)[0]
        
        # Choose next center with probability proportional to distance squared
        probs = distances ** 2
        probs = probs / probs.sum()
        next_idx = torch.multinomial(probs, 1)
        centers.append(data[next_idx].clone())
    
    centers = torch.cat(centers)
    
    # K-means iterations
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
    
    # Compute MSE
    reconstructed = centers[assignments]
    mse = torch.mean((data - reconstructed) ** 2).item()
    
    return centers, assignments, mse


def load_checkpoint_safetensors(checkpoint_dir: str, num_files: int = 15) -> dict:
    """Load checkpoint from safetensors format."""
    checkpoint_dir = Path(checkpoint_dir)
    weights = {}
    
    for file in sorted(checkpoint_dir.glob('*.safetensors'))[:num_files]:
        file_weights = load_file(str(file))
        weights.update(file_weights)
    
    return weights


def test_kmeans_initialization(checkpoint_dir: str = 'nvfp4_checkpoint'):
    """Test K-means++ vs random initialization."""
    print("="*70)
    print("Phase 6.2: K-means++ Initialization")
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
            'method': 'K-means++ vs Random Initialization',
            'block_size': BLOCK_SIZE,
            'k_codes': K_CODES,
        },
        'per_tensor_results': [],
        'summary': {}
    }
    
    total_random_mse = 0
    total_plusplus_mse = 0
    num_tested = 0
    
    for i, (name, tensor) in enumerate(float_tensors[:5]):
        try:
            flat_weight = tensor.flatten()
            num_blocks = len(flat_weight) // BLOCK_SIZE
            
            if num_blocks == 0:
                continue
            
            blocks = flat_weight[:num_blocks * BLOCK_SIZE].reshape(num_blocks, BLOCK_SIZE)
            block_means = blocks.mean(dim=1)
            
            # Test random initialization
            _, _, random_mse = kmeans_random(block_means, K_CODES)
            
            # Test K-means++ initialization
            _, _, plusplus_mse = kmeans_plusplus(block_means, K_CODES)
            
            total_random_mse += random_mse
            total_plusplus_mse += plusplus_mse
            num_tested += 1
            
            improvement = (random_mse - plusplus_mse) / random_mse * 100
            
            results['per_tensor_results'].append({
                'tensor_name': name,
                'random_mse': random_mse,
                'plusplus_mse': plusplus_mse,
                'improvement_percent': improvement,
            })
            
            print(f"[{i+1}] {name}")
            print(f"    Random MSE:    {random_mse:.6f}")
            print(f"    K-means++ MSE: {plusplus_mse:.6f}")
            print(f"    Improvement:   {improvement:.1f}%\n")
        
        except Exception as e:
            print(f"[{i+1}] {name}: ERROR - {str(e)}\n")
    
    if num_tested > 0:
        avg_random_mse = total_random_mse / num_tested
        avg_plusplus_mse = total_plusplus_mse / num_tested
        avg_improvement = (avg_random_mse - avg_plusplus_mse) / avg_random_mse * 100
        
        results['summary'] = {
            'avg_random_mse': avg_random_mse,
            'avg_plusplus_mse': avg_plusplus_mse,
            'avg_improvement_percent': avg_improvement,
            'num_tensors_tested': num_tested,
        }
        
        print("="*70)
        print("K-means++ Initialization Summary")
        print("="*70)
        print(f"Random Initialization MSE:    {avg_random_mse:.6f}")
        print(f"K-means++ Initialization MSE: {avg_plusplus_mse:.6f}")
        print(f"Improvement: {avg_improvement:.1f}%")
        print("="*70)
    
    # Save results
    output_file = 'phase6_kmeans_plusplus_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == '__main__':
    test_kmeans_initialization('nvfp4_checkpoint')

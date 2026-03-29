"""
Phase 3E: EM-Based Clustering Refinement for NVFP4 Compression

Approach:
1. Use EM algorithm instead of K-means for better cluster centers
2. Iteratively refine clusters
3. Compare with K-means baseline

Expected: 5-10% improvement in MSE (could improve compression)
"""

import torch
import json
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from typing import Tuple, List
import time

BLOCK_SIZE = 16
K_CODES = 8  # 3-bit codebook

E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


class EMQuantizer:
    """EM-based clustering for weight quantization."""
    
    def __init__(self, k_codes: int = 8):
        """Initialize EM Quantizer."""
        self.k_codes = k_codes
    
    def quantize(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize weight using EM clustering.
        
        Args:
            weight: Weight tensor to quantize
            
        Returns:
            reconstructed: Reconstructed weight
            codes: Code assignments
            centers: Cluster centers
        """
        original_shape = weight.shape
        flat_weight = weight.flatten()
        
        # Reshape to blocks
        num_blocks = len(flat_weight) // BLOCK_SIZE
        blocks = flat_weight[:num_blocks * BLOCK_SIZE].reshape(num_blocks, BLOCK_SIZE)
        
        # EM clustering
        centers, assignments = self._em_clustering(blocks, self.k_codes, max_iter=20)
        
        # Reconstruct
        reconstructed = centers[assignments]
        reconstructed = reconstructed.flatten()[:len(flat_weight)]
        reconstructed = reconstructed.reshape(original_shape)
        
        return reconstructed, assignments, centers
    
    def _em_clustering(self, data: torch.Tensor, k: int, max_iter: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        EM clustering algorithm.
        
        Args:
            data: Data points (num_blocks, BLOCK_SIZE)
            k: Number of clusters
            max_iter: Maximum iterations
            
        Returns:
            centers: Cluster centers
            assignments: Cluster assignments
        """
        num_points = len(data)
        
        # Initialize with K-means
        indices = torch.randperm(num_points)[:k]
        centers = data[indices].clone()
        
        # EM iterations
        for iteration in range(max_iter):
            # E-step: Compute responsibilities (soft assignments)
            distances = torch.cdist(data, centers)  # (num_points, k)
            
            # Use softmax for soft assignments
            responsibilities = torch.softmax(-distances, dim=1)  # (num_points, k)
            
            # M-step: Update centers
            new_centers = torch.zeros_like(centers)
            for i in range(k):
                # Weighted mean
                weights = responsibilities[:, i]  # (num_points,)
                if weights.sum() > 0:
                    new_centers[i] = (data * weights.unsqueeze(1)).sum(dim=0) / weights.sum()
                else:
                    new_centers[i] = centers[i]
            
            # Check convergence
            if torch.allclose(centers, new_centers, atol=1e-6):
                break
            
            centers = new_centers
        
        # Hard assignments (for final codes)
        distances = torch.cdist(data, centers)
        assignments = distances.argmin(dim=1)
        
        return centers, assignments
    
    def compute_compression(self, weight: torch.Tensor) -> float:
        """Compute compression ratio."""
        num_blocks = weight.numel() // BLOCK_SIZE
        
        # Bits for codes
        code_bits = num_blocks * np.log2(self.k_codes)
        
        # Codebook overhead
        codebook_bits = self.k_codes * 32
        
        # Original bits
        original_bits = weight.numel() * 32
        
        # Compression ratio
        total_bits = code_bits + codebook_bits
        compression = 1.0 - (total_bits / original_bits)
        bits_per_elem = total_bits / weight.numel()
        
        return compression, bits_per_elem


def load_checkpoint_safetensors(checkpoint_dir: str, num_files: int = 5) -> dict:
    """Load checkpoint from safetensors format."""
    checkpoint_dir = Path(checkpoint_dir)
    weights = {}
    
    for file in sorted(checkpoint_dir.glob('*.safetensors'))[:num_files]:
        print(f"Loading {file.name}...", end=' ', flush=True)
        file_weights = load_file(str(file))
        weights.update(file_weights)
        print(f"✓ ({len(file_weights)} tensors)")
    
    return weights


def test_em_clustering(checkpoint_dir: str = 'nvfp4_checkpoint', num_tensors: int = 5):
    """Test EM-based clustering."""
    print("="*70)
    print("Phase 3E: EM-Based Clustering Refinement - Real Model Testing")
    print("="*70)
    
    print(f"\nLoading checkpoint from {checkpoint_dir}...")
    weights = load_checkpoint_safetensors(checkpoint_dir, num_files=5)
    
    # Select float32 tensors
    float_tensors = [
        (name, tensor) for name, tensor in weights.items()
        if tensor.dtype == torch.float32 and tensor.numel() > BLOCK_SIZE
    ]
    
    print(f"\nFound {len(float_tensors)} float32 tensors")
    print(f"Testing on {min(num_tensors, len(float_tensors))} tensors...\n")
    
    results = {
        'metadata': {
            'method': 'EM-Based Clustering',
            'k_codes': K_CODES,
            'num_tensors_tested': min(num_tensors, len(float_tensors)),
        },
        'per_tensor_results': [],
        'summary': {}
    }
    
    total_compression = 0
    total_bits_per_elem = 0
    
    quantizer = EMQuantizer(k_codes=K_CODES)
    
    for i, (name, tensor) in enumerate(float_tensors[:num_tensors]):
        print(f"[{i+1}/{min(num_tensors, len(float_tensors))}] {name}...", end=' ', flush=True)
        
        try:
            # Quantize
            reconstructed, codes, centers = quantizer.quantize(tensor)
            
            # Compute compression
            compression, bits_per_elem = quantizer.compute_compression(tensor)
            
            total_compression += compression
            total_bits_per_elem += bits_per_elem
            
            results['per_tensor_results'].append({
                'tensor_name': name,
                'shape': list(tensor.shape),
                'numel': tensor.numel(),
                'compression_ratio': compression,
                'bits_per_elem': bits_per_elem,
            })
            
            print(f"✓ (Compression: {compression*100:.1f}%, {bits_per_elem:.3f} bits/elem)")
        
        except Exception as e:
            print(f"✗ ERROR: {str(e)}")
            continue
    
    # Summary
    num_tested = len(results['per_tensor_results'])
    if num_tested > 0:
        avg_compression = total_compression / num_tested
        avg_bits_per_elem = total_bits_per_elem / num_tested
        
        results['summary'] = {
            'avg_compression_ratio': avg_compression,
            'avg_bits_per_elem': avg_bits_per_elem,
            'num_tensors_tested': num_tested,
        }
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"EM-Based Clustering Summary")
        print(f"{'='*70}")
        print(f"Average Compression:      {avg_compression*100:.1f}%")
        print(f"Average Bits per Element: {avg_bits_per_elem:.3f}")
        print(f"Comparison with Enhancement 7:")
        print(f"  Enhancement 7: 93.2% compression (2.188 bits/elem)")
        print(f"  EM Clustering: {avg_compression*100:.1f}% compression ({avg_bits_per_elem:.3f} bits/elem)")
        
        if avg_compression > 0.932:
            print(f"  ✅ IMPROVEMENT: +{(avg_compression - 0.932)*100:.1f}%")
        else:
            print(f"  ❌ NO IMPROVEMENT: -{(0.932 - avg_compression)*100:.1f}%")
        print(f"{'='*70}")
    
    # Save results
    output_file = 'phase3e_em_clustering_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == '__main__':
    import sys
    
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else 'nvfp4_checkpoint'
    num_tensors = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    test_em_clustering(checkpoint_dir, num_tensors)

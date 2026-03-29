"""
Phase 3C: Hierarchical Codebooks for NVFP4 Compression
Reference: AQLM (2401.06118)

Approach:
1. Build coarse codebook (4 codes, 2 bits)
2. Build fine codebook (8 codes, 3 bits) for each coarse code
3. Two-stage quantization: coarse + fine

Expected: 45-50% compression (vs 93.2% baseline)
"""

import torch
import json
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from typing import Tuple, List
import time

BLOCK_SIZE = 16

E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


class HierarchicalQuantizer:
    """Hierarchical Codebook Quantization."""
    
    def __init__(self, num_coarse: int = 4, num_fine: int = 8):
        """
        Initialize Hierarchical Quantizer.
        
        Args:
            num_coarse: Number of coarse codes (2 bits)
            num_fine: Number of fine codes per coarse code (3 bits)
        """
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.coarse_codebook = None
        self.fine_codebooks = None
    
    def quantize(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Quantize weight using hierarchical codebooks.
        
        Args:
            weight: Weight tensor to quantize
            
        Returns:
            reconstructed: Reconstructed weight
            coarse_codes: Coarse code assignments
            fine_codes: Fine code assignments
            codebooks: [coarse_codebook, fine_codebooks]
        """
        original_shape = weight.shape
        flat_weight = weight.flatten()
        
        # Reshape to blocks
        num_blocks = len(flat_weight) // BLOCK_SIZE
        blocks = flat_weight[:num_blocks * BLOCK_SIZE].reshape(num_blocks, BLOCK_SIZE)
        
        # Stage 1: Coarse quantization
        coarse_centers = self._kmeans(blocks.mean(dim=1), self.num_coarse, max_iter=10)
        
        # Assign blocks to coarse centers
        block_means = blocks.mean(dim=1)
        distances = torch.cdist(block_means.unsqueeze(1), coarse_centers.unsqueeze(1))
        coarse_codes = distances.argmin(dim=1)
        
        # Stage 2: Fine quantization (per coarse code)
        fine_codebooks = []
        fine_codes = torch.zeros_like(coarse_codes)
        reconstructed_blocks = torch.zeros_like(blocks)
        
        for coarse_idx in range(self.num_coarse):
            # Get blocks assigned to this coarse code
            mask = coarse_codes == coarse_idx
            if not mask.any():
                fine_codebooks.append(coarse_centers[coarse_idx:coarse_idx+1])
                continue
            
            coarse_blocks = blocks[mask]
            
            # Compute residuals
            residuals = coarse_blocks - coarse_centers[coarse_idx]
            
            # K-means on residuals
            fine_centers = self._kmeans(residuals.mean(dim=1), self.num_fine, max_iter=10)
            
            # Assign residuals to fine centers
            residual_means = residuals.mean(dim=1)
            distances = torch.cdist(residual_means.unsqueeze(1), fine_centers.unsqueeze(1))
            fine_codes_local = distances.argmin(dim=1)
            
            # Store fine codes
            fine_codes[mask] = fine_codes_local
            
            # Reconstruct
            reconstructed_blocks[mask] = coarse_centers[coarse_idx] + fine_centers[fine_codes_local]
            
            fine_codebooks.append(fine_centers)
        
        # Reconstruct full weight
        reconstructed = reconstructed_blocks.flatten()[:len(flat_weight)]
        reconstructed = reconstructed.reshape(original_shape)
        
        return reconstructed, coarse_codes, fine_codes, [coarse_centers, fine_codebooks]
    
    def _kmeans(self, data: torch.Tensor, k: int, max_iter: int = 10) -> torch.Tensor:
        """Simple K-means clustering."""
        if len(data) < k:
            return torch.unique(data)[:k]
        
        # Initialize with k random points
        indices = torch.randperm(len(data))[:k]
        centers = data[indices].clone()
        
        for _ in range(max_iter):
            # Assign points to nearest center
            distances = torch.cdist(data.unsqueeze(1), centers.unsqueeze(1))
            assignments = distances.argmin(dim=1)
            
            # Update centers
            new_centers = torch.stack([
                data[assignments == i].mean() if (assignments == i).any() else centers[i]
                for i in range(k)
            ])
            
            # Check convergence
            if torch.allclose(centers, new_centers, atol=1e-6):
                break
            
            centers = new_centers
        
        return centers
    
    def compute_compression(self, weight: torch.Tensor, coarse_codes: torch.Tensor, 
                           fine_codes: torch.Tensor) -> float:
        """Compute compression ratio."""
        # Coarse codes: 2 bits per block
        coarse_bits = len(coarse_codes) * 2
        
        # Fine codes: 3 bits per block
        fine_bits = len(fine_codes) * 3
        
        # Codebook overhead (minimal)
        codebook_bits = (self.num_coarse + self.num_fine * self.num_coarse) * 32
        
        # Original bits
        original_bits = weight.numel() * 32
        
        # Compression ratio
        total_bits = coarse_bits + fine_bits + codebook_bits
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


def test_hierarchical_codebooks(checkpoint_dir: str = 'nvfp4_checkpoint', num_tensors: int = 5):
    """Test Hierarchical Codebooks on real model."""
    print("="*70)
    print("Phase 3C: Hierarchical Codebooks - Real Model Testing")
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
            'method': 'Hierarchical Codebooks',
            'num_coarse': 4,
            'num_fine': 8,
            'num_tensors_tested': min(num_tensors, len(float_tensors)),
        },
        'per_tensor_results': [],
        'summary': {}
    }
    
    total_compression = 0
    total_bits_per_elem = 0
    
    quantizer = HierarchicalQuantizer(num_coarse=4, num_fine=8)
    
    for i, (name, tensor) in enumerate(float_tensors[:num_tensors]):
        print(f"[{i+1}/{min(num_tensors, len(float_tensors))}] {name}...", end=' ', flush=True)
        
        try:
            # Quantize
            reconstructed, coarse_codes, fine_codes, codebooks = quantizer.quantize(tensor)
            
            # Compute compression
            compression, bits_per_elem = quantizer.compute_compression(tensor, coarse_codes, fine_codes)
            
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
        print(f"Hierarchical Codebooks Summary")
        print(f"{'='*70}")
        print(f"Average Compression:      {avg_compression*100:.1f}%")
        print(f"Average Bits per Element: {avg_bits_per_elem:.3f}")
        print(f"Comparison with Enhancement 7:")
        print(f"  Enhancement 7:    93.2% compression (2.188 bits/elem)")
        print(f"  Hierarchical:     {avg_compression*100:.1f}% compression ({avg_bits_per_elem:.3f} bits/elem)")
        
        if avg_compression > 0.932:
            print(f"  ✅ IMPROVEMENT: +{(avg_compression - 0.932)*100:.1f}%")
        else:
            print(f"  ❌ NO IMPROVEMENT: -{(0.932 - avg_compression)*100:.1f}%")
        print(f"{'='*70}")
    
    # Save results
    output_file = 'phase3c_hierarchical_codebooks_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == '__main__':
    import sys
    
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else 'nvfp4_checkpoint'
    num_tensors = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    test_hierarchical_codebooks(checkpoint_dir, num_tensors)

"""
Phase 3B: Product Quantization for NVFP4 Compression
Reference: Product Quantization (1411.4280)

Approach:
1. Divide weight matrix into M sub-matrices
2. Quantize each sub-matrix independently with K-means
3. Combine results for final compression

Expected: 50-75% compression (vs 93.2% baseline)
"""

import torch
import json
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from typing import Tuple, List
import time

BLOCK_SIZE = 16
K_CODES = 8  # 3-bit codebook per sub-matrix

E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


class ProductQuantizer:
    """Product Quantization for weight compression."""
    
    def __init__(self, num_subvectors: int = 4, k_codes: int = 8):
        """
        Initialize Product Quantizer.
        
        Args:
            num_subvectors: Number of sub-matrices to divide into
            k_codes: Number of codes per sub-matrix (8 = 3 bits)
        """
        self.num_subvectors = num_subvectors
        self.k_codes = k_codes
        self.codebooks = []
    
    def quantize(self, weight: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Quantize weight using Product Quantization.
        
        Args:
            weight: Weight tensor to quantize
            
        Returns:
            reconstructed: Reconstructed weight
            codes: List of code assignments for each sub-matrix
            codebooks: List of codebooks for each sub-matrix
        """
        original_shape = weight.shape
        flat_weight = weight.flatten()
        
        # Divide into sub-vectors
        subvector_size = len(flat_weight) // self.num_subvectors
        
        reconstructed_parts = []
        codes_list = []
        codebooks_list = []
        
        for i in range(self.num_subvectors):
            start_idx = i * subvector_size
            end_idx = (i + 1) * subvector_size if i < self.num_subvectors - 1 else len(flat_weight)
            
            subvector = flat_weight[start_idx:end_idx]
            
            # K-means quantization for this sub-vector
            centers = self._kmeans(subvector, self.k_codes, max_iter=10)
            
            # Assign to nearest center
            distances = torch.cdist(subvector.unsqueeze(1), centers.unsqueeze(1))
            codes = distances.argmin(dim=1)
            
            # Reconstruct
            reconstructed = centers[codes]
            
            reconstructed_parts.append(reconstructed)
            codes_list.append(codes)
            codebooks_list.append(centers)
        
        # Combine reconstructed parts
        reconstructed = torch.cat(reconstructed_parts)
        reconstructed = reconstructed.reshape(original_shape)
        
        return reconstructed, codes_list, codebooks_list
    
    def _kmeans(self, data: torch.Tensor, k: int, max_iter: int = 10) -> torch.Tensor:
        """Simple K-means clustering."""
        if len(data) < k:
            # If fewer points than clusters, just return unique values
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
    
    def compute_compression(self, weight: torch.Tensor, codes_list: List[torch.Tensor], 
                           codebooks_list: List[torch.Tensor]) -> float:
        """Compute compression ratio."""
        # Each sub-vector: log2(k_codes) bits per element
        bits_per_code = np.log2(self.k_codes)
        
        # Total bits for codes
        total_code_bits = sum(len(codes) * bits_per_code for codes in codes_list)
        
        # Codebook overhead (minimal, shared across all)
        codebook_bits = sum(len(cb) * 32 for cb in codebooks_list)
        
        # Original bits
        original_bits = weight.numel() * 32
        
        # Compression ratio
        total_bits = total_code_bits + codebook_bits
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


def test_product_quantization(checkpoint_dir: str = 'nvfp4_checkpoint', num_tensors: int = 5):
    """Test Product Quantization on real model."""
    print("="*70)
    print("Phase 3B: Product Quantization - Real Model Testing")
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
            'method': 'Product Quantization',
            'num_subvectors': 4,
            'k_codes': K_CODES,
            'num_tensors_tested': min(num_tensors, len(float_tensors)),
        },
        'per_tensor_results': [],
        'summary': {}
    }
    
    total_compression = 0
    total_bits_per_elem = 0
    
    quantizer = ProductQuantizer(num_subvectors=4, k_codes=K_CODES)
    
    for i, (name, tensor) in enumerate(float_tensors[:num_tensors]):
        print(f"[{i+1}/{min(num_tensors, len(float_tensors))}] {name}...", end=' ', flush=True)
        
        try:
            # Quantize
            reconstructed, codes_list, codebooks_list = quantizer.quantize(tensor)
            
            # Compute compression
            compression, bits_per_elem = quantizer.compute_compression(tensor, codes_list, codebooks_list)
            
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
        print(f"Product Quantization Summary")
        print(f"{'='*70}")
        print(f"Average Compression:      {avg_compression*100:.1f}%")
        print(f"Average Bits per Element: {avg_bits_per_elem:.3f}")
        print(f"Comparison with Enhancement 7:")
        print(f"  Enhancement 7: 93.2% compression (2.188 bits/elem)")
        print(f"  Product VQ:    {avg_compression*100:.1f}% compression ({avg_bits_per_elem:.3f} bits/elem)")
        
        if avg_compression > 0.932:
            print(f"  ✅ IMPROVEMENT: +{(avg_compression - 0.932)*100:.1f}%")
        else:
            print(f"  ❌ NO IMPROVEMENT: -{(0.932 - avg_compression)*100:.1f}%")
        print(f"{'='*70}")
    
    # Save results
    output_file = 'phase3b_product_quantization_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == '__main__':
    import sys
    
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else 'nvfp4_checkpoint'
    num_tensors = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    test_product_quantization(checkpoint_dir, num_tensors)

"""
Phase 3D: Bit-Width Optimization for NVFP4 Compression

Approach:
1. Current: Stage 1 (3 bits) + Stage 2 (2 bits) = 5 bits/block
2. Test: Different allocations (4+1, 2+3, etc.)
3. Find optimal bit allocation

Expected: 5-15% improvement over Enhancement 7 (93.2%)
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


class BitWidthOptimizer:
    """Optimize bit allocation across quantization stages."""
    
    def __init__(self, stage1_bits: int = 3, stage2_bits: int = 2):
        """
        Initialize BitWidth Optimizer.
        
        Args:
            stage1_bits: Bits for Stage 1 (K-means codes)
            stage2_bits: Bits for Stage 2 (residuals)
        """
        self.stage1_bits = stage1_bits
        self.stage2_bits = stage2_bits
        self.stage1_codes = 2 ** stage1_bits
        self.stage2_codes = 2 ** stage2_bits
    
    def quantize(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Quantize weight with specified bit allocation.
        
        Args:
            weight: Weight tensor to quantize
            
        Returns:
            reconstructed: Reconstructed weight
            stage1_codes: Stage 1 code assignments
            stage2_codes: Stage 2 code assignments
            codebooks: [stage1_codebook, stage2_codebook]
        """
        original_shape = weight.shape
        flat_weight = weight.flatten()
        
        # Reshape to blocks
        num_blocks = len(flat_weight) // BLOCK_SIZE
        blocks = flat_weight[:num_blocks * BLOCK_SIZE].reshape(num_blocks, BLOCK_SIZE)
        
        # Stage 1: K-means with stage1_codes clusters
        stage1_centers = self._kmeans(blocks.mean(dim=1), self.stage1_codes, max_iter=10)
        
        # Assign blocks to nearest center
        block_means = blocks.mean(dim=1)
        distances = torch.cdist(block_means.unsqueeze(1), stage1_centers.unsqueeze(1))
        stage1_codes = distances.argmin(dim=1)
        
        # Reconstruct Stage 1
        stage1_reconstructed = stage1_centers[stage1_codes]
        
        # Stage 2: Residual quantization with stage2_codes levels
        residuals = block_means - stage1_reconstructed
        
        # Quantize residuals to stage2_codes levels
        min_val = residuals.min()
        max_val = residuals.max()
        
        if max_val > min_val:
            normalized = (residuals - min_val) / (max_val - min_val) * (self.stage2_codes - 1)
            stage2_codes = normalized.round().clamp(0, self.stage2_codes - 1).to(torch.uint8)
        else:
            stage2_codes = torch.zeros_like(residuals, dtype=torch.uint8)
        
        # Reconstruct Stage 2
        stage2_reconstructed = stage2_codes.float() / (self.stage2_codes - 1) * (max_val - min_val) + min_val
        
        # Final reconstruction
        final_reconstructed = stage1_reconstructed + stage2_reconstructed
        
        # Expand to full tensor
        reconstructed = torch.zeros_like(flat_weight)
        for i in range(num_blocks):
            reconstructed[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE] = final_reconstructed[i]
        
        reconstructed = reconstructed.reshape(original_shape)
        
        return reconstructed, stage1_codes, stage2_codes, [stage1_centers]
    
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
    
    def compute_compression(self, weight: torch.Tensor) -> float:
        """Compute compression ratio."""
        num_blocks = weight.numel() // BLOCK_SIZE
        
        # Bits for codes
        stage1_bits = num_blocks * self.stage1_bits
        stage2_bits = num_blocks * self.stage2_bits
        
        # Codebook overhead (minimal)
        codebook_bits = self.stage1_codes * 32
        
        # Original bits
        original_bits = weight.numel() * 32
        
        # Compression ratio
        total_bits = stage1_bits + stage2_bits + codebook_bits
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


def test_bitwidth_optimization(checkpoint_dir: str = 'nvfp4_checkpoint', num_tensors: int = 5):
    """Test different bit-width allocations."""
    print("="*70)
    print("Phase 3D: Bit-Width Optimization - Real Model Testing")
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
    
    # Test different bit allocations
    bit_allocations = [
        (2, 3),  # 2 bits stage1, 3 bits stage2
        (3, 2),  # 3 bits stage1, 2 bits stage2 (current)
        (4, 1),  # 4 bits stage1, 1 bit stage2
        (1, 4),  # 1 bit stage1, 4 bits stage2
    ]
    
    results = {
        'metadata': {
            'method': 'Bit-Width Optimization',
            'num_tensors_tested': min(num_tensors, len(float_tensors)),
            'bit_allocations_tested': bit_allocations,
        },
        'per_allocation_results': [],
        'summary': {}
    }
    
    best_compression = 0.932  # Enhancement 7 baseline
    best_allocation = (3, 2)
    
    for stage1_bits, stage2_bits in bit_allocations:
        print(f"\nTesting allocation: {stage1_bits} bits (Stage 1) + {stage2_bits} bits (Stage 2)")
        print("-" * 70)
        
        optimizer = BitWidthOptimizer(stage1_bits=stage1_bits, stage2_bits=stage2_bits)
        
        total_compression = 0
        total_bits_per_elem = 0
        num_tested = 0
        
        for i, (name, tensor) in enumerate(float_tensors[:num_tensors]):
            print(f"  [{i+1}/{min(num_tensors, len(float_tensors))}] {name}...", end=' ', flush=True)
            
            try:
                # Quantize
                reconstructed, stage1_codes, stage2_codes, codebooks = optimizer.quantize(tensor)
                
                # Compute compression
                compression, bits_per_elem = optimizer.compute_compression(tensor)
                
                total_compression += compression
                total_bits_per_elem += bits_per_elem
                num_tested += 1
                
                print(f"✓ ({compression*100:.1f}%, {bits_per_elem:.3f} bits/elem)")
            
            except Exception as e:
                print(f"✗ ERROR: {str(e)}")
                continue
        
        if num_tested > 0:
            avg_compression = total_compression / num_tested
            avg_bits_per_elem = total_bits_per_elem / num_tested
            
            results['per_allocation_results'].append({
                'stage1_bits': stage1_bits,
                'stage2_bits': stage2_bits,
                'total_bits': stage1_bits + stage2_bits,
                'avg_compression_ratio': avg_compression,
                'avg_bits_per_elem': avg_bits_per_elem,
                'improvement_vs_e7': (avg_compression - 0.932) * 100,
            })
            
            print(f"  Average: {avg_compression*100:.1f}% compression ({avg_bits_per_elem:.3f} bits/elem)")
            
            if avg_compression > best_compression:
                best_compression = avg_compression
                best_allocation = (stage1_bits, stage2_bits)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Bit-Width Optimization Summary")
    print(f"{'='*70}")
    print(f"Enhancement 7 (baseline): 93.2% compression (3+2 bits)")
    print(f"Best allocation found:    {best_compression*100:.1f}% compression ({best_allocation[0]}+{best_allocation[1]} bits)")
    
    if best_compression > 0.932:
        print(f"✅ IMPROVEMENT: +{(best_compression - 0.932)*100:.1f}%")
    else:
        print(f"❌ NO IMPROVEMENT")
    
    print(f"{'='*70}")
    
    results['summary'] = {
        'best_allocation': best_allocation,
        'best_compression': best_compression,
        'improvement_vs_e7': (best_compression - 0.932) * 100,
    }
    
    # Save results
    output_file = 'phase3d_bitwidth_optimization_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == '__main__':
    import sys
    
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else 'nvfp4_checkpoint'
    num_tensors = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    test_bitwidth_optimization(checkpoint_dir, num_tensors)

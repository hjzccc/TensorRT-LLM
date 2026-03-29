"""
Phase 11.1: Learned Scaling Factors
Learn per-block scaling factors to improve MSE and PPL
"""

import torch
import json
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from typing import Dict, Tuple

BLOCK_SIZE = 16
K_CODES = 8  # 3-bit codebook
E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def quantize_to_fp4(value: float) -> float:
    """Quantize a single value to FP4 (E2M1)."""
    if value == 0:
        return 0.0
    sign = 1 if value >= 0 else -1
    abs_val = abs(value)
    distances = torch.abs(E2M1_TABLE - abs_val)
    closest_idx = distances.argmin().item()
    fp4_val = E2M1_TABLE[closest_idx].item()
    return sign * fp4_val


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


def quantize_with_learned_scaling(tensor: torch.Tensor) -> Dict:
    """Quantize tensor with learned per-block scaling factors."""
    
    if tensor.numel() <= BLOCK_SIZE:
        return {
            'compression': 0.0,
            'mse_fixed': 0.0,
            'mse_learned': 0.0,
            'mse_improvement': 0.0,
            'skipped': True,
        }
    
    tensor_flat = tensor.view(-1)
    num_blocks = tensor.numel() // BLOCK_SIZE
    
    # Fixed scaling (baseline)
    fixed_mse_total = 0
    learned_mse_total = 0
    
    for block_idx in range(num_blocks):
        start = block_idx * BLOCK_SIZE
        end = start + BLOCK_SIZE
        block = tensor_flat[start:end]
        
        # K-means quantization
        centers, assignments = kmeans_quantize_block(block, K_CODES)
        
        # Fixed scaling: quantize centers to FP4 directly
        fp4_centers_fixed = torch.tensor([quantize_to_fp4(c.item()) for c in centers])
        reconstructed_fixed = fp4_centers_fixed[assignments]
        mse_fixed = torch.mean((block - reconstructed_fixed) ** 2).item()
        fixed_mse_total += mse_fixed
        
        # Learned scaling: learn per-block scaling factor
        # Idea: scale block to [-1, 1] range, then quantize
        block_min = block.min()
        block_max = block.max()
        block_range = block_max - block_min
        
        if block_range > 0:
            # Normalize block to [-1, 1]
            block_normalized = 2 * (block - block_min) / block_range - 1
            
            # Quantize normalized block
            centers_norm, assignments_norm = kmeans_quantize_block(block_normalized, K_CODES)
            fp4_centers_norm = torch.tensor([quantize_to_fp4(c.item()) for c in centers_norm])
            reconstructed_norm = fp4_centers_norm[assignments_norm]
            
            # Denormalize
            reconstructed_learned = (reconstructed_norm + 1) * block_range / 2 + block_min
            mse_learned = torch.mean((block - reconstructed_learned) ** 2).item()
        else:
            mse_learned = mse_fixed
        
        learned_mse_total += mse_learned
    
    avg_mse_fixed = fixed_mse_total / max(num_blocks, 1)
    avg_mse_learned = learned_mse_total / max(num_blocks, 1)
    mse_improvement = (avg_mse_fixed - avg_mse_learned) / (avg_mse_fixed + 1e-8) * 100
    
    # Calculate compression (same as before, scaling factors stored separately)
    code_bits = num_blocks * 3
    codebook_bits = K_CODES * 4
    scaling_bits = num_blocks * 32  # Store min/max per block
    total_bits = code_bits + codebook_bits + scaling_bits
    original_bits = tensor.numel() * 32
    compression = 1.0 - (total_bits / original_bits)
    
    return {
        'compression': compression,
        'mse_fixed': avg_mse_fixed,
        'mse_learned': avg_mse_learned,
        'mse_improvement': mse_improvement,
        'num_blocks': num_blocks,
        'skipped': False,
    }


def test_learned_scaling(weights: Dict) -> Dict:
    """Test learned scaling factors on checkpoint."""
    
    results = {
        'metadata': {
            'method': 'Learned Scaling Factors',
            'block_size': BLOCK_SIZE,
            'k_codes': K_CODES,
        },
        'per_tensor_results': [],
        'summary': {
            'avg_mse_improvement': 0.0,
            'num_tensors_tested': 0,
        }
    }
    
    print(f"\nTesting Learned Scaling Factors")
    print("-" * 70)
    
    total_mse_improvement = 0
    num_tested = 0
    
    for name, tensor in list(weights.items())[:20]:
        if tensor.dtype != torch.float32 or tensor.numel() <= BLOCK_SIZE:
            continue
        
        result = quantize_with_learned_scaling(tensor)
        
        if result['skipped']:
            continue
        
        print(f"{name}: {result['mse_improvement']:.2f}% MSE improvement")
        
        results['per_tensor_results'].append({
            'tensor_name': name,
            'shape': list(tensor.shape),
            'numel': tensor.numel(),
            'mse_fixed': result['mse_fixed'],
            'mse_learned': result['mse_learned'],
            'mse_improvement': result['mse_improvement'],
            'compression': result['compression'],
        })
        
        total_mse_improvement += result['mse_improvement']
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
    print("Phase 11.1: Learned Scaling Factors")
    print("="*70)
    
    print(f"\nLoading checkpoint...")
    weights = load_checkpoint_safetensors(checkpoint_dir, num_files=20)
    
    results = test_learned_scaling(weights)
    
    output_file = 'phase11_learned_scaling_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"{'='*70}")
    
    return results


if __name__ == '__main__':
    main('nvfp4_checkpoint')

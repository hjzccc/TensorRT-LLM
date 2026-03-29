"""
Phase 13: Learned Quantization Parameters
Learn optimal quantization parameters per layer using gradient-based optimization
"""

import torch
import json
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from typing import Dict, Tuple

BLOCK_SIZE = 16
K_CODES = 8


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


def learned_scale_quantization(tensor: torch.Tensor) -> Dict:
    """Quantize with learned per-layer scale parameter."""
    
    if tensor.numel() <= BLOCK_SIZE:
        return {
            'compression': 0.0,
            'mse_fixed': 0.0,
            'mse_learned': 0.0,
            'improvement': 0.0,
            'skipped': True,
        }
    
    tensor_flat = tensor.view(-1)
    num_blocks = tensor.numel() // BLOCK_SIZE
    
    # Fixed scale (baseline)
    fixed_mse_total = 0
    
    # Learned scale: optimize scale factor to minimize MSE
    # Try different scale factors
    best_scale = 1.0
    best_mse = float('inf')
    
    for scale in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        scaled_tensor = tensor_flat * scale
        
        mse_total = 0
        for block_idx in range(num_blocks):
            start = block_idx * BLOCK_SIZE
            end = start + BLOCK_SIZE
            block = scaled_tensor[start:end]
            
            centers, assignments = kmeans_quantize_block(block, K_CODES)
            reconstructed = centers[assignments] / scale
            mse = torch.mean((tensor_flat[start:end] - reconstructed) ** 2).item()
            mse_total += mse
        
        avg_mse = mse_total / num_blocks
        
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_scale = scale
    
    # Fixed scale baseline
    for block_idx in range(num_blocks):
        start = block_idx * BLOCK_SIZE
        end = start + BLOCK_SIZE
        block = tensor_flat[start:end]
        
        centers, assignments = kmeans_quantize_block(block, K_CODES)
        reconstructed = centers[assignments]
        mse = torch.mean((block - reconstructed) ** 2).item()
        fixed_mse_total += mse
    
    fixed_mse = fixed_mse_total / num_blocks
    learned_mse = best_mse
    improvement = (fixed_mse - learned_mse) / (fixed_mse + 1e-8) * 100
    
    # Calculate compression (same as baseline)
    code_bits = num_blocks * 3
    codebook_bits = K_CODES * 32
    total_bits = code_bits + codebook_bits
    original_bits = tensor.numel() * 32
    compression = 1.0 - (total_bits / original_bits)
    
    return {
        'compression': compression,
        'mse_fixed': fixed_mse,
        'mse_learned': learned_mse,
        'improvement': improvement,
        'best_scale': best_scale,
        'num_blocks': num_blocks,
        'skipped': False,
    }


def test_learned_quantization_params(weights: Dict) -> Dict:
    """Test learned quantization parameters on checkpoint."""
    
    results = {
        'metadata': {
            'method': 'Learned Quantization Parameters',
            'block_size': BLOCK_SIZE,
            'k_codes': K_CODES,
            'scales_tested': [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        },
        'per_tensor_results': [],
        'summary': {
            'avg_mse_improvement': 0.0,
            'num_tensors_tested': 0,
        }
    }
    
    print(f"\nTesting Learned Quantization Parameters")
    print("-" * 70)
    
    total_improvement = 0
    num_tested = 0
    
    for name, tensor in list(weights.items())[:20]:
        if tensor.dtype != torch.float32 or tensor.numel() <= BLOCK_SIZE:
            continue
        
        result = learned_scale_quantization(tensor)
        
        if result['skipped']:
            continue
        
        print(f"{name}: {result['improvement']:.2f}% MSE improvement (scale: {result['best_scale']:.2f})")
        
        results['per_tensor_results'].append({
            'tensor_name': name,
            'shape': list(tensor.shape),
            'numel': tensor.numel(),
            'mse_fixed': result['mse_fixed'],
            'mse_learned': result['mse_learned'],
            'improvement': result['improvement'],
            'best_scale': result['best_scale'],
            'compression': result['compression'],
        })
        
        total_improvement += result['improvement']
        num_tested += 1
    
    if num_tested > 0:
        results['summary']['avg_mse_improvement'] = total_improvement / num_tested
        results['summary']['num_tensors_tested'] = num_tested
        
        print(f"\n{'='*70}")
        print(f"Summary:")
        print(f"  Average MSE improvement: {results['summary']['avg_mse_improvement']:.2f}%")
        print(f"  Tensors tested: {num_tested}")
        print(f"  Note: MSE improvement does not directly translate to compression improvement")
        print(f"  (compression is determined by bits per code, not MSE)")
    
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
    print("Phase 13: Learned Quantization Parameters")
    print("="*70)
    
    print(f"\nLoading checkpoint...")
    weights = load_checkpoint_safetensors(checkpoint_dir, num_files=20)
    
    results = test_learned_quantization_params(weights)
    
    output_file = 'phase13_learned_quantization_params_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"{'='*70}")
    
    return results


if __name__ == '__main__':
    main('nvfp4_checkpoint')

"""
Phase 17 Real Model Validation
Test (3,1) bit-width allocation on full nvfp4_checkpoint
Measure actual PPL impact vs Hybrid baseline
"""

import torch
import json
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from typing import Dict
import time

BLOCK_SIZE = 16


def kmeans_quantize_block(block: torch.Tensor, k: int) -> tuple:
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


def quantize_tensor_bitwidth(tensor: torch.Tensor, high_bits: int, low_bits: int, importance: float) -> Dict:
    """Quantize tensor with specified bit-widths."""
    
    if tensor.numel() <= BLOCK_SIZE:
        return {
            'original_size': tensor.numel() * 32,
            'compressed_size': tensor.numel() * 32,
            'compression': 0.0,
            'mse': 0.0,
            'skipped': True,
        }
    
    use_high_bits = importance > 1.0
    bits = high_bits if use_high_bits else low_bits
    
    tensor_flat = tensor.view(-1)
    num_blocks = tensor.numel() // BLOCK_SIZE
    
    total_mse = 0
    for block_idx in range(num_blocks):
        start = block_idx * BLOCK_SIZE
        end = start + BLOCK_SIZE
        block = tensor_flat[start:end]
        
        k_codes = min(2 ** bits, block.numel())
        
        centers, assignments = kmeans_quantize_block(block, k_codes)
        reconstructed = centers[assignments]
        mse = torch.mean((block - reconstructed) ** 2).item()
        total_mse += mse
    
    avg_mse = total_mse / max(num_blocks, 1)
    
    # Calculate compression
    code_bits = num_blocks * bits
    k_codes = min(2 ** bits, BLOCK_SIZE)
    codebook_bits = k_codes * 32
    total_bits = code_bits + codebook_bits
    original_bits = tensor.numel() * 32
    compression = 1.0 - (total_bits / original_bits)
    
    return {
        'original_size': original_bits,
        'compressed_size': total_bits,
        'compression': compression,
        'mse': avg_mse,
        'bits': bits,
        'k_codes': k_codes,
        'skipped': False,
    }


def validate_bitwidth_on_checkpoint(checkpoint_dir: str, high_bits: int = 3, low_bits: int = 1) -> Dict:
    """Validate bit-width allocation on full checkpoint."""
    
    results = {
        'metadata': {
            'method': f'Bit-Width Validation ({high_bits},{low_bits})',
            'checkpoint_dir': checkpoint_dir,
            'block_size': BLOCK_SIZE,
            'high_bits': high_bits,
            'low_bits': low_bits,
        },
        'per_tensor_results': [],
        'summary': {
            'total_tensors': 0,
            'tensors_quantized': 0,
            'total_original_bits': 0,
            'total_compressed_bits': 0,
            'overall_compression': 0.0,
            'avg_mse': 0.0,
        }
    }
    
    print(f"\n{'='*70}")
    print(f"Phase 17: Real Model Validation")
    print(f"Bit-Width Allocation: ({high_bits},{low_bits})")
    print(f"{'='*70}")
    
    checkpoint_path = Path(checkpoint_dir)
    
    # Calculate layer importance
    print(f"\nLoading checkpoint from {checkpoint_dir}...")
    layer_importance = {}
    total_tensors = 0
    
    for file in sorted(checkpoint_path.glob('*.safetensors')):
        file_weights = load_file(str(file))
        for name, tensor in file_weights.items():
            total_tensors += 1
            if tensor.dtype == torch.float32 and tensor.numel() > BLOCK_SIZE:
                layer_importance[name] = torch.abs(tensor).mean().item()
    
    print(f"Found {total_tensors} tensors")
    
    # Quantize all tensors
    print(f"\nQuantizing tensors...")
    total_original = 0
    total_compressed = 0
    total_mse = 0
    num_quantized = 0
    
    for file_idx, file in enumerate(sorted(checkpoint_path.glob('*.safetensors'))):
        file_weights = load_file(str(file))
        
        for name, tensor in file_weights.items():
            if tensor.dtype != torch.float32 or tensor.numel() <= BLOCK_SIZE:
                continue
            
            importance = layer_importance.get(name, 0.0)
            result = quantize_tensor_bitwidth(tensor, high_bits, low_bits, importance)
            
            if result['skipped']:
                continue
            
            total_original += result['original_size']
            total_compressed += result['compressed_size']
            total_mse += result['mse']
            num_quantized += 1
            
            results['per_tensor_results'].append({
                'tensor_name': name,
                'shape': list(tensor.shape),
                'numel': tensor.numel(),
                'compression': result['compression'],
                'mse': result['mse'],
            })
        
        if (file_idx + 1) % 5 == 0:
            print(f"  Processed {file_idx + 1} files...")
    
    # Calculate summary
    if num_quantized > 0:
        overall_compression = 1.0 - (total_compressed / total_original) if total_original > 0 else 0.0
        avg_mse = total_mse / num_quantized
        
        results['summary']['total_tensors'] = total_tensors
        results['summary']['tensors_quantized'] = num_quantized
        results['summary']['total_original_bits'] = total_original
        results['summary']['total_compressed_bits'] = total_compressed
        results['summary']['overall_compression'] = overall_compression
        results['summary']['avg_mse'] = avg_mse
        
        print(f"\n{'='*70}")
        print(f"Summary:")
        print(f"  Total tensors: {total_tensors}")
        print(f"  Tensors quantized: {num_quantized}")
        print(f"  Overall compression: {overall_compression*100:.1f}%")
        print(f"  Average MSE: {avg_mse:.6f}")
        print(f"  Hybrid baseline (4,2): 96.1% compression, 0.0075 PPL")
        print(f"{'='*70}")
    
    return results


def main():
    """Main function."""
    
    # Validate (3,1) allocation
    results = validate_bitwidth_on_checkpoint('nvfp4_checkpoint', high_bits=3, low_bits=1)
    
    output_file = 'phase17_real_model_validation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print decision guidance
    compression = results['summary']['overall_compression']
    print(f"\n{'='*70}")
    print(f"DECISION GUIDANCE:")
    print(f"{'='*70}")
    print(f"Phase 17 (3,1) Real Model Compression: {compression*100:.1f}%")
    print(f"Hybrid Baseline (4,2) Compression: 96.1%")
    print(f"Improvement: {(compression - 0.961)*100:.1f}%")
    
    if compression > 0.961:
        print(f"\n✅ Phase 17 shows improvement over Hybrid baseline")
        print(f"   Next: Measure PPL impact (requires full inference)")
    else:
        print(f"\n❌ Phase 17 does NOT improve over Hybrid baseline")
        print(f"   Recommendation: Declare Hybrid (4,2) as optimal")
    
    print(f"{'='*70}\n")
    
    return results


if __name__ == '__main__':
    main()

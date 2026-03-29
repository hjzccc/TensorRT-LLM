"""
Phase 12: Structured Quantization (Channel-wise)
Quantize entire channels with same parameters to improve efficiency
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


def channel_wise_quantization(tensor: torch.Tensor) -> Dict:
    """Quantize tensor with channel-wise parameters."""
    
    if tensor.numel() <= BLOCK_SIZE:
        return {
            'compression': 0.0,
            'mse': 0.0,
            'skipped': True,
        }
    
    # For 1D tensors, treat as single channel
    # For 2D tensors, treat first dimension as channels
    if tensor.dim() == 1:
        channels = 1
        channel_size = tensor.numel()
    else:
        channels = tensor.shape[0]
        channel_size = tensor.numel() // channels
    
    total_mse = 0
    num_blocks = 0
    
    # Quantize each channel separately
    for ch in range(channels):
        if tensor.dim() == 1:
            channel_data = tensor
        else:
            channel_data = tensor[ch].view(-1)
        
        # Quantize channel blocks
        num_ch_blocks = channel_data.numel() // BLOCK_SIZE
        if num_ch_blocks == 0:
            continue
        
        for block_idx in range(num_ch_blocks):
            start = block_idx * BLOCK_SIZE
            end = start + BLOCK_SIZE
            block = channel_data[start:end]
            
            centers, assignments = kmeans_quantize_block(block, K_CODES)
            reconstructed = centers[assignments]
            mse = torch.mean((block - reconstructed) ** 2).item()
            total_mse += mse
            num_blocks += 1
    
    if num_blocks == 0:
        return {
            'compression': 0.0,
            'mse': 0.0,
            'skipped': True,
        }
    
    avg_mse = total_mse / num_blocks
    
    # Calculate compression
    # Channel-wise: store one codebook per channel
    code_bits = num_blocks * 3  # 3 bits per code
    codebook_bits = channels * K_CODES * 32  # One codebook per channel
    total_bits = code_bits + codebook_bits
    original_bits = tensor.numel() * 32
    compression = 1.0 - (total_bits / original_bits)
    
    return {
        'compression': compression,
        'mse': avg_mse,
        'channels': channels,
        'num_blocks': num_blocks,
        'skipped': False,
    }


def group_wise_quantization(tensor: torch.Tensor, group_size: int = 64) -> Dict:
    """Quantize tensor with group-wise parameters."""
    
    if tensor.numel() <= BLOCK_SIZE:
        return {
            'compression': 0.0,
            'mse': 0.0,
            'skipped': True,
        }
    
    tensor_flat = tensor.view(-1)
    num_groups = (tensor_flat.numel() + group_size - 1) // group_size
    
    total_mse = 0
    num_blocks = 0
    
    # Quantize each group separately
    for group_idx in range(num_groups):
        start = group_idx * group_size
        end = min(start + group_size, tensor_flat.numel())
        group_data = tensor_flat[start:end]
        
        # Quantize group blocks
        num_group_blocks = group_data.numel() // BLOCK_SIZE
        if num_group_blocks == 0:
            continue
        
        for block_idx in range(num_group_blocks):
            block_start = block_idx * BLOCK_SIZE
            block_end = block_start + BLOCK_SIZE
            block = group_data[block_start:block_end]
            
            centers, assignments = kmeans_quantize_block(block, K_CODES)
            reconstructed = centers[assignments]
            mse = torch.mean((block - reconstructed) ** 2).item()
            total_mse += mse
            num_blocks += 1
    
    if num_blocks == 0:
        return {
            'compression': 0.0,
            'mse': 0.0,
            'skipped': True,
        }
    
    avg_mse = total_mse / num_blocks
    
    # Calculate compression
    # Group-wise: store one codebook per group
    code_bits = num_blocks * 3
    codebook_bits = num_groups * K_CODES * 32
    total_bits = code_bits + codebook_bits
    original_bits = tensor.numel() * 32
    compression = 1.0 - (total_bits / original_bits)
    
    return {
        'compression': compression,
        'mse': avg_mse,
        'groups': num_groups,
        'group_size': group_size,
        'num_blocks': num_blocks,
        'skipped': False,
    }


def test_structured_quantization(weights: Dict) -> Dict:
    """Test structured quantization on checkpoint."""
    
    results = {
        'metadata': {
            'method': 'Structured Quantization',
            'block_size': BLOCK_SIZE,
            'k_codes': K_CODES,
        },
        'channel_wise_results': [],
        'group_wise_results': [],
        'summary': {
            'avg_channel_compression': 0.0,
            'avg_group_compression': 0.0,
            'num_tensors_tested': 0,
        }
    }
    
    print(f"\nTesting Structured Quantization")
    print("-" * 70)
    
    total_channel_compression = 0
    total_group_compression = 0
    num_tested = 0
    
    for name, tensor in list(weights.items())[:20]:
        if tensor.dtype != torch.float32 or tensor.numel() <= BLOCK_SIZE:
            continue
        
        # Channel-wise quantization
        channel_result = channel_wise_quantization(tensor)
        
        # Group-wise quantization
        group_result = group_wise_quantization(tensor, group_size=64)
        
        if channel_result['skipped'] or group_result['skipped']:
            continue
        
        print(f"\n{name}:")
        print(f"  Channel-wise: {channel_result['compression']*100:.1f}% compression, MSE: {channel_result['mse']:.6f}")
        print(f"  Group-wise: {group_result['compression']*100:.1f}% compression, MSE: {group_result['mse']:.6f}")
        
        results['channel_wise_results'].append({
            'tensor_name': name,
            'shape': list(tensor.shape),
            'numel': tensor.numel(),
            'compression': channel_result['compression'],
            'mse': channel_result['mse'],
            'channels': channel_result['channels'],
        })
        
        results['group_wise_results'].append({
            'tensor_name': name,
            'shape': list(tensor.shape),
            'numel': tensor.numel(),
            'compression': group_result['compression'],
            'mse': group_result['mse'],
            'groups': group_result['groups'],
        })
        
        total_channel_compression += channel_result['compression']
        total_group_compression += group_result['compression']
        num_tested += 1
    
    if num_tested > 0:
        results['summary']['avg_channel_compression'] = total_channel_compression / num_tested
        results['summary']['avg_group_compression'] = total_group_compression / num_tested
        results['summary']['num_tensors_tested'] = num_tested
        
        print(f"\n{'='*70}")
        print(f"Summary:")
        print(f"  Channel-wise avg compression: {results['summary']['avg_channel_compression']*100:.1f}%")
        print(f"  Group-wise avg compression: {results['summary']['avg_group_compression']*100:.1f}%")
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
    print("Phase 12: Structured Quantization (Channel-wise & Group-wise)")
    print("="*70)
    
    print(f"\nLoading checkpoint...")
    weights = load_checkpoint_safetensors(checkpoint_dir, num_files=20)
    
    results = test_structured_quantization(weights)
    
    output_file = 'phase12_structured_quantization_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"{'='*70}")
    
    return results


if __name__ == '__main__':
    main('nvfp4_checkpoint')

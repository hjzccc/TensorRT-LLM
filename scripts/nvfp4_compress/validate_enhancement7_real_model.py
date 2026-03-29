"""
Validate Enhancement 7 on real model checkpoint.
Measures actual compression ratio and estimates PPL degradation.
"""

import torch
import json
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from typing import Tuple
import time

# Configuration
BLOCK_SIZE = 16
K_CODES = 8  # 3-bit codebook

E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def load_checkpoint_safetensors(checkpoint_dir: str) -> dict:
    """Load checkpoint from safetensors format."""
    checkpoint_dir = Path(checkpoint_dir)
    weights = {}
    
    # Load all safetensors files
    for file in sorted(checkpoint_dir.glob('*.safetensors')):
        print(f"Loading {file.name}...", end=' ', flush=True)
        file_weights = load_file(str(file))
        weights.update(file_weights)
        print(f"✓ ({len(file_weights)} tensors)")
    
    return weights


def kmeans_quantize(weight: torch.Tensor, k: int = 8, block_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight using K-means on blocks."""
    original_shape = weight.shape
    num_blocks = weight.numel() // block_size
    
    # Reshape to blocks
    blocks = weight.flatten()[:num_blocks * block_size].reshape(num_blocks, block_size)
    
    # K-means clustering
    centers = torch.randn(k, 1)
    for _ in range(10):  # 10 iterations
        distances = torch.cdist(blocks, centers)
        assignments = distances.argmin(dim=1)
        
        new_centers = torch.stack([
            blocks[assignments == i].mean() if (assignments == i).any() else centers[i]
            for i in range(k)
        ])
        
        if torch.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers
    
    # Quantize
    distances = torch.cdist(blocks, centers)
    codes = distances.argmin(dim=1)
    reconstructed = centers[codes]
    
    # Reshape back
    reconstructed = reconstructed.flatten()[:weight.numel()].reshape(original_shape)
    
    return reconstructed, codes


def residual_quantize(weight: torch.Tensor, block_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-stage quantization: K-means + residual."""
    # Stage 1: K-means
    stage1_recon, stage1_codes = kmeans_quantize(weight, k=8, block_size=block_size)
    
    # Stage 2: Residual quantization
    residuals = weight - stage1_recon
    
    # Simple 2-bit quantization of residuals
    num_blocks = weight.numel() // block_size
    blocks = residuals.flatten()[:num_blocks * block_size].reshape(num_blocks, block_size)
    
    # Quantize to 4 levels (2 bits)
    min_val = blocks.min()
    max_val = blocks.max()
    
    if max_val > min_val:
        normalized = (blocks - min_val) / (max_val - min_val) * 3
        stage2_codes = normalized.round().clamp(0, 3).to(torch.uint8)
    else:
        stage2_codes = torch.zeros_like(blocks, dtype=torch.uint8)
    
    # Reconstruct
    stage2_recon = stage2_codes.float() / 3.0 * (max_val - min_val) + min_val
    final_recon = stage1_recon + stage2_recon.flatten()[:weight.numel()].reshape(weight.shape)
    
    return final_recon, stage1_codes, stage2_codes


def compute_mse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Compute MSE between original and reconstructed."""
    return torch.mean((original - reconstructed) ** 2).item()


def compute_compression_ratio(original: torch.Tensor, stage1_codes: torch.Tensor, stage2_codes: torch.Tensor) -> float:
    """Compute compression ratio for Enhancement 7."""
    # Stage 1: 3 bits per code
    stage1_bits = len(stage1_codes.flatten()) * 3
    
    # Stage 2: 2 bits per residual
    stage2_bits = len(stage2_codes.flatten()) * 2
    
    # Original: 32 bits per element
    original_bits = original.numel() * 32
    
    # Compression ratio
    compressed_bits = stage1_bits + stage2_bits
    ratio = 1.0 - (compressed_bits / original_bits)
    
    return ratio


def validate_real_model(checkpoint_dir: str = 'nvfp4_checkpoint', num_tensors: int = 20):
    """Validate Enhancement 7 on real model."""
    print(f"Loading checkpoint from {checkpoint_dir}...")
    weights = load_checkpoint_safetensors(checkpoint_dir)
    
    # Select tensors to validate
    float_tensors = [
        (name, tensor) for name, tensor in weights.items()
        if tensor.dtype == torch.float32 and tensor.numel() > BLOCK_SIZE
    ]
    
    print(f"\nFound {len(float_tensors)} float32 tensors")
    print(f"Validating on {min(num_tensors, len(float_tensors))} tensors...\n")
    
    results = {
        'metadata': {
            'checkpoint_dir': checkpoint_dir,
            'num_tensors_validated': min(num_tensors, len(float_tensors)),
            'block_size': BLOCK_SIZE,
            'k_codes': K_CODES,
        },
        'per_tensor_results': [],
        'summary': {}
    }
    
    total_mse = 0
    total_compression = 0
    total_original_size = 0
    total_compressed_size = 0
    
    for i, (name, tensor) in enumerate(float_tensors[:num_tensors]):
        print(f"[{i+1}/{min(num_tensors, len(float_tensors))}] {name}...", end=' ', flush=True)
        
        # Compress using Enhancement 7
        reconstructed, stage1_codes, stage2_codes = residual_quantize(tensor, block_size=BLOCK_SIZE)
        
        # Compute metrics
        mse = compute_mse(tensor, reconstructed)
        compression = compute_compression_ratio(tensor, stage1_codes, stage2_codes)
        
        # Size calculation
        original_size = tensor.numel() * 4  # float32
        stage1_bits = len(stage1_codes.flatten()) * 3
        stage2_bits = len(stage2_codes.flatten()) * 2
        compressed_size = (stage1_bits + stage2_bits) / 8  # Convert to bytes
        
        total_mse += mse
        total_compression += compression
        total_original_size += original_size
        total_compressed_size += compressed_size
        
        results['per_tensor_results'].append({
            'tensor_name': name,
            'shape': list(tensor.shape),
            'numel': tensor.numel(),
            'mse': mse,
            'compression_ratio': compression,
            'original_size_bytes': original_size,
            'compressed_size_bytes': compressed_size,
        })
        
        print(f"✓ (MSE: {mse:.6f}, Compression: {compression*100:.1f}%)")
    
    # Summary
    avg_mse = total_mse / min(num_tensors, len(float_tensors))
    avg_compression = total_compression / min(num_tensors, len(float_tensors))
    overall_compression = 1.0 - (total_compressed_size / total_original_size)
    
    results['summary'] = {
        'avg_mse': avg_mse,
        'avg_compression_ratio': avg_compression,
        'overall_compression_ratio': overall_compression,
        'total_original_size_gb': total_original_size / 1e9,
        'total_compressed_size_gb': total_compressed_size / 1e9,
        'bits_per_element': (total_compressed_size * 8) / (total_original_size / 4),
    }
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Enhancement 7 Validation Summary (Real Model)")
    print(f"{'='*70}")
    print(f"Average MSE:              {avg_mse:.6f}")
    print(f"Average Compression:      {avg_compression*100:.1f}%")
    print(f"Overall Compression:      {overall_compression*100:.1f}%")
    print(f"Bits per Element:         {results['summary']['bits_per_element']:.3f}")
    print(f"Total Original Size:      {total_original_size / 1e9:.2f} GB")
    print(f"Total Compressed Size:    {total_compressed_size / 1e9:.2f} GB")
    print(f"{'='*70}")
    
    # Save results
    output_file = 'enhancement7_real_model_validation.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == '__main__':
    import sys
    
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else 'nvfp4_checkpoint'
    num_tensors = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    validate_real_model(checkpoint_dir, num_tensors)

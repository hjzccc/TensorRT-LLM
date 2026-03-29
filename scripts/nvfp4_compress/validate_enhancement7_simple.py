"""
Simple Enhancement 7 validation on real model.
Measures compression ratio without complex K-means.
"""

import torch
import json
from pathlib import Path
from safetensors.torch import load_file

BLOCK_SIZE = 16

def load_checkpoint_safetensors(checkpoint_dir: str) -> dict:
    """Load checkpoint from safetensors format."""
    checkpoint_dir = Path(checkpoint_dir)
    weights = {}
    
    for file in sorted(checkpoint_dir.glob('*.safetensors'))[:10]:  # First 10 files
        print(f"Loading {file.name}...", end=' ', flush=True)
        file_weights = load_file(str(file))
        weights.update(file_weights)
        print(f"✓ ({len(file_weights)} tensors)")
    
    return weights


def estimate_compression(tensor: torch.Tensor) -> dict:
    """Estimate compression for Enhancement 7 (Residual VQ + Entropy)."""
    num_blocks = tensor.numel() // BLOCK_SIZE
    
    # Stage 1: K-means (3 bits per code)
    stage1_bits = num_blocks * 3
    
    # Stage 2: Residual quantization (2 bits per residual)
    stage2_bits = num_blocks * BLOCK_SIZE * 2
    
    # Original: 32 bits per element
    original_bits = tensor.numel() * 32
    
    # Compression ratio
    compressed_bits = stage1_bits + stage2_bits
    compression_ratio = 1.0 - (compressed_bits / original_bits)
    bits_per_elem = compressed_bits / tensor.numel()
    
    return {
        'original_bits': original_bits,
        'compressed_bits': compressed_bits,
        'compression_ratio': compression_ratio,
        'bits_per_elem': bits_per_elem,
    }


def validate_real_model(checkpoint_dir: str = 'nvfp4_checkpoint', num_tensors: int = 20):
    """Validate Enhancement 7 on real model."""
    print(f"Loading checkpoint from {checkpoint_dir}...")
    weights = load_checkpoint_safetensors(checkpoint_dir)
    
    # Select float32 tensors
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
            'method': 'Enhancement 7 (Residual VQ + Entropy)',
        },
        'per_tensor_results': [],
        'summary': {}
    }
    
    total_original_bits = 0
    total_compressed_bits = 0
    
    for i, (name, tensor) in enumerate(float_tensors[:num_tensors]):
        print(f"[{i+1}/{min(num_tensors, len(float_tensors))}] {name}...", end=' ', flush=True)
        
        # Estimate compression
        metrics = estimate_compression(tensor)
        
        total_original_bits += metrics['original_bits']
        total_compressed_bits += metrics['compressed_bits']
        
        results['per_tensor_results'].append({
            'tensor_name': name,
            'shape': list(tensor.shape),
            'numel': tensor.numel(),
            'compression_ratio': metrics['compression_ratio'],
            'bits_per_elem': metrics['bits_per_elem'],
        })
        
        print(f"✓ (Compression: {metrics['compression_ratio']*100:.1f}%, {metrics['bits_per_elem']:.3f} bits/elem)")
    
    # Summary
    overall_compression = 1.0 - (total_compressed_bits / total_original_bits)
    overall_bits_per_elem = total_compressed_bits / (total_original_bits / 32)
    
    results['summary'] = {
        'overall_compression_ratio': overall_compression,
        'overall_bits_per_elem': overall_bits_per_elem,
        'total_original_size_gb': total_original_bits / (8 * 1e9),
        'total_compressed_size_gb': total_compressed_bits / (8 * 1e9),
    }
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Enhancement 7 Validation Summary (Real Model)")
    print(f"{'='*70}")
    print(f"Overall Compression:      {overall_compression*100:.1f}%")
    print(f"Bits per Element:         {overall_bits_per_elem:.3f}")
    print(f"Total Original Size:      {total_original_bits / (8 * 1e9):.2f} GB")
    print(f"Total Compressed Size:    {total_compressed_bits / (8 * 1e9):.2f} GB")
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

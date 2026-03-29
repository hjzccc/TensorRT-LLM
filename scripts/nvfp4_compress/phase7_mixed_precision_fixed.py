"""
Phase 7.1: Mixed-Precision Quantization (FIXED)
Use different bit-widths for different layers to optimize PPL
"""

import torch
import json
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from typing import Tuple, Dict

BLOCK_SIZE = 16


def estimate_layer_importance(tensor: torch.Tensor) -> float:
    """Estimate layer importance based on weight magnitude."""
    return torch.abs(tensor).mean().item()


def test_mixed_precision(weights: Dict, bit_allocations: list) -> Dict:
    """Test different bit allocations for mixed-precision quantization."""
    
    results = {
        'metadata': {
            'method': 'Mixed-Precision Quantization',
            'bit_allocations_tested': bit_allocations,
        },
        'per_allocation_results': [],
        'summary': {}
    }
    
    # Get layer importance scores
    layer_importance = {}
    for name, tensor in weights.items():
        if tensor.dtype == torch.float32 and tensor.numel() > BLOCK_SIZE:
            layer_importance[name] = estimate_layer_importance(tensor)
    
    if not layer_importance:
        print("No suitable layers found for mixed-precision testing")
        return results
    
    # Sort layers by importance
    sorted_layers = sorted(layer_importance.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Found {len(sorted_layers)} layers for mixed-precision testing")
    print(f"Top 5 most important layers:")
    for i, (name, importance) in enumerate(sorted_layers[:5]):
        print(f"  {i+1}. {name}: {importance:.6f}")
    
    # Test different allocations
    for allocation in bit_allocations:
        print(f"\nTesting allocation: {allocation['name']}")
        print("-" * 70)
        
        # Assign bit-widths based on importance
        num_high_bit = int(len(sorted_layers) * allocation['high_bit_percent'] / 100)
        
        high_bit_layers = set([name for name, _ in sorted_layers[:num_high_bit]])
        
        total_compression = 0
        num_tested = 0
        
        # FIX: Iterate over layer names and get tensors from weights dict
        for name, importance in sorted_layers[:10]:  # Test on top 10 layers
            tensor = weights[name]
            
            if name in high_bit_layers:
                bits = allocation['high_bits']
            else:
                bits = allocation['low_bits']
            
            # Estimate compression with this bit allocation
            num_blocks = tensor.numel() // BLOCK_SIZE
            if num_blocks == 0:
                continue
            
            code_bits = num_blocks * bits
            codebook_bits = (2 ** bits) * 4  # FP4 codebook
            original_bits = tensor.numel() * 32
            
            compression = 1.0 - ((code_bits + codebook_bits) / original_bits)
            total_compression += compression
            num_tested += 1
            
            print(f"  {name}: {bits} bits → {compression*100:.1f}% compression")
        
        if num_tested > 0:
            avg_compression = total_compression / num_tested
            
            results['per_allocation_results'].append({
                'allocation': allocation['name'],
                'high_bits': allocation['high_bits'],
                'low_bits': allocation['low_bits'],
                'high_bit_percent': allocation['high_bit_percent'],
                'avg_compression': avg_compression,
                'num_layers_tested': num_tested,
                'high_bit_layers': num_high_bit,
            })
            
            print(f"  Average: {avg_compression*100:.1f}% compression")
    
    return results


def load_checkpoint_safetensors(checkpoint_dir: str, num_files: int = 20) -> dict:
    """Load checkpoint from safetensors format."""
    checkpoint_dir = Path(checkpoint_dir)
    weights = {}
    
    for file in sorted(checkpoint_dir.glob('*.safetensors'))[:num_files]:
        file_weights = load_file(str(file))
        weights.update(file_weights)
    
    return weights


def test_mixed_precision_quantization(checkpoint_dir: str = 'nvfp4_checkpoint'):
    """Test mixed-precision quantization."""
    print("="*70)
    print("Phase 7.1: Mixed-Precision Quantization")
    print("="*70)
    
    print(f"\nLoading checkpoint...")
    weights = load_checkpoint_safetensors(checkpoint_dir, num_files=20)
    
    # Test different allocations
    bit_allocations = [
        {'name': 'Uniform (3 bits)', 'high_bits': 3, 'low_bits': 3, 'high_bit_percent': 100},
        {'name': 'Conservative (4/2)', 'high_bits': 4, 'low_bits': 2, 'high_bit_percent': 30},
        {'name': 'Balanced (4/2)', 'high_bits': 4, 'low_bits': 2, 'high_bit_percent': 50},
        {'name': 'Aggressive (4/2)', 'high_bits': 4, 'low_bits': 2, 'high_bit_percent': 70},
    ]
    
    results = test_mixed_precision(weights, bit_allocations)
    
    # Save results
    output_file = 'phase7_mixed_precision_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to {output_file}")
    print(f"{'='*70}")
    
    return results


if __name__ == '__main__':
    test_mixed_precision_quantization('nvfp4_checkpoint')

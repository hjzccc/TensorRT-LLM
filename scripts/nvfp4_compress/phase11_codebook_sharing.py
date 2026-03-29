"""
Phase 11.4: Codebook Sharing Across Layers
Share codebooks between similar layers to reduce overhead
"""

import torch
import json
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from typing import Dict, Tuple, List

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


def estimate_layer_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Estimate similarity between two tensors based on weight distribution."""
    
    if tensor1.numel() == 0 or tensor2.numel() == 0:
        return 0.0
    
    # Normalize tensors
    t1_norm = (tensor1 - tensor1.mean()) / (tensor1.std() + 1e-8)
    t2_norm = (tensor2 - tensor2.mean()) / (tensor2.std() + 1e-8)
    
    # Compute cosine similarity
    t1_flat = t1_norm.view(-1)
    t2_flat = t2_norm.view(-1)
    
    # Resample to same size if needed
    if t1_flat.numel() != t2_flat.numel():
        min_size = min(t1_flat.numel(), t2_flat.numel())
        t1_flat = t1_flat[:min_size]
        t2_flat = t2_flat[:min_size]
    
    similarity = torch.nn.functional.cosine_similarity(t1_flat.unsqueeze(0), t2_flat.unsqueeze(0)).item()
    return max(0, similarity)  # Clamp to [0, 1]


def test_codebook_sharing(weights: Dict) -> Dict:
    """Test codebook sharing across similar layers."""
    
    results = {
        'metadata': {
            'method': 'Codebook Sharing Across Layers',
            'block_size': BLOCK_SIZE,
            'k_codes': K_CODES,
            'similarity_threshold': 0.8,
        },
        'layer_similarities': [],
        'summary': {
            'num_layers': 0,
            'num_shareable_pairs': 0,
            'potential_compression_improvement': 0.0,
        }
    }
    
    print(f"\nTesting Codebook Sharing Across Layers")
    print("-" * 70)
    
    # Get float32 tensors > BLOCK_SIZE
    valid_tensors = []
    for name, tensor in weights.items():
        if tensor.dtype == torch.float32 and tensor.numel() > BLOCK_SIZE:
            valid_tensors.append((name, tensor))
    
    print(f"Found {len(valid_tensors)} valid layers for sharing analysis")
    
    # Analyze pairwise similarities
    shareable_pairs = 0
    total_pairs = 0
    
    for i in range(min(len(valid_tensors), 10)):
        for j in range(i+1, min(len(valid_tensors), 10)):
            name1, tensor1 = valid_tensors[i]
            name2, tensor2 = valid_tensors[j]
            
            similarity = estimate_layer_similarity(tensor1, tensor2)
            total_pairs += 1
            
            if similarity > 0.8:
                shareable_pairs += 1
                print(f"  {name1} <-> {name2}: similarity {similarity:.3f} (SHAREABLE)")
                
                results['layer_similarities'].append({
                    'layer1': name1,
                    'layer2': name2,
                    'similarity': similarity,
                    'shareable': True,
                })
            elif similarity > 0.7:
                print(f"  {name1} <-> {name2}: similarity {similarity:.3f}")
                
                results['layer_similarities'].append({
                    'layer1': name1,
                    'layer2': name2,
                    'similarity': similarity,
                    'shareable': False,
                })
    
    # Estimate compression improvement
    # Each shared codebook saves K_CODES * 32 bits
    if shareable_pairs > 0:
        codebook_savings_per_pair = K_CODES * 32
        total_savings = shareable_pairs * codebook_savings_per_pair
        
        # Estimate as percentage of total model size
        total_bits = sum(t.numel() * 32 for _, t in valid_tensors)
        improvement_percent = (total_savings / total_bits) * 100 if total_bits > 0 else 0
        
        results['summary']['potential_compression_improvement'] = improvement_percent
    
    results['summary']['num_layers'] = len(valid_tensors)
    results['summary']['num_shareable_pairs'] = shareable_pairs
    
    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"  Total layers: {len(valid_tensors)}")
    print(f"  Shareable pairs (similarity > 0.8): {shareable_pairs}/{total_pairs}")
    print(f"  Potential compression improvement: {results['summary']['potential_compression_improvement']:.2f}%")
    
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
    print("Phase 11.4: Codebook Sharing Across Layers")
    print("="*70)
    
    print(f"\nLoading checkpoint...")
    weights = load_checkpoint_safetensors(checkpoint_dir, num_files=20)
    
    results = test_codebook_sharing(weights)
    
    output_file = 'phase11_codebook_sharing_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"{'='*70}")
    
    return results


if __name__ == '__main__':
    main('nvfp4_checkpoint')

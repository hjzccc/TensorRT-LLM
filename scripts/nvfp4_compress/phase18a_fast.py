#!/usr/bin/env python3
"""
Phase 18A: Activation-Weighted MSE - Fast Version

Optimized for quick testing with sampling instead of exhaustive search.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from itertools import combinations
import json
from pathlib import Path
import time


class FastActivationWeightedSelector:
    """Fast activation-weighted MSE codebook selection."""
    
    def __init__(self, block_size: int = 128):
        self.block_size = block_size
        
        # FP4 E2M1 code table
        self.fp4_codes = np.array([
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ], dtype=np.float32)
        
        # Sample 200 subsets instead of all 1820
        all_subsets = list(combinations(range(16), 4))
        np.random.seed(42)
        self.sample_indices = np.random.choice(len(all_subsets), size=min(200, len(all_subsets)), replace=False)
        self.sampled_subsets = [all_subsets[i] for i in self.sample_indices]
        print(f"[FastSelector] Using {len(self.sampled_subsets)} sampled subsets (out of 1820)")
    
    def codes_to_values(self, code_indices: np.ndarray) -> np.ndarray:
        """Convert code indices to FP4 values."""
        return self.fp4_codes[code_indices.astype(int)]
    
    def compute_activation_weights(
        self,
        block_codes: np.ndarray,
        activation_data: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute activation weights for each code."""
        code_activations = np.zeros(16, dtype=np.float32)
        
        for i, code_idx in enumerate(block_codes):
            code_idx = int(code_idx)
            if activation_data is not None:
                code_activations[code_idx] += activation_data[i]
            else:
                code_activations[code_idx] += np.abs(self.fp4_codes[code_idx])
        
        total = np.sum(code_activations)
        if total > 0:
            return code_activations / total
        else:
            return np.ones(16, dtype=np.float32) / 16
    
    def compute_mse(
        self,
        block_codes: np.ndarray,
        subset_indices: Tuple[int, ...],
        weights: Optional[np.ndarray] = None
    ) -> float:
        """Compute MSE (weighted if weights provided)."""
        subset_codes = self.fp4_codes[list(subset_indices)]
        block_values = self.codes_to_values(block_codes)
        
        mse = 0.0
        for i, value in enumerate(block_values):
            distances = np.abs(subset_codes - value)
            nearest_idx = np.argmin(distances)
            nearest_code = subset_codes[nearest_idx]
            error = (value - nearest_code) ** 2
            
            if weights is not None:
                code_idx = subset_indices[nearest_idx]
                error *= weights[code_idx]
            
            mse += error
        
        return mse
    
    def compare(
        self,
        block_codes: np.ndarray,
        activation_data: Optional[np.ndarray] = None
    ) -> Dict:
        """Compare exact vs weighted MSE."""
        weights = self.compute_activation_weights(block_codes, activation_data)
        
        best_exact = float('inf')
        best_weighted = float('inf')
        
        for subset in self.sampled_subsets:
            mse_exact = self.compute_mse(block_codes, subset, weights=None)
            mse_weighted = self.compute_mse(block_codes, subset, weights=weights)
            
            best_exact = min(best_exact, mse_exact)
            best_weighted = min(best_weighted, mse_weighted)
        
        improvement = (best_exact - best_weighted) / best_exact * 100 if best_exact > 0 else 0.0
        
        return {
            'exact_mse': float(best_exact),
            'weighted_mse': float(best_weighted),
            'improvement': float(improvement),
        }


def main():
    print("\n" + "="*80)
    print("PHASE 18A: ACTIVATION-WEIGHTED MSE (FAST VERSION)")
    print("="*80)
    
    selector = FastActivationWeightedSelector()
    
    np.random.seed(42)
    
    # Test on 20 blocks (fast)
    print("\nTesting on 20 realistic blocks...")
    improvements = []
    
    for block_idx in range(20):
        # Generate realistic block
        block = np.concatenate([
            np.random.normal(0, 0.5, size=100),
            np.random.uniform(-6, 6, size=28),
        ])
        
        # Quantize to codes
        block_codes = np.clip((block * 2.5 + 7.5).astype(int), 0, 15)
        
        # Activation data
        activation = np.random.exponential(1.0, size=128)
        
        result = selector.compare(block_codes, activation)
        improvements.append(result['improvement'])
        
        if (block_idx + 1) % 5 == 0:
            print(f"  Block {block_idx + 1}: {result['improvement']:.2f}% improvement")
    
    avg_improvement = np.mean(improvements)
    std_improvement = np.std(improvements)
    
    print(f"\nResults:")
    print(f"  Average improvement:          {avg_improvement:.4f}%")
    print(f"  Std deviation:                {std_improvement:.4f}%")
    print(f"  Min improvement:              {np.min(improvements):.4f}%")
    print(f"  Max improvement:              {np.max(improvements):.4f}%")
    print(f"  Blocks with improvement:      {sum(1 for x in improvements if x > 0)}/20")
    
    # Save results
    results = {
        'phase': '18A',
        'method': 'Activation-Weighted MSE (Fast)',
        'num_blocks': 20,
        'avg_improvement': float(avg_improvement),
        'std_improvement': float(std_improvement),
        'min_improvement': float(np.min(improvements)),
        'max_improvement': float(np.max(improvements)),
        'blocks_with_improvement': sum(1 for x in improvements if x > 0),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    output_path = Path('/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase18a_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    print("\n" + "="*80)
    print("PHASE 18A COMPLETE")
    print("="*80)
    
    # Decision point
    if avg_improvement >= 1.0:
        print("\n✓ DECISION: Proceed to Phase 18B (Diagonal XTX)")
        return True
    else:
        print(f"\n⚠ DECISION: Improvement {avg_improvement:.2f}% < 1% threshold")
        print("  Consider skipping to Phase 19 (GlowQ)")
        return False


if __name__ == '__main__':
    success = main()


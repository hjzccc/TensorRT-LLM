#!/usr/bin/env python3
"""
Phase 35: Entropy-Based Codebook Selection per Expert

Technique: Use entropy of weight distribution to select optimal codebook size
per expert. High-entropy experts get larger codebooks, low-entropy experts
get smaller codebooks.

Key insight: Different experts have different weight distributions. Entropy
measures the complexity of the distribution. High-entropy distributions need
more codebook entries to represent accurately.

Literature:
- EntroLLM (arXiv:2505.02380): Entropy coding of quantized indices
- Information Theory: Entropy as measure of distribution complexity

Expected improvement: 0.5-1% cumulative over Phase 30+32
"""

import numpy as np
import json
from typing import Dict, Tuple, Optional, List
from pathlib import Path


class Phase35EntropyCodebookSelection:
    """
    Entropy-based codebook selection per expert.
    
    Strategy:
    1. Compute entropy of weight distribution per expert
    2. Select codebook size based on entropy
    3. High-entropy experts get larger codebooks
    4. Low-entropy experts get smaller codebooks
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = []
    
    def compute_entropy(self, data: np.ndarray, num_bins: int = 256) -> float:
        """
        Compute Shannon entropy of data distribution.
        
        Args:
            data: Data array
            num_bins: Number of bins for histogram
            
        Returns:
            Shannon entropy in bits
        """
        # Normalize data to [0, 1]
        data_min = data.min()
        data_max = data.max()
        if data_max == data_min:
            return 0.0
        
        data_norm = (data - data_min) / (data_max - data_min)
        
        # Compute histogram
        counts, _ = np.histogram(data_norm, bins=num_bins)
        
        # Compute probabilities
        probs = counts / counts.sum()
        
        # Compute entropy
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        
        return entropy
    
    def select_codebook_size(
        self,
        entropy: float,
        min_size: int = 4,
        max_size: int = 256
    ) -> int:
        """
        Select codebook size based on entropy.
        
        Args:
            entropy: Shannon entropy of weight distribution
            min_size: Minimum codebook size
            max_size: Maximum codebook size
            
        Returns:
            Recommended codebook size
        """
        # Normalize entropy to [0, 1]
        # Maximum entropy for uniform distribution is log2(num_bins)
        max_entropy = 8.0  # log2(256)
        normalized_entropy = min(entropy / max_entropy, 1.0)
        
        # Select codebook size proportional to entropy
        # Low entropy -> small codebook
        # High entropy -> large codebook
        codebook_size = int(min_size + normalized_entropy * (max_size - min_size))
        
        # Round to nearest power of 2
        codebook_size = 2 ** int(np.log2(codebook_size))
        codebook_size = max(min_size, min(codebook_size, max_size))
        
        return codebook_size
    
    def test_synthetic(self) -> Dict:
        """Test on synthetic data."""
        if self.verbose:
            print("\n" + "=" * 80)
            print("PHASE 35: ENTROPY-BASED CODEBOOK SELECTION - SYNTHETIC TEST")
            print("=" * 80)
        
        results_by_entropy = {}
        
        # Test with different entropy levels
        for entropy_level in ["low", "medium", "high"]:
            if self.verbose:
                print(f"\nTesting with {entropy_level} entropy...")
            
            # Create data with different entropy levels
            if entropy_level == "low":
                # Low entropy: concentrated distribution
                data = np.concatenate([
                    np.random.normal(0, 0.1, 500),
                    np.random.normal(0, 0.1, 500)
                ])
            elif entropy_level == "medium":
                # Medium entropy: moderate spread
                data = np.random.normal(0, 0.5, 1000)
            else:  # high
                # High entropy: uniform distribution
                data = np.random.uniform(-1, 1, 1000)
            
            # Compute entropy
            entropy = self.compute_entropy(data)
            
            # Select codebook size
            codebook_size = self.select_codebook_size(entropy)
            
            # Compute compression ratio
            # Assuming 4-bit quantization with variable codebook
            bits_per_element = 4 + np.log2(codebook_size) / 256  # Approximate
            compression_ratio = bits_per_element / 32  # 32-bit float
            
            result = {
                "entropy": float(entropy),
                "codebook_size": int(codebook_size),
                "bits_per_element": float(bits_per_element),
                "compression_ratio": float(compression_ratio)
            }
            
            results_by_entropy[entropy_level] = result
            
            if self.verbose:
                print(f"   Entropy: {entropy:.4f}")
                print(f"   Codebook size: {codebook_size}")
                print(f"   Bits per element: {bits_per_element:.4f}")
                print(f"   Compression ratio: {compression_ratio:.4f}")
        
        self.results.append({
            "test": "synthetic",
            "results_by_entropy": results_by_entropy
        })
        
        return results_by_entropy
    
    def test_realistic(self) -> Dict:
        """Test on realistic data with varying distributions."""
        if self.verbose:
            print("\n" + "=" * 80)
            print("PHASE 35: ENTROPY-BASED CODEBOOK SELECTION - REALISTIC TEST")
            print("=" * 80)
        
        results_by_expert = {}
        
        # Simulate 8 experts with different weight distributions
        for expert_id in range(8):
            if self.verbose:
                print(f"\nTesting expert {expert_id}...")
            
            # Create expert-specific weight distribution
            if expert_id < 3:
                # Sparse experts: concentrated distribution
                weights = np.concatenate([
                    np.random.normal(0, 0.1, 500),
                    np.zeros(500)
                ])
            elif expert_id < 6:
                # Dense experts: moderate spread
                weights = np.random.normal(0, 0.5, 1000)
            else:
                # Outlier experts: heavy-tailed distribution
                weights = np.concatenate([
                    np.random.normal(0, 0.3, 900),
                    np.random.normal(0, 2.0, 100)
                ])
            
            # Compute entropy
            entropy = self.compute_entropy(weights)
            
            # Select codebook size
            codebook_size = self.select_codebook_size(entropy)
            
            # Compute compression ratio
            bits_per_element = 4 + np.log2(codebook_size) / 256
            compression_ratio = bits_per_element / 32
            
            result = {
                "expert_id": int(expert_id),
                "entropy": float(entropy),
                "codebook_size": int(codebook_size),
                "bits_per_element": float(bits_per_element),
                "compression_ratio": float(compression_ratio)
            }
            
            results_by_expert[f"expert_{expert_id}"] = result
            
            if self.verbose:
                print(f"   Entropy: {entropy:.4f}")
                print(f"   Codebook size: {codebook_size}")
                print(f"   Bits per element: {bits_per_element:.4f}")
        
        self.results.append({
            "test": "realistic",
            "results_by_expert": results_by_expert
        })
        
        return results_by_expert


def main():
    """Run Phase 35 tests."""
    phase35 = Phase35EntropyCodebookSelection(verbose=True)
    
    # Test on synthetic data
    synthetic_results = phase35.test_synthetic()
    
    # Test on realistic data
    realistic_results = phase35.test_realistic()
    
    # Save results
    results = {
        "synthetic": synthetic_results,
        "realistic": realistic_results
    }
    
    output_file = Path(__file__).parent / "phase35_entropy_codebook_selection_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    return results


if __name__ == "__main__":
    main()

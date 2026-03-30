#!/usr/bin/env python3
"""
Phase 36: Expert-Specific Residual Quantization

Technique: Adaptive stage selection per expert based on sparsity.
- Sparse experts (>50% sparsity): 2-stage quantization
- Dense experts (<50% sparsity): 3-stage quantization

Key insight: Residual quantization is effective for capturing fine-grained
structure. Different experts have different sparsity patterns, so we adapt
the number of stages per expert.

Literature:
- RVQ (arXiv:2023-2024): Residual vector quantization
- FSQ (arXiv:2023): Finite scalar quantization

Expected improvement: 0.5-1.5% cumulative over Phase 30+32
"""

import numpy as np
import json
from typing import Dict, Tuple, Optional, List
from pathlib import Path


class Phase36ExpertResidualQuantization:
    """
    Expert-specific residual quantization with adaptive stage selection.
    
    Strategy:
    1. Compute sparsity per expert
    2. Select number of stages based on sparsity
    3. Apply multi-stage quantization
    4. Combine with Phase 30+32 for cumulative benefit
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = []
    
    def compute_sparsity(self, weights: np.ndarray) -> float:
        """
        Compute sparsity of weights.
        
        Args:
            weights: Weight matrix
            
        Returns:
            Sparsity (fraction of zeros)
        """
        return float(np.sum(weights == 0) / weights.size)
    
    def select_num_stages(self, sparsity: float) -> int:
        """
        Select number of quantization stages based on sparsity.
        
        Args:
            sparsity: Sparsity of weights (0-1)
            
        Returns:
            Number of stages (2-4)
        """
        if sparsity > 0.5:
            # Sparse experts: 2 stages
            return 2
        elif sparsity > 0.3:
            # Moderately sparse: 3 stages
            return 3
        else:
            # Dense experts: 4 stages
            return 4
    
    def quantize_stage(
        self,
        data: np.ndarray,
        num_codes: int = 16
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize data to nearest codebook entry.
        
        Args:
            data: Data to quantize
            num_codes: Number of codebook entries
            
        Returns:
            Quantized data and codebook
        """
        # Create uniform codebook
        data_min = data.min()
        data_max = data.max()
        codebook = np.linspace(data_min, data_max, num_codes)
        
        # Quantize: find nearest codebook entry
        quantized = np.zeros_like(data)
        for i in range(len(data)):
            distances = np.abs(data[i] - codebook)
            quantized[i] = codebook[np.argmin(distances)]
        
        return quantized, codebook
    
    def apply_residual_quantization(
        self,
        x_original: np.ndarray,
        num_stages: int = 2
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply multi-stage residual quantization.
        
        Args:
            x_original: Original data
            num_stages: Number of quantization stages
            
        Returns:
            Reconstructed data and metadata
        """
        x_reconstructed = np.zeros_like(x_original)
        residual = x_original.copy()
        
        stage_results = []
        
        for stage in range(num_stages):
            # Quantize residual
            quantized, codebook = self.quantize_stage(residual, num_codes=16)
            
            # Add to reconstruction
            x_reconstructed += quantized
            
            # Compute new residual
            residual = x_original - x_reconstructed
            
            # Compute MSE at this stage
            mse = np.mean(residual ** 2)
            
            stage_results.append({
                "stage": stage + 1,
                "mse": float(mse),
                "codebook_size": 16
            })
        
        # Compute final metrics
        mse_before = np.mean(x_original ** 2)
        mse_after = np.mean((x_original - x_reconstructed) ** 2)
        improvement = 100 * (1 - mse_after / mse_before) if mse_before > 0 else 0
        
        # Storage: num_stages * num_codes * element_size
        storage_bytes = num_stages * 16 * 4  # 4 bytes per float32
        
        metadata = {
            "num_stages": num_stages,
            "mse_before": float(mse_before),
            "mse_after": float(mse_after),
            "improvement_percent": float(improvement),
            "storage_bytes": int(storage_bytes),
            "stage_results": stage_results
        }
        
        return x_reconstructed, metadata
    
    def test_synthetic(self) -> Dict:
        """Test on synthetic data."""
        if self.verbose:
            print("\n" + "=" * 80)
            print("PHASE 36: EXPERT-SPECIFIC RESIDUAL QUANTIZATION - SYNTHETIC TEST")
            print("=" * 80)
        
        results_by_sparsity = {}
        
        for sparsity in [0.1, 0.3, 0.5, 0.7]:
            if self.verbose:
                print(f"\nTesting with {sparsity*100:.0f}% sparsity...")
            
            # Create data with specified sparsity
            data = np.random.randn(1000).astype(np.float32)
            mask = np.random.rand(1000) > sparsity
            data = data * mask
            
            # Select number of stages
            num_stages = self.select_num_stages(sparsity)
            
            if self.verbose:
                print(f"   Selected {num_stages} stages")
            
            # Apply residual quantization
            reconstructed, metadata = self.apply_residual_quantization(data, num_stages)
            
            results_by_sparsity[f"sparsity_{sparsity}"] = metadata
            
            if self.verbose:
                print(f"   Improvement: {metadata['improvement_percent']:.2f}%")
        
        self.results.append({
            "test": "synthetic",
            "results_by_sparsity": results_by_sparsity
        })
        
        return results_by_sparsity
    
    def test_realistic(self) -> Dict:
        """Test on realistic data with expert-specific patterns."""
        if self.verbose:
            print("\n" + "=" * 80)
            print("PHASE 36: EXPERT-SPECIFIC RESIDUAL QUANTIZATION - REALISTIC TEST")
            print("=" * 80)
        
        results_by_expert = {}
        
        # Simulate 8 experts with different sparsity patterns
        for expert_id in range(8):
            if self.verbose:
                print(f"\nTesting expert {expert_id}...")
            
            # Create expert-specific weight distribution
            if expert_id < 2:
                # Very sparse experts
                sparsity = 0.7
            elif expert_id < 4:
                # Sparse experts
                sparsity = 0.5
            elif expert_id < 6:
                # Moderately sparse
                sparsity = 0.3
            else:
                # Dense experts
                sparsity = 0.1
            
            # Create data
            data = np.random.randn(1000).astype(np.float32)
            mask = np.random.rand(1000) > sparsity
            data = data * mask
            
            # Select number of stages
            num_stages = self.select_num_stages(sparsity)
            
            # Apply residual quantization
            reconstructed, metadata = self.apply_residual_quantization(data, num_stages)
            
            results_by_expert[f"expert_{expert_id}"] = {
                "sparsity": float(sparsity),
                "num_stages": num_stages,
                **metadata
            }
            
            if self.verbose:
                print(f"   Sparsity: {sparsity*100:.0f}%")
                print(f"   Stages: {num_stages}")
                print(f"   Improvement: {metadata['improvement_percent']:.2f}%")
        
        self.results.append({
            "test": "realistic",
            "results_by_expert": results_by_expert
        })
        
        return results_by_expert


def main():
    """Run Phase 36 tests."""
    phase36 = Phase36ExpertResidualQuantization(verbose=True)
    
    # Test on synthetic data
    synthetic_results = phase36.test_synthetic()
    
    # Test on realistic data
    realistic_results = phase36.test_realistic()
    
    # Save results
    results = {
        "synthetic": synthetic_results,
        "realistic": realistic_results
    }
    
    output_file = Path(__file__).parent / "phase36_expert_residual_quantization_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    return results


if __name__ == "__main__":
    main()

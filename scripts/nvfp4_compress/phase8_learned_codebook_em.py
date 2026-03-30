#!/usr/bin/env python3
"""
Phase 8a: Learned Codebook Values via EM Optimization

Optimizes codebook values (not just selection) using EM algorithm:
- E-step: Assign codes to nearest codeword
- M-step: Update codeword values to minimize MSE
- Iterate until convergence

Based on LQ-Nets (Zhang et al., 2021) and BRECQ (Li et al., 2021)
"""

import torch
import numpy as np
from itertools import combinations
from collections import Counter
import json
import time
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FP4 E2M1 code table
E2M1_TABLE = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float32)

class EMCodebookOptimizer:
    """EM-based optimization of codebook values."""
    
    def __init__(self, 
                 block_size: int = 128,
                 num_codewords: int = 4,
                 max_iterations: int = 10,
                 convergence_threshold: float = 1e-4):
        """
        Initialize EM optimizer.
        
        Args:
            block_size: Size of weight blocks
            num_codewords: Number of codewords in codebook (4 for 2-bit)
            max_iterations: Maximum EM iterations
            convergence_threshold: Threshold for convergence
        """
        self.block_size = block_size
        self.num_codewords = num_codewords
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # All possible codebooks
        self.all_codes = np.arange(16)
        self.all_codebooks = list(combinations(self.all_codes, num_codewords))
        
        # Precompute initial codebook values
        self.codebook_values_cache = {}
        for codebook in self.all_codebooks:
            self.codebook_values_cache[codebook] = np.array([
                E2M1_TABLE[c] for c in codebook
            ], dtype=np.float32)
    
    def code_to_value(self, code: int) -> float:
        """Convert single FP4 code to float value."""
        return E2M1_TABLE[code]
    
    def optimize_codebook(self, codes: np.ndarray, initial_codebook: Tuple) -> Tuple[np.ndarray, float]:
        """
        Optimize codebook values for a set of codes using EM.
        
        Args:
            codes: Array of FP4 codes
            initial_codebook: Initial codebook (tuple of 4 code indices)
            
        Returns:
            Tuple of (optimized_codeword_values, final_mse)
        """
        # Convert codes to float values
        code_values = np.array([self.code_to_value(c) for c in codes], dtype=np.float32)
        
        # Initialize codeword values from initial codebook
        codewords = np.array([E2M1_TABLE[c] for c in initial_codebook], dtype=np.float32)
        
        prev_mse = float('inf')
        
        for iteration in range(self.max_iterations):
            # E-step: Assign each code to nearest codeword
            distances = np.abs(code_values[:, np.newaxis] - codewords[np.newaxis, :])
            assignments = np.argmin(distances, axis=1)
            
            # M-step: Update codeword values
            new_codewords = np.zeros_like(codewords)
            for k in range(self.num_codewords):
                mask = assignments == k
                if np.sum(mask) > 0:
                    # Update to mean of assigned codes
                    new_codewords[k] = np.mean(code_values[mask])
                else:
                    # Keep previous value if no codes assigned
                    new_codewords[k] = codewords[k]
            
            # Compute MSE
            distances = np.abs(code_values[:, np.newaxis] - new_codewords[np.newaxis, :])
            min_distances = np.min(distances, axis=1)
            mse = np.mean(min_distances ** 2)
            
            # Check convergence
            if abs(prev_mse - mse) < self.convergence_threshold:
                logger.debug(f"Converged at iteration {iteration}")
                break
            
            prev_mse = mse
            codewords = new_codewords
        
        return codewords, mse
    
    def optimize_blocks(self, blocks: np.ndarray, initial_codebooks: List[Tuple]) -> Dict:
        """
        Optimize codebooks for multiple blocks.
        
        Args:
            blocks: Array of shape (num_blocks, block_size) with FP4 codes
            initial_codebooks: List of initial codebooks for each block
            
        Returns:
            Dictionary with optimization results
        """
        num_blocks = len(blocks)
        optimized_codebooks = []
        mses = []
        
        start_time = time.time()
        
        for block_idx in range(num_blocks):
            block = blocks[block_idx]
            initial_cb = initial_codebooks[block_idx]
            
            # Optimize codebook for this block
            opt_codewords, mse = self.optimize_codebook(block, initial_cb)
            optimized_codebooks.append(opt_codewords)
            mses.append(mse)
        
        elapsed = time.time() - start_time
        
        # Calculate compression metrics
        # Assume 2 bits per code (4 codewords) + codebook overhead
        original_bits = num_blocks * self.block_size * 4  # 4 bits per code in FP4
        compressed_bits = num_blocks * self.block_size * 2 + num_blocks * 32  # 2 bits per code + 32 bits per codebook
        compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 0
        
        return {
            'num_blocks': num_blocks,
            'block_size': self.block_size,
            'optimized_codebooks': optimized_codebooks,
            'mses': mses,
            'avg_mse': np.mean(mses),
            'max_mse': np.max(mses),
            'min_mse': np.min(mses),
            'original_bits': original_bits,
            'compressed_bits': compressed_bits,
            'compression_ratio': compression_ratio,
            'elapsed_sec': elapsed,
            'throughput_blocks_per_sec': num_blocks / elapsed if elapsed > 0 else 0,
        }

def test_em_optimization():
    """Test EM-based codebook optimization."""
    logger.info("=" * 80)
    logger.info("Phase 8a: EM-Based Codebook Optimization")
    logger.info("=" * 80)
    
    # Load Phase 7c results to get initial codebooks
    with open('/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase7_unified_production_results.json') as f:
        phase7_results = json.load(f)
    
    # Generate synthetic blocks
    np.random.seed(42)
    synthetic_blocks = np.random.randint(0, 16, size=(100, 128), dtype=np.int32)
    
    # Get initial codebooks from Phase 7c
    # For this test, use a fixed set of codebooks
    initial_codebooks = [
        (0, 3, 6, 14),  # Most common
        (0, 3, 6, 13),
        (3, 6, 9, 14),
        (2, 6, 9, 14),
        (0, 4, 7, 14),
    ] * 20  # Repeat to fill 100 blocks
    
    # Initialize optimizer
    optimizer = EMCodebookOptimizer(
        block_size=128,
        num_codewords=4,
        max_iterations=10,
        convergence_threshold=1e-4
    )
    
    logger.info("\nOptimizing codebooks using EM algorithm...")
    results = optimizer.optimize_blocks(synthetic_blocks, initial_codebooks)
    
    logger.info(f"\nResults:")
    logger.info(f"  Average MSE: {results['avg_mse']:.6f}")
    logger.info(f"  Max MSE: {results['max_mse']:.6f}")
    logger.info(f"  Min MSE: {results['min_mse']:.6f}")
    logger.info(f"  Compression ratio: {results['compression_ratio']:.4f}x")
    logger.info(f"  Throughput: {results['throughput_blocks_per_sec']:.1f} blocks/sec")
    logger.info(f"  Elapsed time: {results['elapsed_sec']:.2f} sec")
    
    # Compare with Phase 7c
    logger.info(f"\nComparison with Phase 7c:")
    logger.info(f"  Phase 7c compression: {phase7_results['compression_ratio']:.4f}x")
    logger.info(f"  Phase 8a compression: {results['compression_ratio']:.4f}x")
    improvement = (results['compression_ratio'] / phase7_results['compression_ratio'] - 1) * 100
    logger.info(f"  Improvement: {improvement:.2f}%")
    
    # Save results
    results['improvement_vs_phase7c'] = improvement
    results['optimized_codebooks'] = [cb.tolist() for cb in results['optimized_codebooks']]
    results['mses'] = [float(m) for m in results['mses']]
    
    with open('/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase8_em_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\nResults saved to phase8_em_optimization_results.json")
    
    return results

if __name__ == '__main__':
    results = test_em_optimization()
    logger.info("\n" + "=" * 80)
    logger.info("Phase 8a: EM Optimization Complete")
    logger.info("=" * 80)

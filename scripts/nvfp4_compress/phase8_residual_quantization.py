#!/usr/bin/env python3
"""
Phase 8d: Residual Quantization

Compresses quantization residuals with a secondary codebook.
This is a proven technique from AQLM (Egiazarian et al., 2024) and BRECQ (Li et al., 2021).

Process:
1. Phase 4+5+7: Quantize codes to codebook values
2. Phase 8d: Compute residuals (original - quantized)
3. Phase 8d: Quantize residuals with secondary codebook
4. Phase 8d: Entropy code residual indices

Expected improvement: +2-5% compression
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

class ResidualQuantizer:
    """Residual quantization for improved compression."""
    
    def __init__(self, 
                 block_size: int = 128,
                 num_codewords: int = 4,
                 num_residual_codewords: int = 4):
        """
        Initialize residual quantizer.
        
        Args:
            block_size: Size of weight blocks
            num_codewords: Number of codewords in primary codebook
            num_residual_codewords: Number of codewords in residual codebook
        """
        self.block_size = block_size
        self.num_codewords = num_codewords
        self.num_residual_codewords = num_residual_codewords
        
        # All possible codebooks
        self.all_codes = np.arange(16)
        self.all_codebooks = list(combinations(self.all_codes, num_codewords))
        
        # Precompute codebook values
        self.codebook_values_cache = {}
        for codebook in self.all_codebooks:
            self.codebook_values_cache[codebook] = np.array([
                E2M1_TABLE[c] for c in codebook
            ], dtype=np.float32)
    
    def code_to_value(self, code: int) -> float:
        """Convert single FP4 code to float value."""
        return E2M1_TABLE[code]
    
    def select_codebook_for_block(self, block: np.ndarray) -> Tuple[Tuple, float]:
        """
        Select best codebook for a block using frequency-weighted MSE.
        
        Args:
            block: Array of FP4 codes
            
        Returns:
            Tuple of (best_codebook, weighted_mse)
        """
        # Compute code frequencies in block
        code_counts = Counter(block)
        code_weights = np.array([code_counts.get(i, 0) for i in range(16)]) / len(block)
        
        best_weighted_mse = float('inf')
        best_codebook = None
        
        for codebook in self.all_codebooks:
            codebook_values = self.codebook_values_cache[codebook]
            
            # Compute weighted MSE
            weighted_mse = 0.0
            for code in range(16):
                if code_weights[code] > 0:
                    code_value = self.code_to_value(code)
                    closest_codeword = min(codebook_values, key=lambda x: (x - code_value) ** 2)
                    error = (code_value - closest_codeword) ** 2
                    weighted_mse += code_weights[code] * error
            
            if weighted_mse < best_weighted_mse:
                best_weighted_mse = weighted_mse
                best_codebook = codebook
        
        return best_codebook, best_weighted_mse
    
    def quantize_block(self, block: np.ndarray, codebook: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize a block to codebook values and compute residuals.
        
        Args:
            block: Array of FP4 codes
            codebook: Codebook to use
            
        Returns:
            Tuple of (quantized_values, residuals)
        """
        codebook_values = self.codebook_values_cache[codebook]
        code_values = np.array([self.code_to_value(c) for c in block], dtype=np.float32)
        
        # Find closest codeword for each code
        quantized_values = np.zeros_like(code_values)
        for i, code_value in enumerate(code_values):
            distances = np.abs(codebook_values - code_value)
            closest_idx = np.argmin(distances)
            quantized_values[i] = codebook_values[closest_idx]
        
        # Compute residuals
        residuals = code_values - quantized_values
        
        return quantized_values, residuals
    
    def build_residual_codebook(self, residuals: np.ndarray) -> np.ndarray:
        """
        Build residual codebook using k-means clustering.
        
        Args:
            residuals: Array of residual values
            
        Returns:
            Array of residual codeword values
        """
        # Simple k-means initialization: use quantiles
        sorted_residuals = np.sort(np.abs(residuals))
        indices = np.linspace(0, len(sorted_residuals) - 1, self.num_residual_codewords, dtype=int)
        residual_codewords = sorted_residuals[indices]
        
        # Add negative versions
        residual_codewords = np.concatenate([-residual_codewords[::-1], residual_codewords])
        residual_codewords = residual_codewords[:self.num_residual_codewords]
        
        return residual_codewords
    
    def compress_block_with_residuals(self, block: np.ndarray) -> Dict:
        """
        Compress a block with residual quantization.
        
        Args:
            block: Array of FP4 codes
            
        Returns:
            Dictionary with compression results
        """
        # Phase 4: Select primary codebook
        primary_codebook, primary_mse = self.select_codebook_for_block(block)
        
        # Phase 8d: Quantize and compute residuals
        quantized_values, residuals = self.quantize_block(block, primary_codebook)
        
        # Phase 8d: Build residual codebook
        residual_codewords = self.build_residual_codebook(residuals)
        
        # Phase 8d: Quantize residuals
        residual_indices = np.zeros(len(residuals), dtype=np.int32)
        for i, residual in enumerate(residuals):
            distances = np.abs(residual_codewords - residual)
            residual_indices[i] = np.argmin(distances)
        
        # Compute final MSE
        residual_values = residual_codewords[residual_indices]
        final_values = quantized_values + residual_values
        code_values = np.array([self.code_to_value(c) for c in block], dtype=np.float32)
        final_mse = np.mean((code_values - final_values) ** 2)
        
        return {
            'primary_codebook': primary_codebook,
            'primary_mse': primary_mse,
            'residual_codewords': residual_codewords,
            'residual_indices': residual_indices,
            'final_mse': final_mse,
            'residual_mse': np.mean((residuals - residual_values) ** 2),
        }
    
    def compress_blocks(self, blocks: np.ndarray) -> Dict:
        """
        Compress multiple blocks with residual quantization.
        
        Args:
            blocks: Array of shape (num_blocks, block_size) with FP4 codes
            
        Returns:
            Dictionary with compression results
        """
        num_blocks = len(blocks)
        results_list = []
        mses = []
        
        start_time = time.time()
        
        for block_idx in range(num_blocks):
            block = blocks[block_idx]
            result = self.compress_block_with_residuals(block)
            results_list.append(result)
            mses.append(result['final_mse'])
        
        elapsed = time.time() - start_time
        
        # Calculate compression metrics
        # Primary: 2 bits per code + codebook index
        # Residual: 2 bits per code (4 residual codewords) + residual codebook
        original_bits = num_blocks * self.block_size * 4  # 4 bits per code in FP4
        compressed_bits = num_blocks * self.block_size * 4 + num_blocks * 64  # 4 bits per code (2+2) + codebooks
        compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 0
        
        return {
            'num_blocks': num_blocks,
            'block_size': self.block_size,
            'results': results_list,
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

def test_residual_quantization():
    """Test residual quantization."""
    logger.info("=" * 80)
    logger.info("Phase 8d: Residual Quantization")
    logger.info("=" * 80)
    
    # Load Phase 7c results
    with open('/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase7_unified_production_results.json') as f:
        phase7_results = json.load(f)
    
    # Generate synthetic blocks
    np.random.seed(42)
    synthetic_blocks = np.random.randint(0, 16, size=(100, 128), dtype=np.int32)
    
    # Initialize quantizer
    quantizer = ResidualQuantizer(
        block_size=128,
        num_codewords=4,
        num_residual_codewords=4
    )
    
    logger.info("\nCompressing blocks with residual quantization...")
    results = quantizer.compress_blocks(synthetic_blocks)
    
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
    logger.info(f"  Phase 8d compression: {results['compression_ratio']:.4f}x")
    improvement = (results['compression_ratio'] / phase7_results['compression_ratio'] - 1) * 100
    logger.info(f"  Improvement: {improvement:.2f}%")
    
    # Save results
    results['improvement_vs_phase7c'] = improvement
    results['results'] = []  # Don't save detailed results
    
    with open('/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase8_residual_quantization_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("\nResults saved to phase8_residual_quantization_results.json")
    
    return results

if __name__ == '__main__':
    results = test_residual_quantization()
    logger.info("\n" + "=" * 80)
    logger.info("Phase 8d: Residual Quantization Complete")
    logger.info("=" * 80)

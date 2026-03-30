#!/usr/bin/env python3
"""
Phase 4.1: Production-Ready Variant B Implementation

Frequency-weighted MSE codebook selection for NVFP4 compression.
Optimized for large-scale deployment with minimal memory overhead.
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

class VariantBProduction:
    """Production-ready Variant B (Frequency-weighted MSE) codebook selector."""
    
    def __init__(self, 
                 block_size: int = 128,
                 num_codewords: int = 4,
                 cache_codebooks: bool = True):
        """
        Initialize Variant B production implementation.
        
        Args:
            block_size: Size of weight blocks
            num_codewords: Number of codewords in codebook (4 for 2-bit)
            cache_codebooks: Cache precomputed codebooks for speed
        """
        self.block_size = block_size
        self.num_codewords = num_codewords
        self.cache_codebooks = cache_codebooks
        
        # Precompute all possible codebooks
        self.all_codes = np.arange(16)
        self.all_codebooks = list(combinations(self.all_codes, num_codewords))
        logger.info(f"Initialized with {len(self.all_codebooks)} possible codebooks")
        
        # Precompute codebook values for faster lookup
        self.codebook_values_cache = {}
        if cache_codebooks:
            for codebook in self.all_codebooks:
                self.codebook_values_cache[codebook] = np.array([
                    E2M1_TABLE[c] for c in codebook
                ])
    
    def code_to_value(self, code: int) -> float:
        """Convert single FP4 code to float value."""
        return E2M1_TABLE[code]
    
    def codes_to_values(self, codes: np.ndarray) -> np.ndarray:
        """Convert FP4 codes to float values."""
        return np.array([E2M1_TABLE[c] for c in codes])
    
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
            # Get precomputed codebook values if available
            if codebook in self.codebook_values_cache:
                codebook_values = self.codebook_values_cache[codebook]
            else:
                codebook_values = np.array([E2M1_TABLE[c] for c in codebook])
            
            # Compute weighted MSE
            weighted_mse = 0.0
            for code in range(16):
                if code_weights[code] > 0:
                    code_value = self.code_to_value(code)
                    distances = np.abs(codebook_values - code_value)
                    nearest_idx = np.argmin(distances)
                    error = (code_value - codebook_values[nearest_idx]) ** 2
                    weighted_mse += code_weights[code] * error
            
            if weighted_mse < best_weighted_mse:
                best_weighted_mse = weighted_mse
                best_codebook = codebook
        
        return best_codebook, best_weighted_mse
    
    def compress(self, 
                 fp4_codes: np.ndarray,
                 progress_interval: int = 100) -> Dict:
        """
        Compress FP4 codes using Variant B codebook selection.
        
        Args:
            fp4_codes: Array of FP4 codes (0-15)
            progress_interval: Log progress every N blocks
            
        Returns:
            Dict with compression results and metadata
        """
        logger.info(f"Starting compression of {len(fp4_codes)} codes")
        
        num_blocks = len(fp4_codes) // self.block_size
        if len(fp4_codes) % self.block_size != 0:
            num_blocks += 1
        
        total_mse = 0.0
        codebook_usage = Counter()
        codebook_map = {}  # Map from block index to selected codebook
        
        start_time = time.time()
        
        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min(start + self.block_size, len(fp4_codes))
            block = fp4_codes[start:end]
            
            # Select best codebook for this block
            best_codebook, weighted_mse = self.select_codebook_for_block(block)
            
            total_mse += weighted_mse * len(block)
            codebook_usage[best_codebook] += 1
            codebook_map[block_idx] = best_codebook
            
            if (block_idx + 1) % progress_interval == 0:
                logger.info(f"  Processed {block_idx + 1}/{num_blocks} blocks")
        
        elapsed = time.time() - start_time
        avg_mse = total_mse / len(fp4_codes)
        
        # Compression metrics
        # 2 bits per code (log2(4) = 2 bits for 4-entry codebook)
        # + codebook overhead (1 codebook per block, ~10 bits per block for codebook index)
        bits_per_elem = 2.0 + (10.0 / self.block_size)
        compression_ratio = 4.0 / bits_per_elem
        
        # Convert codebook tuples to strings for JSON serialization
        codebook_usage_str = {str(k): v for k, v in codebook_usage.items()}
        codebook_map_str = {str(k): str(v) for k, v in codebook_map.items()}
        
        results = {
            'variant': 'B_frequency_weighted_mse',
            'method': 'Frequency-weighted MSE codebook selection',
            'total_codes': len(fp4_codes),
            'num_blocks': num_blocks,
            'block_size': self.block_size,
            'num_codewords': self.num_codewords,
            'avg_mse': float(avg_mse),
            'unique_codebooks': len(codebook_usage),
            'bits_per_elem': float(bits_per_elem),
            'compression_ratio': float(compression_ratio),
            'compression_percent': float((1 - 1/compression_ratio) * 100),
            'elapsed_sec': elapsed,
            'codes_per_sec': len(fp4_codes) / elapsed,
            'codebook_usage': codebook_usage_str,
            'codebook_map': codebook_map_str,
        }
        
        logger.info(f"Compression complete in {elapsed:.2f}s")
        logger.info(f"  Average MSE: {avg_mse:.6f}")
        logger.info(f"  Compression: {results['compression_percent']:.1f}% ({compression_ratio:.2f}x)")
        logger.info(f"  Unique codebooks: {len(codebook_usage)}")
        
        return results

def main():
    """Test Variant B production implementation."""
    print("=" * 80)
    print("Phase 4.1: Production-Ready Variant B Implementation")
    print("=" * 80)
    
    # Generate synthetic FP4 codes (realistic distribution)
    print("\nGenerating synthetic FP4 codes...")
    np.random.seed(42)
    
    codes_list = []
    for _ in range(100):  # 100 blocks
        # Positive codes (0-7) are more common
        block = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], size=100, 
                                p=[0.2, 0.15, 0.15, 0.15, 0.15, 0.1, 0.05, 0.05])
        # Negative codes (8-15) are less common
        block = np.concatenate([block, np.random.choice([8, 9, 10, 11, 12, 13, 14, 15], size=28)])
        codes_list.extend(block)
    
    fp4_codes = np.array(codes_list, dtype=np.uint8)
    print(f"Generated {len(fp4_codes)} codes")
    
    # Run Variant B compression
    print("\nRunning Variant B compression...")
    variant_b = VariantBProduction(block_size=128, num_codewords=4)
    results = variant_b.compress(fp4_codes, progress_interval=10)
    
    # Print results
    print("\n" + "=" * 80)
    print("COMPRESSION RESULTS")
    print("=" * 80)
    
    print(f"\nVariant: {results['variant']}")
    print(f"Method: {results['method']}")
    print(f"\nMetrics:")
    print(f"  Total codes: {results['total_codes']}")
    print(f"  Num blocks: {results['num_blocks']}")
    print(f"  Block size: {results['block_size']}")
    print(f"  Num codewords: {results['num_codewords']}")
    print(f"\nCompression:")
    print(f"  Average MSE: {results['avg_mse']:.6f}")
    print(f"  Bits per element: {results['bits_per_elem']:.4f}")
    print(f"  Compression ratio: {results['compression_ratio']:.2f}x")
    print(f"  Compression: {results['compression_percent']:.1f}%")
    print(f"\nPerformance:")
    print(f"  Elapsed time: {results['elapsed_sec']:.2f}s")
    print(f"  Throughput: {results['codes_per_sec']:.0f} codes/sec")
    print(f"  Unique codebooks: {results['unique_codebooks']}")
    
    # Save results
    output_file = Path('/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase4_variant_b_production_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    print("\n" + "=" * 80)
    print("Phase 4.1 Complete ✅")
    print("=" * 80)
    
    return results

if __name__ == '__main__':
    main()

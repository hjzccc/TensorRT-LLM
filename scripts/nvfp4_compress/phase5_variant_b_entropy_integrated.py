#!/usr/bin/env python3
"""
Phase 5: Variant B + Entropy Coding Integration
Combines Variant B codebook selection with Huffman entropy coding of indices
"""

import torch
import numpy as np
from collections import Counter, defaultdict
import json
import time
from pathlib import Path
import logging
from typing import Tuple, Dict, List, Optional
import heapq
from phase4_variant_b_production import VariantBProduction, E2M1_TABLE
from itertools import combinations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HuffmanNode:
    """Node in Huffman tree."""
    def __init__(self, freq: float, symbol: Optional[int] = None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq

class HuffmanCoder:
    """Huffman encoder/decoder for codebook indices."""
    
    def __init__(self):
        self.codes = {}
        self.reverse_codes = {}
        self.tree = None
    
    def build_tree(self, frequencies: Dict[int, int]):
        """Build Huffman tree from frequencies."""
        if not frequencies:
            return
        
        # Create leaf nodes
        heap = []
        for symbol, freq in frequencies.items():
            node = HuffmanNode(freq, symbol=symbol)
            heapq.heappush(heap, node)
        
        # Build tree
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            parent = HuffmanNode(left.freq + right.freq, left=left, right=right)
            heapq.heappush(heap, parent)
        
        self.tree = heap[0] if heap else None
        self._build_codes(self.tree, "")
    
    def _build_codes(self, node: HuffmanNode, code: str):
        """Recursively build Huffman codes."""
        if node is None:
            return
        
        if node.symbol is not None:
            self.codes[node.symbol] = code if code else "0"
            self.reverse_codes[code if code else "0"] = node.symbol
        else:
            self._build_codes(node.left, code + "0")
            self._build_codes(node.right, code + "1")
    
    def encode(self, symbols: np.ndarray) -> Tuple[str, int]:
        """Encode symbols using Huffman codes."""
        encoded = ""
        for symbol in symbols:
            encoded += self.codes.get(int(symbol), "0")
        return encoded, len(encoded)
    
    def get_stats(self) -> Dict:
        """Get Huffman coding statistics."""
        stats = {
            'codes': {str(k): v for k, v in self.codes.items()},
            'code_lengths': {str(k): len(v) for k, v in self.codes.items()},
        }
        return stats

class VariantBWithEntropyCoding:
    """Variant B + Entropy Coding for codebook indices."""
    
    def __init__(self, 
                 block_size: int = 128,
                 num_codewords: int = 4,
                 cache_codebooks: bool = True):
        """Initialize Variant B with entropy coding."""
        self.variant_b = VariantBProduction(block_size, num_codewords, cache_codebooks)
        self.huffman_coder = None
        self.codebook_frequencies = None
    
    def compress_with_entropy(self, codes: np.ndarray) -> Dict:
        """
        Compress codes using Variant B + Huffman entropy coding.
        
        Args:
            codes: Array of FP4 codes
            
        Returns:
            Dictionary with compression results
        """
        logger.info(f"Compressing {len(codes)} codes with Variant B + Entropy Coding")
        
        block_size = self.variant_b.block_size
        num_blocks = len(codes) // block_size
        
        # Step 1: Select codebooks for each block (Variant B)
        logger.info("Step 1: Selecting codebooks for each block...")
        codebook_indices = []
        codebook_mses = []
        
        for i in range(num_blocks):
            block = codes[i*block_size:(i+1)*block_size]
            best_codebook, weighted_mse = self.variant_b.select_codebook_for_block(block)
            
            # Find index of this codebook in all_codebooks
            codebook_idx = self.variant_b.all_codebooks.index(best_codebook)
            codebook_indices.append(codebook_idx)
            codebook_mses.append(weighted_mse)
        
        codebook_indices = np.array(codebook_indices, dtype=np.uint16)
        
        # Step 2: Build Huffman tree from codebook frequencies
        logger.info("Step 2: Building Huffman tree from codebook frequencies...")
        codebook_freq = Counter(codebook_indices)
        self.codebook_frequencies = codebook_freq
        
        self.huffman_coder = HuffmanCoder()
        self.huffman_coder.build_tree(codebook_freq)
        
        # Step 3: Encode codebook indices with Huffman
        logger.info("Step 3: Encoding codebook indices with Huffman...")
        encoded_str, encoded_bits = self.huffman_coder.encode(codebook_indices)
        
        # Calculate metrics
        original_bits = len(codebook_indices) * np.ceil(np.log2(len(self.variant_b.all_codebooks)))
        compressed_bits = encoded_bits
        compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 0
        bits_per_index = compressed_bits / len(codebook_indices)
        
        # Get statistics
        stats = self.huffman_coder.get_stats()
        
        result = {
            'total_blocks': num_blocks,
            'total_codes': len(codes),
            'block_size': block_size,
            'num_codebooks': len(self.variant_b.all_codebooks),
            'original_bits_per_index': np.ceil(np.log2(len(self.variant_b.all_codebooks))),
            'original_bits': int(original_bits),
            'compressed_bits': int(compressed_bits),
            'compression_ratio': compression_ratio,
            'bits_per_index': bits_per_index,
            'huffman_stats': stats,
            'codebook_frequencies': {str(k): int(v) for k, v in codebook_freq.items()},
            'mean_mse': float(np.mean(codebook_mses)),
            'max_mse': float(np.max(codebook_mses)),
            'min_mse': float(np.min(codebook_mses)),
        }
        
        return result

def generate_synthetic_fp4_data(num_codes: int = 100000, seed: int = 42) -> np.ndarray:
    """Generate synthetic FP4 codes for testing."""
    np.random.seed(seed)
    probs = np.array([
        0.15, 0.12, 0.10, 0.08,
        0.12, 0.10, 0.08, 0.05,
        0.15, 0.12, 0.10, 0.08,
        0.12, 0.10, 0.08, 0.05,
    ], dtype=np.float32)
    probs = probs / probs.sum()
    codes = np.random.choice(16, size=num_codes, p=probs).astype(np.uint8)
    return codes

def main():
    """Main implementation function."""
    logger.info("=" * 80)
    logger.info("Phase 5: Variant B + Entropy Coding Integration")
    logger.info("=" * 80)
    
    # Generate synthetic FP4 data
    logger.info("Generating synthetic FP4 data...")
    codes = generate_synthetic_fp4_data(num_codes=100000)
    
    # Compress with Variant B + Entropy Coding
    logger.info("Compressing with Variant B + Entropy Coding...")
    compressor = VariantBWithEntropyCoding()
    result = compressor.compress_with_entropy(codes)
    
    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("VARIANT B + ENTROPY CODING RESULTS")
    logger.info("=" * 80)
    
    logger.info(f"\nTotal codes: {result['total_codes']}")
    logger.info(f"Total blocks: {result['total_blocks']}")
    logger.info(f"Block size: {result['block_size']}")
    logger.info(f"Number of possible codebooks: {result['num_codebooks']}")
    
    logger.info(f"\nCodebook Index Compression:")
    logger.info(f"  Original bits per index: {result['original_bits_per_index']:.2f}")
    logger.info(f"  Compressed bits per index: {result['bits_per_index']:.4f}")
    logger.info(f"  Compression ratio: {result['compression_ratio']:.4f}x")
    logger.info(f"  Improvement: {(1 - result['bits_per_index']/result['original_bits_per_index']) * 100:.2f}%")
    
    logger.info(f"\nMSE Statistics:")
    logger.info(f"  Mean MSE: {result['mean_mse']:.6f}")
    logger.info(f"  Min MSE: {result['min_mse']:.6f}")
    logger.info(f"  Max MSE: {result['max_mse']:.6f}")
    
    logger.info(f"\nTop 10 Most Frequent Codebooks:")
    sorted_freq = sorted(result['codebook_frequencies'].items(), key=lambda x: int(x[1]), reverse=True)
    for i, (codebook_idx, freq) in enumerate(sorted_freq[:10]):
        logger.info(f"  Codebook {codebook_idx:4s}: {int(freq):4d} ({int(freq)/result['total_blocks']*100:5.1f}%)")
    
    # Save results
    output_file = Path(__file__).parent / "phase5_variant_b_entropy_results.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"\nResults saved to {output_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    
    return result

if __name__ == "__main__":
    result = main()

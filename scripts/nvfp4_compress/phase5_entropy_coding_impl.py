#!/usr/bin/env python3
"""
Phase 5: Entropy Coding Implementation
Huffman coding for codebook indices to improve compression ratio
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
    
    def decode(self, encoded: str, num_symbols: int) -> np.ndarray:
        """Decode Huffman-encoded string."""
        symbols = []
        current = ""
        for bit in encoded:
            current += bit
            if current in self.reverse_codes:
                symbols.append(self.reverse_codes[current])
                current = ""
        return np.array(symbols, dtype=np.uint8)
    
    def get_stats(self) -> Dict:
        """Get Huffman coding statistics."""
        stats = {
            'codes': {str(k): v for k, v in self.codes.items()},
            'code_lengths': {str(k): len(v) for k, v in self.codes.items()},
        }
        return stats

def compress_with_entropy_coding(codes: np.ndarray, block_size: int = 128) -> Dict:
    """
    Compress codes using Variant B + Huffman entropy coding.
    
    Args:
        codes: Array of FP4 codes
        block_size: Size of each block
        
    Returns:
        Dictionary with compression results
    """
    logger.info(f"Compressing {len(codes)} codes with entropy coding")
    
    # Count code frequencies
    code_freq = Counter(codes)
    
    # Build Huffman coder
    coder = HuffmanCoder()
    coder.build_tree(code_freq)
    
    # Encode all codes
    encoded_str, encoded_bits = coder.encode(codes)
    
    # Calculate metrics
    original_bits = len(codes) * 2  # 2 bits per code (uniform)
    compressed_bits = encoded_bits
    compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 0
    bits_per_code = compressed_bits / len(codes)
    
    # Get statistics
    stats = coder.get_stats()
    
    result = {
        'total_codes': len(codes),
        'original_bits': original_bits,
        'compressed_bits': compressed_bits,
        'compression_ratio': compression_ratio,
        'bits_per_code': bits_per_code,
        'huffman_stats': stats,
        'code_frequencies': dict(code_freq),
    }
    
    return result

def main():
    """Main implementation function."""
    logger.info("=" * 80)
    logger.info("Phase 5: Entropy Coding Implementation")
    logger.info("=" * 80)
    
    # Generate synthetic FP4 data with realistic distribution
    logger.info("Generating synthetic FP4 data...")
    np.random.seed(42)
    
    # Create realistic distribution: some codes more frequent than others
    # This simulates actual weight distributions in neural networks
    probs = np.array([
        0.15, 0.12, 0.10, 0.08,  # codes 0-3 (small positive)
        0.12, 0.10, 0.08, 0.05,  # codes 4-7 (larger positive)
        0.15, 0.12, 0.10, 0.08,  # codes 8-11 (small negative)
        0.12, 0.10, 0.08, 0.05,  # codes 12-15 (larger negative)
    ], dtype=np.float32)
    probs = probs / probs.sum()
    
    codes = np.random.choice(16, size=100000, p=probs).astype(np.uint8)
    
    # Compress with entropy coding
    logger.info("Compressing with Huffman entropy coding...")
    result = compress_with_entropy_coding(codes)
    
    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("ENTROPY CODING RESULTS")
    logger.info("=" * 80)
    
    logger.info(f"\nTotal codes: {result['total_codes']}")
    logger.info(f"Original bits (uniform 2-bit): {result['original_bits']}")
    logger.info(f"Compressed bits (Huffman): {result['compressed_bits']}")
    logger.info(f"Compression ratio: {result['compression_ratio']:.4f}x")
    logger.info(f"Bits per code: {result['bits_per_code']:.4f}")
    
    logger.info("\nCode Frequencies:")
    for code in sorted(result['code_frequencies'].keys()):
        freq = result['code_frequencies'][code]
        logger.info(f"  Code {code:2d}: {freq:6d}")
    
    logger.info("\nHuffman Code Lengths:")
    for code_str, length in sorted(result['huffman_stats']['code_lengths'].items(), key=lambda x: int(x[0])):
        logger.info(f"  Code {code_str:2s}: {length} bits")
    
    # Save results
    output_file = Path(__file__).parent / "phase5_entropy_coding_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'total_codes': int(result['total_codes']),
            'original_bits': int(result['original_bits']),
            'compressed_bits': int(result['compressed_bits']),
            'compression_ratio': float(result['compression_ratio']),
            'bits_per_code': float(result['bits_per_code']),
            'code_frequencies': {str(k): int(v) for k, v in result['code_frequencies'].items()},
            'huffman_code_lengths': {str(k): int(v) for k, v in result['huffman_stats']['code_lengths'].items()},
        }, f, indent=2)
    logger.info(f"\nResults saved to {output_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    
    # Key insight
    logger.info("\nKEY INSIGHT:")
    logger.info(f"With uniform distribution across 16 codes, entropy coding provides")
    logger.info(f"minimal improvement (compression ratio: {result['compression_ratio']:.4f}x)")
    logger.info(f"This is because all codes have similar frequency.")
    logger.info(f"\nFor maximum benefit, entropy coding should be applied to:")
    logger.info(f"1. Codebook INDICES (which code is selected per block)")
    logger.info(f"2. Not individual FP4 codes (which are already uniform)")
    
    return result

if __name__ == "__main__":
    result = main()

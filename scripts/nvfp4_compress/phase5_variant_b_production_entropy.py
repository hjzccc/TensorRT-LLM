#!/usr/bin/env python3
"""
Phase 5: Production-Ready Variant B + Entropy Coding
Combines Variant B codebook selection with Huffman entropy coding of indices
"""

import torch
import numpy as np
from collections import Counter
import json
import time
from pathlib import Path
import logging
from typing import Tuple, Dict, List, Optional
import heapq
from itertools import combinations
import pickle

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
    
    def encode(self, symbols: np.ndarray) -> Tuple[bytes, int]:
        """Encode symbols using Huffman codes."""
        encoded = ""
        for symbol in symbols:
            encoded += self.codes.get(int(symbol), "0")
        
        # Convert bit string to bytes
        # Pad to multiple of 8
        padding = (8 - len(encoded) % 8) % 8
        encoded += "0" * padding
        
        # Convert to bytes
        encoded_bytes = bytes(int(encoded[i:i+8], 2) for i in range(0, len(encoded), 8))
        
        return encoded_bytes, len(encoded) - padding
    
    def decode(self, encoded_bytes: bytes, num_bits: int, num_symbols: int) -> np.ndarray:
        """Decode Huffman-encoded bytes."""
        # Convert bytes to bit string
        encoded = ""
        for byte in encoded_bytes:
            encoded += format(byte, '08b')
        
        # Trim to actual number of bits
        encoded = encoded[:num_bits]
        
        # Decode
        symbols = []
        current = ""
        for bit in encoded:
            current += bit
            if current in self.reverse_codes:
                symbols.append(self.reverse_codes[current])
                current = ""
        
        return np.array(symbols[:num_symbols], dtype=np.uint16)
    
    def get_stats(self) -> Dict:
        """Get Huffman coding statistics."""
        stats = {
            'codes': {str(k): v for k, v in self.codes.items()},
            'code_lengths': {str(k): len(v) for k, v in self.codes.items()},
        }
        return stats

class VariantBProduction:
    """Production-ready Variant B (Frequency-weighted MSE) codebook selector."""
    
    def __init__(self, 
                 block_size: int = 128,
                 num_codewords: int = 4,
                 cache_codebooks: bool = True):
        """Initialize Variant B production implementation."""
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
    
    def select_codebook_for_block(self, block: np.ndarray) -> Tuple[Tuple, float]:
        """Select best codebook for a block using frequency-weighted MSE."""
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
                    # Find closest codeword
                    closest_idx = np.argmin(np.abs(codebook_values - code_value))
                    closest_value = codebook_values[closest_idx]
                    mse = (code_value - closest_value) ** 2
                    weighted_mse += code_weights[code] * mse
            
            if weighted_mse < best_weighted_mse:
                best_weighted_mse = weighted_mse
                best_codebook = codebook
        
        return best_codebook, best_weighted_mse

class VariantBWithEntropyCoding:
    """Production-ready Variant B + Entropy Coding."""
    
    def __init__(self, 
                 block_size: int = 128,
                 num_codewords: int = 4,
                 cache_codebooks: bool = True):
        """Initialize Variant B with entropy coding."""
        self.variant_b = VariantBProduction(block_size, num_codewords, cache_codebooks)
        self.huffman_coder = None
        self.codebook_frequencies = None
        self.block_size = block_size
    
    def compress(self, codes: np.ndarray) -> Dict:
        """
        Compress codes using Variant B + Huffman entropy coding.
        
        Args:
            codes: Array of FP4 codes
            
        Returns:
            Dictionary with compression results and metadata
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
        encoded_bytes, num_bits = self.huffman_coder.encode(codebook_indices)
        
        # Calculate metrics
        original_bits = len(codebook_indices) * np.ceil(np.log2(len(self.variant_b.all_codebooks)))
        compressed_bits = num_bits
        compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 0
        bits_per_index = compressed_bits / len(codebook_indices)
        
        # Get statistics
        stats = self.huffman_coder.get_stats()
        
        result = {
            'total_blocks': num_blocks,
            'total_codes': len(codes),
            'block_size': block_size,
            'num_codebooks': len(self.variant_b.all_codebooks),
            'original_bits_per_index': float(np.ceil(np.log2(len(self.variant_b.all_codebooks)))),
            'original_bits': int(original_bits),
            'compressed_bits': int(compressed_bits),
            'compression_ratio': float(compression_ratio),
            'bits_per_index': float(bits_per_index),
            'huffman_stats': stats,
            'codebook_frequencies': {str(k): int(v) for k, v in codebook_freq.items()},
            'mean_mse': float(np.mean(codebook_mses)),
            'max_mse': float(np.max(codebook_mses)),
            'min_mse': float(np.min(codebook_mses)),
            'encoded_bytes': encoded_bytes.hex(),  # Store as hex for JSON
            'num_bits': num_bits,
        }
        
        return result
    
    def save_checkpoint(self, result: Dict, output_path: str):
        """Save compression result and Huffman tree to checkpoint."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save Huffman tree
        tree_file = output_path / "huffman_tree.pkl"
        with open(tree_file, 'wb') as f:
            pickle.dump(self.huffman_coder, f)
        logger.info(f"Saved Huffman tree to {tree_file}")
        
        # Save metadata
        metadata_file = output_path / "compression_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved metadata to {metadata_file}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load Huffman tree from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        # Load Huffman tree
        tree_file = checkpoint_path / "huffman_tree.pkl"
        with open(tree_file, 'rb') as f:
            self.huffman_coder = pickle.load(f)
        logger.info(f"Loaded Huffman tree from {tree_file}")
        
        # Load metadata
        metadata_file = checkpoint_path / "compression_metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata from {metadata_file}")
        
        return metadata

def main():
    """Main implementation function."""
    logger.info("=" * 80)
    logger.info("Phase 5: Production-Ready Variant B + Entropy Coding")
    logger.info("=" * 80)
    
    # Generate synthetic FP4 data
    logger.info("Generating synthetic FP4 data...")
    np.random.seed(42)
    probs = np.array([
        0.15, 0.12, 0.10, 0.08,
        0.12, 0.10, 0.08, 0.05,
        0.15, 0.12, 0.10, 0.08,
        0.12, 0.10, 0.08, 0.05,
    ], dtype=np.float32)
    probs = probs / probs.sum()
    codes = np.random.choice(16, size=100000, p=probs).astype(np.uint8)
    
    # Compress with Variant B + Entropy Coding
    logger.info("Compressing with Variant B + Entropy Coding...")
    compressor = VariantBWithEntropyCoding()
    result = compressor.compress(codes)
    
    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("COMPRESSION RESULTS")
    logger.info("=" * 80)
    
    logger.info(f"\nTotal codes: {result['total_codes']}")
    logger.info(f"Total blocks: {result['total_blocks']}")
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
    
    # Save results
    output_file = Path(__file__).parent / "phase5_variant_b_production_entropy_results.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"\nResults saved to {output_file}")
    
    # Save checkpoint
    checkpoint_dir = Path(__file__).parent / "phase5_checkpoint_entropy"
    compressor.save_checkpoint(result, str(checkpoint_dir))
    
    logger.info("\n" + "=" * 80)
    logger.info("COMPRESSION COMPLETE")
    logger.info("=" * 80)
    
    return result

if __name__ == "__main__":
    result = main()

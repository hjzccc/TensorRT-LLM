#!/usr/bin/env python3
"""
Phase 7c: Unified Production Pipeline
Combines Phase 4 (Variant B) + Phase 5 (Entropy) + Phase 7 (Codebook Pruning)

This is the complete compression pipeline:
1. Phase 4: Frequency-weighted MSE codebook selection
2. Phase 5: Huffman entropy coding on codebook indices
3. Phase 7: Codebook pruning (use only 26 most-used codebooks)
"""

import torch
import numpy as np
from itertools import combinations
from collections import Counter
import json
import time
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import heapq
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
        padding = (8 - len(encoded) % 8) % 8
        encoded += "0" * padding
        
        # Convert to bytes
        encoded_bytes = bytes(int(encoded[i:i+8], 2) for i in range(0, len(encoded), 8))
        
        return encoded_bytes, len(encoded) - padding
    
    def decode(self, encoded_bytes: bytes, num_bits: int, num_symbols: int) -> np.ndarray:
        """Decode Huffman-encoded bytes."""
        # Convert bytes to bit string
        bit_string = ''.join(format(byte, '08b') for byte in encoded_bytes)
        bit_string = bit_string[:num_bits]
        
        # Decode using tree
        symbols = []
        node = self.tree
        for bit in bit_string:
            if bit == '0':
                node = node.left
            else:
                node = node.right
            
            if node.symbol is not None:
                symbols.append(node.symbol)
                node = self.tree
        
        return np.array(symbols, dtype=np.int32)

class Phase7UnifiedCompressor:
    """Unified Phase 4+5+7 compression pipeline."""
    
    def __init__(self, 
                 block_size: int = 128,
                 num_codewords: int = 4,
                 pruned_codebooks: Optional[List[Tuple]] = None):
        """
        Initialize unified compressor.
        
        Args:
            block_size: Size of weight blocks
            num_codewords: Number of codewords in codebook (4 for 2-bit)
            pruned_codebooks: List of codebooks to use (if None, use all 1820)
        """
        self.block_size = block_size
        self.num_codewords = num_codewords
        
        # Phase 4: All possible codebooks
        self.all_codes = np.arange(16)
        self.all_codebooks = list(combinations(self.all_codes, num_codewords))
        
        # Phase 7: Pruned codebooks (26 most-used)
        if pruned_codebooks is None:
            self.pruned_codebooks = self.all_codebooks
            self.use_pruning = False
        else:
            self.pruned_codebooks = pruned_codebooks
            self.use_pruning = True
            # Create mapping from full codebook index to pruned index
            self.codebook_to_pruned = {}
            for pruned_idx, codebook in enumerate(self.pruned_codebooks):
                full_idx = self.all_codebooks.index(codebook)
                self.codebook_to_pruned[full_idx] = pruned_idx
        
        # Precompute codebook values
        self.codebook_values_cache = {}
        for codebook in self.all_codebooks:
            self.codebook_values_cache[codebook] = np.array([
                E2M1_TABLE[c] for c in codebook
            ])
        
        # Phase 5: Huffman coder
        self.huffman_coder = HuffmanCoder()
        
        logger.info(f"Initialized with {len(self.all_codebooks)} total codebooks")
        if self.use_pruning:
            logger.info(f"Using {len(self.pruned_codebooks)} pruned codebooks")
    
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
        
        # Search only pruned codebooks if using pruning
        search_codebooks = self.pruned_codebooks if self.use_pruning else self.all_codebooks
        
        for codebook in search_codebooks:
            codebook_values = self.codebook_values_cache[codebook]
            
            # Compute weighted MSE
            weighted_mse = 0.0
            for code in range(16):
                if code_weights[code] > 0:
                    code_value = self.code_to_value(code)
                    # Find closest codeword
                    closest_codeword = min(codebook_values, key=lambda x: (x - code_value) ** 2)
                    error = (code_value - closest_codeword) ** 2
                    weighted_mse += code_weights[code] * error
            
            if weighted_mse < best_weighted_mse:
                best_weighted_mse = weighted_mse
                best_codebook = codebook
        
        return best_codebook, best_weighted_mse
    
    def compress_block(self, block: np.ndarray) -> Tuple[Tuple, np.ndarray, float]:
        """
        Compress a single block.
        
        Args:
            block: Array of FP4 codes
            
        Returns:
            Tuple of (codebook, indices, mse)
        """
        # Phase 4: Select best codebook
        codebook, mse = self.select_codebook_for_block(block)
        
        # Quantize codes to codebook indices
        codebook_values = self.codebook_values_cache[codebook]
        indices = np.zeros(len(block), dtype=np.int32)
        
        for i, code in enumerate(block):
            code_value = self.code_to_value(code)
            # Find closest codeword index
            distances = np.abs(codebook_values - code_value)
            indices[i] = np.argmin(distances)
        
        return codebook, indices, mse
    
    def compress_blocks(self, blocks: np.ndarray) -> Dict:
        """
        Compress multiple blocks.
        
        Args:
            blocks: Array of shape (num_blocks, block_size) with FP4 codes
            
        Returns:
            Dictionary with compression results
        """
        num_blocks = len(blocks)
        codebooks = []
        all_indices = []
        mses = []
        
        start_time = time.time()
        
        for block_idx in range(num_blocks):
            block = blocks[block_idx]
            codebook, indices, mse = self.compress_block(block)
            
            codebooks.append(codebook)
            all_indices.append(indices)
            mses.append(mse)
        
        # Phase 5: Entropy coding on indices
        # Build Huffman tree from codebook index frequencies
        codebook_indices = np.array([
            self.all_codebooks.index(cb) if not self.use_pruning 
            else self.codebook_to_pruned[self.all_codebooks.index(cb)]
            for cb in codebooks
        ])
        
        codebook_freq = Counter(codebook_indices)
        self.huffman_coder.build_tree(dict(codebook_freq))
        
        # Encode codebook indices
        encoded_codebook_indices, num_codebook_bits = self.huffman_coder.encode(codebook_indices)
        
        # Encode block indices (use fixed 2 bits per index for 4 codewords)
        all_indices_flat = np.concatenate(all_indices)
        encoded_block_indices, num_block_bits = self.huffman_coder.encode(all_indices_flat)
        
        elapsed = time.time() - start_time
        
        # Calculate compression metrics
        original_bits = num_blocks * 10.83 + num_blocks * self.block_size * 2  # codebook + indices
        compressed_bits = num_codebook_bits + num_block_bits
        compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 0
        
        return {
            'num_blocks': num_blocks,
            'block_size': self.block_size,
            'codebooks': codebooks,
            'codebook_indices': codebook_indices,
            'block_indices': all_indices,
            'encoded_codebook_indices': encoded_codebook_indices,
            'encoded_block_indices': encoded_block_indices,
            'num_codebook_bits': num_codebook_bits,
            'num_block_bits': num_block_bits,
            'original_bits': original_bits,
            'compressed_bits': compressed_bits,
            'compression_ratio': compression_ratio,
            'avg_mse': np.mean(mses),
            'elapsed_sec': elapsed,
            'throughput_blocks_per_sec': num_blocks / elapsed if elapsed > 0 else 0,
            'use_pruning': self.use_pruning,
            'num_pruned_codebooks': len(self.pruned_codebooks) if self.use_pruning else len(self.all_codebooks),
        }

def test_unified_compression():
    """Test unified compression on synthetic data."""
    logger.info("=" * 80)
    logger.info("Phase 7c: Unified Compression Test (Phase 4+5+7)")
    logger.info("=" * 80)
    
    # Load pruned codebooks from Phase 7 analysis
    with open('/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase7_codebook_pruning_analysis.json') as f:
        analysis = json.load(f)
    
    pruned_codebooks = [tuple(cb) for cb in analysis['analysis']['used_codebooks_list']]
    logger.info(f"Loaded {len(pruned_codebooks)} pruned codebooks")
    
    # Test 1: Without pruning (Phase 4+5 only)
    logger.info("\n--- Test 1: Phase 4+5 (No Pruning) ---")
    compressor_no_prune = Phase7UnifiedCompressor(pruned_codebooks=None)
    
    # Generate synthetic blocks
    np.random.seed(42)
    synthetic_blocks = np.random.randint(0, 16, size=(100, 128), dtype=np.int32)
    
    results_no_prune = compressor_no_prune.compress_blocks(synthetic_blocks)
    logger.info(f"Compression ratio: {results_no_prune['compression_ratio']:.4f}x")
    logger.info(f"Bits per element: {results_no_prune['compressed_bits'] / (100 * 128):.4f}")
    logger.info(f"Throughput: {results_no_prune['throughput_blocks_per_sec']:.1f} blocks/sec")
    
    # Test 2: With pruning (Phase 4+5+7)
    logger.info("\n--- Test 2: Phase 4+5+7 (With Pruning) ---")
    compressor_prune = Phase7UnifiedCompressor(pruned_codebooks=pruned_codebooks)
    
    results_prune = compressor_prune.compress_blocks(synthetic_blocks)
    logger.info(f"Compression ratio: {results_prune['compression_ratio']:.4f}x")
    logger.info(f"Bits per element: {results_prune['compressed_bits'] / (100 * 128):.4f}")
    logger.info(f"Throughput: {results_prune['throughput_blocks_per_sec']:.1f} blocks/sec")
    
    # Compare
    improvement = (results_prune['compression_ratio'] / results_no_prune['compression_ratio'] - 1) * 100
    logger.info(f"\nImprovement with pruning: {improvement:.2f}%")
    
    # Save results
    results_prune['improvement_pct'] = improvement
    with open('/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase7_unified_production_results.json', 'w') as f:
        json.dump(results_prune, f, indent=2, default=str)
    
    logger.info("\nResults saved to phase7_unified_production_results.json")
    
    return results_prune

if __name__ == '__main__':
    results = test_unified_compression()
    logger.info("\n" + "=" * 80)
    logger.info("Phase 7c: Unified Compression Complete")
    logger.info("=" * 80)

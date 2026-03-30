#!/usr/bin/env python3
"""
Phase 6: Production-Ready Variant B + Entropy Coding + Adaptive Scaling
Combines Variant B, Huffman entropy coding, and adaptive block sizes
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
        
        return np.array(symbols[:num_symbols], dtype=np.int32)


class AdaptiveBlockQuantizer:
    """Quantizes weights using adaptive block sizes and entropy coding."""
    
    def __init__(self, block_sizes: Optional[Dict[str, int]] = None):
        self.block_sizes = block_sizes or {}
        self.huffman_coders = {}
    
    def analyze_layer_distribution(self, weights: np.ndarray) -> int:
        """Determine optimal block size for a layer."""
        abs_weights = np.abs(weights)
        nonzero = abs_weights[abs_weights > 0]
        
        if len(nonzero) == 0:
            return 128  # Default
        
        # Compute entropy
        hist, _ = np.histogram(nonzero, bins=256)
        hist = hist[hist > 0]
        probs = hist / hist.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Compute kurtosis
        mean = nonzero.mean()
        std = nonzero.std()
        if std > 0:
            kurtosis = np.mean(((nonzero - mean) / std) ** 4) - 3
        else:
            kurtosis = 0
        
        # Determine block size
        if entropy > 6.5 and kurtosis < 1.0:
            return 256
        elif entropy > 5.5 and kurtosis < 2.0:
            return 128
        else:
            return 64
    
    def quantize_layer(self, weights: np.ndarray, layer_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Quantize a layer using adaptive block size and entropy coding."""
        
        # Determine block size
        if layer_name not in self.block_sizes:
            block_size = self.analyze_layer_distribution(weights)
            self.block_sizes[layer_name] = block_size
        else:
            block_size = self.block_sizes[layer_name]
        
        # Reshape for block quantization
        original_shape = weights.shape
        weights_flat = weights.reshape(-1)
        
        # Pad to multiple of block_size
        pad_size = (block_size - len(weights_flat) % block_size) % block_size
        if pad_size > 0:
            weights_flat = np.pad(weights_flat, (0, pad_size), mode='constant')
        
        # Reshape into blocks
        num_blocks = len(weights_flat) // block_size
        weights_blocks = weights_flat[:num_blocks * block_size].reshape(num_blocks, block_size)
        
        # Quantize each block
        codes = []
        scales = []
        
        for block in weights_blocks:
            # Find scale (max absolute value)
            scale = np.max(np.abs(block))
            if scale == 0:
                scale = 1.0
            
            # Normalize and quantize
            normalized = block / scale
            
            # Find nearest FP4 code
            code = np.argmin(np.abs(E2M1_TABLE[:, None] - normalized[None, :]), axis=0)
            
            codes.append(code)
            scales.append(scale)
        
        codes = np.concatenate(codes)
        scales = np.array(scales, dtype=np.float32)
        
        # Compute MSE
        reconstructed = E2M1_TABLE[codes] * np.repeat(scales, block_size)[:len(codes)]
        mse = np.mean((weights_flat[:len(codes)] - reconstructed) ** 2)
        
        return codes, scales, np.array([block_size], dtype=np.int32), mse
    
    def build_huffman_coder(self, codes: np.ndarray, layer_name: str):
        """Build Huffman coder for codebook indices."""
        # Count frequencies
        frequencies = Counter(codes)
        
        # Build Huffman tree
        coder = HuffmanCoder()
        coder.build_tree(dict(frequencies))
        
        self.huffman_coders[layer_name] = coder
        
        return coder
    
    def compress_checkpoint(self, checkpoint_path: str, output_dir: str) -> Dict:
        """Compress checkpoint using Phase 6 (Variant B + Entropy + Adaptive)."""
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = {}
        total_original = 0
        total_compressed = 0
        
        start_time = time.time()
        
        for idx, (name, param) in enumerate(state_dict.items()):
            if param.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                weights = param.cpu().numpy().astype(np.float32)
                
                # Quantize
                codes, scales, block_size_arr, mse = self.quantize_layer(weights, name)
                
                # Build Huffman coder
                coder = self.build_huffman_coder(codes, name)
                
                # Encode indices
                encoded_indices, num_bits = coder.encode(codes)
                
                # Compute compression
                original_bits = weights.size * 32  # FP32
                block_size = int(block_size_arr[0])
                num_blocks = (weights.size + block_size - 1) // block_size
                
                # Compressed: encoded indices + scales (32 bits each) + metadata
                compressed_bits = num_bits + num_blocks * 32 + 64  # 64 bits for metadata
                
                total_original += original_bits
                total_compressed += compressed_bits
                
                # Save to file
                layer_file = Path(output_dir) / f"{name.replace('/', '_')}.npz"
                np.savez_compressed(
                    layer_file,
                    codes=codes,
                    scales=scales,
                    block_size=block_size_arr,
                    encoded_indices=encoded_indices,
                    num_bits=np.array([num_bits], dtype=np.int32),
                    huffman_codes=np.array(list(coder.codes.items()), dtype=object)
                )
                
                results[name] = {
                    'mse': float(mse),
                    'original_bits': int(original_bits),
                    'compressed_bits': int(compressed_bits),
                    'compression_ratio': float(original_bits / compressed_bits),
                    'block_size': int(block_size),
                    'num_blocks': int(num_blocks),
                    'huffman_avg_bits': float(num_bits / len(codes))
                }
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"  [{idx+1}] {name}: ratio={original_bits/compressed_bits:.2f}x, block_size={block_size}")
        
        elapsed = time.time() - start_time
        
        overall_ratio = total_original / total_compressed if total_compressed > 0 else 0
        results['overall'] = {
            'total_original_bits': int(total_original),
            'total_compressed_bits': int(total_compressed),
            'overall_compression_ratio': float(overall_ratio),
            'bits_per_element': float(total_compressed / (total_original / 32)),
            'compression_time': float(elapsed),
            'throughput': float(total_original / 32 / elapsed)
        }
        
        logger.info(f"\nCompression complete in {elapsed:.1f}s")
        logger.info(f"Overall compression: {overall_ratio:.2f}x ({total_compressed / (total_original / 32):.4f} bits/elem)")
        logger.info(f"Throughput: {total_original / 32 / elapsed:.0f} elements/sec")
        
        return results


def test_phase6_on_synthetic() -> Dict:
    """Test Phase 6 on synthetic data."""
    logger.info("=" * 80)
    logger.info("PHASE 6: SYNTHETIC DATA COMPRESSION TEST")
    logger.info("=" * 80)
    
    np.random.seed(42)
    
    # Create synthetic checkpoint
    checkpoint = {
        'layer1_uniform': torch.randn(1000, 1000, dtype=torch.float32),
        'layer2_normal': torch.randn(1000, 1000, dtype=torch.float32) * 0.5,
        'layer3_exponential': torch.from_numpy(np.random.exponential(0.5, (1000, 1000))).float(),
    }
    
    # Compress
    quantizer = AdaptiveBlockQuantizer()
    results = {}
    
    total_original = 0
    total_compressed = 0
    
    for name, param in checkpoint.items():
        weights = param.numpy().astype(np.float32)
        codes, scales, block_size, mse = quantizer.quantize_layer(weights, name)
        
        # Build Huffman coder
        coder = quantizer.build_huffman_coder(codes, name)
        encoded_indices, num_bits = coder.encode(codes)
        
        # Compute compression
        original_bits = weights.size * 32
        block_size_val = int(block_size[0])
        num_blocks = (weights.size + block_size_val - 1) // block_size_val
        compressed_bits = num_bits + num_blocks * 32 + 64
        
        total_original += original_bits
        total_compressed += compressed_bits
        
        results[name] = {
            'mse': float(mse),
            'compression_ratio': float(original_bits / compressed_bits),
            'block_size': int(block_size_val),
            'huffman_avg_bits': float(num_bits / len(codes))
        }
        
        logger.info(f"{name}: ratio={original_bits/compressed_bits:.2f}x, block_size={block_size_val}, huffman={num_bits/len(codes):.2f} bits")
    
    overall_ratio = total_original / total_compressed
    results['overall'] = {
        'overall_compression_ratio': float(overall_ratio),
        'bits_per_element': float(total_compressed / (total_original / 32))
    }
    
    logger.info(f"\nOverall: {overall_ratio:.2f}x ({total_compressed / (total_original / 32):.4f} bits/elem)")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "phase6_compressed"
        
        quantizer = AdaptiveBlockQuantizer()
        results = quantizer.compress_checkpoint(checkpoint_path, output_dir)
        
        # Save results
        output_file = "phase6_variant_b_adaptive_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    else:
        # Run synthetic test
        results = test_phase6_on_synthetic()
        
        # Save results
        output_file = "phase6_variant_b_adaptive_synthetic_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")

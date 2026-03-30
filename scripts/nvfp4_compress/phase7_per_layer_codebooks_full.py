#!/usr/bin/env python3
"""
Phase 7: Per-Layer Codebooks (Full Implementation)
Builds on Phase 6 (Adaptive Scaling + Entropy Coding) with per-layer codebooks
"""

import torch
import numpy as np
import json
import time
from pathlib import Path
import logging
from typing import Dict, Tuple, List
from collections import Counter
import heapq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HuffmanEncoder:
    """Huffman encoder for entropy coding."""
    
    def __init__(self):
        self.codes = {}
        self.code_lengths = {}
    
    def build_tree(self, frequencies: Dict[int, int]):
        """Build Huffman tree from frequencies."""
        if not frequencies:
            return
        
        # Create heap of (frequency, unique_id, node)
        heap = [(freq, i, [i, None, None]) for i, freq in frequencies.items()]
        heapq.heapify(heap)
        
        counter = len(frequencies)
        while len(heap) > 1:
            freq1, _, node1 = heapq.heappop(heap)
            freq2, _, node2 = heapq.heappop(heap)
            
            merged = [counter, node1, node2]
            heapq.heappush(heap, (freq1 + freq2, counter, merged))
            counter += 1
        
        if heap:
            _, _, root = heap[0]
            self._build_codes(root, "")
    
    def _build_codes(self, node, code):
        """Recursively build codes from tree."""
        if node[1] is None and node[2] is None:  # Leaf node
            self.codes[node[0]] = code if code else "0"
            self.code_lengths[node[0]] = len(self.codes[node[0]])
        else:
            if node[1]:
                self._build_codes(node[1], code + "0")
            if node[2]:
                self._build_codes(node[2], code + "1")
    
    def encode(self, indices: np.ndarray) -> Tuple[str, int]:
        """Encode indices using Huffman codes."""
        encoded = ""
        for idx in indices:
            encoded += self.codes.get(int(idx), "0")
        return encoded, len(encoded)


class Phase7PerLayerQuantizer:
    """Phase 7: Per-Layer Codebooks with Entropy Coding and Adaptive Scaling."""
    
    def __init__(self, block_size: int = 128):
        self.block_size = block_size
        self.layer_codebooks = {}
        self.layer_huffman = {}
    
    def learn_codebook_fast(self, weights: np.ndarray, layer_name: str, num_clusters: int = 256) -> np.ndarray:
        """Learn codebook using histogram-based approach."""
        
        # Reshape weights into blocks
        weights_flat = weights.reshape(-1)
        
        # Pad to multiple of block_size
        pad_size = (self.block_size - len(weights_flat) % self.block_size) % self.block_size
        if pad_size > 0:
            weights_flat = np.pad(weights_flat, (0, pad_size), mode='constant')
        
        # Reshape into blocks
        num_blocks = len(weights_flat) // self.block_size
        weights_blocks = weights_flat[:num_blocks * self.block_size].reshape(num_blocks, self.block_size)
        
        # Normalize blocks
        scales = np.max(np.abs(weights_blocks), axis=1, keepdims=True)
        scales[scales == 0] = 1.0
        normalized_blocks = weights_blocks / scales
        
        # Sample random values from normalized blocks
        sample_size = min(10000, len(normalized_blocks) * self.block_size)
        flat_normalized = normalized_blocks.reshape(-1)
        sample_indices = np.random.choice(len(flat_normalized), size=sample_size, replace=False)
        samples = flat_normalized[sample_indices]
        
        # Create codebook by quantizing samples
        hist, bin_edges = np.histogram(samples, bins=num_clusters)
        codebook = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Store codebook
        self.layer_codebooks[layer_name] = codebook
        
        return codebook
    
    def quantize_layer(self, weights: np.ndarray, layer_name: str) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Quantize a layer using its learned codebook with entropy coding."""
        
        # Learn codebook if not already learned
        if layer_name not in self.layer_codebooks:
            self.learn_codebook_fast(weights, layer_name)
        
        codebook = self.layer_codebooks[layer_name]
        
        # Reshape weights into blocks
        weights_flat = weights.reshape(-1)
        
        # Pad to multiple of block_size
        pad_size = (self.block_size - len(weights_flat) % self.block_size) % self.block_size
        if pad_size > 0:
            weights_flat = np.pad(weights_flat, (0, pad_size), mode='constant')
        
        # Reshape into blocks
        num_blocks = len(weights_flat) // self.block_size
        weights_blocks = weights_flat[:num_blocks * self.block_size].reshape(num_blocks, self.block_size)
        
        # Quantize each block
        codes = []
        scales = []
        
        for block in weights_blocks:
            # Find scale
            scale = np.max(np.abs(block))
            if scale == 0:
                scale = 1.0
            
            # Normalize
            normalized = block / scale
            
            # Find nearest codebook entry
            distances = np.abs(codebook[:, None] - normalized[None, :])
            code = np.argmin(distances, axis=0)
            
            codes.append(code)
            scales.append(scale)
        
        codes = np.concatenate(codes)
        scales = np.array(scales, dtype=np.float32)
        
        # Compute MSE
        reconstructed = codebook[codes] * np.repeat(scales, self.block_size)[:len(codes)]
        mse = np.mean((weights_flat[:len(codes)] - reconstructed) ** 2)
        
        # Apply Huffman coding to indices
        code_frequencies = Counter(codes)
        huffman = HuffmanEncoder()
        huffman.build_tree(dict(code_frequencies))
        self.layer_huffman[layer_name] = huffman
        
        encoded_bits, num_bits = huffman.encode(codes)
        huffman_bits_per_code = num_bits / len(codes) if len(codes) > 0 else 0
        
        return codes, scales, mse, huffman_bits_per_code


def compare_phase6_vs_phase7_full() -> Dict:
    """Compare Phase 6 vs Phase 7 with full pipeline."""
    logger.info("=" * 80)
    logger.info("PHASE 7: FULL IMPLEMENTATION (Per-Layer + Entropy + Adaptive)")
    logger.info("=" * 80)
    
    np.random.seed(42)
    
    # Create synthetic checkpoint with multiple layers
    checkpoint = {
        'layer1_uniform': torch.randn(1000, 1000, dtype=torch.float32),
        'layer2_normal': torch.randn(1000, 1000, dtype=torch.float32) * 0.5,
        'layer3_exponential': torch.from_numpy(np.random.exponential(0.5, (1000, 1000))).float(),
        'layer4_mixed': torch.randn(1000, 1000, dtype=torch.float32) * 0.3 + 0.5,
        'layer5_sparse': torch.randn(1000, 1000, dtype=torch.float32) * 0.1,
    }
    
    # Phase 6 baseline (from previous results)
    phase6_bits_per_elem = 1.8329
    phase6_compression = 32 / phase6_bits_per_elem
    
    logger.info(f"\nPhase 6 Baseline (Adaptive Scaling + Entropy):")
    logger.info(f"  Bits per element: {phase6_bits_per_elem:.4f}")
    logger.info(f"  Compression ratio: {phase6_compression:.4f}x")
    
    # Compress with Phase 7
    quantizer = Phase7PerLayerQuantizer(block_size=128)
    results = {}
    
    total_original = 0
    total_compressed = 0
    
    start_time = time.time()
    
    for name, param in checkpoint.items():
        weights = param.numpy().astype(np.float32)
        codes, scales, mse, huffman_bits_per_code = quantizer.quantize_layer(weights, name)
        
        # Compute compression
        original_bits = weights.size * 32
        codebook_bits = len(quantizer.layer_codebooks[name]) * 32
        num_blocks = (weights.size + 128 - 1) // 128
        
        # Codes: Huffman-encoded
        code_bits = len(codes) * huffman_bits_per_code
        scale_bits = num_blocks * 32
        compressed_bits = code_bits + scale_bits + codebook_bits
        
        total_original += original_bits
        total_compressed += compressed_bits
        
        results[name] = {
            'mse': float(mse),
            'compression_ratio': float(original_bits / compressed_bits),
            'codebook_size': int(len(quantizer.layer_codebooks[name])),
            'num_blocks': int(num_blocks),
            'huffman_bits_per_code': float(huffman_bits_per_code)
        }
        
        logger.info(f"{name}: ratio={original_bits/compressed_bits:.2f}x, huffman={huffman_bits_per_code:.2f} bits/code, mse={mse:.6f}")
    
    elapsed = time.time() - start_time
    
    overall_ratio = total_original / total_compressed
    phase7_bits_per_elem = total_compressed / (total_original / 32)
    phase7_compression = 32 / phase7_bits_per_elem
    
    improvement = (phase6_bits_per_elem - phase7_bits_per_elem) / phase6_bits_per_elem * 100
    
    results['overall'] = {
        'phase6_bits_per_elem': float(phase6_bits_per_elem),
        'phase6_compression': float(phase6_compression),
        'phase7_bits_per_elem': float(phase7_bits_per_elem),
        'phase7_compression': float(phase7_compression),
        'improvement_percent': float(improvement),
        'improvement_bits': float(phase6_bits_per_elem - phase7_bits_per_elem),
        'compression_time': float(elapsed),
        'num_layers': int(len(checkpoint))
    }
    
    logger.info(f"\nPhase 7 Results (Per-Layer + Entropy + Adaptive):")
    logger.info(f"  Bits per element: {phase7_bits_per_elem:.4f}")
    logger.info(f"  Compression ratio: {phase7_compression:.4f}x")
    logger.info(f"  Improvement over Phase 6: {improvement:.1f}%")
    logger.info(f"  Time: {elapsed:.1f}s")
    logger.info(f"  Layers processed: {len(checkpoint)}")
    
    # Decision logic
    logger.info(f"\n" + "=" * 80)
    logger.info("DECISION LOGIC")
    logger.info("=" * 80)
    
    if improvement >= 19.3:
        logger.info(f"✅ EXCELLENT: {improvement:.1f}% improvement (target: 19.3%)")
        logger.info("   Recommendation: Continue to Phase 8 (Learned Codebooks EM)")
    elif improvement >= 15:
        logger.info(f"✅ GOOD: {improvement:.1f}% improvement (target: 19.3%)")
        logger.info("   Recommendation: Deploy Phase 4+5+6+7 or continue to Phase 8")
    elif improvement >= 10:
        logger.info(f"⚠️  ACCEPTABLE: {improvement:.1f}% improvement (target: 19.3%)")
        logger.info("   Recommendation: Deploy Phase 4+5+6+7")
    else:
        logger.info(f"❌ POOR: {improvement:.1f}% improvement (target: 19.3%)")
        logger.info("   Recommendation: Revert to Phase 4+5+6, try Phase 9 instead")
    
    return results


if __name__ == "__main__":
    results = compare_phase6_vs_phase7_full()
    
    # Save results
    output_file = "phase7_full_implementation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to {output_file}")

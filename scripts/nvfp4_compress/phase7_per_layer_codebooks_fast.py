#!/usr/bin/env python3
"""
Phase 7: Per-Layer Codebooks (Fast Implementation)
Uses fast codebook learning for per-layer quantization
"""

import torch
import numpy as np
from collections import Counter, defaultdict
import json
import time
from pathlib import Path
import logging
from typing import Tuple, Dict, List, Optional

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

class FastPerLayerCodebookQuantizer:
    """Fast per-layer codebook quantization using random sampling."""
    
    def __init__(self, block_size: int = 128):
        self.block_size = block_size
        self.layer_codebooks = {}
    
    def learn_codebook_fast(self, weights: np.ndarray, layer_name: str, num_clusters: int = 256) -> np.ndarray:
        """Learn codebook using random sampling (fast)."""
        
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
        # Use histogram-based approach
        hist, bin_edges = np.histogram(samples, bins=num_clusters)
        codebook = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        logger.info(f"  Learned codebook for {layer_name} ({len(codebook)} clusters)")
        
        return codebook
    
    def quantize_layer(self, weights: np.ndarray, layer_name: str) -> Tuple[np.ndarray, np.ndarray, float]:
        """Quantize a layer using its learned codebook."""
        
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
        
        return codes, scales, mse


def test_phase7_fast() -> Dict:
    """Test Phase 7 fast implementation on synthetic data."""
    logger.info("=" * 80)
    logger.info("PHASE 7: FAST SYNTHETIC DATA TEST")
    logger.info("=" * 80)
    
    np.random.seed(42)
    
    # Create synthetic checkpoint
    checkpoint = {
        'layer1_uniform': torch.randn(1000, 1000, dtype=torch.float32),
        'layer2_normal': torch.randn(1000, 1000, dtype=torch.float32) * 0.5,
        'layer3_exponential': torch.from_numpy(np.random.exponential(0.5, (1000, 1000))).float(),
    }
    
    # Compress
    quantizer = FastPerLayerCodebookQuantizer(block_size=128)
    results = {}
    
    total_original = 0
    total_compressed = 0
    
    start_time = time.time()
    
    for name, param in checkpoint.items():
        weights = param.numpy().astype(np.float32)
        codes, scales, mse = quantizer.quantize_layer(weights, name)
        
        # Compute compression
        original_bits = weights.size * 32
        codebook_bits = len(quantizer.layer_codebooks[name]) * 32
        num_blocks = (weights.size + 128 - 1) // 128
        
        # Codes: 10 bits per code (for 1024 codebook size)
        code_bits = len(codes) * 10
        scale_bits = num_blocks * 32
        compressed_bits = code_bits + scale_bits + codebook_bits
        
        total_original += original_bits
        total_compressed += compressed_bits
        
        results[name] = {
            'mse': float(mse),
            'compression_ratio': float(original_bits / compressed_bits),
            'codebook_size': int(len(quantizer.layer_codebooks[name])),
            'num_blocks': int(num_blocks)
        }
        
        logger.info(f"{name}: ratio={original_bits/compressed_bits:.2f}x, codebook_size={len(quantizer.layer_codebooks[name])}, mse={mse:.6f}")
    
    elapsed = time.time() - start_time
    
    overall_ratio = total_original / total_compressed
    results['overall'] = {
        'overall_compression_ratio': float(overall_ratio),
        'bits_per_element': float(total_compressed / (total_original / 32)),
        'compression_time': float(elapsed)
    }
    
    logger.info(f"\nOverall: {overall_ratio:.2f}x ({total_compressed / (total_original / 32):.4f} bits/elem)")
    logger.info(f"Time: {elapsed:.1f}s")
    
    return results


if __name__ == "__main__":
    results = test_phase7_fast()
    
    # Save results
    output_file = "phase7_per_layer_codebooks_fast_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_file}")

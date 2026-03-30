#!/usr/bin/env python3
"""
Phase 7: Per-Layer Codebooks for NVFP4 Compression
Uses different codebooks for different layers to optimize compression
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
from sklearn.cluster import KMeans

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

class PerLayerCodebookQuantizer:
    """Quantizes weights using per-layer learned codebooks."""
    
    def __init__(self, num_codebooks: int = 1024, block_size: int = 128):
        self.num_codebooks = num_codebooks
        self.block_size = block_size
        self.layer_codebooks = {}
        self.layer_stats = {}
    
    def learn_codebook(self, weights: np.ndarray, layer_name: str, num_clusters: int = 1024) -> np.ndarray:
        """Learn optimal codebook for a layer using K-means."""
        
        # Reshape weights into blocks
        original_shape = weights.shape
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
        
        # Flatten for clustering
        normalized_flat = normalized_blocks.reshape(-1, 1)
        
        # Learn codebook using K-means
        logger.info(f"  Learning codebook for {layer_name} ({num_clusters} clusters)...")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=3, max_iter=100)
        kmeans.fit(normalized_flat)
        
        codebook = kmeans.cluster_centers_.flatten()
        
        # Store codebook
        self.layer_codebooks[layer_name] = codebook
        
        return codebook
    
    def quantize_layer(self, weights: np.ndarray, layer_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Quantize a layer using its learned codebook."""
        
        # Learn codebook if not already learned
        if layer_name not in self.layer_codebooks:
            self.learn_codebook(weights, layer_name)
        
        codebook = self.layer_codebooks[layer_name]
        
        # Reshape weights into blocks
        original_shape = weights.shape
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
        
        return codes, scales, np.array([len(codebook)], dtype=np.int32), mse
    
    def compress_checkpoint(self, checkpoint_path: str, output_dir: str) -> Dict:
        """Compress checkpoint using per-layer codebooks."""
        
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
                codes, scales, codebook_size, mse = self.quantize_layer(weights, name)
                
                # Compute compression
                original_bits = weights.size * 32  # FP32
                codebook_bits = len(self.layer_codebooks[name]) * 32  # Codebook storage
                num_blocks = (weights.size + self.block_size - 1) // self.block_size
                
                # Compressed: codes (10 bits per code) + scales (32 bits each) + codebook
                compressed_bits = len(codes) * 10 + num_blocks * 32 + codebook_bits
                
                total_original += original_bits
                total_compressed += compressed_bits
                
                results[name] = {
                    'mse': float(mse),
                    'original_bits': int(original_bits),
                    'compressed_bits': int(compressed_bits),
                    'compression_ratio': float(original_bits / compressed_bits),
                    'codebook_size': int(len(self.layer_codebooks[name])),
                    'num_blocks': int(num_blocks)
                }
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"  [{idx+1}] {name}: ratio={original_bits/compressed_bits:.2f}x, codebook_size={len(self.layer_codebooks[name])}")
        
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


def test_phase7_on_synthetic() -> Dict:
    """Test Phase 7 on synthetic data."""
    logger.info("=" * 80)
    logger.info("PHASE 7: SYNTHETIC DATA COMPRESSION TEST")
    logger.info("=" * 80)
    
    np.random.seed(42)
    
    # Create synthetic checkpoint with different layer types
    checkpoint = {
        'layer1_uniform': torch.randn(1000, 1000, dtype=torch.float32),
        'layer2_normal': torch.randn(1000, 1000, dtype=torch.float32) * 0.5,
        'layer3_exponential': torch.from_numpy(np.random.exponential(0.5, (1000, 1000))).float(),
    }
    
    # Compress
    quantizer = PerLayerCodebookQuantizer(num_codebooks=1024, block_size=128)
    results = {}
    
    total_original = 0
    total_compressed = 0
    
    for name, param in checkpoint.items():
        weights = param.numpy().astype(np.float32)
        codes, scales, codebook_size, mse = quantizer.quantize_layer(weights, name)
        
        # Compute compression
        original_bits = weights.size * 32
        codebook_bits = len(quantizer.layer_codebooks[name]) * 32
        num_blocks = (weights.size + 128 - 1) // 128
        compressed_bits = len(codes) * 10 + num_blocks * 32 + codebook_bits
        
        total_original += original_bits
        total_compressed += compressed_bits
        
        results[name] = {
            'mse': float(mse),
            'compression_ratio': float(original_bits / compressed_bits),
            'codebook_size': int(len(quantizer.layer_codebooks[name])),
            'num_blocks': int(num_blocks)
        }
        
        logger.info(f"{name}: ratio={original_bits/compressed_bits:.2f}x, codebook_size={len(quantizer.layer_codebooks[name])}, mse={mse:.6f}")
    
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
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "phase7_compressed"
        
        quantizer = PerLayerCodebookQuantizer()
        results = quantizer.compress_checkpoint(checkpoint_path, output_dir)
        
        # Save results
        output_file = "phase7_per_layer_codebooks_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    else:
        # Run synthetic test
        results = test_phase7_on_synthetic()
        
        # Save results
        output_file = "phase7_per_layer_codebooks_synthetic_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")

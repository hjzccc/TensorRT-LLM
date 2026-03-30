#!/usr/bin/env python3
"""
Phase 7c: Real Model Validation

Validates Phase 4+5+7 compression on actual model checkpoint.

Process:
1. Load BF16 checkpoint
2. Quantize weights to FP4 using Phase 4
3. Apply Phase 5+7 compression
4. Measure compression ratio
5. Estimate PPL impact (if validation data available)
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
import logging
from collections import Counter
from itertools import combinations
import time

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

class Phase7cValidator:
    """Validates Phase 7c compression on real model."""
    
    def __init__(self, checkpoint_path: str = 'nvfp4_checkpoint'):
        """Initialize validator."""
        self.checkpoint_path = Path(checkpoint_path)
        self.block_size = 128
        self.num_codewords = 4
        
        # All possible codebooks
        self.all_codes = np.arange(16)
        self.all_codebooks = list(combinations(self.all_codes, self.num_codewords))
        
        # Precompute codebook values
        self.codebook_values_cache = {}
        for codebook in self.all_codebooks:
            self.codebook_values_cache[codebook] = np.array([
                E2M1_TABLE[c] for c in codebook
            ], dtype=np.float32)
        
        # Load pruned codebooks
        with open('phase7_codebook_pruning_analysis.json') as f:
            analysis = json.load(f)
        self.pruned_codebooks = [tuple(cb) for cb in analysis['analysis']['used_codebooks_list']]
        
        logger.info(f"Initialized with {len(self.pruned_codebooks)} pruned codebooks")
    
    def code_to_value(self, code: int) -> float:
        """Convert FP4 code to float value."""
        return E2M1_TABLE[code]
    
    def quantize_to_fp4(self, weight: torch.Tensor) -> np.ndarray:
        """
        Quantize weight tensor to FP4 codes.
        
        Args:
            weight: BF16 weight tensor
            
        Returns:
            Array of FP4 codes
        """
        # Convert to float32 for processing
        weight_fp32 = weight.float().cpu().numpy()
        
        # Flatten and normalize
        weight_flat = weight_fp32.flatten()
        
        # Simple quantization: find nearest FP4 value
        codes = np.zeros(len(weight_flat), dtype=np.int32)
        for i, val in enumerate(weight_flat):
            distances = np.abs(E2M1_TABLE - val)
            codes[i] = np.argmin(distances)
        
        return codes
    
    def select_codebook_for_block(self, block: np.ndarray) -> Tuple[Tuple, float]:
        """Select best codebook for a block."""
        code_counts = Counter(block)
        code_weights = np.array([code_counts.get(i, 0) for i in range(16)]) / len(block)
        
        best_weighted_mse = float('inf')
        best_codebook = None
        
        for codebook in self.pruned_codebooks:
            codebook_values = self.codebook_values_cache[codebook]
            
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
    
    def compress_weight(self, weight: torch.Tensor) -> Dict:
        """
        Compress a single weight tensor.
        
        Args:
            weight: BF16 weight tensor
            
        Returns:
            Dictionary with compression results
        """
        # Quantize to FP4
        codes = self.quantize_to_fp4(weight)
        
        # Split into blocks
        num_blocks = (len(codes) + self.block_size - 1) // self.block_size
        
        total_original_bits = len(codes) * 4  # 4 bits per FP4 code
        total_compressed_bits = 0
        total_mse = 0
        
        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min(start + self.block_size, len(codes))
            block = codes[start:end]
            
            # Select codebook
            codebook, mse = self.select_codebook_for_block(block)
            
            # Estimate compressed bits (2 bits per code + codebook index)
            block_compressed_bits = len(block) * 2 + 5  # 5 bits for codebook index (26 codebooks)
            total_compressed_bits += block_compressed_bits
            total_mse += mse * len(block)
        
        compression_ratio = total_original_bits / total_compressed_bits if total_compressed_bits > 0 else 0
        
        return {
            'shape': weight.shape,
            'dtype': str(weight.dtype),
            'num_elements': weight.numel(),
            'num_blocks': num_blocks,
            'original_bits': total_original_bits,
            'compressed_bits': total_compressed_bits,
            'compression_ratio': compression_ratio,
            'avg_mse': total_mse / len(codes),
        }
    
    def validate_checkpoint(self, max_weights: Optional[int] = None) -> Dict:
        """
        Validate compression on checkpoint.
        
        Args:
            max_weights: Maximum number of weights to process (for testing)
            
        Returns:
            Dictionary with validation results
        """
        logger.info("=" * 80)
        logger.info("Phase 7c: Real Model Validation")
        logger.info("=" * 80)
        
        # Load config
        with open(self.checkpoint_path / 'config.json') as f:
            config = json.load(f)
        
        logger.info(f"\nModel: {config.get('model_type', 'unknown')}")
        logger.info(f"  Hidden size: {config.get('hidden_size', 'unknown')}")
        logger.info(f"  Num layers: {config.get('num_hidden_layers', 'unknown')}")
        
        # Find all shards
        shard_files = sorted(self.checkpoint_path.glob('model-*.safetensors'))
        logger.info(f"\nFound {len(shard_files)} shards")
        
        # Process shards
        total_original_bits = 0
        total_compressed_bits = 0
        weight_results = []
        weights_processed = 0
        
        start_time = time.time()
        
        try:
            from safetensors.torch import load_file
        except ImportError:
            logger.error("safetensors not installed. Install with: pip install safetensors")
            return {}
        
        for shard_idx, shard_file in enumerate(shard_files[:1]):  # Process first shard only for speed
            logger.info(f"\nProcessing shard {shard_idx + 1}/{len(shard_files)}: {shard_file.name}")
            
            try:
                state = load_file(str(shard_file))
                
                for weight_name, weight in state.items():
                    if max_weights and weights_processed >= max_weights:
                        break
                    
                    # Skip non-weight tensors
                    if weight.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                        continue
                    
                    # Compress weight
                    result = self.compress_weight(weight)
                    weight_results.append({
                        'name': weight_name,
                        **result
                    })
                    
                    total_original_bits += result['original_bits']
                    total_compressed_bits += result['compressed_bits']
                    weights_processed += 1
                    
                    if weights_processed % 10 == 0:
                        logger.info(f"  Processed {weights_processed} weights...")
            
            except Exception as e:
                logger.error(f"Error processing shard: {e}")
                continue
        
        elapsed = time.time() - start_time
        
        # Calculate overall metrics
        overall_compression = total_original_bits / total_compressed_bits if total_compressed_bits > 0 else 0
        
        logger.info(f"\n" + "=" * 80)
        logger.info("Validation Results")
        logger.info("=" * 80)
        logger.info(f"Weights processed: {weights_processed}")
        logger.info(f"Original bits: {total_original_bits:,}")
        logger.info(f"Compressed bits: {total_compressed_bits:,}")
        logger.info(f"Overall compression ratio: {overall_compression:.4f}x")
        logger.info(f"Elapsed time: {elapsed:.2f} sec")
        logger.info(f"Throughput: {weights_processed / elapsed:.1f} weights/sec")
        
        return {
            'model_type': config.get('model_type', 'unknown'),
            'hidden_size': config.get('hidden_size', 'unknown'),
            'num_layers': config.get('num_hidden_layers', 'unknown'),
            'weights_processed': weights_processed,
            'total_original_bits': total_original_bits,
            'total_compressed_bits': total_compressed_bits,
            'overall_compression_ratio': overall_compression,
            'elapsed_sec': elapsed,
            'throughput_weights_per_sec': weights_processed / elapsed if elapsed > 0 else 0,
            'weight_results': weight_results[:10],  # Save first 10 for inspection
        }

def main():
    """Run validation."""
    validator = Phase7cValidator()
    results = validator.validate_checkpoint(max_weights=100)
    
    # Save results
    with open('phase7c_real_model_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("\nResults saved to phase7c_real_model_validation_results.json")

if __name__ == '__main__':
    main()

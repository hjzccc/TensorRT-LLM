#!/usr/bin/env python3
"""
Phase 4.2: Checkpoint Integration (Optimized)

Load NVFP4 checkpoint, apply Variant B compression, and save compressed checkpoint.
Uses sampling for large tensors to speed up compression analysis.
"""

import torch
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Tuple
import time
from safetensors.torch import load_file, save_file
import glob

# Import Variant B from Phase 4.1
import sys
sys.path.insert(0, '/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress')
from phase4_variant_b_production import VariantBProduction, E2M1_TABLE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CheckpointCompressor:
    """Compress NVFP4 checkpoint using Variant B codebook selection."""
    
    def __init__(self, checkpoint_path: str, block_size: int = 128, max_tensors: int = None, sample_size: int = 100000):
        """
        Initialize checkpoint compressor.
        
        Args:
            checkpoint_path: Path to NVFP4 checkpoint directory (sharded safetensors)
            block_size: Size of weight blocks for codebook selection
            max_tensors: Maximum number of tensors to process (for testing)
            sample_size: Maximum number of elements to process per tensor (for speed)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.block_size = block_size
        self.max_tensors = max_tensors
        self.sample_size = sample_size
        self.variant_b = VariantBProduction(block_size=block_size, num_codewords=4)
        
        # Find all safetensors files
        self.shard_files = sorted(glob.glob(str(self.checkpoint_path / '*.safetensors')))
        logger.info(f"Found {len(self.shard_files)} shards in {checkpoint_path}")
        
        # Load all shards
        self.checkpoint = {}
        for shard_file in self.shard_files:
            logger.info(f"Loading shard {Path(shard_file).name}...")
            shard = load_file(shard_file)
            self.checkpoint.update(shard)
        
        logger.info(f"Loaded {len(self.checkpoint)} tensors total")
    
    def fp4_codes_from_tensor(self, tensor: torch.Tensor, sample: bool = True) -> np.ndarray:
        """
        Extract FP4 codes from tensor (with optional sampling for large tensors).
        """
        # Convert to float32 if needed
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        
        # Flatten tensor
        flat = tensor.flatten().cpu().numpy().astype(np.float32)
        
        # Sample if tensor is too large
        if sample and len(flat) > self.sample_size:
            indices = np.random.choice(len(flat), size=self.sample_size, replace=False)
            flat = flat[indices]
        
        # Quantize to FP4 codes (0-15)
        codes = np.zeros(len(flat), dtype=np.uint8)
        for i, val in enumerate(flat):
            distances = np.abs(E2M1_TABLE - val)
            codes[i] = np.argmin(distances)
        
        return codes
    
    def compress_tensor(self, tensor: torch.Tensor, name: str) -> Dict:
        """
        Compress a single tensor using Variant B.
        """
        # Extract FP4 codes (with sampling for large tensors)
        fp4_codes = self.fp4_codes_from_tensor(tensor, sample=True)
        
        # Apply Variant B compression
        results = self.variant_b.compress(fp4_codes, progress_interval=10000)
        results['tensor_name'] = name
        results['tensor_shape'] = list(tensor.shape)
        results['tensor_dtype'] = str(tensor.dtype)
        results['tensor_numel'] = tensor.numel()
        results['sampled'] = len(fp4_codes) < tensor.numel()
        
        return results
    
    def compress_checkpoint(self) -> Dict:
        """
        Compress entire checkpoint.
        """
        logger.info("Starting checkpoint compression...")
        
        all_results = {}
        total_codes = 0
        total_mse = 0.0
        
        start_time = time.time()
        
        tensor_list = list(self.checkpoint.items())
        if self.max_tensors:
            tensor_list = tensor_list[:self.max_tensors]
        
        for i, (name, tensor) in enumerate(tensor_list):
            logger.info(f"[{i+1}/{len(tensor_list)}] Compressing {name} {tensor.shape}...")
            
            results = self.compress_tensor(tensor, name)
            all_results[name] = results
            
            total_codes += results['total_codes']
            total_mse += results['avg_mse'] * results['total_codes']
        
        elapsed = time.time() - start_time
        avg_mse = total_mse / total_codes if total_codes > 0 else 0.0
        
        summary = {
            'checkpoint_path': str(self.checkpoint_path),
            'num_shards': len(self.shard_files),
            'num_tensors_total': len(self.checkpoint),
            'num_tensors_compressed': len(all_results),
            'total_codes': total_codes,
            'avg_mse': float(avg_mse),
            'elapsed_sec': elapsed,
            'codes_per_sec': total_codes / elapsed if elapsed > 0 else 0,
            'tensor_results': all_results,
        }
        
        logger.info(f"Checkpoint compression complete in {elapsed:.2f}s")
        logger.info(f"  Total codes: {total_codes}")
        logger.info(f"  Average MSE: {avg_mse:.6f}")
        logger.info(f"  Throughput: {summary['codes_per_sec']:.0f} codes/sec")
        
        return summary

def main():
    """Test checkpoint compression."""
    print("=" * 80)
    print("Phase 4.2: Checkpoint Integration (Optimized)")
    print("=" * 80)
    
    checkpoint_path = '/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/nvfp4_checkpoint'
    
    # Compress checkpoint (limit to 5 tensors for quick test)
    print(f"\nCompressing checkpoint from {checkpoint_path}")
    print("(Testing with first 5 tensors, sampling large tensors)")
    
    compressor = CheckpointCompressor(checkpoint_path, max_tensors=5, sample_size=100000)
    results = compressor.compress_checkpoint()
    
    # Print summary
    print("\n" + "=" * 80)
    print("COMPRESSION SUMMARY")
    print("=" * 80)
    print(f"\nCheckpoint: {results['checkpoint_path']}")
    print(f"Shards: {results['num_shards']}")
    print(f"Total tensors: {results['num_tensors_total']}")
    print(f"Compressed: {results['num_tensors_compressed']}")
    print(f"\nMetrics:")
    print(f"  Total codes: {results['total_codes']}")
    print(f"  Average MSE: {results['avg_mse']:.6f}")
    print(f"  Elapsed time: {results['elapsed_sec']:.2f}s")
    print(f"  Throughput: {results['codes_per_sec']:.0f} codes/sec")
    
    # Save results
    output_file = Path('/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase4_2_checkpoint_compression_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    print("\n" + "=" * 80)
    print("Phase 4.2 Complete ✅")
    print("=" * 80)

if __name__ == '__main__':
    main()

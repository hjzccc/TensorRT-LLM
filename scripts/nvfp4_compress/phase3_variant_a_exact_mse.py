"""
Phase 3: Variant A - Exact MSE Baseline
Brute-force enumeration of all C(16,4) = 1820 4-entry codebook subsets.
Baseline for comparing against optimized variants.
"""

import torch
import numpy as np
from itertools import combinations
import json
from pathlib import Path
import time

def variant_a_exact_mse(fp4_codes: np.ndarray, block_size: int = 128) -> dict:
    """
    Variant A: Exact MSE baseline.
    
    For each block of FP4 codes, find the 4-entry codebook that minimizes MSE
    by exhaustive search over all C(16,4) = 1820 possible subsets.
    
    Args:
        fp4_codes: Array of FP4 code values (0-15)
        block_size: Size of weight blocks
        
    Returns:
        dict with compression metrics and codebook info
    """
    # All possible FP4 code values
    all_codes = np.arange(16)
    
    # Generate all C(16,4) = 1820 possible 4-entry codebooks
    all_codebooks = list(combinations(all_codes, 4))
    print(f"Total possible 4-entry codebooks: {len(all_codebooks)}")
    
    # Reshape codes into blocks
    num_blocks = len(fp4_codes) // block_size
    if len(fp4_codes) % block_size != 0:
        num_blocks += 1
    
    total_mse = 0.0
    total_codes = 0
    codebook_usage = {}
    
    start_time = time.time()
    
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, len(fp4_codes))
        block = fp4_codes[start:end]
        
        # Find best codebook for this block
        best_mse = float('inf')
        best_codebook = None
        
        for codebook in all_codebooks:
            codebook_array = np.array(codebook)
            
            # Quantize block using this codebook
            quantized = np.zeros_like(block)
            for i, code in enumerate(block):
                # Find nearest codebook entry
                distances = np.abs(codebook_array - code)
                nearest_idx = np.argmin(distances)
                quantized[i] = codebook_array[nearest_idx]
            
            # Compute MSE
            mse = np.mean((block - quantized) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_codebook = codebook
        
        # Record results
        total_mse += best_mse * len(block)
        total_codes += len(block)
        
        codebook_key = str(sorted(best_codebook))
        codebook_usage[codebook_key] = codebook_usage.get(codebook_key, 0) + 1
        
        if (block_idx + 1) % 100 == 0:
            print(f"  Processed {block_idx + 1}/{num_blocks} blocks")
    
    elapsed = time.time() - start_time
    
    # Compute metrics
    avg_mse = total_mse / total_codes
    
    # Compression: 4 codes per block, 4 bits per code = 16 bits per block
    # vs. 128 codes × 4 bits = 512 bits per block
    compression_ratio = (block_size * 4) / 16  # 32x for block_size=128
    bits_per_elem = 16 / block_size
    
    results = {
        'variant': 'A_exact_mse',
        'method': 'Brute-force enumeration of C(16,4) subsets',
        'avg_mse': float(avg_mse),
        'compression_ratio': float(compression_ratio),
        'bits_per_elem': float(bits_per_elem),
        'num_blocks': num_blocks,
        'num_codebooks_searched': len(all_codebooks),
        'unique_codebooks_used': len(codebook_usage),
        'elapsed_sec': elapsed,
        'codes_per_sec': total_codes / elapsed,
    }
    
    return results

def load_fp4_codes_from_checkpoint(checkpoint_path: str, max_tensors: int = 10) -> np.ndarray:
    """Load FP4 codes from NVFP4 checkpoint."""
    from safetensors.torch import load_file
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = load_file(checkpoint_path)
    
    all_codes = []
    tensor_count = 0
    
    for key, tensor in state_dict.items():
        if tensor_count >= max_tensors:
            break
        
        if 'weight' in key and tensor.dtype == torch.float32:
            # Extract FP4 codes (assuming they're stored as float32 in range [0, 15])
            codes = (tensor * 15).round().clamp(0, 15).numpy().astype(np.uint8)
            all_codes.append(codes.flatten())
            tensor_count += 1
            print(f"  Loaded {key}: {tensor.shape}")
    
    if all_codes:
        all_codes = np.concatenate(all_codes)
        print(f"Total codes loaded: {len(all_codes)}")
        return all_codes
    else:
        print("No suitable tensors found in checkpoint")
        return None

if __name__ == '__main__':
    # Test on synthetic data first
    print("=" * 80)
    print("Phase 3: Variant A - Exact MSE Baseline")
    print("=" * 80)
    
    # Generate synthetic FP4 codes
    print("\n1. Testing on synthetic data...")
    np.random.seed(42)
    synthetic_codes = np.random.randint(0, 16, size=10000)
    
    results = variant_a_exact_mse(synthetic_codes, block_size=128)
    
    print("\nResults (Synthetic Data):")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    # Try to load real checkpoint
    checkpoint_path = '/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/nvfp4_checkpoint/model-00000-of-00733.safetensors'
    
    if Path(checkpoint_path).exists():
        print("\n2. Testing on real NVFP4 checkpoint...")
        try:
            import torch
            from safetensors.torch import load_file
            
            fp4_codes = load_fp4_codes_from_checkpoint(checkpoint_path, max_tensors=5)
            
            if fp4_codes is not None:
                results_real = variant_a_exact_mse(fp4_codes, block_size=128)
                
                print("\nResults (Real Checkpoint):")
                for key, value in results_real.items():
                    print(f"  {key}: {value}")
                
                # Save results
                output_file = '/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase3_variant_a_results.json'
                with open(output_file, 'w') as f:
                    json.dump(results_real, f, indent=2)
                print(f"\nResults saved to {output_file}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    else:
        print(f"\nCheckpoint not found at {checkpoint_path}")
    
    print("\n" + "=" * 80)
    print("Variant A baseline complete")
    print("=" * 80)

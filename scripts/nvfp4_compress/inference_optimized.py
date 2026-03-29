#!/usr/bin/env python3
"""Step 3: Inference Optimization.

Implement fast codebook lookup and measure decompression latency.
"""

import time
import json
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np

# E2M1 code → float value
E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


class FastCodebookDecompressor:
    """Fast codebook-based decompression with LUT optimization."""
    
    def __init__(self, codebook: List[int]):
        """Initialize with a codebook.
        
        Args:
            codebook: List of FP4 codes (e.g., [0, 2, 5, 7, 8, 10, 13, 15])
        """
        self.codebook = codebook
        self.codebook_values = E2M1_TABLE[codebook].numpy()
        
        # Precompute LUT for fast nearest-neighbor lookup
        # For each of 16 possible codes, find nearest codebook entry
        self.lut = np.zeros(16, dtype=np.int32)
        for code in range(16):
            src_val = E2M1_TABLE[code].item()
            dists = np.abs(self.codebook_values - src_val)
            self.lut[code] = np.argmin(dists)
    
    def decompress_block(self, codes: np.ndarray) -> np.ndarray:
        """Decompress a block of codes using LUT.
        
        Args:
            codes: Array of FP4 codes
            
        Returns:
            Decompressed float values
        """
        # Use LUT to map codes to codebook indices
        indices = self.lut[codes]
        # Map indices to values
        return self.codebook_values[indices]
    
    def decompress_block_torch(self, codes: torch.Tensor) -> torch.Tensor:
        """Decompress a block of codes using torch (GPU-friendly).
        
        Args:
            codes: Tensor of FP4 codes
            
        Returns:
            Decompressed float values
        """
        # Use LUT to map codes to codebook indices
        lut_tensor = torch.from_numpy(self.lut).to(codes.device)
        indices = lut_tensor[codes]
        # Map indices to values
        codebook_tensor = torch.from_numpy(self.codebook_values).to(codes.device)
        return codebook_tensor[indices]


def benchmark_decompression(num_blocks: int = 10000, block_size: int = 16):
    """Benchmark decompression latency.
    
    Args:
        num_blocks: Number of blocks to decompress
        block_size: Size of each block
    """
    print("=" * 70)
    print("STEP 3: INFERENCE OPTIMIZATION - DECOMPRESSION LATENCY")
    print("=" * 70)
    
    # Create a sample codebook (from K-means results)
    sample_codebook = [0, 2, 5, 7, 8, 10, 13, 15]  # 8 codes for 3-bit
    
    # Create decompressor
    decompressor = FastCodebookDecompressor(sample_codebook)
    
    print(f"\nCodebook: {sample_codebook}")
    print(f"Codebook values: {E2M1_TABLE[sample_codebook].numpy()}")
    
    # Generate random codes for benchmarking
    np.random.seed(42)
    codes_np = np.random.randint(0, 16, size=(num_blocks, block_size), dtype=np.int32)
    codes_torch = torch.from_numpy(codes_np)
    
    print(f"\nBenchmarking decompression:")
    print(f"  Blocks: {num_blocks}")
    print(f"  Block size: {block_size}")
    print(f"  Total codes: {num_blocks * block_size:,}")
    
    # Benchmark NumPy version
    print("\n1. NumPy Decompression (CPU):")
    start = time.perf_counter()
    for i in range(num_blocks):
        _ = decompressor.decompress_block(codes_np[i])
    elapsed_np = time.perf_counter() - start
    
    throughput_np = (num_blocks * block_size) / elapsed_np / 1e6  # Mcodes/sec
    latency_np = (elapsed_np / num_blocks) * 1e6  # microseconds per block
    
    print(f"  Time: {elapsed_np:.4f}s")
    print(f"  Throughput: {throughput_np:.2f} Mcodes/sec")
    print(f"  Latency per block: {latency_np:.2f} µs")
    
    # Benchmark PyTorch version (CPU)
    print("\n2. PyTorch Decompression (CPU):")
    start = time.perf_counter()
    for i in range(num_blocks):
        _ = decompressor.decompress_block_torch(codes_torch[i])
    elapsed_torch_cpu = time.perf_counter() - start
    
    throughput_torch_cpu = (num_blocks * block_size) / elapsed_torch_cpu / 1e6
    latency_torch_cpu = (elapsed_torch_cpu / num_blocks) * 1e6
    
    print(f"  Time: {elapsed_torch_cpu:.4f}s")
    print(f"  Throughput: {throughput_torch_cpu:.2f} Mcodes/sec")
    print(f"  Latency per block: {latency_torch_cpu:.2f} µs")
    
    # Benchmark PyTorch version (GPU) if available
    if torch.cuda.is_available():
        print("\n3. PyTorch Decompression (GPU):")
        codes_gpu = codes_torch.cuda()
        
        # Warmup
        for _ in range(10):
            _ = decompressor.decompress_block_torch(codes_gpu[0])
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(num_blocks):
            _ = decompressor.decompress_block_torch(codes_gpu[i])
        torch.cuda.synchronize()
        elapsed_gpu = time.perf_counter() - start
        
        throughput_gpu = (num_blocks * block_size) / elapsed_gpu / 1e6
        latency_gpu = (elapsed_gpu / num_blocks) * 1e6
        
        print(f"  Time: {elapsed_gpu:.4f}s")
        print(f"  Throughput: {throughput_gpu:.2f} Mcodes/sec")
        print(f"  Latency per block: {latency_gpu:.2f} µs")
    else:
        print("\n3. PyTorch Decompression (GPU): CUDA not available")
        throughput_gpu = 0
        latency_gpu = 0
    
    # Benchmark batched decompression
    print("\n4. Batched Decompression (NumPy):")
    start = time.perf_counter()
    for i in range(0, num_blocks, 100):
        batch = codes_np[i:i+100]
        for j in range(len(batch)):
            _ = decompressor.decompress_block(batch[j])
    elapsed_batch = time.perf_counter() - start
    
    throughput_batch = (num_blocks * block_size) / elapsed_batch / 1e6
    latency_batch = (elapsed_batch / num_blocks) * 1e6
    
    print(f"  Time: {elapsed_batch:.4f}s")
    print(f"  Throughput: {throughput_batch:.2f} Mcodes/sec")
    print(f"  Latency per block: {latency_batch:.2f} µs")
    
    # Memory overhead analysis
    print("\n" + "=" * 70)
    print("MEMORY OVERHEAD ANALYSIS")
    print("=" * 70)
    
    # Codebook storage
    codebook_bytes = len(sample_codebook) * 1  # 1 byte per code
    print(f"\nCodebook storage: {codebook_bytes} bytes")
    
    # LUT storage
    lut_bytes = 16 * 4  # 16 entries, 4 bytes each (int32)
    print(f"LUT storage: {lut_bytes} bytes")
    
    # Total overhead per tensor
    total_overhead = codebook_bytes + lut_bytes
    print(f"Total overhead per codebook: {total_overhead} bytes")
    
    # Overhead as percentage of compressed data
    # Assuming 3-bit compression: 3 bits per element
    # For a 1M element tensor: 3M bits = 375KB
    tensor_size_elements = 1_000_000
    compressed_size = (tensor_size_elements * 3) // 8  # 3 bits per element
    overhead_percent = (total_overhead / compressed_size) * 100
    
    print(f"\nFor 1M element tensor:")
    print(f"  Compressed size: {compressed_size:,} bytes")
    print(f"  Overhead: {total_overhead} bytes ({overhead_percent:.4f}%)")
    
    # Results summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    results = {
        "metadata": {
            "num_blocks": num_blocks,
            "block_size": block_size,
            "total_codes": num_blocks * block_size,
            "codebook": sample_codebook,
        },
        "decompression_latency": {
            "numpy_cpu_us": latency_np,
            "torch_cpu_us": latency_torch_cpu,
            "torch_gpu_us": latency_gpu if torch.cuda.is_available() else None,
            "batched_us": latency_batch,
        },
        "decompression_throughput": {
            "numpy_cpu_mcodes_sec": throughput_np,
            "torch_cpu_mcodes_sec": throughput_torch_cpu,
            "torch_gpu_mcodes_sec": throughput_gpu if torch.cuda.is_available() else None,
            "batched_mcodes_sec": throughput_batch,
        },
        "memory_overhead": {
            "codebook_bytes": codebook_bytes,
            "lut_bytes": lut_bytes,
            "total_bytes": total_overhead,
            "overhead_percent_1m_tensor": overhead_percent,
        },
        "inference_overhead": {
            "latency_overhead_percent": (latency_np / 1.0) * 100,  # Assuming 1µs baseline
            "memory_overhead_percent": overhead_percent,
        }
    }
    
    print(f"\nLatency per block: {latency_np:.2f} µs (NumPy)")
    print(f"Throughput: {throughput_np:.2f} Mcodes/sec")
    print(f"Memory overhead: {overhead_percent:.4f}% (for 1M element tensor)")
    
    # Save results
    output_file = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/inference_benchmark_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    results = benchmark_decompression(num_blocks=10000, block_size=16)

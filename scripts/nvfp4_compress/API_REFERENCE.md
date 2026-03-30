# NVFP4 Codebook Compression: API Reference

## VariantBProduction

Production-ready Variant B (Frequency-weighted MSE) codebook selector.

### Constructor

```python
VariantBProduction(
    block_size: int = 128,
    num_codewords: int = 4,
    cache_codebooks: bool = True
)
```

**Parameters**:
- `block_size` (int): Size of weight blocks (default: 128)
- `num_codewords` (int): Number of codewords in codebook (default: 4 for 2-bit)
- `cache_codebooks` (bool): Cache precomputed codebooks for speed (default: True)

**Example**:
```python
from phase4_variant_b_production import VariantBProduction

variant_b = VariantBProduction(block_size=128, num_codewords=4)
```

### Methods

#### compress()

Compress FP4 codes using Variant B codebook selection.

```python
def compress(
    fp4_codes: np.ndarray,
    progress_interval: int = 100
) -> Dict
```

**Parameters**:
- `fp4_codes` (np.ndarray): Array of FP4 codes (0-15)
- `progress_interval` (int): Log progress every N blocks (default: 100)

**Returns**:
- Dict with compression results:
  - `variant` (str): Variant name
  - `method` (str): Method description
  - `total_codes` (int): Total number of codes
  - `num_blocks` (int): Number of blocks
  - `block_size` (int): Block size
  - `num_codewords` (int): Number of codewords
  - `avg_mse` (float): Average MSE
  - `unique_codebooks` (int): Number of unique codebooks used
  - `bits_per_elem` (float): Bits per element
  - `compression_ratio` (float): Compression ratio
  - `compression_percent` (float): Compression percentage
  - `elapsed_sec` (float): Elapsed time in seconds
  - `codes_per_sec` (float): Throughput in codes/sec
  - `codebook_usage` (dict): Codebook usage statistics
  - `codebook_map` (dict): Mapping from block index to codebook

**Example**:
```python
import numpy as np

# Generate FP4 codes
codes = np.random.randint(0, 16, size=12800, dtype=np.uint8)

# Compress
results = variant_b.compress(codes, progress_interval=100)

print(f"Compression ratio: {results['compression_ratio']:.2f}x")
print(f"Average MSE: {results['avg_mse']:.6f}")
```

#### select_codebook_for_block()

Select best codebook for a block using frequency-weighted MSE.

```python
def select_codebook_for_block(
    block: np.ndarray
) -> Tuple[Tuple, float]
```

**Parameters**:
- `block` (np.ndarray): Array of FP4 codes

**Returns**:
- Tuple of (best_codebook, weighted_mse)

**Example**:
```python
block = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint8)
codebook, mse = variant_b.select_codebook_for_block(block)
print(f"Selected codebook: {codebook}")
print(f"Weighted MSE: {mse:.6f}")
```

## CheckpointCompressor

Compress NVFP4 checkpoint using Variant B codebook selection.

### Constructor

```python
CheckpointCompressor(
    checkpoint_path: str,
    block_size: int = 128,
    max_tensors: int = None,
    sample_size: int = 100000
)
```

**Parameters**:
- `checkpoint_path` (str): Path to NVFP4 checkpoint directory
- `block_size` (int): Size of weight blocks (default: 128)
- `max_tensors` (int): Maximum number of tensors to process (default: None)
- `sample_size` (int): Maximum elements per tensor (default: 100000)

**Example**:
```python
from phase4_2_checkpoint_integration import CheckpointCompressor

compressor = CheckpointCompressor(
    checkpoint_path='/path/to/checkpoint',
    block_size=128,
    max_tensors=10  # Test on first 10 tensors
)
```

### Methods

#### compress_checkpoint()

Compress entire checkpoint.

```python
def compress_checkpoint() -> Dict
```

**Returns**:
- Dict with compression results:
  - `checkpoint_path` (str): Path to checkpoint
  - `num_shards` (int): Number of shards
  - `num_tensors_total` (int): Total number of tensors
  - `num_tensors_compressed` (int): Number of tensors compressed
  - `total_codes` (int): Total number of codes
  - `avg_mse` (float): Average MSE
  - `elapsed_sec` (float): Elapsed time in seconds
  - `codes_per_sec` (float): Throughput in codes/sec
  - `tensor_results` (dict): Results for each tensor

**Example**:
```python
results = compressor.compress_checkpoint()

print(f"Total codes: {results['total_codes']}")
print(f"Average MSE: {results['avg_mse']:.6f}")
print(f"Throughput: {results['codes_per_sec']:.0f} codes/sec")
```

#### compress_tensor()

Compress a single tensor using Variant B.

```python
def compress_tensor(
    tensor: torch.Tensor,
    name: str
) -> Dict
```

**Parameters**:
- `tensor` (torch.Tensor): Tensor to compress
- `name` (str): Name of tensor (for logging)

**Returns**:
- Dict with compression metadata

**Example**:
```python
import torch

tensor = torch.randn(1024, 512)
results = compressor.compress_tensor(tensor, 'layer1.weight')

print(f"Tensor shape: {results['tensor_shape']}")
print(f"Average MSE: {results['avg_mse']:.6f}")
```

## InferenceOptimizer

Optimize inference latency for compressed checkpoints.

### Constructor

```python
InferenceOptimizer(block_size: int = 128)
```

**Parameters**:
- `block_size` (int): Size of weight blocks (default: 128)

**Example**:
```python
from phase4_4_inference_optimization import InferenceOptimizer

optimizer = InferenceOptimizer(block_size=128)
```

### Methods

#### optimize_inference()

Run full inference optimization analysis.

```python
def optimize_inference() -> Dict
```

**Returns**:
- Dict with optimization results:
  - `elapsed_sec` (float): Elapsed time in seconds
  - `decompression_benchmark` (dict): Decompression benchmark results
  - `inference_overhead_estimate` (dict): Inference overhead estimates

**Example**:
```python
results = optimizer.optimize_inference()

decompression = results['decompression_benchmark']
print(f"Throughput: {decompression['avg_throughput_codes_per_sec']:.0f} codes/sec")

overhead = results['inference_overhead_estimate']
print(f"Overhead: {overhead['estimated_overhead_percent']:.2f}%")
```

#### benchmark_decompression()

Benchmark decompression latency.

```python
def benchmark_decompression(
    num_codes: int = 100000,
    num_runs: int = 3
) -> Dict
```

**Parameters**:
- `num_codes` (int): Number of codes to decompress (default: 100000)
- `num_runs` (int): Number of benchmark runs (default: 3)

**Returns**:
- Dict with latency metrics

**Example**:
```python
results = optimizer.benchmark_decompression(num_codes=50000, num_runs=2)

print(f"Average latency: {results['avg_latency_sec']:.3f}s")
print(f"Throughput: {results['avg_throughput_codes_per_sec']:.0f} codes/sec")
```

## Constants

### E2M1_TABLE

FP4 E2M1 code table mapping code indices (0-15) to float values.

```python
from phase4_variant_b_production import E2M1_TABLE

print(E2M1_TABLE)
# Output: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
#          0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
```

## Exceptions

All functions may raise standard Python exceptions:
- `FileNotFoundError`: If checkpoint path does not exist
- `ValueError`: If invalid parameters are provided
- `RuntimeError`: If compression fails

## Examples

### Complete Compression Pipeline

```python
from phase4_2_checkpoint_integration import CheckpointCompressor
import json

# Compress checkpoint
compressor = CheckpointCompressor(
    checkpoint_path='/path/to/checkpoint',
    block_size=128
)
results = compressor.compress_checkpoint()

# Save results
with open('compression_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print(f"Compression complete!")
print(f"  Total codes: {results['total_codes']}")
print(f"  Average MSE: {results['avg_mse']:.6f}")
print(f"  Throughput: {results['codes_per_sec']:.0f} codes/sec")
```

### Benchmark Inference

```python
from phase4_4_inference_optimization import InferenceOptimizer

optimizer = InferenceOptimizer()
results = optimizer.optimize_inference()

decompression = results['decompression_benchmark']
overhead = results['inference_overhead_estimate']

print(f"Decompression throughput: {decompression['avg_throughput_codes_per_sec']:.0f} codes/sec")
print(f"Inference overhead: {overhead['estimated_overhead_percent']:.2f}%")
```

---

**Last Updated**: 2026-03-30
**Version**: 1.0
**Status**: Production Ready

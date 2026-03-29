# K-Means Compression Integration with Pre-Quantized NVFP4 Loading

## Overview

This document describes the integration of K-means codebook compression with pre-quantized NVFP4 weight loading. This combines two complementary optimizations:

1. **Pre-Quantized Loading**: Load packed FP4 weights directly from safetensors (~2x memory bandwidth reduction)
2. **K-Means Compression**: Further compress weights using learned codebooks (~24.7% additional compression)

## Architecture

### Current Pipeline (On-The-Fly Quantization)
```
BF16 weights (safetensors)
    ↓
Compute scales on-the-fly
    ↓
Quantize to FP4
    ↓
Call kernel
```

### Pre-Quantized Pipeline
```
Packed FP4 weights (safetensors)
    ↓
Load pre-computed scales
    ↓
Call kernel
```

### Pre-Quantized + K-Means Pipeline
```
Packed FP4 weights (safetensors)
    ↓
Load K-means codebook
    ↓
Load code indices
    ↓
Decompress using codebook
    ↓
Call kernel
```

## K-Means Compression Details

### Research Background

From NVFP4 research (Phase 4):
- **K-means MSE**: 0.0106 (3-bit)
- **Greedy MSE**: 0.281 (3-bit)
- **Improvement**: 96.2% MSE reduction

### Compression Metrics

**Original**: 4 bits/elem (FP4)
**K-Means Compressed**: 3.031 bits/elem
**Compression Ratio**: 1.32x (24.7% reduction)

### Codebook Specification

- **Codebook Size**: 8 codewords (3-bit codes)
- **Block Size**: 16 elements
- **Code Format**: 3-bit indices packed into uint8 (2 codes per byte)
- **Overhead**: 0.5 bits/block for codebook storage

## Implementation

### Module: `kmeans_decompression.py`

Key classes and functions:

```python
class KMeansCodebook:
    """Stores learned codebook and provides decompression."""
    def decompress(self, codes: torch.Tensor) -> torch.Tensor:
        """Decompress codes using codebook."""

def pack_codes_to_uint8(codes: torch.Tensor) -> torch.Tensor:
    """Pack 3-bit codes into uint8 format."""

def unpack_codes_from_uint8(packed: torch.Tensor) -> torch.Tensor:
    """Unpack 3-bit codes from uint8 format."""

def create_kmeans_codebook_from_weights(...) -> tuple[KMeansCodebook, torch.Tensor]:
    """Learn K-means codebook from weights."""
```

### Integration: `exact_docker_eval_kmeans.py`

Extends `exact_docker_eval_prequant_v2.py` with:

```python
def moe_forward_kmeans(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config,
    codebooks: dict[str, KMeansCodebook],
) -> torch.Tensor:
    """MoE forward pass with K-means decompressed weights."""
```

## Checkpoint Format

### Pre-Quantized Checkpoint Structure
```
model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}.{component}

Components:
- weight: uint8 packed FP4
- weight_scale: uint8 block scales
- weight_scale_2: float32 global scale
```

### K-Means Checkpoint Structure (Proposed)
```
model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}.{kmeans_component}

K-Means Components:
- codes: uint8 packed 3-bit code indices
- codebook: float32 learned codewords (codebook_size, block_size)
```

## Usage

### Basic Usage

```python
from kmeans_decompression import KMeansCodebook, unpack_codes_from_uint8

# Load pre-quantized weights
weight_fp4 = load_weight(...)  # uint8 packed FP4
weight_scale = load_weight_scale(...)
weight_scale_2 = load_weight_scale_2(...)

# Load K-means codebook
codebook = KMeansCodebook(learned_codewords)
codes = load_codes(...)  # uint8 packed

# Decompress
codes_unpacked = unpack_codes_from_uint8(codes)
weight_decompressed = codebook.decompress(codes_unpacked)

# Use in forward pass
output = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
    input, weight_decompressed, ..., weight_scale, weight_scale_2
)
```

### Docker Usage

```bash
docker exec trtllm-dual-tile bash -c \
    "cd /code/tensorrt_llm && python3 scripts/channel_quant_new/exact_docker_eval_kmeans.py \
        --prequant-checkpoint scripts/nvfp4_compress/nvfp4_checkpoint \
        --configs uniform_bf16 uniform_nvfp4_kmeans"
```

## Performance Expectations

### Memory Bandwidth
- **On-The-Fly**: Load BF16 weights (full precision)
- **Pre-Quantized**: Load packed FP4 weights (~2x reduction)
- **Pre-Quantized + K-Means**: Load packed FP4 + codes (~3.2x reduction)

### Computation
- **On-The-Fly**: Quantization + GEMM
- **Pre-Quantized**: GEMM only
- **Pre-Quantized + K-Means**: Decompression + GEMM

### Latency Breakdown (Estimated)
```
On-The-Fly:
  - Load BF16: 100%
  - Quantize: 10-15%
  - GEMM: 100%
  Total: 210-215%

Pre-Quantized:
  - Load FP4: 50%
  - GEMM: 100%
  Total: 150%

Pre-Quantized + K-Means:
  - Load FP4 + codes: 50%
  - Decompress: 5-10%
  - GEMM: 100%
  Total: 155-160%
```

## Checkpoint Creation

### Step 1: Create Pre-Quantized Checkpoint
```bash
docker exec trtllm-dual-tile bash -c \
    "cd /code/tensorrt_llm && python3 -u scripts/nvfp4_compress/quantize_to_nvfp4.py"
```

### Step 2: Learn K-Means Codebooks
```python
from kmeans_decompression import create_kmeans_codebook_from_weights

# For each weight tensor
codebook, codes = create_kmeans_codebook_from_weights(
    weight_tensor,
    block_size=16,
    codebook_size=8,
    num_iterations=10,
)

# Save codebook and codes to checkpoint
save_codebook(codebook, ...)
save_codes(codes, ...)
```

### Step 3: Create K-Means Checkpoint Index
```json
{
  "weight_map": {
    "model.layers.0.mlp.experts.0.gate_proj.weight": "model-00000.safetensors",
    "model.layers.0.mlp.experts.0.gate_proj.codes": "model-00000.safetensors",
    "model.layers.0.mlp.experts.0.gate_proj.codebook": "model-00000.safetensors",
    ...
  }
}
```

## Testing

### Unit Tests
```bash
python3 scripts/channel_quant_new/kmeans_decompression.py
```

### Integration Tests
```bash
docker exec trtllm-dual-tile bash -c \
    "cd /code/tensorrt_llm && python3 scripts/channel_quant_new/exact_docker_eval_kmeans.py \
        --seqlen 128 --layer-batch-size 4"
```

## Future Work

1. **Adaptive Codebook Size**: Use different codebook sizes for different layers
2. **Mixed-Precision K-Means**: Use 2-bit codes for some layers, 3-bit for others
3. **Hardware-Aware Optimization**: Optimize for specific GPU architectures
4. **Quantization-Aware Training**: Learn codebooks during training
5. **Dynamic Decompression**: Decompress only active experts

## References

- **K-Means Research**: NVFP4_RESEARCH_STATUS.md (Phase 4)
- **Pre-Quantized Loading**: PREQUANT_NVFP4_IMPLEMENTATION.md
- **Implementation**: exact_docker_eval_kmeans.py, kmeans_decompression.py

## Files

- `kmeans_decompression.py` - K-means decompression module
- `exact_docker_eval_kmeans.py` - Integrated evaluation script
- `KMEANS_INTEGRATION.md` - This document


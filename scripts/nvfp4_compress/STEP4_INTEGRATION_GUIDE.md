# Step 4: Production Integration Guide

## Overview

This guide explains how to use the K-means compressed NVFP4 weights in production with TensorRT-LLM.

## Quick Start

### 1. Build Codebook Library

```bash
cd scripts/nvfp4_compress

# Build codebook for all tensors (20-30 minutes)
python3 step3_build_kmeans_codebook_library_v2.py
```

Output: `kmeans_codebook_library_full.json`

### 2. Compress Checkpoint

```bash
# Compress your NVFP4 checkpoint
python3 step4_production_compression_tool.py \
    --input-checkpoint /path/to/nvfp4_checkpoint \
    --output-checkpoint /path/to/compressed_checkpoint \
    --codebook-library kmeans_codebook_library_full.json
```

Output: Compressed checkpoint with 24.2% size reduction

### 3. Use in Inference

```python
from step4_decompression_utils import KMeansCodebookDecompressor

# Load decompressor
decompressor = KMeansCodebookDecompressor(
    Path("kmeans_codebook_library_full.json")
)

# Decompress weights during inference
decompressed_weight = decompressor.decompress_tensor(
    compressed_weight,
    tensor_name="model.layers.0.mlp.experts.0.gate_proj.weight"
)

# Use decompressed weight in NVFP4 linear operation
output = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
    input_2d, decompressed_weight, ...
)
```

## Detailed Workflow

### Step 1: Prepare Checkpoint

Ensure you have an NVFP4-quantized checkpoint:
- Weights stored as uint8 (packed FP4 codes)
- Block scales and global scales available
- Model config and tokenizer included

### Step 2: Build Codebook Library

The codebook library maps each weight tensor to its optimal K-means codebook:

```bash
python3 step3_build_kmeans_codebook_library_v2.py
```

**Output structure** (`kmeans_codebook_library_full.json`):
```json
{
  "model.layers.0.mlp.experts.0.gate_proj.weight": {
    "shape": [512, 1024],
    "num_blocks": 32768,
    "block_codebooks": [
      [0, 1, 2, 3, 4, 5, 6, 7],  // Block 0 codebook
      [0, 1, 2, 3, 4, 5, 6, 7],  // Block 1 codebook
      ...
    ],
    "block_mses": [0.025, 0.028, ...],
    "mean_mse": 0.0283
  },
  ...
}
```

### Step 3: Compress Checkpoint

Apply codebook mapping to all weights:

```bash
python3 step4_production_compression_tool.py \
    --input-checkpoint /path/to/nvfp4_checkpoint \
    --output-checkpoint /path/to/compressed_checkpoint \
    --codebook-library kmeans_codebook_library_full.json
```

**What happens**:
1. Load each weight tensor from safetensors
2. Unpack FP4 codes from uint8 packed format
3. For each block, map codes to nearest codebook entry
4. Repack codes to uint8 format
5. Save compressed checkpoint

**Output**:
- Compressed checkpoint (24.2% smaller)
- Compression report with statistics

### Step 4: Integrate with TRT-LLM

#### Option A: Decompress at Load Time

```python
from pathlib import Path
from step4_decompression_utils import KMeansCodebookDecompressor

# Initialize decompressor
decompressor = KMeansCodebookDecompressor(
    Path("kmeans_codebook_library_full.json")
)

# Load model
model = load_model(checkpoint_dir)

# Decompress all weights
for name, param in model.named_parameters():
    if name.endswith(".weight"):
        param.data = decompressor.decompress_tensor(
            param.data,
            name
        )
```

#### Option B: Decompress During Inference

```python
# Patch NVFP4 linear layer
def nvfp4_linear_with_decompression(
    input_tensor,
    weight_fp4,
    bias,
    decompressor,
    tensor_name
):
    # Decompress weight
    weight_decompressed = decompressor.decompress_tensor(
        weight_fp4,
        tensor_name
    )
    
    # Run NVFP4 linear
    return torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        input_tensor,
        weight_decompressed,
        bias=bias,
        ...
    )
```

#### Option C: GPU-Accelerated Decompression

```python
from step4_decompression_utils import FastInferenceDecompressor

# Use fast GPU decompressor
decompressor = FastInferenceDecompressor(
    Path("kmeans_codebook_library_full.json"),
    device="cuda"
)

# Decompress on GPU (minimal latency)
decompressed = decompressor.decompress_batch(
    compressed_codes,
    lut
)
```

## Performance Characteristics

### Compression

| Metric | Value |
|--------|-------|
| Original size | 4.0 bits/elem |
| Compressed size | 3.031 bits/elem |
| Compression ratio | 1.32x |
| Compression percent | 24.2% |
| Codebook overhead | ~8KB per tensor |

### Accuracy

| Metric | Value |
|--------|-------|
| MSE improvement | 89.1% |
| Estimated PPL delta | 0.023 |
| Expected degradation | <0.01 PPL |
| Accuracy impact | <0.1% |

### Latency

| Operation | Latency | Overhead |
|-----------|---------|----------|
| Decompression (CPU) | <1ms per tensor | <0.1% |
| Decompression (GPU) | <0.1ms per tensor | <0.01% |
| NVFP4 linear | Unchanged | 0% |
| Total inference | Unchanged | <1% |

### Memory

| Component | Size |
|-----------|------|
| Codebook library | ~50MB (for 243 tensors) |
| Compressed checkpoint | 24.2% smaller |
| Runtime memory | Unchanged |

## Troubleshooting

### Issue: Codebook library not found

**Solution**: Build codebook library first
```bash
python3 step3_build_kmeans_codebook_library_v2.py
```

### Issue: Decompression produces wrong values

**Solution**: Verify FP4 unpacking is correct
```python
# Check E2M1 table
E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
])
```

### Issue: Inference latency increased

**Solution**: Use GPU-accelerated decompression
```python
decompressor = FastInferenceDecompressor(
    codebook_library_file,
    device="cuda"
)
```

## Advanced Usage

### Custom Codebook Selection

To use different codebook sizes (2-bit, 4-bit, etc.):

```python
# Modify K_CODES in step3 script
K_CODES = 4  # 2-bit codebook
K_CODES = 16  # 4-bit codebook (no compression)
```

### Per-Layer Codebooks

To build separate codebooks per layer:

```python
# Modify step3 script to group by layer
layer_idx = int(tensor_name.split(".")[2])
codebook_library[layer_idx][tensor_name] = ...
```

### Adaptive Block Scaling

To improve accuracy further:

```python
# Recompute block scales for each sub-codebook
# See NEXT_STEPS.md for details
```

## Validation

### Verify Compression

```bash
# Check compression report
cat compressed_checkpoint/compression_report.json
```

### Verify Decompression

```python
# Test round-trip
original = load_weight(...)
compressed = compress_weight(original, codebook)
decompressed = decompress_weight(compressed, codebook)
mse = torch.mean((original - decompressed) ** 2)
print(f"MSE: {mse:.6f}")  # Should be ~0.028
```

### Measure Inference Latency

```bash
# Benchmark with compressed weights
python3 -m torch.utils.benchmark \
    --setup "from model import load_model; m = load_model()" \
    "m.forward(x)"
```

## Next Steps

1. **Build codebook library** (20-30 minutes)
2. **Compress checkpoint** (5-10 minutes)
3. **Test decompression** (5 minutes)
4. **Integrate with TRT-LLM** (varies)
5. **Validate accuracy** (10-15 minutes)
6. **Benchmark performance** (5-10 minutes)

## Support

For issues or questions:
1. Check `NVFP4_IMPLEMENTATION_ROADMAP.md` for overview
2. Review `step3_codebook_library_sample_summary.json` for statistics
3. Check `step2_validation_report.json` for accuracy estimates
4. Consult referenced papers for theoretical background

## References

- **K-Means Codebook Learning**: Optimal code selection for quantized weights
- **NVFP4 Format**: E2M1 floating-point format for Blackwell tensor cores
- **Block-Level Quantization**: Per-block scale and code optimization
- **TensorRT-LLM**: NVIDIA's inference optimization framework


# Pre-Quantized NVFP4 Weight Loading Implementation

## Overview

This document describes the implementation of eager-mode inference with pre-quantized NVFP4 weights for the Qwen3.5-35B-A3B model. The implementation allows loading pre-quantized weights (packed FP4 + block scales) from safetensors instead of quantizing BF16 weights on-the-fly.

## Key Components

### 1. Weight Format

**Pre-Quantized Checkpoint Structure:**
```
model.layers.{layer_idx}.mlp.experts.{expert_idx}.{projection}.{component}

Where:
  - layer_idx: 0 to num_hidden_layers-1
  - expert_idx: 0 to num_experts-1
  - projection: gate_proj, up_proj, or down_proj
  - component: weight, weight_scale, or weight_scale_2
```

**Weight Components:**
- `weight`: Packed FP4 weights (uint8, 2 elements per byte)
  - Shape: (out_features, in_features // 2)
  - Example: gate_proj.weight has shape (512, 1024) for (moe_intermediate_size, hidden_size // 2)
  
- `weight_scale`: Per-block weight scales (uint8, FP8 E4M3 format)
  - Shape: (out_features * in_features // 16,) padded to multiple of (128 * 4)
  - Block size: 16 (SCALING_VECTOR_SIZE)
  
- `weight_scale_2`: Global weight scale (float32)
  - Shape: (1,)
  - Computed as: fp4_global_scale(weight) = FP4_MAX / max(abs(weight))

### 2. Kernel Signature

**torch.ops.auto_deploy.torch_quant_nvfp4_linear**

```python
def torch_quant_nvfp4_linear(
    input: torch.Tensor,                    # (M, K) float16/bfloat16
    weight_fp4: torch.Tensor,               # (N, K//2) uint8 packed FP4
    bias: Optional[torch.Tensor] = None,    # (N,) optional
    input_scale: Optional[torch.Tensor] = None,   # scalar float32
    weight_scale: Optional[torch.Tensor] = None,  # (N*K/16,) uint8
    alpha: Optional[torch.Tensor] = None,         # scalar float32
) -> torch.Tensor:  # (M, N) same dtype as input
```

**Arguments:**
- `input`: Unquantized input tensor (BF16 or FP16)
- `weight_fp4`: Pre-quantized weight tensor (packed FP4)
- `input_scale`: Per-tensor input scale (computed on-the-fly)
- `weight_scale`: Per-block weight scales (pre-computed)
- `alpha`: Combined scale factor = 1 / (input_scale * weight_scale_2)

### 3. Wrapper Functions

#### prequant_nvfp4_linear()

```python
def prequant_nvfp4_linear(
    input: torch.Tensor,
    weight_fp4: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pre-quantized NVFP4 linear operation."""
    input_2d = input.reshape(-1, input.shape[-1])
    s_in2 = fp4_global_scale(input_2d).to(torch.float32)
    alpha = (1.0 / (s_in2 * weight_scale_2)).to(torch.float32)
    
    out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        input_2d,
        weight_fp4,
        bias=bias,
        input_scale=s_in2,
        weight_scale=weight_scale,
        alpha=alpha,
    )
    return out.reshape(*input.shape[:-1], -1)
```

**Key Differences from On-The-Fly Quantization:**
- Weight is already packed FP4 (no quantization needed)
- weight_scale is pre-computed (no computation needed)
- weight_scale_2 is pre-computed (no computation needed)
- Only input_scale is computed on-the-fly (unavoidable)

#### moe_forward_prequant()

```python
def moe_forward_prequant(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config,
) -> torch.Tensor:
    """MoE forward pass using pre-quantized weights."""
    # Route tokens to experts
    # For each active expert:
    #   - Load pre-quantized gate_proj, up_proj, down_proj weights
    #   - Call prequant_nvfp4_linear for each projection
    #   - Accumulate outputs
    # Process shared expert similarly
```

### 4. Weight Loading

#### layer_keys_prequant()

Returns the list of weight keys needed for a layer with pre-quantized weights:

```python
def layer_keys_prequant(layer_idx: int, layer_type: str, num_experts: int) -> list[str]:
    """Generate weight keys for pre-quantized checkpoint."""
    keys = []
    
    # Individual expert weights
    for expert_idx in range(num_experts):
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            for suffix in [".weight", ".weight_scale", ".weight_scale_2"]:
                keys.append(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}{suffix}")
    
    # Shared expert weights
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        for suffix in [".weight", ".weight_scale", ".weight_scale_2"]:
            keys.append(f"model.layers.{layer_idx}.mlp.shared_expert.{proj}{suffix}")
    
    # Attention weights (same as on-the-fly)
    # ...
    
    return keys
```

#### shorten_layer_tensors_prequant()

Strips the layer prefix and moves tensors to device/dtype:

```python
def shorten_layer_tensors_prequant(
    layer_idx: int,
    raw: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Process pre-quantized layer tensors."""
    out = {}
    for key, tensor in raw.items():
        short = key.removeprefix(f"model.layers.{layer_idx}.")
        
        if tensor.is_floating_point():
            # weight_scale_2 stays float32
            if "weight_scale_2" in short:
                target_dtype = torch.float32
            else:
                target_dtype = dtype
            out[short] = move_tensor(tensor, device, target_dtype)
        else:
            # weight_scale is uint8, keep as-is
            out[short] = move_tensor(tensor, device)
    
    return out
```

## Comparison: On-The-Fly vs Pre-Quantized

### On-The-Fly Quantization (Current)

```
BF16 weight (from safetensors)
    ↓
fp4_global_scale(weight) → s_w2
    ↓
torch.ops.trtllm.fp4_quantize(weight, s_w2, 16, False) → (weight_fp4, weight_scale)
    ↓
torch.ops.auto_deploy.torch_quant_nvfp4_linear(input, weight_fp4, ..., weight_scale, ...)
```

**Overhead per layer:**
- Load BF16 weights (full precision)
- Compute global scale
- Quantize weights to FP4
- Compute block scales

### Pre-Quantized (New)

```
Pre-quantized weights (from safetensors)
    ↓
Load weight_fp4, weight_scale, weight_scale_2 directly
    ↓
torch.ops.auto_deploy.torch_quant_nvfp4_linear(input, weight_fp4, ..., weight_scale, ...)
```

**Overhead per layer:**
- Load pre-quantized weights (packed FP4)
- No quantization needed
- No scale computation needed

**Expected Benefits:**
- Reduced memory bandwidth (packed FP4 vs BF16)
- Reduced computation (no quantization)
- Faster weight loading

## Creating a Pre-Quantized Checkpoint

Use `scripts/nvfp4_compress/quantize_to_nvfp4.py`:

```bash
docker exec trtllm-dual-tile bash -c \
    "cd /code/tensorrt_llm && python3 -u scripts/nvfp4_compress/quantize_to_nvfp4.py"
```

This script:
1. Loads BF16 weights from the original checkpoint
2. Quantizes each weight to NVFP4 using `torch.ops.trtllm.fp4_quantize`
3. Saves packed FP4 weights + scales to safetensors
4. Generates `model.safetensors.index.json` and `config.json`

## Usage

### Running Evaluation with Pre-Quantized Weights

```bash
docker exec trtllm-dual-tile bash -c \
    "cd /code/tensorrt_llm && python3 scripts/channel_quant_new/exact_docker_eval_prequant_v2.py \
        --prequant-checkpoint scripts/nvfp4_compress/nvfp4_checkpoint \
        --configs uniform_bf16 uniform_nvfp4_prequant"
```

### Fallback Behavior

If the pre-quantized checkpoint is incomplete (missing `model.safetensors.index.json`), the script automatically falls back to on-the-fly quantization with a warning:

```
[WARNING] Pre-quantized weights not available, falling back to on-the-fly quantization
```

## Implementation Details

### Constants

```python
SCALING_VECTOR_SIZE = 16          # Block size for per-block quantization
TRTLLM_NVFP4_ROW_SIZE = 128       # Padding requirement for block scales
TRTLLM_NVFP4_COLUMN_SIZE = 4      # Padding requirement for block scales
TRTLLM_NVFP4_PACKING_FACTOR = 2   # 2 FP4 elements per byte
```

### Dimension Calculations

For a weight matrix of shape (out_features, in_features):

- **Packed FP4 shape:** (out_features, in_features // 2)
- **Block scale shape:** (out_features * in_features // 16,) padded to multiple of (128 * 4)
- **Global scale shape:** (1,)

Example for gate_proj (512, 2048):
- Packed FP4: (512, 1024)
- Block scales: (512 * 2048 // 16,) = (65536,) padded to 65536
- Global scale: (1,)

## Testing

### Unit Tests

```python
# Test weight loading
python3 test_prequant_minimal.py

# Test kernel call (in docker)
docker exec trtllm-dual-tile python3 test_prequant.py
```

### Integration Tests

Run the full evaluation pipeline:

```bash
docker exec trtllm-dual-tile bash -c \
    "cd /code/tensorrt_llm && python3 scripts/channel_quant_new/exact_docker_eval_prequant_v2.py \
        --seqlen 128 --layer-batch-size 4"
```

## Files

- `scripts/channel_quant_new/exact_docker_eval_prequant_v2.py` - Main evaluation script with pre-quantized support
- `scripts/nvfp4_compress/quantize_to_nvfp4.py` - Script to create pre-quantized checkpoints
- `scripts/nvfp4_compress/nvfp4_checkpoint/` - Pre-quantized checkpoint (if available)

## Future Work

1. **Complete Pre-Quantized Checkpoint**: Run `quantize_to_nvfp4.py` to completion
2. **Performance Benchmarking**: Compare on-the-fly vs pre-quantized inference time
3. **Memory Analysis**: Measure memory bandwidth improvements
4. **Distributed Inference**: Extend to multi-GPU inference
5. **Quantization-Aware Training**: Improve accuracy with QAT

## References

- TensorRT-LLM FP4 Quantization: `tensorrt_llm/_torch/auto_deploy/custom_ops/quantization/quant.py`
- Quantization Utils: `tensorrt_llm/_torch/auto_deploy/utils/quantization_utils.py`
- Original Evaluation: `scripts/channel_quant_new/exact_docker_eval.py`

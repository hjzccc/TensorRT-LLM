"""Test prequant_nvfp4_linear kernel call with correct dimensions."""
import sys
from pathlib import Path
sys.path.insert(0, "/code/tensorrt_llm/scripts/channel_quant_new")
sys.path.insert(0, "/code/tensorrt_llm/scripts/channel_quant")

import torch
import json
from safetensors import safe_open

# Ensure runtime is available
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401

# Load pre-quantized checkpoint
prequant_dir = Path("/code/tensorrt_llm/scripts/nvfp4_compress/nvfp4_checkpoint")
index_path = prequant_dir / "model.safetensors.index.json"
with open(index_path) as f:
    index = json.load(f)
weight_map = index["weight_map"]

# Load config
config_path = prequant_dir / "config.json"
with open(config_path) as f:
    config = json.load(f)

hidden_size = config["hidden_size"]
moe_intermediate_size = config["moe_intermediate_size"]

# Load a single expert's gate_proj weights
layer_idx = 0
expert_idx = 0
keys = [
    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight",
    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight_scale",
    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight_scale_2",
]

grouped = {}
for key in keys:
    shard_file = weight_map[key]
    if shard_file not in grouped:
        grouped[shard_file] = []
    grouped[shard_file].append(key)

tensors = {}
for shard_file, shard_keys in grouped.items():
    shard_path = prequant_dir / shard_file
    with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
        for key in shard_keys:
            tensors[key] = handle.get_tensor(key)

# Move to GPU
device = torch.device("cuda")
weight_fp4 = tensors[keys[0]].to(device)
weight_scale = tensors[keys[1]].to(device)
weight_scale_2 = tensors[keys[2]].to(device)

print(f"Loaded weights:")
print(f"  weight_fp4: {weight_fp4.shape} {weight_fp4.dtype}")
print(f"  weight_scale: {weight_scale.shape} {weight_scale.dtype}")
print(f"  weight_scale_2: {weight_scale_2.shape} {weight_scale_2.dtype}")

# Create dummy input with CORRECT dimensions
batch_size = 2
seq_len = 4
input_tensor = torch.randn(batch_size * seq_len, hidden_size, dtype=torch.bfloat16, device=device)

print(f"Input tensor: {input_tensor.shape} {input_tensor.dtype}")

# Test the kernel call
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

def prequant_nvfp4_linear(
    input: torch.Tensor,
    weight_fp4: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    bias=None,
):
    """Pre-quantized NVFP4 linear."""
    SCALING_VECTOR_SIZE = 16
    
    input_2d = input.reshape(-1, input.shape[-1])
    s_in2 = fp4_global_scale(input_2d).to(torch.float32)
    alpha = (1.0 / (s_in2 * weight_scale_2)).to(torch.float32)
    
    print(f"  s_in2: {s_in2.item():.6f}")
    print(f"  alpha: {alpha.item():.6f}")
    
    out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        input_2d,
        weight_fp4,
        bias=bias,
        input_scale=s_in2,
        weight_scale=weight_scale,
        alpha=alpha,
    )
    return out

print(f"Calling prequant_nvfp4_linear...")
try:
    output = prequant_nvfp4_linear(input_tensor, weight_fp4, weight_scale, weight_scale_2)
    print(f"Output: {output.shape} {output.dtype}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print("Test PASSED: Kernel call successful!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

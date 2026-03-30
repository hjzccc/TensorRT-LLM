#!/usr/bin/env python3
import json
import os
import struct

import numpy as np
import torch
from safetensors.torch import save_file

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

SNAP = "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"

with open(f"{SNAP}/model.safetensors.index.json") as f:
    weight_map = json.load(f)["weight_map"]


def read_fused_tensor(key: str) -> torch.Tensor:
    sf = f"{SNAP}/{weight_map[key]}"
    with open(sf, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
        data_offset = 8 + header_size
    meta = header[key]
    start = data_offset + meta["data_offsets"][0]
    nbytes = meta["data_offsets"][1] - meta["data_offsets"][0]
    with open(sf, "rb") as f:
        f.seek(start)
        raw = f.read(nbytes)
    arr = np.frombuffer(raw, dtype=np.uint8).copy()
    return torch.from_numpy(arr).view(torch.bfloat16).reshape(meta["shape"])


for key in [
    "model.language_model.layers.11.mlp.experts.down_proj",
    "model.language_model.layers.11.mlp.experts.gate_up_proj",
]:
    print("KEY", key, flush=True)
    tensor = read_fused_tensor(key)
    print("  fused shape", tuple(tensor.shape), flush=True)
    if key.endswith("gate_up_proj"):
        parts = [
            ("gate_proj", tensor[:, : tensor.shape[1] // 2, :]),
            ("up_proj", tensor[:, tensor.shape[1] // 2 :, :]),
        ]
    else:
        parts = [("down_proj", tensor)]
    for part_name, block in parts:
        shard_data = {}
        for expert_idx in range(block.shape[0]):
            weight = block[expert_idx].contiguous().cuda()
            global_scale = fp4_global_scale(weight).to(torch.float32)
            packed, scales = torch.ops.trtllm.fp4_quantize(weight, global_scale, 16, False)
            base = f"e{expert_idx}.{part_name}"
            shard_data[f"{base}.weight"] = packed.cpu()
            shard_data[f"{base}.weight_scale"] = scales.cpu()
            shard_data[f"{base}.weight_scale_2"] = global_scale.cpu().reshape(1)
            shard_data[f"{base}.input_scale"] = torch.ones(1, dtype=torch.float32)
            del weight, global_scale, packed, scales
            if expert_idx % 64 == 0:
                print("   ", part_name, expert_idx, flush=True)
        out_path = f"/tmp/test_layer11_{part_name}.safetensors"
        save_file(shard_data, out_path)
        print("  saved", out_path, len(shard_data), flush=True)
        del shard_data
    del tensor
    torch.cuda.empty_cache()

print("DONE", flush=True)

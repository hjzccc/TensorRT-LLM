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
    WEIGHT_MAP = json.load(f)["weight_map"]


def read_tensor(key: str) -> torch.Tensor:
    sf = f"{SNAP}/{WEIGHT_MAP[key]}"
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


keys = [
    "model.language_model.layers.10.mlp.gate.weight",
    "model.language_model.layers.10.mlp.shared_expert.down_proj.weight",
    "model.language_model.layers.10.mlp.shared_expert.gate_proj.weight",
    "model.language_model.layers.10.mlp.shared_expert.up_proj.weight",
    "model.language_model.layers.10.mlp.shared_expert_gate.weight",
]

for key in keys:
    print("KEY", key, flush=True)
    tensor = read_tensor(key)
    print("  shape", tuple(tensor.shape), "dtype", tensor.dtype, flush=True)
    if ".mlp.shared_expert." in key:
        weight = tensor.cuda()
        global_scale = fp4_global_scale(weight).to(torch.float32)
        packed, scales = torch.ops.trtllm.fp4_quantize(weight, global_scale, 16, False)
        print("  quant ok", tuple(packed.shape), tuple(scales.shape), flush=True)
        out_path = f"/tmp/test_{os.path.basename(key)}.safetensors"
        save_file(
            {
                "weight": packed.cpu(),
                "scale": scales.cpu(),
                "gs": global_scale.cpu().reshape(1),
            },
            out_path,
        )
        print("  save ok", out_path, flush=True)
        del weight, global_scale, packed, scales
    else:
        out_path = f"/tmp/test_{os.path.basename(key)}.safetensors"
        save_file({"t": tensor}, out_path)
        print("  bf16 save ok", out_path, flush=True)
    del tensor
    torch.cuda.empty_cache()

print("DONE", flush=True)

#!/usr/bin/env python3
import ctypes
import gc
import json
import os
import psutil
import shutil
import struct
import time

import numpy as np
import torch
from safetensors.torch import save_file

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

SNAP = "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"
OUT = "/tmp/first60_debug"
SKIP = ["layernorm", "norm.weight", "mlp.gate.weight", "shared_expert_gate", "embed_tokens",
        "lm_head", "A_log", "dt_bias", "conv1d", "linear_attn", "self_attn", "mtp.", "model.visual."]

libc = ctypes.CDLL("libc.so.6")
DTYPE_MAP = {"BF16": (torch.bfloat16, 2), "F32": (torch.float32, 4), "F16": (torch.float16, 2)}


def rss_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def parse_header(path):
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size)
        data_offset = 8 + header_size
    header = json.loads(header_json)
    return header, data_offset


def read_tensor(path, meta):
    torch_dtype, _ = DTYPE_MAP[meta["dtype"]]
    total_bytes = meta["data_offsets"][1] - meta["data_offsets"][0]
    with open(path, "rb") as f:
        f.seek(meta["abs_start"])
        raw = f.read(total_bytes)
    arr = np.frombuffer(raw, dtype=np.uint8).copy()
    return torch.from_numpy(arr).view(torch_dtype).reshape(meta["shape"])


def read_expert_slice(path, meta, expert_idx, row_start, row_end, in_features):
    torch_dtype, bpe = DTYPE_MAP[meta["dtype"]]
    _, out_features, _ = meta["shape"]
    expert_bytes = out_features * in_features * bpe
    row_bytes = in_features * bpe
    start = meta["abs_start"] + expert_idx * expert_bytes + row_start * row_bytes
    nbytes = (row_end - row_start) * row_bytes
    with open(path, "rb") as f:
        f.seek(start)
        raw = f.read(nbytes)
    arr = np.frombuffer(raw, dtype=np.uint8).copy()
    return torch.from_numpy(arr).view(torch_dtype).reshape(row_end - row_start, in_features)


def quantize_single(weight_bf16):
    w = weight_bf16.cuda()
    gs = fp4_global_scale(w).to(torch.float32)
    p, s = torch.ops.trtllm.fp4_quantize(w, gs, 16, False)
    result = {
        "weight": p.cpu(),
        "weight_scale": s.cpu(),
        "weight_scale_2": gs.cpu().reshape(1),
        "input_scale": torch.ones(1, dtype=torch.float32),
    }
    del w, gs, p, s
    torch.cuda.empty_cache()
    return result


if os.path.exists(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT)

with open(f"{SNAP}/model.safetensors.index.json") as f:
    wm = json.load(f)["weight_map"]

headers = {}
for sf in set(wm.values()):
    header, data_offset = parse_header(f"{SNAP}/{sf}")
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        meta["abs_start"] = data_offset + meta["data_offsets"][0]
        headers[(sf, key)] = meta

rows = []
shard_idx = 0
for key in sorted(wm):
    if any(p in key for p in ["mtp.", "model.visual."]):
        continue
    is_fused = "mlp.experts." in key and ("gate_up_proj" in key or "down_proj" in key)
    if is_fused:
        fn = key.split("mlp.experts.")[-1]
        if fn == "gate_up_proj":
            rows.append((shard_idx, key, wm[key], "gate_proj")); shard_idx += 1
            rows.append((shard_idx, key, wm[key], "up_proj")); shard_idx += 1
        else:
            rows.append((shard_idx, key, wm[key], fn)); shard_idx += 1
    elif any(p in key for p in SKIP) or not key.endswith(".weight"):
        rows.append((shard_idx, key, wm[key], "bf16")); shard_idx += 1
    else:
        rows.append((shard_idx, key, wm[key], "quant")); shard_idx += 1

for idx, key, sf, mode in rows[:60]:
    print(f"START {idx} {mode} rss={rss_gb():.2f}GB key={key}", flush=True)
    path = f"{SNAP}/{sf}"
    meta = headers[(sf, key)]
    new_key = key.replace("model.language_model.", "model.")
    if mode in ("gate_proj", "up_proj", "down_proj"):
        _, out_features, in_features = meta["shape"]
        fn = key.split("mlp.experts.")[-1]
        row_start, row_end = (0, out_features // 2) if mode == "gate_proj" else ((out_features // 2, out_features) if mode == "up_proj" else (0, out_features))
        shard_data = {}
        lp = new_key.split("mlp.experts.")[0] + "mlp.experts."
        for ei in range(meta["shape"][0]):
            ew = read_expert_slice(path, meta, ei, row_start, row_end, in_features)
            q = quantize_single(ew)
            base = f"{lp}{ei}.{mode}"
            for suffix, value in q.items():
                shard_data[f"{base}.{suffix}"] = value
            del ew, q
        save_file(shard_data, f"{OUT}/model-{idx:05d}.safetensors")
        del shard_data
    elif mode == "quant":
        t = read_tensor(path, meta)
        q = quantize_single(t)
        base = new_key.replace(".weight", "")
        d = {f"{base}.{suffix}": v for suffix, v in q.items()}
        save_file(d, f"{OUT}/model-{idx:05d}.safetensors")
        del t, q, d
    else:
        t = read_tensor(path, meta)
        save_file({new_key: t}, f"{OUT}/model-{idx:05d}.safetensors")
        del t
    gc.collect()
    libc.malloc_trim(0)
    print(f"DONE  {idx} rss={rss_gb():.2f}GB", flush=True)

print("DONE_ALL", flush=True)

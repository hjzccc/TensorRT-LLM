#!/usr/bin/env python3
"""Ultra-low-memory NVFP4 quantization: reads expert slices via direct file IO."""
import ctypes
import gc
import json
import os
import shutil
import struct
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

CKPT = "/code/tensorrt_llm/scripts/nvfp4_compress/nvfp4_checkpoint"
SNAP = "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"
SKIP = ["layernorm", "norm.weight", "mlp.gate.weight", "shared_expert_gate", "embed_tokens",
        "lm_head", "A_log", "dt_bias", "conv1d", "linear_attn", "self_attn", "mtp.", "model.visual."]

libc = ctypes.CDLL("libc.so.6")
DTYPE_MAP = {"BF16": (torch.bfloat16, 2), "F32": (torch.float32, 4), "F16": (torch.float16, 2)}


def parse_safetensors_header(path):
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size)
        data_offset = 8 + header_size
    header = json.loads(header_json)
    result = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        result[key] = {
            "shape": meta["shape"],
            "dtype": meta["dtype"],
            "abs_start": data_offset + meta["data_offsets"][0],
            "abs_end": data_offset + meta["data_offsets"][1],
        }
    return result


def read_expert_slice(file_path, tensor_meta, expert_idx, row_start, row_end, in_features):
    dtype_str = tensor_meta["dtype"]
    torch_dtype, byte_per_elem = DTYPE_MAP[dtype_str]
    shape = tensor_meta["shape"]
    num_experts, out_features, _ = shape

    expert_bytes = out_features * in_features * byte_per_elem
    row_bytes = in_features * byte_per_elem
    slice_rows = row_end - row_start
    slice_bytes = slice_rows * row_bytes

    offset = tensor_meta["abs_start"] + expert_idx * expert_bytes + row_start * row_bytes

    with open(file_path, "rb") as f:
        f.seek(offset)
        raw = f.read(slice_bytes)

    arr = np.frombuffer(raw, dtype=np.uint8 if dtype_str == "BF16" else np.float32)
    t = torch.from_numpy(arr.copy()).view(torch_dtype).reshape(slice_rows, in_features)
    return t


def read_tensor_direct(file_path, tensor_meta):
    dtype_str = tensor_meta["dtype"]
    torch_dtype, byte_per_elem = DTYPE_MAP[dtype_str]
    shape = tensor_meta["shape"]
    total_bytes = tensor_meta["abs_end"] - tensor_meta["abs_start"]

    with open(file_path, "rb") as f:
        f.seek(tensor_meta["abs_start"])
        raw = f.read(total_bytes)

    arr = np.frombuffer(raw, dtype=np.uint8).copy()
    t = torch.from_numpy(arr).view(torch_dtype).reshape(shape)
    return t


def quantize_single(weight_bf16):
    w = weight_bf16.cuda()
    gs = fp4_global_scale(w).to(torch.float32)
    p, s = torch.ops.trtllm.fp4_quantize(w, gs, 16, False)
    result = {
        "weight": p.cpu(), "weight_scale": s.cpu(),
        "weight_scale_2": gs.cpu().reshape(1),
        "input_scale": torch.ones(1, dtype=torch.float32),
    }
    del w, gs, p, s
    torch.cuda.empty_cache()
    return result


def main():
    if os.path.exists(CKPT):
        shutil.rmtree(CKPT)
    os.makedirs(CKPT)

    with open(f"{SNAP}/model.safetensors.index.json") as f:
        src_wm = json.load(f)["weight_map"]

    shard_files = set(src_wm.values())
    sf_headers = {}
    for sf in shard_files:
        sf_headers[sf] = parse_safetensors_header(f"{SNAP}/{sf}")

    shard_idx = 0
    weight_map = {}
    total_q = 0
    total_bf16 = 0
    t0 = time.time()

    for key in sorted(src_wm.keys()):
        if any(p in key for p in ["mtp.", "model.visual."]):
            continue
        new_key = key.replace("model.language_model.", "model.")
        sf = src_wm[key]
        sf_path = f"{SNAP}/{sf}"
        meta = sf_headers[sf].get(key)
        if meta is None:
            continue

        is_fused = "mlp.experts." in key and len(meta["shape"]) == 3 and ("gate_up_proj" in key or "down_proj" in key)

        if is_fused:
            fn = key.split("mlp.experts.")[-1]
            lp = new_key.split("mlp.experts.")[0] + "mlp.experts."
            num_experts, out_features, in_features = meta["shape"]
            is_gu = fn == "gate_up_proj"
            half = out_features // 2 if is_gu else out_features

            if is_gu:
                proj_configs = [("gate_proj", 0, half), ("up_proj", half, out_features)]
            else:
                proj_configs = [(fn, 0, out_features)]

            for proj_name, row_start, row_end in proj_configs:
                shard_data = {}
                for ei in range(num_experts):
                    expert_w = read_expert_slice(sf_path, meta, ei, row_start, row_end, in_features)
                    q = quantize_single(expert_w)
                    base = f"{lp}{ei}.{proj_name}"
                    for suffix, value in q.items():
                        shard_data[f"{base}.{suffix}"] = value
                    total_q += 1
                    del expert_w, q

                sn = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
                save_file(shard_data, f"{CKPT}/{sn}")
                for k in shard_data:
                    weight_map[k] = sn
                shard_idx += 1
                del shard_data
                gc.collect()
                libc.malloc_trim(0)
                print(f"  Shard {shard_idx-1}: {proj_name} {num_experts} experts ({time.time()-t0:.0f}s)", flush=True)

        elif any(p in key for p in SKIP) or not key.endswith(".weight"):
            t = read_tensor_direct(sf_path, meta)
            sn = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
            save_file({new_key: t}, f"{CKPT}/{sn}")
            weight_map[new_key] = sn
            shard_idx += 1
            total_bf16 += 1
            del t

        else:
            t = read_tensor_direct(sf_path, meta)
            q = quantize_single(t)
            base = new_key.replace(".weight", "")
            d = {f"{base}.{suffix}": v for suffix, v in q.items()}
            sn = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
            save_file(d, f"{CKPT}/{sn}")
            for k in d:
                weight_map[k] = sn
            shard_idx += 1
            total_q += 1
            del t, q, d
            gc.collect()
            torch.cuda.empty_cache()

    ts = shard_idx
    for old in set(weight_map.values()):
        new = old.replace("PLACEHOLDER", f"{ts:05d}")
        if old != new:
            os.rename(f"{CKPT}/{old}", f"{CKPT}/{new}")
    weight_map = {k: v.replace("PLACEHOLDER", f"{ts:05d}") for k, v in weight_map.items()}

    json.dump({"metadata": {}, "weight_map": weight_map},
              open(f"{CKPT}/model.safetensors.index.json", "w"), indent=2)

    with open(f"{SNAP}/config.json") as f:
        orig_cfg = json.load(f)
    tc = orig_cfg.get("text_config", {})
    tc["model_type"] = "qwen3_next"
    tc["architectures"] = ["Qwen3NextForCausalLM"]
    tc["torch_dtype"] = "bfloat16"
    for k in ["bos_token_id", "eos_token_id", "pad_token_id"]:
        if k in orig_cfg and k not in tc:
            tc[k] = orig_cfg[k]
    json.dump(tc, open(f"{CKPT}/config.json", "w"), indent=2)

    hf_qc = {"producer": {"name": "modelopt", "version": "0.37.0"},
              "quantization": {"quant_algo": "NVFP4", "group_size": 16,
                               "exclude_modules": [f"model.layers.{i}.mlp.gate" for i in range(tc["num_hidden_layers"])] + ["lm_head"]}}
    json.dump(hf_qc, open(f"{CKPT}/hf_quant_config.json", "w"), indent=2)

    for fn in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "generation_config.json"]:
        src = f"{SNAP}/{fn}"
        if os.path.exists(src):
            shutil.copy2(src, f"{CKPT}/{fn}")

    total_gb = sum(os.path.getsize(f"{CKPT}/{f}") for f in os.listdir(CKPT) if not f.startswith(".")) / 1e9
    print(f"\nDone: {ts} shards, {total_q} quantized, {total_bf16} bf16, {total_gb:.1f}GB ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

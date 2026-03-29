#!/usr/bin/env python3
"""Compress NVFP4 checkpoint: remap FP4 codes via sub-codebook LUT.

For each weight tensor, unpacks FP4 codes, maps through a LUT, repacks.
All scales are preserved unchanged. The output is a standard NVFP4 checkpoint
with fewer unique code values (lossy compression).

This is the DECOMPRESSED format — still 4 bits/element, but restricted to
a sub-codebook. True sub-4-bit packing is a future step.

Usage:
    python3 compress_checkpoint.py --codebook identity    # verify: should match original
    python3 compress_checkpoint.py --codebook 3bit_uniform
"""
import argparse
import gc
import json
import os
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

CKPT_DIR = "/code/tensorrt_llm/scripts/nvfp4_compress/nvfp4_checkpoint"

E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

CODEBOOKS = {
    "identity": [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6],
    "3bit_uniform": [-6, -4, -2, 0, 2, 4, 6],
    "3bit_dense": [-6, -2, -1, 0, 1, 2, 6],
    "3bit_truncate": [-4, -2, -1, 0, 1, 2, 4],
}


def build_lut(sub_values):
    cb = torch.tensor(sub_values, dtype=torch.float32)
    lut = torch.zeros(16, dtype=torch.uint8)
    for src in range(16):
        dists = (cb - E2M1_TABLE[src]).abs()
        nearest_val = cb[dists.argmin()].item()
        for dst in range(16):
            if abs(E2M1_TABLE[dst].item() - nearest_val) < 1e-6:
                lut[src] = dst
                break
    return lut


def remap_packed_weight(packed, lut):
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    codes = torch.stack([low, high], dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)
    mapped = lut[codes.long()]
    M, K = mapped.shape
    mapped = mapped.view(M, K // 2, 2)
    return (mapped[:, :, 0] | (mapped[:, :, 1] << 4)).to(torch.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codebook", type=str, default="identity", choices=list(CODEBOOKS.keys()))
    parser.add_argument("--input", type=str, default=CKPT_DIR)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"/code/tensorrt_llm/scripts/nvfp4_compress/ckpt_{args.codebook}"

    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)

    lut = build_lut(CODEBOOKS[args.codebook])
    print(f"Codebook: {args.codebook}", flush=True)
    for i in range(16):
        s, d = E2M1_TABLE[i].item(), E2M1_TABLE[lut[i].item()].item()
        if i != lut[i].item():
            print(f"  {s:+5.1f} -> {d:+5.1f}", flush=True)

    with open(f"{args.input}/model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    t0 = time.time()
    processed_shards = set()
    remapped_count = 0

    for shard_file in sorted(set(weight_map.values())):
        if shard_file in processed_shards:
            continue

        shard_keys = [k for k, v in weight_map.items() if v == shard_file]
        src_path = f"{args.input}/{shard_file}"

        has_expert_weight = any(k.endswith(".weight") and "experts." in k for k in shard_keys)

        if has_expert_weight:
            new_data = {}
            with safe_open(src_path, framework="pt", device="cpu") as sf:
                for k in shard_keys:
                    t = sf.get_tensor(k)
                    if k.endswith(".weight") and "experts." in k:
                        new_data[k] = remap_packed_weight(t, lut)
                        remapped_count += 1
                    else:
                        new_data[k] = t
            save_file(new_data, f"{args.output}/{shard_file}")
            del new_data
            gc.collect()
        else:
            os.symlink(os.path.abspath(src_path), f"{args.output}/{shard_file}")

        processed_shards.add(shard_file)

    json.dump(idx, open(f"{args.output}/model.safetensors.index.json", "w"), indent=2)

    for fn in os.listdir(args.input):
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
            continue
        src = f"{args.input}/{fn}"
        dst = f"{args.output}/{fn}"
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)

    print(f"Remapped {remapped_count} weights in {time.time()-t0:.0f}s", flush=True)
    print(f"Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()

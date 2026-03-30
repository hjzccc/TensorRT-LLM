#!/usr/bin/env python3
"""Decompress a sub-4-bit checkpoint back into standard NVFP4 shards."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_INPUT = "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/compressed_identity"


def repack_fp4_codes(codes: torch.Tensor) -> torch.Tensor:
    m, k = codes.shape
    codes = codes.view(m, k // 2, 2)
    return (codes[:, :, 0] | (codes[:, :, 1] << 4)).to(torch.uint8)


def unpack_bits(packed: torch.Tensor, bits: int, count: int) -> torch.Tensor:
    if bits == 0:
        return torch.zeros(count, dtype=torch.uint8)
    arr = packed.reshape(-1).cpu().numpy().astype(np.uint8, copy=False)
    if bits == 4:
        vals = np.empty(arr.size * 2, dtype=np.uint8)
        vals[0::2] = arr & 0x0F
        vals[1::2] = (arr >> 4) & 0x0F
        return torch.from_numpy(vals[:count].astype(np.uint8, copy=False))
    if bits == 2:
        vals = np.empty(arr.size * 4, dtype=np.uint8)
        vals[0::4] = arr & 0x03
        vals[1::4] = (arr >> 2) & 0x03
        vals[2::4] = (arr >> 4) & 0x03
        vals[3::4] = (arr >> 6) & 0x03
        return torch.from_numpy(vals[:count].astype(np.uint8, copy=False))
    if bits == 1:
        vals = np.empty(arr.size * 8, dtype=np.uint8)
        vals[0::8] = arr & 0x01
        vals[1::8] = (arr >> 1) & 0x01
        vals[2::8] = (arr >> 2) & 0x01
        vals[3::8] = (arr >> 3) & 0x01
        vals[4::8] = (arr >> 4) & 0x01
        vals[5::8] = (arr >> 5) & 0x01
        vals[6::8] = (arr >> 6) & 0x01
        vals[7::8] = (arr >> 7) & 0x01
        return torch.from_numpy(vals[:count].astype(np.uint8, copy=False))
    raise ValueError(f"Unsupported bit width: {bits}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input)
    if args.output is None:
        args.output = str(input_dir.parent / f"decompressed_{input_dir.name}")
    output_dir = Path(args.output)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    with (input_dir / "compression_manifest.json").open() as f:
        manifest = cast(dict[str, Any], json.load(f))
    with (input_dir / "model.safetensors.index.json").open() as f:
        input_index = cast(dict[str, Any], json.load(f))

    storage_mode = str(manifest.get("storage_mode", "global_id"))
    bits_per_index = int(manifest["bits_per_index"])
    codebook_id_bits = int(manifest["codebook_id_bits"])
    codebooks_raw = cast(list[list[int]], manifest["codebooks"])
    codebooks = [torch.tensor(cb, dtype=torch.uint8) for cb in codebooks_raw]
    fixed_codes = torch.tensor(cast(list[int], manifest.get("fixed_codes", [])), dtype=torch.uint8)
    stored_codebook_codes_per_block = int(manifest.get("stored_codebook_codes_per_block", 0))
    compressed_weights = cast(dict[str, dict[str, object]], manifest["compressed_weights"])
    input_weight_map = cast(dict[str, str], input_index["weight_map"])
    shard_files = sorted(set(input_weight_map.values()))

    output_weight_map: dict[str, str] = {}
    t0 = time.time()
    decompressed_count = 0

    for shard_file in shard_files:
        shard_path = input_dir / shard_file
        shard_keys = [k for k, v in input_weight_map.items() if v == shard_file]
        compressed_bases = sorted(
            {
                key[: -len(".weight_indices")]
                for key in shard_keys
                if key.endswith(".weight_indices")
            }
        )

        if not compressed_bases:
            os.symlink(shard_path.resolve(), output_dir / shard_file)
            for key in shard_keys:
                output_weight_map[key] = shard_file
            continue

        new_data: dict[str, torch.Tensor] = {}
        skip_keys: set[str] = set()
        with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
            for base in compressed_bases:
                info = cast(dict[str, Any], compressed_weights[base])
                shape = tuple(int(v) for v in cast(list[int], info["shape"]))
                packed_shape = tuple(int(v) for v in cast(list[int], info["packed_shape"]))
                num_values = shape[0] * shape[1]
                num_blocks = int(info["num_blocks"])

                indices_packed = sf.get_tensor(f"{base}.weight_indices")
                flat_indices = unpack_bits(indices_packed, bits_per_index, num_values)
                indices = flat_indices.view(shape)

                if storage_mode == "per_block_codebook":
                    codebook_entries_packed = sf.get_tensor(f"{base}.weight_codebook_entries")
                    extra_codes = unpack_bits(
                        codebook_entries_packed,
                        4,
                        num_blocks * stored_codebook_codes_per_block,
                    ).view(num_blocks, stored_codebook_codes_per_block)
                    fixed = fixed_codes.view(1, -1).expand(num_blocks, -1)
                    block_codebooks = torch.cat([fixed, extra_codes], dim=1)
                    flat_indices = indices.view(num_blocks, -1).long()
                    recon_blocks = torch.gather(block_codebooks, 1, flat_indices)
                elif codebook_id_bits > 0:
                    codebook_ids_packed = sf.get_tensor(f"{base}.weight_codebook_ids")
                    codebook_ids = unpack_bits(codebook_ids_packed, codebook_id_bits, num_blocks)
                    flat_indices = indices.view(num_blocks, -1)
                    recon_blocks = torch.empty_like(flat_indices, dtype=torch.uint8)
                    for cb_id, codebook in enumerate(codebooks):
                        mask = codebook_ids == cb_id
                        if not torch.any(mask):
                            continue
                        recon_blocks[mask] = codebook[flat_indices[mask].long()]
                else:
                    codebook_ids = torch.zeros(num_blocks, dtype=torch.uint8)
                    flat_indices = indices.view(num_blocks, -1)
                    recon_blocks = codebooks[0][flat_indices.long()]

                codes = recon_blocks.view(shape)
                packed_weight = repack_fp4_codes(codes).view(packed_shape)
                new_data[f"{base}.weight"] = packed_weight
                output_weight_map[f"{base}.weight"] = shard_file
                decompressed_count += 1

                skip_keys.add(f"{base}.weight_indices")
                skip_keys.add(f"{base}.weight_codebook_ids")
                skip_keys.add(f"{base}.weight_codebook_entries")

            for key in shard_keys:
                if key in skip_keys:
                    continue
                tensor = sf.get_tensor(key)
                new_data[key] = tensor
                output_weight_map[key] = shard_file

        save_file(new_data, str(output_dir / shard_file))
        del new_data
        gc.collect()

    output_index = dict(input_index)
    output_index["weight_map"] = output_weight_map
    with (output_dir / "model.safetensors.index.json").open("w") as f:
        json.dump(output_index, f, indent=2)

    for fn in os.listdir(input_dir):
        if fn.endswith(".safetensors") or fn in {"model.safetensors.index.json", "compression_manifest.json"}:
            continue
        src = input_dir / fn
        dst = output_dir / fn
        if dst.exists():
            continue
        os.symlink(src.resolve(), dst)

    print(f"Decompressed weights: {decompressed_count}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Elapsed: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compress an NVFP4 checkpoint with resume support.

Identical to compress_checkpoint.py but adds --resume flag to skip already-compressed shards.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Import everything from compress_checkpoint
import sys
sys.path.insert(0, str(Path(__file__).parent))
from compress_checkpoint import (
    E2M1_TABLE, BLOCK_SIZE, BLOCK_SEARCH_CHUNK_BLOCKS, SCHEMES,
    unpack_fp4_codes, repack_fp4_codes, pack_bits, is_quantized_weight,
    build_scheme_tables, compress_codes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="scripts/nvfp4_compress/nvfp4_checkpoint")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--scheme", type=str, default="3b1b_4free_exact", choices=sorted(SCHEMES))
    parser.add_argument("--resume", action="store_true", help="Resume from existing output directory")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if args.output is None:
        args.output = str(input_dir.parent / f"compressed_{args.scheme}")
    output_dir = Path(args.output)

    if output_dir.exists() and not args.resume:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = build_scheme_tables(args.scheme)
    bits_per_index = int(cast(int, tables["bits_per_index"]))
    codebook_id_bits = int(cast(int, tables["codebook_id_bits"]))

    with (input_dir / "model.safetensors.index.json").open() as f:
        input_index = json.load(f)
    input_weight_map: dict[str, str] = input_index["weight_map"]
    shard_files = sorted(set(input_weight_map.values()))

    # Load existing manifest if resuming
    manifest_path = output_dir / "compression_manifest.json"
    if args.resume and manifest_path.exists():
        with manifest_path.open() as f:
            manifest = json.load(f)
        compressed_weights = cast(dict[str, dict[str, object]], manifest["compressed_weights"])
        print(f"Resuming: {len(compressed_weights)} weights already compressed", flush=True)
    else:
        manifest = {
            "format_version": 1,
            "source_checkpoint": str(input_dir),
            "scheme": args.scheme,
            "description": cast(str, tables["description"]),
            "storage_mode": cast(str, tables["storage_mode"]),
            "loss_mode": str(cast(str, tables.get("loss_mode", "mse"))),
            "block_size": BLOCK_SIZE,
            "bits_per_index": bits_per_index,
            "codebook_id_bits": codebook_id_bits,
            "codebooks": [[int(v) for v in cb.tolist()] for cb in cast(list[torch.Tensor], tables.get("codebooks", []))],
            "fixed_codes": [int(v) for v in cast(torch.Tensor, tables.get("fixed_codes", torch.empty(0, dtype=torch.uint8))).tolist()],
            "stored_codebook_codes_per_block": int(cast(int, tables.get("stored_codebook_codes_per_block", 0))),
            "compressed_weights": {},
        }
        compressed_weights = cast(dict[str, dict[str, object]], manifest["compressed_weights"])

    # Load existing weight map if resuming
    output_index_path = output_dir / "model.safetensors.index.json"
    if args.resume and output_index_path.exists():
        with output_index_path.open() as f:
            output_index = json.load(f)
        output_weight_map: dict[str, str] = output_index["weight_map"]
    else:
        output_weight_map = {}

    t0 = time.time()
    compressed_count = 0
    remapped_count = 0
    skipped_count = 0

    for shard_idx, shard_file in enumerate(shard_files):
        shard_path = input_dir / shard_file
        output_shard_path = output_dir / shard_file

        # Skip if already done
        if args.resume and output_shard_path.exists():
            skipped_count += 1
            # Make sure weight map entries are populated
            shard_keys = [k for k, v in input_weight_map.items() if v == shard_file]
            for key in shard_keys:
                if key not in output_weight_map:
                    output_weight_map[key] = shard_file
            continue

        shard_keys = [k for k, v in input_weight_map.items() if v == shard_file]
        key_set = set(shard_keys)
        quantized_weight_keys = [k for k in shard_keys if is_quantized_weight(k, key_set)]

        if not quantized_weight_keys:
            if not output_shard_path.exists():
                os.symlink(os.path.relpath(shard_path, output_dir), output_shard_path)
            for key in shard_keys:
                output_weight_map[key] = shard_file
            continue

        new_data: dict[str, torch.Tensor] = {}
        with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
            for key in shard_keys:
                tensor = sf.get_tensor(key)
                if key in quantized_weight_keys:
                    base = key[: -len(".weight")]
                    codes = unpack_fp4_codes(tensor)
                    scale_key = f"{base}.weight_scale"
                    blk_scales = None
                    if scale_key in key_set and cast(str, tables.get("loss_mode", "mse")) == "scale_weighted":
                        blk_scales_raw = sf.get_tensor(scale_key)
                        v = blk_scales_raw.long()
                        sign = (v >> 7) & 1
                        exp_bits = (v >> 3) & 0xF
                        mant_bits = v & 0x7
                        blk_scales = ((1 - 2 * sign.float()) * (2.0 ** (exp_bits.float() - 7)) * (1 + mant_bits.float() / 8))
                    compressed = compress_codes(codes, tables, block_scales=blk_scales)
                    indices = cast(torch.Tensor, compressed["indices"])
                    codebook_ids = cast(torch.Tensor | None, compressed["codebook_ids"])
                    codebook_entries = cast(torch.Tensor | None, compressed["codebook_entries"])
                    recon_codes = cast(torch.Tensor, compressed["recon_codes"])
                    packed_indices = pack_bits(indices, bits_per_index)

                    new_data[f"{base}.weight_indices"] = packed_indices
                    output_weight_map[f"{base}.weight_indices"] = shard_file
                    if codebook_ids is not None and codebook_id_bits > 0:
                        packed_codebook_ids = pack_bits(codebook_ids, codebook_id_bits)
                        new_data[f"{base}.weight_codebook_ids"] = packed_codebook_ids
                        output_weight_map[f"{base}.weight_codebook_ids"] = shard_file
                    if codebook_entries is not None:
                        packed_codebook_entries = pack_bits(codebook_entries.reshape(-1), 4)
                        new_data[f"{base}.weight_codebook_entries"] = packed_codebook_entries
                        output_weight_map[f"{base}.weight_codebook_entries"] = shard_file

                    compressed_weights[base] = {
                        "shape": list(tensor.shape[0:1]) + [tensor.shape[1] * 2],
                        "packed_shape": list(tensor.shape),
                        "num_blocks": int(tensor.shape[0] * tensor.shape[1] * 2 // BLOCK_SIZE),
                        "shard_file": shard_file,
                    }
                    compressed_count += 1
                    if not torch.equal(recon_codes, codes):
                        remapped_count += 1
                else:
                    new_data[key] = tensor
                    output_weight_map[key] = shard_file

        save_file(new_data, str(output_shard_path))
        del new_data
        gc.collect()

        # Save progress every 10 shards
        if (shard_idx + 1) % 10 == 0:
            output_index_tmp = dict(input_index)
            output_index_tmp["weight_map"] = output_weight_map
            with (output_dir / "model.safetensors.index.json").open("w") as f:
                json.dump(output_index_tmp, f, indent=2)
            with manifest_path.open("w") as f:
                json.dump(manifest, f, indent=2)
            elapsed = time.time() - t0
            done = shard_idx + 1 - skipped_count
            total_todo = len(shard_files) - skipped_count
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total_todo - done) / rate if rate > 0 else 0
            print(f"Progress: {shard_idx+1}/{len(shard_files)} shards "
                  f"({skipped_count} skipped, {compressed_count} compressed) "
                  f"ETA: {eta/60:.1f}min", flush=True)

    # Final save
    output_index_final = dict(input_index)
    output_index_final["weight_map"] = output_weight_map
    with (output_dir / "model.safetensors.index.json").open("w") as f:
        json.dump(output_index_final, f, indent=2)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    for fn in os.listdir(input_dir):
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
            continue
        src = input_dir / fn
        dst = output_dir / fn
        if dst.exists():
            continue
        os.symlink(os.path.relpath(src, output_dir), dst)

    storage_mode = cast(str, tables["storage_mode"])
    bits_per_elem = bits_per_index + (codebook_id_bits / BLOCK_SIZE)
    if storage_mode == "per_block_codebook":
        bits_per_elem += (4 * int(cast(int, tables["stored_codebook_codes_per_block"]))) / BLOCK_SIZE
    print(f"Scheme: {args.scheme}", flush=True)
    print(f"Skipped (already done): {skipped_count}", flush=True)
    print(f"Compressed weights: {compressed_count}", flush=True)
    print(f"Remapped weights: {remapped_count}", flush=True)
    print(f"Effective bits/elem: {bits_per_elem:.4f}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Elapsed: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

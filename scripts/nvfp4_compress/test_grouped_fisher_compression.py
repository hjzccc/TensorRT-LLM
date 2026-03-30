#!/usr/bin/env python3
"""Test grouped_fisher compression on a subset of shards."""

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from compress_checkpoint import (
    build_scheme_tables,
    compress_codes,
    unpack_fp4_codes,
    pack_bits,
    is_quantized_weight,
    BLOCK_SIZE,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="nvfp4_checkpoint")
    parser.add_argument("--output", type=str, default="nvfp4_checkpoint_grouped_fisher_test")
    parser.add_argument("--num-shards", type=int, default=5, help="Number of shards to compress")
    parser.add_argument("--scheme", type=str, default="2b075b_zero_fixed_grouped_fisher")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    tables = build_scheme_tables(args.scheme)
    bits_per_index = int(cast(int, tables["bits_per_index"]))
    codebook_id_bits = int(cast(int, tables["codebook_id_bits"]))

    with (input_dir / "model.safetensors.index.json").open() as f:
        input_index = json.load(f)
    input_weight_map: dict[str, str] = input_index["weight_map"]
    shard_files = sorted(set(input_weight_map.values()))[:args.num_shards]

    print(f"Compressing {len(shard_files)} shards with {args.scheme}...")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print()

    output_weight_map: dict[str, str] = {}
    manifest: dict[str, Any] = {
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

    t0 = time.time()
    compressed_count = 0
    remapped_count = 0

    for shard_idx, shard_file in enumerate(shard_files):
        shard_path = input_dir / shard_file
        shard_keys = [k for k, v in input_weight_map.items() if v == shard_file]
        key_set = set(shard_keys)
        quantized_weight_keys = [k for k in shard_keys if is_quantized_weight(k, key_set)]

        print(f"[{shard_idx+1}/{len(shard_files)}] {shard_file}: {len(quantized_weight_keys)} quantized weights")

        if not quantized_weight_keys:
            os.symlink(os.path.relpath(shard_path, output_dir), output_dir / shard_file)
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
                    compressed = compress_codes(codes, tables)
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

        save_file(new_data, str(output_dir / shard_file))
        del new_data
        gc.collect()

    elapsed = time.time() - t0
    print()
    print(f"✅ Compression complete in {elapsed:.1f}s")
    print(f"  Compressed: {compressed_count} weights")
    print(f"  Remapped: {remapped_count} weights")
    print(f"  Output size: {sum(p.stat().st_size for p in output_dir.glob('*.safetensors')) / 1e9:.2f} GB")

    # Save manifest
    with (output_dir / "compression_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest saved to {output_dir / 'compression_manifest.json'}")

if __name__ == "__main__":
    main()

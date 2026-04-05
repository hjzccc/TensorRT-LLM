#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

BLOCK_SIZE = 16
N_WORKERS = 4


def unpack_4bit(packed: torch.Tensor, count: int) -> np.ndarray:
    arr = packed.reshape(-1).numpy().astype(np.uint8)
    vals = np.empty(arr.size * 2, dtype=np.uint8)
    vals[0::2] = arr & 0x0F
    vals[1::2] = (arr >> 4) & 0x0F
    return vals[:count]


def pack_8bit(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(values.astype(np.uint8, copy=False))


def _pass1_worker(args: tuple[str, str, dict[str, Any], int]) -> Counter[tuple[int, ...]]:
    key, shard_path, info, n_stored = args
    counts: Counter[tuple[int, ...]] = Counter()
    n_blocks = int(info["num_blocks"])
    with safe_open(shard_path, framework="pt", device="cpu") as sf:
        packed = sf.get_tensor(key)
    raw = unpack_4bit(packed, n_blocks * n_stored).reshape(n_blocks, n_stored)
    tuples = [tuple(int(v) for v in row) for row in raw]
    counts.update(tuples)
    return counts


def _pass2_worker(
    args: tuple[str, str, str, list[str], dict[str, Any], int, dict[tuple[int, ...], int], np.ndarray, np.ndarray, list[int]],
) -> tuple[str, dict[str, str], int]:
    shard_file, input_dir, output_dir, shard_keys, compressed_weights, n_stored, tuple_to_id, library_values, E2M1, fixed_codes = args

    entry_keys_in_shard = [k for k in shard_keys if k.endswith(".weight_codebook_entries")]
    if not entry_keys_in_shard:
        src_path = os.path.realpath(os.path.join(input_dir, shard_file))
        dst_path = os.path.join(output_dir, shard_file)
        if os.path.exists(src_path) and not os.path.exists(dst_path):
            os.symlink(src_path, dst_path)
        return shard_file, {k: shard_file for k in shard_keys}, 0

    new_data: dict[str, torch.Tensor] = {}
    key_map: dict[str, str] = {}
    rewritten = 0

    with safe_open(os.path.join(input_dir, shard_file), framework="pt", device="cpu") as sf:
        for key in shard_keys:
            if key.endswith(".weight_codebook_entries"):
                base = key.removesuffix(".weight_codebook_entries")
                info = compressed_weights[base]
                n_blocks = int(info["num_blocks"])
                packed = sf.get_tensor(key)
                entries = unpack_4bit(packed, n_blocks * n_stored).reshape(n_blocks, n_stored)

                ids = np.empty(n_blocks, dtype=np.uint8)
                fixed_arr = np.array(fixed_codes, dtype=np.int64)

                for b in range(n_blocks):
                    tup = tuple(int(v) for v in entries[b])
                    tid = tuple_to_id.get(tup)
                    if tid is not None:
                        ids[b] = tid
                    else:
                        block_codes = np.concatenate([fixed_arr, entries[b].astype(np.int64)])
                        block_vals = E2M1[block_codes]
                        dists = ((library_values - block_vals[None, :]) ** 2).sum(axis=1)
                        ids[b] = int(dists.argmin())

                new_data[f"{base}.weight_library_id"] = pack_8bit(ids)
                key_map[f"{base}.weight_library_id"] = shard_file
                rewritten += 1
            else:
                new_data[key] = sf.get_tensor(key)
                key_map[key] = shard_file

    save_file(new_data, os.path.join(output_dir, shard_file))
    del new_data
    gc.collect()
    return shard_file, key_map, rewritten


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="scripts/nvfp4_compress/compressed_2b075b_zero_fixed_exact")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-library-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    args = parser.parse_args()

    input_dir = Path(args.input)
    if args.output is None:
        args.output = str(input_dir.parent / f"compressed_shared_library_K{args.max_library_size}")
    output_dir = Path(args.output)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    with (input_dir / "compression_manifest.json").open() as f:
        src_manifest = cast(dict[str, Any], json.load(f))
    with (input_dir / "model.safetensors.index.json").open() as f:
        src_index = cast(dict[str, Any], json.load(f))

    src_weight_map = cast(dict[str, str], src_index["weight_map"])
    compressed_weights = cast(dict[str, dict[str, Any]], src_manifest["compressed_weights"])
    fixed_codes = cast(list[int], src_manifest["fixed_codes"])
    n_stored = int(src_manifest["stored_codebook_codes_per_block"])
    bits_per_index = int(src_manifest["bits_per_index"])

    entry_keys = sorted(k for k in src_weight_map if k.endswith(".weight_codebook_entries"))

    print(f"Pass 1: collecting unique codebook tuples ({len(entry_keys)} keys, {args.workers} workers)...", flush=True)
    t0 = time.time()

    pass1_args = []
    for key in entry_keys:
        base = key.removesuffix(".weight_codebook_entries")
        shard_path = str(input_dir / src_weight_map[key])
        pass1_args.append((key, shard_path, compressed_weights[base], n_stored))

    all_tuples: Counter[tuple[int, ...]] = Counter()
    with Pool(args.workers) as pool:
        for i, partial in enumerate(pool.imap_unordered(_pass1_worker, pass1_args, chunksize=64)):
            all_tuples += partial
            if (i + 1) % 5000 == 0:
                print(f"  {i+1}/{len(entry_keys)}", flush=True)

    unique_tuples = sorted(all_tuples.keys(), key=lambda t: all_tuples[t], reverse=True)
    print(f"  unique tuples: {len(unique_tuples)}", flush=True)

    if len(unique_tuples) > args.max_library_size:
        print(f"  truncating {len(unique_tuples)} -> {args.max_library_size}", flush=True)
        unique_tuples = unique_tuples[: args.max_library_size]

    tuple_to_id = {t: i for i, t in enumerate(unique_tuples)}
    K = len(unique_tuples)
    print(f"  library size K={K}, pass1 time={time.time()-t0:.1f}s", flush=True)

    codebook_size = 1 << bits_per_index
    full_library = np.zeros((K, codebook_size), dtype=np.uint8)
    for i, tup in enumerate(unique_tuples):
        for j, fc in enumerate(fixed_codes):
            full_library[i, j] = fc
        for j, code in enumerate(tup):
            full_library[i, len(fixed_codes) + j] = code

    E2M1 = np.array(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=np.float32,
    )
    library_values = E2M1[full_library.astype(np.int64)]

    print(f"Pass 2: rewriting shards ({args.workers} workers)...", flush=True)
    t1 = time.time()
    shard_files = sorted(set(src_weight_map.values()))

    shard_to_keys: dict[str, list[str]] = {}
    for k, v in src_weight_map.items():
        shard_to_keys.setdefault(v, []).append(k)

    pass2_args = []
    for shard_file in shard_files:
        shard_keys = shard_to_keys[shard_file]
        pass2_args.append((
            shard_file,
            str(input_dir),
            str(output_dir),
            shard_keys,
            compressed_weights,
            n_stored,
            tuple_to_id,
            library_values,
            E2M1,
            fixed_codes,
        ))

    output_weight_map: dict[str, str] = {}
    total_rewritten = 0
    with Pool(args.workers) as pool:
        for i, (sf, km, rw) in enumerate(pool.imap_unordered(_pass2_worker, pass2_args, chunksize=1)):
            output_weight_map.update(km)
            total_rewritten += rw
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(shard_files)} shards", flush=True)

    library_manifest = {
        "format_version": 2,
        "source_checkpoint": str(input_dir),
        "storage_mode": "shared_library",
        "library_size": K,
        "codebook_size": codebook_size,
        "bits_per_index": bits_per_index,
        "fixed_codes": fixed_codes,
        "stored_codebook_codes_per_block": n_stored,
        "library": full_library.tolist(),
        "block_size": BLOCK_SIZE,
        "compressed_weights": compressed_weights,
    }
    with (output_dir / "compression_manifest.json").open("w") as f:
        json.dump(library_manifest, f, indent=2)

    output_index = dict(src_index)
    output_index["weight_map"] = output_weight_map
    with (output_dir / "model.safetensors.index.json").open("w") as f:
        json.dump(output_index, f, indent=2)

    for fn in os.listdir(str(input_dir)):
        if fn.endswith(".safetensors") or fn in {"model.safetensors.index.json", "compression_manifest.json"}:
            continue
        src = (input_dir / fn).resolve()
        dst = output_dir / fn
        if not dst.exists() and src.exists():
            os.symlink(src, dst)

    bpe = bits_per_index + (8 / BLOCK_SIZE)
    print(f"Library size: {K}", flush=True)
    print(f"Rewritten weights: {total_rewritten}", flush=True)
    print(f"Effective bits/elem: {bpe:.4f}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Pass2 time: {time.time()-t1:.1f}s", flush=True)
    print(f"Total time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

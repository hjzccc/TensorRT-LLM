#!/usr/bin/env python3
"""Compress an NVFP4 checkpoint into a sub-4-bit on-disk format.

The compressed checkpoint stores, for each quantized weight tensor:
- per-element indices into a small codebook
- per-block codebook ids (one id per 16-element block)
- original scales unchanged

`decompress_checkpoint.py` reconstructs a standard NVFP4 checkpoint from this format.
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

# Phase 30 + Phase 32: Layer-wise adaptive and expert-specific correction
try:
    from phase30_32_integration import (
        detect_layer_type,
        extract_expert_id,
        compute_layer_wise_correction_params,
        compute_expert_specific_affine,
        apply_layer_wise_correction,
        apply_expert_specific_correction,
    )
    PHASE30_32_AVAILABLE = True
except ImportError:
    PHASE30_32_AVAILABLE = False
    print("Warning: Phase 30/32 integration module not found. Compression will proceed without Phase 30/32 corrections.")



DEFAULT_INPUT = "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/nvfp4_checkpoint"
BLOCK_SIZE = 16
BLOCK_SEARCH_CHUNK_BLOCKS = 65536

E2M1_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)

SCHEMES: dict[str, dict[str, object]] = {
    "identity": {
        "description": "Exact round-trip verification format (4-bit indices, 1 codebook)",
        "bits_per_index": 4,
        "codebooks": [list(range(16))],
    },
    "2b1b_demo": {
        "description": "Demo 2-bit index + 1-bit codebook selector format",
        "bits_per_index": 2,
        "codebooks": [
            [15, 12, 0, 4],
            [14, 10, 2, 6],
        ],
    },
    "2b1b_freq_symmetric": {
        "description": "Data-driven symmetric 2-bit index + 1-bit codebook selector",
        "bits_per_index": 2,
        "codebooks": [
            [9, 14, 1, 6],
            [9, 15, 1, 7],
        ],
    },
    "2b075b_zero_fixed_exact": {
        "description": "Exact per-block MSE search with 0 fixed and 3 stored FP4 codes",
        "bits_per_index": 2,
        "storage_mode": "per_block_codebook",
        "fixed_codes": [0],
    },
    "2b075b_zero_fixed_weighted_abs": {
        "description": "Per-block weighted MSE with 0 fixed and magnitude emphasis",
        "bits_per_index": 2,
        "storage_mode": "per_block_codebook",
        "fixed_codes": [0],
        "loss_mode": "weighted_abs",
    },
    "2b075b_zero_fixed_scale_weighted": {
        "description": "Per-block MSE weighted by block scale^2 (high-scale blocks get better codebooks)",
        "bits_per_index": 2,
        "storage_mode": "per_block_codebook",
        "fixed_codes": [0],
        "loss_mode": "scale_weighted",
    },
    "2b075b_zero_fixed_freq_sq": {
        "description": "Per-block MSE with frequency-squared weighting (dominant codes emphasized)",
        "bits_per_index": 2,
        "storage_mode": "per_block_codebook",
        "fixed_codes": [0],
        "loss_mode": "freq_sq",
    },
    "3b1b_4free_exact": {
        "description": "Exact per-block MSE search with 4 free FP4 codes (no fixed zero), 3.0 bits/elem",
        "bits_per_index": 2,
        "storage_mode": "per_block_codebook",
        "fixed_codes": [],
    },
    "3b1b_4free_weighted_abs": {
        "description": "Per-block weighted MSE with 4 free FP4 codes and magnitude emphasis, 3.0 bits/elem",
        "bits_per_index": 2,
        "storage_mode": "per_block_codebook",
        "fixed_codes": [],
        "loss_mode": "weighted_abs",
    },
    "2b075b_zero_fixed_grouped_fisher": {
        "description": "Per-block MSE with grouped-diagonal Fisher weighting (magnitude-based grouping)",
        "bits_per_index": 2,
        "storage_mode": "per_block_codebook",
        "fixed_codes": [0],
        "loss_mode": "grouped_fisher",
    },
    "2b075b_zero_fixed_scale_linear": {
        "description": "Per-block MSE weighted by block scale (linear, not squared) — more moderate than scale^2",
        "bits_per_index": 2,
        "storage_mode": "per_block_codebook",
        "fixed_codes": [0],
        "loss_mode": "scale_linear",
    },
    "3b1b_4free_scale_weighted": {
        "description": "Exact per-block MSE with 4 free codes + scale^2 weighting, 3.0 bits/elem",
        "bits_per_index": 2,
        "storage_mode": "per_block_codebook",
        "fixed_codes": [],
        "loss_mode": "scale_weighted",
    },
    "3b1b_4free_freq_sq": {
        "description": "Exact per-block MSE with 4 free codes + freq^2 weighting, 3.0 bits/elem",
        "bits_per_index": 2,
        "storage_mode": "per_block_codebook",
        "fixed_codes": [],
        "loss_mode": "freq_sq",
    },
    "2b075b_zero_fixed_entropy": {
        "description": "Per-block entropy-based codebook selection with 0 fixed and magnitude emphasis",
        "bits_per_index": 2,
        "storage_mode": "per_block_codebook",
        "fixed_codes": [0],
        "loss_mode": "entropy",
    },
    "3b1b_4free_entropy": {
        "description": "Per-block entropy-based codebook selection with 4 free FP4 codes, 3.0 bits/elem",
        "bits_per_index": 2,
        "storage_mode": "per_block_codebook",
        "fixed_codes": [],
        "loss_mode": "entropy",
    },
}


def unpack_fp4_codes(packed: torch.Tensor) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack([low, high], dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)


def repack_fp4_codes(codes: torch.Tensor) -> torch.Tensor:
    m, k = codes.shape
    codes = codes.view(m, k // 2, 2)
    return (codes[:, :, 0] | (codes[:, :, 1] << 4)).to(torch.uint8)


def pack_bits(values: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 0:
        return torch.empty(0, dtype=torch.uint8)
    arr = values.reshape(-1).cpu().numpy().astype(np.uint8, copy=False)
    if bits == 4:
        if arr.size % 2:
            arr = np.pad(arr, (0, 1))
        packed = arr[0::2] | (arr[1::2] << 4)
        return torch.from_numpy(packed.astype(np.uint8, copy=False))
    if bits == 2:
        pad = (-arr.size) % 4
        if pad:
            arr = np.pad(arr, (0, pad))
        packed = (
            arr[0::4]
            | (arr[1::4] << 2)
            | (arr[2::4] << 4)
            | (arr[3::4] << 6)
        )
        return torch.from_numpy(packed.astype(np.uint8, copy=False))
    if bits == 1:
        pad = (-arr.size) % 8
        if pad:
            arr = np.pad(arr, (0, pad))
        packed = (
            arr[0::8]
            | (arr[1::8] << 1)
            | (arr[2::8] << 2)
            | (arr[3::8] << 3)
            | (arr[4::8] << 4)
            | (arr[5::8] << 5)
            | (arr[6::8] << 6)
            | (arr[7::8] << 7)
        )
        return torch.from_numpy(packed.astype(np.uint8, copy=False))
    raise ValueError(f"Unsupported bit width: {bits}")


def is_quantized_weight(key: str, key_set: set[str]) -> bool:
    if not key.endswith(".weight"):
        return False
    base = key[: -len(".weight")]
    return f"{base}.weight_scale" in key_set and f"{base}.weight_scale_2" in key_set


def build_scheme_tables(scheme_name: str) -> dict[str, object]:
    scheme = cast(dict[str, Any], SCHEMES[scheme_name])
    storage_mode = str(scheme.get("storage_mode", "global_id"))

    if storage_mode == "per_block_codebook":
        fixed_codes_list = cast(list[int], scheme["fixed_codes"])
        fixed_codes = torch.tensor(fixed_codes_list, dtype=torch.uint8)
        n_fixed = len(fixed_codes_list)
        n_free = 4 - n_fixed  # total codebook size is always 4
        all_candidate_codes = [c for c in range(16) if c not in fixed_codes_list]
        candidate_extra_codes = torch.tensor(
            list(itertools.combinations(all_candidate_codes, n_free)),
            dtype=torch.uint8,
        )
        if n_fixed > 0:
            candidate_codebooks = torch.cat(
                [
                    fixed_codes.unsqueeze(0).expand(candidate_extra_codes.shape[0], -1),
                    candidate_extra_codes,
                ],
                dim=1,
            )
        else:
            candidate_codebooks = candidate_extra_codes
        candidate_values = E2M1_TABLE[candidate_codebooks.long()]
        dists = (E2M1_TABLE.view(1, 16, 1) - candidate_values[:, None, :]).abs()
        best_idx = dists.argmin(dim=2).to(torch.uint8)
        candidate_best_codes = torch.gather(candidate_codebooks, 1, best_idx.long())
        candidate_mse_luts = (E2M1_TABLE.view(1, 16) - E2M1_TABLE[candidate_best_codes.long()]) ** 2
        loss_mode = str(scheme.get("loss_mode", "mse"))
        if loss_mode == "weighted_abs":
            value_weights = 1.0 + E2M1_TABLE.abs()
            candidate_mse_luts = candidate_mse_luts * value_weights.view(1, 16)
        # scale_weighted and freq_sq: base LUT is plain MSE; weighting applied per-block in compress_codes
        return {
            "scheme_name": scheme_name,
            "storage_mode": storage_mode,
            "loss_mode": loss_mode,
            "bits_per_index": int(cast(int, scheme["bits_per_index"])),
            "codebook_id_bits": 0,
            "fixed_codes": fixed_codes,
            "stored_codebook_codes_per_block": int(n_free),
            "candidate_extra_codes": candidate_extra_codes,
            "candidate_best_index_luts": best_idx,
            "candidate_best_code_luts": candidate_best_codes.to(torch.uint8),
            "candidate_mse_luts": candidate_mse_luts.to(torch.float32),
            "description": str(cast(str, scheme["description"])),
        }

    codebooks_raw = cast(list[list[int]], scheme["codebooks"])
    codebooks = [torch.tensor(cb, dtype=torch.uint8) for cb in codebooks_raw]
    bits_per_index = int(cast(int, scheme["bits_per_index"]))
    codebook_id_bits = 0 if len(codebooks) == 1 else math.ceil(math.log2(len(codebooks)))

    best_index_luts: list[torch.Tensor] = []
    best_code_luts: list[torch.Tensor] = []
    mse_luts: list[torch.Tensor] = []
    for codebook in codebooks:
        cb_values = E2M1_TABLE[codebook.long()]
        dists = (E2M1_TABLE[:, None] - cb_values[None, :]).abs()
        best_idx = dists.argmin(dim=1).to(torch.uint8)
        best_codes = codebook[best_idx.long()].to(torch.uint8)
        mse = (E2M1_TABLE - cb_values[best_idx.long()]) ** 2
        best_index_luts.append(best_idx)
        best_code_luts.append(best_codes)
        mse_luts.append(mse)

    return {
        "scheme_name": scheme_name,
        "storage_mode": storage_mode,
        "bits_per_index": bits_per_index,
        "codebook_id_bits": codebook_id_bits,
        "codebooks": codebooks,
        "best_index_luts": best_index_luts,
        "best_code_luts": best_code_luts,
        "mse_luts": mse_luts,
        "description": str(cast(str, scheme["description"])),
    }


def compute_code_entropy(codes: torch.Tensor, num_codes: int = 16) -> float:
    """Compute Shannon entropy of code distribution.
    
    Args:
        codes: 1D tensor of code indices
        num_codes: total number of possible codes
        
    Returns:
        Shannon entropy in bits
    """
    counts = torch.bincount(codes.long(), minlength=num_codes).float()
    probs = counts / counts.sum()
    # Avoid log(0)
    probs = probs[probs > 0]
    entropy = -(probs * torch.log2(probs)).sum().item()
    return entropy


def compress_codes(
    codes: torch.Tensor,
    tables: dict[str, object],
    block_scales: torch.Tensor | None = None,
    key: str | None = None,
) -> dict[str, torch.Tensor | None]:
    # Phase 30 + Phase 32: Apply layer-wise adaptive and expert-specific corrections
    if PHASE30_32_AVAILABLE and key is not None:
        layer_type = detect_layer_type(key)
        
        # Apply Phase 30: Layer-wise adaptive correction
        correction_params = compute_layer_wise_correction_params(codes, layer_type, block_scales)
        if correction_params.get("correction_type") != "none":
            codes = apply_layer_wise_correction(codes, correction_params)
        
        # Apply Phase 32: Expert-specific affine correction
        if layer_type == "expert":
            expert_id = extract_expert_id(key)
            if expert_id is not None:
                scale, bias = compute_expert_specific_affine(codes, expert_id, block_scales)
                codes = apply_expert_specific_correction(codes, expert_id, scale, bias)

    if codes.shape[1] % BLOCK_SIZE != 0:
        raise ValueError(f"Expected K divisible by {BLOCK_SIZE}, got {tuple(codes.shape)}")

    scheme_name = cast(str, tables["scheme_name"])
    if scheme_name == "identity":
        blocks = codes.numel() // BLOCK_SIZE
        return {
            "indices": codes.to(torch.uint8),
            "codebook_ids": torch.zeros(blocks, dtype=torch.uint8),
            "codebook_entries": None,
            "recon_codes": codes.to(torch.uint8),
        }

    storage_mode = cast(str, tables["storage_mode"])
    if storage_mode == "per_block_codebook":
        flat_blocks = codes.reshape(-1, BLOCK_SIZE)
        candidate_extra_codes = cast(torch.Tensor, tables["candidate_extra_codes"])
        candidate_best_index_luts = cast(torch.Tensor, tables["candidate_best_index_luts"])
        candidate_best_code_luts = cast(torch.Tensor, tables["candidate_best_code_luts"])
        candidate_mse_luts = cast(torch.Tensor, tables["candidate_mse_luts"])

        block_indices = torch.empty_like(flat_blocks, dtype=torch.uint8)
        recon_blocks = torch.empty_like(flat_blocks, dtype=torch.uint8)
        block_codebook_entries = torch.empty(
            (flat_blocks.shape[0], candidate_extra_codes.shape[1]),
            dtype=torch.uint8,
        )

        loss_mode = cast(str, tables.get("loss_mode", "mse"))
                # Pre-compute thresholds for grouped_fisher loss mode (using fast median instead of quantile)
        grouped_fisher_thresholds = None
        if loss_mode == "grouped_fisher":
            all_magnitudes = flat_blocks.float().abs()
            # Use median for fast O(n) computation instead of O(n log n) quantile
            # This gives us a simple binary grouping: high (>= median) and low (< median)
            median = torch.median(all_magnitudes)
            grouped_fisher_thresholds = {
                "high": median,
                "low": median * 0.5,  # Approximate lower threshold
            }

        for start in range(0, flat_blocks.shape[0], BLOCK_SEARCH_CHUNK_BLOCKS):
            end = min(start + BLOCK_SEARCH_CHUNK_BLOCKS, flat_blocks.shape[0])
            chunk = flat_blocks[start:end].to(torch.long)
            counts = torch.nn.functional.one_hot(chunk, num_classes=16).sum(dim=1).to(torch.float32)
            if loss_mode == "scale_weighted" and block_scales is not None:
                # Weight each block's code counts by its block scale^2
                # block_scales shape: [num_blocks] (one per 16-element block)
                scales_chunk = block_scales[start:end].to(torch.float32).unsqueeze(1)
                weighted_counts = counts * (scales_chunk ** 2)
                costs = weighted_counts @ candidate_mse_luts.T
            elif loss_mode == "entropy":
                # Fast entropy-based codebook selection using frequency skewness
                # Prefer codebooks that produce more skewed (lower entropy) distributions
                # Compute frequency skewness: sum of squared frequencies (higher = more skewed)
                # This is a fast proxy for entropy that avoids log calculations
                freq = counts / counts.sum(dim=1, keepdim=True).clamp(min=1)
                skewness = (freq ** 2).sum(dim=1, keepdim=True)  # Higher skewness = lower entropy
                # Weight MSE by inverse skewness (prefer high-skewness codebooks)
                weighted_counts = counts / (skewness + 1e-8)
                costs = weighted_counts @ candidate_mse_luts.T
            elif loss_mode == "scale_linear" and block_scales is not None:
                # Weight each block's code counts by its block scale (linear, not squared)
                # More moderate than scale^2: reduces dynamic range from 357000x to 600x
                scales_chunk = block_scales[start:end].to(torch.float32).unsqueeze(1)
                weighted_counts = counts * scales_chunk
                costs = weighted_counts @ candidate_mse_luts.T
            elif loss_mode == "freq_sq":
                # Weight by frequency^2: emphasize dominant codes more
                freq = counts / counts.sum(dim=1, keepdim=True).clamp(min=1)
                weighted_counts = counts * freq
                costs = weighted_counts @ candidate_mse_luts.T
            elif loss_mode == "grouped_fisher":
                # Weight by grouped Fisher: magnitude-based grouping (using pre-computed median)
                # High-magnitude elements (>= median) get 2x weight, low (< median) get 0.5x
                magnitudes = chunk.float().abs()
                high_threshold = grouped_fisher_thresholds["high"]
                
                # Simple binary grouping for speed
                weights = torch.where(magnitudes >= high_threshold, 2.0, 0.5)
                
                # Normalize weights per block
                weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
                
                # Compute weighted costs
                weighted_counts = counts * weights
                costs = weighted_counts @ candidate_mse_luts.T
            else:
                costs = counts @ candidate_mse_luts.T
            chosen = costs.argmin(dim=1)
            chosen_index_luts = candidate_best_index_luts[chosen]
            chosen_code_luts = candidate_best_code_luts[chosen]
            block_indices[start:end] = torch.gather(chosen_index_luts, 1, chunk)
            recon_blocks[start:end] = torch.gather(chosen_code_luts, 1, chunk)
            block_codebook_entries[start:end] = candidate_extra_codes[chosen]

        return {
            "indices": block_indices.reshape_as(codes),
            "codebook_ids": None,
            "codebook_entries": block_codebook_entries,
            "recon_codes": recon_blocks.reshape_as(codes),
        }

    flat_blocks = codes.reshape(-1, BLOCK_SIZE)
    codebooks = cast(list[torch.Tensor], tables["codebooks"])
    best_index_luts = cast(list[torch.Tensor], tables["best_index_luts"])
    best_code_luts = cast(list[torch.Tensor], tables["best_code_luts"])
    mse_luts = cast(list[torch.Tensor], tables["mse_luts"])

    if len(codebooks) == 1:
        chosen_codebooks = torch.zeros(flat_blocks.shape[0], dtype=torch.uint8)
    else:
        all_costs = torch.stack(
            [mse_lut[flat_blocks.long()].sum(dim=1) for mse_lut in mse_luts],
            dim=1,
        )
        chosen_codebooks = all_costs.argmin(dim=1).to(torch.uint8)

    block_indices = torch.empty_like(flat_blocks, dtype=torch.uint8)
    recon_blocks = torch.empty_like(flat_blocks, dtype=torch.uint8)
    for cb_id in range(len(codebooks)):
        mask = chosen_codebooks == cb_id
        if not torch.any(mask):
            continue
        selected = flat_blocks[mask]
        idx_lut = best_index_luts[cb_id]
        code_lut = best_code_luts[cb_id]
        block_indices[mask] = idx_lut[selected.long()]
        recon_blocks[mask] = code_lut[selected.long()]

    return {
        "indices": block_indices.reshape_as(codes),
        "codebook_ids": chosen_codebooks,
        "codebook_entries": None,
        "recon_codes": recon_blocks.reshape_as(codes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--scheme", type=str, default="identity", choices=sorted(SCHEMES))
    parser.add_argument("--resume", action="store_true", help="Resume interrupted compression (skip existing shards)")
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

    for shard_file in shard_files:
        shard_path = input_dir / shard_file
        shard_keys = [k for k, v in input_weight_map.items() if v == shard_file]
        key_set = set(shard_keys)
        quantized_weight_keys = [k for k in shard_keys if is_quantized_weight(k, key_set)]

        # Skip already-processed shards when resuming
        if args.resume and (output_dir / shard_file).exists():
            with safe_open(str(output_dir / shard_file), framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    output_weight_map[key] = shard_file
            continue

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
                    # Load block scales for scale-weighted compression
                    scale_key = f"{base}.weight_scale"
                    blk_scales: torch.Tensor | None = None
                    if scale_key in key_set and cast(str, tables.get("loss_mode", "mse")) in ("scale_weighted", "scale_linear"):
                        blk_scales_raw = sf.get_tensor(scale_key)
                        # Decode FP8 E4M3: sign=bit7, exp=bits6-3, mant=bits2-0
                        v = blk_scales_raw.long()
                        sign = (v >> 7) & 1
                        exp_bits = (v >> 3) & 0xF
                        mant_bits = v & 0x7
                        blk_scales = ((1 - 2 * sign.float()) * (2.0 ** (exp_bits.float() - 7)) * (1 + mant_bits.float() / 8))
                    compressed = compress_codes(codes, tables, block_scales=blk_scales, key=key)
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

    output_index = dict(input_index)
    output_index["weight_map"] = output_weight_map
    with (output_dir / "model.safetensors.index.json").open("w") as f:
        json.dump(output_index, f, indent=2)
    with (output_dir / "compression_manifest.json").open("w") as f:
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
    print(f"Compressed weights: {compressed_count}", flush=True)
    print(f"Remapped weights: {remapped_count}", flush=True)
    print(f"Effective bits/elem: {bits_per_elem:.4f}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Elapsed: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

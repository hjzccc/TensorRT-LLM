#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import hadamard
from safetensors import safe_open

BLOCK_SIZE = 16
CKPT = Path("/root/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3")

E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

FP4_GRID = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)

GRID_TO_CODE = torch.tensor(
    [15, 14, 13, 12, 11, 10, 9, 0, 1, 2, 3, 4, 5, 6, 7],
    dtype=torch.uint8,
)

H16 = torch.tensor(hadamard(BLOCK_SIZE), dtype=torch.float32) / (BLOCK_SIZE ** 0.5)


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack([low, high], dim=-1).reshape(packed.shape[0], packed.shape[1] * 2).to(torch.uint8)


def codes_to_values(codes: torch.Tensor) -> torch.Tensor:
    return E2M1_TABLE[codes.long()]


def values_to_codes(values: torch.Tensor) -> torch.Tensor:
    dists = (values.unsqueeze(-1) - FP4_GRID.view(1, 1, -1)).abs()
    return GRID_TO_CODE[dists.argmin(dim=-1)]


def block_real_values(packed_weight: torch.Tensor, block_scale: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    codes = unpack_fp4(packed_weight)
    out_features, in_features = codes.shape
    num_blocks_per_row = in_features // BLOCK_SIZE
    codes_blocked = codes.reshape(out_features * num_blocks_per_row, BLOCK_SIZE)
    scale_flat = block_scale.reshape(-1, 1).float()
    gs = global_scale.float().item()
    fp4_values = codes_to_values(codes_blocked)
    return fp4_values * scale_flat * gs


def build_candidate_mse_luts() -> tuple[torch.Tensor, torch.Tensor]:
    nonzero_codes = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
    candidates = list(itertools.combinations(nonzero_codes, 3))
    candidate_codebooks = torch.zeros((len(candidates), 4), dtype=torch.uint8)
    for i, (a, b, c) in enumerate(candidates):
        candidate_codebooks[i] = torch.tensor([0, a, b, c], dtype=torch.uint8)

    candidate_values = E2M1_TABLE[candidate_codebooks.long()]
    all_values = E2M1_TABLE.unsqueeze(0)
    dists = (all_values.unsqueeze(-1) - candidate_values.unsqueeze(1)).abs()
    best_idx = dists.argmin(dim=-1)
    nearest_codes = torch.gather(
        candidate_codebooks.unsqueeze(1).expand(-1, 16, -1), 2, best_idx.unsqueeze(-1).long()
    ).squeeze(-1)
    nearest_values = E2M1_TABLE[nearest_codes.long()]
    mse_per_code = (E2M1_TABLE.unsqueeze(0) - nearest_values) ** 2
    return mse_per_code, candidate_codebooks


def baseline_mse(real_values: torch.Tensor, scale_flat: torch.Tensor, gs: float, mse_per_code: torch.Tensor, candidate_codebooks: torch.Tensor) -> torch.Tensor:
    sf = (scale_flat * gs).unsqueeze(1)
    normalized = real_values / sf
    codes = values_to_codes(normalized)

    histograms = torch.zeros((codes.shape[0], 16), dtype=torch.float32)
    for c in range(16):
        histograms[:, c] = (codes.long() == c).sum(dim=1).float()
    block_costs = histograms @ mse_per_code.T
    best_cb_idx = block_costs.argmin(dim=1)
    best_cbs = candidate_codebooks[best_cb_idx]

    recon_normalized = torch.zeros_like(normalized)
    for b in range(codes.shape[0]):
        cb_vals = E2M1_TABLE[best_cbs[b].long()]
        for j in range(BLOCK_SIZE):
            d = (E2M1_TABLE[codes[b, j].long()] - cb_vals).abs()
            recon_normalized[b, j] = cb_vals[d.argmin()]

    recon_real = recon_normalized * sf
    return ((recon_real - real_values) ** 2).mean(dim=1)


def hadamard_mse(real_values: torch.Tensor, scale_flat: torch.Tensor, gs: float, mse_per_code: torch.Tensor, candidate_codebooks: torch.Tensor) -> torch.Tensor:
    rotated = real_values @ H16.T
    sf = (scale_flat * gs).unsqueeze(1)

    rot_abs_max = rotated.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    rot_scale = rot_abs_max / 6.0
    rot_normalized = rotated / rot_scale

    codes_rot = values_to_codes(rot_normalized)

    histograms = torch.zeros((codes_rot.shape[0], 16), dtype=torch.float32)
    for c in range(16):
        histograms[:, c] = (codes_rot.long() == c).sum(dim=1).float()
    block_costs = histograms @ mse_per_code.T
    best_cb_idx = block_costs.argmin(dim=1)
    best_cbs = candidate_codebooks[best_cb_idx]

    recon_rot_normalized = torch.zeros_like(rot_normalized)
    for b in range(codes_rot.shape[0]):
        cb_vals = E2M1_TABLE[best_cbs[b].long()]
        for j in range(BLOCK_SIZE):
            d = (E2M1_TABLE[codes_rot[b, j].long()] - cb_vals).abs()
            recon_rot_normalized[b, j] = cb_vals[d.argmin()]

    recon_rotated = recon_rot_normalized * rot_scale
    recon_original = recon_rotated @ H16
    return ((recon_original - real_values) ** 2).mean(dim=1)


def load_quantized_blocks(num_tensors: int = 10) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    wm = json.loads((CKPT / "model.safetensors.index.json").read_text())["weight_map"]
    expert_keys = [
        k for k in sorted(wm)
        if "mlp.experts." in k
        and k.endswith(".weight")
        and "scale" not in k
    ][:num_tensors]

    results = []
    for key in expert_keys:
        scale_key = key.replace(".weight", ".weight_scale")
        gs_key = key.replace(".weight", ".weight_scale_2")
        if scale_key not in wm or gs_key not in wm:
            continue

        with safe_open(str(CKPT / wm[key]), framework="pt", device="cpu") as sf:
            packed = sf.get_tensor(key)
        with safe_open(str(CKPT / wm[scale_key]), framework="pt", device="cpu") as sf:
            block_scale = sf.get_tensor(scale_key)
        with safe_open(str(CKPT / wm[gs_key]), framework="pt", device="cpu") as sf:
            global_scale = sf.get_tensor(gs_key)

        print(f"  {key}: packed={tuple(packed.shape)} scale={tuple(block_scale.shape)} gs={global_scale.item():.4f}", flush=True)
        results.append((packed, block_scale, global_scale))

    return results


def main() -> None:
    print("Building codebook candidate tables...", flush=True)
    mse_per_code, candidate_codebooks = build_candidate_mse_luts()

    print("Loading quantized weights...", flush=True)
    t0 = time.time()
    tensor_data = load_quantized_blocks(num_tensors=10)
    print(f"Loaded {len(tensor_data)} tensors in {time.time()-t0:.1f}s\n", flush=True)

    all_baseline_mse = []
    all_hadamard_mse = []

    for ti, (packed, block_scale, global_scale) in enumerate(tensor_data):
        real_vals = block_real_values(packed, block_scale, global_scale)
        gs = global_scale.float().item()
        bs_flat = block_scale.float().reshape(-1)
        num_blocks = real_vals.shape[0]

        chunk = 2000
        baseline_mses = []
        hadamard_mses = []

        for start in range(0, num_blocks, chunk):
            end = min(start + chunk, num_blocks)
            rv = real_vals[start:end]
            bsf = bs_flat[start:end]

            bm = baseline_mse(rv, bsf, gs, mse_per_code, candidate_codebooks)
            baseline_mses.append(bm)

            hm = hadamard_mse(rv, bsf, gs, mse_per_code, candidate_codebooks)
            hadamard_mses.append(hm)

            if (start // chunk) % 10 == 0:
                print(f"  tensor {ti}: {end}/{num_blocks} blocks...", flush=True)

        b = torch.cat(baseline_mses)
        h = torch.cat(hadamard_mses)

        print(f"Tensor {ti}: {num_blocks} blocks", flush=True)
        print(f"  Baseline MSE: mean={b.mean():.8f} median={b.median():.8f} p95={b.quantile(0.95):.8f} p99={b.quantile(0.99):.8f}", flush=True)
        print(f"  Hadamard MSE: mean={h.mean():.8f} median={h.median():.8f} p95={h.quantile(0.95):.8f} p99={h.quantile(0.99):.8f}", flush=True)
        print(f"  Hadamard wins: {(h < b).float().mean()*100:.1f}% of blocks", flush=True)
        print(f"  Mean ratio (had/baseline): {(h.mean() / b.mean()):.4f}", flush=True)
        print(flush=True)

        all_baseline_mse.append(b)
        all_hadamard_mse.append(h)

    all_b = torch.cat(all_baseline_mse)
    all_h = torch.cat(all_hadamard_mse)

    print("=" * 60, flush=True)
    print(f"OVERALL ({all_b.shape[0]} blocks)", flush=True)
    print(f"  Baseline MSE: mean={all_b.mean():.8f} median={all_b.median():.8f} p95={all_b.quantile(0.95):.8f} p99={all_b.quantile(0.99):.8f}", flush=True)
    print(f"  Hadamard MSE: mean={all_h.mean():.8f} median={all_h.median():.8f} p95={all_h.quantile(0.95):.8f} p99={all_h.quantile(0.99):.8f}", flush=True)
    print(f"  Hadamard wins: {(all_h < all_b).float().mean()*100:.1f}% of blocks", flush=True)
    print(f"  Mean ratio (had/baseline): {(all_h.mean() / all_b.mean()):.4f}", flush=True)
    print(f"  Total time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

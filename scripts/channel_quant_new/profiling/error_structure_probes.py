#!/usr/bin/env python3
"""Three structural probes into NVFP4 -> 3-bit quantization error.

Probe 1: Residual predictability — can a tiny model predict the error from cheap features?
Probe 2: Stochastic rounding — does randomizing 3-bit rounding help or hurt vs deterministic?
Probe 3: Scale-swap decomposition — is error from bad scales or bad code assignment?

All probes operate on normalized FP4 code space (after block scaling).
No PPL eval needed — pure error analysis on weight tensors.
"""
import sys
import time
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from safetensors import safe_open

BLOCK_SIZE = 16
FP4_VALUES = np.array([-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
FP4_UNIQUE = np.array([-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
CB_3BIT = np.array([-6, -4, -2, 0, 2, 4, 6], dtype=np.float32)
BOUNDARIES_FP4 = np.array([-5, -3.5, -2.5, -1.75, -1.25, -0.75, -0.25, 0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=np.float32)
BOUNDARIES_3BIT = np.array([-5, -3, -1, 0, 1, 3, 5], dtype=np.float32)

MODEL_DIR = "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots"


def get_model_dir():
    p = Path(MODEL_DIR)
    return list(p.iterdir())[0]


def quantize_block_to_fp4(block):
    """block: (16,) float. Returns FP4 codes and reconstructed values."""
    amax = np.abs(block).max()
    if amax == 0:
        return np.zeros(16, dtype=np.float32), np.zeros(16, dtype=np.float32), 0.0
    scale = np.float16(amax / 6.0).astype(np.float32)
    if scale == 0:
        scale = 1.0
    normalized = block / scale
    idx = np.searchsorted(BOUNDARIES_FP4, normalized)
    fp4_vals = FP4_UNIQUE[np.clip(idx, 0, len(FP4_UNIQUE) - 1)]
    return fp4_vals, fp4_vals * scale, scale


def fp4_to_3bit_deterministic(fp4_vals):
    """Round FP4 codes to nearest 3-bit code: ±{0,2,4,6}."""
    idx = np.searchsorted(BOUNDARIES_3BIT, fp4_vals)
    return CB_3BIT[np.clip(idx, 0, len(CB_3BIT) - 1)]


def fp4_to_3bit_stochastic(fp4_vals, rng):
    """Stochastic rounding to 3-bit: probability proportional to distance."""
    result = np.empty_like(fp4_vals)
    for i, v in enumerate(fp4_vals):
        lower_idx = np.searchsorted(CB_3BIT, v, side='right') - 1
        lower_idx = max(0, min(lower_idx, len(CB_3BIT) - 2))
        lower = CB_3BIT[lower_idx]
        upper = CB_3BIT[lower_idx + 1]
        if upper == lower:
            result[i] = lower
        else:
            p_upper = (v - lower) / (upper - lower)
            p_upper = np.clip(p_upper, 0, 1)
            result[i] = upper if rng.random() < p_upper else lower
    return result


def collect_block_data(max_experts=50):
    """Collect normalized FP4 codes, 3-bit codes, residuals, and features for a sample of experts."""
    model_dir = get_model_dir()
    files = sorted(model_dir.glob("*.safetensors"))

    all_fp4 = []
    all_3bit = []
    all_residuals = []
    all_features = []
    all_scales = []
    n_blocks = 0

    for fpath in files:
        with safe_open(str(fpath), framework='pt', device='cpu') as f:
            for key in f.keys():
                if 'experts' not in key:
                    continue
                if 'gate_up_proj' not in key and 'down_proj' not in key:
                    continue
                if 'mtp.' in key or 'shared_expert' in key:
                    continue

                packed = f.get_tensor(key).float().numpy()
                n_exp = packed.shape[0]
                sample_experts = min(max_experts, n_exp)
                expert_indices = np.random.choice(n_exp, sample_experts, replace=False)

                for ei in expert_indices:
                    weight = packed[ei]
                    rows, cols = weight.shape
                    pad = (BLOCK_SIZE - cols % BLOCK_SIZE) % BLOCK_SIZE
                    if pad > 0:
                        weight = np.pad(weight, ((0, 0), (0, pad)))

                    blocks = weight.reshape(-1, BLOCK_SIZE)

                    for bi in range(len(blocks)):
                        block = blocks[bi]
                        fp4_vals, _, scale = quantize_block_to_fp4(block)
                        if scale == 0:
                            continue

                        bit3_vals = fp4_to_3bit_deterministic(fp4_vals)
                        residual = fp4_vals - bit3_vals

                        all_fp4.append(fp4_vals)
                        all_3bit.append(bit3_vals)
                        all_residuals.append(residual)
                        all_scales.append(scale)

                        block_var = np.var(fp4_vals)
                        block_mean = np.mean(fp4_vals)
                        block_max = np.max(np.abs(fp4_vals))
                        all_features.append([scale, block_var, block_mean, block_max])
                        n_blocks += 1

                if n_blocks >= 500000:
                    break
        if n_blocks >= 500000:
            break

    return (np.array(all_fp4), np.array(all_3bit), np.array(all_residuals),
            np.array(all_features), np.array(all_scales), n_blocks)


def probe1_residual_predictability(fp4, bit3, residuals, features, n_blocks):
    """Can we predict the residual sign from cheap features?"""
    print(f"\n{'='*60}")
    print("PROBE 1: Residual Predictability")
    print(f"{'='*60}")
    print(f"Blocks sampled: {n_blocks:,}")

    flat_residuals = residuals.flatten()
    nonzero_mask = flat_residuals != 0
    nonzero_residuals = flat_residuals[nonzero_mask]

    print(f"Total elements: {len(flat_residuals):,}")
    print(f"Non-zero residuals: {nonzero_mask.sum():,} ({100*nonzero_mask.mean():.1f}%)")
    print(f"Zero residuals (3-bit == FP4): {(~nonzero_mask).sum():,} ({100*(~nonzero_mask).mean():.1f}%)")

    signs = np.sign(nonzero_residuals)
    print(f"\nResidual sign distribution:")
    print(f"  Positive (3-bit < FP4, rounded down): {(signs > 0).sum():,} ({100*(signs > 0).mean():.1f}%)")
    print(f"  Negative (3-bit > FP4, rounded up): {(signs < 0).sum():,} ({100*(signs < 0).mean():.1f}%)")

    mags = np.abs(nonzero_residuals)
    print(f"\nResidual magnitude stats:")
    print(f"  Mean: {mags.mean():.4f}")
    print(f"  Median: {np.median(mags):.4f}")
    print(f"  Max: {mags.max():.4f}")
    print(f"  Unique values: {len(np.unique(nonzero_residuals))}")

    unique_res, counts = np.unique(nonzero_residuals, return_counts=True)
    print(f"\nResidual value histogram:")
    for v, c in sorted(zip(unique_res, counts), key=lambda x: -x[1])[:15]:
        print(f"  {v:+6.2f}: {c:>10,} ({100*c/len(nonzero_residuals):5.1f}%)")

    flat_fp4 = fp4.flatten()
    flat_3bit = bit3.flatten()

    print(f"\nPredictability by FP4 code:")
    for code in FP4_UNIQUE:
        mask = flat_fp4 == code
        if mask.sum() == 0:
            continue
        res_for_code = flat_residuals[mask]
        nonzero = res_for_code[res_for_code != 0]
        if len(nonzero) == 0:
            print(f"  FP4={code:+5.1f}: always exact (3-bit has this value)")
            continue
        mean_res = nonzero.mean()
        std_res = nonzero.std()
        print(f"  FP4={code:+5.1f}: {len(nonzero):>8,} nonzero, mean={mean_res:+.3f}, std={std_res:.3f}")

    fp4_code_feature = flat_fp4[nonzero_mask]
    pos_in_block = np.tile(np.arange(BLOCK_SIZE), n_blocks)[nonzero_mask]
    scale_per_elem = np.repeat(features[:, 0], BLOCK_SIZE)[nonzero_mask]

    from numpy.linalg import lstsq
    X = np.column_stack([fp4_code_feature, pos_in_block, scale_per_elem, np.ones(len(signs))])
    y = signs
    coeffs, residual_sum, _, _ = lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    sign_pred = np.sign(y_pred)
    accuracy = (sign_pred == signs).mean()
    print(f"\nLinear probe for residual SIGN:")
    print(f"  Features: [fp4_code, position_in_block, block_scale, bias]")
    print(f"  Coefficients: {coeffs}")
    print(f"  Sign prediction accuracy: {accuracy:.4f} (baseline 50%)")
    print(f"  {'PREDICTABLE' if accuracy > 0.6 else 'NOT PREDICTABLE' if accuracy < 0.55 else 'WEAKLY PREDICTABLE'}")

    return {
        'nonzero_fraction': float(nonzero_mask.mean()),
        'sign_balance': float((signs > 0).mean()),
        'mean_magnitude': float(mags.mean()),
        'linear_probe_accuracy': float(accuracy),
    }


def probe2_stochastic_rounding(fp4, scales, n_blocks):
    """Compare deterministic vs stochastic 3-bit rounding MSE."""
    print(f"\n{'='*60}")
    print("PROBE 2: Stochastic vs Deterministic Rounding")
    print(f"{'='*60}")

    rng = np.random.default_rng(42)
    n_trials = 5

    det_mse_total = 0.0
    stoch_mse_trials = []

    for bi in range(n_blocks):
        fp4_block = fp4[bi]
        det_3bit = fp4_to_3bit_deterministic(fp4_block)
        det_mse_total += np.mean((fp4_block - det_3bit) ** 2)

    det_mse = det_mse_total / n_blocks

    for trial in range(n_trials):
        trial_rng = np.random.default_rng(42 + trial)
        stoch_mse_total = 0.0
        for bi in range(n_blocks):
            fp4_block = fp4[bi]
            stoch_3bit = fp4_to_3bit_stochastic(fp4_block, trial_rng)
            stoch_mse_total += np.mean((fp4_block - stoch_3bit) ** 2)
        stoch_mse_trials.append(stoch_mse_total / n_blocks)

    stoch_mse_mean = np.mean(stoch_mse_trials)
    stoch_mse_std = np.std(stoch_mse_trials)

    print(f"Deterministic MSE: {det_mse:.6f}")
    print(f"Stochastic MSE:    {stoch_mse_mean:.6f} +/- {stoch_mse_std:.6f}")
    print(f"Ratio (stoch/det): {stoch_mse_mean / det_mse:.4f}")

    if stoch_mse_mean < det_mse * 0.95:
        verdict = "STOCHASTIC WINS: deterministic error has harmful coherent structure"
    elif stoch_mse_mean > det_mse * 1.05:
        verdict = "DETERMINISTIC WINS: error already has beneficial cancellation"
    else:
        verdict = "ROUGHLY EQUAL: error is close to structureless noise"
    print(f"Verdict: {verdict}")

    return {
        'deterministic_mse': float(det_mse),
        'stochastic_mse_mean': float(stoch_mse_mean),
        'stochastic_mse_std': float(stoch_mse_std),
        'ratio': float(stoch_mse_mean / det_mse),
        'verdict': verdict,
    }


def probe3_scale_swap(fp4, scales, n_blocks):
    """Decompose error into scale error vs code error by swapping scales between blocks."""
    print(f"\n{'='*60}")
    print("PROBE 3: Scale-Swap Decomposition")
    print(f"{'='*60}")

    det_3bit = np.array([fp4_to_3bit_deterministic(fp4[bi]) for bi in range(n_blocks)])

    original_mse_per_block = np.array([
        np.mean((fp4[bi] - det_3bit[bi]) ** 2) for bi in range(n_blocks)
    ])

    rng = np.random.default_rng(123)
    n_swaps = min(100000, n_blocks)
    swap_indices = rng.choice(n_blocks, size=(n_swaps, 2), replace=True)

    code_error_samples = []
    scale_error_samples = []

    for si in range(n_swaps):
        i, j = swap_indices[si]
        if i == j:
            continue

        real_i = fp4[i] * scales[i]
        real_j = fp4[j] * scales[j]

        recon_i_own_scale = det_3bit[i] * scales[i]
        recon_i_swapped_scale = det_3bit[i] * scales[j]

        mse_own = np.mean((real_i - recon_i_own_scale) ** 2)
        mse_swapped = np.mean((real_i - recon_i_swapped_scale) ** 2)

        code_error_samples.append(mse_own)
        scale_error_samples.append(mse_swapped)

    code_err = np.mean(code_error_samples)
    swap_err = np.mean(scale_error_samples)

    print(f"Original code error (own scale):    {code_err:.6f}")
    print(f"Swapped scale error (wrong scale):  {swap_err:.6f}")
    print(f"Ratio (swapped/original):           {swap_err / code_err:.2f}x")

    scale_values = scales.flatten()
    scale_cv = np.std(scale_values) / np.mean(scale_values)
    print(f"\nBlock scale statistics:")
    print(f"  Mean: {np.mean(scale_values):.6f}")
    print(f"  Std:  {np.std(scale_values):.6f}")
    print(f"  CV:   {scale_cv:.4f}")
    print(f"  Min:  {np.min(scale_values):.6f}")
    print(f"  Max:  {np.max(scale_values):.6f}")

    high_scale = scale_values > np.percentile(scale_values, 90)
    low_scale = scale_values < np.percentile(scale_values, 10)
    high_mse = original_mse_per_block[high_scale[:n_blocks] if len(high_scale) >= n_blocks else high_scale].mean()
    low_mse = original_mse_per_block[low_scale[:n_blocks] if len(low_scale) >= n_blocks else low_scale].mean()
    print(f"\nMSE by scale magnitude:")
    print(f"  Top 10% scales: MSE = {high_mse:.6f}")
    print(f"  Bottom 10% scales: MSE = {low_mse:.6f}")
    print(f"  Ratio: {high_mse / low_mse:.2f}x")

    if swap_err > code_err * 3:
        verdict = "SCALE-DOMINATED: error is mostly from scale mismatch, code assignment is relatively good"
    elif swap_err < code_err * 1.5:
        verdict = "CODE-DOMINATED: error is intrinsic to code cell geometry, scale barely matters"
    else:
        verdict = "MIXED: both scale and code contribute significantly"
    print(f"Verdict: {verdict}")

    return {
        'code_error_mse': float(code_err),
        'swap_error_mse': float(swap_err),
        'ratio': float(swap_err / code_err),
        'scale_cv': float(scale_cv),
        'high_scale_mse': float(high_mse),
        'low_scale_mse': float(low_mse),
        'verdict': verdict,
    }


def main():
    print("Collecting block data from expert weights...", flush=True)
    t0 = time.time()
    np.random.seed(42)
    fp4, bit3, residuals, features, scales, n_blocks = collect_block_data(max_experts=30)
    print(f"Collected {n_blocks:,} blocks in {time.time()-t0:.0f}s\n", flush=True)

    results = {}

    t1 = time.time()
    results['probe1'] = probe1_residual_predictability(fp4, bit3, residuals, features, n_blocks)
    print(f"Probe 1 took {time.time()-t1:.0f}s", flush=True)

    t2 = time.time()
    results['probe2'] = probe2_stochastic_rounding(fp4, scales, n_blocks)
    print(f"Probe 2 took {time.time()-t2:.0f}s", flush=True)

    t3 = time.time()
    results['probe3'] = probe3_scale_swap(fp4, scales, n_blocks)
    print(f"Probe 3 took {time.time()-t3:.0f}s", flush=True)

    total = time.time() - t0
    print(f"\nTotal time: {total:.0f}s")

    out_path = "/code/tensorrt_llm/scripts/channel_quant_new/profiling/error_structure_probes.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()

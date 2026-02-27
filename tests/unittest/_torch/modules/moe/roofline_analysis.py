# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Roofline analysis of per-expert GEMMs under skewed vs uniform distributions.

Generates a roofline plot showing where each expert's GEMM falls relative
to the GPU's compute and memory ceilings.

Usage:
    python roofline_analysis.py
    # Saves roofline_moe.png in the current directory
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# GPU specs (H100 SXM defaults — override for your GPU)
# ---------------------------------------------------------------------------
PEAK_TFLOPS_BF16 = 990        # TFLOPS (BF16 tensor core)
PEAK_BW_TB_S = 3.35            # TB/s HBM bandwidth

PEAK_FLOPS = PEAK_TFLOPS_BF16 * 1e12   # FLOPS
PEAK_BW = PEAK_BW_TB_S * 1e12          # bytes/s
RIDGE_POINT = PEAK_FLOPS / PEAK_BW     # FLOPs/byte


# ---------------------------------------------------------------------------
# MoE configuration (match your benchmark)
# ---------------------------------------------------------------------------
NUM_EXPERTS = 31
HIDDEN_SIZE = 4096       # K dimension
INTERMEDIATE_SIZE = 1536 # N dimension
SEQ_LEN = 6144
BYTES_PER_ELEM = 2       # BF16


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


def make_skewed_distribution():
    head = [4000, 1200, 500, 200, 100, 20, 4]
    tail_count = NUM_EXPERTS - len(head)
    remaining = SEQ_LEN - sum(head)
    base, extra = divmod(remaining, tail_count)
    tail = [base + 1] * extra + [base] * (tail_count - extra)
    return head + tail


def make_even_distribution():
    base, extra = divmod(SEQ_LEN, NUM_EXPERTS)
    return [base + 1] * extra + [base] * (NUM_EXPERTS - extra)


# ---------------------------------------------------------------------------
# Roofline math
# ---------------------------------------------------------------------------


def gemm_arithmetic_intensity(M, N, K, elem_bytes=BYTES_PER_ELEM):
    """Arithmetic intensity for GEMM: [M,K] x [K,N] -> [M,N]."""
    flops = 2 * M * N * K
    # bytes: read A[M,K] + read B[K,N] + write C[M,N]
    mem_bytes = elem_bytes * (M * K + K * N + M * N)
    return flops / mem_bytes if mem_bytes > 0 else 0


def roofline_attainable(ai):
    """Attainable FLOPS at a given arithmetic intensity."""
    return min(PEAK_FLOPS, PEAK_BW * ai)


def gemm_flops(M, N, K):
    return 2 * M * N * K


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def main():
    skewed = make_skewed_distribution()
    even = make_even_distribution()

    K, N = HIDDEN_SIZE, INTERMEDIATE_SIZE

    # Compute AI for each expert under both distributions
    skewed_ai = [gemm_arithmetic_intensity(m, N, K) for m in skewed]
    even_ai = [gemm_arithmetic_intensity(m, N, K) for m in even]

    # Attainable FLOPS for each
    skewed_att = [roofline_attainable(ai) for ai in skewed_ai]
    even_att = [roofline_attainable(ai) for ai in even_ai]

    # Efficiency = attainable / peak_compute
    skewed_eff = [a / PEAK_FLOPS * 100 for a in skewed_att]
    even_eff = [a / PEAK_FLOPS * 100 for a in even_att]

    # ---- Print table ----
    print(f"GPU: {PEAK_TFLOPS_BF16} TFLOPS BF16, {PEAK_BW_TB_S} TB/s HBM")
    print(f"Ridge point: {RIDGE_POINT:.1f} FLOPs/byte")
    print(f"GEMM shape per expert: [M, {K}] x [{K}, {N}]")
    print()
    print(f"{'Expert':>6} | {'M(skew)':>8} {'AI':>10} {'Eff%':>6} | "
          f"{'M(even)':>8} {'AI':>10} {'Eff%':>6}")
    print("-" * 72)
    for i in range(NUM_EXPERTS):
        print(f"{i:>6} | {skewed[i]:>8} {skewed_ai[i]:>10.1f} {skewed_eff[i]:>5.1f}% | "
              f"{even[i]:>8} {even_ai[i]:>10.1f} {even_eff[i]:>5.1f}%")

    # Weighted average efficiency (weighted by FLOPs)
    skewed_total_flops = sum(gemm_flops(m, N, K) for m in skewed)
    even_total_flops = sum(gemm_flops(m, N, K) for m in even)
    skewed_weighted_eff = sum(
        gemm_flops(m, N, K) / skewed_total_flops * eff
        for m, eff in zip(skewed, skewed_eff)
    )
    even_weighted_eff = sum(
        gemm_flops(m, N, K) / even_total_flops * eff
        for m, eff in zip(even, even_eff)
    )
    print()
    print(f"Weighted avg efficiency — Skewed: {skewed_weighted_eff:.1f}%,  "
          f"Even: {even_weighted_eff:.1f}%")

    # ---- Roofline plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left panel: classic roofline ---
    ai_range = np.logspace(-1, 4, 500)
    roofline = np.minimum(PEAK_FLOPS, PEAK_BW * ai_range)

    ax1.loglog(ai_range, roofline / 1e12, 'k-', linewidth=2.5,
              label='Roofline')
    ax1.axvline(RIDGE_POINT, color='gray', linestyle=':', alpha=0.5,
               label=f'Ridge = {RIDGE_POINT:.0f} FLOPs/B')

    # Plot each expert as a dot
    for i, (ai, m) in enumerate(zip(skewed_ai, skewed)):
        att = roofline_attainable(ai)
        ax1.plot(ai, att / 1e12, 'rv', markersize=8 if m > 100 else 5,
                alpha=0.8)
    for i, (ai, m) in enumerate(zip(even_ai, even)):
        att = roofline_attainable(ai)
        ax1.plot(ai, att / 1e12, 'bs', markersize=6, alpha=0.6)

    # Invisible points for legend
    ax1.plot([], [], 'rv', markersize=8, label='Skewed experts')
    ax1.plot([], [], 'bs', markersize=6, label='Uniform experts')

    # Annotate a few key skewed experts
    for i, (ai, m) in enumerate(zip(skewed_ai, skewed)):
        if m in (skewed[0], skewed[-1]):
            att = roofline_attainable(ai)
            label = f'M={m}'
            ax1.annotate(label, (ai, att / 1e12),
                        textcoords='offset points', xytext=(8, -5),
                        fontsize=8, color='red')

    ax1.set_xlabel('Arithmetic Intensity (FLOPs / Byte)', fontsize=12)
    ax1.set_ylabel('Attainable Performance (TFLOPS)', fontsize=12)
    ax1.set_title('Roofline: Per-Expert GEMM\n'
                  f'H100 SXM — BF16 — [{"M"}, {K}] × [{K}, {N}]',
                  fontsize=13)
    ax1.legend(fontsize=10, loc='lower right')
    ax1.set_xlim(0.5, 5000)
    ax1.set_ylim(1, PEAK_TFLOPS_BF16 * 2)
    ax1.grid(True, which='both', alpha=0.3)

    # Fill memory-bound and compute-bound regions
    ax1.fill_between(ai_range, roofline / 1e12, 0.01,
                     where=(ai_range < RIDGE_POINT),
                     alpha=0.05, color='blue')
    ax1.fill_between(ai_range, roofline / 1e12, 0.01,
                     where=(ai_range >= RIDGE_POINT),
                     alpha=0.05, color='red')
    ax1.text(2, 3, 'Memory\nBound', fontsize=11, color='blue', alpha=0.6)
    ax1.text(1500, 50, 'Compute\nBound', fontsize=11, color='red', alpha=0.6)

    # --- Right panel: efficiency bar chart ---
    x_pos = np.arange(NUM_EXPERTS)
    width = 0.35

    bars_s = ax2.bar(x_pos - width / 2, skewed_eff, width,
                     label='Skewed', color='#d62728', alpha=0.8)
    bars_e = ax2.bar(x_pos + width / 2, even_eff, width,
                     label='Uniform', color='#1f77b4', alpha=0.8)

    ax2.axhline(100, color='gray', linestyle='--', alpha=0.4,
               label='100% (compute ceiling)')
    ax2.set_xlabel('Expert ID', fontsize=12)
    ax2.set_ylabel('Roofline Efficiency (%)', fontsize=12)
    ax2.set_title('Per-Expert Efficiency\n'
                  f'Skewed avg={skewed_weighted_eff:.1f}%, '
                  f'Uniform avg={even_weighted_eff:.1f}%',
                  fontsize=13)
    ax2.set_xticks(x_pos[::2])
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 110)
    ax2.grid(True, axis='y', alpha=0.3)

    # Add token counts on top of skewed bars for head experts
    for i, (bar, m) in enumerate(zip(bars_s, skewed)):
        if m >= 100 or m <= 5:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{m}', ha='center', va='bottom', fontsize=6,
                    color='red', rotation=90)

    plt.tight_layout()
    out_path = 'roofline_moe.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {out_path}")


if __name__ == '__main__':
    main()
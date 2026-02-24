#!/usr/bin/env python3
"""
Experiment 1: CTA Tile Waste Analysis for MoE Grouped GEMM

Calculates theoretical tile waste and GEMM kernel time for:
- Different batch sizes (decode=1, small prefill, large prefill)
- Different token distributions (uniform, zipf-skewed, extreme)
- Different CTA_M tile sizes (M32, M64, M128, M256)

Model: Qwen3-30B-A3B-NVFP4
  128 experts, top_k=8, hidden=2048, inter=768
  GEMM1: M×K=M×2048, N=768*2=1536 (gated)
  GEMM2: M×K=M×768, N=2048

GPU: RTX 5090 (84 SMs)
"""
import numpy as np
from collections import defaultdict

NUM_EXPERTS = 128
TOP_K = 8
HIDDEN = 2048
INTER = 768
NUM_SMS = 84  # RTX 5090

# GEMM1: [expanded_M, 2048] × [2048, 1536] → [expanded_M, 1536]
# GEMM2: [expanded_M, 768]  × [768, 2048]  → [expanded_M, 2048]
GEMM1_K, GEMM1_N = 2048, 1536  # gated: inter*2
GEMM2_K, GEMM2_N = 768, 2048

# CTA_N tile sizes (typical for these N dimensions)
CTA_N_GEMM1 = 128  # 1536 / 128 = 12 N-tiles per expert
CTA_N_GEMM2 = 128  # 2048 / 128 = 16 N-tiles per expert


def generate_uniform_distribution(batch_size, num_experts, top_k):
    """Every expert gets exactly batch*top_k/num_experts tokens."""
    total = batch_size * top_k
    base = total // num_experts
    remainder = total % num_experts
    counts = np.full(num_experts, base)
    counts[:remainder] += 1
    return counts


def generate_zipf_distribution(batch_size, num_experts, top_k, s=1.0):
    """Zipf-like skewed distribution (some experts get many tokens, others few)."""
    total = batch_size * top_k
    ranks = np.arange(1, num_experts + 1, dtype=np.float64)
    weights = 1.0 / (ranks ** s)
    weights /= weights.sum()
    counts = np.round(weights * total).astype(int)
    # Fix rounding to match total
    diff = total - counts.sum()
    if diff > 0:
        counts[:diff] += 1
    elif diff < 0:
        for i in range(num_experts - 1, -1, -1):
            take = min(-diff, counts[i])
            counts[i] -= take
            diff += take
            if diff == 0:
                break
    return counts


def generate_extreme_distribution(batch_size, num_experts, top_k):
    """Extreme: 8 experts get all tokens (like decode batch=1)."""
    total = batch_size * top_k
    counts = np.zeros(num_experts, dtype=int)
    # Spread across top_k experts (each token picks 8 unique experts)
    active = min(total, num_experts)
    if batch_size <= 8:
        # For very small batches, each token picks 8 experts
        # With batch=1: exactly 8 experts get 1 token each
        active = min(batch_size * top_k, num_experts)
        per_expert = total // active
        remainder = total % active
        counts[:active] = per_expert
        counts[:remainder] += 1
    else:
        # For larger batches, still skewed but more spread
        counts = generate_zipf_distribution(batch_size, num_experts, top_k, s=1.5)
    return counts


def compute_tiles(expert_counts, cta_m, cta_n, gemm_n):
    """Compute total CTA tiles, wasted compute, and per-expert breakdown."""
    n_tiles = int(np.ceil(gemm_n / cta_n))
    total_tiles = 0
    total_useful_compute = 0
    total_padded_compute = 0
    per_expert = []

    for i, count in enumerate(expert_counts):
        if count == 0:
            per_expert.append((0, 0, 0))
            continue
        m_tiles = int(np.ceil(count / cta_m))
        expert_tiles = m_tiles * n_tiles
        total_tiles += expert_tiles

        useful_m = count * gemm_n  # actual useful M×N elements
        padded_m = m_tiles * cta_m * gemm_n  # padded M×N elements
        total_useful_compute += useful_m
        total_padded_compute += padded_m
        per_expert.append((count, m_tiles, expert_tiles))

    waste_pct = (1 - total_useful_compute / max(total_padded_compute, 1)) * 100
    waves = np.ceil(total_tiles / NUM_SMS)
    tail_idle = int(waves * NUM_SMS - total_tiles)  # idle SM-slots in last wave
    tail_pct = tail_idle / (waves * NUM_SMS) * 100 if waves > 0 else 0

    return {
        'total_tiles': total_tiles,
        'waves': int(waves),
        'tail_idle_slots': tail_idle,
        'tail_waste_pct': tail_pct,
        'padding_waste_pct': waste_pct,
        'per_expert': per_expert,
    }


def estimate_kernel_time_us(tiles_info, time_per_tile_us=1.0):
    """Rough kernel time estimate based on waves × time_per_tile."""
    return tiles_info['waves'] * time_per_tile_us


def run_analysis():
    batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256, 512]
    cta_m_sizes = [32, 64, 128, 256]
    distributions = {
        'uniform': generate_uniform_distribution,
        'zipf_s1': lambda bs, ne, tk: generate_zipf_distribution(bs, ne, tk, s=1.0),
        'zipf_s1.5': lambda bs, ne, tk: generate_zipf_distribution(bs, ne, tk, s=1.5),
        'extreme': generate_extreme_distribution,
    }

    print("=" * 120)
    print("EXPERIMENT 1: CTA Tile Waste Analysis for MoE Grouped GEMM")
    print(f"Model: Qwen3-30B-A3B-NVFP4 | 128 experts, top_k=8 | GPU: RTX 5090 (84 SMs)")
    print(f"GEMM1: M×{GEMM1_K} × {GEMM1_K}×{GEMM1_N} | GEMM2: M×{GEMM2_K} × {GEMM2_K}×{GEMM2_N}")
    print("=" * 120)

    # === Part 1: Expert count distributions ===
    print("\n" + "=" * 120)
    print("PART 1: Token Distribution Across Experts")
    print("=" * 120)
    for bs in [1, 8, 32, 128, 512]:
        total = bs * TOP_K
        print(f"\n--- Batch={bs}, Total expanded tokens={total} ---")
        for dist_name, dist_fn in distributions.items():
            counts = dist_fn(bs, NUM_EXPERTS, TOP_K)
            active = np.sum(counts > 0)
            print(f"  {dist_name:12s}: active={active:3d}/128  "
                  f"min={counts[counts > 0].min() if active > 0 else 0:4d}  "
                  f"max={counts.max():4d}  "
                  f"mean={counts[counts > 0].mean() if active > 0 else 0:6.1f}  "
                  f"std={counts[counts > 0].std() if active > 0 else 0:6.1f}  "
                  f"cv={counts[counts > 0].std() / counts[counts > 0].mean() if active > 0 and counts[counts > 0].mean() > 0 else 0:.2f}")

    # === Part 2: Tile counts and waste for GEMM1 ===
    print("\n" + "=" * 120)
    print("PART 2: GEMM1 Tile Analysis (per batch size × distribution × CTA_M)")
    print("=" * 120)
    print(f"\n{'Batch':>5s}  {'Dist':>10s}  {'CTA_M':>5s}  {'Tiles':>6s}  {'Waves':>5s}  "
          f"{'Tail%':>6s}  {'Pad%':>6s}  {'TotalWaste%':>11s}  {'RelToM128':>9s}")
    print("-" * 90)

    gemm1_results = defaultdict(dict)
    for bs in batch_sizes:
        for dist_name, dist_fn in distributions.items():
            counts = dist_fn(bs, NUM_EXPERTS, TOP_K)
            m128_tiles = None
            for cta_m in cta_m_sizes:
                info = compute_tiles(counts, cta_m, CTA_N_GEMM1, GEMM1_N)
                if cta_m == 128:
                    m128_tiles = info['total_tiles']
                rel = info['total_tiles'] / m128_tiles if m128_tiles and m128_tiles > 0 else 0
                total_waste = info['tail_waste_pct'] + info['padding_waste_pct']
                print(f"{bs:>5d}  {dist_name:>10s}  M{cta_m:<4d} {info['total_tiles']:>6d}  "
                      f"{info['waves']:>5d}  {info['tail_waste_pct']:>5.1f}%  "
                      f"{info['padding_waste_pct']:>5.1f}%  {total_waste:>10.1f}%  "
                      f"{rel:>8.2f}x")
                gemm1_results[(bs, dist_name)][cta_m] = info

    # === Part 3: Improvement from M128→M32 (what the optimization does) ===
    print("\n" + "=" * 120)
    print("PART 3: Tile Reduction from M128 → M32 (= proxy for GEMM speedup)")
    print("  (Fewer tiles = faster kernel, IF compute-bound)")
    print("=" * 120)
    print(f"\n{'Batch':>5s}  {'Dist':>10s}  {'M128_tiles':>10s}  {'M32_tiles':>10s}  "
          f"{'Reduction%':>10s}  {'M128_waves':>10s}  {'M32_waves':>10s}  {'WaveReduction%':>14s}")
    print("-" * 100)

    for bs in batch_sizes:
        for dist_name, dist_fn in distributions.items():
            counts = dist_fn(bs, NUM_EXPERTS, TOP_K)
            info128 = compute_tiles(counts, 128, CTA_N_GEMM1, GEMM1_N)
            info32 = compute_tiles(counts, 32, CTA_N_GEMM1, GEMM1_N)
            tile_red = (1 - info32['total_tiles'] / max(info128['total_tiles'], 1)) * 100
            wave_red = (1 - info32['waves'] / max(info128['waves'], 1)) * 100
            print(f"{bs:>5d}  {dist_name:>10s}  {info128['total_tiles']:>10d}  "
                  f"{info32['total_tiles']:>10d}  {tile_red:>9.1f}%  "
                  f"{info128['waves']:>10d}  {info32['waves']:>10d}  {wave_red:>13.1f}%")

    # === Part 4: Decode deep-dive (batch=1) ===
    print("\n" + "=" * 120)
    print("PART 4: Decode Deep-Dive (batch=1, 8 expanded tokens)")
    print("  Each token picks 8 experts → 8 experts get 1 token, 120 experts get 0")
    print("=" * 120)

    counts_decode = generate_extreme_distribution(1, NUM_EXPERTS, TOP_K)
    active = np.sum(counts_decode > 0)
    print(f"\n  Active experts: {active}/128")
    print(f"  Per-expert tokens: {counts_decode[counts_decode > 0]}")

    for gemm_name, gemm_n, cta_n in [("GEMM1", GEMM1_N, CTA_N_GEMM1), ("GEMM2", GEMM2_N, CTA_N_GEMM2)]:
        print(f"\n  {gemm_name} (N={gemm_n}):")
        n_tiles = int(np.ceil(gemm_n / cta_n))
        print(f"    N-tiles per expert: {n_tiles}")
        for cta_m in cta_m_sizes:
            info = compute_tiles(counts_decode, cta_m, cta_n, gemm_n)
            # Each active expert: ceil(1/CTA_M) = 1 M-tile × n_tiles = n_tiles tiles
            # So 8 experts × n_tiles tiles
            print(f"    CTA_M={cta_m:3d}: tiles={info['total_tiles']:4d}  waves={info['waves']:2d}  "
                  f"tail_idle={info['tail_idle_slots']:3d}/{NUM_SMS}  "
                  f"padding_waste={info['padding_waste_pct']:.1f}%  "
                  f"(each expert: 1 token in {cta_m}-row tile = {(cta_m-1)/cta_m*100:.0f}% wasted)")

    # === Part 5: Where does M32 actually help? ===
    print("\n" + "=" * 120)
    print("PART 5: Where M32/M64 ACTUALLY Helps (wave reduction > 10%)")
    print("=" * 120)
    print(f"\n{'Batch':>5s}  {'Dist':>10s}  {'M128→M64':>10s}  {'M128→M32':>10s}  {'Verdict':>30s}")
    print("-" * 75)

    for bs in batch_sizes:
        for dist_name, dist_fn in distributions.items():
            counts = dist_fn(bs, NUM_EXPERTS, TOP_K)
            info128 = compute_tiles(counts, 128, CTA_N_GEMM1, GEMM1_N)
            info64 = compute_tiles(counts, 64, CTA_N_GEMM1, GEMM1_N)
            info32 = compute_tiles(counts, 32, CTA_N_GEMM1, GEMM1_N)
            red64 = (1 - info64['waves'] / max(info128['waves'], 1)) * 100
            red32 = (1 - info32['waves'] / max(info128['waves'], 1)) * 100
            if info128['waves'] == info32['waves']:
                verdict = "NO HELP (same waves)"
            elif red32 < 10:
                verdict = f"MINIMAL ({red32:.0f}% wave reduction)"
            elif red32 < 25:
                verdict = f"MODERATE ({red32:.0f}% wave reduction)"
            else:
                verdict = f"SIGNIFICANT ({red32:.0f}% wave reduction)"
            print(f"{bs:>5d}  {dist_name:>10s}  {red64:>9.1f}%  {red32:>9.1f}%  {verdict:>30s}")

    # === Part 6: Compute vs Memory bound analysis ===
    print("\n" + "=" * 120)
    print("PART 6: Compute vs Memory Bound Analysis per Expert")
    print("  RTX 5090: ~1790 GB/s bandwidth, ~3352 TOPS FP4")
    print("  Arithmetic intensity threshold: 3352e12 / 1790e9 ≈ 1873 OPs/byte")
    print("=" * 120)

    bw_bytes_per_s = 1790e9      # ~1.79 TB/s
    fp4_tops = 3352e12           # FP4 tensor core TOPS (operations/s)
    ai_threshold = fp4_tops / bw_bytes_per_s  # ops/byte for compute-bound

    print(f"\n  Arithmetic Intensity threshold: {ai_threshold:.0f} ops/byte")
    print(f"\n  {'M_expert':>8s}  {'GEMM':>5s}  {'FLOPs':>12s}  {'Bytes_read':>12s}  "
          f"{'AI(ops/B)':>10s}  {'Bound':>12s}  {'Tile_helps?':>12s}")
    print("-" * 85)

    for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        for gemm_name, K, N in [("GEMM1", GEMM1_K, GEMM1_N), ("GEMM2", GEMM2_K, GEMM2_N)]:
            # FP4: each element = 0.5 bytes
            flops = 2 * m * K * N  # standard GEMM FLOPs
            # Weight: K×N in FP4 = K*N*0.5 bytes (+ scale factors, ~2% overhead)
            # Activation: M×K in FP4 = M*K*0.5 bytes
            bytes_w = K * N * 0.5  # weight tensor
            bytes_a = m * K * 0.5  # activation tensor
            bytes_total = bytes_w + bytes_a  # ignoring output write for simplicity
            ai = flops / bytes_total
            bound = "COMPUTE" if ai > ai_threshold else "MEMORY"
            tile_helps = "YES" if bound == "COMPUTE" else "NO (BW-limited)"
            print(f"{m:>8d}  {gemm_name:>5s}  {flops:>12,d}  {bytes_total:>12,.0f}  "
                  f"{ai:>10.0f}  {bound:>12s}  {tile_helps:>12s}")


if __name__ == '__main__':
    run_analysis()

#!/usr/bin/env python3
"""Figures for BF16+NVFP4 analysis — full 40-layer × 8790-expert data."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

BASE = Path(__file__).parent
OUT = BASE / "figures"; OUT.mkdir(exist_ok=True)

with open(BASE / "results" / "full_error_profile.json") as f:
    profile = json.load(f)
with open(BASE / "results" / "analysis_data.json") as f:
    analysis = json.load(f)

C_BF16 = "#0072B2"
C_NVFP4 = "#D55E00"
C_FP8 = "#E69F00"
C_MIXED = "#009E73"
C_W1 = "#0072B2"
C_W2 = "#D55E00"
C_RAND = "#999999"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ============================================================
# Collect per-layer stats from full profile
# ============================================================
layer_stats = {}
all_w1_gini, all_w2_gini = [], []
all_w1_top15, all_w2_top15 = [], []
all_w1_top15_w, all_w2_top15_w = [], []
all_w1_mae, all_w2_mae = [], []

for li in range(40):
    ld = profile[str(li)]
    w1_g, w2_g = [], []
    w1_t, w2_t = [], []
    w1_tw, w2_tw = [], []
    w1_m, w2_m = [], []
    for e in ld["experts"]:
        if e["proj"] == "w1":
            w1_g.append(e["gini"]); all_w1_gini.append(e["gini"])
            w1_t.append(e["top15_unweighted"]); all_w1_top15.append(e["top15_unweighted"])
            w1_tw.append(e["top15_weighted"]); all_w1_top15_w.append(e["top15_weighted"])
            w1_m.append(e["total_mae"]); all_w1_mae.append(e["total_mae"])
        else:
            w2_g.append(e["gini"]); all_w2_gini.append(e["gini"])
            w2_t.append(e["top15_unweighted"]); all_w2_top15.append(e["top15_unweighted"])
            w2_tw.append(e["top15_weighted"]); all_w2_top15_w.append(e["top15_weighted"])
            w2_m.append(e["total_mae"]); all_w2_mae.append(e["total_mae"])
    layer_stats[li] = {
        "w1_gini_mean": np.mean(w1_g) if w1_g else 0,
        "w2_gini_mean": np.mean(w2_g) if w2_g else 0,
        "w1_top15_mean": np.mean(w1_t) if w1_t else 15,
        "w2_top15_mean": np.mean(w2_t) if w2_t else 15,
        "w1_mae_mean": np.mean(w1_m) if w1_m else 0,
        "w2_mae_mean": np.mean(w2_m) if w2_m else 0,
        "n_experts": len(w1_g),
    }

# ============================================================
# Figure 1: Main analysis (2×2)
# ============================================================
fig = plt.figure(figsize=(7.2, 5.5))
gs = GridSpec(2, 2, hspace=0.42, wspace=0.38, left=0.10, right=0.97, top=0.94, bottom=0.08)

# --- (a) Error concentration by layer depth ---
ax = fig.add_subplot(gs[0, 0])
layers = np.arange(40)
w1_gini_by_layer = [layer_stats[li]["w1_gini_mean"] for li in range(40)]
w2_gini_by_layer = [layer_stats[li]["w2_gini_mean"] for li in range(40)]

ax.plot(layers, w1_gini_by_layer, "o-", color=C_W1, linewidth=1.2, markersize=3, label="W1 (gate_up)")
ax.plot(layers, w2_gini_by_layer, "s-", color=C_W2, linewidth=1.2, markersize=3, label="W2 (down)")
ax.axhline(0, color="gray", linewidth=0.4)

ax.annotate("Layer 0: W1 anomaly\n(Gini=0.49)", xy=(0, w1_gini_by_layer[0]),
            xytext=(8, 0.42), fontsize=6.5, color=C_W1,
            arrowprops=dict(arrowstyle="->", color=C_W1, lw=0.8))

ax.set_xlabel("Layer index")
ax.set_ylabel("Mean Gini coefficient of\nNVFP4 error distribution")
ax.set_xlim(-0.5, 39.5)
ax.legend(loc="upper right", frameon=False)
ax.set_title("(a) Error concentration by layer depth\n(all 8,790 expert instances)")

# --- (b) Gini histogram — all experts, all layers ---
ax = fig.add_subplot(gs[0, 1])
bins = np.linspace(0, 0.9, 35)
ax.hist(all_w1_gini, bins=bins, color=C_W1, alpha=0.5, label=f"W1 (n={len(all_w1_gini)}, med={np.median(all_w1_gini):.2f})")
ax.hist(all_w2_gini, bins=bins, color=C_W2, alpha=0.5, label=f"W2 (n={len(all_w2_gini)}, med={np.median(all_w2_gini):.2f})")
ax.axvline(np.median(all_w1_gini), color=C_W1, linestyle="--", linewidth=1.0)
ax.axvline(np.median(all_w2_gini), color=C_W2, linestyle="--", linewidth=1.0)
ax.set_xlabel("Gini coefficient")
ax.set_ylabel("Number of expert instances")
ax.legend(loc="upper right", frameon=False, fontsize=6.5)
ax.set_title("(b) NVFP4 error concentration\n(all 40 layers × active experts)")

# --- (c) Top-15% error captured by weight magnitude ---
ax = fig.add_subplot(gs[1, 0])
w1_top15_by_layer = [layer_stats[li]["w1_top15_mean"] for li in range(40)]
w2_top15_by_layer = [layer_stats[li]["w2_top15_mean"] for li in range(40)]

ax.plot(layers, w1_top15_by_layer, "o-", color=C_W1, linewidth=1.2, markersize=3, label="W1 (gate_up)")
ax.plot(layers, w2_top15_by_layer, "s-", color=C_W2, linewidth=1.2, markersize=3, label="W2 (down)")
ax.axhline(15, color="gray", linestyle=":", linewidth=0.8, label="Uniform baseline (15%)")

ax.annotate(f"Layer 0 W1: {w1_top15_by_layer[0]:.0f}%\n(1.9× uniform)",
            xy=(0, w1_top15_by_layer[0]), xytext=(8, 32), fontsize=6.5, color=C_W1,
            arrowprops=dict(arrowstyle="->", color=C_W1, lw=0.8))

ax.set_xlabel("Layer index")
ax.set_ylabel("% of NVFP4 error in top 15%\nchannels (by weight magnitude)")
ax.set_xlim(-0.5, 39.5)
ax.legend(loc="upper right", frameon=False, fontsize=6.5)
ax.set_title("(c) Weight-magnitude metric effectiveness\nby layer depth")

# --- (d) NVFP4 MAE by layer depth ---
ax = fig.add_subplot(gs[1, 1])
w1_mae_by_layer = [layer_stats[li]["w1_mae_mean"] for li in range(40)]
w2_mae_by_layer = [layer_stats[li]["w2_mae_mean"] for li in range(40)]

ax.plot(layers, w1_mae_by_layer, "o-", color=C_W1, linewidth=1.2, markersize=3, label="W1 (gate_up)")
ax.plot(layers, w2_mae_by_layer, "s-", color=C_W2, linewidth=1.2, markersize=3, label="W2 (down)")
ax.set_xlabel("Layer index")
ax.set_ylabel("Mean per-channel NVFP4 MAE\n(real activations)")
ax.set_xlim(-0.5, 39.5)
ax.legend(loc="upper left", frameon=False)
ax.set_title("(d) Absolute NVFP4 error by layer depth")

plt.savefig(OUT / "bf16_nvfp4_full_analysis.pdf", bbox_inches="tight")
plt.savefig(OUT / "bf16_nvfp4_full_analysis.png", bbox_inches="tight", dpi=300)
print(f"Saved {OUT / 'bf16_nvfp4_full_analysis.png'}")

# ============================================================
# Figure 2: Selection strategy + PPL curve (1×3)
# ============================================================
fig2, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.42})

ax = axes[0]
strategies = ["Weight\nmagnitude", "Random\n(3 seeds avg)"]
ppls_strat = [6.5499, 6.5953]
colors_s = [C_MIXED, C_RAND]
bars = ax.bar(range(len(strategies)), ppls_strat, color=colors_s, width=0.5)
ax.axhline(6.5987, color=C_NVFP4, linestyle="--", linewidth=0.8, label="Uniform NVFP4 (6.5987)")
ax.axhline(6.5116, color=C_BF16, linestyle="--", linewidth=0.8, label="Uniform BF16 (6.5116)")
ax.set_xticks(range(len(strategies)))
ax.set_xticklabels(strategies)
ax.set_ylabel("PPL (20-chunk, 15% BF16)")
ax.set_ylim(6.50, 6.62)
ax.legend(loc="upper right", frameon=False, fontsize=6)
ax.set_title("(a) Channel selection strategy\n(same 15% BF16 budget, 20-chunk)")
for bar, val in zip(bars, ppls_strat):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.002,
            f"{val:.4f}", ha="center", fontsize=7, fontweight="bold")
ax.annotate(f"Δ = {ppls_strat[1]-ppls_strat[0]:.4f}\n(16× more effective)",
            xy=(0.5, (ppls_strat[0]+ppls_strat[1])/2),
            fontsize=7, ha="center", color="red", fontweight="bold")

# --- (b) PPL vs BF16 fraction ---
ax = axes[1]
ppl_res = analysis["ppl_results"]
fracs = ppl_res["bf16_fractions"]
ppls = ppl_res["ppl_20chunk"]
baselines = ppl_res["baselines_20chunk"]

ax.plot(fracs, ppls, "o-", color=C_MIXED, linewidth=1.5, markersize=4, label="BF16+NVFP4")
ax.axhline(baselines["fp8"], color=C_FP8, linestyle="--", linewidth=0.8, label=f"FP8 ({baselines['fp8']:.4f})")
ax.axhline(baselines["nvfp4"], color=C_NVFP4, linestyle="--", linewidth=0.8, label=f"NVFP4 ({baselines['nvfp4']:.4f})")
ax.axhline(baselines["bf16"], color=C_BF16, linestyle="--", linewidth=0.8, label=f"BF16 ({baselines['bf16']:.4f})")
best_idx = np.argmin(ppls)
ax.plot(fracs[best_idx], ppls[best_idx], "s", color="red", markersize=7, zorder=5)
ax.annotate(f"{fracs[best_idx]}%: {ppls[best_idx]:.4f}",
            xy=(fracs[best_idx], ppls[best_idx]),
            xytext=(fracs[best_idx]+15, ppls[best_idx]-0.012),
            fontsize=7, fontweight="bold", color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
ax.set_xlabel("BF16 channel fraction (%)")
ax.set_ylabel("PPL (20-chunk)")
ax.set_xlim(-2, 105)
ax.legend(loc="upper right", frameon=False, fontsize=5.5)
ax.set_title("(b) PPL vs BF16 fraction\n(weight-magnitude selection)")

# --- (c) Number of active experts by layer ---
ax = axes[2]
n_experts_by_layer = [layer_stats[li]["n_experts"] for li in range(40)]
ax.bar(layers, n_experts_by_layer, color=C_MIXED, alpha=0.7, width=0.8)
ax.set_xlabel("Layer index")
ax.set_ylabel("Active experts\n(≥3 routed tokens)")
ax.set_xlim(-0.5, 39.5)
ax.set_title("(c) Expert utilization by depth\n(2048-token evaluation chunk)")

plt.savefig(OUT / "bf16_nvfp4_strategies.pdf", bbox_inches="tight")
plt.savefig(OUT / "bf16_nvfp4_strategies.png", bbox_inches="tight", dpi=300)
print(f"Saved {OUT / 'bf16_nvfp4_strategies.png'}")

print("Done.")

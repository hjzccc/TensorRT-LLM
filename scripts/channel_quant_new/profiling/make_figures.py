#!/usr/bin/env python3
"""Generate analysis figures from the full 145-chunk error profile."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROFILE_PATH = Path(__file__).parent / "error_profile.json"
FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

with open(PROFILE_PATH) as f:
    profile = json.load(f)

NUM_LAYERS = 40
C_W1 = "#0072B2"
C_W2 = "#D55E00"
C_HOT = "#D55E00"
C_COLD = "#56B4E9"
C_MIXED = "#009E73"
C_GRAY = "#999999"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})


def compute_gini(values):
    s = np.sort(values)
    n = len(s)
    t = s.sum()
    if t < 1e-15:
        return 0.0
    return float((2 * np.sum(np.arange(1, n + 1) * s) / (n * t)) - (n + 1) / n)


# ---------------------------------------------------------------------------
# Extract per-layer statistics
# ---------------------------------------------------------------------------
layer_indices = np.arange(NUM_LAYERS)

within_gini_w1 = []
within_gini_w2 = []
within_top15_w1 = []
within_top15_w2 = []
within_mae_w1 = []
within_mae_w2 = []

across_gini_w1 = []
across_top25_w1 = []
across_top10_w1 = []

for layer_idx in range(NUM_LAYERS):
    layer_data = profile[str(layer_idx)]
    experts = layer_data["experts"]

    w1_gini_vals, w2_gini_vals = [], []
    w1_top15_vals, w2_top15_vals = [], []
    w1_mae_vals, w2_mae_vals = [], []
    moe_contributions = []

    for expert in experts:
        routing_weight = expert["routing_weight_sum"]

        if expert["w1"]:
            w1_gini_vals.append(expert["w1"]["gini_unweighted"])
            w1_top15_vals.append(expert["w1"]["top15_by_mag_uw"])
            w1_mae_vals.append(expert["w1"]["total_mae"])
            moe_contributions.append(routing_weight * expert["w1"]["total_mae"])

        if expert["w2"]:
            w2_gini_vals.append(expert["w2"]["gini_unweighted"])
            w2_top15_vals.append(expert["w2"]["top15_by_mag_uw"])
            w2_mae_vals.append(expert["w2"]["total_mae"])

    within_gini_w1.append(np.mean(w1_gini_vals) if w1_gini_vals else 0)
    within_gini_w2.append(np.mean(w2_gini_vals) if w2_gini_vals else 0)
    within_top15_w1.append(np.mean(w1_top15_vals) if w1_top15_vals else 15)
    within_top15_w2.append(np.mean(w2_top15_vals) if w2_top15_vals else 15)
    within_mae_w1.append(np.mean(w1_mae_vals) if w1_mae_vals else 0)
    within_mae_w2.append(np.mean(w2_mae_vals) if w2_mae_vals else 0)

    contributions = np.array(moe_contributions)
    across_gini_w1.append(compute_gini(contributions) if len(contributions) > 1 else 0)
    if len(contributions) > 0:
        sorted_contribs = np.sort(contributions)[::-1]
        total_contrib = sorted_contribs.sum()
        n10 = max(1, len(sorted_contribs) // 10)
        n25 = max(1, len(sorted_contribs) // 4)
        across_top10_w1.append(sorted_contribs[:n10].sum() / total_contrib * 100 if total_contrib > 0 else 0)
        across_top25_w1.append(sorted_contribs[:n25].sum() / total_contrib * 100 if total_contrib > 0 else 0)
    else:
        across_top10_w1.append(0)
        across_top25_w1.append(0)


# ===========================================================================
# Compute routing-weighted total error per layer
# ===========================================================================
rw_total_w1 = []
rw_total_w2 = []
for layer_idx in range(NUM_LAYERS):
    experts = profile[str(layer_idx)]["experts"]
    rw_total_w1.append(sum(e["routing_weight_sum"] * e["w1"]["total_mae"] for e in experts if e["w1"]))
    rw_total_w2.append(sum(e["routing_weight_sum"] * e["w2"]["total_mae"] for e in experts if e["w2"]))

# ===========================================================================
# Figure 1: Within-Expert View (per-channel error distribution)
# ===========================================================================
fig1, axes1 = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"wspace": 0.38})

ax = axes1[0]
ax.plot(layer_indices, within_gini_w1, "o-", color=C_W1, markersize=3, linewidth=1.2, label="W1 (gate_up)")
ax.plot(layer_indices, within_gini_w2, "s-", color=C_W2, markersize=3, linewidth=1.2, label="W2 (down)")
ax.set_xlabel("Layer index")
ax.set_ylabel("Mean Gini coefficient")
ax.set_xlim(-0.5, 39.5); ax.set_ylim(0, 0.55)
ax.legend(loc="upper right", frameon=False)
ax.set_title("(a) Per-channel error\nconcentration within experts")

ax = axes1[1]
ax.plot(layer_indices, within_top15_w1, "o-", color=C_W1, markersize=3, linewidth=1.2, label="W1")
ax.plot(layer_indices, within_top15_w2, "s-", color=C_W2, markersize=3, linewidth=1.2, label="W2")
ax.axhline(15, color=C_GRAY, linestyle=":", linewidth=0.8, label="Random baseline")
ax.set_xlabel("Layer index")
ax.set_ylabel("% of error in top 15%\nchannels (by weight mag)")
ax.set_xlim(-0.5, 39.5); ax.set_ylim(13, 32)
ax.legend(loc="upper right", frameon=False)
ax.set_title("(b) Weight-magnitude metric\neffectiveness per layer")

ax = axes1[2]
ax.plot(layer_indices, rw_total_w1, "o-", color=C_W1, markersize=3, linewidth=1.2, label="W1 (gate_up)")
ax.set_xlabel("Layer index")
ax.set_ylabel("Routing-weighted total\nNVFP4 error (W1)", color=C_W1)
ax.tick_params(axis="y", labelcolor=C_W1)
ax.set_xlim(-0.5, 39.5)
ax2 = ax.twinx()
ax2.plot(layer_indices, rw_total_w2, "s-", color=C_W2, markersize=3, linewidth=1.2, label="W2 (down)")
ax2.set_ylabel("Routing-weighted total\nNVFP4 error (W2)", color=C_W2)
ax2.tick_params(axis="y", labelcolor=C_W2)
ax2.spines["right"].set_visible(True)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False, fontsize=6.5)
ax.set_title("(c) Routing-weighted NVFP4 error\nby depth (dual axes)")

fig1.savefig(FIG_DIR / "within_expert_view.pdf", bbox_inches="tight")
fig1.savefig(FIG_DIR / "within_expert_view.png", bbox_inches="tight", dpi=300)
print(f"Saved {FIG_DIR / 'within_expert_view.png'}")


# ===========================================================================
# Figure 2: Across-Expert View (routing-weighted MoE error concentration)
# ===========================================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"wspace": 0.38})

ax = axes2[0]
ax.plot(layer_indices, across_gini_w1, "o-", color=C_W1, markersize=3, linewidth=1.2)
ax.set_xlabel("Layer index")
ax.set_ylabel("Gini of routing-weighted\nexpert error contributions")
ax.set_xlim(-0.5, 39.5); ax.set_ylim(0, 0.85)
ax.set_title("(a) MoE-level error\nconcentration by depth")

ax = axes2[1]
ax.plot(layer_indices, across_top25_w1, "o-", color=C_HOT, markersize=3, linewidth=1.2, label="Top 25% experts")
ax.plot(layer_indices, across_top10_w1, "s-", color=C_W1, markersize=3, linewidth=1.2, label="Top 10% experts")
ax.axhline(25, color=C_GRAY, linestyle=":", linewidth=0.6)
ax.axhline(10, color=C_GRAY, linestyle=":", linewidth=0.6)
ax.set_xlabel("Layer index")
ax.set_ylabel("% of total MoE W1 error\nfrom top experts")
ax.set_xlim(-0.5, 39.5); ax.set_ylim(0, 90)
ax.legend(loc="lower right", frameon=False)
ax.set_title("(b) Hot experts dominate\nMoE output error")

ax = axes2[2]
for layer_show, color, label in [
    (20, C_W1, f"Layer 20 (Gini={across_gini_w1[20]:.2f})"),
    (0, C_COLD, f"Layer 0 (Gini={across_gini_w1[0]:.2f})"),
    (39, C_HOT, f"Layer 39 (Gini={across_gini_w1[39]:.2f})"),
]:
    experts = profile[str(layer_show)]["experts"]
    contribs = []
    for e in experts:
        if e["w1"]:
            contribs.append(e["routing_weight_sum"] * e["w1"]["total_mae"])
    contribs = np.sort(contribs)[::-1]
    cumulative = contribs.cumsum() / contribs.sum() * 100
    x_pct = np.arange(1, len(cumulative) + 1) / len(cumulative) * 100
    ax.plot(x_pct, cumulative, color=color, linewidth=1.2, label=label)
ax.plot([0, 100], [0, 100], "k--", linewidth=0.5, alpha=0.3)
ax.set_xlabel("% of experts (sorted by\nrouting-weighted error)")
ax.set_ylabel("Cumulative % of\ntotal MoE error")
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.legend(loc="lower right", frameon=False, fontsize=6.5)
ax.set_title("(c) Lorenz curves:\nexpert-level concentration")

fig2.savefig(FIG_DIR / "across_expert_view.pdf", bbox_inches="tight")
fig2.savefig(FIG_DIR / "across_expert_view.png", bbox_inches="tight", dpi=300)
print(f"Saved {FIG_DIR / 'across_expert_view.png'}")


# ===========================================================================
# Figure 3: Combined two-level view + PPL results
# ===========================================================================
fig3 = plt.figure(figsize=(7.2, 5.0))
from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 2, hspace=0.45, wspace=0.38, left=0.10, right=0.97, top=0.93, bottom=0.08)

# (a) Within vs Across Gini comparison
ax = fig3.add_subplot(gs[0, 0])
ax.plot(layer_indices, within_gini_w1, "o-", color=C_COLD, markersize=3, linewidth=1.2,
        label="Within-expert (per-channel)")
ax.plot(layer_indices, across_gini_w1, "s-", color=C_HOT, markersize=3, linewidth=1.2,
        label="Across-expert (routing-weighted)")
ax.set_xlabel("Layer index")
ax.set_ylabel("Gini coefficient (W1)")
ax.set_xlim(-0.5, 39.5); ax.set_ylim(0, 0.85)
ax.legend(loc="center right", frameon=False, fontsize=6.5)
ax.set_title("(a) Two levels of error concentration")

# (b) Channel selection: weight-magnitude vs random (20-chunk validated)
ax = fig3.add_subplot(gs[0, 1])
strategies = ["Weight\nmagnitude", "Random\n(3 seeds)"]
ppls = [6.5499, 6.5953]
colors = [C_MIXED, C_GRAY]
bars = ax.bar(range(2), ppls, color=colors, width=0.5)
ax.axhline(6.5987, color=C_HOT, linestyle="--", linewidth=0.8, label="Uniform NVFP4")
ax.axhline(6.5116, color=C_W1, linestyle="--", linewidth=0.8, label="Uniform BF16")
ax.set_xticks(range(2)); ax.set_xticklabels(strategies)
ax.set_ylabel("PPL (20-chunk, 15% BF16)")
ax.set_ylim(6.50, 6.62)
ax.legend(loc="upper right", frameon=False, fontsize=6)
ax.set_title("(b) Channel selection strategy\n(same 15% BF16 budget)")
for bar, val in zip(bars, ppls):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
            f"{val:.4f}", ha="center", fontsize=7, fontweight="bold")

# (c) PPL vs BF16 fraction
ax = fig3.add_subplot(gs[1, 0])
fracs = [0, 2, 5, 10, 15, 20, 30, 50, 100]
ppls_curve = [6.5987, 6.5892, 6.5766, 6.5549, 6.5499, 6.5681, 6.5534, 6.5658, 6.5116]
ax.plot(fracs, ppls_curve, "o-", color=C_MIXED, linewidth=1.5, markersize=4)
ax.axhline(6.5159, color="#E69F00", linestyle="--", linewidth=0.8, label="FP8 (6.5159)")
ax.axhline(6.5987, color=C_HOT, linestyle="--", linewidth=0.8, label="NVFP4 (6.5987)")
ax.axhline(6.5116, color=C_W1, linestyle="--", linewidth=0.8, label="BF16 (6.5116)")
best_idx = np.argmin(ppls_curve[:-1])
ax.plot(fracs[best_idx], ppls_curve[best_idx], "s", color="red", markersize=7, zorder=5)
ax.annotate(f"Best: {fracs[best_idx]}%\n({ppls_curve[best_idx]:.4f})",
            xy=(fracs[best_idx], ppls_curve[best_idx]),
            xytext=(fracs[best_idx] + 18, ppls_curve[best_idx] - 0.012),
            fontsize=7, fontweight="bold", color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
ax.set_xlabel("BF16 channel fraction (%)")
ax.set_ylabel("PPL (20-chunk)")
ax.set_xlim(-2, 105)
ax.legend(loc="upper right", frameon=False, fontsize=6)
ax.set_title("(c) PPL vs BF16 fraction\n(weight-magnitude selection)")

ax = fig3.add_subplot(gs[1, 1])
ax.plot(layer_indices, rw_total_w1, "o-", color=C_W1, markersize=3, linewidth=1.2, label="W1 (gate_up)")
ax.set_xlabel("Layer index")
ax.set_ylabel("Routing-weighted error (W1)", color=C_W1)
ax.tick_params(axis="y", labelcolor=C_W1)
ax.set_xlim(-0.5, 39.5)
ax_d2 = ax.twinx()
ax_d2.plot(layer_indices, rw_total_w2, "s-", color=C_W2, markersize=3, linewidth=1.2, label="W2 (down)")
ax_d2.set_ylabel("Routing-weighted error (W2)", color=C_W2)
ax_d2.tick_params(axis="y", labelcolor=C_W2)
ax_d2.spines["right"].set_visible(True)
lines_d1, labels_d1 = ax.get_legend_handles_labels()
lines_d2, labels_d2 = ax_d2.get_legend_handles_labels()
ax.legend(lines_d1 + lines_d2, labels_d1 + labels_d2, loc="upper left", frameon=False, fontsize=6.5)
ax.set_title("(d) Routing-weighted NVFP4 error\nby depth (Σ rw × MAE)")

fig3.savefig(FIG_DIR / "combined_analysis.pdf", bbox_inches="tight")
fig3.savefig(FIG_DIR / "combined_analysis.png", bbox_inches="tight", dpi=300)
print(f"Saved {FIG_DIR / 'combined_analysis.png'}")

# ---------------------------------------------------------------------------
# Print summary statistics
# ---------------------------------------------------------------------------
print("\n=== Summary (full WikiText-2 test, 145 chunks, 296,960 tokens) ===")
print(f"Total expert instances profiled: {sum(profile[str(li)]['n_profiled_experts'] for li in range(NUM_LAYERS))}")
print(f"\nWithin-expert Gini (per-channel error uniformity):")
print(f"  W1: mean={np.mean(within_gini_w1):.3f}  W2: mean={np.mean(within_gini_w2):.3f}")
print(f"\nAcross-expert Gini (routing-weighted MoE error concentration):")
print(f"  W1: mean={np.mean(across_gini_w1):.3f}")
print(f"\nW1 MAE by depth: layer 0={within_mae_w1[0]:.4f}, layer 39={within_mae_w1[39]:.4f} ({within_mae_w1[39]/within_mae_w1[0]:.1f}× growth)")
print(f"W2 MAE by depth: layer 0={within_mae_w2[0]:.4f}, layer 39={within_mae_w2[39]:.4f} ({within_mae_w2[39]/max(within_mae_w2[0], 1e-8):.1f}× growth)")

print("\nDone.")

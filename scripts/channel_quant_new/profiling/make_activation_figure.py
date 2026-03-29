#!/usr/bin/env python3
"""Figure: Activation distribution shift explains NVFP4 error growth with depth."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent
FIG_DIR = BASE.parent / "figures"; FIG_DIR.mkdir(exist_ok=True)

with open(BASE / "activation_distributions.json") as f:
    data = json.load(f)

C_EARLY = "#56B4E9"
C_MID = "#009E73"
C_LATE = "#D55E00"
C_W1 = "#0072B2"
C_W2 = "#D55E00"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

layers_to_show = [0, 20, 39]
colors = [C_EARLY, C_MID, C_LATE]
labels = ["Layer 0", "Layer 20", "Layer 39"]

fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), gridspec_kw={"hspace": 0.45, "wspace": 0.35})

# --- Row 1: W1 input (hidden state) distributions ---
for col, (li, color, label) in enumerate(zip(layers_to_show, colors, labels)):
    ax = axes[0, col]
    d = data[str(li)]
    edges = np.array(d["w1_edges"])
    hist = np.array(d["w1_hist"])
    centers = (edges[:-1] + edges[1:]) / 2
    widths = edges[1:] - edges[:-1]
    hist_density = hist / (hist.sum() * widths[0])
    ax.bar(centers, hist_density, width=widths[0], color=color, alpha=0.7, edgecolor="none")
    ax.set_xlabel("|activation value|")
    if col == 0:
        ax.set_ylabel("Density (W1 input)")
    ax.set_title(f"{label}\namax={d['w1_amax']:.1f}  mean={d['w1_mean']:.3f}")
    ax.set_xlim(0, min(d["w1_amax"] * 1.1, 60))

# --- Row 2: W2 input (SwiGLU output) distributions ---
for col, (li, color, label) in enumerate(zip(layers_to_show, colors, labels)):
    ax = axes[1, col]
    d = data[str(li)]
    edges = np.array(d["w2_edges"])
    hist = np.array(d["w2_hist"])
    centers = (edges[:-1] + edges[1:]) / 2
    widths = edges[1:] - edges[:-1]
    hist_density = hist / (hist.sum() * widths[0])
    ax.bar(centers, hist_density, width=widths[0], color=color, alpha=0.7, edgecolor="none")
    ax.set_xlabel("|activation value|")
    if col == 0:
        ax.set_ylabel("Density (W2 input)")
    near0 = d["w2_near_zero_frac"]
    ax.set_title(f"{label}\namax={d['w2_amax']:.2f}  mean={d['w2_mean']:.4f}  near-0={near0:.0%}")
    ax.set_xlim(0, min(d["w2_amax"] * 1.1, 8))

fig.savefig(FIG_DIR / "activation_distributions.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "activation_distributions.png", bbox_inches="tight", dpi=300)
print(f"Saved {FIG_DIR / 'activation_distributions.png'}")

# --- Summary figure: activation stats by depth ---
fig2, axes2 = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"wspace": 0.38})

all_layers = sorted([int(k) for k in data.keys()])

ax = axes2[0]
w1_amax = [data[str(li)]["w1_amax"] for li in all_layers]
w2_amax = [data[str(li)]["w2_amax"] for li in all_layers]
ax.plot(all_layers, w1_amax, "o-", color=C_W1, markersize=4, linewidth=1.2, label="W1 input")
ax.set_ylabel("Activation amax (W1)", color=C_W1)
ax.tick_params(axis="y", labelcolor=C_W1)
ax2r = ax.twinx()
ax2r.plot(all_layers, w2_amax, "s-", color=C_W2, markersize=4, linewidth=1.2, label="W2 input")
ax2r.set_ylabel("Activation amax (W2)", color=C_W2)
ax2r.tick_params(axis="y", labelcolor=C_W2)
ax2r.spines["right"].set_visible(True)
lines1, lab1 = ax.get_legend_handles_labels()
lines2, lab2 = ax2r.get_legend_handles_labels()
ax.legend(lines1 + lines2, lab1 + lab2, loc="upper left", frameon=False, fontsize=6.5)
ax.set_xlabel("Layer index")
ax.set_title("(a) Activation amax by depth")

ax = axes2[1]
w1_mean = [data[str(li)]["w1_mean"] for li in all_layers]
w2_mean = [data[str(li)]["w2_mean"] for li in all_layers]
ax.plot(all_layers, w1_mean, "o-", color=C_W1, markersize=4, linewidth=1.2, label="W1 input")
ax.set_ylabel("Activation mean |x| (W1)", color=C_W1)
ax.tick_params(axis="y", labelcolor=C_W1)
ax2r = ax.twinx()
ax2r.plot(all_layers, w2_mean, "s-", color=C_W2, markersize=4, linewidth=1.2, label="W2 input")
ax2r.set_ylabel("Activation mean |x| (W2)", color=C_W2)
ax2r.tick_params(axis="y", labelcolor=C_W2)
ax2r.spines["right"].set_visible(True)
lines1, lab1 = ax.get_legend_handles_labels()
lines2, lab2 = ax2r.get_legend_handles_labels()
ax.legend(lines1 + lines2, lab1 + lab2, loc="upper left", frameon=False, fontsize=6.5)
ax.set_xlabel("Layer index")
ax.set_title("(b) Activation mean by depth")

ax = axes2[2]
w2_nz = [data[str(li)]["w2_near_zero_frac"] * 100 for li in all_layers]
ax.plot(all_layers, w2_nz, "s-", color=C_W2, markersize=4, linewidth=1.2)
ax.set_xlabel("Layer index")
ax.set_ylabel("W2 input values near zero (%)")
ax.set_title("(c) W2 sparsity collapses\nwith depth")
ax.annotate(f"Layer 0: {w2_nz[0]:.0f}% near-zero\n→ most values free to quantize",
            xy=(all_layers[0], w2_nz[0]), xytext=(10, w2_nz[0] - 10),
            fontsize=6.5, arrowprops=dict(arrowstyle="->", color=C_W2, lw=0.8), color=C_W2)
ax.annotate(f"Layer 39: {w2_nz[-1]:.0f}%\n→ all values incur error",
            xy=(all_layers[-1], w2_nz[-1]), xytext=(25, w2_nz[-1] + 15),
            fontsize=6.5, arrowprops=dict(arrowstyle="->", color=C_W2, lw=0.8), color=C_W2)

fig2.savefig(FIG_DIR / "activation_stats_by_depth.pdf", bbox_inches="tight")
fig2.savefig(FIG_DIR / "activation_stats_by_depth.png", bbox_inches="tight", dpi=300)
print(f"Saved {FIG_DIR / 'activation_stats_by_depth.png'}")

print("Done.")

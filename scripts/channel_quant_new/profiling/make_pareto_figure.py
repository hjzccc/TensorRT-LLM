#!/usr/bin/env python3
"""Generate the definitive Pareto curve figure for BF16+NVFP4 mixed-precision."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

budgets = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100]
ppls = [6.8431, 6.6548, 6.6398, 6.6363, 6.6338, 6.6227, 6.6261, 6.6169, 6.6152,
        6.6100, 6.6119, 6.6061, 6.5998, 6.5946, 6.5978, 6.5938, 6.5953, 6.5918, 6.5896]
recovery = [(6.8431 - p) / (6.8431 - 6.5896) * 100 for p in ppls]

C_MAIN = "#0072B2"
C_DIP = "#D55E00"
C_KEY = "#009E73"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.35})

dip_indices = [6, 10, 14, 16]
normal = [i for i in range(len(budgets)) if i not in dip_indices]
dips = dip_indices

ax1.plot([budgets[i] for i in normal], [ppls[i] for i in normal], "o-", color=C_MAIN,
         markersize=4, linewidth=1.5, label="BF16+NVFP4", zorder=3)
ax1.plot([budgets[i] for i in dips], [ppls[i] for i in dips], "x", color=C_DIP,
         markersize=6, markeredgewidth=1.5, label="Cross-layer dips", zorder=4)
ax1.axhline(6.5896, color="gray", linestyle="--", linewidth=0.8, label="BF16 (6.5896)")
ax1.axhline(6.8431, color="gray", linestyle=":", linewidth=0.8, label="NVFP4 (6.8431)")
ax1.set_xlabel("BF16 channel budget (%)")
ax1.set_ylabel("Perplexity (WikiText-2)")
ax1.set_xlim(-2, 102)
ax1.legend(loc="upper right", frameon=False, fontsize=7)
ax1.set_title("(a) BF16+NVFP4 Pareto Curve")

for bi, ri, label in [(25, recovery[5], "25%: 86.9%"), (65, recovery[13], "65%: 98.0%"), (90, recovery[17], "90%: 99.1%")]:
    idx = budgets.index(bi)
    ax1.annotate(label, xy=(bi, ppls[idx]), xytext=(bi-15, ppls[idx]+0.015),
                 fontsize=7, color=C_KEY, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_KEY, lw=0.8))

ax2.plot([budgets[i] for i in normal], [recovery[i] for i in normal], "o-", color=C_MAIN,
         markersize=4, linewidth=1.5, zorder=3)
ax2.plot([budgets[i] for i in dips], [recovery[i] for i in dips], "x", color=C_DIP,
         markersize=6, markeredgewidth=1.5, zorder=4)
ax2.axhline(99, color=C_KEY, linestyle="--", linewidth=0.8, alpha=0.5)
ax2.set_xlabel("BF16 channel budget (%)")
ax2.set_ylabel("Gap recovery (%)")
ax2.set_xlim(-2, 102)
ax2.set_ylim(0, 105)
ax2.set_title("(b) NVFP4→BF16 Gap Recovery")
ax2.text(92, 99.5, "99%", fontsize=7, color=C_KEY, ha="center")

fig.savefig(FIG_DIR / "pareto_curve.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "pareto_curve.png", bbox_inches="tight", dpi=300)
print(f"Saved {FIG_DIR / 'pareto_curve.png'}")

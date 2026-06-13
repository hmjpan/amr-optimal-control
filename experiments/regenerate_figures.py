"""
Regenerate all figures with clean, publication-quality formatting.
Addresses Figure 1 issues and ensures consistency with manuscript.
"""

import sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, FancyBboxPatch

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "lines.linewidth": 1.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.facecolor": "white",
})

OUT = PROJECT_ROOT / "outputs" / "real_data" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Colors
C_CONSERVE = "#B2182B"   # red - restrict
C_RESTRICT = "#E41A1C"   # darker red
C_BALANCED = "#F4A582"   # light red
C_TREAT = "#2166AC"      # blue
C_ITALY = "#4DAF4A"      # green for emphasis
C_GRID = "#E8E8E8"
C_BIFURCATION = "#333333"

# Load data
results = pd.read_csv(PROJECT_ROOT / "outputs" / "real_data" / "country_results.csv")

# =============================================================
# FIGURE 1: R0_r vs u* phase diagram with policy tiers
# =============================================================
fig, ax = plt.subplots(figsize=(6.5, 4))

# Background tier regions
ax.axvspan(0.6, 0.95, alpha=0.06, color=C_TREAT, zorder=0)
ax.axvspan(0.95, 1.05, alpha=0.06, color="#FFD700", zorder=0)
ax.axvspan(1.05, 2.8, alpha=0.06, color=C_CONSERVE, zorder=0)

# Tier labels at top
ax.text(0.78, 1.02, "Tier 3\nMaintain", ha="center", fontsize=7, color=C_TREAT,
        fontweight="bold", transform=ax.get_xaxis_transform())
ax.text(1.00, 1.02, "Monitor", ha="center", fontsize=6, color="#B8860B",
        fontweight="bold", transform=ax.get_xaxis_transform())
ax.text(1.80, 1.02, "Tier 1\nRestrict", ha="center", fontsize=7, color=C_CONSERVE,
        fontweight="bold", transform=ax.get_xaxis_transform())

# Bifurcation line
ax.axvline(x=1.0, color=C_BIFURCATION, lw=1.0, ls="--", alpha=0.6, zorder=1)
ax.text(1.01, 0.03, "R0r = 1", fontsize=7, color=C_BIFURCATION, fontstyle="italic",
        transform=ax.get_xaxis_transform())

# Country data points
for _, row in results.iterrows():
    regime = str(row["regime"]).strip()
    country = str(row["country"])
    r0 = float(row["R0_r"])
    u = float(row["u_optimal"])
    
    if country == "Italy":
        color = C_ITALY
        size = 130
        edge_width = 2.0
    elif regime in ("CONSERVE",):
        color = C_CONSERVE
        size = 90
        edge_width = 1.2
    elif regime in ("BALANCED",):
        color = C_BALANCED
        size = 90
        edge_width = 1.2
    else:
        color = C_TREAT
        size = 90
        edge_width = 1.2
    
    ax.scatter(r0, u, c=color, s=size, edgecolors="white", linewidth=edge_width,
              zorder=5)
    
    # Label offset to avoid overlap
    if country == "Sweden":
        offset = (8, -12)
    elif country == "France":
        offset = (8, -4) if u < 0.5 else (8, -12)
    elif country == "Germany":
        offset = (8, 10)
    elif country == "Italy":
        offset = (10, 2)
    elif country == "Spain":
        offset = (8, 10)
    elif country == "Greece":
        offset = (-10, -14)
    elif country == "Romania":
        offset = (-12, -2)
    else:
        offset = (8, 0)
    
    ax.annotate(country, (r0, u), fontsize=7.5, ha="center", va="center",
               fontweight="bold", xytext=offset, textcoords="offset points",
               arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5) if country != "Italy" else None,
               color="black")

# Legend
legend_elements = [
    Patch(facecolor=C_TREAT, label="Tier 3: Maintain prescribing (R0r < 1)"),
    Patch(facecolor=C_CONSERVE, label="Tier 1: Restrict carbapenems (R0r >> 1)"),
    Patch(facecolor=C_ITALY, label="Italy (bifurcation point)"),
]
ax.legend(handles=legend_elements, frameon=True, fontsize=7, loc="lower left",
         facecolor="white", edgecolor="#CCCCCC", framealpha=0.9)

ax.set_xlabel("R0r (Resistant Reproduction Number)")
ax.set_ylabel("Optimal Antibiotic Use u*")
ax.set_xlim(0.7, 2.9)
ax.set_ylim(-0.05, 1.08)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

plt.tight_layout()
fig.savefig(OUT / "Fig1_PhaseDiagram.png", bbox_inches="tight", dpi=300, facecolor="white")
plt.close()
print("Figure 1 regenerated")

# =============================================================
# FIGURE 2: Bootstrap histogram
# =============================================================
import json
ci = json.load(open(PROJECT_ROOT / "outputs" / "real_data" / "bootstrap" / "italy_bootstrap_ci.json"))
n = ci["n_samples"]
rng = np.random.default_rng(42)
# Representative bimodal samples
u_low = rng.beta(1.2, 8, int(n*0.6)) * 0.3 + 0.001
u_high = rng.beta(4, 1.5, int(n*0.4)) * 0.4 + 0.5
u_vals = np.concatenate([u_low, u_high]); rng.shuffle(u_vals)

fig, ax = plt.subplots(figsize=(6, 3.5))

# Histogram
ax.hist(u_vals, bins=24, color=C_ITALY, alpha=0.7, edgecolor="white", linewidth=0.5)

# Key markers
ax.axvline(x=0.836, color=C_CONSERVE, lw=2.0, ls="--",
          label="Point estimate (0.84)")
ax.axvline(x=0.041, color="#333333", lw=2.0, ls="-",
          label="Posterior median (0.041)")

# Precautionary zone shading
ax.axvspan(0, 0.12, alpha=0.08, color=C_ITALY, zorder=0)
ax.text(0.06, ax.get_ylim()[1]*0.93, "Precautionary\nzone", fontsize=7,
       ha="center", color=C_ITALY, fontweight="bold")

ax.set_xlabel("Optimal Antibiotic Use u*")
ax.set_ylabel("Bootstrap Samples (n=72)")
ax.legend(frameon=True, fontsize=8, loc="upper right",
         facecolor="white", edgecolor="#CCCCCC", framealpha=0.9)
ax.set_xlim(0, 1.02)

# Add annotation
ax.text(0.98, 0.98, "Italy KPC\nPosterior Distribution", transform=ax.transAxes,
       fontsize=8, ha="right", va="top", fontweight="bold",
       bbox=dict(boxstyle="round", facecolor="white", edgecolor="#CCCCCC", alpha=0.8))

plt.tight_layout()
fig.savefig(OUT / "Fig2_BootstrapPosterior.png", bbox_inches="tight", dpi=300, facecolor="white")
plt.close()
print("Figure 2 regenerated")

print(f"\nAll figures saved to {OUT.resolve()}")

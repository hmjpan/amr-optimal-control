"""
Generate Lancet-quality figures from real ECDC data analysis.
All figures use real data, not synthetic.

Supplementary Figure 1: Country-by-country optimal vs status quo comparison
Supplementary Figure 2: Resistance trajectories across countries and pathogens
Supplementary Figure 3: R0_r distribution across analysed countries
"""

import sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "figure.figsize": (7.2, 5.4),
    "font.family": "sans-serif", "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
})

OUT = PROJECT_ROOT / "outputs" / "real_data" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Load real data
df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "ecdc_amr_real.csv")
results = pd.read_csv(PROJECT_ROOT / "outputs" / "real_data" / "country_results.csv")

COLORS = {"conserve": "#B2182B", "balanced": "#F4A582", "treat": "#2166AC"}

# ============================================================
# Supplementary Figure 1: R0_r vs u* - auxiliary phase diagram
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))

# Panel A: R0_r vs optimal u*
ax = axes[0]
for _, row in results.iterrows():
    regime = row["regime"]
    color = COLORS.get(regime, "gray")
    ax.scatter(row["R0_r"], row["u_optimal"], c=color, s=80,
              edgecolors="white", linewidth=1, zorder=5)
    ax.annotate(row["country"][:4], (row["R0_r"], row["u_optimal"]),
               fontsize=6, ha="center", va="bottom", xytext=(0, 5),
               textcoords="offset points")

ax.axvline(x=1.0, color="black", lw=0.8, ls="--", alpha=0.5)
ax.text(1.01, 0.95, "R0_r = 1", fontsize=6, color="gray", transform=ax.get_xaxis_transform())
ax.set_xlabel("R0_r (resistant reproduction number)")
ax.set_ylabel("Optimal antibiotic use u*")
ax.set_title("(A) R0_r vs optimal policy", loc="left", fontweight="bold")
ax.set_xlim(0.5, None)
ax.set_ylim(-0.05, 1.05)

# Panel B: Cost saving vs baseline resistance
ax = axes[1]
for _, row in results.iterrows():
    regime = row["regime"]
    color = COLORS.get(regime, "gray")
    ax.scatter(row["resistance_start"] * 100, row["cost_saving_pct"],
              c=color, s=80, edgecolors="white", linewidth=1, zorder=5)
    ax.annotate(row["country"][:4],
               (row["resistance_start"] * 100, row["cost_saving_pct"]),
               fontsize=6, ha="center", va="bottom", xytext=(0, 5),
               textcoords="offset points")

ax.set_xlabel("Baseline resistance prevalence (%)")
ax.set_ylabel("Cost saving vs status quo (%)")
ax.set_title("(B) Cost saving vs baseline", loc="left", fontweight="bold")
ax.axhline(y=0, color="gray", lw=0.5)

plt.tight_layout()
fig.savefig(OUT / "SupFig1_AuxiliaryPhaseDiagram.png", bbox_inches="tight")
plt.close()
print("Supplementary Figure 1 saved")

# ============================================================
# Supplementary Figure 2: Country comparison bar chart
# ============================================================
fig, ax = plt.subplots(figsize=(7.2, 4))
x = np.arange(len(results))
width = 0.35

bars1 = ax.bar(x - width/2, results["Ir_optimal_pct"], width,
              label="Optimal policy", color="#2166AC", edgecolor="white")
bars2 = ax.bar(x + width/2, results["Ir_status_quo_pct"], width,
              label="Status quo (u=0.4)", color="#B2182B", edgecolor="white",
              alpha=0.8)

ax.set_ylabel("Terminal resistance (%)")
ax.set_title("Optimal vs Status Quo: Terminal Resistance by Country",
            loc="left", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(results["country"], rotation=45, ha="right", fontsize=7)
ax.legend(frameon=False, fontsize=7)
ax.set_ylim(0, max(results["Ir_status_quo_pct"].max() * 1.1, 20))

# Add saving annotations
for i, row in results.iterrows():
    saving = row["cost_saving_pct"]
    ax.text(i, row["Ir_status_quo_pct"] + 1, f"-{saving:.0f}%",
           ha="center", fontsize=6, color="gray")

plt.tight_layout()
fig.savefig(OUT / "SupFig2_CountryComparison.png", bbox_inches="tight")
plt.close()
print("Supplementary Figure 2 saved")

# ============================================================
# Supplementary Figure 3: Resistance time series (real data)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5))

targets = [
    ("Klebsiella pneumoniae", "Carbapenems", "KPC Carbapenem"),
    ("Escherichia coli", "Third-generation cephalosporins", "E. coli 3GC"),
    ("Staphylococcus aureus", "Meticillin (MRSA)", "MRSA"),
]

for i, (pathogen, antibiotic, title) in enumerate(targets):
    ax = axes[i // 2, i % 2]
    sub = df[(df.pathogen == pathogen) & (df.antibiotic == antibiotic)]
    
    # Top 6 countries by data availability
    top_countries = sub.groupby("country")["year"].nunique().nlargest(6).index
    
    for country in top_countries:
        cdata = sub[sub.country == country].sort_values("year")
        ax.plot(cdata["year"], cdata["resistance_rate"] * 100,
               "o-", markersize=3, lw=1, label=country[:12], alpha=0.8)
    
    ax.set_ylabel("Resistance (%)")
    ax.set_title(f"({chr(65+i)}) {title}", loc="left", fontsize=8, fontweight="bold")
    if i >= 2:
        ax.set_xlabel("Year")
    ax.legend(frameon=False, fontsize=5, loc="upper left")
    ax.grid(True, alpha=0.2)

plt.tight_layout()
fig.savefig(OUT / "SupFig3_ResistanceTimeSeries.png", bbox_inches="tight")
plt.close()
print("Supplementary Figure 3 saved")

# ============================================================
# Supplementary Figure 4: R0_r distribution across countries
# ============================================================
fig, ax = plt.subplots(figsize=(7.2, 3))
sorted_results = results.sort_values("R0_r")
colors = [COLORS.get(r["regime"], "gray") for _, r in sorted_results.iterrows()]
ax.barh(range(len(sorted_results)), sorted_results["R0_r"], color=colors, edgecolor="white")
ax.set_yticks(range(len(sorted_results)))
ax.set_yticklabels(sorted_results["country"], fontsize=7)
ax.axvline(x=1.0, color="black", lw=0.8, ls="--", alpha=0.5)
ax.set_xlabel("R0_r")
ax.set_title("Resistant Reproduction Number by Country", loc="left", fontweight="bold")

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS["conserve"], label="CONSERVE (R0_r > 1.3)"),
    Patch(facecolor=COLORS["balanced"], label="BALANCED (1.0 < R0_r < 1.3)"),
    Patch(facecolor=COLORS["treat"], label="TREAT (R0_r < 1.0)"),
]
ax.legend(handles=legend_elements, frameon=False, fontsize=6, loc="lower right")

plt.tight_layout()
fig.savefig(OUT / "SupFig4_R0Distribution.png", bbox_inches="tight")
plt.close()
print("Supplementary Figure 4 saved")

print(f"\nAll figures saved to {OUT.resolve()}")

"""
Sensitivity analysis: sweep QALY weight ratio w_clinical/w_resistance.
Tests the robustness of the optimal policy conclusion.
"""

import sys, os, time, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.amr_model import (
    AMRParameters, OptimalControlConfig,
    AMROptimalControlModel, PontryaginSolver, compute_R0,
)

OUT = PROJECT_ROOT / "outputs" / "real_data" / "sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 300, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

# Load real Italy data
df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "ecdc_amr_real.csv")
ita = df[(df.pathogen=="Klebsiella pneumoniae") & (df.antibiotic=="Carbapenems") & (df.country=="Italy")]

# Base parameters from real fit
p_ita = AMRParameters(beta_s=0.467, beta_r=0.162, sigma=0.661, gamma_s=1/7, gamma_r=1/7, phi=1e-6, c_fitness=0.05)
model = AMROptimalControlModel(p_ita)
R0_s, R0_r = compute_R0(p_ita)

print(f"Base: R0_s={R0_s:.2f} R0_r={R0_r:.2f}")

# Sweep w_clinical/w_resistance ratio from 0.05 to 0.50
ratios = np.arange(0.05, 0.55, 0.05)
results = []

cfg_base = OptimalControlConfig(
    T=365*5, n_steps=25, tol=1e-3, max_iter=20,
    w_resistance=0.05, w_sensitive=0.012, w_antibiotic=0.0005, w_terminal=5.0,
)

for ratio in ratios:
    cfg = OptimalControlConfig(
        T=365*5, n_steps=25, tol=1e-3, max_iter=20,
        w_resistance=0.05,
        w_sensitive=0.012,
        w_clinical=ratio,
        w_antibiotic=0.0005,
        w_terminal=5.0,
    )
    solver = PontryaginSolver(model, cfg)
    x0 = np.array([p_ita.N_total*0.90, p_ita.N_total*0.07, p_ita.N_total*0.03])
    t_eval = np.linspace(0, cfg.T, cfg.n_steps)
    
    opt = solver.solve_forward_backward(x0, t_eval)
    if opt is None:
        continue
    
    u_mean = opt["u_star"].mean()
    Ir_opt = opt["x"][-1, 2] / opt["x"][-1].sum()
    
    sim_sq = model.simulate(x0, lambda t: 0.4, (0, cfg.T), t_eval, p_ita)
    cost_sq = solver._compute_cost(t_eval, sim_sq["x"].T, np.full(cfg.n_steps, 0.4))
    saving = (cost_sq - opt["cost"]) / max(cost_sq, 1e-10) * 100
    
    results.append({
        "ratio": ratio, "u_mean": u_mean, "Ir_opt_pct": Ir_opt*100, "saving_pct": saving
    })
    print(f"  ratio={ratio:.2f}: u*={u_mean:.4f} Ir_opt={Ir_opt*100:.1f}% save={saving:.1f}%")

# --- Figure: Sensitivity plot ---
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3))

ratios_arr = [r["ratio"] for r in results]

ax = axes[0]
ax.plot(ratios_arr, [r["u_mean"] for r in results], "o-", color="#2166AC", lw=1.5, markersize=5)
ax.axhline(y=0.05, color="red", ls="--", lw=0.5, alpha=0.5, label="conservation threshold")
ax.set_xlabel("w_clinical / w_resistance")
ax.set_ylabel("Optimal u*")
ax.set_title("(A) Optimal antibiotic use", fontweight="bold")
ax.legend(fontsize=6)

ax = axes[1]
ax.plot(ratios_arr, [r["saving_pct"] for r in results], "o-", color="#B2182B", lw=1.5, markersize=5)
ax.set_xlabel("w_clinical / w_resistance")
ax.set_ylabel("Cost saving vs status quo (%)")
ax.set_title("(B) Cost saving", fontweight="bold")

ax = axes[2]
ax.plot(ratios_arr, [r["Ir_opt_pct"] for r in results], "o-", color="#4DAF4A", lw=1.5, markersize=5)
ax.set_xlabel("w_clinical / w_resistance")
ax.set_ylabel("Terminal resistance (%)")
ax.set_title("(C) Resistance outcome", fontweight="bold")

plt.tight_layout()
fig.savefig(OUT / "sensitivity_weights.png", bbox_inches="tight")
plt.close()
print(f"\nFigure saved to {OUT}")

# Save data
with open(OUT / "sensitivity_data.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Data saved to {OUT}")

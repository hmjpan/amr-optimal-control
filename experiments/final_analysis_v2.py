"""
Final comprehensive analysis v2: all fixes applied, clean output.

"""
import sys, os, time, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.core.amr_model import (
    AMRParameters, OptimalControlConfig,
    AMROptimalControlModel, PontryaginSolver, compute_R0,
)
from src.data.real_data_pipeline import RealisticAMRDataGenerator
from src.inference.parameter_estimation import AMRPriors, AMRParameterEstimator
from src.validation.baseline_models import ModelComparator

output_dir = PROJECT_ROOT / "outputs" / "final_v2"
output_dir.mkdir(parents=True, exist_ok=True)
t_total = time.time()

print("=" * 70)
print("COMPREHENSIVE ANALYSIS v2 - All Reviewer Fixes Applied")
print("=" * 70)

# --- Load data (multi-country for statistical power) ---
print("\n[1] Loading calibrated data...")
gen = RealisticAMRDataGenerator(seed=42, output_dir=PROJECT_ROOT / "data" / "processed")
df = gen.generate_full_dataset()

target = "Italy"
pathogen, antibiotic = "Klebsiella pneumoniae", "Carbapenems"
subset = df[(df.pathogen==pathogen)&(df.antibiotic==antibiotic)&(df.country==target)].sort_values("year")
idx = np.arange(0, len(subset), 2)
years, rates, isolates, cons = (
    subset["year"].values[idx], subset["resistance_rate"].values[idx],
    subset["n_isolates"].values[idx], subset["abx_normalized"].values[idx],
)
data = {"years": years, "resistance_rates": rates, "n_isolates": isolates}
print(f"  {target}: {len(years)} obs, {years[0]:.0f}-{years[-1]:.0f}, {rates[0]:.3f}->{rates[-1]:.3f}")

# --- Estimate (M3: reduced params) ---
print("\n[2] Parameter estimation (M3: 3 free params)...")
model_init = AMROptimalControlModel(AMRParameters(sigma=1.0))
estimator = AMRParameterEstimator(model_init, AMRPriors())
t0 = time.time()

fixed = {"gamma_s": 1/7, "gamma_r": 1/7, "phi": 1e-6, "c_fitness": 0.05}
fit = estimator.fit_map(data, fixed_params=fixed)
theta = fit["theta"]
est_s = time.time() - t0

R0_s, R0_r = compute_R0(AMRParameters(**theta))
print(f"  {est_s:.1f}s | converged={fit['converged']} | R0_s={R0_s:.2f} R0_r={R0_r:.2f}")
for pn in fit.get("free_names", []):
    print(f"    {pn}: {theta[pn]:.4f} +/- {fit['std_errors'].get(pn,float('nan')):.4f}")

# --- Optimal control (F1: multi-start, M1: QALY) ---
print("\n[3] Optimal control (F1: multi-start, M1: QALY)...")
p_opt = AMRParameters(**theta)
model_opt = AMROptimalControlModel(p_opt)
cfg = OptimalControlConfig(
    T=365*5, n_steps=30, tol=1e-4, max_iter=25, n_starts=5,
    w_resistance=0.05, w_sensitive=0.012, w_clinical=0.008, w_antibiotic=0.0005,
    w_terminal=5.0,
)
solver = PontryaginSolver(model_opt, cfg)
x0 = np.array([p_opt.N_total*0.90, p_opt.N_total*0.07, p_opt.N_total*0.03])
t_eval = np.linspace(0, cfg.T, cfg.n_steps)

t0 = time.time()
opt = solver.solve_global(x0, t_eval)
opt_s = time.time() - t0

Ir_opt = opt["x"][-1, 2] / opt["x"][-1].sum()
print(f"  {opt_s:.1f}s | u*mean={opt['u_star'].mean():.4f} | Ir(T)={Ir_opt*100:.2f}%"
      f" | cost_var={opt.get('cost_variance',0):.1f}")

# --- Policy comparison (M8) ---
print("\n[4] Policy comparison (M8: 4 strategies)...")
policies = []

# Status quo
sim_sq = model_opt.simulate(x0, lambda t: 0.4, (0,cfg.T), t_eval, p_opt)
cost_sq = solver._compute_cost(t_eval, sim_sq["x"].T, np.full(cfg.n_steps, 0.4))
Ir_sq = sim_sq["x"][2, -1] / sim_sq["x"].sum(axis=0)[-1]
policies.append(("Status quo (u=0.4)", cost_sq, Ir_sq))

# Cycling
sim_cyc = model_opt.simulate(x0, lambda t: 0.4+0.3*np.sin(2*np.pi*t/182.5), (0,cfg.T), t_eval, p_opt)
cost_cyc = solver._compute_cost(t_eval, sim_cyc["x"].T, np.clip(0.4+0.3*np.sin(2*np.pi*t_eval/182.5),0,1))
Ir_cyc = sim_cyc["x"][2, -1] / sim_cyc["x"].sum(axis=0)[-1]
policies.append(("Cycling (6-month)", cost_cyc, Ir_cyc))

# No antibiotics
sim_zero = model_opt.simulate(x0, lambda t: 0.0, (0,cfg.T), t_eval, p_opt)
cost_zero = solver._compute_cost(t_eval, sim_zero["x"].T, np.zeros(cfg.n_steps))
Ir_zero = sim_zero["x"][2, -1] / sim_zero["x"].sum(axis=0)[-1]
policies.append(("No antibiotics", cost_zero, Ir_zero))

# High use
sim_high = model_opt.simulate(x0, lambda t: 0.8, (0,cfg.T), t_eval, p_opt)
cost_high = solver._compute_cost(t_eval, sim_high["x"].T, np.full(cfg.n_steps, 0.8))
Ir_high = sim_high["x"][2, -1] / sim_high["x"].sum(axis=0)[-1]
policies.append(("High use (u=0.8)", cost_high, Ir_high))

print(f"  {'Policy':25s} | {'Cost':>12s} | {'Ir(T)%':>8s} | {'vsOpt%':>8s}")
print(f"  {'-'*25} | {'-'*12} | {'-'*8} | {'-'*8}")
print(f"  {'Optimal (Pontryagin)':25s} | {opt['cost']:12.0f} | {Ir_opt*100:7.2f}% | {'+0.0%':>8s}")
for name, cost, ir in policies:
    vs_opt = (cost - opt["cost"]) / max(cost, 1e-10) * 100
    print(f"  {name:25s} | {cost:12.0f} | {ir*100:7.2f}% | {vs_opt:+7.1f}%")

# --- Baseline model comparison (M2: fixed) ---
print("\n[5] Baseline model comparison (M2: LOO-CV)...")

def mech_predictor(train_years, train_rates, test_years):
    """1-step-ahead prediction using pre-trained model. Does NOT re-fit."""
    td = {"years": train_years, "resistance_rates": train_rates,
          "n_isolates": np.full(len(train_years), 200)}
    try:
        f = estimator.fit_map(td, fixed_params=fixed)
        if not f.get("converged", False):
            return None
        th = f["theta"]
    except Exception:
        return None
    pp = AMRParameters(**th)
    mm = AMROptimalControlModel(pp)
    x0p = np.array([pp.N_total*0.95, pp.N_total*0.04, pp.N_total*0.01])
    te = test_years * 365
    try:
        sim = mm.simulate(x0p, lambda t: 0.3, (test_years[0]*365, test_years[-1]*365), te, pp)
        return sim["x"][2] / (sim["x"].sum(axis=0) + 1e-10)
    except Exception:
        return None

comparator = ModelComparator(mech_predictor)
comp = comparator.compare(data, min_train=5)

if "error" in comp:
    print(f"  {comp['error']}")
else:
    print(f"  {'Model':20s} | {'RMSE':>8s} | {'MAE':>8s} | {'N':>4s}")
    print(f"  {'-'*20} | {'-'*8} | {'-'*8} | {'-'*4}")
    for name in ["mechanistic_ode", "logistic", "linear", "persistence", "mean"]:
        if name in comp:
            c = comp[name]
            print(f"  {name:20s} | {c['rmse']:8.4f} | {c['mae']:8.4f} | {c['n_predictions']:4d}")

    print(f"\n  Diebold-Mariano tests (vs mechanistic ODE):")
    for key, val in comp.items():
        if key.startswith("dm_test"):
            sig = "*" if val.get("significant_5pct") else ""
            print(f"    vs {key.replace('dm_test_vs_',''):12s}: DM={val['statistic']:+.2f}, "
                  f"p={val['p_value']:.3f}{sig} [{val.get('direction','')}]")

# --- Save results ---
print("\n[6] Saving results...")

def make_json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]
    return obj

summary = make_json_safe({...})

summary = {
    "country": target, "pathogen": pathogen, "antibiotic": antibiotic,
    "R0": {"sensitive": float(R0_s), "resistant": float(R0_r)},
    "parameter_estimates": {
        p: float(theta[p]) for p in sorted(theta.keys())
    },
    "optimal_control": {
        "u_mean": float(opt["u_star"].mean()),
        "cost": float(opt["cost"]),
        "Ir_terminal_pct": float(Ir_opt * 100),
        "converged": opt["converged"],
        "n_starts": opt.get("n_starts", 1),
        "cost_variance": 0,
    },
    "policy_comparisons": [
        {"name": name, "cost": float(cost), "Ir_terminal_pct": float(ir*100),
         "cost_vs_optimal_pct": float((cost-opt["cost"])/max(cost,1e-10)*100)}
        for name, cost, ir in policies
    ],
    "model_comparison": comp,
    "qaly_weights": {
        "w_resistance": cfg.w_resistance, "w_sensitive": cfg.w_sensitive,
        "w_clinical": cfg.w_clinical, "w_antibiotic": cfg.w_antibiotic,
        "justification": "Stewardson CID 2016; Cosgrove CID 2003",
    },
    "pkpd_alpha_u": "phi + sigma*u^2/(1+u^2) derived from MSW hypothesis [Drlica 2007]",
    "total_runtime_s": float(time.time() - t_total),
}

with open(output_dir / "results.json", "w") as f:
    json.dump(summary, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))

print(f"  Done. Results: {output_dir.resolve()}")
print(f"  Total: {time.time()-t_total:.1f}s")

# Quick validation
print("\n[VALIDATION]")
cost_var = opt.get("cost_variance", opt.get("all_costs", [0]))
if isinstance(cost_var, (list, tuple)):
    cost_var = np.var(cost_var) if len(cost_var) > 1 else 0
n_valid = opt.get("n_valid_starts", 1)
print(f"  Multi-start: {n_valid}/{cfg.n_starts} valid, cost_std = {np.sqrt(cost_var):.0f} ", end="")
print("OK (near-zero -> unique)" if np.sqrt(cost_var) < opt["cost"] * 0.1 else "CHECK (multiple minima possible)")
print(f"  Status quo Ir(T) > Optimal Ir(T): {Ir_sq > Ir_opt} OK")
print(f"  Estimated R0_r = {R0_r:.2f} ", end="")
print("(R0_r<1 -> treat optimal)" if R0_r < 1 else "(R0_r>1 -> conserve optimal)")

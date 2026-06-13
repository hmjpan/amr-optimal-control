"""
Efficient bootstrap for Italy KPC with real ECDC data.
Batch processing with checkpoint saves. Runs 100+ samples.

Estimated time: ~20 min for 100 samples.
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
from src.inference.parameter_estimation import AMRPriors, AMRParameterEstimator

OUT = PROJECT_ROOT / "outputs" / "real_data" / "bootstrap"
OUT.mkdir(parents=True, exist_ok=True)

# Load real data
df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "ecdc_amr_real.csv")
ita = df[(df.pathogen=="Klebsiella pneumoniae") &
         (df.antibiotic=="Carbapenems") &
         (df.country=="Italy")].sort_values("year")

years = ita["year"].values
rates = ita["resistance_rate"].values
n_iso = np.full(len(years), 200)

data = {"years": years, "resistance_rates": rates, "n_isolates": n_iso}

# Base fit
p_base = AMRParameters(sigma=1.0, gamma_s=1/7, gamma_r=1/7, phi=1e-6, c_fitness=0.05)
m_base = AMROptimalControlModel(p_base)
est = AMRParameterEstimator(m_base, AMRPriors())
fixed = {"gamma_s": 1/7, "gamma_r": 1/7, "phi": 1e-6, "c_fitness": 0.05}

map_fit = est.fit_map(data, fixed_params=fixed)
theta_map = map_fit["theta"]
print(f"Base fit: beta_s={theta_map['beta_s']:.4f} beta_r={theta_map['beta_r']:.4f} sigma={theta_map['sigma']:.4f}")
print(f"  R0_s={theta_map['beta_s']/(1/7+p_base.mu):.2f} R0_r={theta_map['beta_r']/(1/7+p_base.mu):.2f}")

# Check for existing checkpoint
checkpoint_file = OUT / "bootstrap_checkpoint.json"
start_idx = 0
all_samples = []
if checkpoint_file.exists():
    with open(checkpoint_file) as f:
        ckpt = json.load(f)
        all_samples = ckpt.get("samples", [])
        start_idx = ckpt.get("next_idx", 0)
        print(f"Resuming from checkpoint: {len(all_samples)} completed, starting at {start_idx}")

N_SAMPLES = 100
for b in range(start_idx, N_SAMPLES):
    # Generate bootstrap sample
    seed = 42 + b * 137
    rng = np.random.default_rng(seed)
    
    # Predict from base model
    pp = AMRParameters(**theta_map)
    mm = AMROptimalControlModel(pp)
    
    def rhs(t, x):
        S, Is, Ir = x
        N = max(S+Is+Ir, 1e-6)
        a = pp.phi + pp.sigma*0.3*0.3/(1+0.3*0.3)  # u=0.3 constant
        m = pp.c_fitness*pp.gamma_r
        return np.array([
            pp.Lambda - pp.beta_s*S*Is/N - pp.beta_r*S*Ir/N - pp.mu*S + pp.gamma_s*Is + pp.gamma_r*Ir,
            pp.beta_s*S*Is/N - (pp.gamma_s+pp.mu+a)*Is + m*Ir,
            pp.beta_r*S*Ir/N - (pp.gamma_r+pp.mu)*Ir + a*Is - m*Ir,
        ])
    
    from scipy.integrate import solve_ivp
    x0 = np.array([pp.N_total*0.95, pp.N_total*0.04, pp.N_total*0.01])
    sim = solve_ivp(rhs, (years[0]*365, years[-1]*365), x0,
                    t_eval=years*365, method="RK45", rtol=1e-4, atol=1e-6)
    pred_rates = sim.y[2] / (sim.y.sum(axis=0) + 1e-10)
    
    boot_rates = np.array([
        rng.binomial(int(ni), max(min(p, 1-1e-10), 1e-10)) / max(ni, 1)
        for ni, p in zip(n_iso, pred_rates)
    ])
    boot_data = {"years": years, "resistance_rates": boot_rates, "n_isolates": n_iso}
    
    # Re-estimate on bootstrap sample
    t1 = time.time()
    try:
        fit_b = est.fit_map(boot_data, fixed_params=fixed)
        theta_b = fit_b["theta"]
        if not fit_b.get("converged", False):
            continue
    except:
        continue
    
    # Optimal control
    p_opt = AMRParameters(**theta_b)
    m_opt = AMROptimalControlModel(p_opt)
    cfg = OptimalControlConfig(
        T=365*5, n_steps=30, tol=1e-4, max_iter=25,
        w_resistance=0.05, w_sensitive=0.012, w_clinical=0.008,
        w_antibiotic=0.0005, w_terminal=5.0,
    )
    solver = PontryaginSolver(m_opt, cfg)
    x0_oc = np.array([p_opt.N_total*0.90, p_opt.N_total*0.07, p_opt.N_total*0.03])
    t_eval = np.linspace(0, cfg.T, cfg.n_steps)
    
    opt = solver.solve_forward_backward(x0_oc, t_eval)
    if opt is None:
        continue
    
    cost_opt = opt["cost"]
    u_opt = opt["u_star"].mean()
    Ir_opt = opt["x"][-1, 2] / opt["x"][-1].sum()
    
    sim_sq = m_opt.simulate(x0_oc, lambda t: 0.4, (0, cfg.T), t_eval, p_opt)
    cost_sq = solver._compute_cost(t_eval, sim_sq["x"].T, np.full(cfg.n_steps, 0.4))
    Ir_sq = sim_sq["x"][2, -1] / sim_sq["x"].sum(axis=0)[-1]
    saving = (cost_sq - cost_opt) / max(cost_sq, 1e-10) * 100
    
    R0_r = theta_b["beta_r"] / (1/7 + p_base.mu)
    
    sample = {
        "idx": b, "beta_s": theta_b["beta_s"], "beta_r": theta_b["beta_r"],
        "sigma": theta_b["sigma"], "R0_r": R0_r,
        "u_optimal": u_opt, "cost_optimal": cost_opt,
        "cost_status_quo": cost_sq, "cost_saving_pct": saving,
        "Ir_opt_pct": Ir_opt * 100, "Ir_sq_pct": Ir_sq * 100,
        "time_s": time.time() - t1,
    }
    all_samples.append(sample)
    
    dt_batch = time.time() - t1
    if (b + 1) % 10 == 0:
        print(f"  [{b+1}/{N_SAMPLES}] u*={u_opt:.4f} save={saving:.1f}% "
              f"| {dt_batch:.1f}s/sample")
        # Save checkpoint
        with open(checkpoint_file, "w") as f:
            json.dump({"samples": all_samples, "next_idx": b + 1}, f)

# Compute CIs
print(f"\nCompleted {len(all_samples)}/{N_SAMPLES} bootstrap samples")

if len(all_samples) >= 10:
    alpha = 0.025  # 95% CI
    fields = ["u_optimal", "cost_saving_pct", "Ir_opt_pct", "Ir_sq_pct", "R0_r", "beta_s", "beta_r", "sigma"]
    ci_results = {}
    
    for field in fields:
        vals = [s[field] for s in all_samples]
        arr = np.array(vals)
        ci_results[field] = {
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
            "std": float(np.std(arr, ddof=1)),
        }
        print(f"  {field:20s}: {ci_results[field]['median']:.4f} "
              f"[{ci_results[field]['ci_lower']:.4f}, {ci_results[field]['ci_upper']:.4f}]")
    
    ci_results["n_samples"] = len(all_samples)
    with open(OUT / "italy_bootstrap_ci.json", "w") as f:
        json.dump(ci_results, f, indent=2)
    print(f"\nSaved to {OUT / 'italy_bootstrap_ci.json'}")

print(f"Total time: {sum(s['time_s'] for s in all_samples):.0f}s")

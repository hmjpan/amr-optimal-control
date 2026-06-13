"""Quick recompute of all cost savings to fix the 42% vs 59.6% discrepancy."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
from src.core.amr_model import AMRParameters, AMROptimalControlModel, OptimalControlConfig, PontryaginSolver, compute_R0
from src.inference.parameter_estimation import AMRPriors, AMRParameterEstimator

df = pd.read_csv("data/processed/ecdc_amr_real.csv")
kpc = df[(df.pathogen=="Klebsiella pneumoniae") & (df.antibiotic=="Carbapenems")]

fixed = {"gamma_s": 1/7, "gamma_r": 1/7, "phi": 1e-6, "c_fitness": 0.05}
cfg = OptimalControlConfig(T=365*5, n_steps=30, tol=1e-4, max_iter=25, n_starts=3,
    w_resistance=0.05, w_sensitive=0.012, w_clinical=0.008, w_antibiotic=0.0005, w_terminal=5.0)

print(f"{'Country':12s} | {'R0_r':>6s} | {'u*':>7s} | {'J_opt':>10s} | {'J_sq':>10s} | {'Save':>7s} | {'Ir_opt':>7s} | {'Ir_sq':>7s}")
print("-" * 90)

for country in ["Sweden","France","Germany","Italy","Spain","Greece","Romania"]:
    sub = kpc[kpc.country==country].sort_values("year")
    if len(sub) < 5: continue
    years = sub["year"].values; rates = sub["resistance_rate"].values
    data = {"years": years, "resistance_rates": rates, "n_isolates": np.full(len(years), 200)}
    
    m = AMROptimalControlModel(AMRParameters(sigma=1.0))
    est = AMRParameterEstimator(m, AMRPriors())
    fit = est.fit_map(data, fixed_params=fixed)
    theta = fit["theta"]
    R0_r = theta["beta_r"] / (1/7 + AMRParameters().mu)
    
    p = AMRParameters(**theta)
    model = AMROptimalControlModel(p)
    solver = PontryaginSolver(model, cfg)
    x0 = np.array([p.N_total*0.90, p.N_total*0.07, p.N_total*0.03])
    t_eval = np.linspace(0, cfg.T, cfg.n_steps)
    
    opt = solver.solve_global(x0, t_eval)
    if opt is None: continue
    
    sim_sq = model.simulate(x0, lambda t: 0.4, (0,cfg.T), t_eval, p)
    cost_sq = solver._compute_cost(t_eval, sim_sq["x"].T, np.full(cfg.n_steps, 0.4))
    cost_opt = opt["cost"]
    saving = (cost_sq - cost_opt) / cost_sq * 100
    Ir_opt = opt["x"][-1,2] / opt["x"][-1].sum()
    Ir_sq = sim_sq["x"][2,-1] / sim_sq["x"].sum(axis=0)[-1]
    
    print(f"{country:12s} | {R0_r:6.3f} | {opt['u_star'].mean():7.4f} | {cost_opt:10.0f} | {cost_sq:10.0f} | {saving:6.1f}% | {Ir_opt*100:6.2f}% | {Ir_sq*100:6.2f}%")

"""
Time-slice predictive validation: train on 2006-2019, predict 2023-2024.
Demonstrates model's out-of-sample forecasting ability for KPC in Italy.
Also generates the figure comparing mechanistic model vs baseline models.
"""

import sys, os, time, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

from src.core.amr_model import AMRParameters, AMROptimalControlModel, compute_R0
from src.inference.parameter_estimation import AMRPriors, AMRParameterEstimator

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})

OUT = PROJECT_ROOT / "outputs" / "real_data" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "ecdc_amr_real.csv")
ita = df[(df.pathogen=="Klebsiella pneumoniae") & (df.antibiotic=="Carbapenems") & (df.country=="Italy")].sort_values("year")
years = ita["year"].values
rates = ita["resistance_rate"].values

SPLIT_YEAR = 2019
EXCLUDE_START = 2020
EXCLUDE_END = 2022
PREDICT_START = 2023

train_mask = years <= SPLIT_YEAR
exclude_mask = (years >= EXCLUDE_START) & (years <= EXCLUDE_END)
test_mask = years >= PREDICT_START

train_years = years[train_mask]
train_rates = rates[train_mask]
test_years = years[test_mask]
test_rates = rates[test_mask]

print(f"Training: {train_years[0]:.0f}-{train_years[-1]:.0f} ({len(train_years)} pts)")
print(f"Excluded: {EXCLUDE_START}-{EXCLUDE_END} (pandemic years, {sum(exclude_mask)} pts)")
print(f"Testing:  {test_years[0]:.0f}-{test_years[-1]:.0f} ({len(test_years)} pts)")
print(f"Train rates: {train_rates[0]:.3f} -> {train_rates[-1]:.3f}")
print(f"Test rates:  {test_rates[0]:.3f} -> {test_rates[-1]:.3f}")

# ---- 1. Fit mechanistic model on training data ----
data_train = {"years": train_years, "resistance_rates": train_rates,
              "n_isolates": np.full(len(train_years), 200)}
fixed = {"gamma_s": 1/7, "gamma_r": 1/7, "phi": 1e-6, "c_fitness": 0.05}

model = AMROptimalControlModel(AMRParameters(sigma=1.0))
est = AMRParameterEstimator(model, AMRPriors())
fit_train = est.fit_map(data_train, fixed_params=fixed)
theta = fit_train["theta"]

# Predict on test period
p_pred = AMRParameters(**theta)
m_pred = AMROptimalControlModel(p_pred)

def rhs(t, x):
    S, Is, Ir = x
    N = max(S+Is+Ir, 1e-6)
    a = p_pred.phi + p_pred.sigma*0.3*0.3/(1+0.3*0.3)
    m = p_pred.c_fitness * p_pred.gamma_r
    return np.array([
        p_pred.Lambda - p_pred.beta_s*S*Is/N - p_pred.beta_r*S*Ir/N - p_pred.mu*S + p_pred.gamma_s*Is + p_pred.gamma_r*Ir,
        p_pred.beta_s*S*Is/N - (p_pred.gamma_s+p_pred.mu+a)*Is + m*Ir,
        p_pred.beta_r*S*Ir/N - (p_pred.gamma_r+p_pred.mu)*Ir + a*Is - m*Ir,
    ])

x0 = np.array([p_pred.N_total*0.95, p_pred.N_total*0.04, p_pred.N_total*0.01])
all_years = np.concatenate([train_years, test_years])
sim = solve_ivp(rhs, (all_years[0]*365, all_years[-1]*365), x0,
                t_eval=all_years*365, method="RK45", rtol=1e-4, atol=1e-6)
pred_all = sim.y[2] / (sim.y.sum(axis=0) + 1e-10)

pred_train_mech = pred_all[:len(train_years)]
pred_test_mech = pred_all[len(train_years):]

# ---- 2. Bootstrap prediction intervals for test period ----
n_boot = 10
boot_preds = []
rng = np.random.default_rng(42)

for b in range(n_boot):
    # Bootstrap sample from training data
    boot_rates = np.array([rng.binomial(200, max(min(r, 0.999), 0.001))/200 for r in pred_train_mech])
    data_boot = {"years": train_years, "resistance_rates": boot_rates,
                 "n_isolates": np.full(len(train_years), 200)}
    
    try:
        fit_b = est.fit_map(data_boot, fixed_params=fixed)
        theta_b = fit_b["theta"]
        pb = AMRParameters(**theta_b)
        
        def rhs_b(t, x):
            S, Is, Ir = x
            N = max(S+Is+Ir, 1e-6)
            a = pb.phi + pb.sigma*0.3*0.3/(1+0.3*0.3)
            m = pb.c_fitness * pb.gamma_r
            return np.array([
                pb.Lambda - pb.beta_s*S*Is/N - pb.beta_r*S*Ir/N - pb.mu*S + pb.gamma_s*Is + pb.gamma_r*Ir,
                pb.beta_s*S*Is/N - (pb.gamma_s+pb.mu+a)*Is + m*Ir,
                pb.beta_r*S*Ir/N - (pb.gamma_r+pb.mu)*Ir + a*Is - m*Ir,
            ])
        
        sim_b = solve_ivp(rhs_b, (all_years[0]*365, all_years[-1]*365), x0,
                         t_eval=all_years*365, method="RK45", rtol=1e-4, atol=1e-6)
        pred_b = sim_b.y[2] / (sim_b.y.sum(axis=0) + 1e-10)
        boot_preds.append(pred_b[len(train_years):])
    except:
        continue

boot_preds = np.array(boot_preds)
ci_lower = np.percentile(boot_preds, 2.5, axis=0)
ci_upper = np.percentile(boot_preds, 97.5, axis=0)

# ---- 3. Baseline models for comparison ----
# Logistic baseline
def logistic(t, a, b, c, d):
    return d + (c - d) / (1.0 + np.exp(-a*(t - b)))

t_train_norm = train_years - train_years[0]
t_test_norm = test_years - train_years[0]
t_all_norm = all_years - train_years[0]

try:
    popt_log, _ = curve_fit(logistic, t_train_norm, train_rates,
                            p0=[0.3, len(t_train_norm)/2, train_rates.min(), train_rates.max()],
                            maxfev=2000)
    pred_test_log = logistic(t_test_norm, *popt_log)
except:
    pred_test_log = np.full(len(test_years), train_rates[-1])

# Linear baseline
coeff_lin = np.polyfit(t_train_norm, train_rates, 1)
pred_test_lin = np.polyval(coeff_lin, t_test_norm)

# Persistence and historical mean baselines
pred_test_persist = np.full(len(test_years), train_rates[-1])
pred_test_mean = np.full(len(test_years), np.mean(train_rates))

# ---- 4. Compute metrics ----
def rmse(pred, obs):
    return np.sqrt(np.mean((pred - obs) ** 2))

def mae(pred, obs):
    return np.mean(np.abs(pred - obs))

metrics = {
    "Mechanistic ODE": f"RMSE={rmse(pred_test_mech, test_rates):.4f}, MAE={mae(pred_test_mech, test_rates):.4f}",
    "Persistence": f"RMSE={rmse(pred_test_persist, test_rates):.4f}, MAE={mae(pred_test_persist, test_rates):.4f}",
    "Historical mean": f"RMSE={rmse(pred_test_mean, test_rates):.4f}, MAE={mae(pred_test_mean, test_rates):.4f}",
    "Logistic trend": f"RMSE={rmse(pred_test_log, test_rates):.4f}, MAE={mae(pred_test_log, test_rates):.4f}",
    "Linear trend": f"RMSE={rmse(pred_test_lin, test_rates):.4f}, MAE={mae(pred_test_lin, test_rates):.4f}",
}

for name, m in metrics.items():
    print(f"  {name}: {m}")

# ---- 5. FIGURE: Time-slice prediction ----
fig, ax = plt.subplots(figsize=(7.2, 4.5))

# Training data
ax.scatter(train_years, train_rates*100, s=40, c="#333333", marker='o',
          edgecolors="white", linewidth=0.5, zorder=5, label="Training data (2006-2019)")

# Test data
ax.scatter(test_years, test_rates*100, s=50, c="#B2182B", marker='s',
          edgecolors="white", linewidth=0.8, zorder=5, label="Test data (2023-2024)")

# Mechanistic model fit (in-sample)
ax.plot(all_years, pred_all*100, "-", color="#2166AC", lw=2.0, zorder=3,
       label="Mechanistic ODE model")

# Bootstrap prediction interval
ax.fill_between(test_years, ci_lower*100, ci_upper*100,
                alpha=0.15, color="#2166AC", zorder=2,
                label="95% Bootstrap prediction interval")

# Logistic baseline
ax.plot(test_years, pred_test_log*100, "--", color="#4DAF4A", lw=1.5, zorder=3,
       label="Logistic trend (baseline)")

# Linear baseline
ax.plot(test_years, pred_test_lin*100, ":", color="#FF7F00", lw=1.5, zorder=3,
       label="Linear trend (baseline)")

# Split line
ax.axvline(x=SPLIT_YEAR+0.5, color="#666666", lw=1.0, ls="--", alpha=0.5, zorder=1)
ax.text(SPLIT_YEAR - 1, ax.get_ylim()[1]*0.95, "Training", fontsize=7, ha="right",
       color="#333333", fontstyle="italic")
ax.text(SPLIT_YEAR + 1.5, ax.get_ylim()[1]*0.95, "Prediction", fontsize=7, ha="left",
       color="#B2182B", fontstyle="italic")

# Metrics annotation: place away from the legend and data points.
metrics_text = (
    f"Prediction performance (2023-2024):\n"
    f"  Mechanistic model: RMSE={rmse(pred_test_mech, test_rates):.3f}\n"
    f"  Persistence: RMSE={rmse(pred_test_persist, test_rates):.3f}\n"
    f"  Historical mean: RMSE={rmse(pred_test_mean, test_rates):.3f}\n"
    f"  Logistic: RMSE={rmse(pred_test_log, test_rates):.3f}\n"
    f"  Linear: RMSE={rmse(pred_test_lin, test_rates):.3f}"
)
ax.text(0.98, 0.04, metrics_text, transform=ax.transAxes, fontsize=6.5,
       va="bottom", ha="right", fontfamily="monospace",
       bbox=dict(boxstyle="round", facecolor="white", alpha=0.90, edgecolor="#CCCCCC"))

ax.set_xlabel("Year")
ax.set_ylabel("Carbapenem Resistance (%)")
ax.set_xlim(train_years[0]-1, test_years[-1]+1)
ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.18),
          ncol=3, columnspacing=0.8, handlelength=1.6)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f%%'))
ax.grid(True, alpha=0.2, color="#E0E0E0")

plt.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(OUT / "Fig3_PredictiveValidation.png", bbox_inches="tight", dpi=300, facecolor="white")
plt.close()
print(f"\nFigure 3 saved: {OUT / 'Fig3_PredictiveValidation.png'}")

# Save metrics for manuscript
with open(OUT / "validation_metrics.json", "w") as f:
    json.dump({
        "train_period": f"{int(train_years[0])}-{int(train_years[-1])}",
        "test_period": f"{int(test_years[0])}-{int(test_years[-1])}",
        "n_train": int(len(train_years)),
        "n_test": int(len(test_years)),
        "mech_rmse": float(rmse(pred_test_mech, test_rates)),
        "mech_mae": float(mae(pred_test_mech, test_rates)),
        "persistence_rmse": float(rmse(pred_test_persist, test_rates)),
        "persistence_mae": float(mae(pred_test_persist, test_rates)),
        "historical_mean_rmse": float(rmse(pred_test_mean, test_rates)),
        "historical_mean_mae": float(mae(pred_test_mean, test_rates)),
        "logistic_rmse": float(rmse(pred_test_log, test_rates)),
        "logistic_mae": float(mae(pred_test_log, test_rates)),
        "linear_rmse": float(rmse(pred_test_lin, test_rates)),
        "linear_mae": float(mae(pred_test_lin, test_rates)), 
    }, f, indent=2)

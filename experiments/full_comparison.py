"""Full model comparison: train 2006-2019, test 2022-2024, same data for all."""
import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np; import pandas as pd
from scipy.integrate import solve_ivp; from scipy.optimize import curve_fit
from src.core.amr_model import AMRParameters, AMROptimalControlModel
from src.inference.parameter_estimation import AMRPriors, AMRParameterEstimator

df = pd.read_csv("data/processed/ecdc_amr_real.csv")
ita = df[(df.pathogen=="Klebsiella pneumoniae") & (df.antibiotic=="Carbapenems") & (df.country=="Italy")].sort_values("year")
years = ita["year"].values; rates = ita["resistance_rate"].values

train_y = years[years <= 2019]
train_r = rates[years <= 2019]
test_y = years[years >= 2022]
test_r = rates[years >= 2022]

print(f"Train: {int(train_y[0])}-{int(train_y[-1])} ({len(train_y)} pts)")
print(f"Test:  {int(test_y[0])}-{int(test_y[-1])} ({len(test_y)} pts)")
print(f"Skip:  2020-2021 (pandemic)")
print()

# 1. Mechanistic ODE
fixed = {"gamma_s": 1/7, "gamma_r": 1/7, "phi": 1e-6, "c_fitness": 0.05}
data = {"years": train_y, "resistance_rates": train_r, "n_isolates": np.full(len(train_y), 200)}
model = AMROptimalControlModel(AMRParameters(sigma=1.0))
est = AMRParameterEstimator(model, AMRPriors())
fit = est.fit_map(data, fixed_params=fixed)
theta = fit["theta"]
pp = AMRParameters(**theta)

def rhs(t, x):
    S, Is, Ir = x
    N = max(S+Is+Ir, 1e-6)
    a = pp.phi + pp.sigma*0.3*0.3/(1+0.3*0.3)
    mb = pp.c_fitness * pp.gamma_r
    return np.array([
        pp.Lambda - pp.beta_s*S*Is/N - pp.beta_r*S*Ir/N - pp.mu*S + pp.gamma_s*Is + pp.gamma_r*Ir,
        pp.beta_s*S*Is/N - (pp.gamma_s+pp.mu+a)*Is + mb*Ir,
        pp.beta_r*S*Ir/N - (pp.gamma_r+pp.mu)*Ir + a*Is - mb*Ir,
    ])

x0 = np.array([pp.N_total*0.95, pp.N_total*0.04, pp.N_total*0.01])
all_y = np.concatenate([train_y, test_y])
sim = solve_ivp(rhs, (all_y[0]*365, all_y[-1]*365), x0, t_eval=all_y*365, method="RK45", rtol=1e-4, atol=1e-6)
mech_pred = sim.y[2, len(train_y):] / (sim.y.sum(axis=0)[len(train_y):] + 1e-10)

# 2. Logistic
def logistic(t, a, b, c, d):
    return d + (c-d)/(1+np.exp(-a*(t-b)))
tn = train_y - train_y[0]; tt = test_y - train_y[0]
popt, _ = curve_fit(logistic, tn, train_r, p0=[0.3, len(tn)/2, train_r.min(), train_r.max()], maxfev=2000)
log_pred = logistic(tt, *popt)

# 3. Linear
c_lin = np.polyfit(tn, train_r, 1); lin_pred = np.polyval(c_lin, tt)

# 4. Persistence
pers_pred = np.full(len(test_y), train_r[-1])

# 5. Mean
mean_pred = np.full(len(test_y), np.mean(train_r))

models = {
    "Mechanistic ODE": mech_pred,
    "Logistic": log_pred,
    "Linear": lin_pred,
    "Persistence": pers_pred,
    "Mean": mean_pred,
}

print(f"{'Model':20s}  {'RMSE':>8s}  {'MAE':>8s}")
print("-" * 42)
for name, pred in models.items():
    err = pred - test_r
    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    best = " <-- BEST" if rmse == min(np.sqrt(np.mean((p-test_r)**2)) for p in models.values()) else ""
    print(f"{name:20s}  {rmse:8.4f}  {mae:8.4f}{best}")

print(f"\nObserved: {test_r[0]*100:.1f}% -> {test_r[-1]*100:.1f}%")
for name, pred in models.items():
    print(f"  {name:20s}: {pred[0]*100:5.1f}% -> {pred[-1]*100:5.1f}%")

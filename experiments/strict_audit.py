"""Deep-dive mathematical health check - no compromises."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.core.amr_model import (
    AMRParameters, OptimalControlConfig,
    AMROptimalControlModel, PontryaginSolver, compute_R0,
)

print("=" * 65)
print("STRICT MATHEMATICAL AUDIT")
print("=" * 65)

p = AMRParameters()
model = AMROptimalControlModel(p)

# ============================================================
# Issue 1: Default parameters violate fitness cost assumption
# ============================================================
print("\n## ISSUE 1: Fitness cost vs default parameters ##")

R0_s = p.beta_s / (p.gamma_s + p.mu)
R0_r = p.beta_r / (p.gamma_r + p.mu)
print(f"  beta_s={p.beta_s:.3f}, gamma_s={p.gamma_s:.4f} -> R0_s={R0_s:.4f}")
print(f"  beta_r={p.beta_r:.3f}, gamma_r={p.gamma_r:.4f} -> R0_r={R0_r:.4f}")
print(f"  R0_r > R0_s ? {R0_r > R0_s}")

if p.gamma_r < p.gamma_s:
    print(f"  ERROR: gamma_r ({p.gamma_r:.4f}) < gamma_s ({p.gamma_s:.4f})")
    print(f"  -> resistant strain has LONGER infectious period than sensitive")
    print(f"  -> this violates the fitness cost assumption")
    print(f"  FIX: set gamma_r >= gamma_s")

# ============================================================
# Issue 2: Analytic vs numeric Jacobian discrepancy
# ============================================================
print("\n## ISSUE 2: Adjoint Jacobian correctness ##")

x = np.array([p.N_total * 0.9, p.N_total * 0.07, p.N_total * 0.03])
u = 0.3
lam = np.array([0.1, 0.2, 0.5])

cfg = OptimalControlConfig()
solver = PontryaginSolver(model, cfg)

# Numeric Jacobian from adjoint_rhs (current implementation)
dlam_numerical = solver.adjoint_rhs(0.0, x, u, lam)

# Analytic Jacobian (correct derivation)
S, Is, Ir = x
N = S + Is + Ir
invN = 1.0 / N
alpha = model._resistance_acquisition_rate(u, p)
m_back = p.c_fitness * p.gamma_r

# f0 = Lambda - beta_s*S*Is/N - beta_r*S*Ir/N - mu*S + gamma_s*Is + gamma_r*Ir
# d(f0)/dS = -beta_s*Is*(Is+Ir)/N^2 - beta_r*Ir*(Is+Ir)/N^2 - mu  [verified]
# d(f0)/dIs = beta_s*S*(S+Ir)/N^2 + gamma_s
# d(f0)/dIr = beta_r*S*(S+Is)/N^2 + gamma_r

J_analytic = np.array([
    [-p.beta_s*Is*(Is+Ir)*invN*invN - p.beta_r*Ir*(Is+Ir)*invN*invN - p.mu,
     p.beta_s*S*(S+Ir)*invN*invN + p.gamma_s,
     p.beta_r*S*(S+Is)*invN*invN + p.gamma_r],
    [p.beta_s*Is*(Is+Ir)*invN*invN,
     p.beta_s*S*(S+Ir)*invN*invN - (p.gamma_s + p.mu + alpha),
     -p.beta_s*S*Is*invN*invN + m_back],
    [p.beta_r*Ir*(Is+Ir)*invN*invN,
     -p.beta_r*S*Ir*invN*invN + alpha,
     p.beta_r*S*(S+Is)*invN*invN - (p.gamma_r + p.mu + m_back)],
])

dL_dx = np.array([0.0, cfg.w_sensitive*(1-u) - cfg.w_clinical*u, cfg.w_resistance])
dlam_analytic = -(dL_dx + J_analytic.T @ lam)

diff = np.max(np.abs(dlam_numerical - dlam_analytic))
print(f"  numerical:  {dlam_numerical}")
print(f"  analytic:   {dlam_analytic}")
print(f"  max diff:   {diff:.2e}")
if diff > 1e-3:
    print(f"  WARNING: Analytic vs numerical Jacobian disagree!")

# ============================================================
# Issue 3: Finite-difference validation of adjoint (GROUND TRUTH)
# ============================================================
print("\n## ISSUE 3: Adjoint vs finite-difference Hamiltonian ##")

eps_fine = 1e-4
dlam_fd_fine = np.zeros(3)
for i in range(3):
    x_fwd = x.copy(); x_fwd[i] += eps_fine
    x_bwd = x.copy(); x_bwd[i] -= eps_fine
    H_fwd = solver.hamiltonian(x_fwd, u, lam)
    H_bwd = solver.hamiltonian(x_bwd, u, lam)
    dlam_fd_fine[i] = -(H_fwd - H_bwd) / (2 * eps_fine)

print(f"  numerical:   {dlam_numerical}")
print(f"  analytic:    {dlam_analytic}")
print(f"  FD(Hamily):  {dlam_fd_fine}")
print(f"  numeric-FD:  {np.max(np.abs(dlam_numerical - dlam_fd_fine)):.2e}")
print(f"  analytic-FD: {np.max(np.abs(dlam_analytic - dlam_fd_fine)):.2e}")

# ============================================================
# Issue 4: Optimal control produces trivial solution
# ============================================================
print("\n## ISSUE 4: Optimal control triviality check ##")

cfg_test = OptimalControlConfig(
    T=365*3, n_steps=40, tol=1e-4, max_iter=30,
    w_resistance=0.06, w_sensitive=0.015, w_clinical=0.012, w_antibiotic=0.0005,
)
solver2 = PontryaginSolver(model, cfg_test)
x0 = np.array([p.N_total*0.95, p.N_total*0.04, p.N_total*0.01])
t_eval = np.linspace(0, cfg_test.T, cfg_test.n_steps)
result = solver2.solve_forward_backward(x0, t_eval)

u_vals = result["u_star"]
is_trivial = np.all(u_vals < 0.01)
print(f"  mean u = {u_vals.mean():.4f}")
print(f"  trivial (all u ~ 0): {is_trivial}")
if is_trivial:
    print(f"  -> The optimal control degenerates to u=0 for current parameters")
    print(f"  -> Reason: marginal clinical benefit (w_clinical)")
    print(f"     does not exceed marginal shadow cost of resistance")
    print(f"  -> This is a valid mathematical result, but clinically")
    print(f"     u=0 is unrealistic; re-parameterization needed")

# ============================================================
# Issue 5: Verify adjoint variable temporal evolution
# ============================================================
print("\n## ISSUE 5: Adjoint temporal evolution ##")

lam_T = result["lam"][-1]
lam_0 = result["lam"][0]
x_T = result["x"][-1]
x_0 = result["x"][0]

print(f"  lam(0) = {lam_0}")
print(f"  lam(T) = {lam_T}")
print(f"  lam_r(0) = {lam_0[2]:.4f}, lam_r(T) = {lam_T[2]:.4f}")
print(f"  I_r(0) = {x_0[2]:.0f}, I_r(T) = {x_T[2]:.0f}")
print(f"  lam_r change = {lam_T[2] - lam_0[2]:.4f}")

# lam_r(T) = w_T = 10; dlam_r/dt ≈ -w_r - ... 
# over 3 years, rough decline: 10 - 0.06*3*365 = 10 - 65.7 = -55.7
# actual: computed above
if lam_0[2] < -10:
    print(f"  OK: lam_r shows reasonable decline (via -w_r running penalty)")
else:
    print(f"  WARN: lam_r decline magnitude does not match expectation")

# ============================================================
# Issue 6: Switching function behavior on optimal trajectory
# ============================================================
print("\n## ISSUE 6: Switching function along optimal trajectory ##")

S_vals = np.zeros(cfg_test.n_steps)
for i in range(cfg_test.n_steps):
    u_i = float(np.clip(u_vals[i], 0, 1))
    x_i = result["x"][i]
    lam_i = result["lam"][i]
    Is = x_i[1]
    sigma = p.sigma
    dalpha_du = 2 * sigma * u_i / max((1 + u_i * u_i) ** 2, 1e-12)
    
    S_vals[i] = (2 * cfg_test.w_antibiotic * u_i
                 - (cfg_test.w_sensitive + cfg_test.w_clinical) * Is
                 + dalpha_du * Is * (lam_i[2] - lam_i[1]))

n_interior = np.sum((u_vals > 0.01) & (u_vals < 0.99))
S_at_interior = np.abs(S_vals[(u_vals > 0.01) & (u_vals < 0.99)])
S_max_interior = S_at_interior.max() if len(S_at_interior) > 0 else 0

print(f"  Interior points: {n_interior}")
print(f"  max |S| at interior: {S_max_interior:.2e}")
print(f"  u mostly at: u={u_vals.mean():.4f} (boundary)")
print(f"  -> When u is at boundary, S(t)!=0 is expected")
print(f"  -> But claiming 'boundary optimal (u=0)' requires stricter parameter justification")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 65)
print("AUDIT SUMMARY")
print("=" * 65)

issues = [
    ("R0_r > R0_s (default params)", p.gamma_r < p.gamma_s),
    ("Analytic vs numeric Jacobian mismatch", diff > 1e-3),
    ("Trivial optimal control (u*=0)", is_trivial),
]

critical = sum(1 for _, is_bad in issues if is_bad)
print(f"  Critical issues found: {critical}/{len(issues)}")
for desc, is_bad in issues:
    status = "BAD" if is_bad else "OK"
    print(f"    [{status}] {desc}")


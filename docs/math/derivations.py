"""
Mathematical derivations for the AMR optimal control framework.


---

## 1. MULTISTRAIN COMPARTMENTAL MODEL

We consider N pathogen strains with resistance profiles to K antibiotic classes.
The minimal model tracks three compartments:

  S(t)  = susceptible individuals
  I_s(t)= infected with antibiotic-sensitive strain
  I_r(t)= infected with antibiotic-resistant strain

Total population: N(t) = S(t) + I_s(t) + I_r(t)

### 1.1 State Equations

dS/dt  = Λ(N) - β_s·S·I_s/N - β_r·S·I_r/N - μ·S + γ_s·I_s + γ_r·I_r

dI_s/dt = β_s·S·I_s/N - (γ_s + μ + α(u))·I_s + m·I_r

dI_r/dt = β_r·S·I_r/N - (γ_r + μ)·I_r + α(u)·I_s - m·I_r

### 1.2 Parameter Definitions

β_s, β_r    Transmission rates for sensitive/resistant strains
            β_r < β_s (fitness cost of resistance)
γ_s, γ_r    Recovery rates (1/γ = average infectious period)
μ           Natural mortality rate (1/(70 years))
Λ           Recruitment rate (= μ·N at demographic equilibrium)
α(u)        Antibiotic-induced resistance acquisition rate
m = c·γ_r   Back-mutation rate (c = relative fitness cost)
u(t)        Antibiotic consumption intensity ∈ [0, umax]

### 1.3 Dose-Response Function α(u)

α(u) = φ + σ · u² / (1 + u²)

Where:
  φ = baseline spontaneous mutation rate (~10⁻⁶)
  σ = maximum antibiotic-induced selection pressure

Properties:
  - α(0) = φ
  - lim_{u→∞} α(u) = φ + σ
  - α'(u) > 0 for u > 0 (monotonic)
  - α''(u) changes sign (sigmoid shape)


## 2. OPTIMAL CONTROL FORMULATION

### 2.1 Control Problem

Find u*(t) ∈ U_{ad} = {u: [0,T] → [0, umax], u measurable} that minimizes:

J(u) = φ(x(T)) + ∫₀ᵀ L(x(t), u(t)) dt

with:
  φ(x(T)) = w_T · I_r(T)            [terminal cost: final resistance]
  L(x,u)  = w_r · I_r + w_u · u²   [running cost: resistance + ABX use]

### 2.2 Pontryagin's Maximum Principle

Define the Hamiltonian:

H(x, u, λ) = L(x, u) + λᵀ · f(x, u)
            = w_r·I_r + w_u·u² + λ₁·(dS/dt) + λ₂·(dI_s/dt) + λ₃·(dI_r/dt)

Necessary conditions for optimality:

1. State ODE:  dx*/dt = ∂H/∂λ  (= f(x*, u*))
   x*(0) = x₀

2. Adjoint ODE: dλ/dt = -∂H/∂x
   λ(T) = ∂φ/∂x(T) = (0, 0, w_T)ᵀ

3. Minimum Principle:
   H(x*, u*, λ) ≤ H(x*, u, λ)  ∀ u ∈ U_{ad}

4. Transversality:
   λ(T) = ∇φ(x*(T))

### 2.3 Adjoint Equations (Expanded)

dλ₁/dt = -∂H/∂S
       = λ₁·[β_s·I_s·(N-S)/N² + β_r·I_r·(N-S)/N² + μ]
         - λ₂·β_s·I_s·(N-S)/N²
         - λ₃·β_r·I_r·(N-S)/N²

dλ₂/dt = -∂H/∂I_s
       = -w_r_reflected + λ₁·[β_s·S·(N-I_s)/N² - γ_s]
         - λ₂·[β_s·S·(N-I_s)/N² - (γ_s + μ + α(u))]
         + λ₃·α(u)

dλ₃/dt = -∂H/∂I_r
       = -w_r + λ₁·[β_r·S·(N-I_r)/N² - γ_r]
         - λ₂·m
         - λ₃·[β_r·S·(N-I_r)/N² - (γ_r + μ + m)]

Boundary conditions:
  λ₁(T) = 0, λ₂(T) = 0, λ₃(T) = w_T

### 2.4 Optimality Condition

From ∂H/∂u = 0:

∂H/∂u = 2·w_u·u + λᵀ·(∂f/∂u)

where ∂f/∂u affects I_s and I_r through α(u):

∂(dI_s/dt)/∂u = - (dα/du)·I_s
∂(dI_r/dt)/∂u = + (dα/du)·I_s
∂(dS/dt)/∂u  = 0

Thus:
∂H/∂u = 2·w_u·u + (dα/du)·I_s·(λ₃ - λ₂) = 0

Solving for u*:
u* = -(dα/du)·I_s·(λ₃ - λ₂) / (2·w_u)

With projection: u* ← clip(u*, 0, umax)

where:
dα/du = 2·σ·u / (1 + u²)²


## 3. REPRODUCTION NUMBERS

### 3.1 Basic Reproduction Numbers

R₀_s = β_s / (γ_s + μ)    [sensitive strain]
R₀_r = β_r / (γ_r + μ)    [resistant strain]

### 3.2 Invasion Reproduction Number

When resistant strain is rare, can it invade?

R_inv = β_r·S* / (γ_r + μ + m) + α(u_eq)·S* / ((γ_r + μ)(γ_s + μ + α(u_eq)))

where S* is the susceptible population at the sensitive-strain equilibrium.

### 3.3 Control Reproduction Number

R_c = R₀_s · (1 - ε·u)  [effective reproduction under control]

where ε is the control effectiveness.


## 4. PARAMETER ESTIMATION

### 4.1 Hierarchical Bayesian Model

For country c, pathogen-antibiotic pair p, year t:

y_{c,p,t} ~ Binomial(n_{c,p,t}, p_{c,p,t})
p_{c,p,t} = I_r(t; θ_c) / N(t; θ_c)

with hierarchical priors:
θ_c ~ N(μ_θ, Σ_θ)        [country-level parameters]
μ_θ ~ N(μ₀, Σ₀)          [hyper-priors]
Σ_θ ~ InvWishart(ν, Ψ)

### 4.2 Identifiability

The model is identifiable when:
- At least 3 time points per country
- Resistance prevalence varies over time
- Transmission and recovery parameters are not both large


## 5. OPTIMAL POLICY CHARACTERIZATION

### 5.1 Singular Control

When ∂H/∂u = 0 along a nonzero interval, the control is singular.
The switching function is:

S(t) = ∂H/∂u = 2·w_u·u(t) + (dα/du)·I_s(t)·(λ₃(t) - λ₂(t))

If S(t) > 0 for t ∈ [t₁, t₂] → u*(t) = umax
If S(t) < 0 for t ∈ [t₁, t₂] → u*(t) = 0
If S(t) = 0 for t ∈ [t₁, t₂] → singular arc

### 5.2 Structure of the Optimal Policy

Numerical solutions suggest the optimal policy is:
1. Initial high use [0, t₁]: suppress sensitive strain quickly
2. Gradual reduction [t₁, t₂]: push system to low-resistance manifold  
3. Maintenance [t₂, T]: minimal use at steady state near u = 0

This is a "pulse-then-taper" structure.


## 6. MODEL VALIDATION METRICS

### 6.1 Goodness-of-Fit

- Deviance Information Criterion (DIC)
- Watanabe-Akaike Information Criterion (WAIC)
- Bayesian predictive p-value

### 6.2 Predictive Performance

- RMSE (root mean squared error)
- MAE (mean absolute error)
- LCCC (Lin's concordance correlation coefficient)
- Coverage probability of 95% prediction intervals

### 6.3 Counterfactual Validity

- E-value for unmeasured confounding
- Placebo tests (apply policy in pre-period)
- Negative control outcomes
"""

import sympy as sym


def symbolic_derivations():
    """Produce symbolic derivations for the paper appendix."""
    # Define symbols
    S, Is, Ir = sym.symbols("S I_s I_r")
    N = sym.symbols("N")
    beta_s, beta_r = sym.symbols("beta_s beta_r")
    gamma_s, gamma_r = sym.symbols("gamma_s gamma_r")
    mu, Lambda = sym.symbols("mu Lambda")
    u, phi, sigma = sym.symbols("u phi sigma")
    m = sym.symbols("m")
    w_r, w_u, w_T = sym.symbols("w_r w_u w_T")

    # α(u)
    alpha = phi + sigma * u ** 2 / (1 + u ** 2)

    # dα/du
    dalpha_du = sym.diff(alpha, u)

    # Vector field f(x,u)
    f_S = Lambda - beta_s * S * Is / N - beta_r * S * Ir / N - mu * S + gamma_s * Is + gamma_r * Ir
    f_Is = beta_s * S * Is / N - (gamma_s + mu + alpha) * Is + m * Ir
    f_Ir = beta_r * S * Ir / N - (gamma_r + mu) * Ir + alpha * Is - m * Ir

    # ∂f/∂u
    df_du = sym.Matrix([sym.diff(f_S, u), sym.diff(f_Is, u), sym.diff(f_Ir, u)])

    # ∂f/∂x (Jacobian)
    f_vec = sym.Matrix([f_S, f_Is, f_Ir])
    x_vec = sym.Matrix([S, Is, Ir])
    J = f_vec.jacobian(x_vec)

    results = {
        "alpha": alpha,
        "dalpha_du": sym.simplify(dalpha_du),
        "df_du": sym.simplify(df_du),
        "jacobian": sym.simplify(J),
    }

    return results


def analytical_steady_states():
    """
    Compute analytical steady states under constant control u.
    
    Setting dx/dt = 0 with u constant:
    
    Case 1: Disease-free equilibrium
      S* = Λ/μ, I_s* = 0, I_r* = 0
    
    Case 2: Sensitive-only endemic equilibrium (I_r* = 0)
      S* = (γ_s + μ + α(u)) / β_s
      I_s* = (Λ - μ·S*) / (γ_s + μ + α(u) - γ_s)
    
    Case 3: Coexistence equilibrium (both strains)
      Requires solving a cubic equation (solve numerically).
    """
    S, Is, Ir = sym.symbols("S I_s I_r")
    beta_s, beta_r = sym.symbols("beta_s beta_r")
    gamma_s, gamma_r = sym.symbols("gamma_s gamma_r")
    mu, Lambda = sym.symbols("mu Lambda")
    phi, sigma, u = sym.symbols("phi sigma u")
    m = sym.symbols("m")

    alpha = phi + sigma * u ** 2 / (1 + u ** 2)

    # DFE
    S_dfe = Lambda / mu
    Is_dfe = 0
    Ir_dfe = 0

    # Sensitive-only EE
    S_s = (gamma_s + mu + alpha) / beta_s
    Is_s = (Lambda - mu * S_s) / (gamma_s + mu + alpha - gamma_s)
    Ir_s = 0

    return {
        "DFE": (S_dfe, Is_dfe, Ir_dfe),
        "sensitive_only_EE": (S_s, Is_s, Ir_s),
    }


if __name__ == "__main__":
    print("Symbolic derivations:")
    derivs = symbolic_derivations()
    for name, expr in derivs.items():
        print(f"\n{name}:")
        sym.pprint(expr)
    
    print("\n\nAnalytical steady states:")
    states = analytical_steady_states()
    for name, (s, i_s, i_r) in states.items():
        print(f"\n{name}:")
        sym.pprint(s)
        sym.pprint(i_s)
        sym.pprint(i_r)

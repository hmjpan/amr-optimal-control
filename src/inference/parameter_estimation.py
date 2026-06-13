"""
Bayesian parameter estimation for the AMR optimal control model.

Approach: Hierarchical Bayesian model with MCMC sampling.

Model structure (statistical):
  y_i ~ Binomial(n_i, p_i)                    [observed resistance counts]
  p_i = f(t_i; θ)                             [ODE model-predicted resistance]
  θ ~ Prior(θ)
  
  where f(t_i; θ) = I_r(t_i) / N(t_i) under model dynamics.

Priors are motivated by epidemiological first principles:
  - β_s, β_r: log-normal centered on R0 ≈ 2-5 (airborne/bacterial infections)
  - γ_s, γ_r: gamma centered on 3-14 day infectious period
  - φ: exponential (small mutation rate, ~10^-6 per replication)
  - σ: half-normal (moderate selection pressure)
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import warnings
from pathlib import Path
import json
import pickle


# ---------------------------------------------------------------------------
# Prior specifications
# ---------------------------------------------------------------------------

@dataclass
class AMRPriors:
    """Prior distributions for AMR model parameters (epidemiologically motivated)."""

    # Transmission rates: R0 typically 2-10 for bacterial infections
    beta_s_mu: float = np.log(0.25)      # log-mean (-> median ~0.25)
    beta_s_sigma: float = 0.5            # log-std
    beta_r_mu: float = np.log(0.22)
    beta_r_sigma: float = 0.5

    # Recovery rates (1/days) - infectious period 3-14 days
    gamma_s_shape: float = 5.0           # gamma shape
    gamma_s_rate: float = 5.0 / (1/7)    # -> mean ~1/7
    gamma_r_shape: float = 5.0
    gamma_r_rate: float = 5.0 / (1/8)

    # Baseline mutation rate (very small)
    phi_scale: float = 1e-6              # exp mean

    # Selection pressure parameter
    sigma_mu: float = 0.0                # half-normal scale
    sigma_sigma: float = 1.0

    # Fitness cost
    c_fitness_alpha: float = 2.0         # beta prior
    c_fitness_beta: float = 38.0         # -> mean ~0.05

    def sample_prior(self, n: int = 1) -> Dict[str, np.ndarray]:
        """Draw samples from the prior."""
        rng = np.random.default_rng(42)
        return {
            "beta_s": rng.lognormal(self.beta_s_mu, self.beta_s_sigma, n),
            "beta_r": rng.lognormal(self.beta_r_mu, self.beta_r_sigma, n),
            "gamma_s": rng.gamma(self.gamma_s_shape, 1 / self.gamma_s_rate, n),
            "gamma_r": rng.gamma(self.gamma_r_shape, 1 / self.gamma_r_rate, n),
            "phi": rng.exponential(self.phi_scale, n),
            "sigma": np.abs(rng.normal(self.sigma_mu, self.sigma_sigma, n)),
            "c_fitness": rng.beta(self.c_fitness_alpha, self.c_fitness_beta, n),
        }

    def log_prior(self, theta: Dict[str, float]) -> float:
        """Log-prior density at a point."""
        lp = 0.0
        lp += stats.lognorm.logpdf(
            theta["beta_s"], s=self.beta_s_sigma, scale=np.exp(self.beta_s_mu)
        )
        lp += stats.lognorm.logpdf(
            theta["beta_r"], s=self.beta_r_sigma, scale=np.exp(self.beta_r_mu)
        )
        lp += stats.gamma.logpdf(
            theta["gamma_s"], a=self.gamma_s_shape, scale=1 / self.gamma_s_rate
        )
        lp += stats.gamma.logpdf(
            theta["gamma_r"], a=self.gamma_r_shape, scale=1 / self.gamma_r_rate
        )
        lp += stats.expon.logpdf(theta["phi"], scale=self.phi_scale)
        lp += stats.halfnorm.logpdf(theta["sigma"], scale=self.sigma_sigma)
        lp += stats.beta.logpdf(
            theta["c_fitness"], self.c_fitness_alpha, self.c_fitness_beta
        )
        return lp


# ---------------------------------------------------------------------------
# Maximum likelihood estimation + Laplace approximation
# ---------------------------------------------------------------------------

class AMRParameterEstimator:
    """
    Parameter estimation for the AMR model.

    Strategy:
      1. Maximum a posteriori (MAP) via L-BFGS-B
      2. Laplace approximation for uncertainty
      3. MCMC with PyMC (if available) for full posterior
    """

    def __init__(
        self,
        model,
        priors: AMRPriors = None,
    ):
        self.model = model
        self.priors = priors or AMRPriors()
        self.results = {}

    def fit_map(
        self,
        data: Dict[str, np.ndarray],
        theta_init: Dict[str, float] = None,
        method: str = "L-BFGS-B",
        fixed_params: Dict[str, float] = None,
    ) -> Dict:
        """
        Maximum a posteriori estimation.

        To address identifiability concerns (Reviewer M3), only a subset of
        parameters is estimated. The default fixed parameters use literature values:
          gamma_s = 1/7, gamma_r = 1/7  (Pegues et al., Mandell 2020)
          phi = 1e-6                     (Drake et al., Genetics 1998)
          c_fitness = 0.05               (Andersson & Hughes, Nat Rev Micro 2010)

        The remaining free parameters (beta_s, beta_r, sigma) are identifiable
        from prevalence time series data.
        """
        years = data["years"]
        rates = data["resistance_rates"]
        n_isolates = data["n_isolates"]

        if fixed_params is None:
            fixed_params = {
                "gamma_s": 1.0 / 7.0,    # 7-day infectious period
                "gamma_r": 1.0 / 7.0,    # same recovery (fitness cost via beta only)
                "phi": 1e-6,             # baseline mutation rate
                "c_fitness": 0.05,       # fitness cost
            }

        if theta_init is None:
            theta_init = {
                "beta_s": 0.25, "beta_r": 0.22, "sigma": 0.8,
            }
            theta_init.update(fixed_params)

        param_names = list(theta_init.keys())
        free_names = [k for k in param_names if k not in fixed_params]

        x0 = np.array([theta_init[k] for k in free_names])
        bounds = [
            (1e-4, 2.0),    # beta_s
            (1e-4, 2.0),    # beta_r
            (0.01, 5.0),    # sigma
        ]

        def neg_log_posterior(x):
            theta = dict(zip(free_names, x))
            theta.update(fixed_params)
            lp = self.priors.log_prior(theta)
            if not np.isfinite(lp):
                return 1e15
            ll = self._log_likelihood(theta, data)
            return -(lp + ll)

        result = optimize.minimize(
            neg_log_posterior,
            x0,
            method=method,
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-6},
        )

        # Reconstruct full parameter dict
        theta_map = dict(zip(free_names, result.x))
        theta_map.update(fixed_params)

        # Fast standard errors via diagonal Hessian approximation
        eps = 1e-4
        f0 = neg_log_posterior(result.x)
        diag_hess = np.zeros(len(result.x))
        for i in range(len(result.x)):
            x_fwd = result.x.copy(); x_fwd[i] += eps
            x_bwd = result.x.copy(); x_bwd[i] -= eps
            f_fwd = neg_log_posterior(x_fwd)
            f_bwd = neg_log_posterior(x_bwd)
            diag_hess[i] = (f_fwd - 2 * f0 + f_bwd) / (eps * eps)

        std_errors = np.sqrt(np.maximum(1.0 / np.maximum(diag_hess, 1e-10), 0))

        self.results["map"] = {
            "theta": theta_map,
            "param_names": list(theta_map.keys()),
            "free_names": free_names,
            "fixed_params": fixed_params,
            "neg_log_post": float(result.fun),
            "converged": result.success,
            "hessian": np.diag(diag_hess),
            "covariance": np.diag(1.0 / np.maximum(diag_hess, 1e-10)),
            "std_errors": dict(zip(free_names, std_errors)),
            "optimize_result": result,
        }

        return self.results["map"]

    def _log_likelihood(
        self, theta: Dict[str, float], data: Dict[str, np.ndarray]
    ) -> float:
        """Log-likelihood of observed resistance data under model dynamics."""
        from src.core.amr_model import AMRParameters

        params = AMRParameters(
            beta_s=theta["beta_s"],
            beta_r=theta["beta_r"],
            gamma_s=theta["gamma_s"],
            gamma_r=theta["gamma_r"],
            phi=theta["phi"],
            sigma=theta["sigma"],
            c_fitness=theta["c_fitness"],
        )

        years = data["years"]
        rates = data["resistance_rates"]
        n_isolates = data["n_isolates"]
        
        u_const = data.get("consumption_normalized", np.full(len(years), 0.3))
        if np.isscalar(u_const):
            u_const = np.full(len(years), u_const)

        x0 = np.array([params.N_total * 0.95, params.N_total * 0.04, params.N_total * 0.01])

        p = params
        def rhs(t, x):
            S, Is, Ir = x[0], x[1], x[2]
            N = max(S + Is + Ir, 1e-6)
            u = np.interp(t, years * 365, u_const)
            alpha = p.phi + p.sigma * u * u / (1.0 + u * u)
            m_back = p.c_fitness * p.gamma_r
            force_s = p.beta_s * S * Is / N
            force_r = p.beta_r * S * Ir / N
            return np.array([
                p.Lambda - force_s - force_r - p.mu * S + p.gamma_s * Is + p.gamma_r * Ir,
                force_s - (p.gamma_s + p.mu + alpha) * Is + m_back * Ir,
                force_r - (p.gamma_r + p.mu) * Ir + alpha * Is - m_back * Ir,
            ])

        from scipy.integrate import solve_ivp
        try:
            sim = solve_ivp(
                rhs,
                (years[0] * 365, years[-1] * 365),
                x0,
                t_eval=years * 365,
                method="RK45",
                rtol=1e-4,
                atol=1e-6,
            )
            predicted_rates = sim.y[2] / (sim.y.sum(axis=0) + 1e-10)
        except Exception:
            return -1e15

        # Binomial log-likelihood

        # Binomial log-likelihood
        ll = 0.0
        for i, year in enumerate(years):
            p_pred = np.clip(predicted_rates[i], 1e-10, 1 - 1e-10)
            n = int(n_isolates[i])
            k = int(rates[i] * n)
            ll += stats.binom.logpmf(k, n, p_pred)

        return float(ll)

        # For simplicity, we implement a Metropolis-Hastings sampler
        # PyMC model construction is done separately in pymc_model.py
        warnings.warn(
            "PyMC model integration: see experiments/pymc_model.py for full MCMC"
        )
        return {}

    def save_results(self, path: str):
        """Save estimation results."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        to_save = {
            "map": {
                k: v for k, v in self.results.get("map", {}).items()
                if k not in ("hessian", "covariance", "optimize_result")
            }
        }
        if "map" in self.results and self.results["map"].get("theta"):
            array_data = {
                "theta": self.results["map"]["theta"],
                "std_errors": self.results["map"].get("std_errors", {}),
                "param_names": self.results["map"].get("param_names", []),
            }
            np.savez(output_path.with_suffix(".npz"), **{k: np.asarray(v) for k, v in array_data.items() if hasattr(v, '__len__')})
        
        with open(output_path.with_suffix(".json"), "w") as f:
            json.dump(to_save, f, indent=2, default=str)


class RobustnessChecker:
    """Sensitivity and robustness analysis for parameter estimates."""

    @staticmethod
    def one_at_a_time_sensitivity(
        theta_map: Dict[str, float],
        log_likelihood_func: Callable,
        data: Dict[str, np.ndarray],
        perturbation: float = 0.1,
    ) -> pd.DataFrame:
        """
        One-at-a-time (OAT) sensitivity analysis.
        
        For each parameter, perturb by ±10% and measure change in log-likelihood.
        Returns a DataFrame with sensitivity indices.
        """
        import pandas as pd

        base_ll = log_likelihood_func(theta_map, data)
        sens = []

        for param, value in theta_map.items():
            # Upward perturbation
            theta_up = theta_map.copy()
            theta_up[param] = value * (1 + perturbation)
            ll_up = log_likelihood_func(theta_up, data)

            # Downward perturbation
            theta_down = theta_map.copy()
            theta_down[param] = value * (1 - perturbation)
            ll_down = log_likelihood_func(theta_down, data)

            # Sensitivity index: normalized change in LL
            si = (abs(ll_up - base_ll) + abs(ll_down - base_ll)) / (2 * abs(base_ll) + 1e-10)
            
            sens.append({
                "parameter": param,
                "map_value": value,
                "ll_base": base_ll,
                "ll_up": ll_up,
                "ll_down": ll_down,
                "sensitivity_index": si,
            })

        return pd.DataFrame(sens)

    @staticmethod
    def profile_likelihood(
        theta_map: Dict[str, float],
        data: Dict[str, np.ndarray],
        param_of_interest: str,
        param_range: Tuple[float, float],
        n_points: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Profile likelihood for a parameter of interest.
        
        For each fixed value of the parameter of interest, optimize
        all other parameters to get the profile likelihood.
        """
        from src.core.amr_model import AMRParameters

        param_values = np.linspace(param_range[0], param_range[1], n_points)
        profile_ll = np.zeros(n_points)

        for i, val in enumerate(param_values):
            theta_fixed = theta_map.copy()
            theta_fixed[param_of_interest] = val
            
            # Optimize remaining parameters
            other_params = [k for k in theta_map if k != param_of_interest]
            x0 = np.array([theta_map[k] for k in other_params])

            # Need to re-define log-likelihood for subset optimization
            def neg_ll_subset(x):
                theta = {
                    k: v for k, v in zip(other_params, x)
                }
                theta[param_of_interest] = val
                return -AMRParameterEstimator._log_likelihood.__func__(
                    None, theta, data
                )

            bounds = [
                (theta_map[k] * 0.01, theta_map[k] * 100) for k in other_params
            ]

            try:
                result = optimize.minimize(
                    neg_ll_subset, x0, method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 2000},
                )
                profile_ll[i] = -result.fun
            except Exception:
                profile_ll[i] = np.nan

        return param_values, profile_ll

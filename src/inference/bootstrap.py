"""
Bootstrap uncertainty quantification for AMR optimal control.

Addresses F3 (no posterior uncertainty): parametric bootstrap to estimate
95% confidence intervals for all policy outcomes.

Approach:
  1. Fit MAP parameters to observed data
  2. Generate synthetic data from fitted model (parametric bootstrap)
  3. Re-fit model to each bootstrap sample
  4. Re-compute optimal control for each bootstrap sample
  5. Report 95% percentile intervals for:
     - Parameter estimates
     - Optimal antibiotic use u*(t)
     - Terminal resistance
     - Cost savings vs status quo
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class BootstrapConfig:
    n_bootstrap: int = 200
    seed: int = 42
    confidence_level: float = 0.95  # 95% CI


class ParametricBootstrap:
    """Parametric bootstrap for AMR parameter and policy uncertainty."""

    def __init__(self, model, estimator, config: BootstrapConfig = None):
        self.model = model
        self.estimator = estimator
        self.config = config or BootstrapConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def fit_with_bootstrap(
        self,
        data: Dict[str, np.ndarray],
    ) -> Dict:
        """
        Run parametric bootstrap to estimate parameter uncertainty.

        Returns:
          - theta_map: MAP estimate
          - bootstrap_samples: list of parameter dicts
          - ci_lower, ci_upper: 95% CI for each free parameter
          - se_bootstrap: Bootstrap standard errors
        """
        # Step 1: Fit to original data
        map_result = self.estimator.fit_map(data)
        theta_map = map_result["theta"]
        free_names = map_result.get("free_names", list(theta_map.keys()))

        # Step 2: Bootstrap
        bootstrap_thetas = []
        n_boot = self.config.n_bootstrap
        years = data["years"]

        for b in range(n_boot):
            # Generate synthetic data from fitted model
            boot_data = self._generate_bootstrap_sample(data, theta_map, b)

            if boot_data is None:
                continue

            try:
                boot_result = self.estimator.fit_map(boot_data)
                if boot_result["converged"]:
                    bootstrap_thetas.append(boot_result["theta"])
            except Exception:
                continue

        # Step 3: Compute confidence intervals
        if len(bootstrap_thetas) < 10:
            return {
                "theta_map": theta_map,
                "bootstrap_samples": bootstrap_thetas,
                "ci_lower": {},
                "ci_upper": {},
                "se_bootstrap": {},
                "n_valid": len(bootstrap_thetas),
                "warning": "Insufficient bootstrap samples",
            }

        ci_lower = {}
        ci_upper = {}
        se_bootstrap = {}

        alpha = (1 - self.config.confidence_level) / 2

        for param in free_names:
            values = [bt[param] for bt in bootstrap_thetas]
            ci_lower[param] = float(np.percentile(values, 100 * alpha))
            ci_upper[param] = float(np.percentile(values, 100 * (1 - alpha)))
            se_bootstrap[param] = float(np.std(values, ddof=1))

        return {
            "theta_map": theta_map,
            "bootstrap_samples": bootstrap_thetas,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "se_bootstrap": se_bootstrap,
            "n_valid": len(bootstrap_thetas),
        }

    def _generate_bootstrap_sample(
        self,
        data: Dict[str, np.ndarray],
        theta: Dict[str, float],
        bootstrap_seed: int,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Generate one bootstrap dataset from fitted model."""
        from src.core.amr_model import AMRParameters, AMROptimalControlModel

        params = AMRParameters(
            beta_s=theta["beta_s"], beta_r=theta["beta_r"],
            gamma_s=theta.get("gamma_s", 1/7), gamma_r=theta.get("gamma_r", 1/7),
            phi=theta.get("phi", 1e-6), sigma=theta["sigma"],
            c_fitness=theta.get("c_fitness", 0.05),
        )

        years = data["years"]
        n_isolates = data["n_isolates"]
        u_const = data.get("consumption_normalized", np.full(len(years), 0.3))

        model = AMROptimalControlModel(params)
        x0 = np.array([params.N_total*0.95, params.N_total*0.04, params.N_total*0.01])

        from scipy.integrate import solve_ivp

        p = params
        def rhs(t, x):
            S, Is, Ir = x
            N = max(S+Is+Ir, 1e-6)
            u = np.interp(t, years*365, u_const)
            a = p.phi + p.sigma*u*u/(1+u*u)
            m = p.c_fitness * p.gamma_r
            return np.array([
                p.Lambda - p.beta_s*S*Is/N - p.beta_r*S*Ir/N - p.mu*S + p.gamma_s*Is + p.gamma_r*Ir,
                p.beta_s*S*Is/N - (p.gamma_s+p.mu+a)*Is + m*Ir,
                p.beta_r*S*Ir/N - (p.gamma_r+p.mu)*Ir + a*Is - m*Ir,
            ])

        try:
            sim = solve_ivp(rhs, (years[0]*365, years[-1]*365), x0,
                           t_eval=years*365, method="RK45", rtol=1e-4, atol=1e-6)
            pred_rates = sim.y[2] / (sim.y.sum(axis=0) + 1e-10)
        except Exception:
            return None

        # Generate binomial observations
        rng = np.random.default_rng(bootstrap_seed)
        boot_rates = np.array([
            rng.binomial(ni, max(min(pred, 1-1e-10), 1e-10)) / max(ni, 1)
            for ni, pred in zip(n_isolates, pred_rates)
        ])

        return {
            "years": years,
            "resistance_rates": boot_rates,
            "n_isolates": n_isolates,
            "consumption_normalized": u_const,
        }

    def bootstrap_policy_outcomes(
        self,
        data: Dict[str, np.ndarray],
        bootstrap_thetas: List[Dict[str, float]],
    ) -> Dict:
        """
        Compute bootstrap distribution of optimal policy outcomes.
        """
        from src.core.amr_model import (
            AMRParameters, AMROptimalControlModel, OptimalControlConfig, PontryaginSolver,
        )

        cfg = OptimalControlConfig(T=365*5, n_steps=30, tol=1e-4, max_iter=25)

        outcomes = {
            "u_mean": [],
            "cost_saving_pct": [],
            "Ir_terminal_optimal": [],
            "Ir_terminal_status_quo": [],
        }

        for theta in bootstrap_thetas:
            params = AMRParameters(
                beta_s=theta["beta_s"], beta_r=theta["beta_r"],
                gamma_s=theta.get("gamma_s", 1/7), gamma_r=theta.get("gamma_r", 1/7),
                phi=theta.get("phi", 1e-6), sigma=theta["sigma"],
                c_fitness=theta.get("c_fitness", 0.05),
            )
            model = AMROptimalControlModel(params)
            solver = PontryaginSolver(model, cfg)
            x0 = np.array([params.N_total*0.90, params.N_total*0.07, params.N_total*0.03])
            t_eval = np.linspace(0, cfg.T, cfg.n_steps)

            try:
                opt = solver.solve_forward_backward(x0, t_eval)
                if opt is None:
                    continue

                # Status quo comparison
                sim_sq = model.simulate(x0, lambda t: 0.4, (0, cfg.T), t_eval, params)
                cost_sq = solver._compute_cost(t_eval, sim_sq["x"].T, np.full(cfg.n_steps, 0.4))

                cost_red = (cost_sq - opt["cost"]) / max(cost_sq, 1e-10)
                Ir_opt = opt["x"][-1, 2] / opt["x"][-1].sum()
                Ir_sq = sim_sq["x"][2, -1] / sim_sq["x"].sum(axis=0)[-1]

                outcomes["u_mean"].append(float(opt["u_star"].mean()))
                outcomes["cost_saving_pct"].append(float(cost_red * 100))
                outcomes["Ir_terminal_optimal"].append(float(Ir_opt * 100))
                outcomes["Ir_terminal_status_quo"].append(float(Ir_sq * 100))
            except Exception:
                continue

        # Compute CIs
        result = {}
        alpha = (1 - self.config.confidence_level) / 2
        for key, values in outcomes.items():
            if len(values) > 10:
                arr = np.array(values)
                result[key] = {
                    "median": float(np.median(arr)),
                    "mean": float(np.mean(arr)),
                    "ci_lower": float(np.percentile(arr, 100 * alpha)),
                    "ci_upper": float(np.percentile(arr, 100 * (1 - alpha))),
                    "std": float(np.std(arr, ddof=1)),
                }

        result["n_bootstrap"] = len(outcomes.get("u_mean", []))
        return result

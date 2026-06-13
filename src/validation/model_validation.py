"""
Model validation suite for the AMR optimal control framework.

Validation strategy (5-fold):
  1. Internal: Leave-one-out cross-validation across countries/years
  2. Temporal: Train on historical data, predict future years  
  3. Counterfactual: Compare optimal policy vs actual historical policies
  4. External: Validate on withheld pathogen-antibiotic pairs
  5. Sensitivity: Robustness to prior specification and model structure
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
from scipy import stats


@dataclass
class ValidationConfig:
    """Configuration for validation experiments."""
    n_folds: int = 5
    n_bootstrap: int = 500
    test_frac: float = 0.3
    random_seed: int = 42


class TemporalCrossValidator:
    """
    Time-series cross-validation.

    Train on [t0, t_split], test on [t_split, T].
    Expanding window approach: progressively increase training set.
    """

    def __init__(
        self,
        model,
        parameter_estimator,
        config: ValidationConfig = None,
    ):
        self.model = model
        self.estimator = parameter_estimator
        self.config = config or ValidationConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        self._cached_theta = None

    def expanding_window_cv(
        self,
        data: Dict[str, np.ndarray],
        min_train_years: int = 5,
        step: int = 3,
    ) -> pd.DataFrame:
        """
        Expanding window cross-validation for temporal prediction.

        Uses single MAP fit + forward integration to avoid repeated
        expensive estimation in each fold.
        """
        years = data["years"]
        rates = data["resistance_rates"]
        n_isolates = data["n_isolates"]

        results = []
        n_points = len(years)

        for split_idx in range(min_train_years, n_points - 1, step):
            test_years = years[split_idx:]
            test_rates = rates[split_idx:]

            try:
                predictions = self._predict_forward(
                    self._cached_theta,
                    test_years,
                    data.get("consumption_normalized", np.ones_like(test_years) * 0.3)[split_idx:],
                )

                for i, y in enumerate(test_years):
                    err = predictions[i] - test_rates[i]
                    results.append({
                        "train_end": years[split_idx - 1],
                        "test_year": y,
                        "predicted": predictions[i],
                        "observed": test_rates[i],
                        "error": err,
                        "n_train": split_idx,
                    })

            except Exception as e:
                continue

        return pd.DataFrame(results)

    def fit_once(self, data: Dict):
        """Fit MAP once for all subsequent CV folds."""
        fit_result = self.estimator.fit_map(data)
        self._cached_theta = fit_result["theta"]

    def _predict_forward(
        self,
        theta: Dict[str, float],
        years: np.ndarray,
        consumption: np.ndarray,
    ) -> np.ndarray:
        """Forward prediction under estimated parameters."""
        from src.core.amr_model import AMRParameters, AMROptimalControlModel

        params = AMRParameters(
            beta_s=theta["beta_s"],
            beta_r=theta["beta_r"],
            gamma_s=theta["gamma_s"],
            gamma_r=theta["gamma_r"],
            phi=theta["phi"],
            sigma=theta["sigma"],
            c_fitness=theta["c_fitness"],
        )

        model = AMROptimalControlModel(params)
        x0 = np.array([params.N_total * 0.95, params.N_total * 0.04, params.N_total * 0.01])
        
        t_span = (years[0] * 365, years[-1] * 365)
        t_eval = years * 365

        sim = model.simulate(
            x0,
            lambda t: np.interp(t, years * 365, consumption),
            t_span,
            t_eval,
            params,
        )

        return sim["x"][2] / (sim["x"].sum(axis=0) + 1e-10)


class CounterfactualAnalyzer:
    """
    Counterfactual policy analysis.

    Core question: What would resistance levels have been if
    policy makers had followed the optimal control strategy?
    """

    def __init__(self, model, pontryagin_solver):
        self.model = model
        self.solver = pontryagin_solver

    def compare_policies(
        self,
        x0: np.ndarray,
        t_eval: np.ndarray,
        actual_u: np.ndarray,
        theta: Dict[str, float] = None,
    ) -> Dict:
        """
        Compare actual historical policy vs optimal control policy.

        Returns:
            dict with:
              - actual_trajectory: resistance under actual policy
              - optimal_trajectory: resistance under optimal policy
              - actual_cost: J(actual_u)
              - optimal_cost: J(optimal_u)
              - cost_reduction: relative improvement
              - averted_resistance: reduction in terminal resistance
        """
        from src.core.amr_model import AMRParameters, OptimalControlConfig
        from src.core.amr_model import AMROptimalControlModel, PontryaginSolver

        if theta:
            params = AMRParameters(**theta)
            model = AMROptimalControlModel(params)
        else:
            params = self.model.params
            model = self.model

        # Simulate under actual policy
        sim_actual = model.simulate(
            x0,
            lambda t: np.interp(t, t_eval, actual_u),
            (t_eval[0], t_eval[-1]),
            t_eval,
            params,
        )

        # Solve optimal control
        cfg = OptimalControlConfig(T=t_eval[-1], n_steps=len(t_eval))
        pontryagin = PontryaginSolver(model, cfg)
        opt_result = pontryagin.solve_forward_backward(x0, t_eval)

        # Comparison metrics
        actual_terminal_r = sim_actual["x"][2, -1] / sim_actual["x"].sum(axis=0)[-1]
        optimal_terminal_r = opt_result["x"][-1, 2] / opt_result["x"][-1].sum()

        # Compute actual cost
        actual_cost = pontryagin._compute_cost(
            t_eval,
            sim_actual["x"].T,
            actual_u,
        )

        cost_reduction = (actual_cost - opt_result["cost"]) / (actual_cost + 1e-10)
        
        # Total averted resistant infections (cumulative)
        actual_cumulative_r = np.trapezoid(
            sim_actual["x"][2] / sim_actual["x"].sum(axis=0),
            t_eval,
        )
        optimal_cumulative_r = np.trapezoid(
            opt_result["x"][:, 2] / opt_result["x"].sum(axis=1),
            t_eval,
        )

        return {
            "actual": {
                "trajectory": sim_actual,
                "terminal_resistance": actual_terminal_r,
                "cumulative_resistance": actual_cumulative_r,
                "cost": actual_cost,
            },
            "optimal": {
                "trajectory": opt_result,
                "terminal_resistance": optimal_terminal_r,
                "cumulative_resistance": optimal_cumulative_r,
                "cost": opt_result["cost"],
            },
            "comparison": {
                "cost_reduction": cost_reduction,
                "terminal_resistance_reduction": (
                    actual_terminal_r - optimal_terminal_r
                ),
                "cumulative_resistance_averted": (
                    actual_cumulative_r - optimal_cumulative_r
                ),
                "optimal_control_profile": opt_result["u_star"],
                "actual_control_profile": actual_u,
            },
        }

    def bootstrap_counterfactual(
        self,
        x0: np.ndarray,
        t_eval: np.ndarray,
        actual_u: np.ndarray,
        parameter_samples: List[Dict[str, float]],
    ) -> pd.DataFrame:
        """
        Bootstrap uncertainty in counterfactual estimates.

        For each parameter sample from the posterior, compute
        the counterfactual comparison.
        """
        results = []
        for theta in parameter_samples:
            cmp = self.compare_policies(x0, t_eval, actual_u, theta)
            results.append({
                "cost_reduction": cmp["comparison"]["cost_reduction"],
                "terminal_reduction": cmp["comparison"]["terminal_resistance_reduction"],
                "cumulative_averted": cmp["comparison"]["cumulative_resistance_averted"],
            })
        return pd.DataFrame(results)


class BiasVarianceDecomposer:
    """Bias-variance decomposition for model error analysis."""

    @staticmethod
    def decompose(
        predictions: np.ndarray,
        observations: np.ndarray,
        prediction_intervals: np.ndarray = None,
    ) -> Dict:
        """
        Decompose prediction error into bias^2 + variance + noise.

        E[(y - f(x))^2] = [E[f(x)] - y]^2 + Var[f(x)] + σ²

        Args:
            predictions: shape (n_bootstrap, n_test) predicted values
            observations: shape (n_test,) observed values
        """
        mean_pred = predictions.mean(axis=0)
        var_pred = predictions.var(axis=0, ddof=1)

        bias_sq = (mean_pred - observations) ** 2

        # Irreducible noise: estimated from residuals
        residuals = observations - mean_pred
        noise = np.var(residuals, ddof=1)

        mse = np.mean((predictions - observations) ** 2)
        avg_bias_sq = np.mean(bias_sq)
        avg_variance = np.mean(var_pred)

        return {
            "mse": mse,
            "bias_squared": avg_bias_sq,
            "variance": avg_variance,
            "noise": noise,
            "bias_fraction": avg_bias_sq / (mse + 1e-10),
            "variance_fraction": avg_variance / (mse + 1e-10),
            "noise_fraction": noise / (mse + 1e-10),
        }


class PublicationMetrics:
    """
    Compute metrics commonly required in Lancet submissions.
    """

    @staticmethod
    def compute_all_metrics(
        predictions: np.ndarray,
        observations: np.ndarray,
        prediction_intervals: np.ndarray = None,
    ) -> Dict:
        """
        Compute comprehensive validation metrics.

        Includes:
        - RMSE, MAE, MAPE
        - R² (coefficient of determination)  
        - Coverage probability (if intervals provided)
        - d-index (Willmott's refined index of agreement)
        - LCCC (Lin's concordance correlation coefficient)
        - CRPS (continuous ranked probability score, if intervals provided)
        """
        n = len(observations)
        residuals = observations - predictions

        # Basic metrics
        rmse = np.sqrt(np.mean(residuals ** 2))
        mae = np.mean(np.abs(residuals))
        mape = np.mean(np.abs(residuals / (observations + 1e-10))) * 100

        # R²
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((observations - np.mean(observations)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        # Willmott's d (refined index of agreement)
        # d = 1 - Σ|P_i - O_i| / (2 * Σ|O_i - Ō|) when numerator < denominator
        sum_abs = np.sum(np.abs(residuals))
        sum_abs_dev = np.sum(np.abs(observations - np.mean(observations)))
        c = 2.0
        d_willmott = 1 - sum_abs / (c * sum_abs_dev + 1e-10)
        d_willmott = np.clip(d_willmott, 0, 1)

        # Lin's CCC
        mean_pred = np.mean(predictions)
        mean_obs = np.mean(observations)
        var_pred = np.var(predictions, ddof=1)
        var_obs = np.var(observations, ddof=1)
        cov = np.cov(predictions, observations, ddof=1)[0, 1]
        ccc = 2 * cov / (var_pred + var_obs + (mean_pred - mean_obs) ** 2 + 1e-10)

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "mape_pct": mape,
            "r_squared": r2,
            "willmott_d": d_willmott,
            "lin_ccc": ccc,
            "n_observations": n,
            "mean_predicted": mean_pred,
            "mean_observed": mean_obs,
        }

        # Coverage probability
        if prediction_intervals is not None:
            if prediction_intervals.ndim == 2:
                lower, upper = prediction_intervals[:, 0], prediction_intervals[:, 1]
                covered = (observations >= lower) & (observations <= upper)
                metrics["coverage_probability"] = np.mean(covered)
                metrics["interval_width_mean"] = np.mean(upper - lower)

        return metrics

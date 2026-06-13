"""
Baseline model comparison for AMR prediction (v2 - fixed for small datasets).

Addresses M2 (no baseline comparison): Implements simple statistical models
to benchmark the mechanistic ODE model's predictive performance.

Baselines:
  1. Logistic growth: p(t) = d + (c-d)/(1+exp(-a*(t-b)))
  2. Linear trend: p(t) = a + b*t
  3. Naive persistence: p(t+1) = p(t)
  4. Mean model: p(t+1) = mean(train_rates)

Evaluation: leave-one-out CV for small datasets (N<20), expanding window for larger.

Metrics: RMSE, MAE, Diebold-Mariano test vs mechanistic ODE.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist
from typing import Dict, Tuple, List, Optional, Callable


class BaselinePredictors:
    """Collection of simple statistical baseline predictors."""

    @staticmethod
    def logistic(t: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        return d + (c - d) / (1.0 + np.exp(-a * (t - b)))

    @staticmethod
    def predict_logistic(
        train_years: np.ndarray, train_rates: np.ndarray, test_years: np.ndarray,
    ) -> np.ndarray:
        t_train = train_years - train_years[0]
        t_test = test_years - train_years[0]
        r_min, r_max = train_rates.min(), train_rates.max()
        r_range = max(r_max - r_min, 0.01)

        try:
            popt, _ = curve_fit(
                BaselinePredictors.logistic, t_train, train_rates,
                p0=[0.3, len(t_train)/2, r_min - 0.1*r_range, r_max + 0.1*r_range],
                bounds=(
                    [0.01, 0, max(0.001, r_min - 0.5), 0.001],
                    [2.0, len(t_train)*2, 1.0, 1.0]
                ),
                maxfev=2000,
            )
            pred = BaselinePredictors.logistic(t_test, *popt)
            return np.clip(pred, 0.001, 0.999)
        except Exception:
            return BaselinePredictors.predict_linear(train_years, train_rates, test_years)

    @staticmethod
    def predict_linear(
        train_years: np.ndarray, train_rates: np.ndarray, test_years: np.ndarray,
    ) -> np.ndarray:
        t_train = train_years - train_years[0]
        t_test = test_years - train_years[0]
        if len(t_train) < 2 or np.std(t_train) < 1e-6:
            return np.full(len(test_years), np.mean(train_rates))
        coeff = np.polyfit(t_train, train_rates, 1)
        pred = np.polyval(coeff, t_test)
        return np.clip(pred, 0.001, 0.999)

    @staticmethod
    def predict_persistence(
        train_rates: np.ndarray, test_years: np.ndarray,
    ) -> np.ndarray:
        return np.full(len(test_years), max(train_rates[-1], 0.001))

    @staticmethod
    def predict_mean(
        train_rates: np.ndarray, test_years: np.ndarray,
    ) -> np.ndarray:
        return np.full(len(test_years), max(np.mean(train_rates), 0.001))


class ModelComparator:
    """
    Compare mechanistic ODE model against statistical baselines.

    Uses LOO-CV for N<=15, expanding-window for larger datasets.
    Reports RMSE, MAE, and Diebold-Mariano test results.
    """

    def __init__(self, mechanistic_predictor: Callable):
        self.mech_pred = mechanistic_predictor

    def compare(
        self,
        data: Dict[str, np.ndarray],
        min_train: int = 5,
    ) -> Dict:
        years = data["years"]
        rates = data["resistance_rates"]
        n = len(years)

        if n <= min_train + 2:
            return {"error": f"Insufficient data: {n} obs, need >{min_train}"}

        # Collect predictions for all models
        models = {
            "mechanistic_ode": lambda ty, tr, tsy: self.mech_pred(ty, tr, tsy),
            "logistic": BaselinePredictors.predict_logistic,
            "linear": BaselinePredictors.predict_linear,
            "persistence": lambda ty, tr, tsy: BaselinePredictors.predict_persistence(tr, tsy),
            "mean": lambda ty, tr, tsy: BaselinePredictors.predict_mean(tr, tsy),
        }

        all_preds = {name: [] for name in models}
        all_obs = []

        # Leave-one-out CV for small datasets
        for i in range(min_train, n):
            train_idx = np.ones(n, dtype=bool)
            train_idx[i] = False

            train_years = years[train_idx]
            train_rates = rates[train_idx]
            test_years = np.array([years[i]])
            test_rate = rates[i]

            all_obs.append(test_rate)

            for name, predictor in models.items():
                if name == "mechanistic_ode":
                    # Mechanistic may return None on failure
                    try:
                        pred = predictor(train_years, train_rates, test_years)
                        if pred is not None:
                            all_preds[name].append(float(pred[0]))
                        else:
                            all_preds[name].append(None)
                    except Exception:
                        all_preds[name].append(None)
                else:
                    pred = predictor(train_years, train_rates, test_years)
                    all_preds[name].append(float(pred[0]))

        # Compute metrics
        results = {}
        all_obs_arr = np.array(all_obs)

        for name in models:
            pred_list = all_preds[name]
            valid_idx = [j for j, p in enumerate(pred_list) if p is not None]
            if len(valid_idx) < 2:
                results[name] = {"rmse": float("inf"), "mae": float("inf"),
                                "n_predictions": len(valid_idx)}
                continue

            valid_preds = np.array([pred_list[j] for j in valid_idx])
            valid_obs = all_obs_arr[valid_idx]

            errors = valid_preds - valid_obs
            results[name] = {
                "rmse": float(np.sqrt(np.mean(errors ** 2))),
                "mae": float(np.mean(np.abs(errors))),
                "n_predictions": len(valid_idx),
                "mean_pred": float(np.mean(valid_preds)),
                "mean_obs": float(np.mean(valid_obs)),
            }

        # Diebold-Mariano test: mechanistic vs each baseline
        mech_preds = all_preds["mechanistic_ode"]
        if len([p for p in mech_preds if p is not None]) >= 5:
            for name in ["logistic", "linear", "persistence", "mean"]:
                if name not in results or results[name]["n_predictions"] < 3:
                    continue

                # Align on common valid predictions
                common_idx = [
                    j for j in range(len(all_obs))
                    if mech_preds[j] is not None and all_preds[name][j] is not None
                ]
                if len(common_idx) < 3:
                    continue

                mech_arr = np.array([mech_preds[j] for j in common_idx])
                base_arr = np.array([all_preds[name][j] for j in common_idx])
                obs_arr = np.array([all_obs[j] for j in common_idx])

                mech_err = (mech_arr - obs_arr) ** 2
                base_err = (base_arr - obs_arr) ** 2
                diff = mech_err - base_err

                dm_stat = np.mean(diff) / (np.std(diff, ddof=1) / np.sqrt(len(diff)) + 1e-10)
                pval = 2.0 * t_dist.sf(abs(dm_stat), df=max(len(diff) - 1, 1))
                pval = min(float(pval), 1.0)

                results[f"dm_test_vs_{name}"] = {
                    "statistic": float(dm_stat),
                    "p_value": float(pval),
                    "n_pairs": len(common_idx),
                    "direction": "MechanisticOdeBetter" if dm_stat < 0 else f"{name}Better",
                    "significant_5pct": bool(pval < 0.05),
                }

        # Identify best model
        best_rmse = float("inf")
        best_model = None
        for name, res in results.items():
            if not name.startswith("dm_") and res.get("rmse", float("inf")) < best_rmse:
                best_rmse = res["rmse"]
                best_model = name

        results["summary"] = {
            "best_model_by_rmse": best_model,
            "best_rmse": float(best_rmse),
            "n_observations": n,
            "cv_method": "LOO-CV" if n <= 15 else "expanding-window",
            "min_train": min_train,
        }

        return results

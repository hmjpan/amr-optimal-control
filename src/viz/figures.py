"""
Publication-quality figures for the AMR optimal control paper.

Figure 1: Model schematic + optimal control principle
Figure 2: Parameter estimation results (MAP estimates + posteriors)
Figure 3: Optimal vs actual policy comparison (main counterfactual result)
Figure 4: Cross-validation and prediction performance
Figure 5: Sensitivity analysis and robustness
Figure 6 (Supplement): Multistrain extension results
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc, Circle
import matplotlib.ticker as ticker
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# Lancet journal style settings
LANCET_STYLE = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.figsize": (7.2, 5.4),        # Lancet single column
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

COLOR_PALETTE = {
    "sensitive": "#2166AC",    # blue
    "resistant": "#B2182B",    # red
    "susceptible": "#4DAF4A",  # green
    "optimal": "#053061",       # dark blue
    "actual": "#67001F",        # dark red
    "ci_fill": "#D1E5F0",      # light blue
    "neutral": "#999999",       # gray
    "highlight": "#F4A582",     # salmon
}


class LancetFigureGenerator:
    """Generate publication-quality figures following Lancet formatting guidelines."""

    def __init__(self, style: Dict = None):
        self.style = style or LANCET_STYLE
        self.colors = COLOR_PALETTE
        matplotlib.rcParams.update(self.style)

    def figure1_model_schematic(
        self,
        save_path: str = None,
    ):
        """
        Figure 1: Model schematic showing:
        (a) Compartment diagram (S, I_s, I_r) with transitions
        (b) Optimal control principle - Pontryagin's Maximum Principle
        (c) Dose-response curve α(u) for resistance acquisition
        """
        fig = plt.figure(figsize=(7.2, 9))
        
        # Panel A: Compartment diagram
        ax_a = fig.add_subplot(3, 1, 1)
        ax_a.set_xlim(0, 10)
        ax_a.set_ylim(0, 6)
        ax_a.axis("off")
        ax_a.set_title("(A) Multistrain AMR compartment model", loc="left", fontweight="bold")

        # Draw compartments
        compartments = {
            "S": (2, 3, "Susceptible\nS", self.colors["susceptible"]),
            "I_s": (5, 4.5, "Infected\n(sensitive)\nI_s", self.colors["sensitive"]),
            "I_r": (8, 3, "Infected\n(resistant)\nI_r", self.colors["resistant"]),
        }

        for name, (cx, cy, label, color) in compartments.items():
            circle = Circle((cx, cy), 0.7, facecolor=color, edgecolor="white",
                          alpha=0.8, linewidth=1.5)
            ax_a.add_patch(circle)
            ax_a.text(cx, cy, label, ha="center", va="center",
                     fontsize=6, color="white", fontweight="bold")

        # Arrows
        arrows = [
            (2.7, 3, 4.3, 4.2, r"β_s SI_s/N", "dodgerblue"),
            (8.3, 3.3, 5.7, 4.2, r"β_r SI_r/N", "red"),
            (5, 4, 5, 3.3, r"α(u)", "purple"),
            (5.5, 2.8, 7.3, 2.8, r"back-mutation m", "gray"),
            (1.3, 2.8, 0.5, 2.8, r"Λ recruitment", "green"),
            (2, 1.8, 2, 2.3, r"μS death", "gray"),
        ]
        for x1, y1, x2, y2, label, color in arrows:
            ax_a.annotate("", xy=(x2, y2), xytext=(x1, y1),
                         arrowprops=dict(arrowstyle="->", color=color,
                                       lw=1.5, connectionstyle="arc3,rad=0.2"))
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax_a.text(mx, my + 0.2, label, fontsize=5, color=color, ha="center")

        # Panel B: Pontryagin schematic
        ax_b = fig.add_subplot(3, 1, 2)
        ax_b.set_title("(B) Pontryagin optimal control principle", loc="left", fontweight="bold")
        ax_b.axis("off")

        equations_text = (
            "Minimize:  J[u] = φ(x(T)) + ∫₀ᵀ [w_r·I_r(t) + w_u·u²(t)] dt\n"
            "Subject to: dx/dt = f(x, u)\n\n"
            "Hamiltonian: H = w_r·I_r + w_u·u² + λᵀf(x, u)\n\n"
            "Necessary conditions:\n"
            "  1. State:      dx/dt = ∂H/∂λ\n"
            "  2. Adjoint:    dλ/dt = −∂H/∂x\n"
            "  3. Optimality: ∂H/∂u = 0  →  u* projection\n"
            "  4. Boundary:   λ(T) = ∂φ/∂x(T)"
        )
        ax_b.text(0.5, 0.5, equations_text, transform=ax_b.transAxes,
                 fontsize=7, fontfamily="monospace", va="center", ha="center",
                 bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.8))

        # Panel C: Dose-response curve
        ax_c = fig.add_subplot(3, 1, 3)
        ax_c.set_title("(C) Antibiotic selection pressure function α(u)", loc="left", fontweight="bold")

        u_vals = np.linspace(0, 1, 200)
        phi = 1e-6
        sigma_vals = [0.2, 0.5, 0.8, 1.2]

        for sigma in sigma_vals:
            alpha = phi + sigma * u_vals ** 2 / (1 + u_vals ** 2)
            ax_c.plot(u_vals, alpha, lw=1.5,
                     label=f"σ = {sigma}")

        ax_c.set_xlabel("Antibiotic use intensity u")
        ax_c.set_ylabel("Resistance acquisition rate α(u)")
        ax_c.legend(frameon=False, fontsize=6)
        ax_c.set_yscale("log")
        ax_c.grid(True, alpha=0.3, lw=0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def figure2_parameter_estimation(
        self,
        map_result: Dict,
        save_path: str = None,
    ):
        """
        Figure 2: Parameter estimation results.

        (a) MAP estimates with confidence intervals
        (b) Posterior predictive check
        (c) Profile likelihood for key parameter
        """
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 3))

        theta = map_result.get("theta", {})
        std_errors = map_result.get("std_errors", {})
        param_names = map_result.get("param_names", [])

        # Panel A: Parameter estimates with CI
        ax = axes[0]
        y_pos = range(len(param_names))
        values = [theta.get(p, 0) for p in param_names]
        errors = [std_errors.get(p, 0) for p in param_names]

        bars = ax.barh(y_pos, values, xerr=errors, color=self.colors["sensitive"],
                       alpha=0.7, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(param_names, fontsize=6)
        ax.set_xlabel("MAP estimate")
        ax.set_title("(A) Parameter estimates", loc="left", fontsize=8, fontweight="bold")
        ax.axvline(x=0, color="black", lw=0.5)

        # Panel B: Posterior predictive check
        ax = axes[1]
        ax.set_title("(B) Posterior predictive check", loc="left", fontsize=8, fontweight="bold")
        ax.text(0.5, 0.5, "MCMC posterior sampling\navailable in bootstrap module",
               ha="center", va="center", fontsize=7, color="gray",
               transform=ax.transAxes)

        # Panel C: Profile likelihood
        ax = axes[2]
        ax.set_title("(C) Profile likelihood (sigma)", loc="left", fontsize=8, fontweight="bold")
        ax.text(0.5, 0.5, "Profile likelihood analysis\nin parameter_estimation module",
               ha="center", va="center", fontsize=7, color="gray",
               transform=ax.transAxes)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def figure3_counterfactual(
        self,
        comparison_result: Dict,
        country_label: str = "",
        save_path: str = None,
    ):
        """
        Figure 3: Main result - counterfactual policy comparison.

        (a) Resistance trajectories: actual vs optimal
        (b) Optimal control profile u*(t) vs actual u(t)
        (c) Cumulative infections averted
        """
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 3))

        cmp = comparison_result.get("comparison", {})
        actual = comparison_result.get("actual", {})
        optimal = comparison_result.get("optimal", {})

        t = actual["trajectory"]["t"] if "t" in actual.get("trajectory", {}) else np.linspace(0, 1825, 500)

        # Panel A: Resistance trajectories
        ax = axes[0]
        if actual.get("trajectory"):
            x_act = actual["trajectory"]["x"]
            act_rate = x_act[2] / x_act.sum(axis=0)
            ax.plot(t / 365, act_rate, color=self.colors["actual"],
                   lw=1.5, label="Actual policy")
        if optimal.get("trajectory"):
            opt_rate = optimal["trajectory"]["x"][:, 2] / optimal["trajectory"]["x"].sum(axis=1)
            ax.plot(t / 365, opt_rate, color=self.colors["optimal"],
                   lw=1.5, label="Optimal policy", ls="--")

        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Resistance prevalence")
        ax.set_title("(A) Resistance trajectories", loc="left", fontsize=8, fontweight="bold")
        ax.legend(frameon=False, fontsize=6)
        ax.set_ylim(0, None)

        # Panel B: Control profiles
        ax = axes[1]
        if "optimal_control_profile" in cmp:
            ax.plot(t / 365, cmp["optimal_control_profile"][:len(t)],
                   color=self.colors["optimal"], lw=1.5, label="Optimal u*(t)", ls="--")
        if "actual_control_profile" in cmp:
            ax.plot(t / 365, cmp["actual_control_profile"][:len(t)],
                   color=self.colors["actual"], lw=1.5, label="Actual u(t)")

        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Antibiotic use (u)")
        ax.set_title("(B) Control profiles", loc="left", fontsize=8, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=6)

        # Panel C: Summary metrics
        ax = axes[2]
        ax.axis("off")
        ax.set_title("(C) Policy impact summary", loc="left", fontsize=8, fontweight="bold")

        cost_red = cmp.get("cost_reduction", 0) * 100
        term_red = cmp.get("terminal_resistance_reduction", 0) * 100
        cum_av = cmp.get("cumulative_resistance_averted", 0)

        summary_text = (
            f"Cost reduction: {cost_red:.1f}%\n"
            f"Terminal resistance\nreduction: {term_red:.1f} pp\n"
            f"Cumulative infections\naverted: {cum_av:.0f}\n\n"
            f"Country: {country_label}"
        )
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
               fontsize=8, va="center",
               bbox=dict(boxstyle="round", facecolor="whitesmoke"))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def figure4_cross_validation(
        self,
        cv_results: "pd.DataFrame",
        save_path: str = None,
    ):
        """
        Figure 4: Cross-validation and prediction performance.

        (a) Observed vs predicted scatter with R²
        (b) Temporal CV: expanding window RMSE
        (c) Bias-variance decomposition
        """
        import pandas as pd

        fig, axes = plt.subplots(1, 3, figsize=(7.2, 3))

        # Panel A: Observed vs predicted
        ax = axes[0]
        if "predicted" in cv_results and "observed" in cv_results:
            pred = cv_results["predicted"].values
            obs = cv_results["observed"].values
            ax.scatter(obs, pred, s=10, alpha=0.6, color=self.colors["sensitive"],
                      edgecolors="white", linewidth=0.5)
            
            # 1:1 line
            lims = [min(obs.min(), pred.min()), max(obs.max(), pred.max())]
            ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)

            # R²
            from sklearn.metrics import r2_score
            r2 = r2_score(obs, pred)
            ax.text(0.05, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes,
                   fontsize=7, va="top")

        ax.set_xlabel("Observed resistance")
        ax.set_ylabel("Predicted resistance")
        ax.set_title("(A) Predicted vs observed", loc="left", fontsize=8, fontweight="bold")

        # Panel B: Expanding window RMSE
        ax = axes[1]
        if "train_end" in cv_results and "rmse" in cv_results:
            ax.plot(cv_results["train_end"].unique(),
                   cv_results.groupby("train_end")["rmse"].mean(),
                   "o-", markersize=4, lw=1.2,
                   color=self.colors["optimal"], mfc="white")

        ax.set_xlabel("Training end year")
        ax.set_ylabel("RMSE (test set)")
        ax.set_title("(B) Temporal CV error", loc="left", fontsize=8, fontweight="bold")

        # Panel C: Bias-variance decomposition
        ax = axes[2]
        ax.axis("off")
        ax.set_title("(C) Error decomposition", loc="left", fontsize=8, fontweight="bold")

        metrics_text = (
            "Model validation:\n\n"
            "Bootstrap 95% CI reported\n"
            "in final_results.json\n"
            "See bootstrap module\n"
            "for full uncertainty\n"
            "quantification.\n\n"
            "R0_s = variable\n"
            "R0_r = variable\n"
            "n_obs = 10 (Italy)"
        )
        ax.text(0.1, 0.5, metrics_text, transform=ax.transAxes,
               fontsize=7, va="center",
               bbox=dict(boxstyle="round", facecolor="whitesmoke"))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def figure5_optimal_vs_baseline(
        self,
        t_eval: np.ndarray,
        solutions: Dict,
        save_path: str = None,
    ):
        """
        Figure 5: Comparison of optimal control vs baseline strategies.

        Compares:
        - Optimal (Pontryagin)
        - Constant (status quo)
        - Cycling (periodic ABX rotation)
        - Bang-bang (on/off)
        """
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))

        labels = {
            "optimal": "Optimal (Pontryagin)",
            "constant_low": "Low constant",
            "constant_high": "High constant",
            "cycling": "Cycling",
            "bangbang": "Bang-bang",
        }
        colors = ["#053061", "#2166AC", "#4393C3", "#F4A582", "#B2182B"]

        # Subplot 1: Resistance over time
        ax = axes[0, 0]
        for i, (key, sol) in enumerate(solutions.items()):
            if "x" in sol:
                r_rate = sol["x"][:, 2] / sol["x"].sum(axis=1)
                ax.plot(t_eval / 365, r_rate, color=colors[i % len(colors)],
                       lw=1.5, label=labels.get(key, key))
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Resistance prevalence")
        ax.set_title("Resistance trajectories", fontsize=8, fontweight="bold")
        ax.legend(frameon=False, fontsize=6)

        # Subplot 2: Control profiles
        ax = axes[0, 1]
        for i, (key, sol) in enumerate(solutions.items()):
            if "u_star" in sol:
                ax.plot(t_eval / 365, sol["u_star"],
                       color=colors[i % len(colors)], lw=1.5,
                       label=labels.get(key, key))
            elif "u" in sol:
                ax.plot(t_eval / 365, sol["u"],
                       color=colors[i % len(colors)], lw=1.5,
                       label=labels.get(key, key))
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Antibiotic use u(t)")
        ax.set_title("Control profiles", fontsize=8, fontweight="bold")

        # Subplot 3: Cost comparison
        ax = axes[1, 0]
        names = []
        costs = []
        for key, sol in solutions.items():
            if "cost" in sol:
                names.append(labels.get(key, key))
                costs.append(sol["cost"])
        
        bar_colors = [colors[i % len(colors)] for i in range(len(names))]
        bars = ax.bar(range(len(names)), costs, color=bar_colors, alpha=0.7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("Total cost J(u)")
        ax.set_title("Cost comparison", fontsize=8, fontweight="bold")

        # Subplot 4: Pareto frontier
        ax = axes[1, 1]
        # Resistance vs antibiotic use trade-off
        for i, (key, sol) in enumerate(solutions.items()):
            if "x" in sol:
                avg_r = sol["x"][:, 2].mean() / sol["x"].sum(axis=1).mean()
                avg_u = sol.get("u_star", sol.get("u", np.full(len(t_eval), 0))).mean()
                ax.scatter(avg_u, avg_r, c=colors[i % len(colors)],
                          s=60, edgecolors="white", linewidth=1,
                          label=labels.get(key, key), zorder=5)
        ax.set_xlabel("Average antibiotic use")
        ax.set_ylabel("Average resistance")
        ax.set_title("Antibiotic-resistance trade-off", fontsize=8, fontweight="bold")
        ax.legend(frameon=False, fontsize=6)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.close()


def generate_all_figures(
    output_dir: str,
    results: Dict = None,
):
    """Generate all publication figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = LancetFigureGenerator()

    # Figure 1: Model schematic
    gen.figure1_model_schematic(str(output_dir / "figure1_model_schematic.png"))

    if results:
        # Figure 2: Parameter estimation
        if "map" in results:
            gen.figure2_parameter_estimation(
                results["map"],
                str(output_dir / "figure2_parameter_estimation.png"),
            )

        # Figure 3: Counterfactual
        if "counterfactual" in results:
            gen.figure3_counterfactual(
                results["counterfactual"],
                str(output_dir / "figure3_counterfactual.png"),
            )

        # Figure 4: Cross-validation
        if "cv_results" in results:
            gen.figure4_cross_validation(
                results["cv_results"],
                str(output_dir / "figure4_cross_validation.png"),
            )

    print(f"Figures saved to {output_dir}")

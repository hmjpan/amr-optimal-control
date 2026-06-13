"""Tests for AMR optimal control core model."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from src.core.amr_model import (
    AMRParameters,
    OptimalControlConfig,
    AMROptimalControlModel,
    PontryaginSolver,
    compute_R0,
    invasion_reproduction_number,
)


class TestAMRParameters:
    def test_default_parameters(self):
        p = AMRParameters()
        assert p.N_total > 0
        assert p.beta_s > p.beta_r  # fitness cost
        assert p.gamma_s > 0
        assert p.phi > 0

    def test_recruitment_rate(self):
        p = AMRParameters(N_total=1e6, mu=1.0 / (70 * 365))
        assert abs(p.Lambda - p.mu * p.N_total) < 1e-10

    def test_custom_parameters(self):
        p = AMRParameters(
            beta_s=0.5, beta_r=0.3, gamma_s=0.1, gamma_r=0.05, phi=1e-5
        )
        assert p.beta_s == 0.5
        assert p.phi == 1e-5


class TestAMROptimalControlModel:
    @pytest.fixture
    def model(self):
        return AMROptimalControlModel(AMRParameters())

    def test_dynamics_sum_conservation(self, model):
        """Test that total population stays bounded (not strictly conserved due to μ)."""
        x = np.array([950000, 40000, 10000])
        N0 = x.sum()
        dx = model.dynamics(0, x, 0.0)
        # With μ, population should decline (removal > recruitment in infected)
        assert abs(dx.sum()) < 1e-5 * N0

    def test_no_resistance_without_source(self, model):
        """Without transmission or mutation, resistant should not appear."""
        x = np.array([1e6, 0, 0])
        dx = model.dynamics(0, x, 0.0)
        assert dx[2] == 0  # no Ir without Is

    def test_resistance_acquisition_monotonic(self, model):
        """α(u) should be monotonically increasing in u."""
        alphas = [model._resistance_acquisition_rate(u, model.p) for u in np.linspace(0, 1, 100)]
        for i in range(len(alphas) - 1):
            assert alphas[i] <= alphas[i + 1]

    def test_resistance_acquisition_bounds(self, model):
        """α(u) should be in [φ, φ + σ]."""
        for u in np.linspace(0, 10, 100):
            alpha = model._resistance_acquisition_rate(u, model.p)
            assert model.p.phi <= alpha <= model.p.phi + model.p.sigma + 1e-10

    def test_simulate_baseline(self, model):
        x0 = np.array([model.p.N_total * 0.95, model.p.N_total * 0.04, model.p.N_total * 0.01])
        result = model.simulate(x0, lambda t: 0.0, (0, 100), np.linspace(0, 100, 100))
        assert result["success"]
        assert result["x"].shape == (3, 100)

    def test_simulate_no_negative(self, model):
        """Simulation should not produce negative populations."""
        x0 = np.array([model.p.N_total * 0.95, model.p.N_total * 0.04, model.p.N_total * 0.01])
        result = model.simulate(x0, lambda t: 0.2, (0, 365 * 5), np.linspace(0, 365 * 5, 500))
        assert np.all(result["x"] >= -1e-10)


class TestPontryaginSolver:
    @pytest.fixture
    def solver(self):
        model = AMROptimalControlModel()
        return PontryaginSolver(model)

    def test_hamiltonian_nonnegative_cost(self, solver):
        x = np.array([0.7, 0.2, 0.1])
        u = 0.3
        lam = np.array([0.1, 0.2, 0.5])
        H = solver.hamiltonian(x, u, lam)
        assert H >= 0

    def test_optimal_control_bounds(self, solver):
        x = np.array([0.7, 0.2, 0.1])
        lam = np.array([-0.1, -0.2, 0.8])
        u_star = solver._optimal_control_from_conditions(x, lam)
        assert solver.p.u_min <= u_star <= solver.p.u_max

    def test_solve_forward_backward(self, solver):
        n = 200
        t_eval = np.linspace(0, 365, n)
        x0 = np.array([solver.p.N_total * 0.95, solver.p.N_total * 0.04, solver.p.N_total * 0.01])

        result = solver.solve_forward_backward(x0, t_eval, method="convergent")

        assert "x" in result
        assert "u_star" in result
        assert "lam" in result
        assert "cost" in result
        assert result["u_star"].shape == (n,)
        assert np.all(result["u_star"] >= solver.p.u_min - 1e-10)
        assert np.all(result["u_star"] <= solver.p.u_max + 1e-10)

    def test_solution_reduces_resistance(self, solver):
        """Optimal control should reduce terminal resistance vs constant high use."""
        n = 200
        t_eval = np.linspace(0, 365 * 5, n)
        x0 = np.array([solver.p.N_total * 0.95, solver.p.N_total * 0.04, solver.p.N_total * 0.01])

        result = solver.solve_forward_backward(x0, t_eval)

        # Constant high use trajectory
        const_u = np.full(n, 0.8)
        from src.core.amr_model import AMROptimalControlModel
        model = AMROptimalControlModel(solver.p)
        sim_const = model.simulate(
            x0,
            lambda t: 0.8,
            (0, 365 * 5),
            t_eval,
        )
        const_terminal_r = sim_const["x"][2, -1] / sim_const["x"].sum(axis=0)[-1]
        opt_terminal_r = result["x"][-1, 2] / result["x"][-1].sum()

        # Optimal should do better (or at least not worse)
        assert opt_terminal_r <= const_terminal_r + 0.01


class TestEpidemiologicalQuantities:
    def test_R0(self):
        p = AMRParameters(beta_s=0.25, beta_r=0.20, gamma_s=1 / 7, gamma_r=1 / 7, mu=1 / (70 * 365))
        R0_s, R0_r = compute_R0(p)
        assert R0_s > 1  # should be epidemic
        assert R0_s > R0_r  # fitness cost (lower beta gives lower R0 at equal gamma)

    def test_invasion_reproduction_number(self):
        p = AMRParameters()
        S_star = p.N_total * (1 / (2.5))  # S* at endemic equilibrium
        R_inv = invasion_reproduction_number(p, S_star, 0.3)
        assert R_inv > 0
        assert isinstance(R_inv, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

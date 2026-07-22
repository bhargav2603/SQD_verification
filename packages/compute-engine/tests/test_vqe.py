"""The variational optimisation loop.

Tested against closed-form objectives rather than molecules, so a failure points at the
optimiser wrapper rather than at the chemistry.
"""

import numpy as np
import pytest

from qdr_compute_engine.solvers.vqe import SUPPORTED_OPTIMIZERS, run_vqe
from qdr_core.errors import ValidationError


def quadratic(params: np.ndarray) -> float:
    """Minimum of -3.0 at (1, -2)."""
    return float((params[0] - 1.0) ** 2 + (params[1] + 2.0) ** 2 - 3.0)


@pytest.mark.parametrize("method", ["COBYLA", "L-BFGS-B", "Powell"])
def test_finds_the_minimum_of_a_quadratic(method: str) -> None:
    outcome = run_vqe(quadratic, np.array([0.0, 0.0]), method=method, tolerance=1e-10)
    assert outcome.energy == pytest.approx(-3.0, abs=1e-4)
    np.testing.assert_allclose(outcome.optimal_params, [1.0, -2.0], atol=1e-3)


def test_reports_the_starting_energy() -> None:
    outcome = run_vqe(quadratic, np.array([0.0, 0.0]))
    assert outcome.initial_energy == pytest.approx(quadratic(np.array([0.0, 0.0])))


def test_never_returns_worse_than_its_starting_point() -> None:
    """The variational principle means the search cannot legitimately regress; if the
    optimiser wanders, we keep the better point rather than reporting the worse one."""
    calls = {"n": 0}

    def deceptive(params: np.ndarray) -> float:
        # Genuinely worse everywhere except exactly at the start.
        calls["n"] += 1
        return 0.0 if np.allclose(params, 0.0) else 10.0

    outcome = run_vqe(deceptive, np.zeros(2), method="COBYLA", max_iterations=20)
    assert outcome.energy <= 0.0


def test_counts_energy_evaluations() -> None:
    outcome = run_vqe(quadratic, np.array([0.5, 0.5]), method="COBYLA")
    assert outcome.n_energy_evaluations > 0
    assert len(outcome.history) == outcome.n_energy_evaluations


def test_history_can_be_disabled() -> None:
    outcome = run_vqe(quadratic, np.array([0.5, 0.5]), track_history=False)
    assert outcome.history == []
    assert outcome.n_energy_evaluations > 0


def test_zero_parameter_ansatz_is_evaluated_once() -> None:
    """A fixed reference state has nothing to optimise, but still has an energy."""
    outcome = run_vqe(lambda p: 42.0, np.array([]), method="COBYLA")

    assert outcome.energy == 42.0
    assert outcome.converged
    assert outcome.n_iterations == 0
    assert outcome.n_energy_evaluations == 1
    assert "no free parameters" in outcome.message


def test_unsupported_optimizer_is_rejected() -> None:
    """Only optimisers we have actually exercised are claimed to work."""
    with pytest.raises(ValidationError, match="unsupported optimizer"):
        run_vqe(quadratic, np.zeros(2), method="definitely-not-an-optimizer")


def test_supported_optimizers_are_all_usable() -> None:
    for method in SUPPORTED_OPTIMIZERS:
        outcome = run_vqe(quadratic, np.array([0.9, -1.9]), method=method)
        assert np.isfinite(outcome.energy)


def test_is_deterministic_for_a_fixed_start() -> None:
    first = run_vqe(quadratic, np.array([0.3, 0.7]), method="COBYLA")
    second = run_vqe(quadratic, np.array([0.3, 0.7]), method="COBYLA")
    assert first.energy == second.energy
    np.testing.assert_array_equal(first.optimal_params, second.optimal_params)

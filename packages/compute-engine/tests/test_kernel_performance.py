"""Guard the performance work from silent regression.

The in-place kernel and the adjoint gradient are the difference between a molecule being
tractable and not. Both are easy to undo by accident -- routing ``simulate_statevector``
back through the reference path, or letting the driver fall back to a derivative-free
optimiser -- and neither mistake changes a single energy, so no correctness test would
notice.

These are deliberately loose. They assert *asymptotic behaviour and wiring*, not wall-clock
targets, because a shared CI runner's absolute timings are not reproducible. A regression
that reintroduces per-gate allocation or finite-difference gradients breaks them by an
order of magnitude; ordinary machine-to-machine variation does not.
"""

import time

import numpy as np
import pytest

from qdr_compute_engine.active_space.active_space_selector import select_active_space
from qdr_compute_engine.ansatze.uccsd import build_uccsd_ansatz
from qdr_compute_engine.drivers.statevector.adjoint import energy_and_gradient
from qdr_compute_engine.drivers.statevector.kernel import simulate as fast_simulate
from qdr_compute_engine.drivers.statevector.statevector_driver import (
    StatevectorDriver,
    hamiltonian_matrix,
    simulate_statevector,
    simulate_statevector_reference,
)
from qdr_compute_engine.hamiltonian.fermionic_builder import build_fermionic_hamiltonian
from qdr_compute_engine.mappers.jordan_wigner import map_to_qubits
from qdr_compute_engine.types import SolveRequest


def _best_of(fn, repeats: int = 3) -> float:
    """Fastest of several runs -- the least noisy estimator on a shared machine."""
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - started)
    return best


@pytest.fixture
def h2_problem(h2_integrals):
    active_space = select_active_space(h2_integrals)
    fermionic = build_fermionic_hamiltonian(h2_integrals, active_space)
    qubit_hamiltonian = map_to_qubits(fermionic)
    ansatz = build_uccsd_ansatz(
        n_electrons=active_space.n_active_electrons,
        n_spin_orbitals=active_space.n_spin_orbitals,
    )
    return active_space, qubit_hamiltonian, hamiltonian_matrix(qubit_hamiltonian), ansatz


def test_production_simulate_uses_the_fast_kernel(monkeypatch, h2_problem) -> None:
    """`simulate_statevector` must delegate, not fall back to the tensordot path.

    Checked by identity of the returned buffer's provenance rather than by timing: the
    fast path and the reference path agree numerically, so only wiring can be asserted
    cheaply here.
    """
    _, _, _, ansatz = h2_problem
    params = np.zeros(ansatz.n_params)

    from qdr_compute_engine.drivers.statevector import statevector_driver

    calls = {"n": 0}
    original = statevector_driver.kernel_simulate

    def counting(circuit, p=None):
        calls["n"] += 1
        return original(circuit, p)

    monkeypatch.setattr(statevector_driver, "kernel_simulate", counting)
    simulate_statevector(ansatz, params)

    assert calls["n"] == 1, "simulate_statevector no longer delegates to the kernel"


def test_kernel_is_not_slower_than_the_reference(h2_problem) -> None:
    """The whole point of the kernel. A regression makes it *slower*, not merely equal."""
    _, _, _, ansatz = h2_problem
    params = np.random.default_rng(0).normal(size=ansatz.n_params)

    reference = _best_of(lambda: simulate_statevector_reference(ansatz, params))
    fast = _best_of(lambda: fast_simulate(ansatz, params))

    # Generous: H2 is only 4 qubits, where the advantage is smallest. At 14 qubits the
    # measured ratio is ~4x; here we only require that it has not gone backwards.
    assert fast < reference * 1.5


def test_adjoint_cost_is_independent_of_parameter_count(h2_problem) -> None:
    """Adjoint returns every derivative for ~constant cost; finite differences do not.

    This is the property that makes larger active spaces affordable, and it is what breaks
    if someone swaps the analytic gradient for a numerical one.
    """
    _, _, sparse, ansatz = h2_problem
    params = np.random.default_rng(1).normal(size=ansatz.n_params)

    one_energy = _best_of(lambda: fast_simulate(ansatz, params))
    gradient = _best_of(lambda: energy_and_gradient(ansatz, params, sparse))

    # Theory says ~3 simulations. Allow generous headroom for a small circuit where
    # per-call overhead dominates, but far below the 2*n_params a finite difference costs.
    assert gradient < one_energy * 40


def test_driver_uses_analytic_gradients_by_default(h2_problem) -> None:
    """The default path must be L-BFGS-B with adjoint gradients."""
    active_space, qubit_hamiltonian, _, ansatz = h2_problem

    result = StatevectorDriver().solve(
        SolveRequest(
            hamiltonian=qubit_hamiltonian,
            active_space=active_space,
            ansatz=ansatz,
        )
    )

    assert result.metadata["used_analytic_gradient"] is True
    assert result.metadata["optimizer"] == "L-BFGS-B"


def test_gradient_path_needs_far_fewer_evaluations_than_cobyla(h2_problem) -> None:
    """Same physics, two searches: the gradient one must be markedly cheaper."""
    active_space, qubit_hamiltonian, _, ansatz = h2_problem

    def solve(options):
        return StatevectorDriver().solve(
            SolveRequest(
                hamiltonian=qubit_hamiltonian,
                active_space=active_space,
                ansatz=ansatz,
                options=options,
            )
        )

    gradient_result = solve({})
    cobyla_result = solve({"optimizer": "COBYLA"})

    assert gradient_result.n_energy_evaluations < cobyla_result.n_energy_evaluations

    # And both must still land on the same answer -- a faster wrong answer is not a win.
    assert gradient_result.energy == pytest.approx(cobyla_result.energy, abs=1e-6)

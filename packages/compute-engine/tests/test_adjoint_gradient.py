"""The analytic gradient must agree with finite differences.

An analytic gradient that is silently wrong is worse than no gradient at all: the
optimiser still converges, just to the wrong place, and the resulting energy is a
plausible number with no symptom. So the adjoint implementation is checked against a
central-difference oracle on the real UCCSD ansatz, and separately on synthetic circuits
that exercise RX and RY generators the H2 ansatz never produces.
"""

import numpy as np
import pytest

from qdr_compute_engine.active_space.active_space_selector import select_active_space
from qdr_compute_engine.ansatze.uccsd import build_uccsd_ansatz
from qdr_compute_engine.drivers.statevector.adjoint import (
    energy_and_gradient,
    finite_difference_gradient,
)
from qdr_compute_engine.drivers.statevector.statevector_driver import (
    exact_ground_state_energy,
    hamiltonian_matrix,
)
from qdr_compute_engine.hamiltonian.fermionic_builder import build_fermionic_hamiltonian
from qdr_compute_engine.mappers.jordan_wigner import map_to_qubits
from qdr_compute_engine.types import Circuit, Gate, GateName, QubitHamiltonian


@pytest.fixture
def h2_problem(h2_integrals):
    """The H2 qubit Hamiltonian, its sparse matrix, and the UCCSD ansatz."""
    active_space = select_active_space(h2_integrals)
    fermionic = build_fermionic_hamiltonian(h2_integrals, active_space)
    qubit_hamiltonian = map_to_qubits(fermionic)
    ansatz = build_uccsd_ansatz(
        n_electrons=active_space.n_active_electrons,
        n_spin_orbitals=active_space.n_spin_orbitals,
    )
    return qubit_hamiltonian, hamiltonian_matrix(qubit_hamiltonian), ansatz


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_gradient_matches_finite_difference_h2(h2_problem, seed: int) -> None:
    _, sparse, ansatz = h2_problem
    params = np.random.default_rng(seed).normal(scale=0.8, size=ansatz.n_params)

    _, analytic = energy_and_gradient(ansatz, params, sparse)
    numerical = finite_difference_gradient(ansatz, params, sparse)

    np.testing.assert_allclose(analytic, numerical, atol=1e-7, rtol=1e-5)


def test_gradient_at_zero_is_not_trivially_zero(h2_problem) -> None:
    """At theta=0 the ansatz is the identity but the gradient must still point downhill.

    A gradient that is identically zero here would look like convergence at Hartree-Fock
    and quietly stop the optimiser at the mean-field answer.
    """
    _, sparse, ansatz = h2_problem
    _, gradient = energy_and_gradient(ansatz, np.zeros(ansatz.n_params), sparse)
    assert np.linalg.norm(gradient) > 1e-6


def test_energy_returned_matches_direct_evaluation(h2_problem) -> None:
    from qdr_compute_engine.drivers.statevector.kernel import simulate

    _, sparse, ansatz = h2_problem
    params = np.random.default_rng(42).normal(scale=0.5, size=ansatz.n_params)

    energy, _ = energy_and_gradient(ansatz, params, sparse)
    state = simulate(ansatz, params).reshape(-1)
    direct = float(np.vdot(state, sparse @ state).real)

    assert abs(energy - direct) < 1e-13


def _rotation_circuit(name: GateName, n_qubits: int = 3) -> Circuit:
    """A circuit whose parameterised gates use ``name`` as their generator."""
    gates = (
        Gate(GateName.H, (0,)),
        Gate(GateName.CX, (0, 1)),
        Gate(name, (0,), param_index=0, param_scale=1.0),
        Gate(name, (1,), param_index=0, param_scale=-0.5),  # shared parameter
        Gate(GateName.CX, (1, 2)),
        Gate(name, (2,), param_index=1, param_scale=2.0),
        Gate(GateName.H, (2,)),
    )
    return Circuit(n_qubits=n_qubits, gates=gates, n_params=2)


@pytest.mark.parametrize("name", [GateName.RX, GateName.RY, GateName.RZ])
def test_gradient_for_each_generator(name: GateName) -> None:
    """RX and RY never appear as free parameters in UCCSD, so cover them explicitly."""
    from openfermion import QubitOperator

    circuit = _rotation_circuit(name)
    operator = (
        QubitOperator("Z0", 0.5)
        + QubitOperator("X1", 0.25)
        + QubitOperator("Z0 Z1", -0.75)
        + QubitOperator("Y2", 0.3)
    )
    hamiltonian = QubitHamiltonian(operator=operator, n_qubits=3, mapping="jordan_wigner")
    sparse = hamiltonian_matrix(hamiltonian)

    params = np.array([0.37, -1.12])
    _, analytic = energy_and_gradient(circuit, params, sparse)
    numerical = finite_difference_gradient(circuit, params, sparse)

    np.testing.assert_allclose(analytic, numerical, atol=1e-7, rtol=1e-5)


def test_shared_parameter_accumulates(h2_problem) -> None:
    """UCCSD fans one amplitude out to many rotations; the chain rule must sum them.

    Verified by construction: the H2 ansatz has more parameterised gates than parameters,
    so a gradient that overwrote instead of accumulating would disagree with finite
    differences -- which the parametrised test above already covers. This asserts the
    precondition that makes that test meaningful.
    """
    _, _, ansatz = h2_problem
    n_parameterised_gates = sum(1 for g in ansatz.gates if g.is_parameterized)
    assert n_parameterised_gates > ansatz.n_params


def test_gradient_vanishes_at_the_variational_minimum(h2_problem) -> None:
    """At the optimum the gradient must be zero -- the definition of a stationary point."""
    from scipy.optimize import minimize

    qubit_hamiltonian, sparse, ansatz = h2_problem

    result = minimize(
        lambda x: energy_and_gradient(ansatz, x, sparse),
        np.zeros(ansatz.n_params),
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1e-14, "gtol": 1e-12},
    )

    _, gradient = energy_and_gradient(ansatz, result.x, sparse)
    assert np.linalg.norm(gradient) < 1e-6

    # And it must have found the exact ground state, not merely a stationary point.
    exact = exact_ground_state_energy(qubit_hamiltonian)
    assert abs(float(result.fun) - exact) < 1e-9

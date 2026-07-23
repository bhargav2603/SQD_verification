"""Matrix-free qubit-Hamiltonian application -- Rung 3.

The whole rung is one claim: applying the Hamiltonian *without* materialising the
``2^n x 2^n`` sparse matrix produces the same result as the sparse path. Every test here
is a form of that claim, checked against OpenFermion's ``get_sparse_operator`` -- which is
the operator the statevector driver already takes expectation values against, so agreeing
with it is agreeing with the engine's verified convention (endianness included).

None of this needs PySCF: a ``QubitOperator`` is built directly, or the H2 qubit
Hamiltonian is reconstructed from the golden integrals through the pure mapper path.
"""

import numpy as np
import pytest
from openfermion import QubitOperator, get_sparse_operator, hermitian_conjugated

from qdr_compute_engine.active_space.active_space_selector import select_active_space
from qdr_compute_engine.drivers.statevector.statevector_driver import (
    exact_ground_state_energy,
    exact_ground_state_energy_at_particle_number,
)
from qdr_compute_engine.hamiltonian.fermionic_builder import build_fermionic_hamiltonian
from qdr_compute_engine.hamiltonian.matrix_free import (
    apply_qubit_operator,
    ground_state_energy_at_particle_number_matrix_free,
    ground_state_energy_matrix_free,
    qubit_operator_as_linear_operator,
)
from qdr_compute_engine.mappers.jordan_wigner import map_to_qubits
from qdr_compute_engine.types import IntegralSet, QubitHamiltonian


def _random_state(n_qubits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim = 2**n_qubits
    psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return psi / np.linalg.norm(psi)


def _h2_qubit_hamiltonian(integrals: IntegralSet) -> QubitHamiltonian:
    active_space = select_active_space(integrals)
    return map_to_qubits(build_fermionic_hamiltonian(integrals, active_space))


# --------------------------------------------------------------------------------------
# Single Pauli strings: the endianness/phase convention must match get_sparse_operator
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
@pytest.mark.parametrize("qubit", [0, 1, 2])
def test_single_pauli_matches_sparse(pauli: str, qubit: int) -> None:
    n_qubits = 3
    operator = QubitOperator(f"{pauli}{qubit}")
    psi = _random_state(n_qubits, seed=qubit)

    expected = get_sparse_operator(operator, n_qubits=n_qubits) @ psi
    got = apply_qubit_operator(operator, psi, n_qubits)

    np.testing.assert_allclose(got, expected, atol=1e-13)


def test_identity_term_is_a_scalar_multiple() -> None:
    operator = QubitOperator("", 2.5)  # the constant term
    psi = _random_state(2, seed=7)
    np.testing.assert_allclose(
        apply_qubit_operator(operator, psi, 2), 2.5 * psi, atol=1e-13
    )


# --------------------------------------------------------------------------------------
# Full Hermitian operators: the sum-of-Pauli-strings path vs the assembled matrix
# --------------------------------------------------------------------------------------


def test_random_hermitian_operator_matches_sparse() -> None:
    n_qubits = 4
    rng = np.random.default_rng(0)
    paulis = ["X", "Y", "Z"]
    operator = QubitOperator("", rng.standard_normal())
    for _ in range(25):
        string = " ".join(
            f"{rng.choice(paulis)}{q}"
            for q in sorted(rng.choice(n_qubits, size=2, replace=False))
        )
        operator += QubitOperator(string, rng.standard_normal())
    # Make it Hermitian: H + H^dagger has real spectrum, which is what a Hamiltonian is.
    operator += hermitian_conjugated(operator)

    psi = _random_state(n_qubits, seed=3)
    expected = get_sparse_operator(operator, n_qubits=n_qubits) @ psi
    got = apply_qubit_operator(operator, psi, n_qubits)

    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_apply_matches_sparse_on_h2(h2_integrals: IntegralSet) -> None:
    """The real molecular operator, not a synthetic one: matrix-free H @ psi == sparse."""
    hamiltonian = _h2_qubit_hamiltonian(h2_integrals)
    psi = _random_state(hamiltonian.n_qubits, seed=11)

    expected = (
        get_sparse_operator(hamiltonian.operator, n_qubits=hamiltonian.n_qubits) @ psi
    )
    got = apply_qubit_operator(hamiltonian.operator, psi, hamiltonian.n_qubits)

    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_linear_operator_reproduces_sparse_matvec(h2_integrals: IntegralSet) -> None:
    hamiltonian = _h2_qubit_hamiltonian(h2_integrals)
    linop = qubit_operator_as_linear_operator(hamiltonian)
    sparse = get_sparse_operator(hamiltonian.operator, n_qubits=hamiltonian.n_qubits)

    psi = _random_state(hamiltonian.n_qubits, seed=5)
    np.testing.assert_allclose(linop @ psi, sparse @ psi, atol=1e-12)


# --------------------------------------------------------------------------------------
# Ground-state energy: the reference the whole rung exists to keep computable above 14q
# --------------------------------------------------------------------------------------


def test_ground_state_energy_matches_sparse_path(h2_integrals: IntegralSet) -> None:
    """THE RUNG-3 GATE: matrix-free lowest eigenvalue == the sparse driver's answer."""
    hamiltonian = _h2_qubit_hamiltonian(h2_integrals)

    sparse_energy = exact_ground_state_energy(hamiltonian)
    matrix_free_energy = ground_state_energy_matrix_free(hamiltonian)

    assert matrix_free_energy == pytest.approx(sparse_energy, abs=1e-9)


def test_particle_number_restricted_matches_sparse_path(
    h2_integrals: IntegralSet,
) -> None:
    hamiltonian = _h2_qubit_hamiltonian(h2_integrals)
    n_electrons = 2

    sparse_energy = exact_ground_state_energy_at_particle_number(
        hamiltonian, n_electrons
    )
    matrix_free_energy = ground_state_energy_at_particle_number_matrix_free(
        hamiltonian, n_electrons
    )

    assert matrix_free_energy == pytest.approx(sparse_energy, abs=1e-9)


def test_random_hermitian_ground_state_matches_dense() -> None:
    """On a synthetic operator small enough for a dense check, the eigenvalue agrees."""
    n_qubits = 4
    rng = np.random.default_rng(1)
    operator = QubitOperator("", rng.standard_normal())
    for _ in range(30):
        string = " ".join(
            f"{rng.choice(['X', 'Y', 'Z'])}{q}"
            for q in sorted(rng.choice(n_qubits, size=2, replace=False))
        )
        operator += QubitOperator(string, rng.standard_normal())
    operator += hermitian_conjugated(operator)

    hamiltonian = QubitHamiltonian(
        operator=operator, n_qubits=n_qubits, mapping="jordan_wigner"
    )
    dense = get_sparse_operator(operator, n_qubits=n_qubits).toarray()
    dense_ground = float(np.linalg.eigvalsh(dense)[0])

    assert ground_state_energy_matrix_free(hamiltonian) == pytest.approx(
        dense_ground, abs=1e-9
    )

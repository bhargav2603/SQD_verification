"""Jordan-Wigner mapping, and the qubit-ordering convention it depends on.

The endianness tests here are the most valuable in the suite. Our simulator and
OpenFermion's ``get_sparse_operator`` must agree on which bit is which qubit; if they do
not, every expectation value is taken against a permuted operator and the VQE converges
smoothly to a wrong answer with no symptom. These tests pin the convention against
hand-constructed states whose energies are known by inspection.
"""

import numpy as np
import pytest
from openfermion import FermionOperator, QubitOperator, get_sparse_operator
from openfermion import jordan_wigner as openfermion_jw

from qdr_compute_engine.drivers.statevector.statevector_driver import (
    expectation_value,
    simulate_statevector,
)
from qdr_compute_engine.hamiltonian.fermionic_builder import (
    build_fermionic_hamiltonian,
    hartree_fock_occupations,
)
from qdr_compute_engine.mappers import IMPLEMENTED_MAPPERS, MAPPERS, get_mapper
from qdr_compute_engine.mappers.jordan_wigner import (
    annihilation_operator,
    creation_operator,
    map_to_qubits,
)
from qdr_compute_engine.types import Circuit, FermionicHamiltonian, Gate, GateName, IntegralSet
from qdr_core.errors import QubitCapExceeded, ValidationError
from qdr_core.limits import MAX_QUBITS

# --------------------------------------------------------------------------------------
# Qubit ordering -- the convention everything else depends on
# --------------------------------------------------------------------------------------


def _basis_state_circuit(n_qubits: int, occupied: tuple[int, ...]) -> Circuit:
    return Circuit(
        n_qubits=n_qubits,
        gates=tuple(Gate(GateName.X, (q,)) for q in occupied),
        n_params=0,
    )


@pytest.mark.parametrize("qubit", [0, 1, 2, 3])
def test_single_qubit_z_expectation_identifies_the_qubit(qubit: int) -> None:
    """<Z_q> = -1 exactly on the flipped qubit and +1 on every other one.

    This is what pins our axis-to-qubit mapping to OpenFermion's bit ordering.
    """
    circuit = _basis_state_circuit(4, (qubit,))
    state = simulate_statevector(circuit)

    for probe in range(4):
        operator = get_sparse_operator(QubitOperator(f"Z{probe}"), n_qubits=4)
        expected = -1.0 if probe == qubit else 1.0
        assert expectation_value(operator, state) == pytest.approx(expected, abs=1e-12)


def test_statevector_amplitude_index_is_big_endian() -> None:
    """Qubit 0 occupies the most significant bit of the amplitude index."""
    state = simulate_statevector(_basis_state_circuit(3, (0,)))
    # Qubit 0 set, others clear -> binary 100 -> index 4.
    assert np.argmax(np.abs(state)) == 4

    state = simulate_statevector(_basis_state_circuit(3, (2,)))
    # Qubit 2 set -> binary 001 -> index 1.
    assert np.argmax(np.abs(state)) == 1


def test_number_operator_counts_electrons() -> None:
    """The JW-mapped number operator must return the electron count of a basis state."""
    n_qubits = 4
    occupied = hartree_fock_occupations(2, n_qubits)
    state = simulate_statevector(_basis_state_circuit(n_qubits, occupied))

    number_operator = sum(
        (openfermion_jw(FermionOperator(f"{q}^ {q}")) for q in range(n_qubits)),
        QubitOperator(),
    )
    matrix = get_sparse_operator(number_operator, n_qubits=n_qubits)
    assert expectation_value(matrix, state) == pytest.approx(2.0, abs=1e-12)


# --------------------------------------------------------------------------------------
# Jordan-Wigner algebra
# --------------------------------------------------------------------------------------


def test_map_to_qubits_matches_openfermion(h2_integrals: IntegralSet) -> None:
    fermionic = build_fermionic_hamiltonian(h2_integrals)
    ours = map_to_qubits(fermionic)
    theirs = openfermion_jw(fermionic.operator)
    theirs.compress()

    difference = ours.operator - theirs
    difference.compress()
    assert difference.terms == {}


def test_mapped_hamiltonian_is_hermitian(h2_integrals: IntegralSet) -> None:
    qubit_hamiltonian = map_to_qubits(build_fermionic_hamiltonian(h2_integrals))
    matrix = get_sparse_operator(
        qubit_hamiltonian.operator, n_qubits=qubit_hamiltonian.n_qubits
    ).toarray()
    np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-12)


@pytest.mark.parametrize("index", [0, 1, 3])
def test_creation_operator_matches_openfermion(index: int) -> None:
    difference = creation_operator(index) - openfermion_jw(FermionOperator(f"{index}^"))
    difference.compress()
    assert difference.terms == {}


@pytest.mark.parametrize("index", [0, 2])
def test_annihilation_operator_matches_openfermion(index: int) -> None:
    difference = annihilation_operator(index) - openfermion_jw(FermionOperator(f"{index}"))
    difference.compress()
    assert difference.terms == {}


def test_canonical_anticommutation_relation() -> None:
    """{a_i, a_j+} = delta_ij, the defining property of fermionic operators."""
    n_qubits = 4
    for i in range(3):
        for j in range(3):
            anticommutator = annihilation_operator(i) * creation_operator(j) + creation_operator(
                j
            ) * annihilation_operator(i)
            matrix = get_sparse_operator(anticommutator, n_qubits=n_qubits).toarray()
            expected = np.eye(2**n_qubits) if i == j else np.zeros((2**n_qubits,) * 2)
            np.testing.assert_allclose(matrix, expected, atol=1e-12)


def test_creation_operators_anticommute() -> None:
    """{a_i+, a_j+} = 0, which is what the Z string exists to enforce."""
    for i, j in [(0, 1), (0, 2), (1, 3)]:
        anticommutator = creation_operator(i) * creation_operator(j) + creation_operator(
            j
        ) * creation_operator(i)
        anticommutator.compress()
        assert anticommutator.terms == {}


def test_qubit_count_is_twice_the_orbital_count(h2_integrals: IntegralSet) -> None:
    qubit_hamiltonian = map_to_qubits(build_fermionic_hamiltonian(h2_integrals))
    assert qubit_hamiltonian.n_qubits == 2 * h2_integrals.n_orbitals
    assert qubit_hamiltonian.mapping == "jordan_wigner"


def test_mapping_rejects_oversized_systems() -> None:
    """The cap is enforced in the mapper too, before a huge operator is built."""
    oversized = FermionicHamiltonian(
        operator=FermionOperator(),
        n_spin_orbitals=MAX_QUBITS + 2,
    )
    with pytest.raises(QubitCapExceeded) as exc_info:
        map_to_qubits(oversized)
    assert exc_info.value.required_qubits == MAX_QUBITS + 2


# --------------------------------------------------------------------------------------
# Mapper registry
# --------------------------------------------------------------------------------------


def test_registry_exposes_all_three_mappings() -> None:
    assert set(MAPPERS) == {"jordan_wigner", "parity", "bravyi_kitaev"}


def test_only_jordan_wigner_is_advertised_as_implemented() -> None:
    assert frozenset({"jordan_wigner"}) == IMPLEMENTED_MAPPERS


@pytest.mark.parametrize("name", ["parity", "bravyi_kitaev"])
def test_unimplemented_mappings_fail_loudly(name: str, h2_integrals: IntegralSet) -> None:
    """Registered but unbuilt mappings must raise, never return a plausible wrong answer."""
    mapper = get_mapper(name)
    with pytest.raises(NotImplementedError):
        mapper(build_fermionic_hamiltonian(h2_integrals))


def test_unknown_mapping_is_a_validation_error() -> None:
    with pytest.raises(ValidationError):
        get_mapper("not_a_mapping")

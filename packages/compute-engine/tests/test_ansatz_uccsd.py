"""UCCSD construction and its compilation into the gate IR.

Two properties matter most and are easy to get subtly wrong:

* **Particle-number conservation.** UCCSD's excitations are number-conserving by
  construction, so the state must stay in the 2-electron sector for *any* parameters.
  A compilation bug that mis-orders a CNOT ladder breaks this while still producing a
  normalised state and a believable energy.
* **theta = 0 is the identity.** The reference state must survive an all-zero parameter
  vector untouched, which is what makes Hartree-Fock the guaranteed starting point.
"""

import numpy as np
import pytest
from openfermion import FermionOperator, QubitOperator, get_sparse_operator
from openfermion import jordan_wigner as openfermion_jw

from qdr_compute_engine.ansatze import IMPLEMENTED_ANSATZE, get_ansatz_builder
from qdr_compute_engine.ansatze.uccsd import (
    build_uccsd_ansatz,
    double_excitations,
    initial_parameters,
    reference_state_gates,
    single_excitations,
)
from qdr_compute_engine.drivers.statevector.statevector_driver import (
    expectation_value,
    simulate_statevector,
)
from qdr_compute_engine.types import Circuit, GateName
from qdr_core.errors import ValidationError


def _number_operator_matrix(n_qubits: int):
    operator = sum(
        (openfermion_jw(FermionOperator(f"{q}^ {q}")) for q in range(n_qubits)),
        QubitOperator(),
    )
    return get_sparse_operator(operator, n_qubits=n_qubits)


# --------------------------------------------------------------------------------------
# Excitation enumeration
# --------------------------------------------------------------------------------------


def test_h2_has_two_singles_and_one_double() -> None:
    """2 electrons in 4 spin orbitals: alpha->alpha, beta->beta, and one double."""
    assert single_excitations(2, 4) == [(0, 2), (1, 3)]
    assert double_excitations(2, 4) == [(0, 1, 2, 3)]


def test_excitations_conserve_spin() -> None:
    """Spin orbital 2p is alpha and 2p+1 is beta, so parity must match."""
    for i, a in single_excitations(4, 8):
        assert i % 2 == a % 2
    for i, j, a, b in double_excitations(4, 8):
        assert (i % 2) + (j % 2) == (a % 2) + (b % 2)


def test_parameter_count_matches_excitation_count() -> None:
    circuit = build_uccsd_ansatz(n_electrons=4, n_spin_orbitals=8)
    expected = len(single_excitations(4, 8)) + len(double_excitations(4, 8))
    assert circuit.n_params == expected


def test_singles_and_doubles_can_be_disabled() -> None:
    doubles_only = build_uccsd_ansatz(2, 4, include_singles=False)
    assert doubles_only.n_params == len(double_excitations(2, 4))

    singles_only = build_uccsd_ansatz(2, 4, include_doubles=False)
    assert singles_only.n_params == len(single_excitations(2, 4))


# --------------------------------------------------------------------------------------
# Reference state
# --------------------------------------------------------------------------------------


def test_reference_state_is_a_row_of_x_gates() -> None:
    gates = reference_state_gates((0, 1))
    assert [g.name for g in gates] == [GateName.X, GateName.X]
    assert [g.qubits for g in gates] == [(0,), (1,)]


def test_zero_parameters_leave_the_reference_state_untouched() -> None:
    """theta = 0 must be exactly the Hartree-Fock determinant, not approximately."""
    circuit = build_uccsd_ansatz(n_electrons=2, n_spin_orbitals=4)
    state = simulate_statevector(circuit, initial_parameters(circuit))

    # Qubits 0 and 1 occupied -> binary 1100 -> index 12.
    expected = np.zeros(16, dtype=complex)
    expected[12] = 1.0
    np.testing.assert_allclose(state, expected, atol=1e-12)


# --------------------------------------------------------------------------------------
# Physical invariants of the compiled circuit
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_particle_number_is_conserved_for_arbitrary_parameters(seed: int) -> None:
    """The compiled circuit must never leak out of the 2-electron sector."""
    circuit = build_uccsd_ansatz(n_electrons=2, n_spin_orbitals=4)
    rng = np.random.default_rng(seed)
    params = rng.normal(0.0, 1.0, size=circuit.n_params)

    state = simulate_statevector(circuit, params)
    n_electrons = expectation_value(_number_operator_matrix(4), state)
    assert n_electrons == pytest.approx(2.0, abs=1e-10)


@pytest.mark.parametrize("seed", [0, 5])
def test_state_stays_normalised(seed: int) -> None:
    """Non-unitary compilation would show up here first."""
    circuit = build_uccsd_ansatz(n_electrons=2, n_spin_orbitals=4)
    rng = np.random.default_rng(seed)
    state = simulate_statevector(circuit, rng.normal(size=circuit.n_params))
    assert float(np.vdot(state, state).real) == pytest.approx(1.0, abs=1e-12)


def test_larger_system_conserves_particle_number() -> None:
    """4 electrons in 8 spin orbitals exercises multi-parameter Trotterisation."""
    circuit = build_uccsd_ansatz(n_electrons=4, n_spin_orbitals=8)
    rng = np.random.default_rng(7)
    state = simulate_statevector(circuit, rng.normal(0.0, 0.3, size=circuit.n_params))
    assert expectation_value(_number_operator_matrix(8), state) == pytest.approx(4.0, abs=1e-9)


# --------------------------------------------------------------------------------------
# IR well-formedness
# --------------------------------------------------------------------------------------


def test_every_declared_parameter_is_actually_used() -> None:
    """An unused parameter is a dead optimiser dimension and a sign of a dropped term."""
    circuit = build_uccsd_ansatz(n_electrons=2, n_spin_orbitals=4)
    assert circuit.bound_parameter_indices() == frozenset(range(circuit.n_params))


def test_circuit_reports_two_qubit_gate_cost() -> None:
    circuit = build_uccsd_ansatz(n_electrons=2, n_spin_orbitals=4)
    assert circuit.n_two_qubit_gates > 0
    assert circuit.n_two_qubit_gates < circuit.n_gates


def test_metadata_records_the_construction() -> None:
    circuit = build_uccsd_ansatz(n_electrons=2, n_spin_orbitals=4)
    assert circuit.metadata["ansatz"] == "uccsd"
    assert circuit.metadata["trotter_order"] == 1
    assert circuit.metadata["n_doubles"] == 1


def test_gates_stay_within_the_register() -> None:
    circuit = build_uccsd_ansatz(n_electrons=4, n_spin_orbitals=8)
    for gate in circuit.gates:
        assert all(0 <= q < circuit.n_qubits for q in gate.qubits)


def test_initial_parameters_default_to_zero() -> None:
    circuit = build_uccsd_ansatz(n_electrons=2, n_spin_orbitals=4)
    np.testing.assert_array_equal(initial_parameters(circuit), np.zeros(circuit.n_params))


def test_initial_parameters_are_reproducible_under_a_seed() -> None:
    circuit = build_uccsd_ansatz(n_electrons=2, n_spin_orbitals=4)
    first = initial_parameters(circuit, seed=42, spread=0.1)
    second = initial_parameters(circuit, seed=42, spread=0.1)
    np.testing.assert_array_equal(first, second)
    assert not np.allclose(first, np.zeros_like(first))


# --------------------------------------------------------------------------------------
# Validation and the ansatz registry
# --------------------------------------------------------------------------------------


def test_more_electrons_than_orbitals_is_rejected() -> None:
    with pytest.raises(ValidationError, match="cannot place"):
        build_uccsd_ansatz(n_electrons=6, n_spin_orbitals=4)


def test_a_fully_occupied_space_has_no_excitations() -> None:
    with pytest.raises(ValidationError, match="no excitations"):
        build_uccsd_ansatz(n_electrons=4, n_spin_orbitals=4)


def test_only_uccsd_is_advertised_as_implemented() -> None:
    assert frozenset({"uccsd"}) == IMPLEMENTED_ANSATZE


@pytest.mark.parametrize("name", ["hardware_efficient", "adapt"])
def test_unimplemented_ansatze_fail_loudly(name: str) -> None:
    builder = get_ansatz_builder(name)
    with pytest.raises(NotImplementedError):
        builder(n_electrons=2, n_spin_orbitals=4)


def test_unknown_ansatz_is_a_validation_error() -> None:
    with pytest.raises(ValidationError):
        get_ansatz_builder("not_an_ansatz")


def test_uccsd_builds_a_circuit_type() -> None:
    assert isinstance(build_uccsd_ansatz(2, 4), Circuit)

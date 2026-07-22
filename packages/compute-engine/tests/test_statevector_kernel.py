"""The fast kernel must be indistinguishable from the reference implementation.

``kernel.py`` exists purely for speed. Speed is worthless if it changes an answer, and a
statevector that is subtly wrong produces an energy that still looks like a plausible
eigenvalue -- the exact failure mode this engine is built to refuse. So every gate type is
compared against ``statevector_driver``'s original tensordot path at machine precision,
and the whole H2 ansatz is compared end to end.
"""

import numpy as np
import pytest

from qdr_compute_engine.active_space.active_space_selector import select_active_space
from qdr_compute_engine.ansatze.uccsd import build_uccsd_ansatz
from qdr_compute_engine.drivers.statevector import kernel
from qdr_compute_engine.drivers.statevector.statevector_driver import (
    simulate_statevector_reference as simulate_statevector,
)
from qdr_compute_engine.hamiltonian.fermionic_builder import build_fermionic_hamiltonian
from qdr_compute_engine.mappers.jordan_wigner import map_to_qubits
from qdr_compute_engine.types import Circuit, Gate, GateName

ALL_ONE_QUBIT = [
    GateName.H,
    GateName.X,
    GateName.Y,
    GateName.Z,
    GateName.S,
    GateName.SDG,
]
ROTATIONS = [GateName.RX, GateName.RY, GateName.RZ]


def _random_circuit(n_qubits: int, seed: int, n_params: int = 4) -> Circuit:
    """A circuit touching every gate type the kernel implements."""
    rng = np.random.default_rng(seed)
    gates: list[Gate] = []
    for step in range(40):
        choice = rng.integers(0, 4)
        if choice == 0:
            name = ALL_ONE_QUBIT[int(rng.integers(0, len(ALL_ONE_QUBIT)))]
            gates.append(Gate(name, (int(rng.integers(0, n_qubits)),)))
        elif choice == 1:
            name = ROTATIONS[int(rng.integers(0, len(ROTATIONS)))]
            gates.append(Gate(name, (int(rng.integers(0, n_qubits)),), angle=float(rng.normal())))
        elif choice == 2:
            name = ROTATIONS[int(rng.integers(0, len(ROTATIONS)))]
            gates.append(
                Gate(
                    name,
                    (int(rng.integers(0, n_qubits)),),
                    param_index=int(step % n_params),
                    param_scale=float(rng.normal()),
                )
            )
        else:
            a, b = rng.choice(n_qubits, size=2, replace=False)
            name = GateName.CX if rng.integers(0, 2) == 0 else GateName.CZ
            gates.append(Gate(name, (int(a), int(b))))
    return Circuit(n_qubits=n_qubits, gates=tuple(gates), n_params=n_params)


@pytest.mark.parametrize("name", ALL_ONE_QUBIT)
@pytest.mark.parametrize("qubit", [0, 1, 2])
def test_one_qubit_gates_match_reference(name: GateName, qubit: int) -> None:
    circuit = Circuit(n_qubits=3, gates=(Gate(GateName.H, (0,)), Gate(name, (qubit,))))
    expected = simulate_statevector(circuit)
    actual = kernel.simulate(circuit).reshape(-1)
    np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)


@pytest.mark.parametrize("name", ROTATIONS)
@pytest.mark.parametrize("qubit", [0, 1, 2])
def test_rotation_gates_match_reference(name: GateName, qubit: int) -> None:
    circuit = Circuit(
        n_qubits=3,
        gates=(Gate(GateName.H, (0,)), Gate(name, (qubit,), angle=0.7321)),
    )
    expected = simulate_statevector(circuit)
    actual = kernel.simulate(circuit).reshape(-1)
    np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)


@pytest.mark.parametrize("name", [GateName.CX, GateName.CZ])
@pytest.mark.parametrize("pair", [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2)])
def test_two_qubit_gates_match_reference(name: GateName, pair: tuple[int, int]) -> None:
    # H on both qubits first, so every amplitude is non-zero and a wrong permutation
    # cannot hide behind a zero.
    circuit = Circuit(
        n_qubits=3,
        gates=(
            Gate(GateName.H, (0,)),
            Gate(GateName.H, (1,)),
            Gate(GateName.H, (2,)),
            Gate(GateName.S, (pair[0],)),
            Gate(name, pair),
        ),
    )
    expected = simulate_statevector(circuit)
    actual = kernel.simulate(circuit).reshape(-1)
    np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_random_circuits_match_reference(seed: int) -> None:
    circuit = _random_circuit(n_qubits=4, seed=seed)
    params = np.random.default_rng(seed + 100).normal(size=circuit.n_params)
    expected = simulate_statevector(circuit, params)
    actual = kernel.simulate(circuit, params).reshape(-1)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)


def test_uccsd_h2_matches_reference(h2_integrals) -> None:
    """The real ansatz, not a synthetic circuit."""
    active_space = select_active_space(h2_integrals)
    ansatz = build_uccsd_ansatz(
        n_electrons=active_space.n_active_electrons,
        n_spin_orbitals=active_space.n_spin_orbitals,
    )
    params = np.random.default_rng(7).normal(size=ansatz.n_params)
    expected = simulate_statevector(ansatz, params)
    actual = kernel.simulate(ansatz, params).reshape(-1)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)


def test_inverse_undoes_forward() -> None:
    """The adjoint gradient reconstructs states by undoing gates, so this is load bearing."""
    circuit = _random_circuit(n_qubits=4, seed=11)
    params = np.random.default_rng(11).normal(size=circuit.n_params)

    state = kernel.simulate(circuit, params)
    for gate in reversed(circuit.gates):
        kernel.apply_gate_inverse(state, gate, params)

    np.testing.assert_allclose(
        state.reshape(-1), kernel.zero_state(4).reshape(-1), atol=1e-13, rtol=0
    )


def test_kernel_preserves_norm(h2_integrals) -> None:
    active_space = select_active_space(h2_integrals)
    ansatz = build_uccsd_ansatz(
        n_electrons=active_space.n_active_electrons,
        n_spin_orbitals=active_space.n_spin_orbitals,
    )
    params = np.random.default_rng(3).normal(size=ansatz.n_params)
    state = kernel.simulate(ansatz, params).reshape(-1)
    assert abs(float(np.vdot(state, state).real) - 1.0) < 1e-12


def test_hartree_fock_energy_unchanged(h2_integrals) -> None:
    """The physics claim the engine rests on, recomputed through the new kernel."""
    active_space = select_active_space(h2_integrals)
    fermionic = build_fermionic_hamiltonian(h2_integrals, active_space)
    qubit_hamiltonian = map_to_qubits(fermionic)
    ansatz = build_uccsd_ansatz(
        n_electrons=active_space.n_active_electrons,
        n_spin_orbitals=active_space.n_spin_orbitals,
    )

    from qdr_compute_engine.drivers.statevector.statevector_driver import hamiltonian_matrix

    sparse = hamiltonian_matrix(qubit_hamiltonian)
    state = kernel.simulate(ansatz, np.zeros(ansatz.n_params)).reshape(-1)
    energy = float(np.vdot(state, sparse @ state).real)

    assert abs(energy - h2_integrals.e_hf) < 1e-9

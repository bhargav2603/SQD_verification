"""The H2 correctness gate.

Split into two tiers:

* Tests that load integrals from the golden file. These exercise the whole quantum path
  (fermionic Hamiltonian -> JW -> UCCSD -> statevector VQE -> validation) and run on any
  platform. If one of these fails, the engine's physics is wrong.
* Tests marked ``requires_pyscf``. These recompute the golden file's contents from first
  principles with PySCF and assert agreement, which is what makes the committed numbers
  trustworthy rather than merely self-consistent. They run in Linux CI.
"""

import math
from typing import Any

import numpy as np
import pytest

from qdr_compute_engine.active_space.active_space_selector import select_active_space
from qdr_compute_engine.ansatze.uccsd import build_uccsd_ansatz, initial_parameters
from qdr_compute_engine.drivers.statevector.statevector_driver import (
    StatevectorDriver,
    exact_ground_state_energy,
    expectation_value,
    hamiltonian_matrix,
    simulate_statevector,
)
from qdr_compute_engine.hamiltonian.fermionic_builder import build_fermionic_hamiltonian
from qdr_compute_engine.mappers.jordan_wigner import map_to_qubits
from qdr_compute_engine.pipeline.stage_00_fci_gate import run_gate
from qdr_compute_engine.types import IntegralSet, SolveRequest
from qdr_core.limits import CHEMICAL_ACCURACY_HA

# --------------------------------------------------------------------------------------
# Platform-independent: the engine's own correctness
# --------------------------------------------------------------------------------------


def _qubit_hamiltonian(integrals: IntegralSet):
    active_space = select_active_space(integrals)
    return active_space, map_to_qubits(build_fermionic_hamiltonian(integrals, active_space))


def test_golden_file_is_self_consistent(golden_h2: dict[str, Any]) -> None:
    """The stored correlation energy must equal e_fci - e_hf, and be physically sane."""
    correlation = golden_h2["e_fci"] - golden_h2["e_hf"]
    assert correlation == pytest.approx(golden_h2["correlation_energy"], abs=1e-12)
    # Correlation energy is negative by definition, and for H2/STO-3G is about -0.02 Ha.
    assert -0.05 < correlation < 0.0
    assert golden_h2["n_qubits"] == 2 * golden_h2["n_orbitals"]


def test_active_space_is_four_qubits(h2_integrals: IntegralSet) -> None:
    active_space, qubit_hamiltonian = _qubit_hamiltonian(h2_integrals)
    assert active_space.n_active_electrons == 2
    assert active_space.n_active_orbitals == 2
    assert qubit_hamiltonian.n_qubits == 4


def test_circuit_at_zero_parameters_reproduces_hartree_fock(
    h2_integrals: IntegralSet, golden_h2: dict[str, Any]
) -> None:
    """theta = 0 makes UCCSD the identity, so the energy must be exactly HF.

    This isolates the Hamiltonian construction, the JW mapping, the reference state and
    the simulator from anything the optimiser does -- if it fails, no VQE result below
    means anything.
    """
    _, qubit_hamiltonian = _qubit_hamiltonian(h2_integrals)
    ansatz = build_uccsd_ansatz(n_electrons=2, n_spin_orbitals=4)

    energy = expectation_value(
        hamiltonian_matrix(qubit_hamiltonian),
        simulate_statevector(ansatz, initial_parameters(ansatz)),
    )
    assert energy == pytest.approx(golden_h2["e_hf"], abs=1e-10)


def test_exact_diagonalisation_matches_golden_fci(
    h2_integrals: IntegralSet, golden_h2: dict[str, Any]
) -> None:
    """For H2/STO-3G the 4-qubit space is the full determinant space, so the lowest
    eigenvalue is the FCI energy exactly -- not merely close to it."""
    _, qubit_hamiltonian = _qubit_hamiltonian(h2_integrals)
    assert exact_ground_state_energy(qubit_hamiltonian) == pytest.approx(
        golden_h2["e_fci"], abs=1e-10
    )


def test_h2_vqe_matches_golden_fci(h2_integrals: IntegralSet, golden_h2: dict[str, Any]) -> None:
    """THE GATE: the full quantum path reaches FCI to chemical accuracy."""
    active_space, qubit_hamiltonian = _qubit_hamiltonian(h2_integrals)
    ansatz = build_uccsd_ansatz(n_electrons=2, n_spin_orbitals=4)

    result = StatevectorDriver().solve(
        SolveRequest(
            hamiltonian=qubit_hamiltonian,
            active_space=active_space,
            ansatz=ansatz,
            integrals=h2_integrals,
        )
    )

    error = abs(result.energy - golden_h2["e_fci"])
    assert error < CHEMICAL_ACCURACY_HA, (
        f"VQE energy {result.energy:.10f} differs from FCI {golden_h2['e_fci']:.10f} "
        f"by {error:.3e} Ha"
    )


def test_vqe_respects_the_variational_principle(h2_integrals: IntegralSet) -> None:
    """No ansatz can go below the exact ground state of its own Hamiltonian."""
    active_space, qubit_hamiltonian = _qubit_hamiltonian(h2_integrals)
    ansatz = build_uccsd_ansatz(n_electrons=2, n_spin_orbitals=4)
    exact = exact_ground_state_energy(qubit_hamiltonian)

    result = StatevectorDriver().solve(
        SolveRequest(
            hamiltonian=qubit_hamiltonian,
            active_space=active_space,
            ansatz=ansatz,
            integrals=h2_integrals,
        )
    )
    assert result.energy >= exact - 1e-9


def test_vqe_improves_on_hartree_fock(h2_integrals: IntegralSet, golden_h2: dict[str, Any]) -> None:
    """Recovering correlation energy is the entire point; equalling HF is a failure."""
    active_space, qubit_hamiltonian = _qubit_hamiltonian(h2_integrals)
    ansatz = build_uccsd_ansatz(n_electrons=2, n_spin_orbitals=4)

    result = StatevectorDriver().solve(
        SolveRequest(
            hamiltonian=qubit_hamiltonian,
            active_space=active_space,
            ansatz=ansatz,
            integrals=h2_integrals,
        )
    )
    recovered = golden_h2["e_hf"] - result.energy
    assert recovered > 0.01, f"only {recovered:.2e} Ha of correlation energy recovered"


def test_run_gate_passes(h2_integrals: IntegralSet, golden_h2: dict[str, Any]) -> None:
    """The packaged gate function agrees with the assertions above."""
    report = run_gate(h2_integrals, e_fci_reference=golden_h2["e_fci"])
    assert report.passed, f"gate failed: {report.messages}"
    assert report.error_vs_fci < CHEMICAL_ACCURACY_HA


def test_run_gate_detects_a_corrupted_hamiltonian(h2_integrals: IntegralSet) -> None:
    """A gate that cannot fail is not a gate.

    Perturbing a one-electron integral shifts the spectrum, so the reported HF energy no
    longer matches the circuit and the gate must say so.
    """
    corrupted_h1 = np.array(h2_integrals.h1, dtype=float)
    corrupted_h1[0, 0] += 0.05
    corrupted = IntegralSet(
        h1=corrupted_h1,
        h2=h2_integrals.h2,
        e_nuc=h2_integrals.e_nuc,
        n_electrons=h2_integrals.n_electrons,
        e_hf=h2_integrals.e_hf,
    )

    report = run_gate(corrupted, e_fci_reference=None)
    assert not report.passed
    assert any("theta=0" in message for message in report.messages)


# --------------------------------------------------------------------------------------
# PySCF tier: validates the golden file itself
# --------------------------------------------------------------------------------------


@pytest.mark.requires_pyscf
def test_pyscf_reproduces_golden_integrals(golden_h2: dict[str, Any], h2_geometry: Any) -> None:
    """PySCF must reproduce the analytically derived integrals in the golden file.

    This is the check that makes the committed numbers trustworthy: two independent
    derivations -- closed-form Gaussian integrals here, PySCF's implementation there --
    agreeing to near machine precision.
    """
    from qdr_compute_engine.integrals.pyscf_integrals import compute_integrals

    computed = compute_integrals(h2_geometry, basis="sto-3g")

    assert computed.e_nuc == pytest.approx(golden_h2["e_nuc"], abs=1e-9)
    assert computed.e_hf == pytest.approx(golden_h2["e_hf"], abs=1e-8)

    # MO phase is a gauge freedom: PySCF may return an orbital negated relative to ours,
    # which flips the sign of off-diagonal h1 elements without changing any observable.
    # Compare magnitudes for h1 and, for the two-electron tensor, quantities that are
    # invariant under that sign choice.
    np.testing.assert_allclose(np.abs(computed.h1), np.abs(np.array(golden_h2["h1"])), atol=1e-8)
    np.testing.assert_allclose(np.abs(computed.h2), np.abs(np.array(golden_h2["h2"])), atol=1e-8)


@pytest.mark.requires_pyscf
def test_pyscf_fci_matches_golden_fci(golden_h2: dict[str, Any], h2_geometry: Any) -> None:
    """PySCF's FCI solver must agree with the golden file's exact diagonalisation."""
    from qdr_compute_engine.integrals.pyscf_integrals import compute_integrals
    from qdr_compute_engine.solvers.classical_baseline import solve_fci

    computed = compute_integrals(h2_geometry, basis="sto-3g")
    outcome = solve_fci(computed)

    assert outcome.energy == pytest.approx(golden_h2["e_fci"], abs=1e-8)


@pytest.mark.requires_pyscf
def test_end_to_end_gate_from_pyscf_integrals(golden_h2: dict[str, Any], h2_geometry: Any) -> None:
    """The full gate, with integrals computed from the geometry rather than loaded."""
    from qdr_compute_engine.integrals.pyscf_integrals import compute_integrals

    computed = compute_integrals(h2_geometry, basis="sto-3g")
    report = run_gate(computed, e_fci_reference=golden_h2["e_fci"])

    assert report.passed, f"gate failed on PySCF integrals: {report.messages}"
    assert math.isfinite(report.e_vqe)

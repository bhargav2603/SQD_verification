"""Stage 0 -- the H2 correctness gate.

A precondition on the *engine*, not a step in a user's calculation, which is why it sits
outside the 1 -> 9 chain in ``pipeline.py``.

The claim it defends: for a system whose exact answer is known, the quantum path
(integrals -> active space -> fermionic Hamiltonian -> Jordan-Wigner -> UCCSD -> VQE)
reproduces full configuration interaction to chemical accuracy. H2 in a minimal basis is
the right subject because every stage is exercised end to end while the answer stays
exactly computable -- and because UCCSD there has a single double excitation, so the
first-order Trotter approximation is exact and the gate isolates *implementation* error
rather than method error.

A regression here means the engine's physics is wrong, and no result it produces can be
trusted. Fail the gate, fail the build.
"""

from dataclasses import dataclass

import numpy as np

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
from qdr_compute_engine.pipeline.stage_context import StageContext
from qdr_compute_engine.types import IntegralSet, SolveRequest
from qdr_core.limits import CHEMICAL_ACCURACY_HA

STAGE_NAME = "00_fci_gate"

#: Reference geometry: H2 at its near-equilibrium separation, minimal basis.
H2_BOND_LENGTH_ANGSTROM = 0.735
H2_BASIS = "sto-3g"


@dataclass(frozen=True)
class GateReport:
    """Outcome of the gate. ``passed`` is the only thing callers should branch on."""

    passed: bool
    e_hf_circuit: float
    e_hf_reference: float
    e_vqe: float
    e_fci: float
    error_vs_fci: float
    tolerance: float
    messages: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "e_hf_circuit": self.e_hf_circuit,
            "e_hf_reference": self.e_hf_reference,
            "e_vqe": self.e_vqe,
            "e_fci": self.e_fci,
            "error_vs_fci": self.error_vs_fci,
            "tolerance": self.tolerance,
            "messages": list(self.messages),
        }


def run_gate(
    integrals: IntegralSet,
    e_fci_reference: float | None = None,
    tolerance: float = CHEMICAL_ACCURACY_HA,
) -> GateReport:
    """Run the H2 quantum path and compare against FCI.

    ``e_fci_reference`` may be supplied (from the golden file, or from PySCF). When it is
    omitted the exact ground state of the same qubit Hamiltonian is used instead, which
    still catches ansatz and optimiser faults but cannot catch an error in the Hamiltonian
    itself -- a wrong Hamiltonian would simply be compared against its own spectrum.
    """
    messages: list[str] = []

    active_space = select_active_space(integrals)
    fermionic = build_fermionic_hamiltonian(integrals, active_space)
    qubit_hamiltonian = map_to_qubits(fermionic)
    ansatz = build_uccsd_ansatz(
        n_electrons=active_space.n_active_electrons,
        n_spin_orbitals=active_space.n_spin_orbitals,
    )

    # Check 1: at theta = 0 the ansatz is the identity, so the circuit must reproduce the
    # Hartree-Fock energy exactly. This isolates the Hamiltonian construction, the JW
    # mapping and the reference state from anything the optimiser does.
    sparse = hamiltonian_matrix(qubit_hamiltonian)
    e_hf_circuit = expectation_value(
        sparse, simulate_statevector(ansatz, initial_parameters(ansatz))
    )
    e_hf_reference = float(integrals.e_hf) if integrals.e_hf is not None else float("nan")

    if integrals.e_hf is not None and not np.isclose(e_hf_circuit, e_hf_reference, atol=1e-9):
        messages.append(
            f"circuit at theta=0 gives {e_hf_circuit:.10f} Ha but the reference "
            f"Hartree-Fock energy is {e_hf_reference:.10f} Ha"
        )

    # Check 2: the variational result must reach FCI.
    e_exact = exact_ground_state_energy(qubit_hamiltonian)
    target = e_fci_reference if e_fci_reference is not None else e_exact
    if e_fci_reference is None:
        messages.append(
            "no external FCI reference supplied; compared against the exact ground "
            "state of the same qubit Hamiltonian"
        )

    result = StatevectorDriver().solve(
        SolveRequest(
            hamiltonian=qubit_hamiltonian,
            active_space=active_space,
            ansatz=ansatz,
            integrals=integrals,
        )
    )

    error = abs(result.energy - target)
    if error >= tolerance:
        messages.append(
            f"VQE energy {result.energy:.10f} Ha differs from FCI {target:.10f} Ha "
            f"by {error:.2e}, above the {tolerance:.1e} tolerance"
        )

    # Check 3: the variational principle. Landing meaningfully below the exact ground
    # state of the same Hamiltonian is impossible, so it indicates a bug rather than luck.
    if result.energy < e_exact - 1e-9:
        messages.append(
            f"VQE energy {result.energy:.10f} Ha lies below the exact ground state "
            f"{e_exact:.10f} Ha, violating the variational principle"
        )

    return GateReport(
        passed=not messages,
        e_hf_circuit=e_hf_circuit,
        e_hf_reference=e_hf_reference,
        e_vqe=result.energy,
        e_fci=target,
        error_vs_fci=error,
        tolerance=tolerance,
        messages=tuple(messages),
    )


def run(context: StageContext) -> StageContext:
    """Pipeline-shaped wrapper, for running the gate as a preflight step."""
    integrals = context.require_integrals()
    report = run_gate(integrals)
    context = context.evolve(config={**context.config, "fci_gate": report.to_dict()})
    for message in report.messages:
        context = context.with_warning(f"FCI gate: {message}")
    return context

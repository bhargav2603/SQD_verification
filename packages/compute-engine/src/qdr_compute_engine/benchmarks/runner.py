"""Run a benchmark system and attribute every Hartree of the error.

The point of separating the terms
---------------------------------
"VQE gave -75.01 Ha and FCI gives -75.02 Ha" is not a result, because it does not say
*which* approximation cost the 0.01. This runner therefore computes three energies for
every case and reports the two differences between them separately:

    E_VQE   variational result in the active space
    E_CASCI exact diagonalisation of the *same* active space   (PySCF FCI on the
            restricted integrals -- an independent implementation of the same physics)
    E_FCI   exact diagonalisation of the *full* orbital space

giving

    E_VQE - E_CASCI  ansatz + Trotter + optimiser error   <- what the quantum path costs
    E_CASCI - E_FCI  active-space truncation error        <- what the truncation costs

The second term is almost always the larger, and it is the one a bare "VQE vs FCI"
comparison hides. Keeping them apart is what lets the active-space sweep in
``systems.py`` measure the truncation error rather than assume it.

Both differences are signed and reported as-is. ``E_VQE - E_CASCI`` is non-negative by
the variational principle; a negative value is a bug, not a good result, and is flagged
rather than smoothed over.

PySCF
-----
Everything here needs PySCF and therefore a Linux or macOS host -- see
``docs/REMOTE_EXECUTION.md``. Nothing in this module is imported by the engine itself,
so the package stays importable where PySCF is not installed.
"""

import platform
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from qdr_compute_engine.active_space.active_space_selector import (
    restrict_integrals,
    select_active_space,
)
from qdr_compute_engine.ansatze.uccsd import build_uccsd_ansatz, initial_parameters
from qdr_compute_engine.benchmarks.systems import ActiveSpaceSpec, BenchmarkSystem
from qdr_compute_engine.drivers.statevector.statevector_driver import (
    StatevectorDriver,
    exact_ground_state_energy,
)
from qdr_compute_engine.hamiltonian.fermionic_builder import build_fermionic_hamiltonian
from qdr_compute_engine.integrals.pyscf_integrals import compute_integrals
from qdr_compute_engine.mappers.jordan_wigner import map_to_qubits
from qdr_compute_engine.solvers.classical_baseline import solve_fci
from qdr_compute_engine.types import GateName, SolveRequest
from qdr_core.limits import (
    CHEMICAL_ACCURACY_HA,
    HARTREE_TO_KCAL_PER_MOL,
    MAX_EXACT_REFERENCE_QUBITS,
)


@dataclass
class BenchmarkRecord:
    """One (system, active space) result. Every field is measured, none is assumed."""

    system: str
    basis: str
    active_space_label: str

    n_qubits: int
    n_parameters: int
    n_gates: int
    n_cnot: int
    n_hamiltonian_terms: int

    e_hf: float
    e_vqe: float
    e_casci: float | None
    e_fci_full: float | None

    #: E_VQE - E_CASCI: ansatz + Trotter + optimiser. Non-negative, variationally.
    error_quantum_path: float | None
    #: E_CASCI - E_FCI: the cost of truncating the orbital space.
    error_active_space: float | None
    #: E_VQE - E_FCI: the total, and the only one a naive comparison would report.
    error_total: float | None

    optimizer: str
    used_analytic_gradient: bool
    converged: bool
    n_iterations: int
    n_energy_evaluations: int

    seconds_integrals: float
    seconds_vqe: float
    seconds_reference: float

    warnings: list[str] = field(default_factory=list)

    @property
    def within_chemical_accuracy(self) -> bool:
        """Against the *total* error -- the honest bar, not the flattering one."""
        return self.error_total is not None and abs(self.error_total) < CHEMICAL_ACCURACY_HA

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["within_chemical_accuracy"] = self.within_chemical_accuracy
        return payload


def _count_cnots(gates: tuple[Any, ...]) -> int:
    return sum(1 for gate in gates if gate.name == GateName.CX)


def run_case(
    system: BenchmarkSystem,
    spec: ActiveSpaceSpec,
    *,
    optimizer: str | None = None,
    max_iterations: int = 2000,
    compute_full_fci: bool = True,
) -> BenchmarkRecord:
    """Run one system in one active space and decompose the error.

    ``optimizer=None`` lets the driver choose, which means L-BFGS-B driven by the
    analytic adjoint gradient. Passing ``"COBYLA"`` selects the derivative-free path on
    identical physics, which is how the two searches are compared.
    """
    warnings: list[str] = []

    started = time.perf_counter()
    integrals = compute_integrals(system.geometry, basis=system.basis)
    seconds_integrals = time.perf_counter() - started

    # Reference 1: exact in the FULL orbital space. Independent of any active-space
    # choice, so it is the yardstick for the truncation error.
    started = time.perf_counter()
    e_fci_full: float | None = None
    if compute_full_fci:
        try:
            e_fci_full = solve_fci(integrals).energy
        except (MemoryError, ValueError) as exc:
            warnings.append(f"full-space FCI unavailable: {exc}")
    seconds_reference = time.perf_counter() - started

    active_space = select_active_space(
        integrals,
        n_active_orbitals=spec.n_active_orbitals,
        n_active_electrons=spec.n_active_electrons,
    )
    restricted = restrict_integrals(integrals, active_space)

    fermionic = build_fermionic_hamiltonian(restricted, active_space)
    qubit_hamiltonian = map_to_qubits(fermionic)
    ansatz = build_uccsd_ansatz(
        n_electrons=active_space.n_active_electrons,
        n_spin_orbitals=active_space.n_spin_orbitals,
    )

    # Reference 2: exact in the SAME active space. Computed two ways where possible --
    # PySCF's FCI on the restricted integrals, and direct diagonalisation of the qubit
    # Hamiltonian. They must agree; a disagreement means the Hamiltonian handed to the
    # quantum path is not the one the classical reference solved.
    started = time.perf_counter()
    e_casci: float | None = None
    try:
        e_casci = solve_fci(restricted).energy
    except (MemoryError, ValueError, NotImplementedError) as exc:
        warnings.append(f"CASCI reference unavailable: {exc}")

    if qubit_hamiltonian.n_qubits <= MAX_EXACT_REFERENCE_QUBITS:
        e_qubit_exact = exact_ground_state_energy(qubit_hamiltonian)
        if e_casci is None:
            e_casci = e_qubit_exact
        elif abs(e_qubit_exact - e_casci) > 1e-7:
            warnings.append(
                f"CASCI from integrals ({e_casci:.10f} Ha) and exact diagonalisation of "
                f"the qubit Hamiltonian ({e_qubit_exact:.10f} Ha) disagree by "
                f"{abs(e_qubit_exact - e_casci):.2e} Ha; the two paths are not solving "
                "the same operator"
            )
    seconds_reference += time.perf_counter() - started

    options: dict[str, Any] = {}
    if optimizer is not None:
        options["optimizer"] = optimizer

    started = time.perf_counter()
    result = StatevectorDriver().solve(
        SolveRequest(
            hamiltonian=qubit_hamiltonian,
            active_space=active_space,
            ansatz=ansatz,
            initial_params=initial_parameters(ansatz),
            integrals=restricted,
            max_iterations=max_iterations,
            options=options,
        )
    )
    seconds_vqe = time.perf_counter() - started

    error_quantum_path = None if e_casci is None else result.energy - e_casci
    error_active_space = None if (e_casci is None or e_fci_full is None) else e_casci - e_fci_full
    error_total = None if e_fci_full is None else result.energy - e_fci_full

    if error_quantum_path is not None and error_quantum_path < -1e-9:
        warnings.append(
            f"VQE energy lies {abs(error_quantum_path):.2e} Ha below the exact ground "
            "state of the same active space, which violates the variational principle"
        )
    if not result.converged:
        warnings.append("optimizer did not report convergence")

    return BenchmarkRecord(
        system=system.name,
        basis=system.basis,
        active_space_label=spec.label,
        n_qubits=qubit_hamiltonian.n_qubits,
        n_parameters=ansatz.n_params,
        n_gates=len(ansatz.gates),
        n_cnot=_count_cnots(ansatz.gates),
        n_hamiltonian_terms=qubit_hamiltonian.n_terms,
        e_hf=float(integrals.e_hf) if integrals.e_hf is not None else float("nan"),
        e_vqe=result.energy,
        e_casci=e_casci,
        e_fci_full=e_fci_full,
        error_quantum_path=error_quantum_path,
        error_active_space=error_active_space,
        error_total=error_total,
        optimizer=str(result.metadata.get("optimizer", "unknown")),
        used_analytic_gradient=bool(result.metadata.get("used_analytic_gradient", False)),
        converged=result.converged,
        n_iterations=result.n_iterations,
        n_energy_evaluations=result.n_energy_evaluations,
        seconds_integrals=seconds_integrals,
        seconds_vqe=seconds_vqe,
        seconds_reference=seconds_reference,
        warnings=warnings,
    )


def run_system(system: BenchmarkSystem, **kwargs: Any) -> list[BenchmarkRecord]:
    """Run every active space listed for ``system``."""
    return [run_case(system, spec, **kwargs) for spec in system.active_spaces]


def environment_metadata() -> dict[str, Any]:
    """What the numbers were measured on. Timings are meaningless without it."""
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
    }
    try:
        import pyscf

        metadata["pyscf"] = pyscf.__version__
    except ImportError:  # pragma: no cover - the runner cannot get here without PySCF
        metadata["pyscf"] = "not installed"
    try:
        import openfermion

        metadata["openfermion"] = openfermion.__version__
    except ImportError:  # pragma: no cover
        metadata["openfermion"] = "not installed"
    return metadata


def hartree_to_kcal(value: float | None) -> float | None:
    return None if value is None else value * HARTREE_TO_KCAL_PER_MOL

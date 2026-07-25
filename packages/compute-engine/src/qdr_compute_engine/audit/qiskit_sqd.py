"""Optional compatibility adapter for Qiskit's maintained SQD implementation.

The imports are lazy so the core audit remains usable on Windows and on credential-free
machines.  The chemistry workflow itself is intended for Linux/WSL, matching PySCF and
Qiskit's published support guidance.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from itertools import product
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from qdr_compute_engine.audit.models import (
    AuditBudget,
    AuditProblem,
    PostprocessResult,
    ResourceLedger,
    SampleBatch,
)
from qdr_compute_engine.solvers.sqd import sector_dimension, sector_mask
from qdr_core.errors import BackendUnavailable, ValidationError
from qdr_core.provenance import is_content_hash

BACKEND_NAME = "qiskit-sqd"


@dataclass(frozen=True, slots=True, init=False)
class FermionicAuditProblem:
    """Spatial-orbital tensors needed by Qiskit's fermionic SQD entrypoint."""

    problem_hash: str
    one_body_tensor: NDArray[np.float64]
    two_body_tensor: NDArray[np.float64]
    constant_energy: float
    n_orbitals: int

    def __init__(
        self,
        *,
        problem_hash: str,
        one_body_tensor: NDArray[np.floating[Any]],
        two_body_tensor: NDArray[np.floating[Any]],
        constant_energy: float = 0.0,
    ) -> None:
        if not is_content_hash(problem_hash):
            raise ValidationError(
                "problem_hash must be a canonical content hash",
                problem_hash=problem_hash,
            )
        one_body = np.asarray(one_body_tensor, dtype=float)
        two_body = np.asarray(two_body_tensor, dtype=float)
        if (
            one_body.ndim != 2
            or one_body.shape[0] != one_body.shape[1]
            or one_body.shape[0] == 0
        ):
            raise ValidationError(
                "one_body_tensor must be a non-empty square matrix",
                shape=list(one_body.shape),
            )
        n_orbitals = int(one_body.shape[0])
        expected_two_body = (n_orbitals,) * 4
        if two_body.shape != expected_two_body:
            raise ValidationError(
                "two_body_tensor must have shape (norb, norb, norb, norb)",
                shape=list(two_body.shape),
                expected_shape=list(expected_two_body),
            )
        if not bool(np.all(np.isfinite(one_body))) or not bool(
            np.all(np.isfinite(two_body))
        ):
            raise ValidationError("fermionic integral tensors must be finite")
        if not math.isfinite(constant_energy):
            raise ValidationError(
                "constant_energy must be finite",
                constant_energy=constant_energy,
            )
        one_body_copy = np.array(one_body, dtype=np.float64, copy=True)
        two_body_copy = np.array(two_body, dtype=np.float64, copy=True)
        one_body_copy.flags.writeable = False
        two_body_copy.flags.writeable = False
        object.__setattr__(self, "problem_hash", problem_hash)
        object.__setattr__(self, "one_body_tensor", one_body_copy)
        object.__setattr__(self, "two_body_tensor", two_body_copy)
        object.__setattr__(self, "constant_energy", float(constant_energy))
        object.__setattr__(self, "n_orbitals", n_orbitals)


@dataclass(frozen=True, slots=True)
class QiskitBindings:
    """Narrow external boundary, injectable for contract tests."""

    bit_array_from_samples: Callable[[tuple[str, ...], int], Any]
    diagonalize_fermionic_hamiltonian: Callable[..., Any]
    addon_version: str


def integer_samples_to_bitstrings(
    configurations: NDArray[np.int64],
    *,
    n_qubits: int,
) -> tuple[str, ...]:
    """Render QDR integer samples as fixed-width Qiskit big-endian bitstrings."""

    raw = np.asarray(configurations)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise ValidationError("configurations must be a one-dimensional integer draw")
    if isinstance(n_qubits, bool) or n_qubits <= 0:
        raise ValidationError("n_qubits must be positive", n_qubits=n_qubits)
    values = np.asarray(raw, dtype=np.int64)
    if values.size == 0:
        raise ValidationError("configurations must not be empty")
    if bool(np.any(values < 0)) or bool(np.any(values >= (1 << n_qubits))):
        raise ValidationError(
            "configuration lies outside the Qiskit register",
            n_qubits=n_qubits,
        )
    return tuple(format(int(value), f"0{n_qubits}b") for value in values)


def _load_qiskit_bindings() -> QiskitBindings | None:
    try:
        from qiskit.primitives import BitArray
        from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian
    except ImportError:
        return None
    try:
        addon_version = version("qiskit-addon-sqd")
    except (
        PackageNotFoundError
    ):  # pragma: no cover - import implies a nonstandard install
        addon_version = "unknown"
    return QiskitBindings(
        bit_array_from_samples=lambda samples, num_bits: BitArray.from_samples(
            samples,
            num_bits=num_bits,
        ),
        diagonalize_fermionic_hamiltonian=diagonalize_fermionic_hamiltonian,
        addon_version=addon_version,
    )


def qiskit_sqd_available() -> bool:
    """Whether both Qiskit ``BitArray`` and the SQD addon can be imported."""

    return _load_qiskit_bindings() is not None


def _qdr_configuration(alpha: int, beta: int, n_orbitals: int) -> int:
    configuration = 0
    n_qubits = 2 * n_orbitals
    for orbital in range(n_orbitals):
        if (alpha >> orbital) & 1:
            configuration |= 1 << (n_qubits - 1 - 2 * orbital)
        if (beta >> orbital) & 1:
            configuration |= 1 << (n_qubits - 1 - (2 * orbital + 1))
    return configuration


def _state_configurations(sci_state: Any, n_orbitals: int) -> tuple[int, ...]:
    alpha = tuple(int(value) for value in np.asarray(sci_state.ci_strs_a).reshape(-1))
    beta = tuple(int(value) for value in np.asarray(sci_state.ci_strs_b).reshape(-1))
    return tuple(
        sorted(
            _qdr_configuration(alpha_string, beta_string, n_orbitals)
            for alpha_string, beta_string in product(alpha, beta)
        )
    )


@dataclass(frozen=True, slots=True, init=False)
class QiskitSQDPostprocessor:
    """Canonical self-consistent configuration recovery using Qiskit SQD 0.12.x."""

    fermionic_problems: Mapping[str, FermionicAuditProblem]
    bindings: QiskitBindings | None
    num_batches: int
    max_iterations: int
    energy_tol: float
    occupancies_tol: float
    symmetrize_spin: bool
    postprocessor_id: str
    dimension_policy: str

    def __init__(
        self,
        *,
        fermionic_problems: Mapping[str, FermionicAuditProblem],
        bindings: QiskitBindings | None = None,
        num_batches: int = 1,
        max_iterations: int = 5,
        energy_tol: float = 1.0e-3,
        occupancies_tol: float = 1.0e-3,
        symmetrize_spin: bool = True,
    ) -> None:
        if num_batches <= 0 or max_iterations <= 0:
            raise ValidationError(
                "Qiskit SQD batch and iteration counts must be positive"
            )
        if energy_tol <= 0.0 or occupancies_tol <= 0.0:
            raise ValidationError("Qiskit SQD convergence tolerances must be positive")
        object.__setattr__(self, "fermionic_problems", dict(fermionic_problems))
        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(self, "num_batches", num_batches)
        object.__setattr__(self, "max_iterations", max_iterations)
        object.__setattr__(self, "energy_tol", energy_tol)
        object.__setattr__(self, "occupancies_tol", occupancies_tol)
        object.__setattr__(self, "symmetrize_spin", symmetrize_spin)
        object.__setattr__(self, "postprocessor_id", BACKEND_NAME)
        object.__setattr__(self, "dimension_policy", "measured")

    def process(
        self,
        problem: AuditProblem,
        batch: SampleBatch,
        budget: AuditBudget,
    ) -> tuple[PostprocessResult, ...]:
        fermionic = self.fermionic_problems.get(problem.identity_hash)
        if fermionic is None:
            raise BackendUnavailable(
                BACKEND_NAME,
                "no fermionic integrals are registered for the frozen problem hash",
                problem_hash=problem.identity_hash,
            )
        bindings = self.bindings or _load_qiskit_bindings()
        if bindings is None:
            raise BackendUnavailable(
                BACKEND_NAME,
                "qiskit-addon-sqd and Qiskit BitArray are not installed; "
                "install the audit-qiskit dependency group on Linux/WSL",
            )
        bitstrings = integer_samples_to_bitstrings(
            batch.configurations,
            n_qubits=batch.n_qubits,
        )
        bit_array = bindings.bit_array_from_samples(bitstrings, batch.n_qubits)
        if batch.shots_returned is None:  # Defensive narrowing; SampleBatch fills this.
            raise ValidationError("sample batch is missing returned-shot accounting")
        shots_returned = batch.shots_returned
        _, sector_membership = sector_mask(
            batch.configurations,
            batch.n_qubits,
            problem.n_alpha,
            problem.n_beta,
        )
        n_in_sector = int(sector_membership.sum())
        full_sector_dimension = sector_dimension(
            batch.n_qubits,
            problem.n_alpha,
            problem.n_beta,
        )

        outputs: list[PostprocessResult] = []
        for requested_dimension in budget.subspace_dimensions:
            history: list[list[Any]] = []

            def callback(results: list[Any]) -> None:
                history.append(results)

            started_wall = time.perf_counter()
            started_cpu = time.process_time()
            result = bindings.diagonalize_fermionic_hamiltonian(
                fermionic.one_body_tensor,
                fermionic.two_body_tensor,
                bit_array,
                samples_per_batch=requested_dimension,
                norb=fermionic.n_orbitals,
                nelec=(problem.n_alpha, problem.n_beta),
                num_batches=self.num_batches,
                energy_tol=self.energy_tol,
                occupancies_tol=self.occupancies_tol,
                max_iterations=self.max_iterations,
                symmetrize_spin=self.symmetrize_spin,
                max_dim=requested_dimension,
                callback=callback,
                seed=batch.seed,
            )
            elapsed_wall = time.perf_counter() - started_wall
            elapsed_cpu = time.process_time() - started_cpu
            configurations = _state_configurations(
                result.sci_state, fermionic.n_orbitals
            )
            energy = float(result.energy) + fermionic.constant_energy
            outputs.append(
                PostprocessResult(
                    problem_hash=problem.identity_hash,
                    subspace_dimension=len(configurations),
                    energy=energy,
                    configurations=configurations,
                    is_exact=len(configurations) == full_sector_dimension,
                    n_samples=shots_returned,
                    n_samples_in_sector=n_in_sector,
                    ledger=ResourceLedger(
                        cpu_seconds=elapsed_cpu,
                        wall_seconds=elapsed_wall,
                        recovery_seconds=elapsed_wall,
                    ),
                    metadata={
                        "postprocessor_id": self.postprocessor_id,
                        "qiskit_addon_sqd_version": bindings.addon_version,
                        "requested_samples_per_batch": requested_dimension,
                        "num_batches": self.num_batches,
                        "max_iterations": self.max_iterations,
                        "recovery_iterations": len(history),
                        "symmetrize_spin": self.symmetrize_spin,
                        "spin_squared": getattr(result, "spin_sq", None),
                    },
                )
            )
        return tuple(outputs)


__all__ = [
    "FermionicAuditProblem",
    "QiskitBindings",
    "QiskitSQDPostprocessor",
    "integer_samples_to_bitstrings",
    "qiskit_sqd_available",
]

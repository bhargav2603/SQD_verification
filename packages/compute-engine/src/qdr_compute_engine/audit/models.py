"""Immutable evidence contracts for sampler audits.

Raw sampler output and post-processed results are different kinds of evidence.  The
types in this module keep that boundary explicit: :class:`SampleBatch` retains every
draw, including duplicates and wrong-sector configurations, while later postprocessors
record what they retained.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from qdr_compute_engine.reproducibility.input_hashing import normalize
from qdr_compute_engine.types import QubitHamiltonian
from qdr_core.errors import ValidationError
from qdr_core.provenance import content_hash, is_content_hash


class SamplerKind(StrEnum):
    """Scientifically distinct sources of determinant samples."""

    ORACLE = "oracle"
    IDEAL_QUANTUM = "ideal_quantum"
    NOISY_QUANTUM = "noisy_quantum"
    HARDWARE_QUANTUM = "hardware_quantum"
    CLASSICAL_SELECTED = "classical_selected"
    CLASSICAL_EXCITATION = "classical_excitation"
    NEGATIVE_CONTROL = "negative_control"


class DatasetSplit(StrEnum):
    """Whether an instance may influence method selection."""

    DEVELOPMENT = "development"
    HELD_OUT = "held_out"


def _validate_optional_non_negative_float(name: str, value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or value < 0.0):
        raise ValidationError(
            f"{name} must be a finite non-negative measurement or None",
            field=name,
            value=value,
        )


def _validate_optional_non_negative_int(name: str, value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or value < 0):
        raise ValidationError(
            f"{name} must be a non-negative integer measurement or None",
            field=name,
            value=value,
        )


@dataclass(frozen=True, slots=True)
class ResourceLedger:
    """Measured resources for one sampler or post-processing operation.

    ``None`` means unmeasured.  It never means zero.
    """

    circuit_evaluations: int | None = None
    qpu_seconds: float | None = None
    cpu_seconds: float | None = None
    wall_seconds: float | None = None
    preprocessing_seconds: float | None = None
    recovery_seconds: float | None = None
    diagonalization_seconds: float | None = None
    peak_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        _validate_optional_non_negative_int(
            "circuit_evaluations", self.circuit_evaluations
        )
        _validate_optional_non_negative_int("peak_memory_bytes", self.peak_memory_bytes)
        for name in (
            "qpu_seconds",
            "cpu_seconds",
            "wall_seconds",
            "preprocessing_seconds",
            "recovery_seconds",
            "diagonalization_seconds",
        ):
            _validate_optional_non_negative_float(name, getattr(self, name))

    def to_dict(self) -> dict[str, int | float | None]:
        """Return a JSON-native record without changing unknowns into zeroes."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class AuditBudget:
    """A sample budget and the strictly nested SQD dimensions evaluated from it."""

    shots: int
    subspace_dimensions: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.shots, bool)
            or not isinstance(self.shots, int)
            or self.shots <= 0
        ):
            raise ValidationError(
                "shots must be a positive integer",
                shots=self.shots,
            )
        dimensions = tuple(self.subspace_dimensions)
        object.__setattr__(self, "subspace_dimensions", dimensions)
        if not dimensions:
            raise ValidationError(
                "subspace_dimensions must contain at least one dimension",
                subspace_dimensions=[],
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in dimensions
        ):
            raise ValidationError(
                "subspace_dimensions must contain positive integers",
                subspace_dimensions=list(dimensions),
            )
        if any(
            right <= left
            for left, right in zip(dimensions, dimensions[1:], strict=False)
        ):
            raise ValidationError(
                "subspace_dimensions must be strictly increasing",
                subspace_dimensions=list(dimensions),
            )


def _freeze_json(value: Any) -> Any:
    """Defensively freeze a JSON-like value while retaining its scalar types."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, np.ndarray):
        return tuple(_freeze_json(item) for item in value.tolist())
    if isinstance(value, np.generic):
        return _freeze_json(value.item())
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    raise ValidationError(
        "metadata and descriptors must contain only JSON-compatible values",
        value_type=type(value).__name__,
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _hamiltonian_payload(hamiltonian: QubitHamiltonian) -> dict[str, Any]:
    terms: list[dict[str, Any]] = []
    for pauli_term, coefficient in hamiltonian.operator.terms.items():
        coefficient_complex = complex(coefficient)
        terms.append(
            {
                "pauli": [[int(qubit), str(axis)] for qubit, axis in pauli_term],
                "coefficient": [coefficient_complex.real, coefficient_complex.imag],
            }
        )
    terms.sort(key=lambda item: item["pauli"])
    return {
        "n_qubits": hamiltonian.n_qubits,
        "mapping": hamiltonian.mapping,
        "terms": terms,
    }


@dataclass(frozen=True, slots=True)
class AuditProblem:
    """One frozen Hamiltonian and particle sector used across every sampler arm."""

    problem_id: str
    hamiltonian: QubitHamiltonian
    n_alpha: int
    n_beta: int
    reference_energy: float
    split: DatasetSplit
    descriptors: Mapping[str, Any] = field(default_factory=dict)
    identity_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.problem_id.strip():
            raise ValidationError(
                "problem_id must be non-empty", problem_id=self.problem_id
            )
        if self.hamiltonian.n_qubits <= 0 or self.hamiltonian.n_qubits % 2:
            raise ValidationError(
                "audit Hamiltonians require a positive even number of spin-orbital qubits",
                n_qubits=self.hamiltonian.n_qubits,
            )
        n_orbitals = self.hamiltonian.n_qubits // 2
        if (
            isinstance(self.n_alpha, bool)
            or isinstance(self.n_beta, bool)
            or self.n_alpha < 0
            or self.n_beta < 0
            or self.n_alpha > n_orbitals
            or self.n_beta > n_orbitals
        ):
            raise ValidationError(
                "particle sector cannot be represented by the audit Hamiltonian",
                n_alpha=self.n_alpha,
                n_beta=self.n_beta,
                n_orbitals=n_orbitals,
            )
        if not math.isfinite(self.reference_energy):
            raise ValidationError(
                "reference_energy must be finite",
                reference_energy=self.reference_energy,
            )
        frozen_descriptors = _freeze_json(self.descriptors)
        object.__setattr__(self, "descriptors", frozen_descriptors)
        payload = {
            "problem_id": self.problem_id,
            "hamiltonian": _hamiltonian_payload(self.hamiltonian),
            "n_alpha": self.n_alpha,
            "n_beta": self.n_beta,
            "reference_energy": self.reference_energy,
            "descriptors": _thaw_json(frozen_descriptors),
        }
        object.__setattr__(self, "identity_hash", content_hash(normalize(payload)))


@dataclass(frozen=True, slots=True)
class SampleBatch:
    """Unmodified samples emitted by one source for one audit cell."""

    source_id: str
    kind: SamplerKind
    configurations: NDArray[np.int64]
    n_qubits: int
    shots_requested: int
    seed: int
    shots_returned: int | None = None
    ledger: ResourceLedger = field(default_factory=ResourceLedger)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValidationError(
                "source_id must be non-empty", source_id=self.source_id
            )
        if isinstance(self.n_qubits, bool) or self.n_qubits <= 0:
            raise ValidationError("n_qubits must be positive", n_qubits=self.n_qubits)
        if isinstance(self.shots_requested, bool) or self.shots_requested <= 0:
            raise ValidationError(
                "shots_requested must be positive",
                shots_requested=self.shots_requested,
            )

        raw = np.asarray(self.configurations)
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise ValidationError(
                "configurations must be a one-dimensional integer draw",
                shape=list(raw.shape),
                dtype=str(raw.dtype),
            )
        configurations = np.array(raw, dtype=np.int64, copy=True)
        upper_bound = 1 << self.n_qubits
        if configurations.size == 0:
            raise ValidationError(
                "configurations must contain at least one returned shot"
            )
        if bool(np.any(configurations < 0)) or bool(
            np.any(configurations >= upper_bound)
        ):
            raise ValidationError(
                "configuration lies outside the sampled register",
                n_qubits=self.n_qubits,
                minimum=int(configurations.min()),
                maximum=int(configurations.max()),
            )

        shots_returned = (
            configurations.size if self.shots_returned is None else self.shots_returned
        )
        if shots_returned != configurations.size:
            raise ValidationError(
                "shots_returned must equal the number of raw configurations",
                shots_returned=shots_returned,
                configurations=int(configurations.size),
            )
        if shots_returned > self.shots_requested:
            raise ValidationError(
                "shots_returned cannot exceed shots_requested",
                shots_returned=shots_returned,
                shots_requested=self.shots_requested,
            )

        configurations.flags.writeable = False
        object.__setattr__(self, "configurations", configurations)
        object.__setattr__(self, "shots_returned", int(shots_returned))
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))


@dataclass(frozen=True, slots=True)
class PostprocessResult:
    """One retained-subspace result from a frozen postprocessor."""

    problem_hash: str
    subspace_dimension: int
    energy: float
    configurations: tuple[int, ...]
    is_exact: bool
    n_samples: int
    n_samples_in_sector: int
    ledger: ResourceLedger = field(default_factory=ResourceLedger)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not is_content_hash(self.problem_hash):
            raise ValidationError(
                "problem_hash must be a canonical content hash",
                problem_hash=self.problem_hash,
            )
        configurations = tuple(int(value) for value in self.configurations)
        object.__setattr__(self, "configurations", configurations)
        if (
            self.subspace_dimension <= 0
            or len(configurations) != self.subspace_dimension
        ):
            raise ValidationError(
                "subspace_dimension must equal the retained configuration count",
                subspace_dimension=self.subspace_dimension,
                configuration_count=len(configurations),
            )
        if len(set(configurations)) != len(configurations):
            raise ValidationError("retained configurations must be unique")
        if not math.isfinite(self.energy):
            raise ValidationError(
                "postprocessed energy must be finite", energy=self.energy
            )
        if (
            self.n_samples < 0
            or self.n_samples_in_sector < 0
            or self.n_samples_in_sector > self.n_samples
        ):
            raise ValidationError(
                "postprocessing sample counts are inconsistent",
                n_samples=self.n_samples,
                n_samples_in_sector=self.n_samples_in_sector,
            )
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))


@dataclass(frozen=True, slots=True)
class AuditObservation:
    """One analysis-ready observation from a sampler/postprocessor cell."""

    problem_id: str
    problem_hash: str
    source_id: str
    kind: SamplerKind
    split: DatasetSplit
    seed: int
    shots: int
    subspace_dimension: int
    energy: float
    reference_energy: float
    absolute_error: float
    sector_retention: float
    configurations: tuple[int, ...]
    ledger: ResourceLedger
    shots_returned: int | None = None
    n_qubits: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not is_content_hash(self.problem_hash):
            raise ValidationError(
                "problem_hash must be a canonical content hash",
                problem_hash=self.problem_hash,
            )
        if self.subspace_dimension <= 0:
            raise ValidationError(
                "subspace_dimension must be positive",
                subspace_dimension=self.subspace_dimension,
            )
        if not all(
            math.isfinite(value)
            for value in (
                self.energy,
                self.reference_energy,
                self.absolute_error,
                self.sector_retention,
            )
        ):
            raise ValidationError("observation numerical fields must be finite")
        if self.absolute_error < 0.0:
            raise ValidationError(
                "absolute_error must be non-negative",
                absolute_error=self.absolute_error,
            )
        shots_returned = (
            self.shots if self.shots_returned is None else self.shots_returned
        )
        if (
            isinstance(shots_returned, bool)
            or not isinstance(shots_returned, int)
            or shots_returned <= 0
            or shots_returned > self.shots
        ):
            raise ValidationError(
                "shots_returned must be a positive integer no greater than requested shots",
                shots=self.shots,
                shots_returned=shots_returned,
            )
        object.__setattr__(self, "shots_returned", shots_returned)
        if self.n_qubits is not None and (
            isinstance(self.n_qubits, bool)
            or not isinstance(self.n_qubits, int)
            or self.n_qubits <= 0
        ):
            raise ValidationError("n_qubits must be a positive integer when recorded")
        if not 0.0 <= self.sector_retention <= 1.0:
            raise ValidationError(
                "sector_retention must lie in [0, 1]",
                sector_retention=self.sector_retention,
            )
        configurations = tuple(int(value) for value in self.configurations)
        object.__setattr__(self, "configurations", configurations)
        if len(configurations) != self.subspace_dimension:
            raise ValidationError(
                "observation configuration count must match subspace_dimension",
                subspace_dimension=self.subspace_dimension,
                configuration_count=len(configurations),
            )
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))

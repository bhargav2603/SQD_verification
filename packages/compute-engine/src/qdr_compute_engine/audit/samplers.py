"""Determinant sample sources used in counterfactual SQD comparisons."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Protocol

import numpy as np
from numpy.typing import NDArray

from qdr_compute_engine.audit.models import (
    AuditBudget,
    AuditProblem,
    ResourceLedger,
    SampleBatch,
    SamplerKind,
)
from qdr_compute_engine.solvers.sqd import sample_configurations, sector_configurations
from qdr_core.errors import ValidationError


class DeterminantSampler(Protocol):
    """A source that emits raw computational-basis configurations."""

    @property
    def source_id(self) -> str:
        """Stable arm identity recorded in every observation."""

    @property
    def kind(self) -> SamplerKind:
        """Scientific category of this sampling arm."""

    def sample(
        self, problem: AuditProblem, budget: AuditBudget, seed: int
    ) -> SampleBatch:
        """Draw exactly ``budget.shots`` raw configurations."""


def _timed_ledger(started_wall: float, started_cpu: float) -> ResourceLedger:
    return ResourceLedger(
        cpu_seconds=time.process_time() - started_cpu,
        wall_seconds=time.perf_counter() - started_wall,
    )


@dataclass(frozen=True, slots=True, init=False)
class ExactStateSampler:
    """Oracle control that samples an explicitly supplied exact statevector.

    This is a methodological upper control, not evidence of hardware quantum advantage.
    """

    statevector: NDArray[np.complex128]
    source_id: str
    kind: SamplerKind

    def __init__(
        self,
        statevector: NDArray[np.complex128],
        *,
        source_id: str = "exact-state",
        kind: SamplerKind = SamplerKind.ORACLE,
    ) -> None:
        state = np.asarray(statevector)
        if state.ndim != 1:
            raise ValidationError(
                "statevector must be one-dimensional",
                shape=list(state.shape),
            )
        copied = np.array(state, dtype=np.complex128, copy=True)
        copied.flags.writeable = False
        object.__setattr__(self, "statevector", copied)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "kind", kind)

    def sample(
        self, problem: AuditProblem, budget: AuditBudget, seed: int
    ) -> SampleBatch:
        expected_dimension = 1 << problem.hamiltonian.n_qubits
        if self.statevector.size != expected_dimension:
            raise ValidationError(
                "statevector dimension does not match the audit Hamiltonian",
                statevector_dimension=int(self.statevector.size),
                expected_dimension=expected_dimension,
                n_qubits=problem.hamiltonian.n_qubits,
            )
        started_wall = time.perf_counter()
        started_cpu = time.process_time()
        configurations = sample_configurations(self.statevector, budget.shots, seed)
        ledger = _timed_ledger(started_wall, started_cpu)
        return SampleBatch(
            source_id=self.source_id,
            kind=self.kind,
            configurations=configurations,
            n_qubits=problem.hamiltonian.n_qubits,
            shots_requested=budget.shots,
            seed=seed,
            ledger=ledger,
            metadata={
                "control_role": "exact_distribution_oracle",
                "statevector_norm": float(
                    np.vdot(self.statevector, self.statevector).real
                ),
            },
        )


@dataclass(frozen=True, slots=True)
class UniformSectorSampler:
    """Negative control that samples the target particle sector uniformly."""

    source_id: str = "uniform-sector"
    kind: SamplerKind = SamplerKind.NEGATIVE_CONTROL

    def sample(
        self, problem: AuditProblem, budget: AuditBudget, seed: int
    ) -> SampleBatch:
        started_wall = time.perf_counter()
        started_cpu = time.process_time()
        sector = sector_configurations(
            problem.hamiltonian.n_qubits,
            problem.n_alpha,
            problem.n_beta,
        )
        rng = np.random.default_rng(seed)
        configurations = np.asarray(
            rng.choice(sector, size=budget.shots, replace=True),
            dtype=np.int64,
        )
        ledger = _timed_ledger(started_wall, started_cpu)
        return SampleBatch(
            source_id=self.source_id,
            kind=self.kind,
            configurations=configurations,
            n_qubits=problem.hamiltonian.n_qubits,
            shots_requested=budget.shots,
            seed=seed,
            ledger=ledger,
            metadata={
                "control_role": "uniform_sector_negative_control",
                "sector_dimension": int(sector.size),
            },
        )


def _configuration_from_occupations(
    n_qubits: int,
    occupied_alpha: tuple[int, ...],
    occupied_beta: tuple[int, ...],
) -> int:
    configuration = 0
    for spatial_orbital in occupied_alpha:
        qubit = 2 * spatial_orbital
        configuration |= 1 << (n_qubits - 1 - qubit)
    for spatial_orbital in occupied_beta:
        qubit = 2 * spatial_orbital + 1
        configuration |= 1 << (n_qubits - 1 - qubit)
    return configuration


def _replace_occupations(
    occupied: tuple[int, ...],
    removed: tuple[int, ...],
    added: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(sorted((set(occupied) - set(removed)) | set(added)))


def _hf_excitation_support(
    problem: AuditProblem,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    n_orbitals = problem.hamiltonian.n_qubits // 2
    occupied_alpha = tuple(range(problem.n_alpha))
    occupied_beta = tuple(range(problem.n_beta))
    virtual_alpha = tuple(range(problem.n_alpha, n_orbitals))
    virtual_beta = tuple(range(problem.n_beta, n_orbitals))
    ranked: dict[int, int] = {}

    def add(alpha: tuple[int, ...], beta: tuple[int, ...], rank: int) -> None:
        configuration = _configuration_from_occupations(
            problem.hamiltonian.n_qubits,
            alpha,
            beta,
        )
        ranked[configuration] = min(rank, ranked.get(configuration, rank))

    add(occupied_alpha, occupied_beta, 0)

    for removed in occupied_alpha:
        for added in virtual_alpha:
            add(
                _replace_occupations(occupied_alpha, (removed,), (added,)),
                occupied_beta,
                1,
            )
    for removed in occupied_beta:
        for added in virtual_beta:
            add(
                occupied_alpha,
                _replace_occupations(occupied_beta, (removed,), (added,)),
                1,
            )

    for removed_pair in combinations(occupied_alpha, 2):
        for added_pair in combinations(virtual_alpha, 2):
            add(
                _replace_occupations(occupied_alpha, removed_pair, added_pair),
                occupied_beta,
                2,
            )
    for removed_pair in combinations(occupied_beta, 2):
        for added_pair in combinations(virtual_beta, 2):
            add(
                occupied_alpha,
                _replace_occupations(occupied_beta, removed_pair, added_pair),
                2,
            )
    for removed_alpha in occupied_alpha:
        for added_alpha in virtual_alpha:
            for removed_beta in occupied_beta:
                for added_beta in virtual_beta:
                    add(
                        _replace_occupations(
                            occupied_alpha,
                            (removed_alpha,),
                            (added_alpha,),
                        ),
                        _replace_occupations(
                            occupied_beta,
                            (removed_beta,),
                            (added_beta,),
                        ),
                        2,
                    )

    ordered = sorted(ranked.items())
    configurations = np.array([item[0] for item in ordered], dtype=np.int64)
    ranks = np.array([item[1] for item in ordered], dtype=np.int64)
    return configurations, ranks


@dataclass(frozen=True, slots=True)
class HartreeFockExcitationSampler:
    """Classical proposal over the HF reference and symmetry-valid singles/doubles."""

    rank_weights: tuple[float, float, float] = (1.0, 0.25, 0.0625)
    source_id: str = "hf-excitations"
    kind: SamplerKind = SamplerKind.CLASSICAL_EXCITATION

    def __post_init__(self) -> None:
        weights = tuple(float(value) for value in self.rank_weights)
        object.__setattr__(self, "rank_weights", weights)
        if len(weights) != 3 or any(
            not math.isfinite(value) or value <= 0.0 for value in weights
        ):
            raise ValidationError(
                "rank_weights must contain three finite positive values for ranks 0, 1, and 2",
                rank_weights=list(weights),
            )

    def sample(
        self, problem: AuditProblem, budget: AuditBudget, seed: int
    ) -> SampleBatch:
        started_wall = time.perf_counter()
        started_cpu = time.process_time()
        support, ranks = _hf_excitation_support(problem)
        unnormalized = np.array(
            [self.rank_weights[int(rank)] for rank in ranks], dtype=float
        )
        probabilities = unnormalized / unnormalized.sum()
        rng = np.random.default_rng(seed)
        configurations = np.asarray(
            rng.choice(support, size=budget.shots, replace=True, p=probabilities),
            dtype=np.int64,
        )
        ledger = _timed_ledger(started_wall, started_cpu)
        reference = _configuration_from_occupations(
            problem.hamiltonian.n_qubits,
            tuple(range(problem.n_alpha)),
            tuple(range(problem.n_beta)),
        )
        return SampleBatch(
            source_id=self.source_id,
            kind=self.kind,
            configurations=configurations,
            n_qubits=problem.hamiltonian.n_qubits,
            shots_requested=budget.shots,
            seed=seed,
            ledger=ledger,
            metadata={
                "hartree_fock_configuration": reference,
                "rank_weights": self.rank_weights,
                "support_dimension": int(support.size),
                "maximum_excitation_rank": 2,
            },
        )


@dataclass(frozen=True, slots=True, init=False)
class RecordedSampleSampler:
    """Replay an immutable raw sample artifact from a simulator or QPU."""

    source_id: str
    kind: SamplerKind
    configurations: NDArray[np.int64]
    ledger: ResourceLedger
    metadata: Mapping[str, Any]

    def __init__(
        self,
        *,
        source_id: str,
        kind: SamplerKind,
        configurations: NDArray[np.int64],
        ledger: ResourceLedger | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        raw = np.asarray(configurations)
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise ValidationError(
                "recorded configurations must be a one-dimensional integer draw",
                shape=list(raw.shape),
                dtype=str(raw.dtype),
            )
        copied = np.array(raw, dtype=np.int64, copy=True)
        copied.flags.writeable = False
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "configurations", copied)
        object.__setattr__(self, "ledger", ledger or ResourceLedger())
        object.__setattr__(self, "metadata", dict(metadata or {}))

    def sample(
        self, problem: AuditProblem, budget: AuditBudget, seed: int
    ) -> SampleBatch:
        if budget.shots != self.configurations.size:
            raise ValidationError(
                "recorded sample artifact does not match the requested shot budget",
                recorded_shots=int(self.configurations.size),
                requested_shots=budget.shots,
            )
        return SampleBatch(
            source_id=self.source_id,
            kind=self.kind,
            configurations=self.configurations,
            n_qubits=problem.hamiltonian.n_qubits,
            shots_requested=budget.shots,
            seed=seed,
            ledger=self.ledger,
            metadata=self.metadata,
        )


__all__ = [
    "DeterminantSampler",
    "ExactStateSampler",
    "HartreeFockExcitationSampler",
    "RecordedSampleSampler",
    "UniformSectorSampler",
]

"""Frozen classical post-processing for determinant sampler comparisons."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from qdr_compute_engine.audit.models import (
    AuditBudget,
    AuditProblem,
    PostprocessResult,
    ResourceLedger,
    SampleBatch,
)
from qdr_compute_engine.solvers.sqd import (
    VARIATIONAL_TOLERANCE,
    run_sqd_campaign,
    sector_configurations,
    sector_mask,
    solve_sqd,
)
from qdr_core.errors import ValidationError


class AuditPostprocessor(Protocol):
    """Classical recovery/diagonalization held fixed across sampler arms."""

    @property
    def postprocessor_id(self) -> str:
        """Stable postprocessor identity stored with every result."""

    @property
    def dimension_policy(self) -> str:
        """Whether returned dimensions are exact requests or measured outcomes."""

    def process(
        self,
        problem: AuditProblem,
        batch: SampleBatch,
        budget: AuditBudget,
    ) -> tuple[PostprocessResult, ...]:
        """Post-process one raw draw over the preregistered dimension ladder."""


@dataclass(frozen=True, slots=True)
class QDRSubspacePostprocessor:
    """QDR's fixed ranking and nested selected-subspace diagonalization."""

    postprocessor_id: str = "qdr-fixed-subspace"
    dimension_policy: str = "exact"

    def process(
        self,
        problem: AuditProblem,
        batch: SampleBatch,
        budget: AuditBudget,
    ) -> tuple[PostprocessResult, ...]:
        if batch.n_qubits != problem.hamiltonian.n_qubits:
            raise ValidationError(
                "sample register does not match the frozen audit Hamiltonian",
                sampled_qubits=batch.n_qubits,
                problem_qubits=problem.hamiltonian.n_qubits,
            )
        if batch.shots_requested != budget.shots:
            raise ValidationError(
                "sample batch does not match the frozen audit budget",
                sampled_shots=batch.shots_requested,
                budget_shots=budget.shots,
            )

        started_wall = time.perf_counter()
        started_cpu = time.process_time()
        campaign = run_sqd_campaign(
            problem.hamiltonian,
            problem.n_alpha,
            problem.n_beta,
            configurations=batch.configurations,
            dimensions=budget.subspace_dimensions,
            reference_energy=problem.reference_energy,
        )
        elapsed_wall = time.perf_counter() - started_wall
        elapsed_cpu = time.process_time() - started_cpu
        ledger = ResourceLedger(
            cpu_seconds=elapsed_cpu,
            wall_seconds=elapsed_wall,
            diagonalization_seconds=elapsed_wall,
        )

        return tuple(
            PostprocessResult(
                problem_hash=problem.identity_hash,
                subspace_dimension=step.subspace_dimension,
                energy=step.energy,
                configurations=step.configurations,
                is_exact=step.is_exact,
                n_samples=campaign.n_samples,
                n_samples_in_sector=campaign.n_samples_in_sector,
                ledger=ledger,
                metadata={
                    "postprocessor_id": self.postprocessor_id,
                    "requested_dimension": step.requested_dimension,
                    "sector_dimension": campaign.sector_dimension,
                    "energy_change": step.energy_change,
                    "timing_scope": "ranking_and_nested_diagonalization_campaign",
                },
            )
            for step in campaign.steps
        )


@dataclass(frozen=True, slots=True)
class CanonicalSectorCompletionPostprocessor:
    """Frequency-rank observed determinants, then fill missing rungs canonically.

    The same deterministic recovery is applied to every sampler arm. It exists for
    methodological audits whose locked ladder is wider than a sampler's observed support;
    metadata separates sampled from completed determinants so recovery is never presented
    as discovery by the sampler.
    """

    postprocessor_id: str = "qdr-frequency-rank-canonical-sector-completion"
    dimension_policy: str = "exact"

    def process(
        self,
        problem: AuditProblem,
        batch: SampleBatch,
        budget: AuditBudget,
    ) -> tuple[PostprocessResult, ...]:
        if batch.n_qubits != problem.hamiltonian.n_qubits:
            raise ValidationError(
                "sample register does not match the frozen audit Hamiltonian",
                sampled_qubits=batch.n_qubits,
                problem_qubits=problem.hamiltonian.n_qubits,
            )
        if batch.shots_requested != budget.shots:
            raise ValidationError(
                "sample batch does not match the frozen audit budget",
                sampled_shots=batch.shots_requested,
                budget_shots=budget.shots,
            )

        drawn, mask = sector_mask(
            batch.configurations,
            problem.hamiltonian.n_qubits,
            problem.n_alpha,
            problem.n_beta,
        )
        in_sector = drawn[mask]
        sampled, counts = np.unique(in_sector, return_counts=True)
        if sampled.size:
            frequency_order = np.lexsort((sampled, -counts))
            sampled_ranked = sampled[frequency_order]
        else:
            sampled_ranked = sampled
        full_sector = sector_configurations(
            problem.hamiltonian.n_qubits,
            problem.n_alpha,
            problem.n_beta,
        )
        sampled_set = set(int(value) for value in sampled)
        completion = np.asarray(
            [value for value in full_sector if int(value) not in sampled_set],
            dtype=np.int64,
        )
        ranked = np.concatenate((sampled_ranked, completion))
        if budget.subspace_dimensions[-1] > ranked.size:
            raise ValidationError(
                "locked subspace ladder exceeds the target particle sector",
                requested_dimension=budget.subspace_dimensions[-1],
                sector_dimension=int(ranked.size),
            )

        started_wall = time.perf_counter()
        started_cpu = time.process_time()
        step_payloads: list[tuple[int, float, tuple[int, ...], bool, float | None]] = []
        previous_energy: float | None = None
        for dimension in budget.subspace_dimensions:
            selected = np.sort(ranked[:dimension])
            outcome, _ = solve_sqd(
                problem.hamiltonian,
                problem.n_alpha,
                problem.n_beta,
                configurations=selected,
                reference_energy=problem.reference_energy,
            )
            change = (
                None
                if previous_energy is None
                else abs(outcome.energy - previous_energy)
            )
            if (
                previous_energy is not None
                and outcome.energy > previous_energy + VARIATIONAL_TOLERANCE
            ):
                raise ValidationError(
                    "completed SQD energy increased as the determinant space grew",
                    previous_energy=previous_energy,
                    energy=outcome.energy,
                    dimension=dimension,
                )
            configurations = tuple(int(value) for value in selected)
            step_payloads.append(
                (
                    dimension,
                    outcome.energy,
                    configurations,
                    outcome.is_exact,
                    change,
                )
            )
            previous_energy = outcome.energy

        elapsed_wall = time.perf_counter() - started_wall
        elapsed_cpu = time.process_time() - started_cpu
        ledger = ResourceLedger(
            cpu_seconds=elapsed_cpu,
            wall_seconds=elapsed_wall,
            recovery_seconds=elapsed_wall,
            diagonalization_seconds=elapsed_wall,
        )
        return tuple(
            PostprocessResult(
                problem_hash=problem.identity_hash,
                subspace_dimension=dimension,
                energy=energy,
                configurations=configurations,
                is_exact=is_exact,
                n_samples=int(drawn.size),
                n_samples_in_sector=int(in_sector.size),
                ledger=ledger,
                metadata={
                    "postprocessor_id": self.postprocessor_id,
                    "requested_dimension": dimension,
                    "sector_dimension": int(full_sector.size),
                    "sampled_distinct_configurations": int(sampled.size),
                    "completed_configurations": sum(
                        configuration not in sampled_set
                        for configuration in configurations
                    ),
                    "completion_rule": (
                        "sample_frequency_then_configuration_index_then_"
                        "canonical_unobserved_sector_index"
                    ),
                    "energy_change": energy_change,
                    "timing_scope": "fixed_recovery_and_nested_diagonalization",
                },
            )
            for dimension, energy, configurations, is_exact, energy_change in step_payloads
        )


__all__ = [
    "AuditPostprocessor",
    "CanonicalSectorCompletionPostprocessor",
    "QDRSubspacePostprocessor",
]

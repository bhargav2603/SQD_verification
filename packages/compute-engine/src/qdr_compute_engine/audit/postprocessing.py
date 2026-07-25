"""Frozen classical post-processing for determinant sampler comparisons."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from qdr_compute_engine.audit.models import (
    AuditBudget,
    AuditProblem,
    PostprocessResult,
    ResourceLedger,
    SampleBatch,
)
from qdr_compute_engine.solvers.sqd import run_sqd_campaign
from qdr_core.errors import ValidationError


class AuditPostprocessor(Protocol):
    """Classical recovery/diagonalization held fixed across sampler arms."""

    postprocessor_id: str

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


__all__ = ["AuditPostprocessor", "QDRSubspacePostprocessor"]

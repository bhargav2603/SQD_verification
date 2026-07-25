"""Paired, seeded execution of determinant sampler audits."""

from __future__ import annotations

from collections.abc import Sequence

from qdr_compute_engine.audit.models import (
    AuditBudget,
    AuditObservation,
    AuditProblem,
    ResourceLedger,
)
from qdr_compute_engine.audit.postprocessing import AuditPostprocessor
from qdr_compute_engine.audit.samplers import DeterminantSampler
from qdr_compute_engine.solvers.sqd import VARIATIONAL_TOLERANCE
from qdr_core.errors import ValidationError


def _sum_optional_float(left: float | None, right: float | None) -> float | None:
    if left is None and right is None:
        return None
    return (left or 0.0) + (right or 0.0)


def _sum_optional_int(left: int | None, right: int | None) -> int | None:
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)


def _combine_ledgers(
    sample: ResourceLedger, postprocess: ResourceLedger
) -> ResourceLedger:
    peak_values = [
        value
        for value in (sample.peak_memory_bytes, postprocess.peak_memory_bytes)
        if value is not None
    ]
    return ResourceLedger(
        circuit_evaluations=_sum_optional_int(
            sample.circuit_evaluations,
            postprocess.circuit_evaluations,
        ),
        qpu_seconds=_sum_optional_float(sample.qpu_seconds, postprocess.qpu_seconds),
        cpu_seconds=_sum_optional_float(sample.cpu_seconds, postprocess.cpu_seconds),
        wall_seconds=_sum_optional_float(sample.wall_seconds, postprocess.wall_seconds),
        preprocessing_seconds=_sum_optional_float(
            sample.preprocessing_seconds,
            postprocess.preprocessing_seconds,
        ),
        recovery_seconds=_sum_optional_float(
            sample.recovery_seconds,
            postprocess.recovery_seconds,
        ),
        diagonalization_seconds=_sum_optional_float(
            sample.diagonalization_seconds,
            postprocess.diagonalization_seconds,
        ),
        peak_memory_bytes=max(peak_values) if peak_values else None,
    )


def run_audit_cell(
    problem: AuditProblem,
    sampler: DeterminantSampler,
    budget: AuditBudget,
    *,
    seed: int,
    postprocessor: AuditPostprocessor,
) -> tuple[AuditObservation, ...]:
    """Run one frozen ``(problem, sampler, budget, seed)`` cell."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValidationError("seed must be an integer", seed=seed)
    batch = sampler.sample(problem, budget, seed)
    if batch.source_id != sampler.source_id or batch.kind is not sampler.kind:
        raise ValidationError(
            "sampler returned evidence under a different source identity",
            expected_source_id=sampler.source_id,
            returned_source_id=batch.source_id,
            expected_kind=sampler.kind.value,
            returned_kind=batch.kind.value,
        )
    if batch.seed != seed:
        raise ValidationError(
            "sampler returned evidence under a different seed",
            expected_seed=seed,
            returned_seed=batch.seed,
        )

    results = postprocessor.process(problem, batch, budget)
    dimensions = tuple(result.subspace_dimension for result in results)
    dimension_policy = getattr(postprocessor, "dimension_policy", "exact")
    exact_dimensions_invalid = (
        dimension_policy == "exact" and dimensions != budget.subspace_dimensions
    )
    measured_dimensions_invalid = dimension_policy == "measured" and (
        len(dimensions) != len(budget.subspace_dimensions)
        or any(
            right <= left
            for left, right in zip(dimensions, dimensions[1:], strict=False)
        )
    )
    if dimension_policy not in {"exact", "measured"}:
        raise ValidationError(
            "postprocessor has an unknown dimension policy",
            dimension_policy=dimension_policy,
        )
    if exact_dimensions_invalid or measured_dimensions_invalid:
        raise ValidationError(
            "postprocessor did not return the frozen subspace ladder",
            expected_dimensions=list(budget.subspace_dimensions),
            returned_dimensions=list(dimensions),
            dimension_policy=dimension_policy,
        )

    observations: list[AuditObservation] = []
    for result in results:
        if result.problem_hash != problem.identity_hash:
            raise ValidationError(
                "postprocessor result problem hash does not match the frozen audit problem hash",
                expected_hash=problem.identity_hash,
                returned_hash=result.problem_hash,
            )
        if result.n_samples != batch.shots_returned:
            raise ValidationError(
                "postprocessor sample count does not match the raw evidence",
                raw_samples=batch.shots_returned,
                postprocessed_samples=result.n_samples,
            )
        if result.energy < problem.reference_energy - VARIATIONAL_TOLERANCE:
            raise ValidationError(
                "postprocessed energy lies below the known reference energy",
                energy=result.energy,
                reference_energy=problem.reference_energy,
                tolerance=VARIATIONAL_TOLERANCE,
            )
        retention = (
            1.0
            if result.n_samples == 0
            else result.n_samples_in_sector / result.n_samples
        )
        observations.append(
            AuditObservation(
                problem_id=problem.problem_id,
                problem_hash=problem.identity_hash,
                source_id=batch.source_id,
                kind=batch.kind,
                split=problem.split,
                seed=seed,
                shots=budget.shots,
                subspace_dimension=result.subspace_dimension,
                energy=result.energy,
                reference_energy=problem.reference_energy,
                absolute_error=abs(result.energy - problem.reference_energy),
                sector_retention=retention,
                configurations=result.configurations,
                ledger=_combine_ledgers(batch.ledger, result.ledger),
                metadata={
                    "postprocessor_id": postprocessor.postprocessor_id,
                    "sampler": dict(batch.metadata),
                    "postprocessing": dict(result.metadata),
                },
            )
        )
    return tuple(observations)


def run_paired_audit(
    *,
    problems: Sequence[AuditProblem],
    samplers: Sequence[DeterminantSampler],
    budgets: Sequence[AuditBudget],
    seeds: Sequence[int],
    postprocessor: AuditPostprocessor,
) -> tuple[AuditObservation, ...]:
    """Run the full Cartesian design in deterministic problem-major order."""

    if not problems:
        raise ValidationError("paired audit requires at least one problem")
    if not samplers:
        raise ValidationError("paired audit requires at least one sampler")
    if not budgets:
        raise ValidationError("paired audit requires at least one budget")
    if not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        raise ValidationError(
            "paired audit requires one or more integer seeds", seeds=list(seeds)
        )

    source_ids = [sampler.source_id for sampler in samplers]
    if len(set(source_ids)) != len(source_ids):
        raise ValidationError(
            "duplicate sampler source ids would make paired observations ambiguous",
            source_ids=source_ids,
        )
    problem_ids = [problem.problem_id for problem in problems]
    if len(set(problem_ids)) != len(problem_ids):
        raise ValidationError(
            "duplicate problem ids would make paired observations ambiguous",
            problem_ids=problem_ids,
        )

    observations: list[AuditObservation] = []
    for problem in problems:
        for budget in budgets:
            for seed in seeds:
                for sampler in samplers:
                    observations.extend(
                        run_audit_cell(
                            problem,
                            sampler,
                            budget,
                            seed=seed,
                            postprocessor=postprocessor,
                        )
                    )
    return tuple(observations)


__all__ = ["run_audit_cell", "run_paired_audit"]

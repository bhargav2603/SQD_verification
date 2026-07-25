"""Frozen post-processing and paired audit orchestration."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from qdr_compute_engine.active_space.active_space_selector import select_active_space
from qdr_compute_engine.audit.models import (
    AuditBudget,
    AuditProblem,
    DatasetSplit,
    PostprocessResult,
)
from qdr_compute_engine.audit.postprocessing import (
    AuditPostprocessor,
    CanonicalSectorCompletionPostprocessor,
    QDRSubspacePostprocessor,
)
from qdr_compute_engine.audit.runner import run_audit_cell, run_paired_audit
from qdr_compute_engine.audit.samplers import ExactStateSampler, UniformSectorSampler
from qdr_compute_engine.hamiltonian.fermionic_builder import build_fermionic_hamiltonian
from qdr_compute_engine.mappers.jordan_wigner import map_to_qubits
from qdr_compute_engine.types import IntegralSet
from qdr_core.errors import ValidationError


@pytest.fixture
def h2_problem(h2_integrals: IntegralSet, golden_h2: dict[str, Any]) -> AuditProblem:
    active_space = select_active_space(h2_integrals)
    hamiltonian = map_to_qubits(build_fermionic_hamiltonian(h2_integrals, active_space))
    return AuditProblem(
        problem_id="h2-equilibrium",
        hamiltonian=hamiltonian,
        n_alpha=1,
        n_beta=1,
        reference_energy=float(golden_h2["e_fci"]),
        split=DatasetSplit.HELD_OUT,
    )


def _spread_h2_state() -> np.ndarray:
    state = np.zeros(16, dtype=np.complex128)
    state[[3, 6, 9, 12]] = 0.5
    return state


def test_qdr_postprocessor_records_raw_sample_retention_and_nested_spaces(
    h2_problem: AuditProblem,
) -> None:
    budget = AuditBudget(shots=6, subspace_dimensions=(1, 2, 4))
    batch = ExactStateSampler(_spread_h2_state()).sample(h2_problem, budget, seed=7)
    # Pin all four sector determinants plus one wrong-sector draw and one duplicate.
    batch = replace(
        batch,
        configurations=np.array([12, 12, 0, 3, 9, 6], dtype=np.int64),
    )

    results = QDRSubspacePostprocessor().process(h2_problem, batch, budget)

    assert [result.subspace_dimension for result in results] == [1, 2, 4]
    assert all(result.problem_hash == h2_problem.identity_hash for result in results)
    assert all(result.n_samples == 6 for result in results)
    assert all(result.n_samples_in_sector == 5 for result in results)
    assert set(results[0].configurations) < set(results[1].configurations)
    assert set(results[1].configurations) < set(results[2].configurations)


def test_sector_completion_preserves_sample_ranking_and_fills_the_locked_ladder(
    h2_problem: AuditProblem,
) -> None:
    budget = AuditBudget(shots=4, subspace_dimensions=(1, 2, 4))
    batch = ExactStateSampler(_spread_h2_state()).sample(h2_problem, budget, seed=7)
    batch = replace(
        batch,
        configurations=np.array([12, 12, 12, 12], dtype=np.int64),
    )

    results = CanonicalSectorCompletionPostprocessor().process(
        h2_problem,
        batch,
        budget,
    )

    assert [result.subspace_dimension for result in results] == [1, 2, 4]
    assert results[0].configurations == (12,)
    assert all(result.n_samples == 4 for result in results)
    assert all(result.n_samples_in_sector == 4 for result in results)
    assert results[-1].metadata["sampled_distinct_configurations"] == 1
    assert results[-1].metadata["completed_configurations"] == 3


def test_runner_creates_the_same_problem_budget_seed_cells_for_every_sampler(
    h2_problem: AuditProblem,
) -> None:
    observations = run_paired_audit(
        problems=(h2_problem,),
        samplers=(
            UniformSectorSampler(),
            ExactStateSampler(_spread_h2_state()),
        ),
        budgets=(AuditBudget(shots=512, subspace_dimensions=(1, 2, 4)),),
        seeds=(3, 7),
        postprocessor=QDRSubspacePostprocessor(),
    )

    cells = {
        (observation.source_id, observation.seed, observation.shots)
        for observation in observations
        if observation.subspace_dimension == 4
    }
    assert cells == {
        ("uniform-sector", 3, 512),
        ("uniform-sector", 7, 512),
        ("exact-state", 3, 512),
        ("exact-state", 7, 512),
    }
    assert len(observations) == 2 * 2 * 3


def test_runner_order_is_deterministic_and_problem_major(
    h2_problem: AuditProblem,
) -> None:
    samplers = (UniformSectorSampler(), ExactStateSampler(_spread_h2_state()))
    budget = AuditBudget(shots=256, subspace_dimensions=(1, 2, 4))

    observations = run_paired_audit(
        problems=(h2_problem,),
        samplers=samplers,
        budgets=(budget,),
        seeds=(5,),
        postprocessor=QDRSubspacePostprocessor(),
    )

    assert [(item.source_id, item.subspace_dimension) for item in observations] == [
        ("uniform-sector", 1),
        ("uniform-sector", 2),
        ("uniform-sector", 4),
        ("exact-state", 1),
        ("exact-state", 2),
        ("exact-state", 4),
    ]


def test_runner_rejects_duplicate_sampler_ids(h2_problem: AuditProblem) -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        run_paired_audit(
            problems=(h2_problem,),
            samplers=(UniformSectorSampler(), UniformSectorSampler()),
            budgets=(AuditBudget(shots=64, subspace_dimensions=(1, 2, 4)),),
            seeds=(1,),
            postprocessor=QDRSubspacePostprocessor(),
        )


class _WrongProblemPostprocessor:
    def process(
        self,
        problem: AuditProblem,
        batch: Any,
        budget: AuditBudget,
    ) -> tuple[PostprocessResult, ...]:
        return (
            PostprocessResult(
                problem_hash="sha256:" + "0" * 64,
                subspace_dimension=1,
                energy=problem.reference_energy + 0.1,
                configurations=(12,),
                is_exact=False,
                n_samples=batch.shots_returned,
                n_samples_in_sector=batch.shots_returned,
            ),
        )


def test_runner_rejects_postprocessing_for_a_different_problem(
    h2_problem: AuditProblem,
) -> None:
    with pytest.raises(ValidationError, match="problem hash"):
        run_audit_cell(
            h2_problem,
            UniformSectorSampler(),
            AuditBudget(shots=32, subspace_dimensions=(1,)),
            seed=2,
            postprocessor=_WrongProblemPostprocessor(),
        )


class _BelowReferencePostprocessor:
    def process(
        self,
        problem: AuditProblem,
        batch: Any,
        budget: AuditBudget,
    ) -> tuple[PostprocessResult, ...]:
        return (
            PostprocessResult(
                problem_hash=problem.identity_hash,
                subspace_dimension=1,
                energy=problem.reference_energy - 1.0,
                configurations=(12,),
                is_exact=False,
                n_samples=batch.shots_returned,
                n_samples_in_sector=batch.shots_returned,
            ),
        )


def test_runner_rejects_a_result_below_the_known_reference(
    h2_problem: AuditProblem,
) -> None:
    with pytest.raises(ValidationError, match="below"):
        run_audit_cell(
            h2_problem,
            UniformSectorSampler(),
            AuditBudget(shots=32, subspace_dimensions=(1,)),
            seed=2,
            postprocessor=_BelowReferencePostprocessor(),
        )


def test_observations_combine_sampler_and_postprocessing_costs(
    h2_problem: AuditProblem,
) -> None:
    observations = run_audit_cell(
        h2_problem,
        UniformSectorSampler(),
        AuditBudget(shots=128, subspace_dimensions=(1, 2, 4)),
        seed=13,
        postprocessor=QDRSubspacePostprocessor(),
    )

    assert all(item.ledger.cpu_seconds is not None for item in observations)
    assert all(item.ledger.wall_seconds is not None for item in observations)
    assert all(item.ledger.diagonalization_seconds is not None for item in observations)
    assert all(
        item.absolute_error == abs(item.energy - item.reference_energy)
        for item in observations
    )
    assert all(0.0 <= item.sector_retention <= 1.0 for item in observations)


def test_postprocessor_protocol_accepts_qdr_implementation() -> None:
    postprocessor: AuditPostprocessor = QDRSubspacePostprocessor()

    assert postprocessor.postprocessor_id == "qdr-fixed-subspace"


def test_postprocess_metadata_is_immutable(h2_problem: AuditProblem) -> None:
    observations = run_audit_cell(
        h2_problem,
        UniformSectorSampler(),
        AuditBudget(shots=64, subspace_dimensions=(1, 2, 4)),
        seed=17,
        postprocessor=QDRSubspacePostprocessor(),
    )

    with pytest.raises(TypeError):
        observations[0].metadata["changed"] = True  # type: ignore[index]


def test_runner_rejects_an_empty_seed_set(h2_problem: AuditProblem) -> None:
    with pytest.raises(ValidationError, match="seed"):
        run_paired_audit(
            problems=(h2_problem,),
            samplers=(UniformSectorSampler(),),
            budgets=(AuditBudget(shots=64, subspace_dimensions=(1, 2, 4)),),
            seeds=(),
            postprocessor=QDRSubspacePostprocessor(),
        )


def test_runner_refuses_non_integer_seeds(h2_problem: AuditProblem) -> None:
    with pytest.raises(ValidationError, match="seed"):
        run_paired_audit(
            problems=(h2_problem,),
            samplers=(UniformSectorSampler(),),
            budgets=(AuditBudget(shots=64, subspace_dimensions=(1, 2, 4)),),
            seeds=(True,),
            postprocessor=QDRSubspacePostprocessor(),
        )

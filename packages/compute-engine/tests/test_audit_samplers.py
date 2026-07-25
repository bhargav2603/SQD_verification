"""Determinant-sampler controls for the counterfactual SQD audit."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from qdr_compute_engine.active_space.active_space_selector import select_active_space
from qdr_compute_engine.audit.models import (
    AuditBudget,
    AuditProblem,
    DatasetSplit,
    ResourceLedger,
    SamplerKind,
)
from qdr_compute_engine.audit.samplers import (
    ExactStateSampler,
    HartreeFockExcitationSampler,
    RecordedSampleSampler,
    UniformSectorSampler,
)
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
        split=DatasetSplit.DEVELOPMENT,
    )


@pytest.fixture
def budget() -> AuditBudget:
    return AuditBudget(shots=256, subspace_dimensions=(1, 2, 4))


def _spread_h2_state() -> np.ndarray:
    state = np.zeros(16, dtype=np.complex128)
    state[[3, 6, 9, 12]] = np.sqrt([0.4, 0.3, 0.2, 0.1])
    return state


def test_exact_state_sampler_replays_the_same_draw_for_the_same_seed(
    h2_problem: AuditProblem, budget: AuditBudget
) -> None:
    sampler = ExactStateSampler(_spread_h2_state())

    first = sampler.sample(h2_problem, budget, seed=11)
    second = sampler.sample(h2_problem, budget, seed=11)

    assert first.configurations.tolist() == second.configurations.tolist()
    assert first.kind is SamplerKind.ORACLE
    assert first.source_id == "exact-state"


def test_exact_state_sampler_tracks_the_supplied_distribution(
    h2_problem: AuditProblem,
) -> None:
    batch = ExactStateSampler(_spread_h2_state()).sample(
        h2_problem,
        AuditBudget(shots=20_000, subspace_dimensions=(1, 2, 4)),
        seed=19,
    )
    frequencies = np.bincount(batch.configurations, minlength=16) / batch.shots_returned

    assert frequencies[[3, 6, 9, 12]] == pytest.approx([0.4, 0.3, 0.2, 0.1], abs=0.015)


def test_exact_state_sampler_rejects_wrong_statevector_dimension(
    h2_problem: AuditProblem, budget: AuditBudget
) -> None:
    with pytest.raises(ValidationError, match="statevector"):
        ExactStateSampler(np.ones(8)).sample(h2_problem, budget, seed=0)


def test_uniform_sector_sampler_never_emits_the_wrong_particle_sector(
    h2_problem: AuditProblem, budget: AuditBudget
) -> None:
    batch = UniformSectorSampler().sample(h2_problem, budget, seed=5)

    assert set(batch.configurations).issubset({3, 6, 9, 12})
    assert batch.kind is SamplerKind.NEGATIVE_CONTROL


def test_hartree_fock_excitation_sampler_covers_h2_reference_and_excitations(
    h2_problem: AuditProblem,
) -> None:
    batch = HartreeFockExcitationSampler(rank_weights=(1.0, 1.0, 1.0)).sample(
        h2_problem,
        AuditBudget(shots=20_000, subspace_dimensions=(1, 2, 4)),
        seed=23,
    )

    assert set(batch.configurations) == {3, 6, 9, 12}
    assert batch.metadata["hartree_fock_configuration"] == 12
    assert batch.kind is SamplerKind.CLASSICAL_EXCITATION


def test_hartree_fock_rank_weights_change_the_draw_without_changing_support(
    h2_problem: AuditProblem,
) -> None:
    batch = HartreeFockExcitationSampler(rank_weights=(100.0, 1.0, 1.0)).sample(
        h2_problem,
        AuditBudget(shots=20_000, subspace_dimensions=(1, 2, 4)),
        seed=29,
    )
    reference_fraction = np.mean(batch.configurations == 12)

    assert set(batch.configurations) == {3, 6, 9, 12}
    assert reference_fraction > 0.9


def test_hartree_fock_sampler_rejects_non_positive_rank_weights() -> None:
    with pytest.raises(ValidationError, match="rank_weights"):
        HartreeFockExcitationSampler(rank_weights=(1.0, 0.0, 1.0))


def test_recorded_sampler_retains_wrong_sector_values_and_duplicates(
    h2_problem: AuditProblem,
) -> None:
    sampler = RecordedSampleSampler(
        source_id="ibm-free-run",
        kind=SamplerKind.HARDWARE_QUANTUM,
        configurations=np.array([12, 12, 0, 3], dtype=np.int64),
        ledger=ResourceLedger(qpu_seconds=0.025, circuit_evaluations=1),
        metadata={"backend": "example-device", "job_id": "job-123"},
    )

    batch = sampler.sample(
        h2_problem,
        AuditBudget(shots=4, subspace_dimensions=(1, 2)),
        seed=31,
    )

    assert batch.configurations.tolist() == [12, 12, 0, 3]
    assert batch.ledger.qpu_seconds == pytest.approx(0.025)
    assert batch.metadata["job_id"] == "job-123"


def test_recorded_sampler_refuses_a_budget_that_does_not_match_the_artifact(
    h2_problem: AuditProblem,
) -> None:
    sampler = RecordedSampleSampler(
        source_id="recorded",
        kind=SamplerKind.HARDWARE_QUANTUM,
        configurations=np.array([12, 3], dtype=np.int64),
    )

    with pytest.raises(ValidationError, match="shot budget"):
        sampler.sample(
            h2_problem,
            AuditBudget(shots=3, subspace_dimensions=(1, 2)),
            seed=0,
        )


@pytest.mark.parametrize(
    "sampler",
    [
        ExactStateSampler(_spread_h2_state()),
        UniformSectorSampler(),
        HartreeFockExcitationSampler(),
    ],
)
def test_generated_sampler_records_measured_cpu_and_wall_time(
    sampler: Any, h2_problem: AuditProblem, budget: AuditBudget
) -> None:
    batch = sampler.sample(h2_problem, budget, seed=37)

    assert batch.ledger.cpu_seconds is not None
    assert batch.ledger.cpu_seconds >= 0.0
    assert batch.ledger.wall_seconds is not None
    assert batch.ledger.wall_seconds >= 0.0


def test_generated_sampler_different_seed_changes_non_degenerate_draw(
    h2_problem: AuditProblem, budget: AuditBudget
) -> None:
    sampler = ExactStateSampler(_spread_h2_state())

    first = sampler.sample(h2_problem, budget, seed=41)
    second = sampler.sample(h2_problem, budget, seed=43)

    assert first.configurations.tolist() != second.configurations.tolist()

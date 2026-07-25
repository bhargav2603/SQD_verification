"""Contracts for sampler-audit evidence.

These tests pin the distinction between raw evidence and post-processed evidence.  In
particular, a :class:`SampleBatch` is allowed to contain duplicates and configurations
outside the target particle sector: those are facts about the sampler, not input errors
to be silently cleaned away.
"""

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
    SampleBatch,
    SamplerKind,
)
from qdr_compute_engine.hamiltonian.fermionic_builder import build_fermionic_hamiltonian
from qdr_compute_engine.mappers.jordan_wigner import map_to_qubits
from qdr_compute_engine.types import IntegralSet
from qdr_compute_engine.types import QubitHamiltonian
from qdr_core.errors import ValidationError


@pytest.fixture
def h2_hamiltonian(h2_integrals: IntegralSet) -> QubitHamiltonian:
    active_space = select_active_space(h2_integrals)
    return map_to_qubits(build_fermionic_hamiltonian(h2_integrals, active_space))


@pytest.fixture
def h2_problem(
    h2_hamiltonian: QubitHamiltonian, golden_h2: dict[str, Any]
) -> AuditProblem:
    return AuditProblem(
        problem_id="h2-equilibrium",
        hamiltonian=h2_hamiltonian,
        n_alpha=1,
        n_beta=1,
        reference_energy=float(golden_h2["e_fci"]),
        split=DatasetSplit.DEVELOPMENT,
        descriptors={"bond_length_angstrom": 0.735, "basis": "sto-3g"},
    )


def test_sample_batch_preserves_raw_duplicates_and_wrong_sector_values() -> None:
    raw = np.array([12, 12, 0, 3], dtype=np.int64)

    batch = SampleBatch(
        source_id="recorded-hardware",
        kind=SamplerKind.HARDWARE_QUANTUM,
        configurations=raw,
        n_qubits=4,
        shots_requested=4,
        seed=7,
    )

    assert batch.configurations.tolist() == [12, 12, 0, 3]
    assert batch.shots_returned == 4


def test_sample_batch_defensively_copies_and_freezes_raw_evidence() -> None:
    raw = np.array([12, 3], dtype=np.int64)
    batch = SampleBatch(
        source_id="recorded",
        kind=SamplerKind.HARDWARE_QUANTUM,
        configurations=raw,
        n_qubits=4,
        shots_requested=2,
        seed=1,
    )

    raw[0] = 0

    assert batch.configurations.tolist() == [12, 3]
    with pytest.raises(ValueError, match="read-only"):
        batch.configurations[0] = 0


def test_sample_batch_rejects_configuration_outside_register() -> None:
    with pytest.raises(ValidationError, match="configuration"):
        SampleBatch(
            source_id="bad",
            kind=SamplerKind.NEGATIVE_CONTROL,
            configurations=np.array([16], dtype=np.int64),
            n_qubits=4,
            shots_requested=1,
            seed=0,
        )


def test_sample_batch_requires_returned_shots_to_match_raw_draw() -> None:
    with pytest.raises(ValidationError, match="shots_returned"):
        SampleBatch(
            source_id="bad-accounting",
            kind=SamplerKind.HARDWARE_QUANTUM,
            configurations=np.array([12, 3], dtype=np.int64),
            n_qubits=4,
            shots_requested=3,
            shots_returned=3,
            seed=0,
        )


def test_resource_ledger_rejects_negative_measurements() -> None:
    with pytest.raises(ValidationError, match="cpu_seconds"):
        ResourceLedger(cpu_seconds=-0.1)


def test_resource_ledger_keeps_unmeasured_resources_unknown() -> None:
    ledger = ResourceLedger(cpu_seconds=0.2)

    assert ledger.cpu_seconds == pytest.approx(0.2)
    assert ledger.qpu_seconds is None
    assert ledger.circuit_evaluations is None
    assert ledger.to_dict()["qpu_seconds"] is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"shots": 0, "subspace_dimensions": (1,)}, "shots"),
        ({"shots": 4, "subspace_dimensions": ()}, "subspace_dimensions"),
        ({"shots": 4, "subspace_dimensions": (1, 1)}, "strictly increasing"),
        ({"shots": 4, "subspace_dimensions": (2, 1)}, "strictly increasing"),
        ({"shots": 4, "subspace_dimensions": (0, 1)}, "positive"),
    ],
)
def test_budget_rejects_non_physical_values(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        AuditBudget(**kwargs)


def test_problem_rejects_impossible_particle_sector(
    h2_hamiltonian: QubitHamiltonian,
) -> None:
    with pytest.raises(ValidationError, match="particle sector"):
        AuditProblem(
            problem_id="impossible",
            hamiltonian=h2_hamiltonian,
            n_alpha=3,
            n_beta=0,
            reference_energy=-1.0,
            split=DatasetSplit.DEVELOPMENT,
        )


def test_problem_descriptors_are_defensively_frozen(
    h2_hamiltonian: QubitHamiltonian,
) -> None:
    descriptors = {"basis": "sto-3g"}
    problem = AuditProblem(
        problem_id="h2",
        hamiltonian=h2_hamiltonian,
        n_alpha=1,
        n_beta=1,
        reference_energy=-1.137,
        split=DatasetSplit.HELD_OUT,
        descriptors=descriptors,
    )

    descriptors["basis"] = "changed"

    assert problem.descriptors["basis"] == "sto-3g"
    with pytest.raises(TypeError):
        problem.descriptors["basis"] = "changed"  # type: ignore[index]


def test_problem_identity_is_stable_for_equivalent_descriptor_order(
    h2_hamiltonian: QubitHamiltonian,
) -> None:
    first = AuditProblem(
        problem_id="h2",
        hamiltonian=h2_hamiltonian,
        n_alpha=1,
        n_beta=1,
        reference_energy=-1.137,
        split=DatasetSplit.DEVELOPMENT,
        descriptors={"basis": "sto-3g", "bond": 0.735},
    )
    second = AuditProblem(
        problem_id="h2",
        hamiltonian=h2_hamiltonian,
        n_alpha=1,
        n_beta=1,
        reference_energy=-1.137,
        split=DatasetSplit.DEVELOPMENT,
        descriptors={"bond": 0.735, "basis": "sto-3g"},
    )

    assert first.identity_hash == second.identity_hash


def test_problem_reference_energy_must_be_finite(
    h2_hamiltonian: QubitHamiltonian,
) -> None:
    with pytest.raises(ValidationError, match="reference_energy"):
        AuditProblem(
            problem_id="h2",
            hamiltonian=h2_hamiltonian,
            n_alpha=1,
            n_beta=1,
            reference_energy=float("nan"),
            split=DatasetSplit.DEVELOPMENT,
        )


def test_h2_problem_fixture_is_physically_consistent(h2_problem: AuditProblem) -> None:
    assert h2_problem.problem_id == "h2-equilibrium"
    assert h2_problem.hamiltonian.n_qubits == 4
    assert h2_problem.n_alpha == h2_problem.n_beta == 1

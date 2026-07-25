"""Content-addressed sampler-audit bundles."""

from __future__ import annotations

import json

import numpy as np
import pytest

from qdr_compute_engine.audit.analysis import ComparisonStatus, SamplerComparison
from qdr_compute_engine.audit.artifacts import (
    AuditResult,
    load_audit_bundle,
    scientific_result_hash,
    write_audit_bundle,
)
from qdr_compute_engine.audit.models import (
    AuditObservation,
    DatasetSplit,
    ResourceLedger,
    SamplerKind,
)
from qdr_core.errors import ValidationError

PROBLEM_HASH = "sha256:" + "2" * 64


def _result() -> AuditResult:
    observation = AuditObservation(
        problem_id="h2-held-out",
        problem_hash=PROBLEM_HASH,
        source_id="exact-state",
        kind=SamplerKind.ORACLE,
        split=DatasetSplit.HELD_OUT,
        seed=7,
        shots=3,
        shots_returned=3,
        n_qubits=4,
        subspace_dimension=2,
        energy=-1.13,
        reference_energy=-1.137,
        absolute_error=0.007,
        sector_retention=1.0,
        configurations=(3, 12),
        ledger=ResourceLedger(cpu_seconds=0.01),
        metadata={"postprocessor_id": "qdr-fixed-subspace"},
    )
    comparison = SamplerComparison(
        quantum_source_id="exact-state",
        classical_source_id="hf-excitations",
        point_estimate=-0.002,
        interval_low=-0.003,
        interval_high=-0.001,
        confidence=0.95,
        n_pairs=30,
        missing_pairs=0,
        status=ComparisonStatus.QUANTUM_BENEFIT,
    )
    return AuditResult(
        protocol={"protocol_version": 1, "title": "test audit"},
        observations=(observation,),
        comparison=comparison,
        raw_samples={
            "h2-held-out/exact-state/shots-3/seed-7": np.array(
                [12, 12, 3], dtype=np.int64
            )
        },
        environment={"python": "3.12.test"},
        git_commit="abc123",
    )


def test_bundle_round_trip_preserves_raw_samples_and_full_precision(tmp_path) -> None:
    paths = write_audit_bundle(_result(), tmp_path)

    loaded = load_audit_bundle(paths.manifest)

    assert loaded.observations == _result().observations
    assert loaded.raw_samples["h2-held-out/exact-state/shots-3/seed-7"].tolist() == [
        12,
        12,
        3,
    ]
    assert loaded.comparison == _result().comparison


def test_bundle_hash_is_deterministic_for_identical_evidence(tmp_path) -> None:
    first = write_audit_bundle(_result(), tmp_path / "first")
    second = write_audit_bundle(_result(), tmp_path / "second")

    assert first.manifest_hash == second.manifest_hash
    assert first.samples.read_bytes() == second.samples.read_bytes()
    assert first.observations.read_bytes() == second.observations.read_bytes()


def test_scientific_hash_excludes_runtime_measurements() -> None:
    original = _result()
    observation = original.observations[0]
    replayed_observation = AuditObservation(
        problem_id=observation.problem_id,
        problem_hash=observation.problem_hash,
        source_id=observation.source_id,
        kind=observation.kind,
        split=observation.split,
        seed=observation.seed,
        shots=observation.shots,
        shots_returned=observation.shots_returned,
        n_qubits=observation.n_qubits,
        subspace_dimension=observation.subspace_dimension,
        energy=observation.energy,
        reference_energy=observation.reference_energy,
        absolute_error=observation.absolute_error,
        sector_retention=observation.sector_retention,
        configurations=observation.configurations,
        ledger=ResourceLedger(cpu_seconds=99.0, wall_seconds=100.0),
        metadata=observation.metadata,
    )
    replayed = AuditResult(
        protocol=original.protocol,
        observations=(replayed_observation,),
        comparison=original.comparison,
        raw_samples=original.raw_samples,
        environment=original.environment,
        git_commit=original.git_commit,
    )

    assert scientific_result_hash(original) == scientific_result_hash(replayed)


def test_bundle_rejects_a_modified_sample_file(tmp_path) -> None:
    paths = write_audit_bundle(_result(), tmp_path)
    paths.samples.write_bytes(b"tampered")

    with pytest.raises(ValidationError, match="digest"):
        load_audit_bundle(paths.manifest)


def test_bundle_rejects_a_modified_observation_file(tmp_path) -> None:
    paths = write_audit_bundle(_result(), tmp_path)
    paths.observations.write_text("{}", encoding="utf-8")

    with pytest.raises(ValidationError, match="digest"):
        load_audit_bundle(paths.manifest)


def test_bundle_refuses_to_overwrite_different_evidence(tmp_path) -> None:
    write_audit_bundle(_result(), tmp_path)
    changed = AuditResult(
        protocol={"protocol_version": 1, "title": "changed"},
        observations=_result().observations,
        comparison=_result().comparison,
        raw_samples=_result().raw_samples,
        environment=_result().environment,
        git_commit="abc123",
    )

    with pytest.raises(ValidationError, match="overwrite"):
        write_audit_bundle(changed, tmp_path)


def test_result_rejects_raw_samples_that_do_not_cover_observation_cells() -> None:
    original = _result()

    with pytest.raises(ValidationError, match="raw sample coverage"):
        AuditResult(
            protocol=original.protocol,
            observations=original.observations,
            comparison=original.comparison,
            raw_samples={"wrong-cell": np.array([12, 12, 3])},
            environment=original.environment,
            git_commit=original.git_commit,
        )


def test_result_accepts_partial_returned_shots_and_checks_register_bounds() -> None:
    original = _result()
    observation = original.observations[0]
    partial = AuditObservation(
        problem_id=observation.problem_id,
        problem_hash=observation.problem_hash,
        source_id=observation.source_id,
        kind=observation.kind,
        split=observation.split,
        seed=observation.seed,
        shots=5,
        shots_returned=3,
        n_qubits=4,
        subspace_dimension=observation.subspace_dimension,
        energy=observation.energy,
        reference_energy=observation.reference_energy,
        absolute_error=observation.absolute_error,
        sector_retention=observation.sector_retention,
        configurations=observation.configurations,
        ledger=observation.ledger,
        metadata=observation.metadata,
    )
    result = AuditResult(
        protocol=original.protocol,
        observations=(partial,),
        comparison=original.comparison,
        raw_samples={
            "h2-held-out/exact-state/shots-5/seed-7": np.array([12, 12, 3])
        },
        environment=original.environment,
        git_commit=original.git_commit,
    )

    assert len(result.raw_samples) == 1
    with pytest.raises(ValidationError, match="register"):
        AuditResult(
            protocol=original.protocol,
            observations=(partial,),
            comparison=original.comparison,
            raw_samples={
                "h2-held-out/exact-state/shots-5/seed-7": np.array([16, 12, 3])
            },
            environment=original.environment,
            git_commit=original.git_commit,
        )


def test_bundle_hashes_and_verifies_supplemental_report_artifacts(tmp_path) -> None:
    paths = write_audit_bundle(
        _result(),
        tmp_path,
        supplemental_artifacts={"report": ("report.md", b"# reviewed report\n")},
    )
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))

    assert manifest["artifacts"]["report"]["path"] == "report.md"
    (tmp_path / "report.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValidationError, match="digest"):
        load_audit_bundle(paths.manifest)

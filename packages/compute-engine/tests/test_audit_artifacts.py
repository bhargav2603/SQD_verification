"""Content-addressed sampler-audit bundles."""

from __future__ import annotations

import numpy as np
import pytest

from qdr_compute_engine.audit.analysis import ComparisonStatus, SamplerComparison
from qdr_compute_engine.audit.artifacts import (
    AuditResult,
    load_audit_bundle,
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
        raw_samples={"exact-state/seed-7": np.array([12, 12, 3], dtype=np.int64)},
        environment={"python": "3.12.test"},
        git_commit="abc123",
    )


def test_bundle_round_trip_preserves_raw_samples_and_full_precision(tmp_path) -> None:
    paths = write_audit_bundle(_result(), tmp_path)

    loaded = load_audit_bundle(paths.manifest)

    assert loaded.observations == _result().observations
    assert loaded.raw_samples["exact-state/seed-7"].tolist() == [12, 12, 3]
    assert loaded.comparison == _result().comparison


def test_bundle_hash_is_deterministic_for_identical_evidence(tmp_path) -> None:
    first = write_audit_bundle(_result(), tmp_path / "first")
    second = write_audit_bundle(_result(), tmp_path / "second")

    assert first.manifest_hash == second.manifest_hash
    assert first.samples.read_bytes() == second.samples.read_bytes()
    assert first.observations.read_bytes() == second.observations.read_bytes()


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

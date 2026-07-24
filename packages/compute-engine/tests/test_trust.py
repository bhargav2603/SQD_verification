"""Fail-closed scientific trust verdicts.

The regression this file protects is the original false-positive gate: two exact solvers
can agree on the same chemically inadequate Hamiltonian.  Agreement must survive as useful
engine evidence without being allowed to overrule a failed chemistry diagnostic.
"""

import json

import pytest

from qdr_compute_engine.validation.trust import (
    Evidence,
    EvidenceStatus,
    FailureMode,
    TrustProfile,
    Verdict,
    evaluate_trust,
)
from qdr_core.errors import ValidationError


def _profile(**required_codes: FailureMode) -> TrustProfile:
    return TrustProfile(name="test", required_codes=required_codes)


def test_chemistry_failure_overrules_same_hamiltonian_engine_agreement() -> None:
    """Would fail if the evaluator treated CASCI/QDR agreement as global validation."""
    report = evaluate_trust(
        [
            Evidence(
                code="engine_crosscheck",
                status=EvidenceStatus.PASS,
                failure_mode=FailureMode.ENGINE_DISAGREEMENT,
                value=2.0e-9,
                threshold=1.0e-6,
                units="Ha",
                message="QDR and CASCI agree inside the chosen Hamiltonian",
                metadata={"scope": "engine_only"},
            ),
            Evidence(
                code="basis",
                status=EvidenceStatus.FAIL,
                failure_mode=FailureMode.MINIMAL_BASIS,
                value="sto-3g",
                message="minimal basis cannot support a quantitative transition-metal gap",
            ),
            Evidence(
                code="required_orbitals",
                status=EvidenceStatus.FAIL,
                failure_mode=FailureMode.MISSING_REQUIRED_ORBITALS,
                value=["Fe 3d"],
                message="ligand donor orbitals are absent",
            ),
        ],
        profile=_profile(
            engine_crosscheck=FailureMode.ENGINE_DISAGREEMENT,
            basis=FailureMode.MINIMAL_BASIS,
            required_orbitals=FailureMode.MISSING_REQUIRED_ORBITALS,
        ),
    )

    assert report.verdict is Verdict.UNTRUSTED
    assert report.failure_modes == (
        FailureMode.MINIMAL_BASIS,
        FailureMode.MISSING_REQUIRED_ORBITALS,
    )
    assert report.evidence[0].status is EvidenceStatus.PASS


def test_missing_required_evidence_is_undecidable_not_trusted() -> None:
    """Would fail if unknown active-space error were silently treated as zero."""
    report = evaluate_trust(
        [
            Evidence(
                code="engine_crosscheck",
                status=EvidenceStatus.PASS,
                failure_mode=FailureMode.ENGINE_DISAGREEMENT,
            ),
            Evidence(
                code="active_space_ladder",
                status=EvidenceStatus.MISSING,
                failure_mode=FailureMode.ACTIVE_SPACE_NOT_CONVERGED,
                message="only one active space was computed",
            ),
        ],
        profile=_profile(
            engine_crosscheck=FailureMode.ENGINE_DISAGREEMENT,
            active_space_ladder=FailureMode.ACTIVE_SPACE_NOT_CONVERGED,
        ),
    )

    assert report.verdict is Verdict.UNDECIDABLE
    assert report.failure_modes == ()
    assert report.missing_modes == (FailureMode.ACTIVE_SPACE_NOT_CONVERGED,)


def test_definite_failure_dominates_other_missing_evidence() -> None:
    """Would fail if missing evidence masked an observed invalid model."""
    report = evaluate_trust(
        [
            Evidence(
                code="basis",
                status=EvidenceStatus.FAIL,
                failure_mode=FailureMode.MINIMAL_BASIS,
            ),
            Evidence(
                code="dynamic_correlation",
                status=EvidenceStatus.MISSING,
                failure_mode=FailureMode.DYNAMIC_CORRELATION_MISSING,
            ),
        ],
        profile=_profile(
            basis=FailureMode.MINIMAL_BASIS,
            dynamic_correlation=FailureMode.DYNAMIC_CORRELATION_MISSING,
        ),
    )

    assert report.verdict is Verdict.UNTRUSTED
    assert report.failure_modes == (FailureMode.MINIMAL_BASIS,)
    assert report.missing_modes == (FailureMode.DYNAMIC_CORRELATION_MISSING,)


def test_all_required_evidence_must_pass_for_trusted() -> None:
    report = evaluate_trust(
        [
            Evidence(
                code="basis",
                status=EvidenceStatus.PASS,
                failure_mode=FailureMode.MINIMAL_BASIS,
            ),
            Evidence(
                code="active_space_ladder",
                status=EvidenceStatus.PASS,
                failure_mode=FailureMode.ACTIVE_SPACE_NOT_CONVERGED,
            ),
        ],
        profile=_profile(
            basis=FailureMode.MINIMAL_BASIS,
            active_space_ladder=FailureMode.ACTIVE_SPACE_NOT_CONVERGED,
        ),
    )

    assert report.verdict is Verdict.TRUSTED


def test_optional_missing_evidence_does_not_block_trust() -> None:
    report = evaluate_trust(
        [
            Evidence(
                code="required",
                status=EvidenceStatus.PASS,
                failure_mode=FailureMode.INVALID_RESULT,
            ),
            Evidence(
                code="context_only",
                status=EvidenceStatus.MISSING,
                failure_mode=FailureMode.LITERATURE_REFERENCE_MISSING,
                required=False,
            ),
        ],
        profile=_profile(required=FailureMode.INVALID_RESULT),
    )

    assert report.verdict is Verdict.TRUSTED


def test_report_serializes_to_strict_deterministic_json() -> None:
    report = evaluate_trust(
        [
            Evidence(
                code="spin_purity",
                status=EvidenceStatus.PASS,
                failure_mode=FailureMode.SPIN_CONTAMINATION,
                value=3.0e-7,
                threshold=1.0e-3,
                units="S^2",
                metadata={"target": 0.0},
            )
        ],
        profile=_profile(spin_purity=FailureMode.SPIN_CONTAMINATION),
    )

    first = report.to_json()
    second = report.to_json()
    payload = json.loads(first, parse_constant=lambda value: pytest.fail(value))

    assert first == second
    assert payload["verdict"] == "TRUSTED"
    assert payload["evidence"][0]["failure_mode"] == "SPIN_CONTAMINATION"


def test_non_finite_evidence_is_rejected_before_json_output() -> None:
    with pytest.raises(ValidationError, match="finite"):
        Evidence(
            code="bad_energy",
            status=EvidenceStatus.FAIL,
            failure_mode=FailureMode.INVALID_RESULT,
            value=float("nan"),
        )


def test_duplicate_evidence_codes_are_rejected() -> None:
    repeated = Evidence(
        code="basis",
        status=EvidenceStatus.PASS,
        failure_mode=FailureMode.MINIMAL_BASIS,
    )

    with pytest.raises(ValidationError, match="duplicate"):
        evaluate_trust(
            [repeated, repeated],
            profile=_profile(basis=FailureMode.MINIMAL_BASIS),
        )


def test_lone_engine_pass_cannot_satisfy_default_quantitative_profile() -> None:
    report = evaluate_trust(
        [
            Evidence(
                code="engine_crosscheck",
                status=EvidenceStatus.PASS,
                failure_mode=FailureMode.ENGINE_DISAGREEMENT,
            )
        ]
    )

    assert report.verdict is Verdict.UNDECIDABLE
    assert FailureMode.BASIS_NOT_CONVERGED in report.missing_modes


def test_caller_cannot_demote_profile_required_evidence() -> None:
    with pytest.raises(ValidationError, match="profile-required"):
        evaluate_trust(
            [
                Evidence(
                    code="engine_crosscheck",
                    status=EvidenceStatus.PASS,
                    failure_mode=FailureMode.ENGINE_DISAGREEMENT,
                    required=False,
                )
            ],
            profile=_profile(
                engine_crosscheck=FailureMode.ENGINE_DISAGREEMENT,
            ),
        )

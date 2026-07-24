"""Fail-closed, machine-readable scientific trust verdicts.

This module deliberately knows nothing about PySCF, SQD, or any particular molecule.  It
combines evidence produced by those layers under one rule: numerical agreement is local to
the question it tested.  In particular, an engine cross-check cannot cancel a failed basis,
active-space, geometry, or environment check.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from qdr_core.errors import ValidationError


class Verdict(str, Enum):
    """Overall status of a scientific result."""

    TRUSTED = "TRUSTED"
    UNTRUSTED = "UNTRUSTED"
    UNDECIDABLE = "UNDECIDABLE"


class EvidenceStatus(str, Enum):
    """Outcome of one diagnostic."""

    PASS = "PASS"
    FAIL = "FAIL"
    MISSING = "MISSING"


class FailureMode(str, Enum):
    """Stable failure identifiers for machines and humans."""

    ENGINE_DISAGREEMENT = "ENGINE_DISAGREEMENT"
    INVALID_RESULT = "INVALID_RESULT"
    SCF_NOT_CONVERGED = "SCF_NOT_CONVERGED"
    SCF_UNSTABLE = "SCF_UNSTABLE"
    SPIN_CONTAMINATION = "SPIN_CONTAMINATION"
    WRONG_ELECTRON_OR_SPIN_SECTOR = "WRONG_ELECTRON_OR_SPIN_SECTOR"
    MINIMAL_BASIS = "MINIMAL_BASIS"
    BASIS_NOT_CONVERGED = "BASIS_NOT_CONVERGED"
    MISSING_REQUIRED_ORBITALS = "MISSING_REQUIRED_ORBITALS"
    ACTIVE_SPACE_NOT_CONVERGED = "ACTIVE_SPACE_NOT_CONVERGED"
    ACTIVE_SPACE_BOUNDARY_LEAKAGE = "ACTIVE_SPACE_BOUNDARY_LEAKAGE"
    ACTIVE_SPACE_CAP_FORCED = "ACTIVE_SPACE_CAP_FORCED"
    SQD_NOT_CONVERGED = "SQD_NOT_CONVERGED"
    SQD_NON_VARIATIONAL = "SQD_NON_VARIATIONAL"
    LOW_SECTOR_RETENTION = "LOW_SECTOR_RETENTION"
    SAMPLING_NOT_REPRODUCIBLE = "SAMPLING_NOT_REPRODUCIBLE"
    GEOMETRY_INADEQUATE = "GEOMETRY_INADEQUATE"
    ENVIRONMENT_MISMATCH = "ENVIRONMENT_MISMATCH"
    EMBEDDING_NOT_VALIDATED = "EMBEDDING_NOT_VALIDATED"
    EMBEDDING_PARTITION_UNSTABLE = "EMBEDDING_PARTITION_UNSTABLE"
    DYNAMIC_CORRELATION_MISSING = "DYNAMIC_CORRELATION_MISSING"
    DYNAMIC_CORRELATION_NOT_CONVERGED = "DYNAMIC_CORRELATION_NOT_CONVERGED"
    COUPLED_CLUSTER_NOT_CONVERGED = "COUPLED_CLUSTER_NOT_CONVERGED"
    SINGLE_REFERENCE_BREAKDOWN = "SINGLE_REFERENCE_BREAKDOWN"
    METHODS_DISAGREE = "METHODS_DISAGREE"
    ORBITAL_SPACE_NOT_NESTED = "ORBITAL_SPACE_NOT_NESTED"
    SAMPLING_PROVENANCE_MISSING = "SAMPLING_PROVENANCE_MISSING"
    LITERATURE_REFERENCE_MISSING = "LITERATURE_REFERENCE_MISSING"
    NO_TRUST_EVIDENCE = "NO_TRUST_EVIDENCE"


JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _validate_finite(value: Any, path: str) -> None:
    """Reject non-finite numbers anywhere in an evidence payload."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(
                f"{path} must be finite, got {value!r}",
                field=path,
                value=repr(value),
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite(item, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _validate_finite(item, f"{path}[{index}]")


def _json_value(value: Any) -> JsonValue:
    """Normalize enums, tuples, and mappings to deterministic JSON-native values."""
    if isinstance(value, Enum):
        return str(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    raise ValidationError(
        f"value of type {type(value).__name__} is not JSON serializable",
        value_type=type(value).__name__,
    )


@dataclass(frozen=True, slots=True)
class Evidence:
    """One independently interpretable diagnostic result."""

    code: str
    status: EvidenceStatus
    failure_mode: FailureMode
    required: bool = True
    value: Any = None
    threshold: Any = None
    units: str | None = None
    message: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValidationError("evidence code must not be empty")
        _validate_finite(self.value, f"{self.code}.value")
        _validate_finite(self.threshold, f"{self.code}.threshold")
        _validate_finite(self.metadata, f"{self.code}.metadata")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "code": self.code,
            "status": self.status.value,
            "failure_mode": self.failure_mode.value,
            "required": self.required,
            "value": _json_value(self.value),
            "threshold": _json_value(self.threshold),
            "units": self.units,
            "message": self.message,
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class TrustProfile:
    """Immutable policy naming every evidence axis required for a claim."""

    name: str
    required_codes: Mapping[str, FailureMode]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValidationError("trust profile name must not be empty")
        if not self.required_codes:
            raise ValidationError("trust profile must declare at least one required evidence code")
        if any(not code.strip() for code in self.required_codes):
            raise ValidationError("trust profile evidence codes must not be empty")
        object.__setattr__(
            self,
            "required_codes",
            MappingProxyType(dict(self.required_codes)),
        )


ENGINE_SANITY_PROFILE = TrustProfile(
    name="engine_sanity",
    required_codes={
        "engine_crosscheck": FailureMode.ENGINE_DISAGREEMENT,
        "spin_purity_low": FailureMode.SPIN_CONTAMINATION,
        "spin_purity_high": FailureMode.SPIN_CONTAMINATION,
        "sqd_convergence_low": FailureMode.SQD_NOT_CONVERGED,
        "sqd_convergence_high": FailureMode.SQD_NOT_CONVERGED,
        "sqd_reference_low": FailureMode.SQD_NOT_CONVERGED,
        "sqd_reference_high": FailureMode.SQD_NOT_CONVERGED,
        "sqd_sector_retention_low": FailureMode.LOW_SECTOR_RETENTION,
        "sqd_sector_retention_high": FailureMode.LOW_SECTOR_RETENTION,
        "sqd_reproducibility_low": FailureMode.SAMPLING_NOT_REPRODUCIBLE,
        "sqd_reproducibility_high": FailureMode.SAMPLING_NOT_REPRODUCIBLE,
    },
)

QUANTITATIVE_CHEMISTRY_PROFILE = TrustProfile(
    name="quantitative_chemistry",
    required_codes={
        "basis_representation": FailureMode.MINIMAL_BASIS,
        "basis_convergence": FailureMode.BASIS_NOT_CONVERGED,
        "required_orbitals": FailureMode.MISSING_REQUIRED_ORBITALS,
        "active_space_ladder": FailureMode.ACTIVE_SPACE_NOT_CONVERGED,
        "active_space_boundary": FailureMode.ACTIVE_SPACE_BOUNDARY_LEAKAGE,
        "orbital_space_nesting": FailureMode.ORBITAL_SPACE_NOT_NESTED,
        "dynamic_correlation": FailureMode.DYNAMIC_CORRELATION_MISSING,
        "independent_method_agreement": FailureMode.METHODS_DISAGREE,
        "sampling_provenance": FailureMode.SAMPLING_PROVENANCE_MISSING,
        "geometry_model": FailureMode.GEOMETRY_INADEQUATE,
        "environment_model": FailureMode.ENVIRONMENT_MISMATCH,
        **ENGINE_SANITY_PROFILE.required_codes,
    },
)

EMBEDDED_CHEMISTRY_PROFILE = TrustProfile(
    name="embedded_quantitative_chemistry",
    required_codes={
        **QUANTITATIVE_CHEMISTRY_PROFILE.required_codes,
        "embedding_validation": FailureMode.EMBEDDING_NOT_VALIDATED,
    },
)

TRUST_PROFILES: Mapping[str, TrustProfile] = {
    ENGINE_SANITY_PROFILE.name: ENGINE_SANITY_PROFILE,
    QUANTITATIVE_CHEMISTRY_PROFILE.name: QUANTITATIVE_CHEMISTRY_PROFILE,
    EMBEDDED_CHEMISTRY_PROFILE.name: EMBEDDED_CHEMISTRY_PROFILE,
}


@dataclass(frozen=True, slots=True)
class TrustReport:
    """The verdict and the complete evidence ledger that produced it."""

    verdict: Verdict
    evidence: tuple[Evidence, ...]
    failure_modes: tuple[FailureMode, ...] = ()
    missing_modes: tuple[FailureMode, ...] = ()
    profile: str = QUANTITATIVE_CHEMISTRY_PROFILE.name

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "verdict": self.verdict.value,
            "profile": self.profile,
            "failure_modes": [mode.value for mode in self.failure_modes],
            "missing_modes": [mode.value for mode in self.missing_modes],
            "evidence": [item.to_dict() for item in self.evidence],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Strict JSON: non-finite values are refused instead of emitted as JavaScript."""
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            indent=indent,
            sort_keys=True,
        )


def _unique_modes(items: Iterable[Evidence]) -> tuple[FailureMode, ...]:
    seen: set[FailureMode] = set()
    ordered: list[FailureMode] = []
    for item in items:
        if item.failure_mode not in seen:
            seen.add(item.failure_mode)
            ordered.append(item.failure_mode)
    return tuple(ordered)


def evaluate_trust(
    evidence: Iterable[Evidence],
    *,
    profile: TrustProfile = QUANTITATIVE_CHEMISTRY_PROFILE,
) -> TrustReport:
    """Evaluate required evidence under a fail-closed three-way decision rule."""
    supplied = tuple(evidence)
    supplied_by_code = {item.code: item for item in supplied}
    demoted = sorted(
        code
        for code in profile.required_codes
        if code in supplied_by_code and not supplied_by_code[code].required
    )
    if demoted:
        raise ValidationError(
            f"profile-required evidence cannot be optional: {', '.join(demoted)}",
            profile=profile.name,
            evidence_codes=demoted,
        )
    missing_profile_evidence = tuple(
        Evidence(
            code=code,
            status=EvidenceStatus.MISSING,
            failure_mode=failure_mode,
            message=f"required by trust profile {profile.name!r} but not supplied",
        )
        for code, failure_mode in sorted(profile.required_codes.items())
        if code not in supplied_by_code
    )
    ledger = (*supplied, *missing_profile_evidence)
    codes = [item.code for item in ledger]
    duplicate_codes = sorted({code for code in codes if codes.count(code) > 1})
    if duplicate_codes:
        raise ValidationError(
            f"duplicate evidence codes: {', '.join(duplicate_codes)}",
            duplicate_codes=duplicate_codes,
        )

    required_failures = tuple(
        item for item in ledger if item.required and item.status is EvidenceStatus.FAIL
    )
    required_missing = tuple(
        item for item in ledger if item.required and item.status is EvidenceStatus.MISSING
    )

    if required_failures:
        verdict = Verdict.UNTRUSTED
    elif required_missing or not ledger:
        verdict = Verdict.UNDECIDABLE
    else:
        verdict = Verdict.TRUSTED

    missing_modes = _unique_modes(required_missing)
    if not ledger:
        missing_modes = (FailureMode.NO_TRUST_EVIDENCE,)

    return TrustReport(
        verdict=verdict,
        evidence=ledger,
        failure_modes=_unique_modes(required_failures),
        missing_modes=missing_modes,
        profile=profile.name,
    )


def evidence_from_dict(payload: Mapping[str, Any]) -> Evidence:
    """Parse a JSON-native diagnostic with domain errors instead of raw enum failures."""
    try:
        status = EvidenceStatus(str(payload["status"]))
        failure_mode = FailureMode(str(payload["failure_mode"]))
        code = str(payload["code"])
    except (KeyError, ValueError) as exc:
        raise ValidationError(
            "trust evidence needs valid code, status, and failure_mode fields",
            payload=dict(payload),
        ) from exc

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValidationError(
            "trust evidence metadata must be a mapping",
            code=code,
            metadata_type=type(metadata).__name__,
        )
    return Evidence(
        code=code,
        status=status,
        failure_mode=failure_mode,
        required=bool(payload.get("required", True)),
        value=payload.get("value"),
        threshold=payload.get("threshold"),
        units=None if payload.get("units") is None else str(payload["units"]),
        message=str(payload.get("message", "")),
        metadata=metadata,
    )

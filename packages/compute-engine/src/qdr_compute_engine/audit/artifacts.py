"""Content-addressed, tamper-evident audit bundles."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from qdr_compute_engine.audit.analysis import (
    ComparisonStatus,
    SamplerComparison,
)
from qdr_compute_engine.audit.models import (
    AuditObservation,
    DatasetSplit,
    ResourceLedger,
    SamplerKind,
)
from qdr_core.errors import ValidationError
from qdr_core.provenance import canonical_json, content_hash, hash_bytes

MANIFEST_NAME = "manifest.json"
OBSERVATIONS_NAME = "observations.json"
SAMPLES_NAME = "raw_samples.npz"


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    raise TypeError(f"cannot serialize {type(value).__name__} into an audit bundle")


def _observation_to_dict(observation: AuditObservation) -> dict[str, Any]:
    return {
        "problem_id": observation.problem_id,
        "problem_hash": observation.problem_hash,
        "source_id": observation.source_id,
        "kind": observation.kind.value,
        "split": observation.split.value,
        "seed": observation.seed,
        "shots": observation.shots,
        "subspace_dimension": observation.subspace_dimension,
        "energy": observation.energy,
        "reference_energy": observation.reference_energy,
        "absolute_error": observation.absolute_error,
        "sector_retention": observation.sector_retention,
        "configurations": list(observation.configurations),
        "ledger": observation.ledger.to_dict(),
        "metadata": _json_value(observation.metadata),
    }


def _observation_from_dict(payload: Mapping[str, Any]) -> AuditObservation:
    return AuditObservation(
        problem_id=str(payload["problem_id"]),
        problem_hash=str(payload["problem_hash"]),
        source_id=str(payload["source_id"]),
        kind=SamplerKind(str(payload["kind"])),
        split=DatasetSplit(str(payload["split"])),
        seed=int(payload["seed"]),
        shots=int(payload["shots"]),
        subspace_dimension=int(payload["subspace_dimension"]),
        energy=float(payload["energy"]),
        reference_energy=float(payload["reference_energy"]),
        absolute_error=float(payload["absolute_error"]),
        sector_retention=float(payload["sector_retention"]),
        configurations=tuple(int(value) for value in payload["configurations"]),
        ledger=ResourceLedger(**dict(payload["ledger"])),
        metadata=dict(payload["metadata"]),
    )


def _comparison_to_dict(comparison: SamplerComparison) -> dict[str, Any]:
    return {
        "quantum_source_id": comparison.quantum_source_id,
        "classical_source_id": comparison.classical_source_id,
        "point_estimate": comparison.point_estimate,
        "interval_low": comparison.interval_low,
        "interval_high": comparison.interval_high,
        "confidence": comparison.confidence,
        "n_pairs": comparison.n_pairs,
        "missing_pairs": comparison.missing_pairs,
        "status": comparison.status.value,
        "metric": comparison.metric,
    }


def _comparison_from_dict(payload: Mapping[str, Any]) -> SamplerComparison:
    return SamplerComparison(
        quantum_source_id=str(payload["quantum_source_id"]),
        classical_source_id=str(payload["classical_source_id"]),
        point_estimate=(
            None
            if payload["point_estimate"] is None
            else float(payload["point_estimate"])
        ),
        interval_low=(
            None if payload["interval_low"] is None else float(payload["interval_low"])
        ),
        interval_high=(
            None
            if payload["interval_high"] is None
            else float(payload["interval_high"])
        ),
        confidence=float(payload["confidence"]),
        n_pairs=int(payload["n_pairs"]),
        missing_pairs=int(payload["missing_pairs"]),
        status=ComparisonStatus(str(payload["status"])),
        metric=str(payload["metric"]),
    )


@dataclass(frozen=True, slots=True, init=False)
class AuditResult:
    """Complete evidence needed to reconstruct one audit conclusion."""

    protocol: Mapping[str, Any]
    observations: tuple[AuditObservation, ...]
    comparison: SamplerComparison
    raw_samples: Mapping[str, NDArray[np.int64]]
    environment: Mapping[str, Any]
    git_commit: str

    def __init__(
        self,
        *,
        protocol: Mapping[str, Any],
        observations: tuple[AuditObservation, ...],
        comparison: SamplerComparison,
        raw_samples: Mapping[str, NDArray[np.int64]],
        environment: Mapping[str, Any],
        git_commit: str,
    ) -> None:
        frozen_samples: dict[str, NDArray[np.int64]] = {}
        for key, values in raw_samples.items():
            array = np.asarray(values)
            if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
                raise ValidationError(
                    "raw sample artifacts must be one-dimensional integer arrays",
                    sample_id=key,
                )
            copied = np.array(array, dtype=np.int64, copy=True)
            copied.flags.writeable = False
            frozen_samples[str(key)] = copied
        object.__setattr__(self, "protocol", MappingProxyType(dict(protocol)))
        object.__setattr__(self, "observations", tuple(observations))
        object.__setattr__(self, "comparison", comparison)
        object.__setattr__(self, "raw_samples", MappingProxyType(frozen_samples))
        object.__setattr__(self, "environment", MappingProxyType(dict(environment)))
        object.__setattr__(self, "git_commit", git_commit)


@dataclass(frozen=True, slots=True)
class AuditBundlePaths:
    manifest: Path
    observations: Path
    samples: Path
    manifest_hash: str


def _deterministic_npz(raw_samples: Mapping[str, NDArray[np.int64]]) -> bytes:
    output = io.BytesIO()
    index: dict[str, str] = {}
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for position, key in enumerate(sorted(raw_samples)):
            filename = f"samples-{position:06d}.npy"
            index[key] = filename
            array_buffer = io.BytesIO()
            np.lib.format.write_array(
                array_buffer,
                np.asarray(raw_samples[key], dtype=np.int64),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, array_buffer.getvalue(), compresslevel=9)
        info = zipfile.ZipInfo("index.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o600 << 16
        archive.writestr(info, canonical_json(index), compresslevel=9)
    return output.getvalue()


def _load_npz(data: bytes) -> dict[str, NDArray[np.int64]]:
    loaded: dict[str, NDArray[np.int64]] = {}
    with zipfile.ZipFile(io.BytesIO(data), mode="r") as archive:
        index = json.loads(archive.read("index.json"))
        for key, filename in index.items():
            array = np.lib.format.read_array(
                io.BytesIO(archive.read(filename)),
                allow_pickle=False,
            )
            loaded[str(key)] = np.asarray(array, dtype=np.int64)
    return loaded


def _bundle_bytes(result: AuditResult) -> tuple[bytes, bytes, dict[str, Any]]:
    observations_bytes = canonical_json(
        {"observations": [_observation_to_dict(item) for item in result.observations]}
    )
    samples_bytes = _deterministic_npz(result.raw_samples)
    manifest_core = {
        "bundle_version": 1,
        "protocol": _json_value(result.protocol),
        "comparison": _comparison_to_dict(result.comparison),
        "environment": _json_value(result.environment),
        "git_commit": result.git_commit,
        "artifacts": {
            "observations": {
                "path": OBSERVATIONS_NAME,
                "digest": hash_bytes(observations_bytes),
            },
            "raw_samples": {
                "path": SAMPLES_NAME,
                "digest": hash_bytes(samples_bytes),
            },
        },
    }
    return observations_bytes, samples_bytes, manifest_core


def write_audit_bundle(result: AuditResult, output_dir: Path) -> AuditBundlePaths:
    """Write a deterministic bundle, refusing to overwrite different evidence."""

    observations_bytes, samples_bytes, manifest_core = _bundle_bytes(result)
    manifest_hash = content_hash(manifest_core)
    manifest_payload = {**manifest_core, "manifest_hash": manifest_hash}
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    observations_path = output_dir / OBSERVATIONS_NAME
    samples_path = output_dir / SAMPLES_NAME
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("manifest_hash") != manifest_hash:
            raise ValidationError(
                "refusing to overwrite an audit bundle with different evidence",
                destination=str(output_dir),
                existing_hash=existing.get("manifest_hash"),
                new_hash=manifest_hash,
            )
    observations_path.write_bytes(observations_bytes)
    samples_path.write_bytes(samples_bytes)
    manifest_path.write_bytes(canonical_json(manifest_payload))
    return AuditBundlePaths(
        manifest=manifest_path,
        observations=observations_path,
        samples=samples_path,
        manifest_hash=manifest_hash,
    )


def load_audit_bundle(manifest_path: Path) -> AuditResult:
    """Verify every digest and reconstruct the evidence bundle."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claimed_manifest_hash = str(manifest.pop("manifest_hash"))
    actual_manifest_hash = content_hash(manifest)
    if actual_manifest_hash != claimed_manifest_hash:
        raise ValidationError(
            "manifest digest does not match its contents",
            expected=claimed_manifest_hash,
            actual=actual_manifest_hash,
        )
    parent = manifest_path.parent
    artifact_payloads: dict[str, bytes] = {}
    for name, record in manifest["artifacts"].items():
        data = (parent / record["path"]).read_bytes()
        actual_digest = hash_bytes(data)
        if actual_digest != record["digest"]:
            raise ValidationError(
                f"{name} artifact digest does not match the manifest",
                artifact=name,
                expected=record["digest"],
                actual=actual_digest,
            )
        artifact_payloads[name] = data
    observation_payload = json.loads(artifact_payloads["observations"])
    return AuditResult(
        protocol=dict(manifest["protocol"]),
        observations=tuple(
            _observation_from_dict(item) for item in observation_payload["observations"]
        ),
        comparison=_comparison_from_dict(manifest["comparison"]),
        raw_samples=_load_npz(artifact_payloads["raw_samples"]),
        environment=dict(manifest["environment"]),
        git_commit=str(manifest["git_commit"]),
    )


__all__ = [
    "AuditBundlePaths",
    "AuditResult",
    "load_audit_bundle",
    "write_audit_bundle",
]

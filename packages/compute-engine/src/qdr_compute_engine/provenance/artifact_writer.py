"""Write run artifacts with content-addressed provenance (ADR-0007).

Artifacts are written under their own digest, so the same content always lands at the
same key. That makes writes idempotent -- a retried job overwrites itself byte for byte
instead of creating a second copy -- and makes tampering detectable, since the key is a
claim about the contents that can be re-verified at any time.

Object storage lives in the control plane; this module produces the bytes and the
:class:`~qdr_core.provenance.ArtifactRef`.

.. warning::

   **Nothing calls this yet.** The intended consumer is the worker's
   ``result_publisher``, which is a 0-byte placeholder, and no pipeline stage invokes it
   either -- so no run currently emits a content-addressed artifact. The functions below
   are implemented and unit-tested, but the *provenance chain they describe does not
   exist end to end*.

   Recorded here rather than left for a reader to discover, because a module that reads
   as working infrastructure while being unreachable is exactly the kind of claim this
   codebase is meant not to make. Wire it into stage 9 (or the benchmark runner) before
   citing artifact provenance as a property of a run.
"""

import json
from pathlib import Path
from typing import Any

from qdr_compute_engine.reproducibility.input_hashing import normalize
from qdr_core.errors import ValidationError
from qdr_core.provenance import ArtifactRef, canonical_json, hash_bytes

MEDIA_TYPE_JSON = "application/json"


def serialize_artifact(payload: dict[str, Any]) -> bytes:
    """Canonical JSON bytes for an artifact payload.

    Uses the same canonicalisation as content hashing, so the bytes written are exactly
    the bytes hashed -- there is no window in which a reformat changes the digest.
    """
    return canonical_json(normalize(payload))


def build_artifact_ref(
    payload: dict[str, Any],
    uri_prefix: str = "s3://qdr-artifacts",
) -> tuple[ArtifactRef, bytes]:
    """Build the reference and the bytes for an artifact payload."""
    data = serialize_artifact(payload)
    digest = hash_bytes(data)
    # The digest is the storage key; the prefix only says which bucket holds a copy.
    uri = f"{uri_prefix}/{digest.replace(':', '/')}"
    ref = ArtifactRef.create(
        uri=uri,
        content_hash=digest,
        media_type=MEDIA_TYPE_JSON,
        size_bytes=len(data),
    )
    return ref, data


def write_artifact(
    payload: dict[str, Any],
    destination: Path,
    uri_prefix: str = "s3://qdr-artifacts",
) -> ArtifactRef:
    """Write an artifact to a local path and return its reference.

    Used by the worker to stage artifacts before upload, and by tests.
    """
    ref, data = build_artifact_ref(payload, uri_prefix=uri_prefix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return ref


def build_run_manifest(
    run_id: str,
    input_hash: str,
    version_pins: dict[str, str],
    seed_record: dict[str, Any],
    stage_timings_ms: dict[str, float],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the run manifest -- everything needed to reproduce and audit a run."""
    return {
        "run_id": run_id,
        "input_hash": input_hash,
        "versions": version_pins,
        "seeds": seed_record,
        "stage_timings_ms": stage_timings_ms,
        "result": result,
    }


def load_artifact(path: Path, expected_hash: str | None = None) -> dict[str, Any]:
    """Read an artifact, optionally verifying its content hash.

    A digest mismatch is corruption or tampering, not a recoverable condition, so it
    raises rather than returning the payload with a warning.
    """
    data = path.read_bytes()
    if expected_hash is not None:
        actual = hash_bytes(data)
        if actual != expected_hash:
            raise ValidationError(
                f"artifact at {path} has digest {actual}, expected {expected_hash}",
                path=str(path),
                expected=expected_hash,
                actual=actual,
            )
    payload: dict[str, Any] = json.loads(data)
    return payload

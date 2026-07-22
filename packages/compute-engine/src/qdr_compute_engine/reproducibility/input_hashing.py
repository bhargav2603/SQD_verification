"""Input hashing for run provenance.

Delegates to :mod:`qdr_core.provenance.content_hash` -- this module owns *normalisation*
(turning engine domain objects into JSON-native values) and nothing else. The hashing
itself has exactly one implementation in the platform, by design (ADR-0007).

Float normalisation is the interesting part. Raw coordinates carry noise in their final
bits, so hashing them verbatim makes two runs on the same structure look like different
inputs. Rounding to a fixed number of decimals makes the hash stable against that noise
while staying far finer than any physically meaningful difference.
"""

from typing import Any

import numpy as np

from qdr_core.provenance.content_hash import content_hash

#: Decimals retained when normalising floats. 12 is well below double precision but far
#: beyond the ~4 decimals at which coordinates or energies carry physical meaning.
FLOAT_DECIMALS = 12


def normalize(value: Any) -> Any:
    """Recursively convert a value into a canonical JSON-native form."""
    if isinstance(value, np.ndarray):
        return normalize(value.tolist())
    if isinstance(value, np.generic):
        return normalize(value.item())
    if isinstance(value, bool | int | str) or value is None:
        return value
    if isinstance(value, float):
        # Normalise -0.0 to 0.0: they are numerically equal but distinct JSON tokens.
        rounded = round(value, FLOAT_DECIMALS)
        return rounded + 0.0
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [normalize(v) for v in value]
    raise TypeError(f"cannot normalize {type(value).__name__} for hashing")


def hash_inputs(payload: dict[str, Any]) -> str:
    """Content hash of a normalised input payload."""
    return content_hash(normalize(payload))


def build_run_input_payload(
    *,
    geometry: dict[str, Any],
    config: dict[str, Any],
    basis: str,
    backend: str,
    mapping: str,
    seed: int,
) -> dict[str, Any]:
    """Assemble the inputs that define a run's identity.

    Deliberately excludes the run id, timestamps and version pins: two runs of the same
    science must hash identically so results can be deduplicated and cache-hit. Versions
    are recorded separately in the manifest, where they describe *how* the answer was
    produced rather than *what* was asked.
    """
    return {
        "geometry": geometry,
        "config": config,
        "basis": basis,
        "backend": backend,
        "mapping": mapping,
        "seed": seed,
    }

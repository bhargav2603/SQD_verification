"""Deterministic RNG management.

One run seed derives every stream the pipeline needs, via ``SeedSequence.spawn``. Sharing
a single global generator would make results depend on execution order, so a stage that
starts drawing numbers would silently change every later stage's output. Named streams
also mean a stage can be added or removed without renumbering anyone else's seed.
"""

from typing import Any

import numpy as np

#: Stable ordering of named streams. Appending is safe; reordering or removing an entry
#: changes the seeds of every stream after it and breaks reproducibility of past runs.
STREAM_NAMES = (
    "ansatz_init",
    "optimizer",
    "sampling",
    "fragmentation",
)


def derive_generators(run_seed: int) -> dict[str, np.random.Generator]:
    """Derive one independent generator per named stream from a single run seed."""
    root = np.random.SeedSequence(run_seed)
    children = root.spawn(len(STREAM_NAMES))
    return {
        name: np.random.default_rng(child)
        for name, child in zip(STREAM_NAMES, children, strict=True)
    }


def generator_for(run_seed: int, stream: str) -> np.random.Generator:
    """Generator for one named stream."""
    if stream not in STREAM_NAMES:
        raise ValueError(f"unknown RNG stream {stream!r}; known streams: {list(STREAM_NAMES)}")
    return derive_generators(run_seed)[stream]


def seed_record(run_seed: int) -> dict[str, Any]:
    """The seed metadata written into the run manifest."""
    return {"run_seed": int(run_seed), "streams": list(STREAM_NAMES)}

"""Projection-based WF-in-DFT embedding result (produced by pipeline stage 03b).

Populated only when ``config["embedding"]`` is set. For the R1 (non-embedded) path the
``StageContext.embedding`` field stays ``None``, which is what keeps those runs
byte-identical to a pipeline without the embedding stage.

The type is intentionally small at this stage: it records *which* scheme carved out the
active subsystem and any caveats, so a reader is never left to assume the active-space
integrals were embedded when they were not. It grows as the embedding stage grows (the
orbital partition, the level-shift projector, the natural-orbital-occupation diagnostic).

Re-exported by ``qdr_compute_engine.types`` -- that module remains the one import site for
the whole engine.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """How the active subsystem was carved out of the supersystem.

    ``scheme`` names the embedding actually performed (e.g. ``"projection-wf-in-dft"``), so
    downstream reporting can never pass off an un-embedded number as an embedded one --
    the same discipline the FMO assembly applies to its ``fmo2-no-esp`` label.
    """

    scheme: str
    active_atom_indices: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

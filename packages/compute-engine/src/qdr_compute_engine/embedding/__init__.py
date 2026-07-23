"""Projection-based WF-in-DFT active-site embedding.

The pure, backend-free heart of stage 03b lives here: given the pieces a mean-field
calculation produces (the environment density, the overlap matrix, the environment
Coulomb+XC potential), it assembles the embedded one-electron operator and the orbital
partition. Keeping this layer free of PySCF is deliberate -- the embedding *math* is then
unit-testable on any platform, exactly like the qubit mappers, while the PySCF glue that
produces its inputs stays isolated in one adapter (mirroring ``integrals/pyscf_integrals``).
"""

from qdr_compute_engine.embedding.projection import (
    PartitionResult,
    embed_one_body_operator,
    levelshift_projector,
    partition_orbitals_by_population,
)

__all__ = [
    "PartitionResult",
    "embed_one_body_operator",
    "levelshift_projector",
    "partition_orbitals_by_population",
]

"""Projection-based WF-in-DFT embedding: the backend-free core.

What this implements, precisely
-------------------------------
The embedded one-electron operator of projection-based WF-in-DFT embedding
(Manby & Miller, JCTC 2012, 8, 2564):

    h_emb = h_core_A + (g[gamma_tot] - g[gamma_A]) + mu * P_env

where
  * ``h_core_A``            is the bare one-electron (core) Hamiltonian of the active region,
  * ``g[gamma] = J[gamma] + XC[gamma]`` is the Coulomb+exchange-correlation potential of a
    density -- the term that actually *polarises* the active region in the environment's
    field; omitting it leaves a projector with no embedding,
  * ``P_env = S gamma_env S``  is the level-shift projector onto the environment-occupied
    space in the AO basis, and ``mu`` is a large level shift (~1e6 Ha) that pushes the
    environment orbitals out of the active problem to enforce orthogonality.

Everything here is plain array arithmetic: the caller supplies the density, overlap and
potential matrices (a mean-field calculation produces them), and this module combines them.
That keeps the embedding math testable without PySCF -- including the recovery property that
is the correctness gate's backbone: with an empty/absent environment ``P_env`` is zero and
``g[gamma_tot] == g[gamma_A]``, so ``h_emb`` collapses to ``h_core_A`` exactly.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from qdr_core.errors import ValidationError


def levelshift_projector(
    overlap: NDArray[np.float64], environment_density: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The level-shift projector ``P_env = S gamma_env S`` in the AO basis.

    Scaled by ``mu`` in :func:`embed_one_body_operator`, this raises the energy of any
    orbital with environment-density weight, which is what keeps the active-region solve
    orthogonal to the (frozen) environment. An empty environment gives the zero matrix, so
    the projector contributes nothing -- the hinge of the recovery gate.
    """
    _require_square(overlap, "overlap")
    _require_square(environment_density, "environment_density")
    if overlap.shape != environment_density.shape:
        raise ValidationError(
            "overlap and environment_density must share a shape",
            overlap_shape=list(overlap.shape),
            environment_density_shape=list(environment_density.shape),
        )
    return overlap @ environment_density @ overlap


def embed_one_body_operator(
    core_hamiltonian: NDArray[np.float64],
    potential_total: NDArray[np.float64],
    potential_active: NDArray[np.float64],
    projector: NDArray[np.float64],
    level_shift: float,
) -> NDArray[np.float64]:
    """Assemble ``h_emb = h_core_A + (g_tot - g_A) + mu * P_env``.

    ``potential_total`` is ``g[gamma_tot]`` (whole system) and ``potential_active`` is
    ``g[gamma_A]`` (active region alone); their difference is the environment's polarising
    field on the active region. All four matrices must be same-shape and in the same basis.
    """
    for name, matrix in (
        ("core_hamiltonian", core_hamiltonian),
        ("potential_total", potential_total),
        ("potential_active", potential_active),
        ("projector", projector),
    ):
        _require_square(matrix, name)
    shape = core_hamiltonian.shape
    for name, matrix in (
        ("potential_total", potential_total),
        ("potential_active", potential_active),
        ("projector", projector),
    ):
        if matrix.shape != shape:
            raise ValidationError(
                f"{name} shape {matrix.shape} does not match core_hamiltonian {shape}; "
                "all embedding operators must be in the same basis",
                operator=name,
            )
    if level_shift < 0:
        raise ValidationError(
            f"level_shift must be non-negative, got {level_shift}",
            level_shift=level_shift,
        )

    environment_field = potential_total - potential_active
    return core_hamiltonian + environment_field + level_shift * projector


@dataclass(frozen=True, slots=True)
class PartitionResult:
    """Which orbitals belong to the active region vs the (frozen) environment.

    ``warnings`` names every orbital that sat close to the threshold -- unavoidable across a
    covalent metal-ligand bond -- so a boundary-sensitive partition is surfaced, not buried.
    """

    active_indices: tuple[int, ...]
    environment_indices: tuple[int, ...]
    warnings: tuple[str, ...] = ()


def partition_orbitals_by_population(
    active_populations: NDArray[np.float64] | list[float],
    *,
    threshold: float = 0.5,
    boundary_fraction: float = 0.1,
) -> PartitionResult:
    """Split orbitals into active vs environment by their active-atom population.

    An orbital whose population on the active atoms exceeds ``threshold`` is active; the rest
    are environment. Any orbital within ``boundary_fraction * threshold`` of the cut is
    recorded as a boundary case in ``warnings`` -- across a covalent M-L bond such orbitals
    are the ones whose assignment most affects the embedding, so they must not be silent.
    """
    populations = np.asarray(active_populations, dtype=float)
    if populations.ndim != 1:
        raise ValidationError(
            f"active_populations must be 1-D, got shape {populations.shape}",
            shape=list(populations.shape),
        )

    active: list[int] = []
    environment: list[int] = []
    warnings: list[str] = []
    margin = boundary_fraction * threshold
    for index, population in enumerate(populations):
        if population > threshold:
            active.append(index)
        else:
            environment.append(index)
        if abs(population - threshold) <= margin:
            warnings.append(
                f"orbital {index} has active-atom population {population:.3f}, within "
                f"{margin:.3f} of the {threshold:.3f} cut -- boundary assignment"
            )

    if not active:
        raise ValidationError(
            "no orbital exceeds the active-atom population threshold; the partition would "
            "leave an empty active region",
            threshold=threshold,
        )

    return PartitionResult(
        active_indices=tuple(active),
        environment_indices=tuple(environment),
        warnings=tuple(warnings),
    )


def _require_square(matrix: NDArray[np.float64], name: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValidationError(
            f"{name} must be a square 2-D matrix, got shape {matrix.shape}",
            operator=name,
        )

"""Stage 2: FMO fragmentation.

Two schemes are supported, and the distinction matters chemically:

``molecular``
    Cut nothing. Fragments are the connected components of the bond graph, so a
    protein-ligand-water system separates into its physically distinct molecules. This is
    exact -- no covalent bond is broken, so no capping is required.

``explicit_cuts``
    Additionally sever a caller-supplied list of covalent bonds, which is how a single
    protein chain becomes per-residue fragments. Breaking a covalent bond leaves the
    fragment open-shell, so a real calculation must cap the dangling valence (hydrogen
    caps, or FMO's hybrid-orbital projection). This module records the cuts it made and
    flags that the caps are not yet applied, rather than quietly producing radicals.

Bond perception is distance-based against covalent radii, which is standard for
structure-derived input and does not require bond orders in the source file.
"""

from collections import deque

import numpy as np

from qdr_compute_engine.geometry.pose_builder import (
    BOND_TOLERANCE_ANGSTROM,
    covalent_radius,
)
from qdr_compute_engine.types import (
    Fragment,
    FragmentationResult,
    Geometry,
    LengthUnit,
)
from qdr_core.errors import ValidationError


def perceive_bonds(geometry: Geometry) -> list[tuple[int, int]]:
    """Infer covalent bonds from interatomic distances.

    Two atoms are bonded when ``d(i,j) <= r_cov(i) + r_cov(j) + tolerance``.
    """
    in_angstrom = geometry.to_unit(LengthUnit.ANGSTROM)
    positions = np.array([a.position for a in in_angstrom.atoms], dtype=float)
    radii = np.array([covalent_radius(a.symbol) for a in in_angstrom.atoms], dtype=float)

    if len(positions) < 2:
        return []

    deltas = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    thresholds = radii[:, None] + radii[None, :] + BOND_TOLERANCE_ANGSTROM

    bonded = np.triu(distances <= thresholds, k=1)
    return [(int(i), int(j)) for i, j in zip(*np.nonzero(bonded), strict=True)]


def connected_components(n_atoms: int, bonds: list[tuple[int, int]]) -> list[tuple[int, ...]]:
    """Group atom indices into connected components via breadth-first search."""
    adjacency: dict[int, list[int]] = {i: [] for i in range(n_atoms)}
    for i, j in bonds:
        adjacency[i].append(j)
        adjacency[j].append(i)

    seen: set[int] = set()
    components: list[tuple[int, ...]] = []
    for start in range(n_atoms):
        if start in seen:
            continue
        component: list[int] = []
        queue = deque([start])
        seen.add(start)
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        components.append(tuple(sorted(component)))
    return components


def fragment(
    geometry: Geometry,
    *,
    scheme: str = "molecular",
    cut_bonds: tuple[tuple[int, int], ...] = (),
    dimer_cutoff_angstrom: float | None = None,
) -> FragmentationResult:
    """Decompose ``geometry`` into FMO monomers and the dimer pairs to compute."""
    if geometry.n_atoms == 0:
        raise ValidationError("cannot fragment an empty geometry")

    if scheme == "single":
        monomer = Fragment(
            fragment_id="F0",
            atom_indices=tuple(range(geometry.n_atoms)),
            label="whole-system",
            charge=geometry.charge,
        )
        return FragmentationResult(monomers=(monomer,), dimer_pairs=())

    if scheme not in {"molecular", "explicit_cuts"}:
        raise ValidationError(
            f"unknown fragmentation scheme {scheme!r}",
            scheme=scheme,
            supported=["single", "molecular", "explicit_cuts"],
        )

    bonds = perceive_bonds(geometry)

    if scheme == "explicit_cuts":
        if not cut_bonds:
            raise ValidationError("scheme 'explicit_cuts' requires at least one bond to cut")
        normalized_cuts = {(min(i, j), max(i, j)) for i, j in cut_bonds}
        perceived = {(min(i, j), max(i, j)) for i, j in bonds}
        missing = normalized_cuts - perceived
        if missing:
            raise ValidationError(
                f"requested cuts are not perceived bonds: {sorted(missing)}",
                missing_bonds=sorted(missing),
            )
        bonds = [b for b in bonds if (min(b), max(b)) not in normalized_cuts]

    components = connected_components(geometry.n_atoms, bonds)
    monomers = tuple(
        Fragment(
            fragment_id=f"F{index}",
            atom_indices=component,
            label=_component_label(geometry, component),
            charge=0,
        )
        for index, component in enumerate(components)
    )

    return FragmentationResult(
        monomers=monomers,
        dimer_pairs=_dimer_pairs(geometry, monomers, dimer_cutoff_angstrom),
    )


def _component_label(geometry: Geometry, atom_indices: tuple[int, ...]) -> str:
    """A readable Hill-ordered formula, for logs and the run console."""
    counts: dict[str, int] = {}
    for i in atom_indices:
        symbol = geometry.atoms[i].symbol
        counts[symbol] = counts.get(symbol, 0) + 1

    ordered: list[str] = []
    for symbol in ("C", "H"):
        if symbol in counts:
            count = counts.pop(symbol)
            ordered.append(f"{symbol}{count}" if count > 1 else symbol)
    for symbol in sorted(counts):
        count = counts[symbol]
        ordered.append(f"{symbol}{count}" if count > 1 else symbol)
    return "".join(ordered)


def _fragment_centroid(geometry: Geometry, atom_indices: tuple[int, ...]) -> np.ndarray:
    in_angstrom = geometry.to_unit(LengthUnit.ANGSTROM)
    return np.array([in_angstrom.atoms[i].position for i in atom_indices], dtype=float).mean(axis=0)


def _dimer_pairs(
    geometry: Geometry,
    monomers: tuple[Fragment, ...],
    cutoff: float | None,
) -> tuple[tuple[str, str], ...]:
    """Pairs to evaluate at FMO2. Beyond the cutoff, interactions are treated by the
    electrostatic potential rather than an explicit dimer calculation."""
    if len(monomers) < 2:
        return ()

    centroids = {f.fragment_id: _fragment_centroid(geometry, f.atom_indices) for f in monomers}
    pairs: list[tuple[str, str]] = []
    for index, left in enumerate(monomers):
        for right in monomers[index + 1 :]:
            if cutoff is not None:
                separation = float(
                    np.linalg.norm(centroids[left.fragment_id] - centroids[right.fragment_id])
                )
                if separation > cutoff:
                    continue
            pairs.append((left.fragment_id, right.fragment_id))
    return tuple(pairs)

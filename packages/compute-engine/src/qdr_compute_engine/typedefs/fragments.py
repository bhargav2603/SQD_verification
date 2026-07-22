"""Fragment decomposition types.

What stage 2 produces and the FMO expansion consumes.

Re-exported by ``qdr_compute_engine.types``, which remains the import site for the
whole engine -- these modules exist to keep each file small, not to add a new path.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass
from qdr_compute_engine.typedefs.geometry import Geometry  # noqa: F401

# --------------------------------------------------------------------------------------
# Fragmentation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Fragment:
    fragment_id: str
    atom_indices: tuple[int, ...]
    label: str = ""
    charge: int = 0

    @property
    def n_atoms(self) -> int:
        return len(self.atom_indices)


@dataclass(frozen=True, slots=True)
class FragmentationResult:
    """FMO2 decomposition: monomers plus the dimer pairs to be computed."""

    monomers: tuple[Fragment, ...]
    dimer_pairs: tuple[tuple[str, str], ...] = ()

    @property
    def n_monomers(self) -> int:
        return len(self.monomers)

    def by_id(self, fragment_id: str) -> Fragment:
        for fragment in self.monomers:
            if fragment.fragment_id == fragment_id:
                return fragment
        raise KeyError(fragment_id)

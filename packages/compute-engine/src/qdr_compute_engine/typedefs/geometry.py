"""Geometry and units.

Atoms, molecular geometry, and the Angstrom/Bohr conversion.

Re-exported by ``qdr_compute_engine.types``, which remains the import site for the
whole engine -- these modules exist to keep each file small, not to add a new path.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------


class LengthUnit(StrEnum):
    ANGSTROM = "angstrom"
    BOHR = "bohr"


#: CODATA-2018 Bohr radius in Angstrom.
BOHR_IN_ANGSTROM = 0.529177210903


@dataclass(frozen=True, slots=True)
class Atom:
    symbol: str
    position: tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.position) != 3:
            raise ValueError(f"position must be 3 coordinates, got {len(self.position)}")


@dataclass(frozen=True, slots=True)
class Geometry:
    """A molecular geometry with explicit units and charge/spin state."""

    atoms: tuple[Atom, ...]
    charge: int = 0
    spin_multiplicity: int = 1
    units: LengthUnit = LengthUnit.ANGSTROM

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    def to_unit(self, units: LengthUnit) -> "Geometry":
        """Return this geometry expressed in ``units`` (identity if already there)."""
        if units == self.units:
            return self
        factor = BOHR_IN_ANGSTROM if units == LengthUnit.ANGSTROM else 1.0 / BOHR_IN_ANGSTROM
        scaled = tuple(
            Atom(a.symbol, (a.position[0] * factor, a.position[1] * factor, a.position[2] * factor))
            for a in self.atoms
        )
        return Geometry(scaled, self.charge, self.spin_multiplicity, units)

    def subset(self, atom_indices: tuple[int, ...]) -> "Geometry":
        """Extract a sub-geometry. Charge/spin are *not* inherited -- a fragment's charge
        is a property of the fragmentation scheme, not of the parent molecule."""
        return Geometry(
            atoms=tuple(self.atoms[i] for i in atom_indices),
            charge=0,
            spin_multiplicity=1,
            units=self.units,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "atoms": [{"symbol": a.symbol, "position": list(a.position)} for a in self.atoms],
            "charge": self.charge,
            "spin_multiplicity": self.spin_multiplicity,
            "units": str(self.units),
        }

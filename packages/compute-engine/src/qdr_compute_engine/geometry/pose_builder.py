"""Stage 1 support: geometry ingest, validation and pose preparation.

Also the home of the small element table the engine needs (atomic numbers for electron
counting, covalent radii for bond perception in the fragmenter). It covers the elements
that actually appear in protein-ligand systems; an unknown symbol is a hard error rather
than a guessed default, because a silently wrong atomic number changes the electron count
and therefore every energy downstream.
"""

import math

import numpy as np

from qdr_compute_engine.types import Atom, Geometry, LengthUnit
from qdr_core.errors import ValidationError

ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Se": 34,
    "Br": 35,
    "I": 53,
}

#: Covalent radii in Angstrom (Cordero et al., 2008). Used only for bond perception.
COVALENT_RADII: dict[str, float] = {
    "H": 0.31,
    "He": 0.28,
    "Li": 1.28,
    "Be": 0.96,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Ne": 0.58,
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Ar": 1.06,
    "K": 2.03,
    "Ca": 1.76,
    "Mn": 1.39,
    "Fe": 1.32,
    "Co": 1.26,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Se": 1.20,
    "Br": 1.20,
    "I": 1.39,
}

#: Two atoms are bonded when their separation is under the sum of covalent radii plus
#: this tolerance. The standard value in structure-perception codes.
BOND_TOLERANCE_ANGSTROM = 0.40

#: Below this separation, two atoms are considered coincident -- almost always a broken
#: input file rather than a real structure.
MIN_ATOM_SEPARATION_ANGSTROM = 0.30


def normalize_symbol(symbol: str) -> str:
    """Canonicalise an element symbol ('FE', 'fe' -> 'Fe')."""
    cleaned = symbol.strip()
    if not cleaned:
        raise ValidationError("empty element symbol")
    return cleaned[0].upper() + cleaned[1:].lower()


def atomic_number(symbol: str) -> int:
    normalized = normalize_symbol(symbol)
    try:
        return ATOMIC_NUMBERS[normalized]
    except KeyError:
        raise ValidationError(
            f"unsupported element {normalized!r}; add it to ATOMIC_NUMBERS if it is real",
            symbol=normalized,
        ) from None


def covalent_radius(symbol: str) -> float:
    normalized = normalize_symbol(symbol)
    try:
        return COVALENT_RADII[normalized]
    except KeyError:
        raise ValidationError(f"no covalent radius for {normalized!r}", symbol=normalized) from None


def count_electrons(geometry: Geometry) -> int:
    """Total electron count: sum of atomic numbers, less the net charge."""
    total = sum(atomic_number(a.symbol) for a in geometry.atoms) - geometry.charge
    if total < 0:
        raise ValidationError(
            f"geometry has a negative electron count ({total}); check the charge",
            charge=geometry.charge,
        )
    return total


def parse_xyz(text: str) -> Geometry:
    """Parse a standard XYZ block.

    Line 1 is the atom count, line 2 a free-form comment, then one ``symbol x y z`` per
    line. The declared count is enforced -- a mismatch means a truncated file, and
    silently accepting it would compute the energy of a different molecule.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 1:
        raise ValidationError("empty XYZ input")

    try:
        declared = int(lines[0].split()[0])
    except (ValueError, IndexError):
        raise ValidationError(f"XYZ line 1 must be an atom count, got {lines[0]!r}") from None

    atom_lines = lines[2:] if len(lines) > 1 else []
    if len(atom_lines) != declared:
        raise ValidationError(
            f"XYZ declares {declared} atoms but carries {len(atom_lines)}",
            declared=declared,
            found=len(atom_lines),
        )

    atoms = []
    for lineno, line in enumerate(atom_lines, start=3):
        parts = line.split()
        if len(parts) < 4:
            raise ValidationError(f"XYZ line {lineno} is malformed: {line!r}")
        try:
            coords = (float(parts[1]), float(parts[2]), float(parts[3]))
        except ValueError:
            raise ValidationError(f"XYZ line {lineno} has non-numeric coordinates") from None
        atoms.append(Atom(normalize_symbol(parts[0]), coords))

    return Geometry(atoms=tuple(atoms), units=LengthUnit.ANGSTROM)


def parse_pdb(text: str, charge: int = 0) -> Geometry:
    """Parse ATOM/HETATM records from a PDB block.

    Column-based per the PDB spec rather than whitespace-split: PDB fields run together
    when coordinates are wide, so splitting on spaces mis-parses real files.
    """
    atoms = []
    for line in text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except (ValueError, IndexError):
            raise ValidationError(f"malformed PDB coordinate columns: {line[:54]!r}") from None

        symbol = line[76:78].strip() if len(line) >= 78 else ""
        if not symbol:
            # Fall back to the atom-name column, whose first character is the element
            # for standard biomolecular naming (CA -> C, but also HG1 -> H).
            symbol = line[12:16].strip()[:1]
        atoms.append(Atom(normalize_symbol(symbol), (x, y, z)))

    if not atoms:
        raise ValidationError("PDB input contains no ATOM or HETATM records")
    return Geometry(atoms=tuple(atoms), charge=charge, units=LengthUnit.ANGSTROM)


def validate_geometry(geometry: Geometry) -> None:
    """Reject geometries that cannot be meaningfully computed on."""
    if geometry.n_atoms == 0:
        raise ValidationError("geometry has no atoms")

    for atom in geometry.atoms:
        atomic_number(atom.symbol)  # raises on unknown elements
        if not all(math.isfinite(c) for c in atom.position):
            raise ValidationError(f"non-finite coordinate on {atom.symbol}: {atom.position}")

    if geometry.spin_multiplicity < 1:
        raise ValidationError(
            f"spin multiplicity must be >= 1, got {geometry.spin_multiplicity}",
        )

    n_electrons = count_electrons(geometry)
    n_unpaired = geometry.spin_multiplicity - 1
    if (n_electrons - n_unpaired) % 2 != 0:
        raise ValidationError(
            f"{n_electrons} electrons is inconsistent with spin multiplicity "
            f"{geometry.spin_multiplicity}",
            n_electrons=n_electrons,
            spin_multiplicity=geometry.spin_multiplicity,
        )

    in_angstrom = geometry.to_unit(LengthUnit.ANGSTROM)
    positions = np.array([a.position for a in in_angstrom.atoms])
    if len(positions) > 1:
        deltas = positions[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(deltas, axis=-1)
        np.fill_diagonal(distances, np.inf)
        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        if distances[i, j] < MIN_ATOM_SEPARATION_ANGSTROM:
            raise ValidationError(
                f"atoms {i} and {j} are {distances[i, j]:.3f} A apart, effectively coincident",
                atom_indices=[int(i), int(j)],
                separation=float(distances[i, j]),
            )


def require_closed_shell(geometry: Geometry) -> None:
    """Reject anything the engine's closed-shell machinery cannot honestly compute.

    Three separate components assume a spin-restricted, doubly-occupied reference:

      * ``integrals/pyscf_integrals.py`` builds the reference with ``scf.RHF``;
      * ``active_space_selector`` computes ``n_frozen_core = (n_el - n_act) // 2``, which
        is only the core count when every core orbital holds two electrons;
      * ``fermionic_builder.hartree_fock_occupations`` fills the *lowest* ``n_electrons``
        spin orbitals, which under interleaved indexing is the closed-shell determinant.

    None of the three fails loudly on an open-shell input. Worse, PySCF returns an
    **ROHF** object from ``scf.RHF(mol)`` when ``mol.spin != 0`` rather than raising, so
    an unrestricted system would flow through the whole pipeline and produce a
    plausible-looking energy for the wrong reference state.

    Open-shell support means UHF/ROHF integrals, a spin-adapted active space and a
    reference state that is not a simple row of X gates. Until all three exist, this is a
    hard error -- an explicit refusal is worth more than a number nobody can trust.
    """
    if geometry.spin_multiplicity != 1:
        raise ValidationError(
            f"spin multiplicity {geometry.spin_multiplicity} is not supported: this "
            "engine is closed-shell only (RHF reference, doubly-occupied core, "
            "Hartree-Fock determinant as a product state). Open-shell systems require "
            "UHF/ROHF integrals and a spin-adapted active space, which are not "
            "implemented.",
            spin_multiplicity=geometry.spin_multiplicity,
            supported_spin_multiplicity=1,
        )

    n_electrons = count_electrons(geometry)
    if n_electrons % 2 != 0:
        raise ValidationError(
            f"{n_electrons} electrons cannot form a closed-shell singlet; check the "
            "charge. This engine is closed-shell only.",
            n_electrons=n_electrons,
            charge=geometry.charge,
        )


def center_of_mass_positions(geometry: Geometry) -> np.ndarray:
    """Positions re-expressed relative to the centroid (unweighted)."""
    positions = np.array([a.position for a in geometry.atoms])
    return positions - positions.mean(axis=0)


def center_geometry(geometry: Geometry) -> Geometry:
    """Translate so the centroid sits at the origin.

    Translation-invariant by construction, so this never changes the energy; it only
    keeps coordinates small and comparable across poses.
    """
    centered = center_of_mass_positions(geometry)
    atoms = tuple(
        Atom(a.symbol, (float(p[0]), float(p[1]), float(p[2])))
        for a, p in zip(geometry.atoms, centered, strict=True)
    )
    return Geometry(atoms, geometry.charge, geometry.spin_multiplicity, geometry.units)


def build_pose(
    geometry: Geometry,
    *,
    center: bool = True,
    units: LengthUnit = LengthUnit.ANGSTROM,
) -> Geometry:
    """Validate and normalise a geometry into the engine's canonical pose."""
    validate_geometry(geometry)
    prepared = geometry.to_unit(units)
    if center:
        prepared = center_geometry(prepared)
    return prepared


def to_pyscf_atom_spec(geometry: Geometry) -> list[tuple[str, tuple[float, float, float]]]:
    """Convert to PySCF's ``atom`` argument form.

    Lives here rather than in the PySCF module so it stays importable (and testable)
    on platforms with no PySCF wheel.
    """
    in_angstrom = geometry.to_unit(LengthUnit.ANGSTROM)
    return [(a.symbol, a.position) for a in in_angstrom.atoms]

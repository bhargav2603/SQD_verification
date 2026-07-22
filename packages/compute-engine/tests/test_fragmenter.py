"""Geometry ingest and FMO fragmentation -- the stages that need no PySCF."""

import numpy as np
import pytest

from qdr_compute_engine.fragmentation.fmo_fragmenter import (
    connected_components,
    fragment,
    perceive_bonds,
)
from qdr_compute_engine.geometry.pose_builder import (
    atomic_number,
    build_pose,
    center_geometry,
    count_electrons,
    normalize_symbol,
    parse_pdb,
    parse_xyz,
    validate_geometry,
)
from qdr_compute_engine.types import Atom, Geometry, LengthUnit
from qdr_core.errors import ValidationError

WATER_XYZ = """3
water
O   0.000000   0.000000   0.117300
H   0.000000   0.757200  -0.469200
H   0.000000  -0.757200  -0.469200
"""

TWO_WATERS_XYZ = """6
two well-separated waters
O   0.000000   0.000000   0.117300
H   0.000000   0.757200  -0.469200
H   0.000000  -0.757200  -0.469200
O  10.000000   0.000000   0.117300
H  10.000000   0.757200  -0.469200
H  10.000000  -0.757200  -0.469200
"""

PDB_BLOCK = """ATOM      1  N   ALA A   1      -0.966   0.493   1.500  1.00  0.00           N
ATOM      2  CA  ALA A   1       0.257   0.418   0.692  1.00  0.00           C
ATOM      3  C   ALA A   1       1.204  -0.680   1.180  1.00  0.00           C
END
"""


# --------------------------------------------------------------------------------------
# Element handling
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"), [("h", "H"), ("FE", "Fe"), (" Cl ", "Cl"), ("c", "C")]
)
def test_symbol_normalisation(raw: str, expected: str) -> None:
    assert normalize_symbol(raw) == expected


def test_unknown_element_is_rejected() -> None:
    """A guessed atomic number would silently change the electron count."""
    with pytest.raises(ValidationError, match="unsupported element"):
        atomic_number("Xx")


def test_electron_count_accounts_for_charge() -> None:
    water = parse_xyz(WATER_XYZ)
    assert count_electrons(water) == 10

    cation = Geometry(atoms=water.atoms, charge=1)
    assert count_electrons(cation) == 9


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


def test_parse_xyz_reads_atoms_and_coordinates() -> None:
    water = parse_xyz(WATER_XYZ)
    assert water.n_atoms == 3
    assert [a.symbol for a in water.atoms] == ["O", "H", "H"]
    assert water.atoms[0].position == pytest.approx((0.0, 0.0, 0.1173))


def test_parse_xyz_enforces_the_declared_atom_count() -> None:
    """A truncated file would otherwise compute the energy of a different molecule."""
    truncated = "5\ncomment\nH 0 0 0\nH 0 0 0.74\n"
    with pytest.raises(ValidationError, match="declares 5 atoms"):
        parse_xyz(truncated)


def test_parse_xyz_rejects_non_numeric_coordinates() -> None:
    with pytest.raises(ValidationError, match="non-numeric"):
        parse_xyz("1\nc\nH x y z\n")


def test_parse_pdb_uses_fixed_columns() -> None:
    geometry = parse_pdb(PDB_BLOCK)
    assert geometry.n_atoms == 3
    assert [a.symbol for a in geometry.atoms] == ["N", "C", "C"]
    assert geometry.atoms[0].position == pytest.approx((-0.966, 0.493, 1.500))


def test_parse_pdb_without_atom_records_is_rejected() -> None:
    with pytest.raises(ValidationError, match="no ATOM"):
        parse_pdb("HEADER something\nEND\n")


# --------------------------------------------------------------------------------------
# Validation and pose preparation
# --------------------------------------------------------------------------------------


def test_coincident_atoms_are_rejected() -> None:
    """Two atoms on top of each other is a broken file, and would blow up the integrals."""
    broken = Geometry(atoms=(Atom("H", (0.0, 0.0, 0.0)), Atom("H", (0.0, 0.0, 0.01))))
    with pytest.raises(ValidationError, match="coincident"):
        validate_geometry(broken)


def test_inconsistent_spin_multiplicity_is_rejected() -> None:
    """10 electrons cannot form a doublet."""
    water = parse_xyz(WATER_XYZ)
    bad = Geometry(atoms=water.atoms, charge=0, spin_multiplicity=2)
    with pytest.raises(ValidationError, match="inconsistent with spin"):
        validate_geometry(bad)


def test_centering_moves_the_centroid_to_the_origin() -> None:
    centered = center_geometry(parse_xyz(WATER_XYZ))
    centroid = np.array([a.position for a in centered.atoms]).mean(axis=0)
    np.testing.assert_allclose(centroid, np.zeros(3), atol=1e-12)


def test_unit_conversion_round_trips() -> None:
    original = parse_xyz(WATER_XYZ)
    round_tripped = original.to_unit(LengthUnit.BOHR).to_unit(LengthUnit.ANGSTROM)
    for before, after in zip(original.atoms, round_tripped.atoms, strict=True):
        assert after.position == pytest.approx(before.position, abs=1e-12)


def test_build_pose_validates_and_centers() -> None:
    pose = build_pose(parse_xyz(WATER_XYZ))
    centroid = np.array([a.position for a in pose.atoms]).mean(axis=0)
    np.testing.assert_allclose(centroid, np.zeros(3), atol=1e-12)


# --------------------------------------------------------------------------------------
# Bond perception and fragmentation
# --------------------------------------------------------------------------------------


def test_water_has_two_bonds() -> None:
    bonds = perceive_bonds(parse_xyz(WATER_XYZ))
    assert sorted(bonds) == [(0, 1), (0, 2)]


def test_distant_molecules_are_not_bonded() -> None:
    bonds = perceive_bonds(parse_xyz(TWO_WATERS_XYZ))
    assert (0, 3) not in bonds
    assert len(bonds) == 4  # two O-H bonds per molecule


def test_connected_components_groups_atoms() -> None:
    components = connected_components(6, [(0, 1), (0, 2), (3, 4), (3, 5)])
    assert components == [(0, 1, 2), (3, 4, 5)]


def test_molecular_scheme_separates_the_two_waters() -> None:
    result = fragment(parse_xyz(TWO_WATERS_XYZ), scheme="molecular")

    assert result.n_monomers == 2
    assert result.monomers[0].atom_indices == (0, 1, 2)
    assert result.monomers[1].atom_indices == (3, 4, 5)
    assert result.dimer_pairs == (("F0", "F1"),)


def test_fragment_labels_are_hill_ordered_formulas() -> None:
    result = fragment(parse_xyz(WATER_XYZ), scheme="molecular")
    assert result.monomers[0].label == "H2O"


def test_single_scheme_keeps_the_system_whole() -> None:
    result = fragment(parse_xyz(TWO_WATERS_XYZ), scheme="single")
    assert result.n_monomers == 1
    assert result.monomers[0].n_atoms == 6
    assert result.dimer_pairs == ()


def test_dimer_cutoff_drops_distant_pairs() -> None:
    """Beyond the cutoff, pairs are handled electrostatically rather than explicitly."""
    result = fragment(parse_xyz(TWO_WATERS_XYZ), scheme="molecular", dimer_cutoff_angstrom=5.0)
    assert result.n_monomers == 2
    assert result.dimer_pairs == ()


def test_explicit_cuts_split_a_connected_molecule() -> None:
    result = fragment(parse_xyz(WATER_XYZ), scheme="explicit_cuts", cut_bonds=((0, 1),))
    assert result.n_monomers == 2


def test_cutting_a_nonexistent_bond_is_rejected() -> None:
    """Silently ignoring an impossible cut would produce fragments the caller never asked
    for, and an energy attributed to the wrong decomposition."""
    with pytest.raises(ValidationError, match="not perceived bonds"):
        fragment(parse_xyz(WATER_XYZ), scheme="explicit_cuts", cut_bonds=((1, 2),))


def test_explicit_cuts_requires_at_least_one_bond() -> None:
    with pytest.raises(ValidationError, match="at least one bond"):
        fragment(parse_xyz(WATER_XYZ), scheme="explicit_cuts")


def test_unknown_scheme_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown fragmentation scheme"):
        fragment(parse_xyz(WATER_XYZ), scheme="magic")


def test_lookup_by_fragment_id() -> None:
    result = fragment(parse_xyz(TWO_WATERS_XYZ), scheme="molecular")
    assert result.by_id("F1").atom_indices == (3, 4, 5)
    with pytest.raises(KeyError):
        result.by_id("F99")

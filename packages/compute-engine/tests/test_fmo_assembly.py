"""FMO2 assembly: partitioning, charges, geometry construction and the expansion.

None of this needs PySCF, and all of it is where a silent error would be most expensive:
a dropped atom, an inherited charge or a mis-assembled expansion all produce a number that
looks like an energy. The subsystem *solves* need PySCF and are covered separately.
"""

import pytest

from qdr_compute_engine.fragmentation.fmo_assembly import (
    SCHEME,
    FMOResult,
    FragmentEnergy,
    assemble,
    build_plans,
    dimer_geometry,
    monomer_geometry,
    resolve_fragment_charges,
    scheme_warnings,
    validate_fragmentation,
)
from qdr_compute_engine.pipeline.fmo_runner import subsystem_config
from qdr_compute_engine.types import (
    Atom,
    Fragment,
    FragmentationResult,
    Geometry,
    LengthUnit,
)
from qdr_core.errors import ValidationError

# Two well-separated water molecules: the canonical two-fragment test case.
WATER_DIMER = Geometry(
    atoms=(
        Atom("O", (0.000, 0.000, 0.000)),
        Atom("H", (0.757, 0.586, 0.000)),
        Atom("H", (-0.757, 0.586, 0.000)),
        Atom("O", (0.000, 0.000, 2.900)),
        Atom("H", (0.757, -0.586, 2.900)),
        Atom("H", (-0.757, -0.586, 2.900)),
    ),
    charge=0,
    spin_multiplicity=1,
    units=LengthUnit.ANGSTROM,
)

FRAG_A = Fragment("F0", (0, 1, 2), label="H2O", charge=0)
FRAG_B = Fragment("F1", (3, 4, 5), label="H2O", charge=0)
DIMER_FRAGMENTATION = FragmentationResult(monomers=(FRAG_A, FRAG_B), dimer_pairs=(("F0", "F1"),))


# ---------------------------------------------------------------------------------
# Partition validity
# ---------------------------------------------------------------------------------


def test_valid_partition_accepted() -> None:
    validate_fragmentation(DIMER_FRAGMENTATION, WATER_DIMER)


def test_dropped_atom_is_rejected() -> None:
    """An incomplete partition computes a different molecule."""
    partial = FragmentationResult(monomers=(FRAG_A, Fragment("F1", (3, 4))))
    with pytest.raises(ValidationError, match="belong to no fragment"):
        validate_fragmentation(partial, WATER_DIMER)


def test_double_counted_atom_is_rejected() -> None:
    overlapping = FragmentationResult(
        monomers=(Fragment("F0", (0, 1, 2)), Fragment("F1", (2, 3, 4, 5)))
    )
    with pytest.raises(ValidationError, match="must be disjoint"):
        validate_fragmentation(overlapping, WATER_DIMER)


def test_empty_fragmentation_is_rejected() -> None:
    with pytest.raises(ValidationError, match="no monomers"):
        validate_fragmentation(FragmentationResult(monomers=()), WATER_DIMER)


# ---------------------------------------------------------------------------------
# Charges
# ---------------------------------------------------------------------------------


def test_neutral_default() -> None:
    assert resolve_fragment_charges(DIMER_FRAGMENTATION, 0) == {"F0": 0, "F1": 0}


def test_explicit_charges_must_sum_to_the_system_charge() -> None:
    charges = resolve_fragment_charges(
        DIMER_FRAGMENTATION, total_charge=0, requested={"F0": 1, "F1": -1}
    )
    assert charges == {"F0": 1, "F1": -1}


def test_charges_that_do_not_add_up_are_rejected() -> None:
    """A wrong fragment charge changes the electron count and nothing downstream notices."""
    with pytest.raises(ValidationError, match="sum to"):
        resolve_fragment_charges(DIMER_FRAGMENTATION, total_charge=0, requested={"F0": 1})


def test_charged_system_requires_explicit_assignment() -> None:
    """Defaulting every fragment to neutral cannot describe a charged system."""
    with pytest.raises(ValidationError, match="fragment_charges"):
        resolve_fragment_charges(DIMER_FRAGMENTATION, total_charge=-1)


def test_unknown_fragment_in_charges_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown fragments"):
        resolve_fragment_charges(DIMER_FRAGMENTATION, total_charge=0, requested={"F9": 0})


# ---------------------------------------------------------------------------------
# Sub-geometry construction
# ---------------------------------------------------------------------------------


def test_monomer_geometry_keeps_parent_coordinates() -> None:
    """A fragment must sit where it sits, or the pair interaction describes something else."""
    monomer = monomer_geometry(WATER_DIMER, FRAG_B, charge=0)
    assert monomer.n_atoms == 3
    assert monomer.atoms[0].symbol == "O"
    assert monomer.atoms[0].position == pytest.approx((0.0, 0.0, 2.900))


def test_monomer_geometry_takes_its_own_charge() -> None:
    monomer = monomer_geometry(WATER_DIMER, FRAG_A, charge=-1)
    assert monomer.charge == -1
    assert monomer.spin_multiplicity == 1


def test_dimer_geometry_is_the_union() -> None:
    dimer = dimer_geometry(WATER_DIMER, FRAG_A, FRAG_B, 0, 0)
    assert dimer.n_atoms == 6
    assert [a.symbol for a in dimer.atoms] == ["O", "H", "H", "O", "H", "H"]


def test_dimer_charge_is_the_sum() -> None:
    assert dimer_geometry(WATER_DIMER, FRAG_A, FRAG_B, 1, -1).charge == 0
    assert dimer_geometry(WATER_DIMER, FRAG_A, FRAG_B, -1, -1).charge == -2


def test_dimer_atom_order_is_deterministic() -> None:
    """Stable ordering keeps the input hash reproducible across runs."""
    first = dimer_geometry(WATER_DIMER, FRAG_A, FRAG_B, 0, 0)
    second = dimer_geometry(WATER_DIMER, FRAG_B, FRAG_A, 0, 0)
    assert [a.position for a in first.atoms] == [a.position for a in second.atoms]


def test_overlapping_fragments_rejected_in_dimer() -> None:
    with pytest.raises(ValidationError, match="share atoms"):
        dimer_geometry(WATER_DIMER, FRAG_A, Fragment("F1", (2, 3, 4, 5)), 0, 0)


def test_build_plans_covers_every_subsystem() -> None:
    monomers, dimers = build_plans(WATER_DIMER, DIMER_FRAGMENTATION, {"F0": 0, "F1": 0})
    assert [p.fragment_ids for p in monomers] == [("F0",), ("F1",)]
    assert [p.fragment_ids for p in dimers] == [("F0", "F1")]
    assert dimers[0].geometry.n_atoms == 6


# ---------------------------------------------------------------------------------
# The expansion
# ---------------------------------------------------------------------------------


def _energy(ids: tuple[str, ...], value: float, qubits: int = 12) -> FragmentEnergy:
    return FragmentEnergy(
        fragment_ids=ids,
        energy=value,
        n_atoms=3 * len(ids),
        n_electrons=10 * len(ids),
        charge=0,
        n_qubits=qubits,
        n_parameters=26,
        seconds=1.0,
        converged=True,
    )


def test_two_body_expansion_arithmetic() -> None:
    """E = sum E_I + sum (E_IJ - E_I - E_J)."""
    result = assemble(
        (
            _energy(("F0",), -76.0),
            _energy(("F1",), -76.0),
            _energy(("F0", "F1"), -152.01),
        ),
        requested_pairs=(("F0", "F1"),),
    )
    assert result.total_energy == pytest.approx(-152.01)
    assert result.interaction_energies[("F0", "F1")] == pytest.approx(-0.01)
    assert result.sum_of_monomers == pytest.approx(-152.0)


def test_binding_energy_for_two_fragments() -> None:
    result = assemble(
        (
            _energy(("F0",), -76.0),
            _energy(("F1",), -76.0),
            _energy(("F0", "F1"), -152.008),
        ),
        requested_pairs=(("F0", "F1"),),
    )
    assert result.binding_energy() == pytest.approx(-0.008)


def test_binding_energy_refuses_three_fragments() -> None:
    """With three fragments 'binding' needs a stated partition; guessing is not allowed."""
    result = assemble(
        (_energy(("F0",), -76.0), _energy(("F1",), -76.0), _energy(("F2",), -76.0)),
        requested_pairs=(),
    )
    with pytest.raises(ValidationError, match="only well defined for two fragments"):
        result.binding_energy()


def test_binding_energy_refuses_when_the_pair_was_cut_off() -> None:
    result = assemble(
        (_energy(("F0",), -76.0), _energy(("F1",), -76.0)),
        requested_pairs=(("F0", "F1"),),
    )
    with pytest.raises(ValidationError, match="cutoff excluded it"):
        result.binding_energy()


def test_single_fragment_reproduces_the_supersystem_energy() -> None:
    """The degenerate case: one monomer, no pairs, expansion == that energy."""
    result = assemble((_energy(("F0",), -152.01),), requested_pairs=())
    assert result.total_energy == pytest.approx(-152.01)
    assert result.total_interaction == pytest.approx(0.0)


def test_three_fragment_expansion() -> None:
    result = assemble(
        (
            _energy(("F0",), -76.0),
            _energy(("F1",), -76.0),
            _energy(("F2",), -76.0),
            _energy(("F0", "F1"), -152.01),
            _energy(("F0", "F2"), -152.02),
            _energy(("F1", "F2"), -152.03),
        ),
        requested_pairs=(("F0", "F1"), ("F0", "F2"), ("F1", "F2")),
    )
    # -228 + (-0.01) + (-0.02) + (-0.03)
    assert result.total_energy == pytest.approx(-228.06)
    assert result.total_interaction == pytest.approx(-0.06)


def test_skipped_pairs_are_recorded() -> None:
    result = assemble(
        (_energy(("F0",), -76.0), _energy(("F1",), -76.0)),
        requested_pairs=(("F0", "F1"),),
    )
    assert result.skipped_pairs == (("F0", "F1"),)


def test_assembly_requires_a_monomer() -> None:
    with pytest.raises(ValidationError, match="at least one monomer"):
        assemble((), requested_pairs=())


def test_result_records_the_scheme() -> None:
    result = assemble((_energy(("F0",), -76.0),), requested_pairs=())
    assert result.scheme == SCHEME == "fmo2-no-esp"


# ---------------------------------------------------------------------------------
# Honesty of the reported caveats
# ---------------------------------------------------------------------------------


def test_no_esp_caveat_always_present() -> None:
    messages = scheme_warnings(
        DIMER_FRAGMENTATION, active_space_truncated=False, cutoff_applied=False
    )
    assert any("electrostatic embedding" in m for m in messages)


def test_cutoff_truncation_is_flagged() -> None:
    three = FragmentationResult(
        monomers=(FRAG_A, FRAG_B, Fragment("F2", (6,))),
        dimer_pairs=(("F0", "F1"),),  # F0-F2 and F1-F2 dropped
    )
    messages = scheme_warnings(three, active_space_truncated=False, cutoff_applied=True)
    assert any("dropped entirely" in m for m in messages)


def test_many_body_truncation_flagged_above_two_fragments() -> None:
    three = FragmentationResult(
        monomers=(FRAG_A, FRAG_B, Fragment("F2", (6,))),
        dimer_pairs=(("F0", "F1"), ("F0", "F2"), ("F1", "F2")),
    )
    messages = scheme_warnings(three, active_space_truncated=False, cutoff_applied=False)
    assert any("three-body" in m for m in messages)


def test_active_space_truncation_flags_broken_error_cancellation() -> None:
    messages = scheme_warnings(
        DIMER_FRAGMENTATION, active_space_truncated=True, cutoff_applied=False
    )
    assert any("does not cancel" in m for m in messages)


def test_two_fragments_do_not_get_a_three_body_warning() -> None:
    messages = scheme_warnings(
        DIMER_FRAGMENTATION, active_space_truncated=False, cutoff_applied=False
    )
    assert not any("three-body" in m for m in messages)


# ---------------------------------------------------------------------------------
# Config isolation
# ---------------------------------------------------------------------------------


def test_subsystem_config_strips_parent_charge() -> None:
    """The single most dangerous leak: a fragment inheriting the parent's charge."""
    child = subsystem_config(
        {"basis": "6-31g", "charge": -2, "spin_multiplicity": 3, "backend": "statevector"}
    )
    assert child == {"basis": "6-31g", "backend": "statevector"}


def test_subsystem_config_strips_fragmentation_keys() -> None:
    child = subsystem_config(
        {
            "basis": "sto-3g",
            "fragmentation_scheme": "molecular",
            "cut_bonds": ((0, 1),),
            "dimer_cutoff_angstrom": 5.0,
            "fragment_charges": {"F0": 0},
            "xyz": "...",
        }
    )
    assert child == {"basis": "sto-3g"}


def test_subsystem_config_keeps_method_settings() -> None:
    """Everything describing *how* to solve must survive, or fragments differ from each other."""
    child = subsystem_config(
        {
            "basis": "cc-pvdz",
            "mapping": "jordan_wigner",
            "backend": "statevector",
            "max_iterations": 500,
            "solver_options": {"optimizer": "L-BFGS-B"},
        }
    )
    assert child["mapping"] == "jordan_wigner"
    assert child["solver_options"] == {"optimizer": "L-BFGS-B"}
    assert child["max_iterations"] == 500


def test_fmo_result_is_frozen() -> None:
    result = assemble((_energy(("F0",), -76.0),), requested_pairs=())
    assert isinstance(result, FMOResult)
    with pytest.raises((AttributeError, TypeError)):
        result.total_energy = 0.0  # type: ignore[misc]

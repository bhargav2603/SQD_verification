"""End-to-end FMO2 runs. Requires PySCF, so these execute on the Linux host.

The central check is ``test_single_fragment_matches_supersystem``: with one fragment the
two-body expansion degenerates to a single monomer energy, which must equal what the plain
supersystem chain produces for the same geometry. The two paths run the same stage
functions over the same input, so any disagreement is a bug in the fragment traversal
rather than a difference in physics -- and it is the cheapest possible way to catch one.
"""

import pytest

from qdr_compute_engine.pipeline import fmo_runner, pipeline
from qdr_compute_engine.types import Atom, Geometry, LengthUnit
from qdr_core.errors import ValidationError

pytestmark = pytest.mark.requires_pyscf


def _geometry(atoms, charge=0, multiplicity=1) -> Geometry:
    return Geometry(
        atoms=tuple(Atom(s, p) for s, p in atoms),
        charge=charge,
        spin_multiplicity=multiplicity,
        units=LengthUnit.ANGSTROM,
    )


H2 = _geometry([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))])

# Two H2 molecules far enough apart that their interaction is tiny but non-zero.
H2_PAIR = _geometry(
    [
        ("H", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 0.735)),
        ("H", (0.0, 0.0, 5.000)),
        ("H", (0.0, 0.0, 5.735)),
    ]
)

# Standard water dimer, near its equilibrium hydrogen bond.
WATER_DIMER = _geometry(
    [
        ("O", (-0.0702, -0.0530, 0.0000)),
        ("H", (0.8741, 0.1387, 0.0000)),
        ("H", (-0.5133, 0.7825, 0.0000)),
        ("O", (2.7947, 0.0313, 0.0000)),
        ("H", (3.1157, -0.4324, -0.7738)),
        ("H", (3.1157, -0.4324, 0.7738)),
    ]
)

BASE_CONFIG = {"basis": "sto-3g", "backend": "statevector"}


def test_single_fragment_matches_supersystem() -> None:
    """One fragment: the expansion must reproduce the supersystem energy exactly."""
    _, result = fmo_runner.run({**BASE_CONFIG, "fragmentation_scheme": "single"}, geometry=H2)
    assert result.n_monomers == 1
    assert result.total_interaction == pytest.approx(0.0, abs=1e-12)

    supersystem = pipeline.run(BASE_CONFIG, geometry=H2)
    assert result.total_energy == pytest.approx(supersystem.require_solve_result().energy, abs=1e-9)


def test_two_fragments_produce_a_binding_energy() -> None:
    """The path that never once executed before: A, B, AB and the difference."""
    _, result = fmo_runner.run(
        {**BASE_CONFIG, "fragmentation_scheme": "molecular"}, geometry=H2_PAIR
    )

    assert result.n_monomers == 2
    assert len(result.dimer_energies) == 1

    binding = result.binding_energy()
    # Two H2 at 5 A: bound, but only just.
    assert -0.05 < binding < 0.0

    # The expansion identity must hold exactly, by construction.
    assert result.total_energy == pytest.approx(
        result.sum_of_monomers + result.total_interaction, abs=1e-12
    )


def test_identical_fragments_have_identical_energies() -> None:
    """Two copies of the same molecule must give the same monomer energy."""
    _, result = fmo_runner.run(
        {**BASE_CONFIG, "fragmentation_scheme": "molecular"}, geometry=H2_PAIR
    )
    energies = list(result.monomer_energies.values())
    assert energies[0] == pytest.approx(energies[1], abs=1e-9)


def test_water_dimer_runs_end_to_end() -> None:
    """The real target: 3 subsystems, a hydrogen bond, a binding energy."""
    context, result = fmo_runner.run(
        {**BASE_CONFIG, "fragmentation_scheme": "molecular"}, geometry=WATER_DIMER
    )

    assert result.n_monomers == 2
    assert all(f.n_qubits <= 28 for f in result.fragments)

    binding = result.binding_energy()
    # Bound, and of a physically plausible magnitude for an H-bond. STO-3G with no
    # counterpoise correction cannot give the right number, so this is a sanity bound
    # rather than an accuracy claim -- see the BSSE warning in the docs.
    assert -0.05 < binding < 0.0

    assert any("electrostatic embedding" in w for w in context.warnings)


def test_expansion_is_reported_as_no_esp() -> None:
    _, result = fmo_runner.run({**BASE_CONFIG, "fragmentation_scheme": "single"}, geometry=H2)
    assert result.scheme == "fmo2-no-esp"
    assert any("electrostatic embedding" in w for w in result.warnings)


def test_fragment_charges_must_sum_correctly() -> None:
    with pytest.raises(ValidationError, match="sum to"):
        fmo_runner.run(
            {
                **BASE_CONFIG,
                "fragmentation_scheme": "molecular",
                "fragment_charges": {"F0": 1},
            },
            geometry=H2_PAIR,
        )


def test_supersystem_and_fmo_differ_by_the_interaction() -> None:
    """FMO2 on two fragments is not the supersystem energy, and the gap is meaningful.

    For a genuinely two-body system the expansion is exact up to the missing embedding,
    so the two should be close but not identical -- and the difference is what the
    two-body truncation plus the missing ESP costs.
    """
    _, fmo_result = fmo_runner.run(
        {**BASE_CONFIG, "fragmentation_scheme": "molecular"}, geometry=H2_PAIR
    )
    supersystem = pipeline.run(BASE_CONFIG, geometry=H2_PAIR)
    e_super = supersystem.require_solve_result().energy

    difference = abs(fmo_result.total_energy - e_super)
    assert difference < 0.05, "expansion should be close to the supersystem at 5 A"

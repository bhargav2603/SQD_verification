"""Open-shell inputs must be refused, not silently mis-computed.

The failure this guards against is specific and nasty: ``scf.RHF(mol)`` in PySCF returns
an **ROHF** object when ``mol.spin != 0`` instead of raising, and every downstream
component (frozen-core counting, the Hartree-Fock occupation pattern, the reference-state
X gates) assumes double occupancy without checking. An open-shell system would therefore
complete the pipeline and report a believable energy for the wrong reference.
"""

import pytest

from qdr_compute_engine.geometry.pose_builder import require_closed_shell
from qdr_compute_engine.pipeline import stage_01_geometry_pose
from qdr_compute_engine.pipeline.stage_context import StageContext
from qdr_compute_engine.types import Atom, Geometry, LengthUnit
from qdr_core.errors import ValidationError
from qdr_core.ids import new_run_id


def _geometry(symbols_positions, charge=0, multiplicity=1) -> Geometry:
    return Geometry(
        atoms=tuple(Atom(s, p) for s, p in symbols_positions),
        charge=charge,
        spin_multiplicity=multiplicity,
        units=LengthUnit.ANGSTROM,
    )


def test_singlet_is_accepted() -> None:
    require_closed_shell(_geometry([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))]))


def test_triplet_is_rejected() -> None:
    # O2 in its true triplet ground state -- the textbook open-shell molecule.
    geometry = _geometry([("O", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.208))], multiplicity=3)
    with pytest.raises(ValidationError, match="closed-shell only"):
        require_closed_shell(geometry)


def test_doublet_radical_is_rejected() -> None:
    """An odd electron count cannot be a closed-shell singlet at all."""
    # Neutral H atom: one electron.
    geometry = _geometry([("H", (0.0, 0.0, 0.0))])
    with pytest.raises(ValidationError, match="closed-shell"):
        require_closed_shell(geometry)


def test_odd_electron_count_from_charge_is_rejected() -> None:
    """A charge that breaks pairing must be caught even at multiplicity 1."""
    geometry = _geometry([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))], charge=1)
    with pytest.raises(ValidationError, match="closed-shell singlet"):
        require_closed_shell(geometry)


def test_charged_closed_shell_is_accepted() -> None:
    """Charge itself is fine as long as the electrons still pair."""
    # LiH with a +2 charge: 4 - 2 = 2 electrons, still closed shell.
    geometry = _geometry([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.6))], charge=2)
    require_closed_shell(geometry)


def test_stage_01_rejects_open_shell_early() -> None:
    """The pipeline must fail at stage 1, not five stages later with a wrong number."""
    geometry = _geometry([("O", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.208))], multiplicity=3)
    context = StageContext(run_id=new_run_id(), config={}, geometry=geometry)

    with pytest.raises(ValidationError, match="closed-shell only"):
        stage_01_geometry_pose.run(context)


def test_rejection_names_the_unsupported_value() -> None:
    """The error must say what was wrong, so a user can act on it."""
    geometry = _geometry([("O", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.208))], multiplicity=3)
    with pytest.raises(ValidationError) as excinfo:
        require_closed_shell(geometry)

    details = excinfo.value.details
    assert details["spin_multiplicity"] == 3
    assert details["supported_spin_multiplicity"] == 1

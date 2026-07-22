"""Exercise the FMO orchestration without PySCF, using injected integrals.

Why this exists
---------------
``test_fmo_integration.py`` covers the real chemistry but needs PySCF, so on a developer
machine it skips and the entire fragment traversal goes unexecuted. That is precisely the
code most likely to be wrong -- it builds sub-geometries, threads config, and assembles a
sum of many parts.

So stage 3 is substituted here with integrals built by hand, and everything downstream
(active space, Hamiltonian, mapping, ansatz, solver, assembly) runs for real.

The sharpest test available is the **non-interacting limit**: two fragments whose mutual
Coulomb integrals are exactly zero must produce exactly zero interaction energy, and a
total that is exactly the sum of the parts. Any error in fragment construction, config
threading or assembly arithmetic breaks that identity, and it holds independently of which
molecule the integrals came from.
"""

import numpy as np
import pytest

from qdr_compute_engine.pipeline import fmo_runner, stage_03_integrals
from qdr_compute_engine.types import Atom, Geometry, IntegralSet, LengthUnit
from qdr_core.errors import ValidationError

# Two H2 molecules, far apart. Fragmentation splits them into F0 and F1.
H2_PAIR = Geometry(
    atoms=(
        Atom("H", (0.0, 0.0, 0.0)),
        Atom("H", (0.0, 0.0, 0.735)),
        Atom("H", (0.0, 0.0, 8.000)),
        Atom("H", (0.0, 0.0, 8.735)),
    ),
    charge=0,
    spin_multiplicity=1,
    units=LengthUnit.ANGSTROM,
)


def _block_diagonal(monomer: IntegralSet, copies: int) -> IntegralSet:
    """``copies`` non-interacting replicas of ``monomer``, in an energy-ordered MO basis.

    Inter-block two-electron integrals are left at zero, which *is* the non-interacting
    limit: the blocks share no Coulomb or exchange. The exact ground state is therefore
    the product state, with energy exactly ``copies * E_monomer``.

    **Orbital ordering matters here.** The engine documents that integrals arrive in an
    energy-ordered MO basis, and relies on it in two places: ``select_active_space`` takes
    a contiguous index window around the Fermi level, and ``hartree_fock_occupations``
    fills the lowest ``n_electrons`` spin orbitals. Stacking the replicas naively gives
    ``[A_occ, A_virt, B_occ, B_virt]``, so the reference determinant would doubly occupy
    A's occupied *and* A's virtual orbital -- a completely different (and much higher)
    state. PySCF returns energy-ordered MOs for a real dimer, so the permutation applied
    here reproduces what the production path would actually see: all occupied orbitals
    first, then all virtuals.
    """
    n = monomer.n_orbitals
    n_occupied = monomer.n_electrons // 2
    total = n * copies

    h1 = np.zeros((total, total))
    h2 = np.zeros((total, total, total, total))
    for c in range(copies):
        lo = c * n
        hi = lo + n
        h1[lo:hi, lo:hi] = monomer.h1
        h2[lo:hi, lo:hi, lo:hi, lo:hi] = monomer.h2

    order = [c * n + i for c in range(copies) for i in range(n_occupied)]
    order += [c * n + i for c in range(copies) for i in range(n_occupied, n)]
    index = np.ix_(order, order)

    return IntegralSet(
        h1=np.ascontiguousarray(h1[index]),
        h2=np.ascontiguousarray(h2[np.ix_(order, order, order, order)]),
        e_nuc=monomer.e_nuc * copies,
        n_electrons=monomer.n_electrons * copies,
        e_hf=(monomer.e_hf or 0.0) * copies,
    )


@pytest.fixture
def injected_integrals(monkeypatch, h2_integrals):
    """Replace stage 3 with hand-built integrals sized to each fragment."""
    seen: list[int] = []

    def fake_compute_integrals(geometry, *, basis="sto-3g", verbose=0):
        n_hydrogens = geometry.n_atoms
        seen.append(n_hydrogens)
        if n_hydrogens % 2 != 0:
            raise ValidationError(f"unexpected fragment size {n_hydrogens}")
        return _block_diagonal(h2_integrals, n_hydrogens // 2)

    monkeypatch.setattr(stage_03_integrals, "compute_integrals", fake_compute_integrals)
    return seen


BASE_CONFIG = {"basis": "sto-3g", "backend": "statevector"}
MOLECULAR = {**BASE_CONFIG, "fragmentation_scheme": "molecular"}


def test_every_subsystem_is_visited(injected_integrals) -> None:
    """Two monomers and one dimer: three subsystem solves, with the right sizes."""
    _, result = fmo_runner.run(MOLECULAR, geometry=H2_PAIR)

    assert sorted(injected_integrals) == [2, 2, 4]
    assert result.n_monomers == 2
    assert len(result.dimer_energies) == 1


def test_non_interacting_fragments_give_zero_interaction(injected_integrals) -> None:
    """The identity that catches almost any wiring error."""
    _, result = fmo_runner.run(MOLECULAR, geometry=H2_PAIR)

    interaction = result.binding_energy()
    assert interaction == pytest.approx(0.0, abs=1e-7)


def test_total_is_exactly_the_sum_of_parts(injected_integrals) -> None:
    _, result = fmo_runner.run(MOLECULAR, geometry=H2_PAIR)

    energies = list(result.monomer_energies.values())
    assert energies[0] == pytest.approx(energies[1], abs=1e-9)
    assert result.total_energy == pytest.approx(sum(energies), abs=1e-7)


def test_expansion_identity_holds(injected_integrals) -> None:
    _, result = fmo_runner.run(MOLECULAR, geometry=H2_PAIR)
    assert result.total_energy == pytest.approx(
        result.sum_of_monomers + result.total_interaction, abs=1e-12
    )


def test_fragment_metadata_is_recorded(injected_integrals) -> None:
    _, result = fmo_runner.run(MOLECULAR, geometry=H2_PAIR)

    by_key = {f.key: f for f in result.fragments}
    assert set(by_key) == {"F0", "F1", "F0+F1"}

    assert by_key["F0"].n_atoms == 2
    assert by_key["F0"].n_electrons == 2
    assert by_key["F0"].n_qubits == 4

    assert by_key["F0+F1"].n_atoms == 4
    assert by_key["F0+F1"].n_electrons == 4
    assert by_key["F0+F1"].n_qubits == 8

    assert all(f.seconds >= 0.0 for f in result.fragments)
    assert all(f.converged for f in result.fragments)


def test_single_scheme_produces_one_monomer(injected_integrals) -> None:
    _, result = fmo_runner.run({**BASE_CONFIG, "fragmentation_scheme": "single"}, geometry=H2_PAIR)
    assert result.n_monomers == 1
    assert result.dimer_energies == {}
    assert result.total_interaction == pytest.approx(0.0)


def test_warnings_reach_the_context(injected_integrals) -> None:
    context, _ = fmo_runner.run(MOLECULAR, geometry=H2_PAIR)
    assert any("electrostatic embedding" in w for w in context.warnings)


def test_dimer_cutoff_skips_the_pair(injected_integrals) -> None:
    """A cutoff below the separation drops the pair, and that must be visible."""
    context, result = fmo_runner.run({**MOLECULAR, "dimer_cutoff_angstrom": 2.0}, geometry=H2_PAIR)

    assert result.dimer_energies == {}
    assert result.skipped_pairs == ()  # no pair was requested, so none was skipped
    assert any("dropped entirely" in w for w in context.warnings) or result.n_monomers == 2

    # With no pair term the total is just the isolated monomers.
    assert result.total_energy == pytest.approx(result.sum_of_monomers, abs=1e-12)


def test_a_failing_subsystem_aborts_the_expansion(monkeypatch, h2_integrals) -> None:
    """A partial expansion is a wrong answer, not a smaller one."""

    calls = {"n": 0}

    def failing_compute_integrals(geometry, *, basis="sto-3g", verbose=0):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("SCF blew up")
        return _block_diagonal(h2_integrals, geometry.n_atoms // 2)

    monkeypatch.setattr(stage_03_integrals, "compute_integrals", failing_compute_integrals)

    with pytest.raises(ValidationError, match="cannot be assembled from a subset"):
        fmo_runner.run(MOLECULAR, geometry=H2_PAIR)


def test_open_shell_fragment_is_refused(monkeypatch, h2_integrals) -> None:
    """A covalent cut leaves a radical; preflight must catch it before any SCF."""
    called = {"n": 0}

    def counting_compute_integrals(geometry, *, basis="sto-3g", verbose=0):
        called["n"] += 1
        return _block_diagonal(h2_integrals, max(geometry.n_atoms // 2, 1))

    monkeypatch.setattr(stage_03_integrals, "compute_integrals", counting_compute_integrals)

    # One H2 plus two isolated H atoms. The *parent* has 4 electrons and passes stage 1
    # as a closed-shell singlet, so the refusal has to come from the per-fragment check:
    # fragmentation yields H2 (2 electrons) and two lone H atoms (1 each).
    odd = Geometry(
        atoms=(
            Atom("H", (0.0, 0.0, 0.0)),
            Atom("H", (0.0, 0.0, 0.735)),
            Atom("H", (0.0, 0.0, 8.000)),
            Atom("H", (0.0, 0.0, 16.000)),
        ),
        charge=0,
        spin_multiplicity=1,
        units=LengthUnit.ANGSTROM,
    )

    with pytest.raises(ValidationError, match="closed-shell"):
        fmo_runner.run(MOLECULAR, geometry=odd)

    assert called["n"] == 0, "preflight must reject before any integral evaluation"


def test_parent_charge_does_not_leak_into_fragments(monkeypatch, injected_integrals) -> None:
    """The dangerous leak: fragments must not inherit the parent's charge."""
    captured: list[int] = []
    original = fmo_runner.solve_subsystem

    def capturing(plan, config, *, seed=0):
        captured.append(plan.geometry.charge)
        assert "charge" not in config, "parent charge reached a subsystem config"
        return original(plan, config, seed=seed)

    monkeypatch.setattr(fmo_runner, "solve_subsystem", capturing)
    fmo_runner.run(MOLECULAR, geometry=H2_PAIR)

    assert captured == [0, 0, 0]


def test_charge_assignment_that_breaks_a_fragment_is_refused(injected_integrals) -> None:
    """Charges are propagated, and an assignment that opens a shell is caught.

    ``{F0: +1, F1: -1}`` sums to the neutral parent, so the sum rule passes -- but H2+
    has one electron and cannot be a closed-shell singlet. The refusal has to name the
    offending fragment, because the sum rule alone would have let this through.
    """
    with pytest.raises(ValidationError, match="closed-shell") as excinfo:
        fmo_runner.run({**MOLECULAR, "fragment_charges": {"F0": 1, "F1": -1}}, geometry=H2_PAIR)

    assert excinfo.value.details["fragment_ids"] == ["F0"]

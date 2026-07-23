"""Transition-metal spin-gap runner (Colab / Linux, needs PySCF).

What this produces
------------------
For a molecule and two spin states (low-spin vs high-spin) it computes the spin gap
``dE = E(high-spin) - E(low-spin)`` four ways and lays them side by side:

  * **DFT (B3LYP)** -- the method that is *supposed* to be wrong for these systems; the
    reason spin gaps are the validation wedge in the first place.
  * **CASCI reference (PySCF)** -- exact diagonalisation in the chosen active space, the
    trustworthy classical answer for that active space.
  * **QDR engine (matrix-free, rung 3)** -- the same active-space Hamiltonian solved through
    this engine's own fermion -> qubit -> matrix-free path, in the correct ``(n_alpha,
    n_beta)`` sector (rung-1 open-shell support). This is the number QDR "owns".
  * **Literature CBS**, printed for context only -- matching it needs post-processing beyond
    the active space (NEVPT2 / CCSD(T)); an active-space CASCI is not expected to reproduce
    it, and pretending otherwise would be a number for its own sake.

The load-bearing check is the **cross-check gate**: QDR's active-space energy must equal
PySCF's CASCI energy for the *same* active space and spin sector, per state, to ~1e-6 Ha.
If it does not, the script raises instead of plotting -- a wrong engine must not print a
believable gap. A fast open-shell **sanity molecule (O2)** runs first so a pipeline problem
surfaces in seconds, not after the expensive metal SCF.

How to run in Colab
-------------------
    !pip install -q pyscf matplotlib
    !git clone <this repo> && cd <repo>/packages/compute-engine && pip install -e . ../qdr-core
    !python notebooks/spin_gap_colab.py

Everything here is deliberately explicit (geometries inline, active spaces named) so the
recipe is auditable. The idealised octahedral metal geometries are *illustrative* -- an
adiabatic gap for publication needs a geometry optimised per spin state; swap in your own
coordinates and the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from qdr_compute_engine.active_space.active_space_selector import (
    restrict_integrals,
    select_active_space,
)
from qdr_compute_engine.active_space.avas_selection import avas_active_integrals
from qdr_compute_engine.hamiltonian.fermionic_builder import build_fermionic_hamiltonian
from qdr_compute_engine.hamiltonian.matrix_free import (
    ground_state_energy_at_spin_sector_matrix_free,
)
from qdr_compute_engine.hamiltonian.spin_penalty import spin_penalized_hamiltonian
from qdr_compute_engine.integrals.pyscf_integrals import compute_integrals
from qdr_compute_engine.mappers.jordan_wigner import map_to_qubits
from qdr_compute_engine.solvers.classical_baseline import (
    alpha_beta_from_spin,
    solve_fci,
)
from qdr_compute_engine.types import Atom, Geometry, LengthUnit
from qdr_core.limits import HARTREE_TO_KCAL_PER_MOL

#: QDR-vs-PySCF agreement demanded before any gap is reported (Hartree).
CROSS_CHECK_TOLERANCE = 1e-6


@dataclass(frozen=True)
class SpinState:
    label: str
    spin_multiplicity: int


@dataclass(frozen=True)
class SpinGapCase:
    name: str
    atoms: tuple[Atom, ...]
    charge: int
    basis: str
    low_spin: SpinState
    high_spin: SpinState
    n_active_electrons: int
    n_active_orbitals: int
    #: AVAS target valence orbitals, e.g. ``("Fe 3d",)``. When set, the active space is
    #: chosen by AVAS (the 3d shell is guaranteed active) and ``n_active_*`` above are
    #: ignored -- AVAS decides the size. When ``None``, the index-window selector is used.
    ao_labels: tuple[str, ...] | None = None
    #: Published complete-basis reference, kcal/mol (high - low). Context only.
    literature_cbs_kcal: float | None = None


@dataclass
class StateResult:
    label: str
    spin_multiplicity: int
    n_alpha: int
    n_beta: int
    n_qubits: int
    e_casci: float
    e_qdr: float
    spin_squared: float
    multiplicity: float
    e_dft: float


def _octahedral(metal: str, ligand_atom: str, bond_length: float) -> list[Atom]:
    """A metal atom at the origin with six identical single-atom ligands along the axes.

    A single-atom ligand model (e.g. bare F or Cl donors) keeps the demonstration geometry
    unambiguous. For molecular ligands (H2O, NCH) substitute real coordinates.
    """
    directions = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]
    atoms = [Atom(metal, (0.0, 0.0, 0.0))]
    atoms += [
        Atom(ligand_atom, (d[0] * bond_length, d[1] * bond_length, d[2] * bond_length))
        for d in directions
    ]
    return atoms


def _dft_energy(case: SpinGapCase, state: SpinState) -> float:
    """B3LYP total energy for one spin state -- the 'DFT gets this wrong' baseline."""
    from pyscf import dft, gto

    molecule = gto.Mole()
    molecule.atom = [(a.symbol, a.position) for a in case.atoms]
    molecule.basis = case.basis
    molecule.charge = case.charge
    molecule.spin = state.spin_multiplicity - 1
    molecule.unit = "Angstrom"
    molecule.verbose = 0
    molecule.build()

    mean_field = dft.RKS(molecule) if molecule.spin == 0 else dft.ROKS(molecule)
    mean_field.xc = "b3lyp"
    mean_field.max_cycle = 200
    energy = float(mean_field.kernel())
    if not mean_field.converged:
        # Charged transition-metal centres (e.g. [FeF6]4-) stall first-order DIIS; hand the
        # stalled density to the second-order (Newton) solver, matching the HF path.
        mean_field = mean_field.newton()
        energy = float(mean_field.kernel(mean_field.mo_coeff, mean_field.mo_occ))
    if not mean_field.converged:
        raise RuntimeError(f"B3LYP did not converge for {case.name} {state.label}")
    return energy


def _solve_state(case: SpinGapCase, state: SpinState) -> StateResult:
    """Active-space CASCI (PySCF) and the QDR engine, cross-checked, for one spin state."""
    geometry = Geometry(
        atoms=case.atoms,
        charge=case.charge,
        spin_multiplicity=state.spin_multiplicity,
        units=LengthUnit.ANGSTROM,
    )
    if case.ao_labels is not None:
        # AVAS: the named valence shell (e.g. Fe 3d) is active by construction.
        avas = avas_active_integrals(
            geometry, ao_labels=case.ao_labels, basis=case.basis
        )
        restricted, active_space = avas.integrals, avas.active_space
    else:
        integrals = compute_integrals(geometry, basis=case.basis)
        active_space = select_active_space(
            integrals,
            n_active_orbitals=case.n_active_orbitals,
            n_active_electrons=case.n_active_electrons,
        )
        restricted = restrict_integrals(integrals, active_space)

    n_alpha, n_beta = alpha_beta_from_spin(
        restricted.n_electrons, state.spin_multiplicity
    )

    # (1) Classical reference: PySCF CASCI in this active space and spin sector.
    reference = solve_fci(restricted, spin_multiplicity=state.spin_multiplicity)

    # (2) QDR's own engine: fermion -> qubit -> matrix-free, same active space + sector.
    # restricted.e_nuc (nuclear repulsion + frozen-core shift) is already folded into the
    # InteractionOperator's constant by build_fermionic_hamiltonian and carried into the
    # qubit Hamiltonian's identity term by Jordan-Wigner, so the matrix-free eigenvalue is
    # already the *total* energy. Adding e_nuc here would double-count it -- the golden H2
    # sector test (test_open_shell.test_h2_singlet_sector_is_the_fci_energy) pins this.
    # The (n_alpha, n_beta) sector fixes only S_z, so a requested singlet would otherwise
    # collapse onto a lower triplet's M_s=0 component; the S^2 penalty pins the actual spin
    # without shifting its energy (see hamiltonian/spin_penalty.py), exactly as fix_spin_
    # does for the CASCI reference above -- which is what keeps the cross-check honest.
    hamiltonian = map_to_qubits(build_fermionic_hamiltonian(restricted))
    hamiltonian = spin_penalized_hamiltonian(
        hamiltonian, spin_multiplicity=state.spin_multiplicity
    )
    e_qdr = ground_state_energy_at_spin_sector_matrix_free(hamiltonian, n_alpha, n_beta)

    # The gate: a disagreement means the engine is wrong; do not report a gap built on it.
    if abs(e_qdr - reference.energy) > CROSS_CHECK_TOLERANCE:
        raise AssertionError(
            f"cross-check FAILED for {case.name} {state.label}: "
            f"QDR {e_qdr:.9f} vs CASCI {reference.energy:.9f} Ha "
            f"(|dE| = {abs(e_qdr - reference.energy):.2e}); refusing to report a gap"
        )

    return StateResult(
        label=state.label,
        spin_multiplicity=state.spin_multiplicity,
        n_alpha=n_alpha,
        n_beta=n_beta,
        n_qubits=active_space.n_qubits,
        e_casci=reference.energy,
        e_qdr=e_qdr,
        spin_squared=float(reference.spin_squared or 0.0),
        multiplicity=float(reference.multiplicity or 0.0),
        e_dft=_dft_energy(case, state),
    )


def run_case(case: SpinGapCase) -> dict:
    """Run both spin states and assemble the gap every way."""
    low = _solve_state(case, case.low_spin)
    high = _solve_state(case, case.high_spin)

    to_kcal = HARTREE_TO_KCAL_PER_MOL
    gap_qdr = (high.e_qdr - low.e_qdr) * to_kcal
    gap_casci = (high.e_casci - low.e_casci) * to_kcal
    gap_dft = (high.e_dft - low.e_dft) * to_kcal

    print(f"\n=== {case.name} ===")
    print(
        f"active space: ({case.n_active_electrons}e, {case.n_active_orbitals}o) "
        f"-> {low.n_qubits} qubits"
    )
    for state in (low, high):
        print(
            f"  {state.label:>10}: E(CASCI)={state.e_casci:.6f}  "
            f"E(QDR)={state.e_qdr:.6f}  <S^2>={state.spin_squared:.3f}  "
            f"2S+1={state.multiplicity:.2f}  [cross-check OK]"
        )
    print("  spin gap (high - low):")
    print(f"    DFT B3LYP     : {gap_dft:+8.2f} kcal/mol")
    print(f"    CASCI (PySCF) : {gap_casci:+8.2f} kcal/mol")
    print(f"    QDR engine    : {gap_qdr:+8.2f} kcal/mol")
    if case.literature_cbs_kcal is not None:
        print(
            f"    literature CBS: {case.literature_cbs_kcal:+8.2f} kcal/mol "
            "(context only; needs post-processing to match)"
        )

    return {
        "name": case.name,
        "basis": case.basis,
        "active_space": [case.n_active_electrons, case.n_active_orbitals],
        "n_qubits": low.n_qubits,
        "states": {"low": asdict(low), "high": asdict(high)},
        "gap_kcal": {"dft": gap_dft, "casci": gap_casci, "qdr": gap_qdr},
        "literature_cbs_kcal": case.literature_cbs_kcal,
    }


def plot_gaps(results: list[dict], output_dir: Path) -> None:
    """Bar chart of the spin gap by method, one group per molecule."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["dft", "casci", "qdr"]
    labels = ["DFT B3LYP", "CASCI (ref)", "QDR engine"]
    x = np.arange(len(results))
    width = 0.25

    fig, ax = plt.subplots(figsize=(1.8 * len(results) + 3, 5))
    for offset, (method, label) in enumerate(zip(methods, labels, strict=True)):
        values = [r["gap_kcal"][method] for r in results]
        ax.bar(x + (offset - 1) * width, values, width, label=label)

    for index, r in enumerate(results):
        cbs = r.get("literature_cbs_kcal")
        if cbs is not None:
            ax.plot(
                [index - 1.6 * width, index + 1.6 * width],
                [cbs, cbs],
                "k--",
                linewidth=1.2,
                label="literature CBS" if index == 0 else None,
            )

    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in results], rotation=15, ha="right")
    ax.set_ylabel("spin gap  E(high) - E(low)  [kcal/mol]")
    ax.set_title(
        "Transition-metal spin gap: DFT vs active-space reference vs QDR engine"
    )
    ax.legend()
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "spin_gap.png"
    fig.savefig(path, dpi=300)
    print(f"\nwrote {path}")


#: A fast open-shell sanity check: O2's ground state is the triplet, so the gap must be
#: NEGATIVE (triplet below singlet). If this comes out positive, the open-shell machinery
#: is wrong and nothing below is trustworthy.
O2 = SpinGapCase(
    name="O2 (sanity: triplet ground state)",
    atoms=(Atom("O", (0.0, 0.0, 0.0)), Atom("O", (0.0, 0.0, 1.208))),
    charge=0,
    basis="sto-3g",
    low_spin=SpinState("singlet", 1),
    high_spin=SpinState("triplet", 3),
    n_active_electrons=8,
    n_active_orbitals=6,
)

#: The metal target. [FeF6]^4- as a single-atom-ligand octahedral model keeps the geometry
#: unambiguous while exercising the full open-shell + active-space + matrix-free path on a
#: real 3d^6 Fe(II) centre (low-spin singlet vs high-spin quintet). Start small (Fe 3d ->
#: (6e,5o) = 10 qubits); widen the active space once the cross-check holds.
FE_OCTAHEDRAL = SpinGapCase(
    name="[FeF6]4- (Fe(II) d6, illustrative geometry)",
    atoms=tuple(_octahedral("Fe", "F", bond_length=2.00)),
    charge=-4,
    basis="sto-3g",
    low_spin=SpinState("singlet", 1),
    high_spin=SpinState("quintet", 5),
    n_active_electrons=6,
    n_active_orbitals=5,
    # AVAS pins the active space to the Fe 3d shell -- the orbitals that carry the spin
    # state -- instead of gambling on an index window. This is what makes the selection
    # defensible; widen to ("Fe 3d", "Fe 4d") for the double-shell active space.
    ao_labels=("Fe 3d",),
)


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "results"
    cases = [O2, FE_OCTAHEDRAL]

    results: list[dict] = []
    for case in cases:
        results.append(run_case(case))

    # The sanity gate: O2's triplet must be below its singlet.
    o2 = results[0]
    assert o2["gap_kcal"]["qdr"] < 0.0, (
        f"SANITY FAILED: O2 triplet is not below its singlet "
        f"(gap {o2['gap_kcal']['qdr']:+.2f} kcal/mol); open-shell path is wrong"
    )
    print("\nsanity OK: O2 triplet is the ground state, QDR == CASCI on every state.")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "spin_gap.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"wrote {output_dir / 'spin_gap.json'}")
    plot_gaps(results, output_dir)


if __name__ == "__main__":
    main()

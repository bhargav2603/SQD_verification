"""AVAS active-space selection (atomic valence active space).

Why this exists
---------------
``select_active_space`` takes a contiguous band of MOs around the Fermi level. For a
transition metal that is a gamble: the 3d orbitals that *carry* the spin state are not
guaranteed to land in that window, so a spin gap built on it can be physically meaningless
even when every downstream cross-check passes (QDR would faithfully reproduce CASCI of the
wrong active space). AVAS (Sayfutyarova, Sun, Chan & Knizia, JCTC 2017) instead fixes the
selection to the orbitals that overlap a chosen set of atomic valence orbitals -- e.g.
``["Fe 3d"]`` -- so the metal d shell is active *by construction*.

This wraps PySCF's validated ``mcscf.avas`` (re-deriving the projection by hand is exactly
the believable-wrong risk this engine exists to avoid) and hands its reordered orbitals to
the already-tested frozen-core transform in ``restrict_integrals``. PySCF is the only new
dependency, imported lazily as everywhere else that touches it, so this module still imports
on Windows.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qdr_compute_engine.active_space.active_space_selector import restrict_integrals
from qdr_compute_engine.integrals.pyscf_integrals import (
    _require_pyscf,
    chemist_to_physicist,
    run_reference_scf,
)
from qdr_compute_engine.geometry.pose_builder import to_pyscf_atom_spec
from qdr_compute_engine.types import ActiveSpace, Geometry, IntegralSet
from qdr_core.errors import ValidationError

DEFAULT_AVAS_THRESHOLD = 0.2


@dataclass(frozen=True)
class AvasResult:
    """The AVAS active space and the integrals restricted to it.

    ``integrals`` is already frozen-core-reduced onto the active orbitals (via the tested
    ``restrict_integrals``), so it drops straight into the FCI/CASCI and QDR paths. The
    metadata records *what* AVAS chose, which is the whole point of using it over an index
    window -- the selection is auditable, not implicit.
    """

    integrals: IntegralSet
    active_space: ActiveSpace
    ao_labels: tuple[str, ...]
    threshold: float


def avas_active_integrals(
    geometry: Geometry,
    *,
    ao_labels: tuple[str, ...] | list[str],
    basis: str = "sto-3g",
    threshold: float = DEFAULT_AVAS_THRESHOLD,
    max_active_orbitals: int | None = None,
    verbose: int = 0,
) -> AvasResult:
    """Select an active space by AVAS and return integrals restricted to it.

    ``ao_labels`` names the target valence orbitals, in PySCF's label syntax
    (``["Fe 3d"]``, ``["O 2p"]``, ...). Closed-shell geometries use RHF, open-shell use
    ROHF -- the same single-coefficient-set reference the rest of the spin-gap path relies
    on. The returned active space is guaranteed to span the requested orbitals; how many it
    picks (``ncas``) is AVAS's decision, recorded in the result.

    ``threshold`` is the AVAS overlap cutoff (its count of active orbitals is non-increasing
    in it). ``max_active_orbitals`` optionally caps the selection to a qubit budget: a rich
    target like ``("Fe 3d", "F 2p")`` can pull in every ligand p shell and overrun what an
    exact reference can hold, so when the initial threshold selects too many, this raises the
    threshold on a fine grid and keeps the *largest* space that still fits -- the most
    covalency the budget allows, chosen automatically rather than by hand-tuning. The
    threshold actually used is recorded in the result.
    """
    if not ao_labels:
        raise ValidationError(
            "ao_labels is empty; AVAS needs target valence orbitals to select"
        )

    pyscf = _require_pyscf()
    from pyscf import ao2mo
    from pyscf.mcscf import avas

    molecule = pyscf.gto.Mole()
    molecule.atom = to_pyscf_atom_spec(geometry)
    molecule.basis = basis
    molecule.charge = geometry.charge
    molecule.spin = geometry.spin_multiplicity - 1
    molecule.unit = "Angstrom"
    molecule.verbose = verbose
    molecule.build()

    # Robust SCF: [FeF6]^4- and other charged transition-metal centres routinely stall
    # first-order DIIS, so this escalates to the second-order solver before giving up.
    mean_field = run_reference_scf(molecule)
    e_hf = float(mean_field.e_tot)
    if not mean_field.converged:
        raise ValidationError(
            "SCF did not converge; the geometry or charge/spin state is likely unphysical",
            e_hf=e_hf,
        )

    labels = list(ao_labels)
    used_threshold = threshold
    n_active_orbitals, n_active_electrons, mo_coeff = avas.avas(
        mean_field, labels, threshold=used_threshold
    )
    if max_active_orbitals is not None and int(n_active_orbitals) > max_active_orbitals:
        # Raise the threshold on a fine grid and take the smallest one whose selection fits:
        # since the AVAS orbital count only ever shrinks as the threshold rises, the first
        # fit is the largest active space within the budget. AVAS is cheap on a fixed SCF,
        # so this re-projection loop never repeats the (expensive) mean-field solve.
        for candidate in np.round(np.arange(threshold + 0.05, 1.0, 0.05), 3):
            used_threshold = float(candidate)
            n_active_orbitals, n_active_electrons, mo_coeff = avas.avas(
                mean_field, labels, threshold=used_threshold
            )
            if int(n_active_orbitals) <= max_active_orbitals:
                break
        if int(n_active_orbitals) > max_active_orbitals:
            raise ValidationError(
                "AVAS could not fit the requested orbitals within the active-space budget "
                "even at a near-unit threshold; trim ao_labels or raise the budget",
                max_active_orbitals=max_active_orbitals,
                n_active_orbitals=int(n_active_orbitals),
            )
    # AVAS may report active electrons as a total or an (alpha, beta) split; the active
    # space carries the total (the spin sector is chosen later from the spin multiplicity).
    n_active_electrons = int(np.sum(n_active_electrons))
    n_active_orbitals = int(n_active_orbitals)

    # AVAS returns the full MO set reordered as [frozen-core | active | virtual]. Build the
    # integrals in that basis, then let the tested frozen-core reduction do the rest.
    full = _integrals_in_mo_basis(molecule, mean_field, ao2mo, mo_coeff, e_hf)
    n_frozen_core = (molecule.nelectron - n_active_electrons) // 2
    active_space = ActiveSpace(
        n_active_orbitals=n_active_orbitals,
        n_active_electrons=n_active_electrons,
        orbital_indices=tuple(range(n_frozen_core, n_frozen_core + n_active_orbitals)),
        n_frozen_core_orbitals=n_frozen_core,
    )

    return AvasResult(
        integrals=restrict_integrals(full, active_space),
        active_space=active_space,
        ao_labels=tuple(ao_labels),
        threshold=used_threshold,
    )


def _integrals_in_mo_basis(
    molecule: Any,
    mean_field: Any,
    ao2mo: Any,
    mo_coeff: NDArray[np.float64],
    e_hf: float,
) -> IntegralSet:
    """Full-space integrals in a supplied MO basis (mirrors ``compute_integrals``)."""
    n_orbitals = int(mo_coeff.shape[1])
    h_core_ao = mean_field.get_hcore()
    h1_mo = mo_coeff.T @ h_core_ao @ mo_coeff
    eri_mo_chemist = ao2mo.restore(1, ao2mo.kernel(molecule, mo_coeff), n_orbitals)
    return IntegralSet(
        h1=np.ascontiguousarray(h1_mo),
        h2=chemist_to_physicist(np.ascontiguousarray(eri_mo_chemist)),
        e_nuc=float(molecule.energy_nuc()),
        n_electrons=int(molecule.nelectron),
        e_hf=e_hf,
    )

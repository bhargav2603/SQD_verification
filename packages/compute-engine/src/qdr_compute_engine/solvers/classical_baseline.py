"""Classical reference solvers (PySCF).

FCI is the exact answer within a given basis and active space, which makes it the yardstick
every quantum result in this engine is measured against -- the stage-0 gate is precisely
the claim "our quantum path reproduces FCI for H2". It is also exponentially scaling, so
it only exists as a reference for small active spaces, not as a production method.

PySCF is imported lazily here for the same reason as in ``integrals/pyscf_integrals.py``.
"""

from dataclasses import dataclass
from math import comb

import numpy as np

from qdr_compute_engine.integrals.pyscf_integrals import (
    _require_pyscf,
    physicist_to_chemist,
)
from qdr_compute_engine.types import IntegralSet
from qdr_core.errors import ValidationError

#: Penalty strength (Hartree) on wrong-``S^2`` states in the spin-projected FCI. Matches the
#: qubit-side ``DEFAULT_SPIN_PENALTY_SHIFT`` in ``hamiltonian/spin_penalty.py`` in spirit: a
#: spin gap is at most ~0.1 Ha, so 1.0 Ha separates the target spin from its contaminants far
#: enough that the Davidson solver does not settle on a near-degenerate wrong-spin neighbour.
_SPIN_PENALTY_SHIFT = 1.0
#: How many of the lowest (penalised) roots to compute and screen by ``<S^2>``. ``fix_spin_``
#: alone makes the target the ground root, but a strongly correlated low-spin state (e.g. the
#: singlet of a high-spin d6 metal) can trap a single-root Davidson on a higher solution;
#: iterating a small window of roots gives it the subspace it needs to reach the true minimum.
_SPIN_ROOT_PROBE = 4
#: ``<S^2>`` tolerance for accepting a root as the requested spin (``S(S+1)`` is well
#: separated between multiplicities: 0, 2, 6, ..., so 0.1 is comfortably tight).
_SPIN_MATCH_TOL = 0.1


@dataclass(frozen=True)
class ClassicalResult:
    energy: float
    method: str
    converged: bool = True
    #: ``<S^2>`` of the solved state and the corresponding spin multiplicity ``2S+1``.
    #: Populated for open-shell solves so the caller can confirm it found the intended spin
    #: state (a singlet solver landing on a triplet is a silent, believable-wrong gap).
    spin_squared: float | None = None
    multiplicity: float | None = None


def alpha_beta_from_spin(n_electrons: int, spin_multiplicity: int) -> tuple[int, int]:
    """Split ``n_electrons`` into ``(n_alpha, n_beta)`` for a given spin multiplicity.

    ``spin_multiplicity = 2S + 1``, so the number of unpaired electrons is
    ``spin_multiplicity - 1``; those are the excess alpha electrons. The pairing constraint
    (``n_electrons`` and the unpaired count must have the same parity) and the physical bound
    (``n_beta >= 0``) are the only ways this can be inconsistent, and both are caught here so
    the FCI call downstream never receives a nonsense sector.
    """
    if spin_multiplicity < 1:
        raise ValidationError(
            f"spin_multiplicity must be >= 1, got {spin_multiplicity}",
            spin_multiplicity=spin_multiplicity,
        )
    n_unpaired = spin_multiplicity - 1
    if (n_electrons - n_unpaired) % 2 != 0:
        raise ValidationError(
            f"{n_electrons} electrons is inconsistent with spin multiplicity "
            f"{spin_multiplicity}: the {n_unpaired} unpaired electrons leave an odd remainder "
            "to pair up",
            n_electrons=n_electrons,
            spin_multiplicity=spin_multiplicity,
        )
    n_alpha = (n_electrons + n_unpaired) // 2
    n_beta = (n_electrons - n_unpaired) // 2
    if n_beta < 0:
        raise ValidationError(
            f"{n_electrons} electrons cannot carry {n_unpaired} unpaired spins "
            f"(multiplicity {spin_multiplicity})",
            n_electrons=n_electrons,
            spin_multiplicity=spin_multiplicity,
        )
    return n_alpha, n_beta


def solve_fci(integrals: IntegralSet, *, spin_multiplicity: int = 1) -> ClassicalResult:
    """Exact diagonalisation in the given orbital space, via PySCF's FCI solver.

    ``spin_multiplicity`` selects the target spin. Fixing ``(n_alpha, n_beta)`` alone fixes
    only ``S_z``, whose lowest state is whichever *total* spin is the ground state -- so a
    requested singlet in the ``S_z = 0`` sector silently comes back as the ``M_s = 0``
    component of a lower-lying triplet. Two things pin the actual spin here:

    * a wrong-``S^2`` penalty (PySCF's ``fix_spin_``), zero on the target spin so the returned
      energy is physical, that lifts contaminants above it; and
    * a small window of the lowest roots, screened by ``<S^2>``, with the lowest matching root
      taken. ``fix_spin_`` makes the target the ground root, but a strongly correlated
      low-spin state (the singlet of a high-spin d6 metal is the motivating case) can trap a
      single-root Davidson on a higher solution; the multi-root window is what lets it reach
      the true minimum and match the matrix-free QDR result to the cross-check tolerance.

    ``<S^2>`` is returned so the caller can confirm the solver landed where it asked.
    """
    _require_pyscf()
    from pyscf import fci

    n_orbitals = integrals.n_orbitals
    n_alpha, n_beta = alpha_beta_from_spin(integrals.n_electrons, spin_multiplicity)

    eri_chemist = physicist_to_chemist(np.asarray(integrals.h2, dtype=float))

    spin = (spin_multiplicity - 1) / 2.0
    target_s_squared = spin * (spin + 1.0)

    solver = fci.direct_spin1.FCI()
    solver.conv_tol = 1e-10
    fci.addons.fix_spin_(solver, shift=_SPIN_PENALTY_SHIFT, ss=target_s_squared)
    n_determinants = comb(n_orbitals, n_alpha) * comb(n_orbitals, n_beta)
    solver.nroots = max(1, min(_SPIN_ROOT_PROBE, n_determinants))

    result = solver.kernel(
        np.asarray(integrals.h1, dtype=float),
        eri_chemist,
        n_orbitals,
        (n_alpha, n_beta),
    )
    # kernel returns (energy, civec) for one root and (energies, civecs) for several.
    energies = np.atleast_1d(result[0])
    civecs = result[1] if solver.nroots > 1 else [result[1]]

    # Roots come back energy-ordered, so the first one whose <S^2> matches the request is the
    # lowest state of that spin -- and its penalty contribution is zero, so its energy is
    # physical (the frozen-core-shifted nuclear repulsion in e_nuc is added back below).
    for root_energy, civec in zip(energies, civecs):
        spin_squared, multiplicity = fci.spin_square(
            civec, n_orbitals, (n_alpha, n_beta)
        )
        if abs(spin_squared - target_s_squared) < _SPIN_MATCH_TOL:
            return ClassicalResult(
                energy=float(root_energy) + float(integrals.e_nuc),
                method="fci",
                spin_squared=float(spin_squared),
                multiplicity=float(multiplicity),
            )

    raise ValidationError(
        "no FCI root in the probed window carries the requested spin multiplicity; "
        "the active space may not support it, or the probe window is too small",
        spin_multiplicity=spin_multiplicity,
        target_s_squared=target_s_squared,
        probed_roots=int(solver.nroots),
    )


def solve_ccsd(integrals: IntegralSet) -> ClassicalResult:
    """CCSD in the given orbital space.

    Not implemented from bare integrals: PySCF's CC module is built around a mean-field
    object, and reconstructing a faithful one from integrals alone is exactly the kind of
    plumbing that produces a subtly wrong reference value. CCSD arrives when the pipeline
    carries the mean-field object forward.
    """
    raise NotImplementedError(
        "CCSD from bare integrals is not implemented; use solve_fci for small active spaces"
    )

"""Classical reference solvers (PySCF).

FCI is the exact answer within a given basis and active space, which makes it the yardstick
every quantum result in this engine is measured against -- the stage-0 gate is precisely
the claim "our quantum path reproduces FCI for H2". It is also exponentially scaling, so
it only exists as a reference for small active spaces, not as a production method.

PySCF is imported lazily here for the same reason as in ``integrals/pyscf_integrals.py``.
"""

from dataclasses import dataclass

import numpy as np

from qdr_compute_engine.integrals.pyscf_integrals import (
    _require_pyscf,
    physicist_to_chemist,
)
from qdr_compute_engine.types import IntegralSet
from qdr_core.errors import ValidationError


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

    ``spin_multiplicity`` selects the ``(n_alpha, n_beta)`` sector: the default ``1`` is the
    closed-shell singlet and reproduces the original behaviour exactly. A high-spin state
    (e.g. a quintet, ``spin_multiplicity=5``) is the mechanism behind a transition-metal spin
    gap. ``<S^2>`` is returned so the caller can verify the solver landed on the spin state it
    asked for rather than a lower-energy contaminant of the same ``S_z``.
    """
    _require_pyscf()
    from pyscf import fci

    n_orbitals = integrals.n_orbitals
    n_alpha, n_beta = alpha_beta_from_spin(integrals.n_electrons, spin_multiplicity)

    eri_chemist = physicist_to_chemist(np.asarray(integrals.h2, dtype=float))

    energy, civec = fci.direct_spin1.kernel(
        np.asarray(integrals.h1, dtype=float),
        eri_chemist,
        n_orbitals,
        (n_alpha, n_beta),
    )
    spin_squared, multiplicity = fci.spin_square(civec, n_orbitals, (n_alpha, n_beta))
    # PySCF's FCI kernel returns the electronic energy; the nuclear repulsion (plus any
    # frozen-core shift folded into e_nuc) is added back here.
    return ClassicalResult(
        energy=float(energy) + float(integrals.e_nuc),
        method="fci",
        spin_squared=float(spin_squared),
        multiplicity=float(multiplicity),
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

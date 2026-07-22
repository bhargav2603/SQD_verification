"""Classical reference solvers (PySCF).

FCI is the exact answer within a given basis and active space, which makes it the yardstick
every quantum result in this engine is measured against -- the stage-0 gate is precisely
the claim "our quantum path reproduces FCI for H2". It is also exponentially scaling, so
it only exists as a reference for small active spaces, not as a production method.

PySCF is imported lazily here for the same reason as in ``integrals/pyscf_integrals.py``.
"""

from dataclasses import dataclass

import numpy as np

from qdr_compute_engine.integrals.pyscf_integrals import _require_pyscf, physicist_to_chemist
from qdr_compute_engine.types import IntegralSet
from qdr_core.errors import ValidationError


@dataclass(frozen=True)
class ClassicalResult:
    energy: float
    method: str
    converged: bool = True


def solve_fci(integrals: IntegralSet) -> ClassicalResult:
    """Exact diagonalisation in the given orbital space, via PySCF's FCI solver."""
    _require_pyscf()
    from pyscf import fci

    n_orbitals = integrals.n_orbitals
    n_electrons = integrals.n_electrons
    if n_electrons % 2 != 0:
        raise ValidationError(
            f"solve_fci expects a closed-shell electron count, got {n_electrons}",
            n_electrons=n_electrons,
        )

    eri_chemist = physicist_to_chemist(np.asarray(integrals.h2, dtype=float))
    n_alpha = n_electrons // 2

    energy, _ = fci.direct_spin1.kernel(
        np.asarray(integrals.h1, dtype=float),
        eri_chemist,
        n_orbitals,
        (n_alpha, n_alpha),
    )
    # PySCF's FCI kernel returns the electronic energy; the nuclear repulsion (plus any
    # frozen-core shift folded into e_nuc) is added back here.
    return ClassicalResult(energy=float(energy) + float(integrals.e_nuc), method="fci")


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

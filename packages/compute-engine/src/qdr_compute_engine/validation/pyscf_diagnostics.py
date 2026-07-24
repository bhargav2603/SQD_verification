"""PySCF-backed correlation diagnostics for quantitative active-space claims.

The CASCI/QDR equality gate validates implementation of one Hamiltonian.  This module asks
two different chemistry questions: whether orbital relaxation plus external-space
correlation changes the result (CASSCF/NEVPT2), and whether an independently parameterized
single-reference calculation agrees (CCSD(T)).  Imports remain lazy so Windows installs
without PySCF can still import the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from qdr_compute_engine.active_space.avas_selection import AvasResult
from qdr_compute_engine.solvers.classical_baseline import alpha_beta_from_spin
from qdr_core.errors import ValidationError


@dataclass(frozen=True, slots=True)
class MultireferenceResult:
    converged: bool
    e_casscf: float
    e_nevpt2_correction: float | None
    e_nevpt2: float | None
    spin_squared: float | None
    multiplicity: float | None


@dataclass(frozen=True, slots=True)
class CoupledClusterResult:
    converged: bool
    e_ccsd: float
    triples_correction: float | None
    e_ccsd_t: float | None
    t1_diagnostic: float | None
    frozen_orbitals: int | None


def check_scf_stability(mean_field: Any) -> bool | None:
    """Return PySCF's internal RHF/ROHF stability status when available."""
    stability = getattr(mean_field, "stability", None)
    if stability is None:
        return None
    try:
        result = stability(internal=True, external=False, return_status=True)
    except (NotImplementedError, TypeError):
        return None
    items = result if isinstance(result, tuple) else (result,)
    statuses = [bool(item) for item in items if isinstance(item, (bool, np.bool_))]
    return statuses[0] if statuses else None


def _require_avas_work_objects(avas: AvasResult) -> tuple[Any, Any]:
    if avas.mean_field is None or avas.mo_coeff is None:
        raise ValidationError(
            "AVAS result does not retain the PySCF mean-field and MO coefficients; "
            "rebuild it with avas_active_integrals before post-correlation"
        )
    return avas.mean_field, avas.mo_coeff


def run_casscf_nevpt2(
    avas: AvasResult,
    *,
    spin_multiplicity: int,
    conv_tol: float = 1.0e-8,
    max_cycle_macro: int = 100,
) -> MultireferenceResult:
    """Optimize the AVAS orbitals and apply strongly contracted NEVPT2."""
    from pyscf import mcscf, mrpt

    mean_field, mo_coeff = _require_avas_work_objects(avas)
    active = avas.active_space
    n_alpha, n_beta = alpha_beta_from_spin(
        active.n_active_electrons,
        spin_multiplicity,
    )
    spin = 0.5 * (spin_multiplicity - 1)
    target_s_squared = spin * (spin + 1.0)

    casscf = mcscf.CASSCF(
        mean_field,
        active.n_active_orbitals,
        (n_alpha, n_beta),
    )
    casscf.conv_tol = conv_tol
    casscf.max_cycle_macro = max_cycle_macro
    casscf.fix_spin_(shift=1.0, ss=target_s_squared)
    casscf.kernel(mo_coeff)
    e_casscf = float(casscf.e_tot)

    if not casscf.converged:
        return MultireferenceResult(
            converged=False,
            e_casscf=e_casscf,
            e_nevpt2_correction=None,
            e_nevpt2=None,
            spin_squared=None,
            multiplicity=None,
        )

    spin_squared, multiplicity = casscf.fcisolver.spin_square(
        casscf.ci,
        casscf.ncas,
        casscf.nelecas,
    )
    correction = float(mrpt.NEVPT(casscf).kernel())
    return MultireferenceResult(
        converged=True,
        e_casscf=e_casscf,
        e_nevpt2_correction=correction,
        e_nevpt2=e_casscf + correction,
        spin_squared=float(spin_squared),
        multiplicity=float(multiplicity),
    )


def run_ccsd_t(
    mean_field: Any,
    *,
    frozen_orbitals: int | None = None,
    conv_tol: float = 1.0e-8,
    max_cycle: int = 100,
) -> CoupledClusterResult:
    """Run the PySCF CCSD dispatcher and perturbative triples on its SCF reference."""
    from pyscf import cc

    if frozen_orbitals is not None and frozen_orbitals < 0:
        raise ValidationError(
            "frozen_orbitals must be non-negative",
            frozen_orbitals=frozen_orbitals,
        )
    solver = cc.CCSD(mean_field, frozen=frozen_orbitals)
    solver.conv_tol = conv_tol
    solver.max_cycle = max_cycle
    solver.kernel()
    e_ccsd = float(solver.e_tot)
    diagnostic = getattr(solver, "get_t1_diagnostic", None)
    t1 = None if diagnostic is None else float(diagnostic())
    if not solver.converged:
        return CoupledClusterResult(
            converged=False,
            e_ccsd=e_ccsd,
            triples_correction=None,
            e_ccsd_t=None,
            t1_diagnostic=t1,
            frozen_orbitals=frozen_orbitals,
        )
    triples = float(solver.ccsd_t())
    return CoupledClusterResult(
        converged=True,
        e_ccsd=e_ccsd,
        triples_correction=triples,
        e_ccsd_t=e_ccsd + triples,
        t1_diagnostic=t1,
        frozen_orbitals=frozen_orbitals,
    )

"""Stage 4: active-space selection.

This is the engine-side enforcement point for the platform qubit cap (ADR-0006). The
control plane refuses oversized configurations pre-submit and the dispatcher refuses them
at admission, but the engine checks again anyway -- those two gates protect the queue,
while this one protects the worker process that would otherwise try to allocate the
statevector. Defence in depth is warranted when the failure mode is an OOM kill rather
than an error response.
"""

import numpy as np

from qdr_compute_engine.types import ActiveSpace, IntegralSet
from qdr_core.errors import QubitCapExceeded, ValidationError
from qdr_core.limits import MAX_QUBITS, qubits_for_orbitals


def enforce_qubit_cap(n_active_orbitals: int) -> None:
    """Raise ``QubitCapExceeded`` if an active space would exceed the platform cap."""
    required = qubits_for_orbitals(n_active_orbitals)
    if required > MAX_QUBITS:
        raise QubitCapExceeded(
            required_qubits=required,
            max_qubits=MAX_QUBITS,
            n_active_orbitals=n_active_orbitals,
        )


def select_active_space(
    integrals: IntegralSet,
    *,
    n_active_orbitals: int | None = None,
    n_active_electrons: int | None = None,
) -> ActiveSpace:
    """Select a HOMO/LUMO-centred active space.

    Orbitals are taken as a contiguous window around the Fermi level, which is the
    standard CAS(n,m) construction: the lowest occupied orbitals are frozen, the highest
    virtuals dropped, and correlation is treated in the window between. Selection is by
    orbital index -- valid because the integrals arrive in an energy-ordered MO basis.

    With no arguments this returns the full space, which is exact but only tractable for
    small systems; the qubit cap is what stops that being a footgun.
    """
    n_orbitals = integrals.n_orbitals
    n_electrons = integrals.n_electrons

    if n_active_orbitals is None:
        n_active_orbitals = n_orbitals
    if n_active_electrons is None:
        n_active_electrons = min(n_electrons, 2 * n_active_orbitals)

    if n_active_orbitals <= 0:
        raise ValidationError(f"n_active_orbitals must be positive, got {n_active_orbitals}")
    if n_active_orbitals > n_orbitals:
        raise ValidationError(
            f"cannot select {n_active_orbitals} active orbitals from a {n_orbitals}-orbital basis",
            requested=n_active_orbitals,
            available=n_orbitals,
        )
    if n_active_electrons < 0:
        raise ValidationError(f"n_active_electrons must be non-negative, got {n_active_electrons}")
    if n_active_electrons > n_electrons:
        raise ValidationError(
            f"cannot correlate {n_active_electrons} electrons; the system has {n_electrons}",
            requested=n_active_electrons,
            available=n_electrons,
        )
    if n_active_electrons > 2 * n_active_orbitals:
        raise ValidationError(
            f"{n_active_electrons} electrons cannot occupy {n_active_orbitals} spatial orbitals",
            n_active_electrons=n_active_electrons,
            n_active_orbitals=n_active_orbitals,
        )

    # Check the cap before doing anything else expensive with this selection.
    enforce_qubit_cap(n_active_orbitals)

    n_frozen_core = (n_electrons - n_active_electrons) // 2
    if n_frozen_core + n_active_orbitals > n_orbitals:
        raise ValidationError(
            f"freezing {n_frozen_core} core orbitals leaves too few for "
            f"{n_active_orbitals} active orbitals in a {n_orbitals}-orbital basis",
            n_frozen_core=n_frozen_core,
            n_active_orbitals=n_active_orbitals,
            n_orbitals=n_orbitals,
        )

    orbital_indices = tuple(range(n_frozen_core, n_frozen_core + n_active_orbitals))

    return ActiveSpace(
        n_active_orbitals=n_active_orbitals,
        n_active_electrons=n_active_electrons,
        orbital_indices=orbital_indices,
        n_frozen_core_orbitals=n_frozen_core,
    )


def restrict_integrals(integrals: IntegralSet, active_space: ActiveSpace) -> IntegralSet:
    """Project integrals onto the active orbitals.

    The frozen-core contribution is folded into ``e_nuc`` as a constant energy shift,
    computed from the standard closed-shell core expression

        E_core = sum_i 2*h_ii + sum_ij (2*<ij|ij> - <ij|ji>)

    over frozen orbitals i, j, plus the core-active mean-field term added to the
    one-body operator. Dropping either term leaves the total energy wrong by a constant,
    which is invisible in a variational optimisation but corrupts every energy difference.
    """
    core = list(range(active_space.n_frozen_core_orbitals))
    active = list(active_space.orbital_indices)

    if not core:
        h1_active = integrals.h1[np.ix_(active, active)]
        h2_active = integrals.h2[np.ix_(active, active, active, active)]
        return IntegralSet(
            h1=np.ascontiguousarray(h1_active),
            h2=np.ascontiguousarray(h2_active),
            e_nuc=integrals.e_nuc,
            n_electrons=active_space.n_active_electrons,
            e_hf=integrals.e_hf,
        )

    h1 = integrals.h1
    h2 = integrals.h2  # physicist notation <pq|rs>

    e_core = 2.0 * sum(h1[i, i] for i in core)
    for i in core:
        for j in core:
            e_core += 2.0 * h2[i, j, i, j] - h2[i, j, j, i]

    # Mean field of the frozen core, seen by the active electrons.
    h1_eff = h1.copy()
    for p in range(integrals.n_orbitals):
        for q in range(integrals.n_orbitals):
            correction = 0.0
            for i in core:
                correction += 2.0 * h2[i, p, i, q] - h2[i, p, q, i]
            h1_eff[p, q] += correction

    return IntegralSet(
        h1=np.ascontiguousarray(h1_eff[np.ix_(active, active)]),
        h2=np.ascontiguousarray(h2[np.ix_(active, active, active, active)]),
        e_nuc=integrals.e_nuc + float(e_core),
        n_electrons=active_space.n_active_electrons,
        e_hf=integrals.e_hf,
    )

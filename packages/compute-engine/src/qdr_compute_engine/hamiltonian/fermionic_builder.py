"""Stage 5: build the second-quantised electronic Hamiltonian.

Takes spatial-orbital integrals and produces an OpenFermion ``InteractionOperator`` over
spin orbitals:

    H = E_nuc + sum_pq h_pq a+_p a_q + 1/2 sum_pqrs h_pqrs a+_p a+_q a_r a_s

Spin-orbital indexing is interleaved -- spatial orbital ``p`` becomes spin orbitals
``2p`` (alpha) and ``2p+1`` (beta). This is OpenFermion's convention, and we adopt it
rather than translating because ``get_sparse_operator`` and ``jordan_wigner`` both assume
it. A mismatch here produces a Hermitian operator with a believable spectrum that is
simply not the molecule's, so the H2/FCI gate is the only thing that would catch it.

``integrals.h2`` must be in **physicist** notation ``<pq|rs>``; the chemist-to-physicist
transpose happens once, in ``integrals/pyscf_integrals.py``.
"""

import numpy as np
from openfermion import InteractionOperator, get_fermion_operator

from qdr_compute_engine.types import ActiveSpace, FermionicHamiltonian, IntegralSet
from qdr_core.errors import ValidationError


def build_fermionic_hamiltonian(
    integrals: IntegralSet,
    active_space: ActiveSpace | None = None,
) -> FermionicHamiltonian:
    """Construct the spin-orbital Hamiltonian from spatial-orbital integrals."""
    n_spatial = integrals.n_orbitals
    if active_space is not None and active_space.n_active_orbitals != n_spatial:
        raise ValidationError(
            f"integrals span {n_spatial} orbitals but the active space declares "
            f"{active_space.n_active_orbitals}; restrict the integrals first",
            n_orbitals=n_spatial,
            n_active_orbitals=active_space.n_active_orbitals,
        )

    n_spin_orbitals = 2 * n_spatial
    h1 = np.asarray(integrals.h1, dtype=float)
    h2 = np.asarray(integrals.h2, dtype=float)

    one_body = np.zeros((n_spin_orbitals, n_spin_orbitals), dtype=complex)
    two_body = np.zeros((n_spin_orbitals,) * 4, dtype=complex)

    for p in range(n_spatial):
        for q in range(n_spatial):
            # Spin-diagonal one-body terms: the operator is spin-free.
            one_body[2 * p, 2 * q] = h1[p, q]
            one_body[2 * p + 1, 2 * q + 1] = h1[p, q]

            for r in range(n_spatial):
                for s in range(n_spatial):
                    value = h2[p, q, r, s] / 2.0
                    # Mixed spin (alpha-beta and beta-alpha).
                    two_body[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = value
                    two_body[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = value
                    # Same spin (alpha-alpha and beta-beta).
                    two_body[2 * p, 2 * q, 2 * r, 2 * s] = value
                    two_body[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = value

    operator = InteractionOperator(float(integrals.e_nuc), one_body, two_body)
    return FermionicHamiltonian(
        operator=operator,
        n_spin_orbitals=n_spin_orbitals,
        constant=float(integrals.e_nuc),
    )


def to_fermion_operator(hamiltonian: FermionicHamiltonian) -> object:
    """Expand an ``InteractionOperator`` into an explicit ``FermionOperator`` sum."""
    return get_fermion_operator(hamiltonian.operator)


def hartree_fock_occupations(n_electrons: int, n_spin_orbitals: int) -> tuple[int, ...]:
    """Spin orbitals occupied in the Hartree-Fock reference determinant.

    Under interleaved, energy-ordered indexing the reference simply fills the lowest
    ``n_electrons`` spin orbitals, which is why the reference state preparation is a
    row of X gates and nothing more.
    """
    if n_electrons > n_spin_orbitals:
        raise ValidationError(
            f"cannot place {n_electrons} electrons in {n_spin_orbitals} spin orbitals",
            n_electrons=n_electrons,
            n_spin_orbitals=n_spin_orbitals,
        )
    return tuple(range(n_electrons))

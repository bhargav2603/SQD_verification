"""Spin (S^2) projection for the qubit Hamiltonian.

Why this exists
---------------
Fixing ``(n_alpha, n_beta)`` fixes only ``S_z``, not the total spin ``S``. The ``S_z = 0``
sector of an open-shell molecule contains the ``M_s = 0`` component of *every* spin state it
can support -- singlet, triplet, quintet -- so the lowest eigenvalue there is whichever spin
happens to be the ground state, not the one the caller asked for. For O2 that means a
requested *singlet* comes back as the triplet's ``M_s = 0`` component: a silent,
believable-wrong result that even passes a QDR-vs-CASCI cross-check, because *both* engines
share the same blind spot.

The fix is to add a penalty ``shift * (S^2 - s(s+1))`` to the Hamiltonian. Because the
spin-free electronic Hamiltonian commutes with ``S^2``, its eigenstates are already ``S^2``
eigenstates, so the penalty is **exactly zero** on the target-spin state (its energy stays
physical) while every other-spin contaminant is lifted by ``shift * (s'(s'+1) - s(s+1))``.
For a spin-gap solve the requested state is always the *minimum* spin in its ``S_z`` sector
(a singlet at ``M_s = 0``, a quintet at ``M_s = 2``), so a positive ``shift`` lifts all
contaminants above it and isolates it cleanly.

``S^2`` is built by OpenFermion (``s_squared_operator``) in the same interleaved spin-orbital
convention (spatial ``p`` -> ``2p`` alpha, ``2p+1`` beta) as ``build_fermionic_hamiltonian``
and mapped through the same Jordan-Wigner transform, so it adds to the Hamiltonian operator
term-for-term with no index translation.
"""

from openfermion import QubitOperator, jordan_wigner, s_squared_operator

from qdr_compute_engine.types import QubitHamiltonian

#: Penalty strength (Hartree) on wrong-spin contaminants. A spin gap is at most ~0.1 Ha, so
#: 1.0 Ha lifts the nearest contaminant (``Delta(S^2) >= 1``) far above the target with margin
#: to spare, while staying small enough not to stress the sparse eigensolver.
DEFAULT_SPIN_PENALTY_SHIFT = 1.0


def s_squared_qubit_operator(n_spatial_orbitals: int) -> QubitOperator:
    """The total-spin operator ``S^2`` as a Jordan-Wigner qubit operator.

    Interleaved convention (spatial ``p`` -> spin orbitals ``2p`` alpha, ``2p+1`` beta),
    matching :func:`build_fermionic_hamiltonian` and OpenFermion's ``jordan_wigner``.
    """
    qubit_s2 = jordan_wigner(s_squared_operator(n_spatial_orbitals))
    qubit_s2.compress()
    return qubit_s2


def spin_penalized_hamiltonian(
    hamiltonian: QubitHamiltonian,
    *,
    spin_multiplicity: int,
    shift: float = DEFAULT_SPIN_PENALTY_SHIFT,
) -> QubitHamiltonian:
    """Add an ``S^2`` penalty so a fixed-``S_z`` solve lands on the requested spin.

    ``spin_multiplicity = 2S + 1`` sets the target ``S^2 = S(S+1)``. The returned Hamiltonian
    has the *same* eigenvalue as ``hamiltonian`` on that spin state (the penalty vanishes
    there) and a raised eigenvalue on every other spin, so the lowest eigenvalue in the
    ``(n_alpha, n_beta)`` sector is the target state -- see the module docstring.
    """
    n_spatial_orbitals = hamiltonian.n_qubits // 2
    spin = (spin_multiplicity - 1) / 2.0
    target_s_squared = spin * (spin + 1.0)

    penalty = shift * (
        s_squared_qubit_operator(n_spatial_orbitals)
        - target_s_squared * QubitOperator(())
    )
    penalized = hamiltonian.operator + penalty
    penalized.compress()

    return QubitHamiltonian(
        operator=penalized,
        n_qubits=hamiltonian.n_qubits,
        mapping=hamiltonian.mapping,
    )

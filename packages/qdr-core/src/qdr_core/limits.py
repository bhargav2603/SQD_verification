"""Platform invariants.

These are deliberately module-level constants rather than configuration. The qubit cap
is an architectural decision (ADR-0006), not a tunable: it bounds the statevector memory
a single worker can be asked to allocate, and every enforcement point in the platform
must agree on the same number.

Enforcement points that read from here:
  * pre-submit  -- apps/api/.../modules/configs/estimator/qubit_cap_policy.py
  * dispatch    -- services/dispatcher/.../admission_control.py
  * engine      -- packages/compute-engine/.../active_space/active_space_selector.py
"""

from typing import Final

# --------------------------------------------------------------------------------------
# The qubit ladder
# --------------------------------------------------------------------------------------
#
# There are three different ceilings and they are *not* interchangeable. They used to be
# scattered -- 40 here, 24 hard-coded in the statevector driver, and two differently-named
# constants (14 and 16) for the same idea in stage 9 and the benchmark runner. That drift
# is exactly what this module exists to prevent, so all four now live here.
#
#   MAX_QUBITS                  40   what the platform will schedule       (policy)
#   MAX_DENSE_STATEVECTOR_QUBITS 24  what one worker will allocate         (memory)
#   MAX_EXACT_REFERENCE_QUBITS   14  where the sparse Hamiltonian wins     (measured)
#
# Anything quoting a qubit ceiling must import one of these rather than restate it.

#: Hard ceiling on the number of qubits any single job may require (ADR-0006).
#: A dense statevector of n qubits is 2^n complex128 values; at n=40 that is 16 TiB, so
#: this cap is enforced *before* a job is ever scheduled, never discovered at runtime.
#: This is a scheduling policy, **not** an achievable size -- see the two limits below.
MAX_QUBITS: Final[int] = 40

#: Largest register the dense statevector backend will attempt: 2^24 complex128 = 256 MiB
#: for the state itself. A backend is free to be more conservative than the platform.
MAX_DENSE_STATEVECTOR_QUBITS: Final[int] = 24

#: Largest register for which an *exact reference* is computed by materialising the sparse
#: qubit Hamiltonian.
#:
#: Measured, not assumed. Sparse-Hamiltonian memory is ``n_terms * 2**n * 16`` bytes, which
#: overtakes the statevector and becomes the binding constraint well before it does: a
#: 16-qubit case required 5.66 GiB for the operator alone and failed there, not in the
#: simulator. 14 qubits is the largest that completes comfortably on a 16 GB host.
#:
#: Raising this needs a matrix-free Hamiltonian application (apply Pauli strings directly
#: to the statevector), not a bigger machine.
MAX_EXACT_REFERENCE_QUBITS: Final[int] = 14

#: Spin-orbital accounting: a spatial orbital contributes two spin orbitals, and the
#: Jordan-Wigner family of mappings assigns one qubit per spin orbital.
QUBITS_PER_SPATIAL_ORBITAL: Final[int] = 2

#: Largest active space expressible within MAX_QUBITS.
MAX_ACTIVE_ORBITALS: Final[int] = MAX_QUBITS // QUBITS_PER_SPATIAL_ORBITAL

#: "Chemical accuracy" in Hartree (1 kcal/mol). The threshold a binding-energy result
#: must beat for the validator to mark it trustworthy.
CHEMICAL_ACCURACY_HA: Final[float] = 1.6e-3

#: CODATA-2018 conversion factor.
HARTREE_TO_KCAL_PER_MOL: Final[float] = 627.5094740631


def qubits_for_orbitals(n_spatial_orbitals: int) -> int:
    """Return the qubit count required to encode ``n_spatial_orbitals``.

    This is the single definition of the ``n = 2 * n_orb`` accounting rule; the control
    plane's estimator and the engine's active-space selector both call it so their
    numbers can never drift apart.
    """
    if n_spatial_orbitals < 0:
        raise ValueError(f"n_spatial_orbitals must be non-negative, got {n_spatial_orbitals}")
    return QUBITS_PER_SPATIAL_ORBITAL * n_spatial_orbitals


def exceeds_qubit_cap(n_qubits: int) -> bool:
    """True when ``n_qubits`` is beyond what the platform will schedule."""
    return n_qubits > MAX_QUBITS

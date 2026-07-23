"""Matrix-free application of a qubit Hamiltonian (Rung 3).

Why this exists
---------------
The sparse reference path materialises the full ``2^n x 2^n`` operator via OpenFermion's
``get_sparse_operator``. Its memory cost is ``n_terms * 2^n * 16`` bytes, which overtakes
the ``2^n`` statevector and becomes the binding constraint far below it: a 16-qubit active
space needs ~5.66 GiB for the operator alone and dies there, which is why
``MAX_EXACT_REFERENCE_QUBITS`` sat at 14 (see :mod:`qdr_core.limits`).

This module removes that wall by never forming the matrix. A qubit Hamiltonian is a sum of
weighted Pauli strings, and a single Pauli string maps each computational-basis amplitude
to exactly one other amplitude (a permutation) times a ``+/-1`` or ``+/-i`` phase. So
``H @ psi`` is a sum, over terms, of one permute-and-phase pass across the ``2^n`` vector --
``O(2^n)`` memory, the *same* as the statevector, so the exact reference now scales with the
simulator (to ~24 qubits) instead of dying far below it.

Convention
----------
The state is a ``(2,)*n`` array with axis ``q`` = qubit ``q``; flattening (C order) puts
qubit 0 in the most significant bit. This is exactly the convention the statevector driver
and ``get_sparse_operator`` share (see ``statevector_driver`` module docstring), so a Pauli
applied to axis ``q`` here reproduces the sparse operator's action bit-for-bit up to
floating-point summation order. ``tests/test_matrix_free_hamiltonian.py`` pins every piece
against ``get_sparse_operator`` -- if the convention were off, the single-Pauli tests fail
immediately.

The single-qubit actions, on the two halves of axis ``q``:

* ``Z`` -- ``diag(1, -1)``: negate the ``1`` half in place, no temporary.
* ``X`` -- ``[[0, 1], [1, 0]]``: swap the two halves.
* ``Y`` -- ``[[0, -i], [i, 0]]``: ``new[0] = -i * old[1]``, ``new[1] = i * old[0]``.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, eigsh

from qdr_compute_engine.types import QubitHamiltonian
from qdr_core.errors import ValidationError

State = NDArray[np.complex128]

#: Below this eigsh cannot return ``k=1`` of ``dim`` eigenvalues, and a dense solve is
#: cheaper anyway. Mirrors the ``dimension <= 64`` branch in the sparse driver so the two
#: reference paths agree on the small-matrix handling too.
_DENSE_DIMENSION_CEILING = 64


def _axis(qubit: int, value: int) -> tuple[Any, ...]:
    """Index tuple selecting ``value`` on axis ``qubit`` (a view into the buffer)."""
    return (slice(None),) * qubit + (value,)


def _apply_pauli_string(work: State, paulis: tuple[tuple[int, str], ...]) -> None:
    """Apply one Pauli string to ``work`` in place (shape ``(2,)*n``).

    ``paulis`` is an OpenFermion term key: a tuple of ``(qubit, letter)`` pairs. The empty
    tuple is the identity term and leaves ``work`` untouched.
    """
    for qubit, letter in paulis:
        lower = _axis(qubit, 0)
        upper = _axis(qubit, 1)
        if letter == "Z":
            work[upper] *= -1.0
        elif letter == "X":
            temp = work[lower].copy()
            work[lower] = work[upper]
            work[upper] = temp
        elif letter == "Y":
            old_lower = work[lower].copy()
            work[lower] = -1j * work[upper]
            work[upper] = 1j * old_lower
        else:  # pragma: no cover - OpenFermion only emits X/Y/Z
            raise ValidationError(f"unknown Pauli letter {letter!r}", letter=letter)


def apply_qubit_operator(
    operator: Any, state: NDArray[np.complex128], n_qubits: int
) -> State:
    """Return ``H @ state`` without materialising ``H``.

    ``operator`` is an OpenFermion ``QubitOperator``; ``state`` is a length-``2^n`` vector
    (any shape that reshapes to it). The result is returned flat, matching the sparse path's
    ``matrix @ state``.
    """
    psi = np.asarray(state, dtype=complex).reshape((2,) * n_qubits)
    result = np.zeros_like(psi)
    for paulis, coefficient in operator.terms.items():
        work = psi.copy()
        _apply_pauli_string(work, paulis)
        result += coefficient * work
    return result.reshape(-1)


def qubit_operator_as_linear_operator(hamiltonian: QubitHamiltonian) -> LinearOperator:
    """A SciPy ``LinearOperator`` that applies ``hamiltonian`` matrix-free.

    ``H`` is Hermitian, so its action equals that of its conjugate transpose and the same
    callback serves as both ``matvec`` and ``rmatvec`` -- which is all ``eigsh`` needs.
    """
    dimension = 2**hamiltonian.n_qubits
    operator = hamiltonian.operator
    n_qubits = hamiltonian.n_qubits

    def matvec(vector: NDArray[np.complex128]) -> State:
        return apply_qubit_operator(operator, vector, n_qubits)

    return LinearOperator(
        (dimension, dimension), matvec=matvec, rmatvec=matvec, dtype=complex
    )


def ground_state_energy_matrix_free(hamiltonian: QubitHamiltonian) -> float:
    """Lowest eigenvalue of ``hamiltonian`` over all particle-number sectors, matrix-free.

    The unconstrained counterpart of the sparse driver's ``exact_ground_state_energy``; for
    a full-space molecular Hamiltonian this is the FCI energy, for an active-space effective
    Hamiltonian it is not (use the particle-number-restricted variant for a CASCI compare).
    """
    dimension = 2**hamiltonian.n_qubits
    if dimension <= _DENSE_DIMENSION_CEILING:
        matrix = _dense_from_linear_operator(
            qubit_operator_as_linear_operator(hamiltonian)
        )
        return float(np.linalg.eigvalsh(matrix)[0])

    eigenvalues = eigsh(
        qubit_operator_as_linear_operator(hamiltonian),
        k=1,
        which="SA",
        return_eigenvectors=False,
    )
    return float(eigenvalues[0])


def ground_state_energy_at_particle_number_matrix_free(
    hamiltonian: QubitHamiltonian, n_electrons: int
) -> float:
    """Lowest eigenvalue within the fixed ``n_electrons`` sector, matrix-free.

    Mirrors ``exact_ground_state_energy_at_particle_number``: under Jordan-Wigner the
    electron number of a basis state is its Hamming weight, so the sector is spanned by the
    basis indices with exactly ``n_electrons`` set bits. The Hamiltonian conserves particle
    number, so scattering a sector vector into the full ``2^n`` space, applying ``H``
    matrix-free, and gathering the sector entries back stays inside the sector exactly -- and
    never forms a matrix, keeping the whole thing ``O(2^n)`` in memory.
    """
    n_qubits = hamiltonian.n_qubits
    dimension = 2**n_qubits
    weights = _hamming_weights(n_qubits)
    sector = np.flatnonzero(weights == n_electrons).astype(np.int64)
    if sector.size == 0:
        raise ValidationError(
            f"no basis state holds {n_electrons} electrons in {n_qubits} spin orbitals",
            n_electrons=n_electrons,
            n_qubits=n_qubits,
        )
    return _lowest_eigenvalue_in_sector(
        hamiltonian.operator, n_qubits, dimension, sector
    )


def ground_state_energy_at_spin_sector_matrix_free(
    hamiltonian: QubitHamiltonian, n_alpha: int, n_beta: int
) -> float:
    """Lowest eigenvalue within the fixed ``(n_alpha, n_beta)`` sector, matrix-free.

    A spin gap needs the energy at a *fixed* number of alpha and beta electrons -- not just a
    fixed total -- because the high-spin state is exactly the ``n_alpha != n_beta`` sector.
    Under the interleaved Jordan-Wigner convention this engine uses (spatial orbital ``p`` ->
    spin orbitals ``2p`` alpha, ``2p+1`` beta), a basis state's alpha count is its set bits on
    even qubits and its beta count is the set bits on odd qubits. Fixing both selects a fixed
    ``S_z`` sector; the target spin state is then the lowest eigenvalue there (confirmed by an
    ``<S^2>`` reading, which the PySCF reference reports alongside).
    """
    n_qubits = hamiltonian.n_qubits
    dimension = 2**n_qubits
    alpha, beta = _alpha_beta_counts(n_qubits)
    sector = np.flatnonzero((alpha == n_alpha) & (beta == n_beta)).astype(np.int64)
    if sector.size == 0:
        raise ValidationError(
            f"no basis state has {n_alpha} alpha and {n_beta} beta electrons in "
            f"{n_qubits} spin orbitals",
            n_alpha=n_alpha,
            n_beta=n_beta,
            n_qubits=n_qubits,
        )
    return _lowest_eigenvalue_in_sector(
        hamiltonian.operator, n_qubits, dimension, sector
    )


def _hamming_weights(n_qubits: int) -> NDArray[np.int64]:
    """Population count of every index in ``range(2**n_qubits)``, vectorised."""
    indices = np.arange(2**n_qubits, dtype=np.int64)
    weights = np.zeros_like(indices)
    for bit in range(n_qubits):
        weights += (indices >> bit) & 1
    return weights


def _alpha_beta_counts(n_qubits: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Per-index alpha (even-qubit) and beta (odd-qubit) occupation, vectorised.

    Qubit ``q`` occupies bit ``n_qubits - 1 - q`` of the flat index (axis 0 is the most
    significant bit -- the convention this module shares with ``get_sparse_operator``).
    """
    indices = np.arange(2**n_qubits, dtype=np.int64)
    alpha = np.zeros_like(indices)
    beta = np.zeros_like(indices)
    for qubit in range(n_qubits):
        bit = (indices >> (n_qubits - 1 - qubit)) & 1
        if qubit % 2 == 0:
            alpha += bit
        else:
            beta += bit
    return alpha, beta


def _lowest_eigenvalue_in_sector(
    operator: Any, n_qubits: int, dimension: int, sector: NDArray[np.int64]
) -> float:
    """Lowest eigenvalue of ``H`` restricted to ``sector``, matrix-free.

    A number-conserving ``H`` maps the sector to itself, so scattering a sector vector into
    the full ``2^n`` space, applying ``H`` matrix-free, and gathering the sector entries back
    is exact and never forms a matrix -- ``O(2^n)`` memory, the whole point of the rung.
    """

    def matvec(vector: NDArray[np.complex128]) -> NDArray[np.complex128]:
        full = np.zeros(dimension, dtype=complex)
        full[sector] = vector
        applied = apply_qubit_operator(operator, full, n_qubits)
        return applied[sector]

    block = LinearOperator(
        (sector.size, sector.size), matvec=matvec, rmatvec=matvec, dtype=complex
    )
    if sector.size <= _DENSE_DIMENSION_CEILING:
        return float(np.linalg.eigvalsh(_dense_from_linear_operator(block))[0])

    eigenvalues = eigsh(block, k=1, which="SA", return_eigenvectors=False)
    return float(eigenvalues[0])


def _dense_from_linear_operator(operator: LinearOperator) -> NDArray[np.complex128]:
    """Materialise a small ``LinearOperator`` by applying it to each basis vector.

    Only ever called on ``dim <= 64`` blocks, where ``eigsh`` cannot return ``k=1`` and a
    dense ``eigvalsh`` is both possible and cheaper.
    """
    dimension = operator.shape[0]
    identity = np.eye(dimension, dtype=complex)
    return np.column_stack(
        [operator @ identity[:, index] for index in range(dimension)]
    )

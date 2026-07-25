"""Sample-based quantum diagonalisation (SQD).

What it is
----------
SQD replaces "measure an expectation value on a quantum state" with "let the quantum state
*nominate a basis*, then diagonalise classically inside it". A parameterised circuit is
sampled in the computational basis; each sampled bitstring is one electronic configuration;
the Hamiltonian is projected onto the span of the configurations that came back and the
lowest eigenvalue of that small dense matrix is the answer.

Why the engine wants it
-----------------------
The energy is a **Rayleigh quotient of a genuine state**, so it obeys the variational
principle: it can only ever sit *above* the true ground state of the sector, never below.
That is a materially different failure mode from VQE, where an optimiser stuck in a local
minimum and a noisy estimator both look like "a slightly high energy" and neither is a
bound on anything. Here a wrong answer is still an honest upper bound, and a subspace that
grows can only ever lower it. :func:`assert_variational_bound` is what turns that from a
claim in a docstring into something the code checks on every solve.

Where it sits
-------------
This module is the *algorithm*, in the same role as ``solvers/vqe.py`` (the optimisation
loop) and ``solvers/classical_baseline.py`` (the FCI reference): pure, importable anywhere,
no vendor SDK and no PySCF. A registered ``SolverBackend`` that dispatches to it is a
separate concern living in ``drivers/``; :class:`SQDSolver` deliberately does not decorate
itself with ``@register_backend``, matching how ``vqe.py`` and ``classical_baseline.py`` are
undecorated algorithms that a driver wraps later.

One honest caveat about that seam: :meth:`SQDSolver.solve` samples a circuit, and the only
simulator lives in ``drivers/``, so this module does import from the driver layer -- the
reverse of the ``drivers -> solvers`` direction, and the only such edge in the package. The
import is deferred into the method body, which keeps the pure functions here importable on
their own, but deferral is not a fix: the dependency is real. Removing it means moving
:class:`SQDSolver` into ``drivers/`` or injecting a sampler through the request.

It still speaks :class:`SolveRequest` / :class:`SolveResult` and still refuses by name --
every refusal is a ``BackendUnavailable`` carrying ``"sqd"``, so a caller that dispatches to
this solver gets the same error shape and the same ``qdr.backend_unavailable`` code it would
get from a driver, and a refusal is never mistakable for a computed energy.

Sector filtering
----------------
A sampled bitstring is *not* automatically a usable configuration. Under the interleaved
Jordan-Wigner convention (spatial orbital ``p`` -> spin orbitals ``2p`` alpha, ``2p+1``
beta, and qubit ``q`` at bit ``n_qubits - 1 - q`` of the flat index -- the convention shared
with ``get_sparse_operator``, see ``hamiltonian/matrix_free.py``) a bitstring's alpha count
is its set bits on even qubits and its beta count is its set bits on odd qubits. Shot noise,
hardware error, and an imperfectly particle-conserving ansatz all produce bitstrings in the
wrong ``(n_alpha, n_beta)`` sector. Keeping them does not merely add noise: it drags the
projected minimum into a different sector entirely and returns, with full confidence, the
energy of a molecule with the wrong number of electrons. So they are dropped, and the count
of what was dropped is reported rather than hidden -- a high discard rate is the signal that
the state being sampled is not the state that was intended.

Exactness
---------
With ``subspace_dimension=None`` the subspace *is* the whole sector, the projection is
exact, and the result equals CASCI/FCI in that active space. That is the property
``tests/test_sqd.py`` pins, and it needs no PySCF: the same claim checked against the
engine's own matrix-free sector solver runs on any platform.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, eigsh

from qdr_compute_engine.solvers.classical_baseline import alpha_beta_from_spin
from qdr_compute_engine.types import QubitHamiltonian, SolveRequest, SolveResult
from qdr_core.errors import BackendUnavailable, ValidationError
from qdr_core.limits import MAX_DENSE_STATEVECTOR_QUBITS, MAX_QUBITS

#: Registry-style name. Every refusal raised from this module carries it, so an error
#: reaching a caller says *which* solver declined rather than only that something did.
SOLVER_NAME = "sqd"

#: Largest sampled register this solver will accept. Projection acts directly on selected
#: determinant indices, so truncated SQD scales with subspace size rather than ``2^n``.
MAX_SQD_QUBITS = MAX_QUBITS

#: Exact mode enumerates the full sector and therefore retains the dense-memory ceiling.
MAX_EXACT_SQD_QUBITS = MAX_DENSE_STATEVECTOR_QUBITS

#: Below this, ``eigsh`` cannot return ``k=1`` of ``dim`` eigenvalues and a dense solve is
#: cheaper anyway. Mirrors ``hamiltonian/matrix_free.py`` so the two agree on small blocks.
_DENSE_DIMENSION_CEILING = 64

#: Samples drawn when a truncated subspace is requested without an explicit shot budget.
DEFAULT_SAMPLE_COUNT = 4096

#: Slack on the variational-bound check. The projected energy and the full-space Rayleigh
#: quotient of the same vector are the same quantity computed two ways, so they agree to
#: rounding; this is sized for accumulated floating-point error across ``2^n`` terms, not
#: for genuine physical disagreement.
VARIATIONAL_TOLERANCE = 1e-8

#: How far above the largest imaginary/asymmetric residue a projected block may sit before
#: it is treated as evidence of a convention bug rather than of rounding.
_HERMITICITY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class SQDOutcome:
    """The result of one subspace diagonalisation.

    ``is_exact`` is the load-bearing field: it is True only when the kept configurations
    span the entire ``(n_alpha, n_beta)`` sector, in which case the energy is not an upper
    bound at all but the exact sector ground state. Callers that compare against FCI need
    to know which of the two they were handed.
    """

    energy: float
    #: Configurations actually diagonalised over.
    subspace_dimension: int
    #: Size of the full ``(n_alpha, n_beta)`` sector, for context on the truncation.
    sector_dimension: int
    n_alpha: int
    n_beta: int
    #: Bitstrings drawn, and how many of those landed in the requested sector. Equal to
    #: ``(0, 0)`` when the subspace was taken whole rather than sampled.
    n_samples: int = 0
    n_samples_in_sector: int = 0
    is_exact: bool = False

    @property
    def n_samples_discarded(self) -> int:
        return self.n_samples - self.n_samples_in_sector

    @property
    def sector_retention(self) -> float:
        """Fraction of samples that were usable. 1.0 when nothing was sampled."""
        if self.n_samples == 0:
            return 1.0
        return self.n_samples_in_sector / self.n_samples


@dataclass(frozen=True, slots=True)
class SQDStep:
    """One member of a genuinely nested SQD convergence ladder."""

    requested_dimension: int
    subspace_dimension: int
    energy: float
    energy_change: float | None
    is_exact: bool
    #: Auditable determinant set, in the exact order used by the projected eigensolver.
    configurations: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SQDCampaign:
    """Instrumented SQD results across nested selected determinant spaces."""

    steps: tuple[SQDStep, ...]
    sector_dimension: int
    n_samples: int
    n_samples_in_sector: int
    reference_energy: float | None = None

    @property
    def sector_retention(self) -> float:
        if self.n_samples == 0:
            return 1.0
        return self.n_samples_in_sector / self.n_samples

    @property
    def final_energy_change(self) -> float | None:
        if len(self.steps) < 2:
            return None
        return self.steps[-1].energy_change

    @property
    def reference_deviation(self) -> float | None:
        if self.reference_energy is None:
            return None
        return abs(self.steps[-1].energy - self.reference_energy)


# --------------------------------------------------------------------------------------
# Sector filtering
# --------------------------------------------------------------------------------------


def alpha_beta_occupation(
    configurations: NDArray[np.int64], n_qubits: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Alpha (even-qubit) and beta (odd-qubit) electron counts of each configuration.

    Qubit ``q`` is bit ``n_qubits - 1 - q`` of the flat index: axis 0 is the most
    significant bit, the convention this engine shares with ``get_sparse_operator``. Getting
    this backwards silently swaps alpha and beta, which is invisible for a closed-shell
    singlet and wrong for everything else -- so it is derived here once and nowhere else.
    """
    indices = np.asarray(configurations, dtype=np.int64)
    alpha = np.zeros_like(indices)
    beta = np.zeros_like(indices)
    for qubit in range(n_qubits):
        bit = (indices >> (n_qubits - 1 - qubit)) & 1
        if qubit % 2 == 0:
            alpha += bit
        else:
            beta += bit
    return alpha, beta


def sector_configurations(n_qubits: int, n_alpha: int, n_beta: int) -> NDArray[np.int64]:
    """Every basis index with exactly ``n_alpha`` alpha and ``n_beta`` beta electrons.

    The full sector -- the subspace SQD would reach if it sampled forever. Enumerated
    rather than sampled, which is what makes the ``subspace_dimension=None`` path exact.
    """
    if n_qubits > MAX_EXACT_SQD_QUBITS:
        raise BackendUnavailable(
            SOLVER_NAME,
            f"enumerating the sector of a {n_qubits}-qubit register would materialise "
            f"2^{n_qubits} indices, above exact SQD's limit of {MAX_EXACT_SQD_QUBITS}",
            required_qubits=n_qubits,
            max_qubits=MAX_EXACT_SQD_QUBITS,
        )
    indices = np.arange(2**n_qubits, dtype=np.int64)
    alpha, beta = alpha_beta_occupation(indices, n_qubits)
    return indices[(alpha == n_alpha) & (beta == n_beta)]


def sector_dimension(n_qubits: int, n_alpha: int, n_beta: int) -> int:
    """Combinatorial sector size without enumerating any bitstrings."""
    if n_qubits < 0 or n_qubits % 2:
        raise ValidationError(
            "a spin-orbital register must contain a non-negative even number of qubits",
            n_qubits=n_qubits,
        )
    n_spatial = n_qubits // 2
    if not (0 <= n_alpha <= n_spatial and 0 <= n_beta <= n_spatial):
        return 0
    return comb(n_spatial, n_alpha) * comb(n_spatial, n_beta)


def sector_mask(
    configurations: Iterable[int] | NDArray[np.int64],
    n_qubits: int,
    n_alpha: int,
    n_beta: int,
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """The configurations as an array, plus a mask of which lie in the sector.

    Duplicates survive, because a sampled configuration's multiplicity is its weight and
    the caller needs the *sample-level* retention rate -- "9 of 4096 shots were usable" and
    "9 distinct configurations were usable" are very different diagnoses of the same run.

    Out-of-range indices are a caller bug rather than shot noise (they cannot come from
    measuring an ``n_qubits`` register at all), so they are rejected instead of filtered.
    """
    indices = np.asarray(list(configurations), dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return indices, np.zeros(0, dtype=bool)
    dimension = 2**n_qubits
    if int(indices.min()) < 0 or int(indices.max()) >= dimension:
        raise ValidationError(
            f"configuration index out of range for a {n_qubits}-qubit register",
            n_qubits=n_qubits,
            min_index=int(indices.min()),
            max_index=int(indices.max()),
            dimension=dimension,
        )
    alpha, beta = alpha_beta_occupation(indices, n_qubits)
    return indices, (alpha == n_alpha) & (beta == n_beta)


def filter_to_sector(
    configurations: Iterable[int] | NDArray[np.int64],
    n_qubits: int,
    n_alpha: int,
    n_beta: int,
) -> NDArray[np.int64]:
    """Drop configurations outside the ``(n_alpha, n_beta)`` sector; deduplicate the rest.

    Returned ascending, so the projected matrix depends on the *set* of surviving
    configurations and not on the order they happened to be sampled in -- two runs that
    sample the same configurations produce a bit-identical matrix.
    """
    indices, mask = sector_mask(configurations, n_qubits, n_alpha, n_beta)
    unique: NDArray[np.int64] = np.unique(indices[mask])
    return unique


# --------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------


def sample_configurations(
    state: NDArray[np.complex128], n_samples: int, seed: int
) -> NDArray[np.int64]:
    """Draw ``n_samples`` computational-basis outcomes from ``|state|^2``.

    Seeded through ``default_rng``, so the same ``(state, n_samples, seed)`` always yields
    the same draw. That is not a convenience: a subspace chosen by an unseeded sampler makes
    the reported energy irreproducible, and an irreproducible energy cannot be compared
    across runs, which is most of what this engine's results are for.
    """
    if n_samples < 1:
        raise ValidationError(f"n_samples must be >= 1, got {n_samples}", n_samples=n_samples)
    amplitudes = np.asarray(state, dtype=complex).reshape(-1)
    probabilities = np.abs(amplitudes) ** 2
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValidationError(
            "state has no probability mass to sample from; it is zero or non-finite",
            total_probability=total,
        )
    probabilities = probabilities / total
    rng = np.random.default_rng(seed)
    drawn = rng.choice(probabilities.size, size=n_samples, p=probabilities)
    return np.asarray(drawn, dtype=np.int64)


def _validate_subspace_dimension(value: int | None) -> None:
    """Reject a non-positive requested dimension before it reaches a slice.

    Worth its own check rather than trusting the caller: every consumer of this number
    eventually slices with it, and Python's negative-slice semantics turn ``-1`` into
    "drop the last-ranked configuration" -- a silently smaller subspace that still returns
    a plausible, still-variational energy for the wrong question. ``SolveRequest`` validates
    the same way, but these functions are the module's public API and are called directly,
    so the guard has to live here too.
    """
    if value is not None and value < 1:
        raise ValidationError(
            f"subspace_dimension must be >= 1 when given, got {value}",
            subspace_dimension=value,
        )


def rank_configurations_by_weight(
    samples: NDArray[np.int64], keep: int | None = None
) -> NDArray[np.int64]:
    """The most frequently sampled configurations, ascending, at most ``keep`` of them.

    Frequency is the ranking because a sampled configuration's count estimates its weight
    in the state, and the configurations carrying the most weight are the ones whose span
    contains most of the wavefunction. Ties are broken by index so the truncation is a total
    order -- otherwise two runs with identical samples could keep different subspaces.
    """
    _validate_subspace_dimension(keep)
    unique, counts = np.unique(np.asarray(samples, dtype=np.int64), return_counts=True)
    if keep is None or keep >= unique.size:
        return unique
    order = np.lexsort((unique, -counts))
    return np.sort(unique[order[:keep]])


def _configuration_weight_order(samples: NDArray[np.int64]) -> NDArray[np.int64]:
    """All distinct configurations in deterministic descending-frequency order."""
    unique, counts = np.unique(np.asarray(samples, dtype=np.int64), return_counts=True)
    order = np.lexsort((unique, -counts))
    return unique[order]


# --------------------------------------------------------------------------------------
# Projection and diagonalisation
# --------------------------------------------------------------------------------------


def project_hamiltonian(
    hamiltonian: QubitHamiltonian, configurations: NDArray[np.int64]
) -> NDArray[np.complex128]:
    """Dense ``<c_i|H|c_j>`` over the given configurations.

    Pauli strings map a computational-basis determinant to exactly one other determinant.
    Applying that map directly makes the cost depend on the selected configurations and
    Hamiltonian terms, with no length-``2^n`` statevector.
    """
    n_qubits = hamiltonian.n_qubits
    selected = np.asarray(configurations, dtype=np.int64)
    if selected.size != np.unique(selected).size:
        raise ValidationError("projected configurations must be unique")
    subspace_dimension = int(selected.size)
    block = np.zeros((subspace_dimension, subspace_dimension), dtype=complex)
    lookup = {int(index): row for row, index in enumerate(selected)}
    for column, index in enumerate(selected):
        for paulis, coefficient in hamiltonian.operator.terms.items():
            destination, phase = _pauli_image(int(index), paulis, n_qubits)
            row = lookup.get(destination)
            if row is not None:
                block[row, column] += complex(coefficient) * phase

    # H is Hermitian, so its projection onto any basis is too. An asymmetric block means
    # the operator or the index convention is not what this module thinks it is, and the
    # eigenvalue would still come back looking perfectly plausible.
    asymmetry = float(np.abs(block - block.conj().T).max()) if subspace_dimension else 0.0
    if asymmetry > _HERMITICITY_TOLERANCE:
        raise ValidationError(
            f"projected Hamiltonian is not Hermitian (max asymmetry {asymmetry:.3e}); "
            "the operator or the qubit-ordering convention is wrong",
            asymmetry=asymmetry,
            subspace_dimension=subspace_dimension,
        )
    return block


def _pauli_image(
    index: int,
    paulis: tuple[tuple[int, str], ...],
    n_qubits: int,
) -> tuple[int, complex]:
    """Return ``P|index> = phase |destination>`` in the engine's MSB qubit order."""
    destination = index
    phase = 1.0 + 0.0j
    for qubit, letter in paulis:
        bit_position = n_qubits - 1 - qubit
        bit = (destination >> bit_position) & 1
        if letter == "X":
            destination ^= 1 << bit_position
        elif letter == "Y":
            phase *= 1.0j if bit == 0 else -1.0j
            destination ^= 1 << bit_position
        elif letter == "Z":
            phase *= 1.0 if bit == 0 else -1.0
        else:
            raise ValidationError(f"unknown Pauli letter {letter!r}", letter=letter)
    return destination, phase


def _apply_in_selected_space(
    hamiltonian: QubitHamiltonian,
    configurations: NDArray[np.int64],
    vector: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Apply ``P H P`` directly to coefficients on selected determinants."""
    selected = np.asarray(configurations, dtype=np.int64)
    coefficients = np.asarray(vector, dtype=complex)
    lookup = {int(index): row for row, index in enumerate(selected)}
    result = np.zeros(selected.size, dtype=complex)
    for index, amplitude in zip(selected, coefficients, strict=True):
        if amplitude == 0.0:
            continue
        for paulis, coefficient in hamiltonian.operator.terms.items():
            destination, phase = _pauli_image(
                int(index), paulis, hamiltonian.n_qubits
            )
            row = lookup.get(destination)
            if row is not None:
                result[row] += complex(coefficient) * phase * amplitude
    return result


def _subspace_ground_state(
    hamiltonian: QubitHamiltonian, configurations: NDArray[np.int64]
) -> tuple[float, NDArray[np.complex128]]:
    """Lowest eigenpair of ``H`` restricted to ``configurations``.

    Dense below the ceiling (where ``eigsh`` cannot return ``k=1`` of ``dim`` anyway), and a
    scatter/apply/gather ``LinearOperator`` above it so a large subspace never forms a
    matrix. The eigenvector comes back because :func:`assert_variational_bound` needs it.
    """
    subspace_dimension = int(configurations.size)
    if subspace_dimension <= _DENSE_DIMENSION_CEILING:
        block = project_hamiltonian(hamiltonian, configurations)
        eigenvalues, eigenvectors = np.linalg.eigh(block)
        return float(eigenvalues[0]), np.asarray(eigenvectors[:, 0], dtype=complex)

    def matvec(vector: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return _apply_in_selected_space(hamiltonian, configurations, vector)

    linear_operator = LinearOperator(
        (subspace_dimension, subspace_dimension),
        matvec=matvec,
        rmatvec=matvec,
        dtype=complex,
    )
    # Same convergence settings as the matrix-free sector solve: a near-degenerate low-spin
    # ground state otherwise stops at a Ritz value a few 1e-4 Ha above the true minimum,
    # which is large enough to matter against chemical accuracy.
    eigenvalues, eigenvectors = eigsh(
        linear_operator,
        k=1,
        which="SA",
        ncv=min(subspace_dimension, 40),
        maxiter=subspace_dimension * 20,
        tol=0,
    )
    return float(eigenvalues[0]), np.asarray(eigenvectors[:, 0], dtype=complex)


def assert_variational_bound(
    hamiltonian: QubitHamiltonian,
    configurations: NDArray[np.int64],
    energy: float,
    vector: NDArray[np.complex128],
    *,
    reference_energy: float | None = None,
    tolerance: float = VARIATIONAL_TOLERANCE,
) -> float:
    """Check the returned energy really is a variational upper bound. Returns the quotient.

    The whole value of SQD over VQE is that its answer is a Rayleigh quotient of an actual
    state, so it cannot fall below the true ground state. Two things are checked:

    1. **It is a Rayleigh quotient.** ``<psi|PHP|psi> / <psi|psi>`` is recomputed by
       applying each Pauli term directly to the selected determinant coefficients. It must
       equal the subspace eigenvalue.
    2. **It does not undercut a known reference.** When the caller supplies one -- an FCI
       energy, or the exact sector minimum -- an "upper bound" below it is impossible, so it
       is proof of a bug rather than a good result, and is refused rather than reported.

    A failure is a ``ValidationError``, not a bare ``assert``: this must survive ``python -O``
    (the cost is one extra ``H @ psi``), and it must carry the numbers that show how far off
    it was.
    """
    coefficients = np.asarray(vector, dtype=complex)
    norm_squared = float(np.vdot(coefficients, coefficients).real)
    if norm_squared <= 0.0:
        raise ValidationError(
            "subspace eigenvector has zero norm; the projection produced no state",
            subspace_dimension=int(configurations.size),
        )

    applied = _apply_in_selected_space(hamiltonian, configurations, coefficients)
    quotient = complex(np.vdot(coefficients, applied)) / norm_squared
    if abs(quotient.imag) > tolerance:
        raise ValidationError(
            f"Rayleigh quotient has a significant imaginary part ({quotient.imag:.3e}); "
            "the Hamiltonian is not Hermitian",
            imaginary_part=quotient.imag,
        )
    rayleigh = float(quotient.real)

    if abs(rayleigh - energy) > tolerance:
        raise ValidationError(
            f"subspace eigenvalue {energy:.12f} is not the Rayleigh quotient of its own "
            f"eigenvector ({rayleigh:.12f}); the projection does not represent the state "
            "it claims to",
            subspace_energy=energy,
            rayleigh_quotient=rayleigh,
            difference=rayleigh - energy,
        )

    if reference_energy is not None and rayleigh < reference_energy - tolerance:
        raise ValidationError(
            f"SQD energy {rayleigh:.12f} lies below the reference {reference_energy:.12f}; "
            "a variational upper bound cannot undercut the true minimum, so the subspace "
            "or the Hamiltonian is wrong",
            sqd_energy=rayleigh,
            reference_energy=reference_energy,
            violation=reference_energy - rayleigh,
        )
    return rayleigh


# --------------------------------------------------------------------------------------
# The solver
# --------------------------------------------------------------------------------------


def solve_sqd(
    hamiltonian: QubitHamiltonian,
    n_alpha: int,
    n_beta: int,
    *,
    configurations: Iterable[int] | NDArray[np.int64] | None = None,
    subspace_dimension: int | None = None,
    reference_energy: float | None = None,
) -> tuple[SQDOutcome, NDArray[np.complex128]]:
    """Diagonalise ``hamiltonian`` in the span of ``configurations``.

    ``configurations`` is the **raw** draw: duplicates and out-of-sector bitstrings included.
    All of the filtering, weighting and truncation happens here, in one place, so the
    sample-level discard counts in the returned :class:`SQDOutcome` describe what actually
    arrived rather than what a caller had already cleaned up. Passing ``None`` means the
    whole ``(n_alpha, n_beta)`` sector, which makes the result exact rather than a bound.

    ``subspace_dimension`` keeps only that many of the most frequently sampled
    configurations. A list of *distinct* configurations carries no weight information, so it
    truncates in index order; a draw with repeats truncates by multiplicity, which is the
    real SQD selection rule.

    Returns the outcome alongside the subspace eigenvector, which callers occasionally want
    (it is the SQD wavefunction) and which the bound check has already validated.
    """
    _validate_subspace_dimension(subspace_dimension)
    n_qubits = hamiltonian.n_qubits
    full_sector_dimension = sector_dimension(n_qubits, n_alpha, n_beta)
    if full_sector_dimension == 0:
        raise BackendUnavailable(
            SOLVER_NAME,
            f"no configuration in {n_qubits} spin orbitals holds {n_alpha} alpha and "
            f"{n_beta} beta electrons; the active space cannot express this sector",
            n_alpha=n_alpha,
            n_beta=n_beta,
            n_qubits=n_qubits,
        )

    if configurations is None:
        sector = sector_configurations(n_qubits, n_alpha, n_beta)
        kept = sector if subspace_dimension is None else sector[:subspace_dimension]
        n_samples = 0
        n_in_sector = 0
    else:
        drawn, mask = sector_mask(configurations, n_qubits, n_alpha, n_beta)
        n_samples = int(drawn.size)
        in_sector = drawn[mask]
        n_in_sector = int(in_sector.size)
        if n_in_sector == 0:
            raise BackendUnavailable(
                SOLVER_NAME,
                f"all {n_samples} sampled configurations fell outside the "
                f"({n_alpha}, {n_beta}) sector, leaving nothing to diagonalise; the state "
                "being sampled does not carry the requested electron count",
                n_alpha=n_alpha,
                n_beta=n_beta,
                n_samples=n_samples,
            )
        kept = rank_configurations_by_weight(in_sector, keep=subspace_dimension)

    energy, vector = _subspace_ground_state(hamiltonian, kept)
    energy = assert_variational_bound(
        hamiltonian, kept, energy, vector, reference_energy=reference_energy
    )

    outcome = SQDOutcome(
        energy=energy,
        subspace_dimension=int(kept.size),
        sector_dimension=full_sector_dimension,
        n_alpha=n_alpha,
        n_beta=n_beta,
        n_samples=n_samples,
        n_samples_in_sector=n_in_sector,
        is_exact=bool(kept.size == full_sector_dimension),
    )
    return outcome, vector


def run_sqd_campaign(
    hamiltonian: QubitHamiltonian,
    n_alpha: int,
    n_beta: int,
    *,
    configurations: Iterable[int] | NDArray[np.int64],
    dimensions: Sequence[int],
    reference_energy: float | None = None,
) -> SQDCampaign:
    """Run SQD over one ranking and a strictly nested dimension ladder.

    Ranking the raw draw once is load-bearing.  Ranking each requested size separately can
    produce different tie outcomes and therefore non-nested spaces; then a decreasing energy
    would not be convergence evidence for one systematically enlarged approximation.
    """
    requested = tuple(int(dimension) for dimension in dimensions)
    if (
        not requested
        or any(dimension < 1 for dimension in requested)
        or any(later <= earlier for earlier, later in zip(requested, requested[1:], strict=False))
    ):
        raise ValidationError(
            "dimensions must be a non-empty, strictly increasing sequence of positive integers",
            dimensions=list(requested),
        )

    drawn, mask = sector_mask(configurations, hamiltonian.n_qubits, n_alpha, n_beta)
    in_sector = drawn[mask]
    if in_sector.size == 0:
        raise ValidationError(
            "dimensions cannot be evaluated because the raw draw has no in-sector samples",
            dimensions=list(requested),
            n_samples=int(drawn.size),
        )
    ranked = _configuration_weight_order(in_sector)
    if requested[-1] > int(ranked.size):
        raise ValidationError(
            f"dimensions request {requested[-1]} configurations but only {ranked.size} "
            "distinct in-sector configurations were sampled",
            dimensions=list(requested),
            distinct_in_sector=int(ranked.size),
        )

    steps: list[SQDStep] = []
    previous_energy: float | None = None
    full_sector_dimension = sector_dimension(hamiltonian.n_qubits, n_alpha, n_beta)
    for dimension in requested:
        nested = np.sort(ranked[:dimension])
        outcome, _ = solve_sqd(
            hamiltonian,
            n_alpha,
            n_beta,
            configurations=nested,
            reference_energy=reference_energy,
        )
        change = (
            None if previous_energy is None else abs(outcome.energy - previous_energy)
        )
        if previous_energy is not None and outcome.energy > previous_energy + VARIATIONAL_TOLERANCE:
            raise ValidationError(
                "SQD energy increased when the determinant space was enlarged; "
                "the campaign is not nested or the eigensolver failed",
                previous_energy=previous_energy,
                energy=outcome.energy,
                dimension=dimension,
            )
        steps.append(
            SQDStep(
                requested_dimension=dimension,
                subspace_dimension=outcome.subspace_dimension,
                energy=outcome.energy,
                energy_change=change,
                is_exact=outcome.is_exact,
                configurations=tuple(int(value) for value in nested),
            )
        )
        previous_energy = outcome.energy

    return SQDCampaign(
        steps=tuple(steps),
        sector_dimension=full_sector_dimension,
        n_samples=int(drawn.size),
        n_samples_in_sector=int(in_sector.size),
        reference_energy=reference_energy,
    )


class SQDSolver:
    """Sample-based quantum diagonalisation over a :class:`SolveRequest`.

    Not a registered ``SolverBackend`` -- see the module docstring. It mirrors the interface
    closely enough (``name``, ``solve``, ``check_supported``) that wrapping it in a driver
    later is a five-line class, and every refusal already carries the right name and error
    code.
    """

    name = SOLVER_NAME

    def check_supported(self, request: SolveRequest) -> None:
        """Raise ``BackendUnavailable`` if this solver cannot serve ``request``.

        Every branch names the solver and states the reason, because the alternative --
        proceeding on a request this solver cannot honestly serve -- returns a number that
        answers a different question than the one asked.
        """
        n_qubits = request.n_qubits
        if n_qubits > MAX_QUBITS:
            raise BackendUnavailable(
                self.name,
                f"request needs {n_qubits} qubits, above the platform cap of {MAX_QUBITS}",
                required_qubits=n_qubits,
                max_qubits=MAX_QUBITS,
            )
        if request.ansatz is not None and request.ansatz.n_qubits != n_qubits:
            raise BackendUnavailable(
                self.name,
                f"ansatz spans {request.ansatz.n_qubits} qubits but the Hamiltonian spans "
                f"{n_qubits}",
                ansatz_qubits=request.ansatz.n_qubits,
                hamiltonian_qubits=n_qubits,
            )

        explicit = request.options.get("configurations")
        truncated = request.subspace_dimension is not None
        if n_qubits > MAX_EXACT_SQD_QUBITS and explicit is None:
            raise BackendUnavailable(
                self.name,
                f"a {n_qubits}-qubit SQD request needs explicit sampled configurations; "
                "exact sector enumeration and the built-in ansatz simulator stop at "
                f"{MAX_EXACT_SQD_QUBITS} qubits",
                required_qubits=n_qubits,
                exact_mode_max_qubits=MAX_EXACT_SQD_QUBITS,
            )
        if truncated and explicit is None and request.ansatz is None:
            raise BackendUnavailable(
                self.name,
                f"a subspace of {request.subspace_dimension} configurations was requested "
                "but there is nothing to choose it from: supply an ansatz to sample or an "
                "explicit 'configurations' option, or leave subspace_dimension unset to "
                "take the whole sector",
                subspace_dimension=request.subspace_dimension,
            )
        if request.shots is not None and explicit is None and request.ansatz is None:
            raise BackendUnavailable(
                self.name,
                f"a shot budget of {request.shots} was requested but no ansatz was supplied "
                "to sample from",
                shots=request.shots,
            )

    def supports(self, request: SolveRequest) -> bool:
        """Non-raising form of :meth:`check_supported`, for capability queries."""
        try:
            self.check_supported(request)
        except BackendUnavailable:
            return False
        return True

    def solve(self, request: SolveRequest) -> SolveResult:
        """Compute the ground-state energy by subspace diagonalisation.

        The configuration set comes from the first available of: an explicit
        ``options["configurations"]`` list, sampling the ansatz, or -- when no truncation was
        asked for -- the entire sector. Sampling only happens when a truncation is actually
        requested: with ``subspace_dimension=None`` the sector is taken whole and the answer
        is exact, and drawing samples in order to then keep all of them would be pure cost.
        """
        self.check_supported(request)

        spin_multiplicity = int(request.options.get("spin_multiplicity", 1))
        n_alpha, n_beta = alpha_beta_from_spin(
            request.active_space.n_active_electrons, spin_multiplicity
        )
        reference_energy = request.options.get("reference_energy")

        outcome, _vector = solve_sqd(
            request.hamiltonian,
            n_alpha,
            n_beta,
            configurations=self._draw_configurations(request),
            subspace_dimension=request.subspace_dimension,
            reference_energy=(None if reference_energy is None else float(reference_energy)),
        )

        return SolveResult(
            energy=outcome.energy,
            backend=self.name,
            method="sqd-exact" if outcome.is_exact else "sqd",
            converged=True,
            n_iterations=0,
            n_energy_evaluations=1,
            optimal_params=None,
            std_error=None,
            metadata={
                # The dimension actually used, not the one requested: a sector smaller than
                # the request, or samples that collapsed onto few distinct configurations,
                # both silently shrink it, and a reader comparing two runs needs the real one.
                "subspace_dimension": outcome.subspace_dimension,
                "requested_subspace_dimension": request.subspace_dimension,
                "sector_dimension": outcome.sector_dimension,
                "n_alpha": outcome.n_alpha,
                "n_beta": outcome.n_beta,
                "spin_multiplicity": spin_multiplicity,
                "n_samples": outcome.n_samples,
                "n_samples_in_sector": outcome.n_samples_in_sector,
                "n_samples_discarded": outcome.n_samples_discarded,
                "sector_retention": outcome.sector_retention,
                "seed": request.seed,
                # False means the energy is an upper bound, not the sector minimum. Every
                # downstream comparison against FCI hinges on this flag.
                "is_exact_in_sector": outcome.is_exact,
                "is_variational_bound": True,
            },
        )

    def _draw_configurations(self, request: SolveRequest) -> NDArray[np.int64] | None:
        """The raw configuration draw, or ``None`` for "take the whole sector".

        Filtering and truncation are deliberately *not* done here -- :func:`solve_sqd` owns
        both, so the discard counts it reports are measured against the untouched draw.
        """
        explicit: Any = request.options.get("configurations")
        if explicit is not None:
            return np.asarray(list(explicit), dtype=np.int64).reshape(-1)

        if request.subspace_dimension is None:
            return None

        assert request.ansatz is not None  # guaranteed by check_supported

        # Imported here, not at module scope, so that `solvers.sqd` stays importable
        # without pulling in the driver registry -- the pure functions above are usable
        # from anywhere, and only this sampling path costs you the driver layer.
        #
        # Be clear about what this does *not* buy: deferring an import changes when Python
        # resolves the name, not the dependency graph. `solvers.sqd` genuinely depends on
        # `drivers.statevector`, which is the reverse of the `drivers -> solvers` direction
        # the rest of the engine follows (see `drivers/classical/classical_driver.py`).
        # It is the only such edge in the package. It is tolerated because sampling a
        # circuit is the one thing SQD cannot do without a simulator, and inverting it
        # properly means either moving `SQDSolver` into `drivers/` or passing a sampler
        # callable in on the request.
        from qdr_compute_engine.drivers.statevector.statevector_driver import (
            simulate_statevector,
        )

        n_samples = request.shots if request.shots is not None else DEFAULT_SAMPLE_COUNT
        state = simulate_statevector(request.ansatz, request.initial_params)
        return sample_configurations(state, n_samples, request.seed)

"""Sample-based quantum diagonalisation.

Three claims, in order of how much they matter:

1. **Taking the whole sector reproduces FCI.** SQD's subspace is a truncation of the
   sector; remove the truncation and there is nothing left to approximate, so the answer
   must be the exact one. If this fails, every truncated result is measured against a
   yardstick that is itself wrong.
2. **A truncated subspace returns a variational upper bound.** Never below the exact
   sector minimum, and never rising when the subspace grows. This is the property that
   makes SQD worth having over VQE, so it is tested rather than assumed.
3. **The same seed gives the same subspace.** An energy from an unseeded sampler cannot be
   compared against anything, including itself.

None of this needs PySCF. The H2 Hamiltonian is rebuilt from the golden integrals through
the pure mapper path, and the reference is the engine's own matrix-free sector solver plus
the golden file's ``e_fci`` -- so the whole file runs on Windows, where the PySCF-dependent
tests are skipped and would otherwise leave SQD entirely unexercised during development.
"""

from itertools import pairwise
from typing import Any

import numpy as np
import pytest
from openfermion import QubitOperator

from qdr_compute_engine.active_space.active_space_selector import select_active_space
from qdr_compute_engine.ansatze.uccsd import build_uccsd_ansatz
from qdr_compute_engine.hamiltonian.fermionic_builder import build_fermionic_hamiltonian
from qdr_compute_engine.hamiltonian.matrix_free import (
    ground_state_energy_at_spin_sector_matrix_free,
)
from qdr_compute_engine.mappers.jordan_wigner import map_to_qubits
from qdr_compute_engine.solvers.sqd import (
    SOLVER_NAME,
    SQDSolver,
    alpha_beta_occupation,
    assert_variational_bound,
    filter_to_sector,
    rank_configurations_by_weight,
    run_sqd_campaign,
    sample_configurations,
    sector_configurations,
    solve_sqd,
)
from qdr_compute_engine.types import (
    ActiveSpace,
    IntegralSet,
    QubitHamiltonian,
    SolveRequest,
)
from qdr_core.errors import BackendUnavailable, ValidationError
from qdr_core.limits import MAX_DENSE_STATEVECTOR_QUBITS, MAX_QUBITS

#: H2/STO-3G in the full space: 2 electrons, 2 spatial orbitals, 4 qubits, one alpha and
#: one beta electron. The (1, 1) sector holds four configurations.
H2_N_ALPHA = 1
H2_N_BETA = 1

#: The Hartree-Fock determinant's flat index. Spin orbitals 0 (alpha) and 1 (beta) are
#: occupied, i.e. qubits 0 and 1, which sit at bits 3 and 2: 8 + 4 = 12.
HF_CONFIGURATION = 12

#: UCCSD amplitudes chosen so the prepared state spreads across all four (1, 1)
#: determinants (roughly 0.33 / 0.18 / 0.18 / 0.32) instead of sitting on one of them. Any
#: test that samples the ansatz needs this: at the default zero amplitudes UCCSD is the
#: identity on the reference, and sampling a point mass cannot distinguish a working
#: sampler from a broken one.
SPREAD_PARAMS = np.full(3, 0.5)


@pytest.fixture
def h2_hamiltonian(h2_integrals: IntegralSet) -> QubitHamiltonian:
    active_space = select_active_space(h2_integrals)
    return map_to_qubits(build_fermionic_hamiltonian(h2_integrals, active_space))


@pytest.fixture
def h2_active_space(h2_integrals: IntegralSet) -> ActiveSpace:
    return select_active_space(h2_integrals)


@pytest.fixture
def h2_exact(h2_hamiltonian: QubitHamiltonian) -> float:
    """The exact (1, 1)-sector minimum, from the engine's independent matrix-free path."""
    return ground_state_energy_at_spin_sector_matrix_free(h2_hamiltonian, H2_N_ALPHA, H2_N_BETA)


def _request(
    hamiltonian: QubitHamiltonian, active_space: ActiveSpace, **overrides: Any
) -> SolveRequest:
    return SolveRequest(hamiltonian=hamiltonian, active_space=active_space, **overrides)


# --------------------------------------------------------------------------------------
# THE GATE: the untruncated subspace is not an approximation at all
# --------------------------------------------------------------------------------------


def test_full_subspace_equals_fci(
    h2_hamiltonian: QubitHamiltonian, golden_h2: dict[str, Any]
) -> None:
    """THE SQD GATE: diagonalising the whole sector reproduces the golden FCI energy."""
    outcome, _vector = solve_sqd(h2_hamiltonian, H2_N_ALPHA, H2_N_BETA)

    assert outcome.energy == pytest.approx(float(golden_h2["e_fci"]), abs=1e-10)


def test_full_subspace_equals_the_matrix_free_sector_solver(
    h2_hamiltonian: QubitHamiltonian, h2_exact: float
) -> None:
    """Same claim against a second, independent reference path inside the engine."""
    outcome, _vector = solve_sqd(h2_hamiltonian, H2_N_ALPHA, H2_N_BETA)

    assert outcome.energy == pytest.approx(h2_exact, abs=1e-10)


def test_full_subspace_is_reported_as_exact(h2_hamiltonian: QubitHamiltonian) -> None:
    """The flag downstream comparisons switch on must say "exact", not "bound"."""
    outcome, _vector = solve_sqd(h2_hamiltonian, H2_N_ALPHA, H2_N_BETA)

    assert outcome.is_exact
    assert outcome.subspace_dimension == outcome.sector_dimension == 4


def test_explicitly_listing_every_configuration_also_equals_fci(
    h2_hamiltonian: QubitHamiltonian, h2_exact: float
) -> None:
    """Handing in the sector by hand must take the same path as asking for it implicitly."""
    every = sector_configurations(4, H2_N_ALPHA, H2_N_BETA)

    outcome, _vector = solve_sqd(h2_hamiltonian, H2_N_ALPHA, H2_N_BETA, configurations=every)

    assert outcome.energy == pytest.approx(h2_exact, abs=1e-10)
    assert outcome.is_exact


# --------------------------------------------------------------------------------------
# Sector filtering: the qubit-ordering convention, and dropping what does not belong
# --------------------------------------------------------------------------------------


def test_sector_configurations_have_the_requested_occupation() -> None:
    sector = sector_configurations(6, 2, 1)
    alpha, beta = alpha_beta_occupation(sector, 6)

    assert sector.size > 0
    assert np.all(alpha == 2)
    assert np.all(beta == 1)


def test_h2_sector_is_the_four_expected_determinants() -> None:
    """The H2 sector, hardcoded rather than recomputed.

    Qubit ``q`` sits at bit ``n - 1 - q``, alpha on even qubits and beta on odd, so the
    (1, 1) determinants of a 4-qubit register are 3, 6, 9 and 12.

    This pins the H2 numbers the rest of the file relies on, but note what it *cannot* do:
    H2 has ``n_alpha == n_beta``, and ``{3, 6, 9, 12}`` maps onto itself under bit reversal,
    so swapping alpha with beta or flipping the bit order both leave this list unchanged.
    ``test_asymmetric_sector_pins_the_convention`` is the test that actually pins it.
    """
    assert sector_configurations(4, 1, 1).tolist() == [3, 6, 9, 12]
    assert HF_CONFIGURATION in sector_configurations(4, 1, 1).tolist()


def test_asymmetric_sector_pins_the_convention() -> None:
    """The convention check with the symmetry removed, so the two real bugs cannot hide.

    A sector with ``n_alpha != n_beta`` is not invariant under swapping the spin labels, and
    on 6 qubits the (2, 1) indices are not invariant under reversing the bit order either.
    Both mutations send this list to ``[7, 13, 19, 22, 25, 28, 37, 49, 52]``, so either one
    fails here. That matters beyond bookkeeping: ``matrix_free.py`` derives alpha/beta counts
    with the identical arithmetic, so a swap would agree with itself everywhere and only
    surface as a wrong open-shell energy.
    """
    assert sector_configurations(6, 2, 1).tolist() == [11, 14, 26, 35, 38, 41, 44, 50, 56]

    # The same claim from the other side: every index above really does carry two alpha and
    # one beta electron under the hand-computed convention, without calling the production
    # helper that would cancel its own bug out.
    for index in sector_configurations(6, 2, 1).tolist():
        bits = [(index >> (6 - 1 - qubit)) & 1 for qubit in range(6)]
        assert sum(bits[0::2]) == 2
        assert sum(bits[1::2]) == 1


def test_sector_sizes_are_the_binomial_product() -> None:
    """``C(n_orb, n_alpha) * C(n_orb, n_beta)`` -- the determinant count SQD truncates."""
    from math import comb

    for n_orbitals, n_alpha, n_beta in [(2, 1, 1), (3, 2, 1), (4, 2, 2), (4, 3, 1)]:
        expected = comb(n_orbitals, n_alpha) * comb(n_orbitals, n_beta)
        assert sector_configurations(2 * n_orbitals, n_alpha, n_beta).size == expected


def test_filter_drops_configurations_from_the_wrong_sector() -> None:
    """A wrong-sector bitstring is not noise: it answers for a different molecule."""
    # 15 = every orbital occupied (2, 2); 0 = vacuum (0, 0); 12 and 3 are genuine (1, 1).
    kept = filter_to_sector([15, 12, 0, 3], n_qubits=4, n_alpha=1, n_beta=1)

    assert kept.tolist() == [3, 12]


def test_filter_deduplicates_and_sorts() -> None:
    """The subspace is a set, so a repeated sample must not widen or reorder the basis."""
    kept = filter_to_sector([12, 3, 12, 9, 3, 12], n_qubits=4, n_alpha=1, n_beta=1)

    assert kept.tolist() == [3, 9, 12]


def test_filter_rejects_an_out_of_range_index() -> None:
    """Out of range cannot come from measuring the register -- it is a caller bug."""
    with pytest.raises(ValidationError, match="out of range"):
        filter_to_sector([16], n_qubits=4, n_alpha=1, n_beta=1)


def test_sample_level_discards_are_reported(h2_hamiltonian: QubitHamiltonian) -> None:
    """A high discard rate is the signal the sampled state is not the intended one.

    Reported per *sample*, not per distinct configuration: "6 of 8 shots were unusable" and
    "2 distinct configurations survived" are different facts and only the first one says
    whether the state being sampled conserves particle number.
    """
    draw = [15, 15, 0, 0, 12, 3, 5, 10]  # only 12 and 3 lie in the (1, 1) sector

    outcome, _vector = solve_sqd(h2_hamiltonian, H2_N_ALPHA, H2_N_BETA, configurations=draw)

    assert outcome.n_samples == 8
    assert outcome.n_samples_in_sector == 2
    assert outcome.n_samples_discarded == 6
    assert outcome.sector_retention == pytest.approx(0.25)
    assert outcome.subspace_dimension == 2
    assert not outcome.is_exact


def test_refuses_when_every_sample_leaves_the_sector(
    h2_hamiltonian: QubitHamiltonian,
) -> None:
    """Nothing to diagonalise is a refusal, never a fallback to some other sector."""
    with pytest.raises(BackendUnavailable) as excinfo:
        solve_sqd(h2_hamiltonian, H2_N_ALPHA, H2_N_BETA, configurations=[0, 15, 15])

    assert excinfo.value.backend == SOLVER_NAME
    assert "outside" in excinfo.value.reason


def test_refuses_a_sector_the_active_space_cannot_express(
    h2_hamiltonian: QubitHamiltonian,
) -> None:
    """Two spatial orbitals cannot hold three alpha electrons; say so rather than guess."""
    with pytest.raises(BackendUnavailable) as excinfo:
        solve_sqd(h2_hamiltonian, 3, 0)

    assert excinfo.value.backend == SOLVER_NAME
    assert excinfo.value.details["n_alpha"] == 3


# --------------------------------------------------------------------------------------
# The variational bound -- what SQD has that VQE does not
# --------------------------------------------------------------------------------------


def test_single_configuration_subspace_is_the_hartree_fock_energy(
    h2_hamiltonian: QubitHamiltonian, golden_h2: dict[str, Any]
) -> None:
    """The smallest possible subspace has a known answer, which pins the projection.

    One determinant means the 1x1 projected matrix is exactly ``<HF|H|HF>``, so this
    reproducing the golden ``e_hf`` shows the projection is the operator it claims to be.
    """
    outcome, _vector = solve_sqd(
        h2_hamiltonian, H2_N_ALPHA, H2_N_BETA, configurations=[HF_CONFIGURATION]
    )

    assert outcome.energy == pytest.approx(float(golden_h2["e_hf"]), abs=1e-10)
    assert not outcome.is_exact


def test_truncated_subspace_never_falls_below_the_exact_energy(
    h2_hamiltonian: QubitHamiltonian, h2_exact: float
) -> None:
    """Every proper subspace of the sector: an upper bound, never an undercut."""
    sector = sector_configurations(4, H2_N_ALPHA, H2_N_BETA).tolist()

    for keep in range(1, len(sector) + 1):
        outcome, _vector = solve_sqd(
            h2_hamiltonian, H2_N_ALPHA, H2_N_BETA, configurations=sector[:keep]
        )
        assert outcome.energy >= h2_exact - 1e-12


def test_growing_the_subspace_can_only_lower_the_energy(
    h2_hamiltonian: QubitHamiltonian, h2_exact: float
) -> None:
    """Monotonicity: adding a configuration widens the minimisation, never narrows it."""
    sector = sector_configurations(4, H2_N_ALPHA, H2_N_BETA).tolist()

    energies = [
        solve_sqd(h2_hamiltonian, H2_N_ALPHA, H2_N_BETA, configurations=sector[:keep])[0].energy
        for keep in range(1, len(sector) + 1)
    ]

    for smaller, larger in pairwise(energies):
        assert larger <= smaller + 1e-12
    assert energies[-1] == pytest.approx(h2_exact, abs=1e-10)


def test_bound_check_rejects_an_energy_below_a_known_reference(
    h2_hamiltonian: QubitHamiltonian, h2_exact: float
) -> None:
    """An "upper bound" under the true minimum is proof of a bug, so it must not return."""
    sector = sector_configurations(4, H2_N_ALPHA, H2_N_BETA)

    with pytest.raises(ValidationError, match="lies below the reference"):
        solve_sqd(
            h2_hamiltonian,
            H2_N_ALPHA,
            H2_N_BETA,
            configurations=sector,
            reference_energy=h2_exact + 1.0,
        )


def test_bound_check_rejects_an_eigenvalue_that_is_not_its_own_rayleigh_quotient(
    h2_hamiltonian: QubitHamiltonian,
) -> None:
    """The check that catches a wrong index map: a plausible number that is not the state's."""
    sector = sector_configurations(4, H2_N_ALPHA, H2_N_BETA)
    _outcome, vector = solve_sqd(h2_hamiltonian, H2_N_ALPHA, H2_N_BETA)

    with pytest.raises(ValidationError, match="not the Rayleigh quotient"):
        assert_variational_bound(h2_hamiltonian, sector, -99.0, vector)


def test_bound_check_accepts_a_reference_it_sits_above(
    h2_hamiltonian: QubitHamiltonian, h2_exact: float
) -> None:
    sector = sector_configurations(4, H2_N_ALPHA, H2_N_BETA)
    _outcome, vector = solve_sqd(h2_hamiltonian, H2_N_ALPHA, H2_N_BETA)

    quotient = assert_variational_bound(
        h2_hamiltonian, sector, h2_exact, vector, reference_energy=h2_exact - 1e-9
    )

    assert quotient == pytest.approx(h2_exact, abs=1e-10)


def test_projection_rejects_a_non_hermitian_operator() -> None:
    """A non-Hermitian operator still yields a plausible eigenvalue; refuse it instead.

    ``X0 Y2`` couples two configurations *inside* the (1, 1) sector, so the imaginary
    coefficient lands in the off-diagonal block rather than being projected away -- which is
    where a dropped conjugate or a sign-convention slip in a mapper actually shows up.
    """
    from qdr_compute_engine.solvers.sqd import project_hamiltonian

    hamiltonian = QubitHamiltonian(
        operator=QubitOperator("X0 Y2", 1j), n_qubits=4, mapping="jordan_wigner"
    )
    with pytest.raises(ValidationError, match="not Hermitian"):
        project_hamiltonian(hamiltonian, sector_configurations(4, H2_N_ALPHA, H2_N_BETA))


# --------------------------------------------------------------------------------------
# Sampling and seed determinism
# --------------------------------------------------------------------------------------


def _uniform_state(n_qubits: int) -> np.ndarray:
    dimension = 2**n_qubits
    return np.full(dimension, 1.0 / np.sqrt(dimension), dtype=complex)


def test_the_same_seed_draws_the_same_samples() -> None:
    state = _uniform_state(4)

    first = sample_configurations(state, 64, seed=7)
    second = sample_configurations(state, 64, seed=7)

    assert first.tolist() == second.tolist()


def test_a_different_seed_draws_different_samples() -> None:
    """Otherwise the seed is decorative and the determinism test above proves nothing."""
    state = _uniform_state(4)

    assert (
        sample_configurations(state, 64, seed=7).tolist()
        != sample_configurations(state, 64, seed=8).tolist()
    )


def test_sampling_follows_the_amplitudes() -> None:
    """A state with all its weight on one configuration can only ever yield that one."""
    state = np.zeros(16, dtype=complex)
    state[HF_CONFIGURATION] = 1.0

    samples = sample_configurations(state, 32, seed=0)

    assert set(samples.tolist()) == {HF_CONFIGURATION}


def test_sampling_normalises_an_unnormalised_state() -> None:
    state = np.zeros(16, dtype=complex)
    state[3] = 2.0
    state[12] = 2.0

    samples = sample_configurations(state, 64, seed=1)

    assert set(samples.tolist()) == {3, 12}


def test_sampling_refuses_a_state_with_no_probability_mass() -> None:
    with pytest.raises(ValidationError, match="no probability mass"):
        sample_configurations(np.zeros(16, dtype=complex), 8, seed=0)


def test_sampling_refuses_a_non_positive_sample_count() -> None:
    with pytest.raises(ValidationError, match="n_samples"):
        sample_configurations(_uniform_state(2), 0, seed=0)


def test_ranking_keeps_the_most_frequently_sampled() -> None:
    """Frequency is the SQD selection rule: a configuration's count estimates its weight."""
    samples = np.array([3, 3, 3, 12, 12, 9, 6, 6, 6, 6], dtype=np.int64)

    assert rank_configurations_by_weight(samples, keep=2).tolist() == [3, 6]


def test_ranking_breaks_ties_by_index_so_truncation_is_reproducible() -> None:
    samples = np.array([12, 9, 6, 3], dtype=np.int64)  # all weight 1

    assert rank_configurations_by_weight(samples, keep=2).tolist() == [3, 6]


def test_ranking_returns_everything_when_keep_exceeds_the_draw() -> None:
    samples = np.array([12, 12, 3], dtype=np.int64)

    assert rank_configurations_by_weight(samples, keep=99).tolist() == [3, 12]
    assert rank_configurations_by_weight(samples).tolist() == [3, 12]


# --------------------------------------------------------------------------------------
# SQDSolver: SolveRequest in, SolveResult out
# --------------------------------------------------------------------------------------


def test_solver_untruncated_request_equals_fci(
    h2_hamiltonian: QubitHamiltonian,
    h2_active_space: ActiveSpace,
    golden_h2: dict[str, Any],
) -> None:
    result = SQDSolver().solve(_request(h2_hamiltonian, h2_active_space))

    assert result.energy == pytest.approx(float(golden_h2["e_fci"]), abs=1e-10)
    assert result.backend == SOLVER_NAME
    assert result.method == "sqd-exact"
    assert result.converged
    assert result.metadata["is_exact_in_sector"] is True
    assert result.metadata["is_variational_bound"] is True


def test_solver_reports_the_dimension_it_actually_used(
    h2_hamiltonian: QubitHamiltonian, h2_active_space: ActiveSpace
) -> None:
    """Requested and actual differ whenever samples collapse; both are recorded."""
    request = _request(
        h2_hamiltonian,
        h2_active_space,
        subspace_dimension=99,
        options={"configurations": [HF_CONFIGURATION, 3, 15]},
    )

    result = SQDSolver().solve(request)

    assert result.metadata["subspace_dimension"] == 2  # 15 is out of sector
    assert result.metadata["requested_subspace_dimension"] == 99
    assert result.metadata["sector_dimension"] == 4
    assert result.metadata["n_samples_discarded"] == 1
    assert result.method == "sqd"


def test_solver_records_the_spin_sector(
    h2_hamiltonian: QubitHamiltonian, h2_active_space: ActiveSpace
) -> None:
    result = SQDSolver().solve(_request(h2_hamiltonian, h2_active_space))

    assert result.metadata["n_alpha"] == 1
    assert result.metadata["n_beta"] == 1
    assert result.metadata["spin_multiplicity"] == 1


def test_solver_sampling_is_deterministic_for_a_seed(
    h2_hamiltonian: QubitHamiltonian, h2_active_space: ActiveSpace
) -> None:
    """T16: two runs of the same seeded request agree bit for bit, not just closely.

    ``initial_params`` is set deliberately. Left at ``None`` the driver defaults every
    amplitude to zero, UCCSD reduces to the identity on the reference, and the sampled
    distribution collapses to a point mass on the Hartree-Fock determinant -- at which
    point ``rng.choice`` returns the same index whatever the seed is, and this test passes
    even if the solver ignores ``seed`` entirely. :data:`SPREAD_PARAMS` puts real weight on
    all four determinants, so the draw is genuinely random and the seed is load-bearing.
    """
    ansatz = build_uccsd_ansatz(n_electrons=2, n_spin_orbitals=4)
    request = _request(
        h2_hamiltonian,
        h2_active_space,
        ansatz=ansatz,
        initial_params=SPREAD_PARAMS,
        subspace_dimension=2,
        shots=256,
        seed=1234,
    )

    first = SQDSolver().solve(request)
    second = SQDSolver().solve(request)

    # The guard on the paragraph above: a collapsed distribution can only ever yield a
    # one-configuration subspace, so this failing means the test stopped testing sampling.
    assert first.metadata["subspace_dimension"] == 2

    assert first.energy == second.energy
    assert first.metadata["subspace_dimension"] == second.metadata["subspace_dimension"]
    assert first.metadata["seed"] == 1234


def test_solver_sampled_energy_is_a_variational_bound(
    h2_hamiltonian: QubitHamiltonian, h2_active_space: ActiveSpace, h2_exact: float
) -> None:
    """The sampled path obeys the same bound as the hand-listed one."""
    ansatz = build_uccsd_ansatz(n_electrons=2, n_spin_orbitals=4)
    request = _request(
        h2_hamiltonian,
        h2_active_space,
        ansatz=ansatz,
        initial_params=SPREAD_PARAMS,
        subspace_dimension=2,
        shots=128,
        seed=5,
    )

    result = SQDSolver().solve(request)

    assert result.energy >= h2_exact - 1e-12
    assert result.metadata["n_samples"] == 128
    assert result.metadata["is_exact_in_sector"] is False


# --------------------------------------------------------------------------------------
# Named refusals: every one says "sqd" and why
# --------------------------------------------------------------------------------------


def _oversized_request(n_qubits: int) -> SolveRequest:
    hamiltonian = QubitHamiltonian(
        operator=QubitOperator("Z0", 1.0), n_qubits=n_qubits, mapping="jordan_wigner"
    )
    active_space = ActiveSpace(
        n_active_orbitals=n_qubits // 2,
        n_active_electrons=2,
        orbital_indices=tuple(range(n_qubits // 2)),
    )
    return SolveRequest(hamiltonian=hamiltonian, active_space=active_space)


def test_refuses_above_the_platform_qubit_cap() -> None:
    with pytest.raises(BackendUnavailable) as excinfo:
        SQDSolver().check_supported(_oversized_request(MAX_QUBITS + 2))

    assert excinfo.value.backend == SOLVER_NAME
    assert excinfo.value.details["max_qubits"] == MAX_QUBITS


def test_sampled_sqd_runs_above_the_dense_statevector_ceiling() -> None:
    """Would fail if SQD still materialized a length-2^n vector or full sector.

    Thirty qubits is intentionally beyond the dense backend ceiling.  The selected
    determinant subspace has only two members, so its cost must follow the selected space,
    not the full Hilbert space.
    """
    n_qubits = MAX_DENSE_STATEVECTOR_QUBITS + 6
    hamiltonian = QubitHamiltonian(
        operator=QubitOperator((), -1.25),
        n_qubits=n_qubits,
        mapping="jordan_wigner",
    )
    first = (1 << (n_qubits - 1)) | (1 << (n_qubits - 2))
    second = (1 << (n_qubits - 3)) | (1 << (n_qubits - 4))

    outcome, vector = solve_sqd(
        hamiltonian,
        1,
        1,
        configurations=np.array([first, second], dtype=np.int64),
    )

    assert outcome.energy == pytest.approx(-1.25)
    assert outcome.subspace_dimension == 2
    assert outcome.sector_dimension == 225
    assert vector.shape == (2,)


def test_refuses_a_truncation_with_nothing_to_choose_from(
    h2_hamiltonian: QubitHamiltonian, h2_active_space: ActiveSpace
) -> None:
    """No ansatz and no configuration list means the subspace would be arbitrary."""
    request = _request(h2_hamiltonian, h2_active_space, subspace_dimension=2)

    with pytest.raises(BackendUnavailable) as excinfo:
        SQDSolver().solve(request)

    assert excinfo.value.backend == SOLVER_NAME
    assert "nothing to choose it from" in excinfo.value.reason


def test_refuses_a_shot_budget_with_nothing_to_sample(
    h2_hamiltonian: QubitHamiltonian, h2_active_space: ActiveSpace
) -> None:
    request = _request(h2_hamiltonian, h2_active_space, shots=1024)

    with pytest.raises(BackendUnavailable) as excinfo:
        SQDSolver().solve(request)

    assert excinfo.value.backend == SOLVER_NAME
    assert excinfo.value.details["shots"] == 1024


def test_refuses_an_ansatz_that_does_not_match_the_hamiltonian(
    h2_hamiltonian: QubitHamiltonian, h2_active_space: ActiveSpace
) -> None:
    mismatched = build_uccsd_ansatz(n_electrons=2, n_spin_orbitals=6)
    request = _request(h2_hamiltonian, h2_active_space, ansatz=mismatched)

    with pytest.raises(BackendUnavailable) as excinfo:
        SQDSolver().solve(request)

    assert excinfo.value.backend == SOLVER_NAME
    assert excinfo.value.details["hamiltonian_qubits"] == 4


def test_refusals_carry_the_shared_error_code() -> None:
    """A caller switching on ``code`` must not need to know SQD is not yet a driver."""
    with pytest.raises(BackendUnavailable) as excinfo:
        SQDSolver().check_supported(_oversized_request(MAX_QUBITS + 2))

    assert excinfo.value.to_dict()["code"] == "qdr.backend_unavailable"


def test_supports_is_the_non_raising_form(
    h2_hamiltonian: QubitHamiltonian, h2_active_space: ActiveSpace
) -> None:
    solver = SQDSolver()

    assert solver.supports(_request(h2_hamiltonian, h2_active_space))
    assert not solver.supports(_oversized_request(MAX_QUBITS + 2))


# --------------------------------------------------------------------------------------
# The request field itself
# --------------------------------------------------------------------------------------


def test_subspace_dimension_defaults_to_the_whole_sector(
    h2_hamiltonian: QubitHamiltonian, h2_active_space: ActiveSpace
) -> None:
    assert _request(h2_hamiltonian, h2_active_space).subspace_dimension is None


def test_subspace_dimension_must_be_at_least_one(
    h2_hamiltonian: QubitHamiltonian, h2_active_space: ActiveSpace
) -> None:
    """Zero configurations is not a small subspace, it is no subspace."""
    with pytest.raises(ValidationError, match="subspace_dimension"):
        _request(h2_hamiltonian, h2_active_space, subspace_dimension=0)


@pytest.mark.parametrize("bad", [0, -1, -2])
def test_solve_sqd_rejects_a_non_positive_dimension_too(
    h2_hamiltonian: QubitHamiltonian, bad: int
) -> None:
    """The guard has to live on the function, not only on ``SolveRequest``.

    ``solve_sqd`` is importable and gets called directly. Without its own check the value
    reaches a slice, and a negative one is the dangerous case: ``sector[:-1]`` quietly drops
    the last-ranked configuration and returns a smaller subspace with a perfectly plausible,
    still-variational energy. Nothing downstream can tell that apart from an honest answer,
    which is why this refuses rather than clamps.
    """
    with pytest.raises(ValidationError, match="subspace_dimension"):
        solve_sqd(h2_hamiltonian, H2_N_ALPHA, H2_N_BETA, subspace_dimension=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_ranking_rejects_a_non_positive_keep(bad: int) -> None:
    with pytest.raises(ValidationError, match="subspace_dimension"):
        rank_configurations_by_weight(np.array([3, 3, 12], dtype=np.int64), keep=bad)


# --------------------------------------------------------------------------------------
# Instrumented nested-subspace campaign
# --------------------------------------------------------------------------------------


def test_campaign_tracks_a_nested_variational_convergence_ladder(
    h2_hamiltonian: QubitHamiltonian,
) -> None:
    """Would fail if independently ranked steps were not genuinely nested."""
    sector = sector_configurations(4, H2_N_ALPHA, H2_N_BETA)
    exact = ground_state_energy_at_spin_sector_matrix_free(
        h2_hamiltonian, H2_N_ALPHA, H2_N_BETA
    )

    campaign = run_sqd_campaign(
        h2_hamiltonian,
        H2_N_ALPHA,
        H2_N_BETA,
        configurations=sector,
        dimensions=(1, 2, 4),
        reference_energy=exact,
    )

    assert [step.subspace_dimension for step in campaign.steps] == [1, 2, 4]
    assert all(
        set(earlier.configurations) < set(later.configurations)
        for earlier, later in pairwise(campaign.steps)
    )
    assert all(
        later.energy <= earlier.energy + 1.0e-10
        for earlier, later in pairwise(campaign.steps)
    )
    assert campaign.steps[-1].is_exact
    assert campaign.steps[-1].energy == pytest.approx(exact, abs=1.0e-8)
    assert campaign.final_energy_change == pytest.approx(
        abs(campaign.steps[-1].energy - campaign.steps[-2].energy)
    )


def test_campaign_reports_sector_retention_from_the_raw_draw(
    h2_hamiltonian: QubitHamiltonian,
) -> None:
    raw = np.array([3, 3, 12, 0, 15], dtype=np.int64)

    campaign = run_sqd_campaign(
        h2_hamiltonian,
        H2_N_ALPHA,
        H2_N_BETA,
        configurations=raw,
        dimensions=(1, 2),
    )

    assert campaign.n_samples == 5
    assert campaign.n_samples_in_sector == 3
    assert campaign.sector_retention == pytest.approx(0.6)


@pytest.mark.parametrize("dimensions", [(), (0,), (2, 2), (3, 2), (1, 5)])
def test_campaign_rejects_a_non_nested_or_impossible_dimension_ladder(
    h2_hamiltonian: QubitHamiltonian,
    dimensions: tuple[int, ...],
) -> None:
    sector = sector_configurations(4, H2_N_ALPHA, H2_N_BETA)

    with pytest.raises(ValidationError, match="dimensions"):
        run_sqd_campaign(
            h2_hamiltonian,
            H2_N_ALPHA,
            H2_N_BETA,
            configurations=sector,
            dimensions=dimensions,
        )

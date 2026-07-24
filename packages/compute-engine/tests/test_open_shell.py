"""Open-shell support -- the prerequisite for transition-metal spin gaps.

A spin gap is ``E(high-spin) - E(low-spin)`` for the *same* molecule, and the high-spin
state is open-shell: unequal numbers of alpha and beta electrons. Two pieces make that
computable here, and both are tested without PySCF:

* :func:`alpha_beta_from_spin` -- the pure ``(n_alpha, n_beta)`` bookkeeping from a total
  electron count and a spin multiplicity. All the failure modes (odd/even mismatch, too
  many unpaired electrons) are pure arithmetic and checked here.
* :func:`ground_state_energy_at_spin_sector_matrix_free` -- the exact energy restricted to a
  fixed ``(n_alpha, n_beta)`` sector, matrix-free (rung 3). Checked against the golden H2
  singlet (which is the FCI energy) and against a dense restriction on a synthetic operator.

The PySCF-backed ``solve_fci(spin_multiplicity=...)`` open-shell path and its ``<S^2>`` are
covered by ``requires_pyscf`` tests that run in Linux CI.
"""

import numpy as np
import pytest
from openfermion import QubitOperator, get_sparse_operator, hermitian_conjugated

from qdr_compute_engine.active_space.active_space_selector import select_active_space
from qdr_compute_engine.hamiltonian.fermionic_builder import build_fermionic_hamiltonian
from qdr_compute_engine.hamiltonian.matrix_free import (
    ground_state_energy_at_spin_sector_matrix_free,
)
from qdr_compute_engine.mappers.jordan_wigner import map_to_qubits
from qdr_compute_engine.solvers.classical_baseline import alpha_beta_from_spin
from qdr_compute_engine.types import IntegralSet, QubitHamiltonian
from qdr_core.errors import ValidationError

# --------------------------------------------------------------------------------------
# alpha/beta bookkeeping (pure)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_electrons", "multiplicity", "expected"),
    [
        (2, 1, (1, 1)),  # H2 singlet
        (10, 1, (5, 5)),  # closed shell
        (2, 3, (2, 0)),  # triplet (O2-like)
        (6, 5, (5, 1)),  # Fe(II) d6 high-spin quintet in a 6-electron active space
        (6, 1, (3, 3)),  # the same active space, low-spin singlet
        (5, 2, (3, 2)),  # doublet
    ],
)
def test_alpha_beta_from_spin(n_electrons: int, multiplicity: int, expected: tuple) -> None:
    assert alpha_beta_from_spin(n_electrons, multiplicity) == expected


def test_alpha_beta_rejects_parity_mismatch() -> None:
    # 3 electrons cannot be a singlet: (3 - 0) is odd.
    with pytest.raises(ValidationError, match="inconsistent"):
        alpha_beta_from_spin(3, 1)


def test_alpha_beta_rejects_too_many_unpaired() -> None:
    # 2 electrons cannot carry 4 unpaired spins (a quintet).
    with pytest.raises(ValidationError, match="unpaired"):
        alpha_beta_from_spin(2, 5)


# --------------------------------------------------------------------------------------
# Matrix-free energy in a fixed (n_alpha, n_beta) sector
# --------------------------------------------------------------------------------------


def _h2_qubit_hamiltonian(integrals: IntegralSet) -> QubitHamiltonian:
    active_space = select_active_space(integrals)
    return map_to_qubits(build_fermionic_hamiltonian(integrals, active_space))


def test_h2_singlet_sector_is_the_fci_energy(h2_integrals: IntegralSet, golden_h2: dict) -> None:
    """H2's ground state is the singlet (1 alpha, 1 beta); that sector's floor is FCI."""
    hamiltonian = _h2_qubit_hamiltonian(h2_integrals)
    energy = ground_state_energy_at_spin_sector_matrix_free(hamiltonian, n_alpha=1, n_beta=1)
    assert energy == pytest.approx(golden_h2["e_fci"], abs=1e-9)


def test_spin_sector_matches_dense_restriction() -> None:
    """The sector solve equals a dense diagonalisation restricted to the same indices.

    This checks the interleaved-spin index construction (even spin orbital = alpha) against
    an independent enumeration, on a synthetic Hermitian operator.
    """
    n_qubits = 4  # two spatial orbitals -> qubits 0,2 alpha; 1,3 beta
    rng = np.random.default_rng(2)
    operator = QubitOperator("", rng.standard_normal())
    for _ in range(20):
        string = " ".join(
            f"{rng.choice(['X', 'Y', 'Z'])}{q}"
            for q in sorted(rng.choice(n_qubits, size=2, replace=False))
        )
        operator += QubitOperator(string, rng.standard_normal())
    operator += hermitian_conjugated(operator)
    hamiltonian = QubitHamiltonian(operator=operator, n_qubits=n_qubits, mapping="jordan_wigner")

    n_alpha, n_beta = 1, 1
    dense = get_sparse_operator(operator, n_qubits=n_qubits).toarray()

    def alpha_beta(index: int) -> tuple[int, int]:
        occupied = [q for q in range(n_qubits) if (index >> (n_qubits - 1 - q)) & 1]
        return sum(q % 2 == 0 for q in occupied), sum(q % 2 == 1 for q in occupied)

    sector = [i for i in range(2**n_qubits) if alpha_beta(i) == (n_alpha, n_beta)]
    block = dense[np.ix_(sector, sector)]
    expected = float(np.linalg.eigvalsh(block)[0])

    got = ground_state_energy_at_spin_sector_matrix_free(hamiltonian, n_alpha, n_beta)
    assert got == pytest.approx(expected, abs=1e-9)


def test_empty_spin_sector_is_rejected() -> None:
    operator = QubitOperator("Z0", 1.0)
    hamiltonian = QubitHamiltonian(operator=operator, n_qubits=2, mapping="jordan_wigner")
    # 2 spatial spin orbitals interleaved -> only 1 alpha slot; asking for 2 is impossible.
    with pytest.raises(ValidationError, match="no basis state"):
        ground_state_energy_at_spin_sector_matrix_free(hamiltonian, n_alpha=2, n_beta=0)


# --------------------------------------------------------------------------------------
# Open-shell reference physics (PySCF; the spin gap the Colab runner reports rides on this)
# --------------------------------------------------------------------------------------


def _o2(spin_multiplicity: int):
    from qdr_compute_engine.types import Atom, Geometry, LengthUnit

    return Geometry(
        atoms=(Atom("O", (0.0, 0.0, 0.0)), Atom("O", (0.0, 0.0, 1.208))),
        charge=0,
        spin_multiplicity=spin_multiplicity,
        units=LengthUnit.ANGSTROM,
    )


@pytest.mark.requires_pyscf
def test_open_shell_integrals_converge_via_rohf() -> None:
    """compute_integrals must handle an open-shell geometry (ROHF), not reject it."""
    from qdr_compute_engine.integrals.pyscf_integrals import compute_integrals

    integrals = compute_integrals(_o2(spin_multiplicity=3), basis="sto-3g")
    assert integrals.n_electrons == 16
    assert integrals.n_orbitals == 10


@pytest.mark.requires_pyscf
def test_o2_triplet_is_the_ground_state() -> None:
    """O2's triplet lies below its singlet, and each state carries the right <S^2>.

    This is the end-to-end open-shell gate: a wrong (n_alpha, n_beta) sector or a lost spin
    label would flip the sign of the gap the runner reports.
    """
    from qdr_compute_engine.integrals.pyscf_integrals import compute_integrals
    from qdr_compute_engine.solvers.classical_baseline import solve_fci

    integrals = compute_integrals(_o2(spin_multiplicity=3), basis="sto-3g")
    triplet = solve_fci(integrals, spin_multiplicity=3)
    singlet = solve_fci(integrals, spin_multiplicity=1)

    assert triplet.energy < singlet.energy  # triplet ground state
    assert triplet.spin_squared == pytest.approx(2.0, abs=0.05)  # S=1 -> S(S+1)=2
    assert singlet.spin_squared == pytest.approx(0.0, abs=0.05)


@pytest.mark.requires_pyscf
def test_qdr_engine_matches_pyscf_on_an_open_shell_state() -> None:
    """QDR's matrix-free (n_alpha, n_beta) solve reproduces PySCF FCI for an open-shell state.

    The cross-check the Colab runner enforces before it will print a gap, in miniature.
    """
    from qdr_compute_engine.integrals.pyscf_integrals import compute_integrals
    from qdr_compute_engine.solvers.classical_baseline import solve_fci

    # A 2-electron triplet active space keeps the FCI space tiny while still open-shell.
    integrals = compute_integrals(_o2(spin_multiplicity=3), basis="sto-3g")
    active_space = select_active_space(integrals, n_active_orbitals=4, n_active_electrons=6)
    from qdr_compute_engine.active_space.active_space_selector import restrict_integrals

    restricted = restrict_integrals(integrals, active_space)
    reference = solve_fci(restricted, spin_multiplicity=3)

    hamiltonian = map_to_qubits(build_fermionic_hamiltonian(restricted))
    n_alpha, n_beta = alpha_beta_from_spin(restricted.n_electrons, 3)
    # e_nuc is already in the qubit Hamiltonian's identity term (via the InteractionOperator
    # constant + Jordan-Wigner), so the eigenvalue is the total energy -- do not add it again.
    e_qdr = ground_state_energy_at_spin_sector_matrix_free(hamiltonian, n_alpha, n_beta)

    assert e_qdr == pytest.approx(reference.energy, abs=1e-6)

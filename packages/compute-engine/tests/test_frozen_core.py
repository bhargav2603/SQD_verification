"""Frozen-core active-space reduction, checked against an independent oracle.

``restrict_integrals`` folds the doubly-occupied core orbitals into a constant energy
shift plus a mean-field correction to the one-body term. That algebra is only ever
exercised when a core is actually frozen -- the H2 gate never touches it, because H2 has
no core -- so it needs its own check, and one that does not simply restate the same
formula.

The oracle here is OpenFermion's ``freeze_orbitals``, a completely independent
implementation of the same physics operating on the second-quantised operator rather than
on spatial integrals. If the engine's spatial-integral reduction and OpenFermion's
operator-level reduction give the same ground-state energy at the same electron count,
the reduction is correct. None of this needs PySCF, so it runs on every platform.
"""

import numpy as np
import pytest
from openfermion import freeze_orbitals, get_fermion_operator, get_sparse_operator

from qdr_compute_engine.active_space.active_space_selector import (
    restrict_integrals,
    select_active_space,
)
from qdr_compute_engine.drivers.statevector.statevector_driver import (
    exact_ground_state_energy,
    exact_ground_state_energy_at_particle_number,
)
from qdr_compute_engine.hamiltonian.fermionic_builder import build_fermionic_hamiltonian
from qdr_compute_engine.integrals.pyscf_integrals import chemist_to_physicist
from qdr_compute_engine.mappers.jordan_wigner import map_to_qubits
from qdr_compute_engine.types import IntegralSet


def _random_integrals(n_orbitals: int, n_electrons: int, seed: int) -> IntegralSet:
    """A valid closed-shell integral set: symmetric h1 and a full 8-fold-symmetric ERI."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n_orbitals, n_orbitals))
    h1 = 0.5 * (a + a.T)
    factor = rng.standard_normal((n_orbitals, n_orbitals, 5))
    factor = 0.5 * (factor + factor.transpose(1, 0, 2))  # symmetric in the first pair
    eri_chemist = np.einsum("pqP,rsP->pqrs", factor, factor)  # (pq|rs), full symmetry
    return IntegralSet(
        h1=h1,
        h2=chemist_to_physicist(eri_chemist),
        e_nuc=1.234,
        n_electrons=n_electrons,
        e_hf=0.0,
    )


def _ground_energy_at_n(fermion_operator: object, n_qubits: int, n_electrons: int) -> float:
    """Lowest eigenvalue at fixed electron count, computed straight from the operator."""
    matrix = get_sparse_operator(fermion_operator, n_qubits=n_qubits).toarray()
    sector = [i for i in range(2**n_qubits) if bin(i).count("1") == n_electrons]
    return float(np.linalg.eigvalsh(matrix[np.ix_(sector, sector)])[0].real)


# (n_orbitals, n_electrons, n_active_orbitals, n_active_electrons)
_CASES = [
    (3, 4, 2, 2),
    (4, 4, 3, 2),
    (4, 6, 3, 4),
    (5, 6, 4, 4),
]


@pytest.mark.parametrize(("n_orb", "n_elec", "n_act_orb", "n_act_elec"), _CASES)
def test_restrict_integrals_matches_freeze_orbitals_oracle(
    n_orb: int, n_elec: int, n_act_orb: int, n_act_elec: int
) -> None:
    """The engine's frozen-core reduction must reproduce OpenFermion's, to machine precision.

    This runs the full engine path -- ``restrict_integrals`` then the fermionic build then
    the qubit map then a number-restricted diagonalisation -- and compares its energy to
    the oracle. It fails loudly if the core Coulomb/exchange folding is wrong by so much as
    a constant, which a variational optimisation would otherwise hide.
    """
    integrals = _random_integrals(n_orb, n_elec, seed=n_orb * 10 + n_elec)
    active_space = select_active_space(
        integrals, n_active_orbitals=n_act_orb, n_active_electrons=n_act_elec
    )
    n_core = active_space.n_frozen_core_orbitals

    # Oracle: freeze the core *spin* orbitals (both spins of each core spatial orbital).
    full_operator = get_fermion_operator(build_fermionic_hamiltonian(integrals).operator)
    core_spin_orbitals = sorted([2 * i for i in range(n_core)] + [2 * i + 1 for i in range(n_core)])
    oracle_operator = freeze_orbitals(full_operator, core_spin_orbitals)
    e_oracle = _ground_energy_at_n(oracle_operator, 2 * n_act_orb, n_act_elec)

    # Engine path.
    restricted = restrict_integrals(integrals, active_space)
    qubit_hamiltonian = map_to_qubits(build_fermionic_hamiltonian(restricted))
    e_engine = exact_ground_state_energy_at_particle_number(qubit_hamiltonian, n_act_elec)

    assert e_engine == pytest.approx(e_oracle, abs=1e-9), (
        f"frozen-core reduction gives {e_engine:.10f} Ha but the freeze_orbitals oracle "
        f"gives {e_oracle:.10f} Ha (differ by {e_engine - e_oracle:+.2e})"
    )


def test_number_restriction_ignores_a_lower_lying_wrong_sector() -> None:
    """The number-restricted energy must be the correct sector's floor, not the global one.

    An active-space effective Hamiltonian can have its unconstrained ground state in a
    different electron-number sector. This builds such a case and checks that restricting
    to the intended electron count returns a value at or above the global minimum, and one
    that a plain full-spectrum diagonalisation would miss.
    """
    integrals = _random_integrals(4, 6, seed=99)
    active_space = select_active_space(integrals, n_active_orbitals=3, n_active_electrons=4)
    restricted = restrict_integrals(integrals, active_space)
    qubit_hamiltonian = map_to_qubits(build_fermionic_hamiltonian(restricted))

    global_min = exact_ground_state_energy(qubit_hamiltonian)
    restricted_min = exact_ground_state_energy_at_particle_number(qubit_hamiltonian, 4)

    # The restricted floor can never be below the unconstrained floor.
    assert restricted_min >= global_min - 1e-9

    # And it agrees with an independent number-restricted diagonalisation of the operator.
    operator = get_fermion_operator(build_fermionic_hamiltonian(restricted).operator)
    assert restricted_min == pytest.approx(_ground_energy_at_n(operator, 6, 4), abs=1e-9)

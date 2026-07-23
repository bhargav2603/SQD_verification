"""AVAS active-space selection (PySCF; runs in Colab / Linux CI).

AVAS guarantees the *chosen* valence orbitals are in the active space -- the property an
index window cannot promise. These checks run where PySCF is available:

* the selected space is non-trivial and its integrals are self-consistent in size;
* CASCI in the AVAS space respects the variational bound (energy <= HF);
* QDR's own engine reproduces PySCF CASCI in the AVAS space (the runner's gate, on a real
  AVAS selection rather than an index window).

The input-validation guard (empty ``ao_labels``) is pure and runs everywhere.
"""

import numpy as np
import pytest

from qdr_compute_engine.active_space.avas_selection import avas_active_integrals
from qdr_compute_engine.types import Atom, Geometry, LengthUnit
from qdr_core.errors import ValidationError


def _o2(spin_multiplicity: int) -> Geometry:
    return Geometry(
        atoms=(Atom("O", (0.0, 0.0, 0.0)), Atom("O", (0.0, 0.0, 1.208))),
        charge=0,
        spin_multiplicity=spin_multiplicity,
        units=LengthUnit.ANGSTROM,
    )


def test_empty_ao_labels_is_rejected() -> None:
    with pytest.raises(ValidationError, match="target valence orbitals"):
        avas_active_integrals(_o2(3), ao_labels=())


@pytest.mark.requires_pyscf
def test_avas_selects_a_nontrivial_space_for_o2() -> None:
    result = avas_active_integrals(
        _o2(spin_multiplicity=3), ao_labels=["O 2p"], basis="sto-3g"
    )

    assert result.active_space.n_active_orbitals > 0
    assert result.active_space.n_active_electrons > 0
    # The restricted integrals must span exactly the active orbitals.
    assert result.integrals.n_orbitals == result.active_space.n_active_orbitals
    assert result.integrals.n_electrons == result.active_space.n_active_electrons


@pytest.mark.requires_pyscf
def test_qdr_matches_casci_in_the_avas_space() -> None:
    """The runner's cross-check, on an AVAS selection: QDR engine == PySCF CASCI."""
    from qdr_compute_engine.hamiltonian.fermionic_builder import (
        build_fermionic_hamiltonian,
    )
    from qdr_compute_engine.hamiltonian.matrix_free import (
        ground_state_energy_at_spin_sector_matrix_free,
    )
    from qdr_compute_engine.mappers.jordan_wigner import map_to_qubits
    from qdr_compute_engine.solvers.classical_baseline import (
        alpha_beta_from_spin,
        solve_fci,
    )

    result = avas_active_integrals(
        _o2(spin_multiplicity=3), ao_labels=["O 2p"], basis="sto-3g"
    )
    restricted = result.integrals

    reference = solve_fci(restricted, spin_multiplicity=3)
    assert reference.energy <= (restricted.e_hf or np.inf) + 1e-9  # variational bound

    hamiltonian = map_to_qubits(build_fermionic_hamiltonian(restricted))
    n_alpha, n_beta = alpha_beta_from_spin(restricted.n_electrons, 3)
    e_qdr = (
        ground_state_energy_at_spin_sector_matrix_free(hamiltonian, n_alpha, n_beta)
        + restricted.e_nuc
    )
    assert e_qdr == pytest.approx(reference.energy, abs=1e-6)

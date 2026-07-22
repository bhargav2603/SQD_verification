"""Active-space selection and the engine-side qubit cap (ADR-0006).

The cap has three enforcement points across the platform. This file covers the engine's,
which is the last line of defence: the control plane and dispatcher protect the queue,
but only this one stands between an oversized job and a worker trying to allocate a
statevector that cannot fit in memory.
"""

import numpy as np
import pytest

from qdr_compute_engine.active_space.active_space_selector import (
    enforce_qubit_cap,
    restrict_integrals,
    select_active_space,
)
from qdr_compute_engine.types import ActiveSpace, IntegralSet
from qdr_core.errors import QubitCapExceeded, ValidationError
from qdr_core.limits import MAX_ACTIVE_ORBITALS, MAX_QUBITS


def _integrals(n_orbitals: int, n_electrons: int) -> IntegralSet:
    """A structurally valid, physically meaningless integral set of a given size."""
    rng = np.random.default_rng(0)
    h1 = rng.normal(size=(n_orbitals, n_orbitals))
    h1 = h1 + h1.T
    h2 = np.zeros((n_orbitals,) * 4)
    return IntegralSet(h1=h1, h2=h2, e_nuc=0.0, n_electrons=n_electrons)


# --------------------------------------------------------------------------------------
# The cap
# --------------------------------------------------------------------------------------


def test_cap_allows_exactly_the_maximum() -> None:
    enforce_qubit_cap(MAX_ACTIVE_ORBITALS)  # 20 orbitals -> 40 qubits, allowed


def test_cap_rejects_one_orbital_too_many() -> None:
    with pytest.raises(QubitCapExceeded) as exc_info:
        enforce_qubit_cap(MAX_ACTIVE_ORBITALS + 1)

    error = exc_info.value
    assert error.required_qubits == 2 * (MAX_ACTIVE_ORBITALS + 1)
    assert error.max_qubits == MAX_QUBITS
    assert error.code == "qdr.qubit_cap_exceeded"


def test_selecting_twenty_one_orbitals_is_rejected() -> None:
    """The concrete case from the plan: 21 orbitals needs 42 qubits."""
    integrals = _integrals(n_orbitals=25, n_electrons=20)
    with pytest.raises(QubitCapExceeded) as exc_info:
        select_active_space(integrals, n_active_orbitals=21, n_active_electrons=20)
    assert exc_info.value.required_qubits == 42


def test_selecting_twenty_orbitals_is_allowed() -> None:
    integrals = _integrals(n_orbitals=25, n_electrons=20)
    active_space = select_active_space(integrals, n_active_orbitals=20, n_active_electrons=20)
    assert active_space.n_qubits == MAX_QUBITS


# --------------------------------------------------------------------------------------
# Selection logic
# --------------------------------------------------------------------------------------


def test_defaults_to_the_full_space(h2_integrals: IntegralSet) -> None:
    active_space = select_active_space(h2_integrals)
    assert active_space.n_active_orbitals == 2
    assert active_space.n_active_electrons == 2
    assert active_space.n_frozen_core_orbitals == 0
    assert active_space.orbital_indices == (0, 1)


def test_window_is_centred_above_the_frozen_core() -> None:
    """Freezing electrons freezes the lowest orbitals; the window starts after them."""
    integrals = _integrals(n_orbitals=10, n_electrons=8)
    active_space = select_active_space(integrals, n_active_orbitals=4, n_active_electrons=4)
    assert active_space.n_frozen_core_orbitals == 2  # (8 - 4) / 2
    assert active_space.orbital_indices == (2, 3, 4, 5)


def test_n_qubits_is_twice_the_orbital_count() -> None:
    integrals = _integrals(n_orbitals=10, n_electrons=8)
    active_space = select_active_space(integrals, n_active_orbitals=5, n_active_electrons=4)
    assert active_space.n_qubits == 10
    assert active_space.n_spin_orbitals == 10


def test_requesting_more_orbitals_than_the_basis_has_is_rejected() -> None:
    integrals = _integrals(n_orbitals=4, n_electrons=4)
    with pytest.raises(ValidationError, match="cannot select"):
        select_active_space(integrals, n_active_orbitals=6)


def test_requesting_more_electrons_than_exist_is_rejected() -> None:
    integrals = _integrals(n_orbitals=6, n_electrons=4)
    with pytest.raises(ValidationError, match="cannot correlate"):
        select_active_space(integrals, n_active_electrons=8)


def test_overfilling_the_active_orbitals_is_rejected() -> None:
    """More than two electrons per spatial orbital violates the Pauli principle."""
    integrals = _integrals(n_orbitals=6, n_electrons=10)
    with pytest.raises(ValidationError, match="cannot occupy"):
        select_active_space(integrals, n_active_orbitals=2, n_active_electrons=6)


def test_active_space_rejects_inconsistent_index_count() -> None:
    with pytest.raises(ValueError, match="orbital_indices"):
        ActiveSpace(n_active_orbitals=3, n_active_electrons=2, orbital_indices=(0, 1))


# --------------------------------------------------------------------------------------
# Integral restriction
# --------------------------------------------------------------------------------------


def test_restriction_without_a_frozen_core_is_a_pure_projection(
    h2_integrals: IntegralSet,
) -> None:
    active_space = select_active_space(h2_integrals)
    restricted = restrict_integrals(h2_integrals, active_space)

    assert restricted.n_orbitals == 2
    assert restricted.e_nuc == h2_integrals.e_nuc  # no core energy to fold in
    np.testing.assert_allclose(restricted.h1, h2_integrals.h1)
    np.testing.assert_allclose(restricted.h2, h2_integrals.h2)


def test_restriction_shrinks_the_tensors() -> None:
    integrals = _integrals(n_orbitals=8, n_electrons=6)
    active_space = select_active_space(integrals, n_active_orbitals=3, n_active_electrons=2)
    restricted = restrict_integrals(integrals, active_space)

    assert restricted.h1.shape == (3, 3)
    assert restricted.h2.shape == (3, 3, 3, 3)
    assert restricted.n_electrons == 2


def test_restriction_folds_a_frozen_core_energy_into_e_nuc() -> None:
    """Dropping the core constant would leave every total energy wrong by a fixed shift --
    invisible variationally, but fatal to any energy difference."""
    integrals = _integrals(n_orbitals=8, n_electrons=6)
    active_space = select_active_space(integrals, n_active_orbitals=3, n_active_electrons=2)
    restricted = restrict_integrals(integrals, active_space)

    assert active_space.n_frozen_core_orbitals == 2
    assert restricted.e_nuc != integrals.e_nuc

"""The qubit cap is an architectural invariant (ADR-0006), so it gets a test that fails
loudly if anyone edits the constant without also revisiting the decision record."""

import pytest

from qdr_core.limits import (
    CHEMICAL_ACCURACY_HA,
    MAX_ACTIVE_ORBITALS,
    MAX_QUBITS,
    QUBITS_PER_SPATIAL_ORBITAL,
    exceeds_qubit_cap,
    qubits_for_orbitals,
)


def test_max_qubits_is_forty() -> None:
    assert MAX_QUBITS == 40


def test_active_orbital_cap_is_consistent_with_qubit_cap() -> None:
    """MAX_ACTIVE_ORBITALS must be exactly the largest active space that fits."""
    assert qubits_for_orbitals(MAX_ACTIVE_ORBITALS) <= MAX_QUBITS
    assert qubits_for_orbitals(MAX_ACTIVE_ORBITALS + 1) > MAX_QUBITS


@pytest.mark.parametrize(
    ("n_orbitals", "expected_qubits"),
    [(0, 0), (1, 2), (2, 4), (10, 20), (20, 40)],
)
def test_qubits_for_orbitals(n_orbitals: int, expected_qubits: int) -> None:
    assert qubits_for_orbitals(n_orbitals) == expected_qubits


def test_qubits_for_orbitals_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        qubits_for_orbitals(-1)


def test_exceeds_qubit_cap_boundary() -> None:
    """The cap is inclusive: exactly MAX_QUBITS is allowed, one more is not."""
    assert not exceeds_qubit_cap(MAX_QUBITS)
    assert exceeds_qubit_cap(MAX_QUBITS + 1)


def test_spin_orbital_accounting() -> None:
    assert QUBITS_PER_SPATIAL_ORBITAL == 2


def test_chemical_accuracy_is_one_kcal_per_mol() -> None:
    """1 kcal/mol expressed in Hartree, to the precision the validator relies on."""
    from qdr_core.limits import HARTREE_TO_KCAL_PER_MOL

    assert pytest.approx(1.0, abs=5e-3) == CHEMICAL_ACCURACY_HA * HARTREE_TO_KCAL_PER_MOL

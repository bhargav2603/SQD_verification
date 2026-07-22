"""The three qubit ceilings must stay coherent.

These drifted once already: the statevector driver hard-coded 24, and two differently
named constants for the exact-reference limit -- ``EXACT_REFERENCE_QUBIT_LIMIT = 16`` in
stage 9 and ``EXACT_QUBIT_REFERENCE_LIMIT = 14`` in the benchmark runner -- disagreed
about the same quantity. Both are now imported from :mod:`qdr_core.limits`; these tests
are what stops the next copy from being pasted back in.
"""

import ast
from pathlib import Path

import pytest

from qdr_core.limits import (
    MAX_ACTIVE_ORBITALS,
    MAX_DENSE_STATEVECTOR_QUBITS,
    MAX_EXACT_REFERENCE_QUBITS,
    MAX_QUBITS,
    QUBITS_PER_SPATIAL_ORBITAL,
    exceeds_qubit_cap,
    qubits_for_orbitals,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE_SRC = REPO_ROOT / "packages" / "compute-engine" / "src"


def test_ladder_is_ordered() -> None:
    """Scheduling policy >= what a worker allocates >= what an exact reference costs."""
    assert MAX_QUBITS >= MAX_DENSE_STATEVECTOR_QUBITS >= MAX_EXACT_REFERENCE_QUBITS


def test_ladder_values_are_positive_and_even() -> None:
    """An odd qubit count cannot correspond to a whole number of spatial orbitals."""
    for value in (MAX_QUBITS, MAX_DENSE_STATEVECTOR_QUBITS, MAX_EXACT_REFERENCE_QUBITS):
        assert value > 0
        assert value % QUBITS_PER_SPATIAL_ORBITAL == 0


def test_active_orbital_cap_is_consistent() -> None:
    assert qubits_for_orbitals(MAX_ACTIVE_ORBITALS) == MAX_QUBITS
    assert not exceeds_qubit_cap(MAX_QUBITS)
    assert exceeds_qubit_cap(MAX_QUBITS + 1)


def test_qubits_for_orbitals_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        qubits_for_orbitals(-1)


def _module_level_int_constants(path: Path) -> dict[str, int]:
    """Module-level ``NAME = <int literal>`` assignments in one file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, int] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, int)):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                found[target.id] = node.value.value
    return found


#: Names that would indicate a re-hard-coded qubit ceiling.
_SUSPICIOUS = ("QUBIT", "QUBITS")


def test_no_engine_module_redefines_a_qubit_ceiling() -> None:
    """No module outside limits.py may define its own qubit-limit constant.

    Catches the exact regression that happened before: a literal pasted into a stage,
    slowly diverging from every other copy of the same number.
    """
    offenders: list[str] = []
    for path in sorted(ENGINE_SRC.rglob("*.py")):
        for name, value in _module_level_int_constants(path).items():
            if (
                any(token in name.upper() for token in _SUSPICIOUS) and "LIMIT" in name.upper()
            ) or (name.upper().startswith("MAX_") and "QUBIT" in name.upper()):
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {name} = {value}")

    assert not offenders, (
        "qubit ceilings must be imported from qdr_core.limits, not redefined:\n  "
        + "\n  ".join(offenders)
    )

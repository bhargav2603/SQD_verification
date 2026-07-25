"""Compatibility boundary for the maintained Qiskit SQD implementation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from qdr_compute_engine.active_space.active_space_selector import select_active_space
from qdr_compute_engine.audit.models import AuditBudget, AuditProblem, DatasetSplit
from qdr_compute_engine.audit.qiskit_sqd import (
    FermionicAuditProblem,
    QiskitBindings,
    QiskitSQDPostprocessor,
    integer_samples_to_bitstrings,
    qiskit_sqd_available,
)
from qdr_compute_engine.audit.samplers import UniformSectorSampler
from qdr_compute_engine.hamiltonian.fermionic_builder import build_fermionic_hamiltonian
from qdr_compute_engine.mappers.jordan_wigner import map_to_qubits
from qdr_compute_engine.types import IntegralSet
from qdr_core.errors import BackendUnavailable, ValidationError


@pytest.fixture
def h2_problem(h2_integrals: IntegralSet, golden_h2: dict[str, Any]) -> AuditProblem:
    active_space = select_active_space(h2_integrals)
    return AuditProblem(
        problem_id="h2",
        hamiltonian=map_to_qubits(
            build_fermionic_hamiltonian(h2_integrals, active_space)
        ),
        n_alpha=1,
        n_beta=1,
        reference_energy=float(golden_h2["e_fci"]),
        split=DatasetSplit.DEVELOPMENT,
    )


@pytest.fixture
def fermionic_h2(
    h2_problem: AuditProblem, h2_integrals: IntegralSet
) -> FermionicAuditProblem:
    return FermionicAuditProblem(
        problem_hash=h2_problem.identity_hash,
        one_body_tensor=h2_integrals.h1,
        two_body_tensor=h2_integrals.h2,
        constant_energy=h2_integrals.e_nuc,
    )


def test_integer_conversion_pins_qiskit_big_endian_bitstrings() -> None:
    assert integer_samples_to_bitstrings(
        np.array([12, 3], dtype=np.int64),
        n_qubits=4,
    ) == ("1100", "0011")


def test_integer_conversion_rejects_out_of_register_values() -> None:
    with pytest.raises(ValidationError, match="register"):
        integer_samples_to_bitstrings(np.array([16], dtype=np.int64), n_qubits=4)


def test_fermionic_problem_defensively_freezes_integrals(
    fermionic_h2: FermionicAuditProblem,
) -> None:
    assert fermionic_h2.n_orbitals == 2
    with pytest.raises(ValueError, match="read-only"):
        fermionic_h2.one_body_tensor[0, 0] = 99.0


def test_fermionic_problem_rejects_inconsistent_integral_shapes(
    h2_problem: AuditProblem,
) -> None:
    with pytest.raises(ValidationError, match="two_body_tensor"):
        FermionicAuditProblem(
            problem_hash=h2_problem.identity_hash,
            one_body_tensor=np.eye(2),
            two_body_tensor=np.zeros((2, 2, 2, 3)),
        )


def test_adapter_refuses_cleanly_when_optional_stack_is_absent(
    h2_problem: AuditProblem,
    fermionic_h2: FermionicAuditProblem,
) -> None:
    if qiskit_sqd_available():
        pytest.skip("optional Qiskit SQD stack is installed")
    budget = AuditBudget(shots=64, subspace_dimensions=(1,))
    batch = UniformSectorSampler().sample(h2_problem, budget, seed=3)

    with pytest.raises(BackendUnavailable, match="qiskit-addon-sqd"):
        QiskitSQDPostprocessor(
            fermionic_problems={h2_problem.identity_hash: fermionic_h2}
        ).process(h2_problem, batch, budget)


def test_adapter_forwards_frozen_settings_and_applies_constant_shift(
    h2_problem: AuditProblem,
    fermionic_h2: FermionicAuditProblem,
) -> None:
    calls: list[dict[str, Any]] = []

    def bit_array_factory(samples: tuple[str, ...], num_bits: int) -> Any:
        return {"samples": samples, "num_bits": num_bits}

    def diagonalize(
        one_body: np.ndarray,
        two_body: np.ndarray,
        bit_array: Any,
        **kwargs: Any,
    ) -> Any:
        calls.append(
            {
                "one_body": one_body,
                "two_body": two_body,
                "bit_array": bit_array,
                **kwargs,
            }
        )
        state = SimpleNamespace(
            ci_strs_a=np.array([1], dtype=np.int64),
            ci_strs_b=np.array([1], dtype=np.int64),
            amplitudes=np.array([[1.0]]),
        )
        result = SimpleNamespace(
            energy=-1.856,
            sci_state=state,
            orbital_occupancies=(np.array([1.0, 0.0]), np.array([1.0, 0.0])),
            spin_sq=0.0,
        )
        callback = kwargs.get("callback")
        if callback is not None:
            callback([result])
        return result

    bindings = QiskitBindings(
        bit_array_from_samples=bit_array_factory,
        diagonalize_fermionic_hamiltonian=diagonalize,
        addon_version="0.12.1-test",
    )
    budget = AuditBudget(shots=64, subspace_dimensions=(1,))
    batch = UniformSectorSampler().sample(h2_problem, budget, seed=5)
    postprocessor = QiskitSQDPostprocessor(
        fermionic_problems={h2_problem.identity_hash: fermionic_h2},
        bindings=bindings,
        num_batches=1,
        max_iterations=4,
    )

    (result,) = postprocessor.process(h2_problem, batch, budget)

    assert result.energy == pytest.approx(-1.856 + fermionic_h2.constant_energy)
    assert result.configurations == (12,)
    assert result.subspace_dimension == 1
    assert calls[0]["samples_per_batch"] == 1
    assert calls[0]["max_dim"] == 1
    assert calls[0]["norb"] == 2
    assert calls[0]["nelec"] == (1, 1)
    assert calls[0]["seed"] == 5
    assert result.metadata["qiskit_addon_sqd_version"] == "0.12.1-test"
    assert result.metadata["recovery_iterations"] == 1


def test_adapter_requires_registered_fermionic_integrals(
    h2_problem: AuditProblem,
) -> None:
    bindings = QiskitBindings(
        bit_array_from_samples=lambda samples, num_bits: None,
        diagonalize_fermionic_hamiltonian=lambda *args, **kwargs: None,
        addon_version="test",
    )
    budget = AuditBudget(shots=8, subspace_dimensions=(1,))
    batch = UniformSectorSampler().sample(h2_problem, budget, seed=0)

    with pytest.raises(BackendUnavailable, match="fermionic integrals"):
        QiskitSQDPostprocessor(
            fermionic_problems={},
            bindings=bindings,
        ).process(h2_problem, batch, budget)

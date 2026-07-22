"""SolverBackend contract conformance (ADR-0003).

Parametrized over the *registry* rather than a hardcoded list, so a backend added later
is conformance-tested automatically. That is the mechanism that keeps the seam honest as
it grows: you cannot add a driver and skip these checks without deliberately removing it
from the registry.

The central rule under test is that an unusable backend fails loudly. A driver that
returns a plausible number it cannot actually justify is worse than one that refuses,
because the number flows into a binding-energy comparison and nothing downstream can tell.
"""

from typing import Any

import numpy as np
import pytest

from qdr_compute_engine.active_space.active_space_selector import select_active_space
from qdr_compute_engine.ansatze.uccsd import build_uccsd_ansatz
from qdr_compute_engine.drivers import (
    available_backends,
    backend_capabilities,
    get_backend,
    implemented_backends,
)
from qdr_compute_engine.drivers.capabilities import BackendCapabilities
from qdr_compute_engine.drivers.registry import register_backend
from qdr_compute_engine.drivers.solver_backend import SolverBackend
from qdr_compute_engine.hamiltonian.fermionic_builder import build_fermionic_hamiltonian
from qdr_compute_engine.mappers.jordan_wigner import map_to_qubits
from qdr_compute_engine.types import (
    ActiveSpace,
    IntegralSet,
    QubitHamiltonian,
    SolveRequest,
    SolveResult,
)
from qdr_core.errors import BackendUnavailable, ValidationError
from qdr_core.limits import MAX_QUBITS

ALL_BACKENDS = available_backends()

# Split at collection time rather than skipping at run time: a skip would be
# indistinguishable from a test that quietly stopped running, and CI treats any skip on a
# fully-provisioned runner as a failure.
UNIMPLEMENTED_BACKENDS = [
    name for name in ALL_BACKENDS if not get_backend(name).capabilities.is_implemented
]


@pytest.fixture
def h2_request(h2_integrals: IntegralSet) -> SolveRequest:
    active_space = select_active_space(h2_integrals)
    qubit_hamiltonian = map_to_qubits(build_fermionic_hamiltonian(h2_integrals, active_space))
    return SolveRequest(
        hamiltonian=qubit_hamiltonian,
        active_space=active_space,
        ansatz=build_uccsd_ansatz(n_electrons=2, n_spin_orbitals=4),
        integrals=h2_integrals,
    )


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------


def test_all_four_backends_are_registered() -> None:
    assert set(ALL_BACKENDS) == {"classical", "statevector", "sampled", "qpu"}


def test_only_built_backends_are_advertised_as_implemented() -> None:
    assert set(implemented_backends()) == {"classical", "statevector"}


def test_unknown_backend_is_a_validation_error() -> None:
    """Distinct from BackendUnavailable: this is a client bug, not a capacity condition."""
    with pytest.raises(ValidationError):
        get_backend("not_a_backend")


def test_duplicate_registration_is_rejected() -> None:
    """Backend names are persisted on run records, so collisions must be impossible."""

    class Duplicate(SolverBackend):
        name = "statevector"

        @property
        def capabilities(self) -> BackendCapabilities:  # pragma: no cover - never called
            raise NotImplementedError

        def solve(self, request: SolveRequest) -> SolveResult:  # pragma: no cover
            raise NotImplementedError

    with pytest.raises(ValueError, match="already registered"):
        register_backend(Duplicate)


# --------------------------------------------------------------------------------------
# Contract, applied to every registered backend
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_BACKENDS)
def test_backend_instantiates_and_declares_its_name(name: str) -> None:
    backend = get_backend(name)
    assert isinstance(backend, SolverBackend)
    assert backend.name == name


@pytest.mark.parametrize("name", ALL_BACKENDS)
def test_capabilities_respect_the_platform_cap(name: str) -> None:
    """No backend may claim it can exceed the platform qubit cap."""
    assert get_backend(name).capabilities.max_qubits <= MAX_QUBITS


@pytest.mark.parametrize("name", ALL_BACKENDS)
def test_unimplemented_backends_state_a_reason(name: str) -> None:
    capabilities = get_backend(name).capabilities
    if not capabilities.is_implemented:
        assert capabilities.unavailable_reason


@pytest.mark.parametrize("name", ALL_BACKENDS)
def test_backend_rejects_oversized_requests(name: str, h2_request: SolveRequest) -> None:
    """Every backend refuses a register beyond its declared limit, via the shared check."""
    backend = get_backend(name)
    oversized = SolveRequest(
        hamiltonian=QubitHamiltonian(
            operator=h2_request.hamiltonian.operator,
            n_qubits=MAX_QUBITS + 2,
            mapping="jordan_wigner",
        ),
        active_space=h2_request.active_space,
        ansatz=h2_request.ansatz,
        integrals=h2_request.integrals,
    )
    with pytest.raises(BackendUnavailable):
        backend.solve(oversized)
    assert not backend.supports(oversized)


@pytest.mark.parametrize("name", UNIMPLEMENTED_BACKENDS)
def test_unimplemented_backends_raise_rather_than_return(
    name: str, h2_request: SolveRequest
) -> None:
    """The core honesty property: no silent plausible answers."""
    with pytest.raises(BackendUnavailable):
        get_backend(name).solve(h2_request)


@pytest.mark.parametrize("name", ALL_BACKENDS)
def test_supports_never_raises(name: str, h2_request: SolveRequest) -> None:
    """`supports` is the query form and must be safe to call on anything."""
    assert isinstance(get_backend(name).supports(h2_request), bool)


def test_backend_capabilities_table_covers_every_backend() -> None:
    table = backend_capabilities()
    assert set(table) == set(ALL_BACKENDS)
    assert all(isinstance(v, BackendCapabilities) for v in table.values())


# --------------------------------------------------------------------------------------
# Precondition enforcement
# --------------------------------------------------------------------------------------


def test_ansatz_requiring_backend_rejects_a_request_without_one(
    h2_request: SolveRequest,
) -> None:
    backend = get_backend("statevector")
    without_ansatz = SolveRequest(
        hamiltonian=h2_request.hamiltonian,
        active_space=h2_request.active_space,
        ansatz=None,
        integrals=h2_request.integrals,
    )
    with pytest.raises(BackendUnavailable, match="no ansatz"):
        backend.solve(without_ansatz)


def test_integral_requiring_backend_rejects_a_request_without_them(
    h2_request: SolveRequest,
) -> None:
    backend = get_backend("classical")
    without_integrals = SolveRequest(
        hamiltonian=h2_request.hamiltonian,
        active_space=h2_request.active_space,
        ansatz=h2_request.ansatz,
        integrals=None,
    )
    with pytest.raises(BackendUnavailable, match="integrals"):
        backend.solve(without_integrals)


def test_exact_backend_rejects_a_shot_budget(h2_request: SolveRequest) -> None:
    """Silently ignoring `shots` would misreport how the number was produced."""
    backend = get_backend("statevector")
    with_shots = SolveRequest(
        hamiltonian=h2_request.hamiltonian,
        active_space=h2_request.active_space,
        ansatz=h2_request.ansatz,
        integrals=h2_request.integrals,
        shots=1024,
    )
    with pytest.raises(BackendUnavailable, match="shot-free"):
        backend.solve(with_shots)


def test_mismatched_ansatz_width_is_rejected(h2_request: SolveRequest) -> None:
    """An ansatz spanning a different register than the Hamiltonian is a wiring bug."""
    backend = get_backend("statevector")
    mismatched = SolveRequest(
        hamiltonian=h2_request.hamiltonian,
        active_space=h2_request.active_space,
        ansatz=build_uccsd_ansatz(n_electrons=2, n_spin_orbitals=6),
        integrals=h2_request.integrals,
    )
    with pytest.raises(BackendUnavailable, match="spans"):
        backend.solve(mismatched)


# --------------------------------------------------------------------------------------
# Result shape, for backends that can actually run
# --------------------------------------------------------------------------------------


def test_statevector_result_shape(h2_request: SolveRequest) -> None:
    result = get_backend("statevector").solve(h2_request)

    assert isinstance(result, SolveResult)
    assert result.backend == "statevector"
    assert np.isfinite(result.energy)
    assert result.optimal_params is not None
    assert result.optimal_params.shape == (h2_request.ansatz.n_params,)  # type: ignore[union-attr]
    assert result.n_energy_evaluations > 0
    assert result.std_error is None  # exact backend reports no sampling error


@pytest.mark.requires_pyscf
def test_classical_result_shape(h2_request: SolveRequest) -> None:
    result = get_backend("classical").solve(h2_request)

    assert result.backend == "classical"
    assert result.method == "fci"
    assert np.isfinite(result.energy)
    assert result.optimal_params is None


@pytest.mark.requires_pyscf
def test_classical_and_statevector_agree_on_h2(h2_request: SolveRequest) -> None:
    """Two independent paths to the same ground state must land in the same place."""
    classical: Any = get_backend("classical").solve(h2_request)
    quantum: Any = get_backend("statevector").solve(h2_request)
    assert quantum.energy == pytest.approx(classical.energy, abs=1.6e-3)


def test_active_space_is_carried_through(h2_request: SolveRequest) -> None:
    assert isinstance(h2_request.active_space, ActiveSpace)
    assert h2_request.n_qubits == h2_request.active_space.n_qubits

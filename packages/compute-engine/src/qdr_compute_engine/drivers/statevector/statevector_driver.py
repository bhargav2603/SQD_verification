"""Exact statevector backend.

Simulates the gate IR densely and evaluates ``<psi|H|psi>`` exactly -- no sampling, so
the only error left in the answer is the ansatz's own (Trotter and active-space
truncation). That makes this the reference backend for correctness work: if the H2 gate
disagrees with FCI here, the bug is in the physics, not in the statistics.

Endianness
----------
The state is held as an ``(2,)*n`` array whose axis ``q`` is qubit ``q``, so flattening
puts qubit 0 in the most significant bit. That matches OpenFermion's
``get_sparse_operator`` exactly, which is the point: expectation values are taken against
OpenFermion's matrices, and a silent endianness mismatch would yield a wrong energy that
still looks like a plausible eigenvalue. ``tests/test_qubit_mapper.py`` pins the
convention against a hand-built state.
"""

import math

import numpy as np
from numpy.typing import NDArray
from openfermion import get_sparse_operator
from scipy.sparse import csc_matrix

from qdr_compute_engine.drivers.capabilities import BackendCapabilities
from qdr_compute_engine.drivers.registry import register_backend
from qdr_compute_engine.drivers.solver_backend import SolverBackend
from qdr_compute_engine.drivers.statevector.adjoint import energy_and_gradient
from qdr_compute_engine.drivers.statevector.kernel import simulate as kernel_simulate
from qdr_compute_engine.solvers.vqe import run_vqe
from qdr_compute_engine.types import (
    Circuit,
    Gate,
    GateName,
    QubitHamiltonian,
    SolveRequest,
    SolveResult,
)
from qdr_core.errors import ValidationError
from qdr_core.limits import MAX_DENSE_STATEVECTOR_QUBITS

_INV_SQRT2 = 1.0 / math.sqrt(2.0)

_STATIC_GATES: dict[GateName, NDArray[np.complex128]] = {
    GateName.H: np.array([[_INV_SQRT2, _INV_SQRT2], [_INV_SQRT2, -_INV_SQRT2]], dtype=complex),
    GateName.X: np.array([[0, 1], [1, 0]], dtype=complex),
    GateName.Y: np.array([[0, -1j], [1j, 0]], dtype=complex),
    GateName.Z: np.array([[1, 0], [0, -1]], dtype=complex),
    GateName.S: np.array([[1, 0], [0, 1j]], dtype=complex),
    GateName.SDG: np.array([[1, 0], [0, -1j]], dtype=complex),
}

_TWO_QUBIT_GATES: dict[GateName, NDArray[np.complex128]] = {
    # Basis order |ct>: control is the first factor.
    GateName.CX: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex),
    GateName.CZ: np.diag([1, 1, 1, -1]).astype(complex),
}


def _rotation_matrix(name: GateName, angle: float) -> NDArray[np.complex128]:
    half = angle / 2.0
    cos, sin = math.cos(half), math.sin(half)
    if name == GateName.RX:
        return np.array([[cos, -1j * sin], [-1j * sin, cos]], dtype=complex)
    if name == GateName.RY:
        return np.array([[cos, -sin], [sin, cos]], dtype=complex)
    if name == GateName.RZ:
        return np.array([[np.exp(-1j * half), 0], [0, np.exp(1j * half)]], dtype=complex)
    raise ValidationError(f"{name} is not a rotation gate")


def _apply_one_qubit(
    state: NDArray[np.complex128], matrix: NDArray[np.complex128], qubit: int
) -> NDArray[np.complex128]:
    contracted = np.tensordot(matrix, state, axes=([1], [qubit]))
    return np.moveaxis(contracted, 0, qubit)


def _apply_two_qubit(
    state: NDArray[np.complex128],
    matrix: NDArray[np.complex128],
    control: int,
    target: int,
) -> NDArray[np.complex128]:
    tensor = matrix.reshape(2, 2, 2, 2)
    contracted = np.tensordot(tensor, state, axes=([2, 3], [control, target]))
    return np.moveaxis(contracted, [0, 1], [control, target])


def _apply_gate(
    state: NDArray[np.complex128], gate: Gate, params: NDArray[np.float64]
) -> NDArray[np.complex128]:
    if gate.name in _STATIC_GATES:
        return _apply_one_qubit(state, _STATIC_GATES[gate.name], gate.qubits[0])
    if gate.name in _TWO_QUBIT_GATES:
        return _apply_two_qubit(state, _TWO_QUBIT_GATES[gate.name], gate.qubits[0], gate.qubits[1])
    matrix = _rotation_matrix(gate.name, gate.resolve_angle(params))
    return _apply_one_qubit(state, matrix, gate.qubits[0])


def simulate_statevector_reference(
    circuit: Circuit, params: NDArray[np.float64] | None = None
) -> NDArray[np.complex128]:
    """Run ``circuit`` on |0...0> the obvious way: one ``tensordot`` per gate.

    Superseded for production use by ``kernel.simulate``, which is the same computation
    without the per-gate allocation and transpose. This implementation is **kept
    deliberately** as the oracle: ``tests/test_statevector_kernel.py`` asserts the fast
    path reproduces it exactly, and that check is only meaningful while the two remain
    genuinely independent derivations. Do not reimplement this in terms of the kernel.
    """
    if params is None:
        params = np.zeros(circuit.n_params, dtype=float)
    elif len(params) != circuit.n_params:
        raise ValidationError(
            f"circuit expects {circuit.n_params} parameters, got {len(params)}",
            expected=circuit.n_params,
            received=len(params),
        )

    state = np.zeros((2,) * circuit.n_qubits, dtype=complex)
    state[(0,) * circuit.n_qubits] = 1.0

    for gate in circuit.gates:
        state = _apply_gate(state, gate, params)

    return state.reshape(-1)


def simulate_statevector(
    circuit: Circuit, params: NDArray[np.float64] | None = None
) -> NDArray[np.complex128]:
    """Run ``circuit`` on |0...0> and return the resulting statevector.

    Delegates to the in-place kernel. Verified equivalent to
    :func:`simulate_statevector_reference` at machine precision.
    """
    return kernel_simulate(circuit, params).reshape(-1)


def hamiltonian_matrix(hamiltonian: QubitHamiltonian) -> csc_matrix:
    """Sparse matrix form of a qubit Hamiltonian."""
    return get_sparse_operator(hamiltonian.operator, n_qubits=hamiltonian.n_qubits)


def expectation_value(hamiltonian_sparse: csc_matrix, state: NDArray[np.complex128]) -> float:
    """``<psi|H|psi>`` for a Hermitian ``H``.

    The imaginary part is discarded after an explicit check -- for a Hermitian operator
    it is zero up to rounding, and a non-negligible one means the operator is not what
    the caller thinks it is.
    """
    value = complex(np.vdot(state, hamiltonian_sparse @ state))
    if abs(value.imag) > 1e-8:
        raise ValidationError(
            f"expectation value has a significant imaginary part ({value.imag:.3e}); "
            "the Hamiltonian is not Hermitian",
            imaginary_part=value.imag,
        )
    return float(value.real)


@register_backend
class StatevectorDriver(SolverBackend):
    """Exact simulation of the ansatz, optimised variationally."""

    name = "statevector"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            # Defined in qdr_core.limits alongside the platform cap and the exact-
            # reference limit, so the three ceilings cannot drift apart again.
            max_qubits=MAX_DENSE_STATEVECTOR_QUBITS,
            is_exact=True,
            supports_shots=False,
            requires_ansatz=True,
            requires_integrals=False,
        )

    def solve(self, request: SolveRequest) -> SolveResult:
        self.check_supported(request)
        ansatz = request.ansatz
        assert ansatz is not None  # guaranteed by check_supported

        sparse_hamiltonian = hamiltonian_matrix(request.hamiltonian)

        def energy(params: NDArray[np.float64]) -> float:
            return expectation_value(sparse_hamiltonian, simulate_statevector(ansatz, params))

        def value_and_gradient(
            params: NDArray[np.float64],
        ) -> tuple[float, NDArray[np.float64]]:
            return energy_and_gradient(ansatz, params, sparse_hamiltonian)

        initial = (
            request.initial_params
            if request.initial_params is not None
            else np.zeros(ansatz.n_params, dtype=float)
        )

        # This backend is exact, so the energy is smooth and an analytic gradient is
        # available. `optimizer` may still be set explicitly -- naming a derivative-free
        # method falls back to the energy-only path, which is what the driver contract
        # tests use to compare the two searches on identical physics.
        requested_optimizer = request.options.get("optimizer")
        method = str(requested_optimizer) if requested_optimizer is not None else None

        outcome = run_vqe(
            energy,
            initial,
            max_iterations=request.max_iterations,
            method=method,
            tolerance=float(request.options.get("tolerance", 1e-8)),
            value_and_gradient=value_and_gradient,
        )

        return SolveResult(
            energy=outcome.energy,
            backend=self.name,
            method="vqe-statevector",
            converged=outcome.converged,
            n_iterations=outcome.n_iterations,
            n_energy_evaluations=outcome.n_energy_evaluations,
            optimal_params=outcome.optimal_params,
            std_error=None,
            metadata={
                "optimizer": outcome.method,
                "used_analytic_gradient": outcome.used_analytic_gradient,
                "initial_energy": outcome.initial_energy,
                "message": outcome.message,
                "n_qubits": request.n_qubits,
                "n_hamiltonian_terms": request.hamiltonian.n_terms,
                # Consumed by the stage-9 error budget to estimate optimiser residual.
                "history": outcome.history,
            },
        )


def exact_ground_state_energy(hamiltonian: QubitHamiltonian) -> float:
    """Lowest eigenvalue of the qubit Hamiltonian, bypassing the ansatz entirely.

    This is the variational floor: no ansatz can beat it. Used by the validator to
    quantify how much energy the ansatz leaves on the table, and by tests as an
    ansatz-independent reference.
    """
    from scipy.sparse.linalg import eigsh

    matrix = hamiltonian_matrix(hamiltonian)
    dimension = matrix.shape[0]

    if dimension <= 64:
        # eigsh cannot return all eigenvalues of a small matrix; dense is cheaper anyway.
        return float(np.linalg.eigvalsh(matrix.toarray())[0])

    eigenvalues = eigsh(matrix, k=1, which="SA", return_eigenvectors=False)
    return float(eigenvalues[0])

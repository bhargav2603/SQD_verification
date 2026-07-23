"""Adjoint (reverse-mode) gradients for the exact statevector backend.

Why this exists
---------------
``solvers/vqe.py`` defaults to COBYLA because a *sampled* backend returns a noisy energy
and derivative-free search degrades gracefully under noise. That reasoning does not apply
to the exact backend, where the energy is a smooth deterministic function of the
parameters -- and paying for gradient-free search there is expensive:

  * finite differences cost ``2 * n_params`` energy evaluations per gradient. At CAS(8,8)
    that is 720 full circuit simulations for one gradient.
  * COBYLA needs thousands of energy evaluations to converge 360 parameters at all.

Adjoint differentiation returns **every** partial derivative for the cost of roughly
three circuit simulations, independent of ``n_params``.

The method
----------
Write the energy as ``E(theta) = <0| U_1+ ... U_N+  H  U_N ... U_1 |0>``. Define

    |psi_k>    = U_k ... U_1 |0>              the state after gate k
    |lambda_k> = U_{k+1}+ ... U_N+ H |psi_N>  the co-state at gate k

Then ``dE/dtheta_k = 2 Re[ <lambda_k| dU_k/dtheta_k |psi_{k-1}> ]``.

Every parameterised gate here is a rotation ``U = exp(-i * phi * P / 2)`` about a Pauli
``P`` in ``{X, Y, Z}``, with ``phi = param_scale * theta``. Differentiating,

    dU/dtheta |psi_{k-1}> = param_scale * (-i/2) * P |psi_k>

and substituting collapses the expression to a single real number:

    dE/dtheta_k = param_scale * Im[ <lambda_k| P |psi_k> ]

Both ``|psi_k>`` and ``|lambda_k>`` are needed for every ``k``, walking backwards. Storing
all ``N`` intermediates is not an option -- 60,000 gates at 16 qubits is ~60 GiB -- so
both are *reconstructed* by applying gate inverses as the walk proceeds. That is exact
(every gate is unitary), and costs one extra state per step rather than one per gate.

Total cost: one forward pass, one sparse ``H @ psi``, and two backward passes. Memory: two
statevectors.

Shared parameters
-----------------
A UCCSD amplitude fans out to many Pauli rotations that all carry the same
``param_index`` with different ``param_scale`` values. The chain rule sums their
contributions, which is exactly what accumulating into ``gradient[param_index]`` does --
so this handles the Trotterised ansatz without special-casing it.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator

from qdr_compute_engine.drivers.statevector.kernel import (
    State,
    _axis,
    apply_gate_inverse,
    simulate,
)
from qdr_compute_engine.types import Circuit, GateName
from qdr_core.errors import ValidationError

#: Pauli generator of each parameterised rotation: ``U = exp(-i * phi * P / 2)``.
_GENERATOR: dict[GateName, str] = {
    GateName.RX: "X",
    GateName.RY: "Y",
    GateName.RZ: "Z",
}


def _pauli_braket(lam: State, psi: State, pauli: str, qubit: int) -> complex:
    """``<lambda| P_qubit |psi>`` without materialising ``P``.

    Each Pauli acts on the two halves of one axis, so the inner product is two half-size
    dot products combined -- far cheaper than a sparse matrix application, and this runs
    once per parameterised gate.
    """
    lower = _axis(qubit, 0)
    upper = _axis(qubit, 1)

    if pauli == "Z":
        # Z = diag(1, -1)
        return complex(np.vdot(lam[lower], psi[lower]) - np.vdot(lam[upper], psi[upper]))
    if pauli == "X":
        # X exchanges the halves.
        return complex(np.vdot(lam[lower], psi[upper]) + np.vdot(lam[upper], psi[lower]))
    if pauli == "Y":
        # Y = [[0, -i], [i, 0]]
        return complex(-1j * np.vdot(lam[lower], psi[upper]) + 1j * np.vdot(lam[upper], psi[lower]))
    raise ValidationError(f"unknown Pauli generator {pauli!r}")


def energy_and_gradient(
    circuit: Circuit,
    params: NDArray[np.float64],
    hamiltonian_sparse: csc_matrix | LinearOperator,
) -> tuple[float, NDArray[np.float64]]:
    """Return ``(energy, gradient)`` for ``circuit`` at ``params``.

    ``gradient[i]`` is ``dE/dtheta_i``, exact to machine precision -- this is analytic
    differentiation, not a finite-difference approximation.

    ``hamiltonian_sparse`` is only ever used as ``H @ psi`` (the single co-state seed), so a
    matrix-free ``LinearOperator`` is accepted interchangeably with the sparse matrix -- which
    is what lets the gradient path clear the 14-qubit wall alongside the reference (rung 3).
    """
    n_qubits = circuit.n_qubits

    # Forward pass: |psi_N>.
    psi = simulate(circuit, params)
    psi_flat = psi.reshape(-1)

    h_psi_flat = hamiltonian_sparse @ psi_flat
    energy_complex = complex(np.vdot(psi_flat, h_psi_flat))
    if abs(energy_complex.imag) > 1e-8:
        raise ValidationError(
            f"expectation value has a significant imaginary part "
            f"({energy_complex.imag:.3e}); the Hamiltonian is not Hermitian",
            imaginary_part=energy_complex.imag,
        )
    energy = float(energy_complex.real)

    # Co-state seed: |lambda_N> = H |psi_N>.
    lam = np.array(h_psi_flat, dtype=complex).reshape((2,) * n_qubits)

    gradient = np.zeros(circuit.n_params, dtype=float)

    # Backward pass. On entry to each iteration `psi` holds |psi_k> and `lam` holds
    # |lambda_k| for the gate about to be processed, so the derivative is read off
    # before either is stepped back.
    for gate in reversed(circuit.gates):
        if gate.is_parameterized:
            generator = _GENERATOR.get(gate.name)
            if generator is None:
                raise ValidationError(
                    f"gate {gate.name!r} is parameterised but has no Pauli generator; "
                    "the adjoint gradient cannot differentiate it",
                    gate=str(gate.name),
                )
            braket = _pauli_braket(lam, psi, generator, gate.qubits[0])
            assert gate.param_index is not None  # guaranteed by is_parameterized
            gradient[gate.param_index] += gate.param_scale * braket.imag

        apply_gate_inverse(psi, gate, params)
        apply_gate_inverse(lam, gate, params)

    return energy, gradient


def finite_difference_gradient(
    circuit: Circuit,
    params: NDArray[np.float64],
    hamiltonian_sparse: csc_matrix,
    step: float = 1e-5,
) -> NDArray[np.float64]:
    """Central-difference gradient. Test oracle only -- ``2 * n_params`` simulations.

    Kept beside the adjoint implementation deliberately: an analytic gradient that is
    silently wrong sends the optimiser somewhere plausible and wrong, so the two are
    cross-checked in ``tests/test_adjoint_gradient.py``.
    """
    gradient = np.zeros(circuit.n_params, dtype=float)
    for index in range(circuit.n_params):
        shifted_up = np.array(params, dtype=float)
        shifted_down = np.array(params, dtype=float)
        shifted_up[index] += step
        shifted_down[index] -= step

        psi_up = simulate(circuit, shifted_up).reshape(-1)
        psi_down = simulate(circuit, shifted_down).reshape(-1)
        energy_up = float(np.vdot(psi_up, hamiltonian_sparse @ psi_up).real)
        energy_down = float(np.vdot(psi_down, hamiltonian_sparse @ psi_down).real)
        gradient[index] = (energy_up - energy_down) / (2.0 * step)
    return gradient

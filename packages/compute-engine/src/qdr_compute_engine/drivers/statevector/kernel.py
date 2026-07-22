"""In-place statevector kernel.

The reference implementation in ``statevector_driver`` applies every gate with
``np.tensordot`` followed by ``np.moveaxis``. Both allocate a fresh ``2^n`` array and the
second one transposes it, so a UCCSD energy evaluation at 16 qubits performs ~60,000
full-state allocations *and* ~60,000 cache-hostile transposes. Measured cost at CAS(8,8):
82 s for a single energy evaluation, which puts a VQE run in the region of two days.

This module applies each gate to a **slice pair** of the ``(2,)*n`` array instead. Indexing
axis ``q`` with ``0`` and ``1`` yields two views over disjoint halves of the state; a
one-qubit gate is then four multiply-adds over half-size views, with no transpose and no
full-state allocation anywhere.

Three gate classes get a specialisation rather than the generic 2x2 path, because they
dominate the UCCSD gate count:

``RZ`` (and ``Z``, ``S``, ``SDG``)
    Diagonal. A phase multiply applied in place -- no temporary at all.
``CX`` (and ``X``)
    A permutation. One quarter-size (respectively half-size) temporary and two view
    assignments; no arithmetic.
``CZ``
    Diagonal on one quadrant. A sign flip in place.

In the H2 -> CAS(8,8) ansatz family those cover ~85% of all gates, and CX alone is ~60%.

Correctness is not traded for any of this: ``tests/test_statevector_kernel.py`` asserts
this module reproduces the reference implementation to machine precision gate by gate,
and the H2 FCI gate continues to run through the full stack.
"""

import math

import numpy as np
from numpy.typing import NDArray

from qdr_compute_engine.types import Circuit, Gate, GateName
from qdr_core.errors import ValidationError

State = NDArray[np.complex128]

#: A numpy basic-indexing tuple: full slices on every axis except the pinned ones.
#: Spelled out rather than left as ``tuple[object, ...]`` so numpy's stubs accept it.
AxisIndex = tuple[slice | int, ...]

_INV_SQRT2 = 1.0 / math.sqrt(2.0)

#: Gates that are their own inverse, so ``apply_inverse`` can defer to ``apply``.
_SELF_INVERSE = frozenset(
    {GateName.H, GateName.X, GateName.Y, GateName.Z, GateName.CX, GateName.CZ}
)


def _axis(qubit: int, value: int) -> AxisIndex:
    """Index tuple selecting ``value`` on axis ``qubit``.

    Trailing axes are implicitly full slices, so only the leading ``qubit`` entries need
    to be spelled out. The result is a *view*, which is the whole point -- assigning
    through it writes into the original buffer.
    """
    return (slice(None),) * qubit + (value,)


def _axis2(qubit_a: int, value_a: int, qubit_b: int, value_b: int) -> AxisIndex:
    """Index tuple pinning two axes at once, for two-qubit gates."""
    high = max(qubit_a, qubit_b)
    index: list[slice | int] = [slice(None)] * (high + 1)
    index[qubit_a] = value_a
    index[qubit_b] = value_b
    return tuple(index)


def _apply_matrix(state: State, matrix: NDArray[np.complex128], qubit: int) -> None:
    """Apply a general one-qubit matrix in place."""
    lower = _axis(qubit, 0)
    upper = _axis(qubit, 1)
    # Both halves are read before either is written, so both need a temporary.
    a = state[lower].copy()
    b = state[upper].copy()
    state[lower] = matrix[0, 0] * a + matrix[0, 1] * b
    state[upper] = matrix[1, 0] * a + matrix[1, 1] * b


def _apply_diagonal(state: State, phase_0: complex, phase_1: complex, qubit: int) -> None:
    """Apply ``diag(phase_0, phase_1)`` in place, with no temporary."""
    if phase_0 != 1.0:
        state[_axis(qubit, 0)] *= phase_0
    if phase_1 != 1.0:
        state[_axis(qubit, 1)] *= phase_1


def _apply_swap_halves(state: State, qubit: int) -> None:
    """Exchange the two halves of ``qubit`` -- the action of X."""
    lower = _axis(qubit, 0)
    upper = _axis(qubit, 1)
    temp = state[lower].copy()
    state[lower] = state[upper]
    state[upper] = temp


_H_MATRIX = np.array([[_INV_SQRT2, _INV_SQRT2], [_INV_SQRT2, -_INV_SQRT2]], dtype=complex)
_Y_MATRIX = np.array([[0, -1j], [1j, 0]], dtype=complex)


def apply_gate(state: State, gate: Gate, params: NDArray[np.float64]) -> None:
    """Apply ``gate`` to ``state`` in place.

    ``state`` must have shape ``(2,)*n_qubits``. Nothing is returned: mutating in place is
    the entire performance argument for this module.
    """
    name = gate.name

    if name == GateName.CX:
        control, target = gate.qubits
        # Controlled-X permutes the two target halves within the control-is-one subspace.
        lower = _axis2(control, 1, target, 0)
        upper = _axis2(control, 1, target, 1)
        temp = state[lower].copy()
        state[lower] = state[upper]
        state[upper] = temp
        return

    if name == GateName.CZ:
        control, target = gate.qubits
        state[_axis2(control, 1, target, 1)] *= -1.0
        return

    qubit = gate.qubits[0]

    if name == GateName.X:
        _apply_swap_halves(state, qubit)
    elif name == GateName.Z:
        _apply_diagonal(state, 1.0, -1.0, qubit)
    elif name == GateName.S:
        _apply_diagonal(state, 1.0, 1j, qubit)
    elif name == GateName.SDG:
        _apply_diagonal(state, 1.0, -1j, qubit)
    elif name == GateName.H:
        _apply_matrix(state, _H_MATRIX, qubit)
    elif name == GateName.Y:
        _apply_matrix(state, _Y_MATRIX, qubit)
    elif name == GateName.RZ:
        half = gate.resolve_angle(params) / 2.0
        # Diagonal in the computational basis: the dominant gate in a Trotterised
        # excitation, and the cheapest possible path through this function.
        _apply_diagonal(
            state,
            complex(math.cos(half), -math.sin(half)),
            complex(math.cos(half), math.sin(half)),
            qubit,
        )
    elif name == GateName.RX:
        half = gate.resolve_angle(params) / 2.0
        cos, sin = math.cos(half), math.sin(half)
        _apply_matrix(state, np.array([[cos, -1j * sin], [-1j * sin, cos]], dtype=complex), qubit)
    elif name == GateName.RY:
        half = gate.resolve_angle(params) / 2.0
        cos, sin = math.cos(half), math.sin(half)
        _apply_matrix(state, np.array([[cos, -sin], [sin, cos]], dtype=complex), qubit)
    else:
        raise ValidationError(f"kernel has no implementation for gate {name!r}", gate=str(name))


def apply_gate_inverse(state: State, gate: Gate, params: NDArray[np.float64]) -> None:
    """Apply ``gate``'s inverse in place.

    Needed by the adjoint gradient, which walks the circuit backwards and reconstructs
    each intermediate state by undoing gates rather than storing them -- storing all
    ~60,000 intermediates at 16 qubits would need ~60 GiB.
    """
    name = gate.name

    if name in _SELF_INVERSE:
        apply_gate(state, gate, params)
        return

    if name == GateName.S:
        _apply_diagonal(state, 1.0, -1j, gate.qubits[0])
        return
    if name == GateName.SDG:
        _apply_diagonal(state, 1.0, 1j, gate.qubits[0])
        return

    # The rotations invert by negating the angle. Rebuilding the Gate would re-run
    # validation on every call, so the angle is negated at the point of use instead.
    qubit = gate.qubits[0]
    half = -gate.resolve_angle(params) / 2.0
    if name == GateName.RZ:
        _apply_diagonal(
            state,
            complex(math.cos(half), -math.sin(half)),
            complex(math.cos(half), math.sin(half)),
            qubit,
        )
    elif name == GateName.RX:
        cos, sin = math.cos(half), math.sin(half)
        _apply_matrix(state, np.array([[cos, -1j * sin], [-1j * sin, cos]], dtype=complex), qubit)
    elif name == GateName.RY:
        cos, sin = math.cos(half), math.sin(half)
        _apply_matrix(state, np.array([[cos, -sin], [sin, cos]], dtype=complex), qubit)
    else:
        raise ValidationError(f"kernel has no inverse for gate {name!r}", gate=str(name))


def zero_state(n_qubits: int) -> State:
    """The ``|0...0>`` computational basis state, shaped ``(2,)*n_qubits``."""
    state = np.zeros((2,) * n_qubits, dtype=complex)
    state[(0,) * n_qubits] = 1.0
    return state


def simulate(circuit: Circuit, params: NDArray[np.float64] | None = None) -> State:
    """Run ``circuit`` on ``|0...0>``; returns the state shaped ``(2,)*n_qubits``.

    The flat statevector the rest of the engine expects is ``simulate(...).reshape(-1)``.
    """
    if params is None:
        params = np.zeros(circuit.n_params, dtype=float)
    elif len(params) != circuit.n_params:
        raise ValidationError(
            f"circuit expects {circuit.n_params} parameters, got {len(params)}",
            expected=circuit.n_params,
            received=len(params),
        )

    state = zero_state(circuit.n_qubits)
    for gate in circuit.gates:
        apply_gate(state, gate, params)
    return state

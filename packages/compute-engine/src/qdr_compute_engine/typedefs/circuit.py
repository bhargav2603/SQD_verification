"""The circuit IR.

The vendor-neutral gate representation ADR-0003 rests on: ansaetze emit it,
drivers consume it, and no vendor circuit type appears in the pipeline.

Re-exported by ``qdr_compute_engine.types``, which remains the import site for the
whole engine -- these modules exist to keep each file small, not to add a new path.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

# --------------------------------------------------------------------------------------
# Circuit IR
# --------------------------------------------------------------------------------------


class GateName(StrEnum):
    H = "h"
    X = "x"
    Y = "y"
    Z = "z"
    S = "s"
    SDG = "sdg"
    RX = "rx"
    RY = "ry"
    RZ = "rz"
    CX = "cx"
    CZ = "cz"


#: Gates that take a rotation angle.
ROTATION_GATES = frozenset({GateName.RX, GateName.RY, GateName.RZ})

#: Number of qubits each gate acts on.
GATE_ARITY: dict[GateName, int] = {
    GateName.H: 1,
    GateName.X: 1,
    GateName.Y: 1,
    GateName.Z: 1,
    GateName.S: 1,
    GateName.SDG: 1,
    GateName.RX: 1,
    GateName.RY: 1,
    GateName.RZ: 1,
    GateName.CX: 2,
    GateName.CZ: 2,
}


@dataclass(frozen=True, slots=True)
class Gate:
    """One operation in the IR.

    A rotation is either *bound* (fixed ``angle``) or *free* (``param_index`` into the
    parameter vector). Requiring exactly one of the two makes an unbound parameter a
    construction-time error rather than a confusing failure inside the optimizer.

    ``param_scale`` lets several gates share one parameter with different coefficients,
    which is what a Trotterised excitation operator needs: one variational amplitude per
    excitation fans out to many Pauli rotations, each carrying its own Jordan-Wigner
    coefficient. Without it the ansatz would need one free parameter per Pauli term,
    which is a different (and wrong) ansatz.
    """

    name: GateName
    qubits: tuple[int, ...]
    param_index: int | None = None
    angle: float | None = None
    param_scale: float = 1.0

    def __post_init__(self) -> None:
        expected_arity = GATE_ARITY[self.name]
        if len(self.qubits) != expected_arity:
            raise ValueError(
                f"{self.name} acts on {expected_arity} qubit(s), got {len(self.qubits)}"
            )
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError(f"{self.name} has repeated qubit indices: {self.qubits}")
        if any(q < 0 for q in self.qubits):
            raise ValueError(f"qubit indices must be non-negative, got {self.qubits}")

        is_rotation = self.name in ROTATION_GATES
        has_binding = (self.param_index is not None) or (self.angle is not None)
        if is_rotation and not has_binding:
            raise ValueError(f"{self.name} needs either a param_index or a fixed angle")
        if not is_rotation and has_binding:
            raise ValueError(f"{self.name} takes no angle")
        if self.param_index is not None and self.angle is not None:
            raise ValueError(f"{self.name} cannot be both parameterized and bound")

    @property
    def is_parameterized(self) -> bool:
        return self.param_index is not None

    def resolve_angle(self, params: NDArray[np.float64]) -> float:
        """Return this gate's rotation angle for a given parameter vector."""
        if self.angle is not None:
            return self.angle
        if self.param_index is None:
            raise ValueError(f"{self.name} is not a rotation gate")
        return float(params[self.param_index]) * self.param_scale


@dataclass(frozen=True, slots=True)
class Circuit:
    """An ordered gate list over ``n_qubits``, with ``n_params`` free parameters."""

    n_qubits: int
    gates: tuple[Gate, ...]
    n_params: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_qubits <= 0:
            raise ValueError(f"n_qubits must be positive, got {self.n_qubits}")
        for gate in self.gates:
            if any(q >= self.n_qubits for q in gate.qubits):
                raise ValueError(
                    f"{gate.name} acts on qubit {max(gate.qubits)}, "
                    f"outside the {self.n_qubits}-qubit register"
                )
            if gate.param_index is not None and gate.param_index >= self.n_params:
                raise ValueError(
                    f"{gate.name} references parameter {gate.param_index}, "
                    f"but the circuit declares only {self.n_params}"
                )

    @property
    def n_gates(self) -> int:
        return len(self.gates)

    @property
    def n_two_qubit_gates(self) -> int:
        """The cost metric that actually matters on hardware."""
        return sum(1 for g in self.gates if len(g.qubits) == 2)

    def bound_parameter_indices(self) -> frozenset[int]:
        return frozenset(g.param_index for g in self.gates if g.param_index is not None)


def statevector_index_order(n_qubits: int) -> str:
    """Describe the amplitude layout, for docs and error messages.

    Amplitude index ``i`` encodes qubit ``q``'s value in bit ``n_qubits - 1 - q``
    (qubit 0 is the most significant bit), which is what OpenFermion's
    ``get_sparse_operator`` produces.
    """
    return f"big-endian over {n_qubits} qubits: qubit 0 is the most significant bit"

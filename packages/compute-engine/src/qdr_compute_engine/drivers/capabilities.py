"""Per-driver capability flags.

Capabilities are declarative so the control plane can answer "can this config run on that
backend?" without instantiating a driver or touching a vendor SDK -- the backends module
in the API serves these to the frontend for capability gating.

``is_implemented`` exists so a driver can be *registered but honest*. A registered driver
that silently returns a plausible number when it has no implementation is far worse than
one that refuses; the contract tests assert the refusal.
"""

from dataclasses import dataclass

from qdr_core.limits import MAX_QUBITS


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    #: Largest register this backend can handle. Never above the platform cap.
    max_qubits: int

    #: True when energies are exact expectation values rather than shot estimates.
    is_exact: bool

    #: True when the backend accepts a finite shot budget.
    supports_shots: bool

    #: True when the backend needs a parameterized circuit (variational backends).
    requires_ansatz: bool

    #: True when the backend needs orbital-basis integrals (classical backends).
    requires_integrals: bool

    #: True when the backend needs vendor credentials to run at all.
    requires_credentials: bool = False

    #: False for drivers that are registered but not yet built.
    is_implemented: bool = True

    #: Why the driver is unusable, when ``is_implemented`` is False.
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.max_qubits > MAX_QUBITS:
            raise ValueError(
                f"backend declares max_qubits={self.max_qubits}, above the platform "
                f"cap of {MAX_QUBITS}; the cap is an invariant, not a default"
            )
        if self.max_qubits <= 0:
            raise ValueError(f"max_qubits must be positive, got {self.max_qubits}")
        if not self.is_implemented and not self.unavailable_reason:
            raise ValueError("an unimplemented backend must state why it is unavailable")

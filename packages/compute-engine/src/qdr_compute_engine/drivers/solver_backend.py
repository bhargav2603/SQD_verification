"""THE interface (ADR-0003).

A new backend implements this one class and registers itself. Nothing in the pipeline,
the mappers, the ansaetze or the validator changes to accommodate it -- that is the whole
point of the seam, and it only survives if implementations stay disciplined about two
rules:

  1. Consume the :class:`~qdr_compute_engine.types.Circuit` IR. Never require a vendor
     circuit type in a signature.
  2. Fail loudly. A backend that cannot serve a request raises ``BackendUnavailable``;
     it never returns an approximate answer while pretending it is the requested one.

See ``drivers/README.md`` for the add-a-backend checklist.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

from qdr_compute_engine.drivers.capabilities import BackendCapabilities
from qdr_compute_engine.types import SolveRequest, SolveResult
from qdr_core.errors import BackendUnavailable
from qdr_core.limits import MAX_QUBITS


class SolverBackend(ABC):
    """Computes a ground-state energy for a :class:`SolveRequest`."""

    #: Registry key. Must be unique and stable -- it is persisted on run records.
    name: ClassVar[str]

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Declarative description of what this backend can do."""

    @abstractmethod
    def solve(self, request: SolveRequest) -> SolveResult:
        """Compute the ground-state energy.

        Implementations must call :meth:`check_supported` first.
        """

    def check_supported(self, request: SolveRequest) -> None:
        """Raise ``BackendUnavailable`` if this backend cannot serve ``request``.

        Centralising the common preconditions here means every backend enforces the
        qubit cap and the ansatz/integral requirements identically, including backends
        written later by someone who never read this file.
        """
        caps = self.capabilities

        if not caps.is_implemented:
            raise BackendUnavailable(self.name, caps.unavailable_reason or "not implemented")

        if request.n_qubits > MAX_QUBITS:
            raise BackendUnavailable(
                self.name,
                f"request needs {request.n_qubits} qubits, above the platform cap of {MAX_QUBITS}",
                required_qubits=request.n_qubits,
                max_qubits=MAX_QUBITS,
            )

        if request.n_qubits > caps.max_qubits:
            raise BackendUnavailable(
                self.name,
                f"request needs {request.n_qubits} qubits, above this backend's limit "
                f"of {caps.max_qubits}",
                required_qubits=request.n_qubits,
                backend_max_qubits=caps.max_qubits,
            )

        if caps.requires_ansatz and request.ansatz is None:
            raise BackendUnavailable(self.name, "request carries no ansatz circuit")

        if caps.requires_integrals and request.integrals is None:
            raise BackendUnavailable(self.name, "request carries no orbital-basis integrals")

        if request.shots is not None and not caps.supports_shots:
            raise BackendUnavailable(
                self.name,
                f"a shot budget was requested ({request.shots}) but this backend is shot-free",
            )

        if request.ansatz is not None and request.ansatz.n_qubits != request.n_qubits:
            raise BackendUnavailable(
                self.name,
                f"ansatz spans {request.ansatz.n_qubits} qubits but the Hamiltonian "
                f"spans {request.n_qubits}",
            )

    def supports(self, request: SolveRequest) -> bool:
        """Non-raising form of :meth:`check_supported`, for capability queries."""
        try:
            self.check_supported(request)
        except BackendUnavailable:
            return False
        return True

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"

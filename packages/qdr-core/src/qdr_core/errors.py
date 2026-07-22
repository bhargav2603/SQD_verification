"""Shared error taxonomy.

Every error carries a stable, dotted ``code``. The control plane maps codes to HTTP
responses and the frontend switches on them, so a code is part of the public contract:
rename one and you break clients.
"""

from typing import Any


class QdrError(Exception):
    """Base for every deliberate platform error.

    ``details`` holds structured, machine-readable context (never free-form prose) so a
    caller can render its own message without re-parsing ``str(exc)``.
    """

    code: str = "qdr.internal_error"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


class ValidationError(QdrError):
    """Input failed a structural or physical validity check."""

    code = "qdr.validation_error"


class QubitCapExceeded(QdrError):
    """A job would require more qubits than the platform will schedule.

    Raised at every enforcement point that reads :mod:`qdr_core.limits`. Carries both
    numbers so the caller can tell the user how far over the line they are.
    """

    code = "qdr.qubit_cap_exceeded"

    def __init__(self, required_qubits: int, max_qubits: int, **details: Any) -> None:
        super().__init__(
            f"Configuration requires {required_qubits} qubits, "
            f"which exceeds the platform cap of {max_qubits}.",
            required_qubits=required_qubits,
            max_qubits=max_qubits,
            **details,
        )
        self.required_qubits = required_qubits
        self.max_qubits = max_qubits


class BackendUnavailable(QdrError):
    """A solver backend exists in the registry but cannot service this request.

    Covers unimplemented drivers, missing vendor credentials, and capability mismatches.
    Distinct from a lookup miss: an unknown backend name is a ``ValidationError``.
    """

    code = "qdr.backend_unavailable"

    def __init__(self, backend: str, reason: str, **details: Any) -> None:
        super().__init__(
            f"Backend {backend!r} is unavailable: {reason}",
            backend=backend,
            reason=reason,
            **details,
        )
        self.backend = backend
        self.reason = reason


class ConvergenceError(QdrError):
    """A variational optimization terminated without meeting its convergence criteria."""

    code = "qdr.convergence_error"

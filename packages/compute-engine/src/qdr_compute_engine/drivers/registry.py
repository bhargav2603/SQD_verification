"""Backend discovery and construction.

Registration is by decorator at import time, and the built-in drivers are imported by
``drivers/__init__.py``. This module must never import the ``drivers`` package itself --
that would close an import cycle, since every driver module imports this one.
"""

from qdr_compute_engine.drivers.capabilities import BackendCapabilities
from qdr_compute_engine.drivers.solver_backend import SolverBackend
from qdr_core.errors import ValidationError

_REGISTRY: dict[str, type[SolverBackend]] = {}


def register_backend(cls: type[SolverBackend]) -> type[SolverBackend]:
    """Class decorator that adds a backend to the registry."""
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} must define a non-empty class-level `name`")
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(
            f"backend name {name!r} is already registered to {_REGISTRY[name].__name__}; "
            "names are persisted on run records and must be unique"
        )
    _REGISTRY[name] = cls
    return cls


def get_backend(name: str) -> SolverBackend:
    """Instantiate a registered backend by name.

    An unknown name is a ``ValidationError`` (the caller asked for something that does
    not exist), whereas a known-but-unusable backend raises ``BackendUnavailable`` later,
    from ``check_supported``. Keeping those distinct matters: one is a client bug, the
    other is a capacity or configuration condition.
    """
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValidationError(
            f"unknown solver backend {name!r}",
            requested=name,
            available=sorted(_REGISTRY),
        ) from None
    return cls()


def available_backends() -> list[str]:
    """All registered backend names, including unimplemented ones."""
    return sorted(_REGISTRY)


def implemented_backends() -> list[str]:
    """Only backends that can actually run."""
    return sorted(n for n, cls in _REGISTRY.items() if cls().capabilities.is_implemented)


def backend_capabilities() -> dict[str, BackendCapabilities]:
    """Capability table, as served to the control plane for capability gating."""
    return {name: cls().capabilities for name, cls in sorted(_REGISTRY.items())}


def is_registered(name: str) -> bool:
    return name in _REGISTRY

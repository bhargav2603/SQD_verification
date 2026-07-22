"""Solver backends (ADR-0003).

Importing this package registers every built-in driver. The imports below look unused --
they are not: each module applies ``@register_backend`` at import time, and dropping one
silently removes that backend from the registry.
"""

from qdr_compute_engine.drivers.capabilities import BackendCapabilities
from qdr_compute_engine.drivers.classical.classical_driver import ClassicalDriver
from qdr_compute_engine.drivers.qpu.qpu_driver import QpuDriver
from qdr_compute_engine.drivers.registry import (
    available_backends,
    backend_capabilities,
    get_backend,
    implemented_backends,
    is_registered,
    register_backend,
)
from qdr_compute_engine.drivers.sampled.sampled_driver import SampledDriver
from qdr_compute_engine.drivers.solver_backend import SolverBackend
from qdr_compute_engine.drivers.statevector.statevector_driver import StatevectorDriver

DEFAULT_BACKEND = StatevectorDriver.name

__all__ = [
    "DEFAULT_BACKEND",
    "BackendCapabilities",
    "ClassicalDriver",
    "QpuDriver",
    "SampledDriver",
    "SolverBackend",
    "StatevectorDriver",
    "available_backends",
    "backend_capabilities",
    "get_backend",
    "implemented_backends",
    "is_registered",
    "register_backend",
]

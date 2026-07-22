"""Vendor QPU backend (feature-flagged).

Registered but not implemented.

This module is the *only* place in the engine where importing a vendor SDK is legal --
that is the whole purpose of the driver seam (ADR-0003). Everything upstream speaks the
gate IR, so adding hardware support means writing a transpiler from
:class:`~qdr_compute_engine.types.Circuit` to the vendor's circuit type here, and
changing nothing else.

Job submission itself belongs to ``services/qpu-broker``, which owns credentials,
calibration windows and queue management. This driver's eventual job is to translate the
circuit and delegate, not to hold vendor tokens inside a compute worker.
"""

from qdr_compute_engine.drivers.capabilities import BackendCapabilities
from qdr_compute_engine.drivers.registry import register_backend
from qdr_compute_engine.drivers.solver_backend import SolverBackend
from qdr_compute_engine.types import SolveRequest, SolveResult
from qdr_core.errors import BackendUnavailable


@register_backend
class QpuDriver(SolverBackend):
    name = "qpu"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            # Deliberately below the platform cap: no available device offers 40
            # usable qubits at the fidelity this pipeline needs.
            max_qubits=20,
            is_exact=False,
            supports_shots=True,
            requires_ansatz=True,
            requires_integrals=False,
            requires_credentials=True,
            is_implemented=False,
            unavailable_reason=(
                "no vendor adapter is wired up; QPU execution is brokered by "
                "services/qpu-broker and gated behind a capability flag"
            ),
        )

    def solve(self, request: SolveRequest) -> SolveResult:
        self.check_supported(request)  # always raises, given is_implemented=False
        raise BackendUnavailable(self.name, "not implemented")

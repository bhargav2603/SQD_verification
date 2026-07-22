"""Shot-based sampled simulation.

Registered but not implemented.

The reason this is not a thin wrapper over the statevector driver: a faithful sampled
backend has to group the Hamiltonian's Pauli terms into commuting sets, allocate the shot
budget across those groups, and propagate the resulting variance into the reported
standard error. Skipping that and simply adding Gaussian noise to an exact expectation
value would produce a number that looks like a shot-based estimate, carries an error bar
that is not the real one, and would quietly mislead every downstream comparison against
QPU results. Refusing is the honest behaviour until the grouping exists.
"""

from qdr_compute_engine.drivers.capabilities import BackendCapabilities
from qdr_compute_engine.drivers.registry import register_backend
from qdr_compute_engine.drivers.solver_backend import SolverBackend
from qdr_compute_engine.types import SolveRequest, SolveResult
from qdr_core.errors import BackendUnavailable


@register_backend
class SampledDriver(SolverBackend):
    name = "sampled"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            max_qubits=24,
            is_exact=False,
            supports_shots=True,
            requires_ansatz=True,
            requires_integrals=False,
            is_implemented=False,
            unavailable_reason=(
                "shot-based estimation needs commuting-set grouping and shot allocation, "
                "which are not implemented yet"
            ),
        )

    def solve(self, request: SolveRequest) -> SolveResult:
        self.check_supported(request)  # always raises, given is_implemented=False
        raise BackendUnavailable(self.name, "not implemented")

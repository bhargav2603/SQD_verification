"""Classical reference backend (FCI via PySCF).

Consumes orbital-basis integrals rather than the circuit IR -- it has no ansatz and no
qubits. It still implements the same :class:`SolverBackend` interface so the pipeline can
request a classical reference through exactly the same path it uses for a quantum one,
which is what makes side-by-side comparison in the results dashboard straightforward.
"""

from qdr_compute_engine.drivers.capabilities import BackendCapabilities
from qdr_compute_engine.drivers.registry import register_backend
from qdr_compute_engine.drivers.solver_backend import SolverBackend
from qdr_compute_engine.solvers.classical_baseline import solve_fci
from qdr_compute_engine.types import SolveRequest, SolveResult
from qdr_core.errors import BackendUnavailable


@register_backend
class ClassicalDriver(SolverBackend):
    name = "classical"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            # FCI cost grows combinatorially with the active space; beyond ~16 spin
            # orbitals the determinant space is impractical regardless of the qubit cap.
            max_qubits=16,
            is_exact=True,
            supports_shots=False,
            requires_ansatz=False,
            requires_integrals=True,
        )

    def solve(self, request: SolveRequest) -> SolveResult:
        self.check_supported(request)
        integrals = request.integrals
        assert integrals is not None  # guaranteed by check_supported

        method = str(request.options.get("classical_method", "fci"))
        if method != "fci":
            raise BackendUnavailable(
                self.name,
                f"classical method {method!r} is not implemented; only 'fci' is available",
            )

        outcome = solve_fci(integrals)

        return SolveResult(
            energy=outcome.energy,
            backend=self.name,
            method=outcome.method,
            converged=outcome.converged,
            n_iterations=0,
            n_energy_evaluations=1,
            optimal_params=None,
            std_error=None,
            metadata={
                "n_orbitals": integrals.n_orbitals,
                "n_electrons": integrals.n_electrons,
                "is_exact_in_basis": True,
            },
        )

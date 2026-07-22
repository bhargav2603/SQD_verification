"""Stage 8 -- solve for the ground-state energy through the driver seam."""

import numpy as np

from qdr_compute_engine.ansatze.uccsd import initial_parameters
from qdr_compute_engine.drivers import DEFAULT_BACKEND, get_backend
from qdr_compute_engine.pipeline.stage_context import StageContext
from qdr_compute_engine.types import SolveRequest

STAGE_NAME = "08_solver"


def run(context: StageContext) -> StageContext:
    config = context.config
    backend = get_backend(str(config.get("backend", DEFAULT_BACKEND)))

    ansatz = context.ansatz
    initial = None
    if ansatz is not None:
        initial = initial_parameters(
            ansatz,
            seed=context.seed,
            spread=float(config.get("initial_param_spread", 0.0)),
        )

    shots = config.get("shots")
    request = SolveRequest(
        hamiltonian=context.require_qubit_hamiltonian(),
        active_space=context.require_active_space(),
        ansatz=ansatz,
        initial_params=initial,
        integrals=context.integrals,
        shots=int(shots) if shots is not None else None,
        seed=context.seed,
        max_iterations=int(config.get("max_iterations", 1000)),
        options=dict(config.get("solver_options", {})),
    )

    result = backend.solve(request)

    # The optimiser's energy history feeds the error budget in stage 9, but SolveResult
    # is backend-agnostic and does not carry it as a field; it travels in metadata.
    if isinstance(result.optimal_params, np.ndarray):
        result.metadata.setdefault("n_params", int(result.optimal_params.size))

    return context.evolve(solve_result=result)

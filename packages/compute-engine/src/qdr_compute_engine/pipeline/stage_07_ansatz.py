"""Stage 7 -- ansatz construction.

Skipped for backends that declare ``requires_ansatz=False`` (the classical driver has no
circuit), so a purely classical run does not pay to build one it will never use.
"""

from qdr_compute_engine.ansatze import DEFAULT_ANSATZ, get_ansatz_builder
from qdr_compute_engine.drivers import DEFAULT_BACKEND, get_backend
from qdr_compute_engine.pipeline.stage_context import StageContext

STAGE_NAME = "07_ansatz"


def run(context: StageContext) -> StageContext:
    backend = get_backend(str(context.config.get("backend", DEFAULT_BACKEND)))
    if not backend.capabilities.requires_ansatz:
        return context

    active_space = context.require_active_space()
    builder = get_ansatz_builder(str(context.config.get("ansatz", DEFAULT_ANSATZ)))

    circuit = builder(
        n_electrons=active_space.n_active_electrons,
        n_spin_orbitals=active_space.n_spin_orbitals,
    )
    return context.evolve(ansatz=circuit)

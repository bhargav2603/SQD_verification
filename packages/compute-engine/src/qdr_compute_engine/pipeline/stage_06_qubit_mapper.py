"""Stage 6 -- fermion-to-qubit mapping."""

from qdr_compute_engine.mappers import DEFAULT_MAPPER, get_mapper
from qdr_compute_engine.pipeline.stage_context import StageContext

STAGE_NAME = "06_qubit_mapper"


def run(context: StageContext) -> StageContext:
    mapper = get_mapper(str(context.config.get("mapping", DEFAULT_MAPPER)))
    qubit_hamiltonian = mapper(context.require_fermionic_hamiltonian())
    return context.evolve(qubit_hamiltonian=qubit_hamiltonian)

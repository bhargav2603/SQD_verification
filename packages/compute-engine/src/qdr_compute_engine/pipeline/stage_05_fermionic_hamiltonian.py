"""Stage 5 -- second-quantised Hamiltonian."""

from qdr_compute_engine.hamiltonian.fermionic_builder import build_fermionic_hamiltonian
from qdr_compute_engine.pipeline.stage_context import StageContext

STAGE_NAME = "05_fermionic_hamiltonian"


def run(context: StageContext) -> StageContext:
    hamiltonian = build_fermionic_hamiltonian(
        context.require_integrals(),
        context.require_active_space(),
    )
    return context.evolve(fermionic_hamiltonian=hamiltonian)

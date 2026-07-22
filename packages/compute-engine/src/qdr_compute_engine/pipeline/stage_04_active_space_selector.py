"""Stage 4 -- active-space selection and the engine-side qubit-cap check.

Note that this stage *replaces* ``context.integrals`` with the active-space projection.
Everything downstream works exclusively in the active space, so carrying the full-basis
integrals forward would only invite a stage to pick the wrong one; the frozen-core energy
is folded into ``e_nuc`` so total energies stay correct.
"""

from qdr_compute_engine.active_space.active_space_selector import (
    restrict_integrals,
    select_active_space,
)
from qdr_compute_engine.pipeline.stage_context import StageContext

STAGE_NAME = "04_active_space_selector"


def run(context: StageContext) -> StageContext:
    config = context.config
    integrals = context.require_integrals()

    n_active_orbitals = config.get("n_active_orbitals")
    n_active_electrons = config.get("n_active_electrons")

    active_space = select_active_space(
        integrals,
        n_active_orbitals=int(n_active_orbitals) if n_active_orbitals is not None else None,
        n_active_electrons=int(n_active_electrons) if n_active_electrons is not None else None,
    )

    return context.evolve(
        active_space=active_space,
        integrals=restrict_integrals(integrals, active_space),
    )

"""Stage 3 -- molecular integrals (PySCF).

The only stage that requires PySCF. A context that already carries integrals (a test
fixture, or a cached artifact) skips the calculation entirely, which is what lets stages
4-9 be exercised on platforms with no PySCF wheel.
"""

from qdr_compute_engine.integrals.pyscf_integrals import DEFAULT_BASIS, compute_integrals
from qdr_compute_engine.pipeline.stage_context import StageContext

STAGE_NAME = "03_integrals"


def run(context: StageContext) -> StageContext:
    if context.integrals is not None:
        return context

    integrals = compute_integrals(
        context.require_geometry(),
        basis=str(context.config.get("basis", DEFAULT_BASIS)),
    )
    return context.evolve(integrals=integrals)

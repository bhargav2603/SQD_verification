"""Stage 2 -- FMO fragmentation."""

from qdr_compute_engine.fragmentation.fmo_fragmenter import fragment
from qdr_compute_engine.pipeline.stage_context import StageContext

STAGE_NAME = "02_fmo_fragmenter"


def run(context: StageContext) -> StageContext:
    config = context.config
    cut_bonds = tuple((int(a), int(b)) for a, b in config.get("cut_bonds", ()))
    cutoff = config.get("dimer_cutoff_angstrom")

    result = fragment(
        context.require_geometry(),
        scheme=str(config.get("fragmentation_scheme", "molecular")),
        cut_bonds=cut_bonds,
        dimer_cutoff_angstrom=float(cutoff) if cutoff is not None else None,
    )

    context = context.evolve(fragmentation=result)

    # The linear stage chain (pipeline.STAGES) computes the supersystem: stages 3-9 take
    # the whole geometry regardless of how it was fragmented. The fragment expansion is a
    # different traversal of the same stages and lives in `pipeline/fmo_runner.py`.
    # Reaching here with multiple fragments means the supersystem path was chosen, so say
    # which number is about to be produced rather than letting the fragmentation above
    # imply an FMO result.
    if result.n_monomers > 1:
        context = context.with_warning(
            f"fragmentation found {result.n_monomers} monomers, but this is the "
            "supersystem chain: stages 3-9 will compute the whole system as one. For the "
            "two-body fragment expansion use pipeline.fmo_runner.run() instead."
        )

    return context

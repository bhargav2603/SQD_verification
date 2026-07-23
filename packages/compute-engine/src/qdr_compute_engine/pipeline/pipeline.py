"""Pipeline orchestration: stages 1 -> 9.

Stage 0 is deliberately *not* in this chain. It is a correctness gate on the engine
itself, not a step in computing a user's result -- see ``stage_00_fci_gate``.

Each stage is a plain ``(StageContext) -> StageContext`` function, so the pipeline is an
ordered list and nothing more. That flatness is the point: a reader can see the whole
computation in one screen, and a stage can be run in isolation in a test without
constructing an orchestrator.

Failures propagate. A stage that cannot do its job raises, and the pipeline attaches
which stage failed rather than swallowing it -- a partial result in this domain is
indistinguishable from a wrong one.
"""

import logging
import time
from collections.abc import Callable

from qdr_compute_engine.pipeline import (
    stage_01_geometry_pose,
    stage_02_fmo_fragmenter,
    stage_03_integrals,
    stage_03b_embedding,
    stage_04_active_space_selector,
    stage_05_fermionic_hamiltonian,
    stage_06_qubit_mapper,
    stage_07_ansatz,
    stage_08_solver,
    stage_09_validator,
)
from qdr_compute_engine.pipeline.stage_context import StageContext
from qdr_compute_engine.reproducibility.input_hashing import (
    build_run_input_payload,
    hash_inputs,
)
from qdr_compute_engine.reproducibility.version_pins import capture_version_pins
from qdr_core.errors import QdrError
from qdr_core.ids import RunId, new_run_id

logger = logging.getLogger(__name__)

StageFn = Callable[[StageContext], StageContext]

#: The ordered pipeline. Order is the contract -- each stage consumes what its
#: predecessors produced.
STAGES: tuple[tuple[str, StageFn], ...] = (
    (stage_01_geometry_pose.STAGE_NAME, stage_01_geometry_pose.run),
    (stage_02_fmo_fragmenter.STAGE_NAME, stage_02_fmo_fragmenter.run),
    (stage_03_integrals.STAGE_NAME, stage_03_integrals.run),
    (stage_03b_embedding.STAGE_NAME, stage_03b_embedding.run),
    (stage_04_active_space_selector.STAGE_NAME, stage_04_active_space_selector.run),
    (stage_05_fermionic_hamiltonian.STAGE_NAME, stage_05_fermionic_hamiltonian.run),
    (stage_06_qubit_mapper.STAGE_NAME, stage_06_qubit_mapper.run),
    (stage_07_ansatz.STAGE_NAME, stage_07_ansatz.run),
    (stage_08_solver.STAGE_NAME, stage_08_solver.run),
    (stage_09_validator.STAGE_NAME, stage_09_validator.run),
)

#: Stages 3 -> 8: everything needed to turn *one* geometry into *one* energy.
#:
#: This is the supersystem chain with its ends removed. Geometry preparation (1) and
#: fragmentation (2) belong to the parent system and must not re-run per fragment;
#: validation (9) operates on the assembled result, not on individual subsystem energies.
#: ``pipeline/fmo_runner.py`` drives this once per monomer and once per dimer, so the
#: fragment path and the supersystem path run byte-identical physics.
#:
#: Embedding (03b) is deliberately *not* here: the FMO many-body expansion is a different
#: shrink mechanism (many fragments summed pairwise) from active-site projection embedding
#: (one region in a mean-field bath). R2's single-site benchmark runs the main ``STAGES``
#: path, so the FMO fragment path stays untouched.
SUBSYSTEM_STAGES: tuple[tuple[str, StageFn], ...] = (
    (stage_03_integrals.STAGE_NAME, stage_03_integrals.run),
    (stage_04_active_space_selector.STAGE_NAME, stage_04_active_space_selector.run),
    (stage_05_fermionic_hamiltonian.STAGE_NAME, stage_05_fermionic_hamiltonian.run),
    (stage_06_qubit_mapper.STAGE_NAME, stage_06_qubit_mapper.run),
    (stage_07_ansatz.STAGE_NAME, stage_07_ansatz.run),
    (stage_08_solver.STAGE_NAME, stage_08_solver.run),
)


class PipelineError(QdrError):
    """A stage failed. Carries which one, so the failure is attributable."""

    code = "qdr.pipeline_error"

    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__(
            f"pipeline failed at stage {stage}: {cause}",
            stage=stage,
            cause_type=type(cause).__name__,
        )
        self.stage = stage
        self.cause = cause


def build_context(
    config: dict[str, object],
    *,
    run_id: RunId | None = None,
    seed: int = 0,
    **initial_fields: object,
) -> StageContext:
    """Create the starting context, with input hash and version pins recorded up front.

    Both are captured *before* any stage runs, so the provenance describes the inputs as
    submitted rather than as mutated along the way.
    """
    context = StageContext(
        run_id=run_id or new_run_id(),
        config=dict(config),
        seed=seed,
        version_pins=capture_version_pins(),
        **initial_fields,  # type: ignore[arg-type]
    )

    geometry = context.geometry
    payload = build_run_input_payload(
        geometry=geometry.to_dict() if geometry is not None else {},
        config=dict(config),
        basis=str(config.get("basis", "sto-3g")),
        backend=str(config.get("backend", "statevector")),
        mapping=str(config.get("mapping", "jordan_wigner")),
        seed=seed,
    )
    return context.evolve(input_hash=hash_inputs(payload))


def run_pipeline(
    context: StageContext,
    stages: tuple[tuple[str, StageFn], ...] = STAGES,
) -> StageContext:
    """Run the stages in order, timing each."""
    for stage_name, stage_fn in stages:
        started = time.perf_counter()
        try:
            context = stage_fn(context)
        except Exception as exc:
            logger.exception("stage %s failed for run %s", stage_name, context.run_id)
            raise PipelineError(stage_name, exc) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        context = context.with_timing(stage_name, elapsed_ms)
        logger.info(
            "stage %s completed in %.1f ms (run %s)", stage_name, elapsed_ms, context.run_id
        )

    return context


def run(
    config: dict[str, object],
    *,
    run_id: RunId | None = None,
    seed: int = 0,
    **initial_fields: object,
) -> StageContext:
    """Build a context and run the full pipeline. The usual entry point."""
    return run_pipeline(build_context(config, run_id=run_id, seed=seed, **initial_fields))

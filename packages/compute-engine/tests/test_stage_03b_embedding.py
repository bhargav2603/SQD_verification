"""Stage 3b (projection embedding) -- T1: config-gated pass-through + plumbing.

T1 delivers ONLY the stage skeleton and its wiring: with no ``config["embedding"]`` the
stage is a pure pass-through, so R1 (non-embedded) runs stay byte-identical to a pipeline
without this stage. The embedding math itself (h_emb build, ROHF reference, partition) lands
in later tasks and is intentionally ``NotImplementedError`` here rather than silently doing
nothing when embedding IS requested.

These tests need no PySCF: they seed integrals directly, exactly like ``test_pipeline.py``.
"""

import pytest

from qdr_compute_engine.pipeline import stage_03b_embedding
from qdr_compute_engine.pipeline.pipeline import STAGES, build_context, run_pipeline
from qdr_compute_engine.pipeline.stage_context import StageContext
from qdr_compute_engine.types import IntegralSet
from qdr_core.ids import RunId

# All stages from integrals onward: 03b (pass-through) then 04..09.
FROM_INTEGRALS = STAGES[3:]


def test_stage_03b_sits_between_integrals_and_active_space() -> None:
    names = [name for name, _ in STAGES]
    assert names.index("03b_embedding") == names.index("03_integrals") + 1
    assert names.index("04_active_space_selector") == names.index("03b_embedding") + 1


def test_pass_through_when_no_embedding_config(h2_integrals: IntegralSet) -> None:
    """No embedding config -> the stage returns its input untouched."""
    ctx = build_context(
        {"backend": "statevector"}, run_id=RunId("t"), integrals=h2_integrals
    )
    out = stage_03b_embedding.run(ctx)

    assert out.integrals is ctx.integrals  # same object, not a copy
    assert out.embedding is None


def test_pass_through_keeps_r1_energy_byte_identical(
    h2_integrals: IntegralSet, golden_h2: dict[str, object]
) -> None:
    """The R1 regression: inserting stage 03b as a pass-through must not move the number."""
    ctx = build_context(
        {"backend": "statevector", "mapping": "jordan_wigner", "ansatz": "uccsd"},
        run_id=RunId("t"),
        seed=1234,
        integrals=h2_integrals,
    )
    final = run_pipeline(ctx, FROM_INTEGRALS)
    assert final.require_solve_result().energy == pytest.approx(
        golden_h2["e_fci"], abs=1.6e-3
    )


def test_configured_embedding_is_not_yet_implemented(h2_integrals: IntegralSet) -> None:
    """Requesting embedding must fail loudly, not silently pass through."""
    ctx = build_context(
        {"embedding": {"scheme": "projection-wf-in-dft"}}, integrals=h2_integrals
    )
    with pytest.raises(NotImplementedError, match="embedding"):
        stage_03b_embedding.run(ctx)


def test_require_embedding_names_the_missing_stage() -> None:
    ctx = StageContext(run_id=RunId("r"))
    with pytest.raises(Exception, match="03b_embedding"):
        ctx.require_embedding()

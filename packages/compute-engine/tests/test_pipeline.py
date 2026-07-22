"""End-to-end pipeline orchestration.

Runs stages 4 -> 9 by seeding the context with golden-file integrals, which is what lets
the whole orchestration path be exercised without PySCF: stage 3 short-circuits when
integrals are already present.
"""

from typing import Any

import numpy as np
import pytest

from qdr_compute_engine.pipeline import pipeline as pipeline_module
from qdr_compute_engine.pipeline.pipeline import (
    STAGES,
    PipelineError,
    build_context,
    run_pipeline,
)
from qdr_compute_engine.pipeline.stage_context import StageContext
from qdr_compute_engine.types import IntegralSet, ValidationReport
from qdr_core.errors import QubitCapExceeded
from qdr_core.ids import RunId

# Stages 4 onwards: everything after integrals are available.
POST_INTEGRAL_STAGES = STAGES[3:]


@pytest.fixture
def seeded_context(h2_integrals: IntegralSet) -> StageContext:
    return build_context(
        {"backend": "statevector", "mapping": "jordan_wigner", "ansatz": "uccsd"},
        run_id=RunId("test-run"),
        seed=1234,
        integrals=h2_integrals,
    )


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def test_pipeline_runs_to_completion(
    seeded_context: StageContext, golden_h2: dict[str, Any]
) -> None:
    final = run_pipeline(seeded_context, POST_INTEGRAL_STAGES)

    assert final.active_space is not None
    assert final.qubit_hamiltonian is not None
    assert final.ansatz is not None
    assert final.solve_result is not None
    assert isinstance(final.validation, ValidationReport)
    assert final.solve_result.energy == pytest.approx(golden_h2["e_fci"], abs=1.6e-3)


def test_every_stage_is_timed(seeded_context: StageContext) -> None:
    final = run_pipeline(seeded_context, POST_INTEGRAL_STAGES)
    assert set(final.stage_timings_ms) == {name for name, _ in POST_INTEGRAL_STAGES}
    assert all(v >= 0.0 for v in final.stage_timings_ms.values())


def test_stage_four_replaces_integrals_with_the_active_space_projection(
    seeded_context: StageContext,
) -> None:
    final = run_pipeline(seeded_context, POST_INTEGRAL_STAGES[:1])
    assert final.integrals is not None
    assert final.integrals.n_orbitals == final.require_active_space().n_active_orbitals


def test_failure_reports_which_stage_broke(seeded_context: StageContext) -> None:
    """A pipeline that swallows the stage name makes production failures unattributable."""
    context = seeded_context.evolve(config={**seeded_context.config, "mapping": "parity"})

    with pytest.raises(PipelineError) as exc_info:
        run_pipeline(context, POST_INTEGRAL_STAGES)

    assert exc_info.value.stage == "06_qubit_mapper"
    assert isinstance(exc_info.value.cause, NotImplementedError)


def test_cap_violation_propagates_from_stage_four() -> None:
    # Needs a basis large enough that 21 orbitals is available to select -- otherwise the
    # basis-size check fires first and the cap is never reached.
    n_orbitals = 25
    h1 = np.eye(n_orbitals)
    large_basis = IntegralSet(
        h1=h1,
        h2=np.zeros((n_orbitals,) * 4),
        e_nuc=0.0,
        n_electrons=2,
    )
    context = build_context(
        {"n_active_orbitals": 21, "n_active_electrons": 2},
        integrals=large_basis,
    )
    with pytest.raises(PipelineError) as exc_info:
        run_pipeline(context, POST_INTEGRAL_STAGES)

    assert exc_info.value.stage == "04_active_space_selector"
    # The cap error itself must survive, not be flattened into a generic failure.
    assert isinstance(exc_info.value.cause, QubitCapExceeded)


# --------------------------------------------------------------------------------------
# Context behaviour
# --------------------------------------------------------------------------------------


def test_context_is_immutable_across_stages(seeded_context: StageContext) -> None:
    """A stage returns a new context rather than mutating its input."""
    before = seeded_context
    after = run_pipeline(before, POST_INTEGRAL_STAGES[:1])
    assert before.active_space is None
    assert after.active_space is not None
    assert after is not before


def test_requiring_an_unset_field_names_the_missing_stage() -> None:
    context = StageContext(run_id=RunId("r"))
    with pytest.raises(Exception, match="04_active_space_selector"):
        context.require_active_space()


def test_input_hash_is_recorded_up_front(seeded_context: StageContext) -> None:
    assert seeded_context.input_hash is not None
    assert seeded_context.input_hash.startswith("sha256:")


def test_version_pins_are_captured(seeded_context: StageContext) -> None:
    pins = seeded_context.version_pins
    assert pins["numpy"] == np.__version__
    # Absent optional dependencies are recorded explicitly, not omitted.
    assert "pyscf" in pins


# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------


def test_identical_inputs_hash_identically(h2_integrals: IntegralSet) -> None:
    """Two runs of the same science must be recognisably the same run."""
    config = {"backend": "statevector", "mapping": "jordan_wigner"}
    first = build_context(config, run_id=RunId("run-a"), seed=7, integrals=h2_integrals)
    second = build_context(config, run_id=RunId("run-b"), seed=7, integrals=h2_integrals)

    # Different run ids, same inputs -> same hash. The run id is identity, not input.
    assert first.input_hash == second.input_hash


def test_a_different_seed_changes_the_input_hash(h2_integrals: IntegralSet) -> None:
    config = {"backend": "statevector"}
    first = build_context(config, seed=1, integrals=h2_integrals)
    second = build_context(config, seed=2, integrals=h2_integrals)
    assert first.input_hash != second.input_hash


def test_repeated_runs_produce_the_same_energy(seeded_context: StageContext) -> None:
    first = run_pipeline(seeded_context, POST_INTEGRAL_STAGES)
    second = run_pipeline(seeded_context, POST_INTEGRAL_STAGES)
    assert first.require_solve_result().energy == second.require_solve_result().energy


# --------------------------------------------------------------------------------------
# Validation output
# --------------------------------------------------------------------------------------


def test_validator_quantifies_ansatz_error(seeded_context: StageContext) -> None:
    """For a 4-qubit system the exact ground state is computable, so the ansatz error is
    measured rather than guessed."""
    final = run_pipeline(seeded_context, POST_INTEGRAL_STAGES)
    report = final.require_validation()

    assert "ansatz" in report.error_budget.contributions
    assert report.error_budget.contributions["ansatz"] < 1.6e-3


def test_validator_warns_when_binding_energy_references_are_absent(
    seeded_context: StageContext,
) -> None:
    """Without isolated-fragment energies there is no binding energy to report, and the
    report must say so rather than passing off a total energy as one."""
    final = run_pipeline(seeded_context, POST_INTEGRAL_STAGES)
    report = final.require_validation()

    assert any("reference energies" in w for w in report.warnings)
    assert report.binding_energy_ha == final.require_solve_result().energy


def test_validator_warns_that_active_space_error_is_unquantified(
    seeded_context: StageContext,
) -> None:
    final = run_pipeline(seeded_context, POST_INTEGRAL_STAGES)
    assert any("active-space" in w for w in final.require_validation().warnings)


def test_binding_energy_is_computed_when_references_are_supplied(
    h2_integrals: IntegralSet,
) -> None:
    context = build_context(
        {
            "backend": "statevector",
            "reference_energy_protein": -0.5,
            "reference_energy_ligand": -0.6,
        },
        integrals=h2_integrals,
    )
    final = run_pipeline(context, POST_INTEGRAL_STAGES)
    report = final.require_validation()

    expected = final.require_solve_result().energy - (-0.5 + -0.6)
    assert report.binding_energy_ha == pytest.approx(expected)
    assert report.binding_energy_kcal_per_mol == pytest.approx(expected * 627.5094740631)


# --------------------------------------------------------------------------------------
# Stage list integrity
# --------------------------------------------------------------------------------------


def test_pipeline_has_nine_stages_in_order() -> None:
    names = [name for name, _ in STAGES]
    assert names == [
        "01_geometry_pose",
        "02_fmo_fragmenter",
        "03_integrals",
        "04_active_space_selector",
        "05_fermionic_hamiltonian",
        "06_qubit_mapper",
        "07_ansatz",
        "08_solver",
        "09_validator",
    ]


def test_stage_zero_is_not_part_of_the_chain() -> None:
    """The FCI gate is a check on the engine, not a step in a user's calculation."""
    assert not any(name.startswith("00") for name, _ in STAGES)
    assert not hasattr(pipeline_module, "stage_00_fci_gate")

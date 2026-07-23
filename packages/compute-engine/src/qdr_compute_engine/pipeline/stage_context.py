"""The typed context threaded through pipeline stages 1 -> 9.

The context is immutable: each stage returns a new instance via :meth:`StageContext.evolve`
rather than mutating in place. That costs a shallow copy per stage and buys two things
worth more than the copy -- a stage cannot corrupt its predecessor's output, and the
inputs to any stage can be re-hashed after the fact for provenance.

Every downstream field starts as ``None``. Reading one through the ``require_*``
accessors turns "stage 4 never ran" into a precise error at the point of use, instead of
an ``AttributeError`` on ``None`` several stages later.
"""

from dataclasses import dataclass, field, replace
from typing import Any, TypeVar

from qdr_compute_engine.types import (
    ActiveSpace,
    Circuit,
    EmbeddingResult,
    FermionicHamiltonian,
    FragmentationResult,
    Geometry,
    IntegralSet,
    QubitHamiltonian,
    SolveResult,
    ValidationReport,
)
from qdr_core.errors import ValidationError
from qdr_core.ids import RunId
from qdr_core.provenance import ArtifactRef

T = TypeVar("T")


@dataclass(frozen=True, eq=False)
class StageContext:
    run_id: RunId
    config: dict[str, Any] = field(default_factory=dict)
    seed: int = 0

    # Progressively populated, one field per producing stage.
    geometry: Geometry | None = None  # stage 1
    fragmentation: FragmentationResult | None = None  # stage 2
    integrals: IntegralSet | None = None  # stage 3
    embedding: EmbeddingResult | None = None  # stage 3b (None on the non-embedded path)
    active_space: ActiveSpace | None = None  # stage 4
    fermionic_hamiltonian: FermionicHamiltonian | None = None  # stage 5
    qubit_hamiltonian: QubitHamiltonian | None = None  # stage 6
    ansatz: Circuit | None = None  # stage 7
    solve_result: SolveResult | None = None  # stage 8
    validation: ValidationReport | None = None  # stage 9

    # Provenance and observability, accumulated across stages.
    input_hash: str | None = None
    version_pins: dict[str, str] = field(default_factory=dict)
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    artifacts: tuple[ArtifactRef, ...] = ()
    warnings: tuple[str, ...] = ()

    def evolve(self, **changes: Any) -> "StageContext":
        """Return a copy with ``changes`` applied."""
        return replace(self, **changes)

    def with_timing(self, stage: str, elapsed_ms: float) -> "StageContext":
        return self.evolve(stage_timings_ms={**self.stage_timings_ms, stage: elapsed_ms})

    def with_warning(self, message: str) -> "StageContext":
        return self.evolve(warnings=(*self.warnings, message))

    def with_artifact(self, artifact: ArtifactRef) -> "StageContext":
        return self.evolve(artifacts=(*self.artifacts, artifact))

    # -- required accessors ------------------------------------------------------------
    #
    # Named for the stage that produces them, so the error says which stage is missing.

    def _require(self, value: T | None, field_name: str, producing_stage: str) -> T:
        if value is None:
            raise ValidationError(
                f"{field_name} is not set: stage {producing_stage} has not run",
                missing_field=field_name,
                producing_stage=producing_stage,
            )
        return value

    def require_geometry(self) -> Geometry:
        return self._require(self.geometry, "geometry", "01_geometry_pose")

    def require_fragmentation(self) -> FragmentationResult:
        return self._require(self.fragmentation, "fragmentation", "02_fmo_fragmenter")

    def require_integrals(self) -> IntegralSet:
        return self._require(self.integrals, "integrals", "03_integrals")

    def require_embedding(self) -> EmbeddingResult:
        return self._require(self.embedding, "embedding", "03b_embedding")

    def require_active_space(self) -> ActiveSpace:
        return self._require(self.active_space, "active_space", "04_active_space_selector")

    def require_fermionic_hamiltonian(self) -> FermionicHamiltonian:
        return self._require(
            self.fermionic_hamiltonian, "fermionic_hamiltonian", "05_fermionic_hamiltonian"
        )

    def require_qubit_hamiltonian(self) -> QubitHamiltonian:
        return self._require(self.qubit_hamiltonian, "qubit_hamiltonian", "06_qubit_mapper")

    def require_ansatz(self) -> Circuit:
        return self._require(self.ansatz, "ansatz", "07_ansatz")

    def require_solve_result(self) -> SolveResult:
        return self._require(self.solve_result, "solve_result", "08_solver")

    def require_validation(self) -> ValidationReport:
        return self._require(self.validation, "validation", "09_validator")

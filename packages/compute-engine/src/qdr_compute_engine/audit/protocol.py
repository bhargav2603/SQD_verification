"""Strict loading and execution of the locked H₂ sampler-audit pilot."""

from __future__ import annotations

import math
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from qdr_compute_engine.active_space.active_space_selector import select_active_space
from qdr_compute_engine.audit.analysis import compare_against_best_classical
from qdr_compute_engine.audit.artifacts import (
    AuditBundlePaths,
    AuditResult,
    scientific_result_hash,
    write_audit_bundle,
)
from qdr_compute_engine.audit.h2_pilot import h2_sto3g_integrals
from qdr_compute_engine.audit.models import (
    AuditBudget,
    AuditObservation,
    AuditProblem,
    DatasetSplit,
)
from qdr_compute_engine.audit.postprocessing import (
    CanonicalSectorCompletionPostprocessor,
)
from qdr_compute_engine.audit.report import build_audit_markdown, write_audit_figures
from qdr_compute_engine.audit.runner import run_audit_cell_with_batch
from qdr_compute_engine.audit.samplers import (
    ExactStateSampler,
    HartreeFockExcitationSampler,
    UniformSectorSampler,
)
from qdr_compute_engine.hamiltonian.fermionic_builder import (
    build_fermionic_hamiltonian,
)
from qdr_compute_engine.mappers.jordan_wigner import map_to_qubits
from qdr_compute_engine.solvers.sqd import sector_configurations, solve_sqd
from qdr_core.errors import ValidationError

_PROTOCOL_FIELDS = {
    "protocol_version",
    "title",
    "development_bond_lengths_angstrom",
    "held_out_bond_lengths_angstrom",
    "shot_budgets",
    "subspace_dimensions",
    "seeds",
    "confidence",
    "bootstrap_resamples",
    "minimum_pairs",
    "classical_rank_weights",
    "primary_estimand",
    "interpretation",
}


@dataclass(frozen=True, slots=True)
class AuditProtocol:
    protocol_version: int
    title: str
    development_bond_lengths_angstrom: tuple[float, ...]
    held_out_bond_lengths_angstrom: tuple[float, ...]
    shot_budgets: tuple[int, ...]
    subspace_dimensions: tuple[int, ...]
    seeds: tuple[int, ...]
    confidence: float
    bootstrap_resamples: int
    minimum_pairs: int
    classical_rank_weights: tuple[float, float, float]
    primary_estimand: str
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "title": self.title,
            "development_bond_lengths_angstrom": list(
                self.development_bond_lengths_angstrom
            ),
            "held_out_bond_lengths_angstrom": list(self.held_out_bond_lengths_angstrom),
            "shot_budgets": list(self.shot_budgets),
            "subspace_dimensions": list(self.subspace_dimensions),
            "seeds": list(self.seeds),
            "confidence": self.confidence,
            "bootstrap_resamples": self.bootstrap_resamples,
            "minimum_pairs": self.minimum_pairs,
            "classical_rank_weights": list(self.classical_rank_weights),
            "primary_estimand": self.primary_estimand,
            "interpretation": self.interpretation,
        }


@dataclass(frozen=True, slots=True)
class ProtocolRun:
    result: AuditResult
    bundle: AuditBundlePaths
    scientific_hash: str
    held_out_problem_ids: tuple[str, ...]
    figures: tuple[Path, ...]


def _positive_floats(values: Any, *, field: str) -> tuple[float, ...]:
    if not isinstance(values, list) or not values:
        raise ValidationError(f"{field} must be a non-empty JSON array")
    converted = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value <= 0.0 for value in converted):
        raise ValidationError(f"{field} must contain finite positive values")
    return converted


def _positive_integers(values: Any, *, field: str) -> tuple[int, ...]:
    if (
        not isinstance(values, list)
        or not values
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in values
        )
    ):
        raise ValidationError(f"{field} must contain positive integers")
    return tuple(values)


def load_protocol(path: Path) -> AuditProtocol:
    """Load exactly the preregistered schema; unknown fields are hard failures."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(
            "audit protocol must be readable valid JSON",
            path=str(path),
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValidationError("audit protocol must be a JSON object")
    fields = set(payload)
    unknown = sorted(fields - _PROTOCOL_FIELDS)
    missing = sorted(_PROTOCOL_FIELDS - fields)
    if unknown:
        raise ValidationError("audit protocol contains unknown fields", fields=unknown)
    if missing:
        raise ValidationError(
            "audit protocol is missing required fields", fields=missing
        )

    development = _positive_floats(
        payload["development_bond_lengths_angstrom"],
        field="development_bond_lengths_angstrom",
    )
    held_out = _positive_floats(
        payload["held_out_bond_lengths_angstrom"],
        field="held_out_bond_lengths_angstrom",
    )
    shots = _positive_integers(payload["shot_budgets"], field="shot_budgets")
    dimensions = _positive_integers(
        payload["subspace_dimensions"],
        field="subspace_dimensions",
    )
    seeds_payload = payload["seeds"]
    if (
        not isinstance(seeds_payload, list)
        or not seeds_payload
        or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in seeds_payload
        )
    ):
        raise ValidationError("seeds must contain one or more integers")
    seeds = tuple(seeds_payload)
    rank_weights_raw = _positive_floats(
        payload["classical_rank_weights"],
        field="classical_rank_weights",
    )

    if payload["protocol_version"] != 1:
        raise ValidationError(
            "unsupported audit protocol version",
            protocol_version=payload["protocol_version"],
        )
    if not isinstance(payload["title"], str) or not payload["title"].strip():
        raise ValidationError("title must be a non-empty string")
    if set(development) & set(held_out):
        raise ValidationError("development and held-out bond lengths must be disjoint")
    if len(set(development)) != len(development) or len(set(held_out)) != len(held_out):
        raise ValidationError("bond lengths must be unique within each split")
    if tuple(sorted(shots)) != shots or len(set(shots)) != len(shots):
        raise ValidationError("shot_budgets must be strictly increasing")
    if tuple(sorted(dimensions)) != dimensions or len(set(dimensions)) != len(
        dimensions
    ):
        raise ValidationError("subspace_dimensions must be strictly increasing")
    if dimensions[-1] > 4:
        raise ValidationError(
            "H2/STO-3G has only four determinants in the (1, 1) sector"
        )
    if len(set(seeds)) != len(seeds):
        raise ValidationError("seeds must be unique")
    confidence = float(payload["confidence"])
    if not 0.0 < confidence < 1.0:
        raise ValidationError("confidence must lie strictly between zero and one")
    bootstrap_resamples = payload["bootstrap_resamples"]
    minimum_pairs = payload["minimum_pairs"]
    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or bootstrap_resamples <= 0
    ):
        raise ValidationError("bootstrap_resamples must be a positive integer")
    if (
        isinstance(minimum_pairs, bool)
        or not isinstance(minimum_pairs, int)
        or minimum_pairs <= 0
    ):
        raise ValidationError("minimum_pairs must be a positive integer")
    if len(rank_weights_raw) != 3:
        raise ValidationError("classical_rank_weights must contain ranks 0, 1, and 2")
    for field in ("primary_estimand", "interpretation"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValidationError(f"{field} must be a non-empty string")

    return AuditProtocol(
        protocol_version=1,
        title=payload["title"],
        development_bond_lengths_angstrom=development,
        held_out_bond_lengths_angstrom=held_out,
        shot_budgets=shots,
        subspace_dimensions=dimensions,
        seeds=seeds,
        confidence=confidence,
        bootstrap_resamples=bootstrap_resamples,
        minimum_pairs=minimum_pairs,
        classical_rank_weights=(
            rank_weights_raw[0],
            rank_weights_raw[1],
            rank_weights_raw[2],
        ),
        primary_estimand=payload["primary_estimand"],
        interpretation=payload["interpretation"],
    )


def _problem_id(bond_length: float) -> str:
    encoded = f"{bond_length:.3f}".replace(".", "p")
    return f"h2-sto3g-r-{encoded}-angstrom"


def _build_problem(
    bond_length: float,
    split: DatasetSplit,
) -> tuple[AuditProblem, np.ndarray]:
    integrals = h2_sto3g_integrals(bond_length)
    active_space = select_active_space(integrals)
    hamiltonian = map_to_qubits(build_fermionic_hamiltonian(integrals, active_space))
    exact, sector_vector = solve_sqd(hamiltonian, 1, 1)
    sector = sector_configurations(hamiltonian.n_qubits, 1, 1)
    statevector = np.zeros(1 << hamiltonian.n_qubits, dtype=np.complex128)
    statevector[sector] = sector_vector
    problem = AuditProblem(
        problem_id=_problem_id(bond_length),
        hamiltonian=hamiltonian,
        n_alpha=1,
        n_beta=1,
        reference_energy=exact.energy,
        split=split,
        descriptors={
            "molecule": "H2",
            "basis": "STO-3G",
            "bond_length_angstrom": bond_length,
            "integral_source": "analytic_closed_form_s_gaussians",
            "pilot_role": "methodological_validation_only",
        },
    )
    return problem, statevector


def _environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for distribution in (
        "numpy",
        "scipy",
        "qdr-compute-engine",
        "qiskit",
        "qiskit-addon-sqd",
    ):
        try:
            packages[distribution] = version(distribution)
        except PackageNotFoundError:
            packages[distribution] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_protocol(protocol: AuditProtocol, *, output_dir: Path) -> ProtocolRun:
    """Execute every frozen cell and write a content-addressed evidence directory."""

    problem_states = [
        _build_problem(bond_length, DatasetSplit.DEVELOPMENT)
        for bond_length in protocol.development_bond_lengths_angstrom
    ]
    problem_states.extend(
        _build_problem(bond_length, DatasetSplit.HELD_OUT)
        for bond_length in protocol.held_out_bond_lengths_angstrom
    )
    postprocessor = CanonicalSectorCompletionPostprocessor()
    observations: list[AuditObservation] = []
    raw_samples: dict[str, np.ndarray] = {}
    for problem, exact_state in problem_states:
        samplers = (
            ExactStateSampler(exact_state),
            HartreeFockExcitationSampler(rank_weights=protocol.classical_rank_weights),
            UniformSectorSampler(),
        )
        for shots in protocol.shot_budgets:
            budget = AuditBudget(
                shots=shots,
                subspace_dimensions=protocol.subspace_dimensions,
            )
            for seed in protocol.seeds:
                for sampler in samplers:
                    cell = run_audit_cell_with_batch(
                        problem,
                        sampler,
                        budget,
                        seed=seed,
                        postprocessor=postprocessor,
                    )
                    observations.extend(cell.observations)
                    sample_id = (
                        f"{problem.problem_id}/{sampler.source_id}/"
                        f"shots-{shots}/seed-{seed}"
                    )
                    raw_samples[sample_id] = cell.batch.configurations

    locked = compare_against_best_classical(
        observations,
        quantum_source_id="exact-state",
        classical_source_ids=("hf-excitations",),
        confidence=protocol.confidence,
        n_resamples=protocol.bootstrap_resamples,
        seed=0,
        min_pairs=protocol.minimum_pairs,
    )
    result = AuditResult(
        protocol={
            **protocol.to_dict(),
            "locked_classical_source_id": locked.classical_source_id,
            "development_classical_scores": [
                [source_id, score] for source_id, score in locked.development_scores
            ],
            "postprocessor_id": postprocessor.postprocessor_id,
        },
        observations=tuple(observations),
        comparison=locked.comparison,
        raw_samples=raw_samples,
        environment=_environment(),
        git_commit=_git_commit(),
    )
    scientific_hash = scientific_result_hash(result)
    destination = output_dir / scientific_hash.split(":", maxsplit=1)[-1]
    bundle = write_audit_bundle(result, destination)
    (destination / "report.md").write_text(
        build_audit_markdown(result),
        encoding="utf-8",
    )
    figures = write_audit_figures(result, destination)
    held_out_ids = tuple(
        problem.problem_id
        for problem, _ in problem_states
        if problem.split is DatasetSplit.HELD_OUT
    )
    return ProtocolRun(
        result=result,
        bundle=bundle,
        scientific_hash=scientific_hash,
        held_out_problem_ids=held_out_ids,
        figures=figures,
    )


__all__ = [
    "AuditProtocol",
    "ProtocolRun",
    "load_protocol",
    "run_protocol",
]

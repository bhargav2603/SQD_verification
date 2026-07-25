"""Preregistered estimands and paired uncertainty for sampler audits."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from qdr_compute_engine.audit.models import AuditObservation, DatasetSplit
from qdr_core.errors import ValidationError


class ComparisonStatus(StrEnum):
    """Decision implied by the preregistered paired confidence interval."""

    QUANTUM_BENEFIT = "quantum_benefit"
    CLASSICAL_BENEFIT = "classical_benefit"
    NO_EVIDENCE = "no_evidence"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class SamplerComparison:
    """Paired quantum-minus-classical effect on the primary AUC estimand."""

    quantum_source_id: str
    classical_source_id: str
    point_estimate: float | None
    interval_low: float | None
    interval_high: float | None
    confidence: float
    n_pairs: int
    missing_pairs: int
    status: ComparisonStatus
    metric: str = "normalized_log_dimension_auc_absolute_energy_error"


@dataclass(frozen=True, slots=True)
class LockedClassicalComparison:
    """Held-out comparison after selecting one classical baseline on development data."""

    classical_source_id: str
    development_scores: tuple[tuple[str, float], ...]
    comparison: SamplerComparison


def log_budget_auc(
    dimensions: Iterable[int],
    errors: Iterable[float],
) -> float:
    """Normalized trapezoid AUC of absolute error versus log retained dimension."""

    dimension_values = tuple(dimensions)
    error_values = tuple(float(value) for value in errors)
    if len(dimension_values) != len(error_values):
        raise ValidationError(
            "dimensions and errors must have the same length",
            dimensions=len(dimension_values),
            errors=len(error_values),
        )
    if len(dimension_values) < 2:
        raise ValidationError("AUC requires at least two curve points")
    if any(isinstance(value, bool) or value <= 0 for value in dimension_values):
        raise ValidationError(
            "AUC dimensions must be positive integers",
            dimensions=list(dimension_values),
        )
    if len(set(dimension_values)) != len(dimension_values):
        raise ValidationError(
            "AUC dimensions must be unique",
            dimensions=list(dimension_values),
        )
    if any(not math.isfinite(value) or value < 0.0 for value in error_values):
        raise ValidationError(
            "AUC errors must be finite and non-negative",
            errors=list(error_values),
        )

    ordered = sorted(zip(dimension_values, error_values, strict=True))
    x = np.log(np.asarray([item[0] for item in ordered], dtype=float))
    y = np.asarray([item[1] for item in ordered], dtype=float)
    width = float(x[-1] - x[0])
    if width <= 0.0:
        raise ValidationError("AUC log-dimension width must be positive")
    return float(np.trapezoid(y, x) / width)


def paired_bootstrap_comparison(
    *,
    quantum: NDArray[np.floating[Any]],
    classical: NDArray[np.floating[Any]],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
    min_pairs: int = 3,
    quantum_source_id: str = "quantum",
    classical_source_id: str = "classical",
) -> SamplerComparison:
    """Bootstrap paired cells; negative quantum-minus-classical values are better."""

    quantum_values = np.asarray(quantum, dtype=float)
    classical_values = np.asarray(classical, dtype=float)
    if quantum_values.ndim != 1 or classical_values.ndim != 1:
        raise ValidationError("paired comparison inputs must be one-dimensional")
    if quantum_values.shape != classical_values.shape:
        raise ValidationError(
            "paired comparison inputs must have the same length",
            quantum_count=int(quantum_values.size),
            classical_count=int(classical_values.size),
        )
    if not bool(np.all(np.isfinite(quantum_values))) or not bool(
        np.all(np.isfinite(classical_values))
    ):
        raise ValidationError("paired comparison inputs must be finite")
    if not 0.0 < confidence < 1.0:
        raise ValidationError("confidence must lie strictly between zero and one")
    if isinstance(n_resamples, bool) or n_resamples <= 0:
        raise ValidationError("n_resamples must be positive", n_resamples=n_resamples)
    if isinstance(min_pairs, bool) or min_pairs <= 0:
        raise ValidationError("min_pairs must be positive", min_pairs=min_pairs)

    n_pairs = int(quantum_values.size)
    if n_pairs < min_pairs:
        return SamplerComparison(
            quantum_source_id=quantum_source_id,
            classical_source_id=classical_source_id,
            point_estimate=None,
            interval_low=None,
            interval_high=None,
            confidence=confidence,
            n_pairs=n_pairs,
            missing_pairs=0,
            status=ComparisonStatus.INCONCLUSIVE,
        )

    differences = quantum_values - classical_values
    point_estimate = float(differences.mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n_pairs, size=(n_resamples, n_pairs))
    bootstrap_means = differences[indices].mean(axis=1)
    alpha = 1.0 - confidence
    interval_low, interval_high = (
        float(value)
        for value in np.quantile(
            bootstrap_means,
            (alpha / 2.0, 1.0 - alpha / 2.0),
        )
    )
    if interval_high < 0.0:
        status = ComparisonStatus.QUANTUM_BENEFIT
    elif interval_low > 0.0:
        status = ComparisonStatus.CLASSICAL_BENEFIT
    else:
        status = ComparisonStatus.NO_EVIDENCE
    return SamplerComparison(
        quantum_source_id=quantum_source_id,
        classical_source_id=classical_source_id,
        point_estimate=point_estimate,
        interval_low=interval_low,
        interval_high=interval_high,
        confidence=confidence,
        n_pairs=n_pairs,
        missing_pairs=0,
        status=status,
    )


CellKey = tuple[str, int, int]


def _source_aucs(
    observations: Iterable[AuditObservation],
    *,
    source_id: str,
    split: DatasetSplit,
) -> dict[CellKey, float]:
    curves: dict[CellKey, list[AuditObservation]] = defaultdict(list)
    for observation in observations:
        if observation.source_id == source_id and observation.split is split:
            key = (observation.problem_id, observation.shots, observation.seed)
            curves[key].append(observation)
    return {
        key: log_budget_auc(
            (item.subspace_dimension for item in curve),
            (item.absolute_error for item in curve),
        )
        for key, curve in curves.items()
    }


def compare_against_best_classical(
    observations: Iterable[AuditObservation],
    *,
    quantum_source_id: str,
    classical_source_ids: Iterable[str],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
    min_pairs: int = 3,
) -> LockedClassicalComparison:
    """Lock the lowest-development-AUC classical source, then evaluate held-out cells."""

    evidence = tuple(observations)
    classical_ids = tuple(classical_source_ids)
    if not classical_ids or len(set(classical_ids)) != len(classical_ids):
        raise ValidationError(
            "classical_source_ids must be non-empty and unique",
            classical_source_ids=list(classical_ids),
        )

    development_scores: list[tuple[str, float]] = []
    for source_id in classical_ids:
        aucs = _source_aucs(
            evidence,
            source_id=source_id,
            split=DatasetSplit.DEVELOPMENT,
        )
        if not aucs:
            raise ValidationError(
                "every classical baseline requires development observations",
                source_id=source_id,
            )
        development_scores.append((source_id, float(np.mean(tuple(aucs.values())))))
    development_scores.sort(key=lambda item: (item[1], item[0]))
    selected_id = development_scores[0][0]

    quantum_held_out = _source_aucs(
        evidence,
        source_id=quantum_source_id,
        split=DatasetSplit.HELD_OUT,
    )
    classical_held_out = _source_aucs(
        evidence,
        source_id=selected_id,
        split=DatasetSplit.HELD_OUT,
    )
    common_keys = sorted(set(quantum_held_out) & set(classical_held_out))
    missing_pairs = len(set(quantum_held_out) ^ set(classical_held_out))
    if missing_pairs:
        comparison = SamplerComparison(
            quantum_source_id=quantum_source_id,
            classical_source_id=selected_id,
            point_estimate=None,
            interval_low=None,
            interval_high=None,
            confidence=confidence,
            n_pairs=len(common_keys),
            missing_pairs=missing_pairs,
            status=ComparisonStatus.INCONCLUSIVE,
        )
    else:
        quantum_by_seed: dict[int, list[float]] = defaultdict(list)
        classical_by_seed: dict[int, list[float]] = defaultdict(list)
        for key in common_keys:
            cluster_seed = key[2]
            quantum_by_seed[cluster_seed].append(quantum_held_out[key])
            classical_by_seed[cluster_seed].append(classical_held_out[key])
        cluster_seeds = sorted(quantum_by_seed)
        comparison = replace(
            paired_bootstrap_comparison(
                quantum=np.asarray(
                    [
                        np.mean(quantum_by_seed[cluster_seed])
                        for cluster_seed in cluster_seeds
                    ],
                    dtype=float,
                ),
                classical=np.asarray(
                    [
                        np.mean(classical_by_seed[cluster_seed])
                        for cluster_seed in cluster_seeds
                    ],
                    dtype=float,
                ),
                confidence=confidence,
                n_resamples=n_resamples,
                seed=seed,
                min_pairs=min_pairs,
                quantum_source_id=quantum_source_id,
                classical_source_id=selected_id,
            ),
            metric=(
                "seed_clustered_mean_normalized_log_dimension_auc_absolute_energy_error"
            ),
        )
    return LockedClassicalComparison(
        classical_source_id=selected_id,
        development_scores=tuple(development_scores),
        comparison=comparison,
    )


__all__ = [
    "ComparisonStatus",
    "LockedClassicalComparison",
    "SamplerComparison",
    "compare_against_best_classical",
    "log_budget_auc",
    "paired_bootstrap_comparison",
]

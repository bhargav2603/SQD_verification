"""Preregistered analysis for the sampler audit."""

from __future__ import annotations

import numpy as np
import pytest

from qdr_compute_engine.audit.analysis import (
    ComparisonStatus,
    compare_against_best_classical,
    log_budget_auc,
    paired_bootstrap_comparison,
)
from qdr_compute_engine.audit.models import (
    AuditObservation,
    DatasetSplit,
    ResourceLedger,
    SamplerKind,
)
from qdr_core.errors import ValidationError

PROBLEM_HASH = "sha256:" + "1" * 64


def _observation(
    *,
    problem_id: str,
    source_id: str,
    split: DatasetSplit,
    seed: int,
    dimension: int,
    error: float,
    shots: int = 128,
    kind: SamplerKind = SamplerKind.CLASSICAL_EXCITATION,
) -> AuditObservation:
    return AuditObservation(
        problem_id=problem_id,
        problem_hash=PROBLEM_HASH,
        source_id=source_id,
        kind=kind,
        split=split,
        seed=seed,
        shots=shots,
        subspace_dimension=dimension,
        energy=error,
        reference_energy=0.0,
        absolute_error=error,
        sector_retention=1.0,
        configurations=tuple(range(dimension)),
        ledger=ResourceLedger(),
    )


def _curve(
    *,
    problem_id: str,
    source_id: str,
    split: DatasetSplit,
    seed: int,
    errors: tuple[float, float],
    shots: int = 128,
    kind: SamplerKind = SamplerKind.CLASSICAL_EXCITATION,
) -> list[AuditObservation]:
    return [
        _observation(
            problem_id=problem_id,
            source_id=source_id,
            split=split,
            seed=seed,
            dimension=dimension,
            error=error,
            shots=shots,
            kind=kind,
        )
        for dimension, error in zip((1, 4), errors, strict=True)
    ]


def test_log_budget_auc_matches_hand_integrated_two_point_curve() -> None:
    assert log_budget_auc((1, 4), (0.010, 0.002)) == pytest.approx(0.006)


def test_log_budget_auc_sorts_dimension_error_pairs() -> None:
    assert log_budget_auc((4, 1), (0.002, 0.010)) == pytest.approx(0.006)


@pytest.mark.parametrize(
    ("dimensions", "errors", "message"),
    [
        ((1,), (0.1,), "two"),
        ((1, 1), (0.1, 0.2), "unique"),
        ((0, 2), (0.1, 0.2), "positive"),
        ((1, 2), (0.1, float("nan")), "finite"),
        ((1, 2), (0.1,), "same length"),
    ],
)
def test_log_budget_auc_rejects_invalid_curves(
    dimensions: tuple[int, ...],
    errors: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        log_budget_auc(dimensions, errors)


def test_paired_comparison_detects_a_consistent_lower_quantum_auc() -> None:
    comparison = paired_bootstrap_comparison(
        quantum=np.array([0.2, 0.3, 0.4]),
        classical=np.array([0.8, 0.9, 1.0]),
        confidence=0.95,
        n_resamples=5_000,
        seed=17,
        min_pairs=3,
    )

    assert comparison.status is ComparisonStatus.QUANTUM_BENEFIT
    assert comparison.interval_high is not None
    assert comparison.interval_high < 0.0
    assert comparison.point_estimate == pytest.approx(-0.6)


def test_paired_comparison_detects_classical_benefit() -> None:
    comparison = paired_bootstrap_comparison(
        quantum=np.array([0.8, 0.9, 1.0]),
        classical=np.array([0.2, 0.3, 0.4]),
        n_resamples=2_000,
        seed=3,
        min_pairs=3,
    )

    assert comparison.status is ComparisonStatus.CLASSICAL_BENEFIT
    assert comparison.interval_low is not None
    assert comparison.interval_low > 0.0


def test_paired_comparison_reports_no_evidence_for_a_tie() -> None:
    comparison = paired_bootstrap_comparison(
        quantum=np.array([0.2, 0.3, 0.4]),
        classical=np.array([0.2, 0.3, 0.4]),
        n_resamples=2_000,
        seed=3,
        min_pairs=3,
    )

    assert comparison.status is ComparisonStatus.NO_EVIDENCE
    assert comparison.interval_low == comparison.interval_high == 0.0


def test_paired_comparison_is_deterministic_for_a_seed() -> None:
    kwargs = {
        "quantum": np.array([0.2, 0.5, 0.4, 0.8]),
        "classical": np.array([0.4, 0.4, 0.9, 0.7]),
        "n_resamples": 2_000,
        "seed": 101,
        "min_pairs": 3,
    }

    assert paired_bootstrap_comparison(**kwargs) == paired_bootstrap_comparison(
        **kwargs
    )


def test_paired_comparison_is_inconclusive_below_minimum_pairs() -> None:
    comparison = paired_bootstrap_comparison(
        quantum=np.array([0.2, 0.3]),
        classical=np.array([0.4, 0.5]),
        n_resamples=1_000,
        seed=1,
        min_pairs=3,
    )

    assert comparison.status is ComparisonStatus.INCONCLUSIVE
    assert comparison.point_estimate is None
    assert comparison.n_pairs == 2


def test_baseline_is_locked_on_development_data_not_held_out_results() -> None:
    observations: list[AuditObservation] = []
    for seed in (0, 1, 2):
        observations += _curve(
            problem_id="development",
            source_id="classical-a",
            split=DatasetSplit.DEVELOPMENT,
            seed=seed,
            errors=(0.2, 0.1),
        )
        observations += _curve(
            problem_id="development",
            source_id="classical-b",
            split=DatasetSplit.DEVELOPMENT,
            seed=seed,
            errors=(0.8, 0.7),
        )
        # Classical B looks excellent on held-out data, but selection must already be locked.
        observations += _curve(
            problem_id="held-out",
            source_id="classical-a",
            split=DatasetSplit.HELD_OUT,
            seed=seed,
            errors=(0.5, 0.4),
        )
        observations += _curve(
            problem_id="held-out",
            source_id="classical-b",
            split=DatasetSplit.HELD_OUT,
            seed=seed,
            errors=(0.01, 0.01),
        )
        observations += _curve(
            problem_id="held-out",
            source_id="quantum",
            split=DatasetSplit.HELD_OUT,
            seed=seed,
            errors=(0.1, 0.1),
            kind=SamplerKind.IDEAL_QUANTUM,
        )

    locked = compare_against_best_classical(
        observations,
        quantum_source_id="quantum",
        classical_source_ids=("classical-a", "classical-b"),
        n_resamples=2_000,
        seed=29,
        min_pairs=3,
    )

    assert locked.classical_source_id == "classical-a"
    assert locked.comparison.status is ComparisonStatus.QUANTUM_BENEFIT


def test_missing_held_out_pair_forces_inconclusive_result() -> None:
    observations: list[AuditObservation] = []
    for seed in (0, 1, 2):
        observations += _curve(
            problem_id="development",
            source_id="classical",
            split=DatasetSplit.DEVELOPMENT,
            seed=seed,
            errors=(0.2, 0.1),
        )
        observations += _curve(
            problem_id="held-out",
            source_id="classical",
            split=DatasetSplit.HELD_OUT,
            seed=seed,
            errors=(0.3, 0.2),
        )
        if seed != 2:
            observations += _curve(
                problem_id="held-out",
                source_id="quantum",
                split=DatasetSplit.HELD_OUT,
                seed=seed,
                errors=(0.1, 0.1),
                kind=SamplerKind.IDEAL_QUANTUM,
            )

    locked = compare_against_best_classical(
        observations,
        quantum_source_id="quantum",
        classical_source_ids=("classical",),
        n_resamples=1_000,
        seed=13,
        min_pairs=2,
    )

    assert locked.comparison.status is ComparisonStatus.INCONCLUSIVE
    assert locked.comparison.missing_pairs == 1


def test_primary_comparison_counts_independent_seed_clusters_not_budget_cells() -> None:
    observations: list[AuditObservation] = []
    for seed in range(5):
        for problem_id in ("held-a", "held-b"):
            for shots in (128, 512, 2048):
                observations += _curve(
                    problem_id=problem_id,
                    source_id="classical",
                    split=DatasetSplit.DEVELOPMENT,
                    seed=seed,
                    shots=shots,
                    errors=(0.4, 0.3),
                )
                observations += _curve(
                    problem_id=problem_id,
                    source_id="classical",
                    split=DatasetSplit.HELD_OUT,
                    seed=seed,
                    shots=shots,
                    errors=(0.4, 0.3),
                )
                observations += _curve(
                    problem_id=problem_id,
                    source_id="quantum",
                    split=DatasetSplit.HELD_OUT,
                    seed=seed,
                    shots=shots,
                    errors=(0.1, 0.1),
                    kind=SamplerKind.IDEAL_QUANTUM,
                )

    locked = compare_against_best_classical(
        observations,
        quantum_source_id="quantum",
        classical_source_ids=("classical",),
        n_resamples=1_000,
        min_pairs=30,
    )

    assert locked.comparison.n_pairs == 5
    assert locked.comparison.status is ComparisonStatus.INCONCLUSIVE

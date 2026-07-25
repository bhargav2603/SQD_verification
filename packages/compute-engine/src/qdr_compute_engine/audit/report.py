"""Paper-oriented tables and figures derived from audit bundles."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from qdr_compute_engine.audit.artifacts import AuditResult
from qdr_compute_engine.audit.models import SamplerKind


def _number(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def build_audit_markdown(result: AuditResult) -> str:
    """Render the decision, observations, and resource ledger without hiding unknowns."""

    comparison = result.comparison
    if comparison.interval_low is None or comparison.interval_high is None:
        interval = "n/a"
    else:
        interval = f"[{comparison.interval_low:.3f}, {comparison.interval_high:.3f}]"
    lines = [
        "# Quantum sampler audit",
        "",
        f"Primary decision: **{comparison.status.value}**",
        "",
        f"Primary metric: `{comparison.metric}`",
        "",
        f"Paired effect (quantum - classical): {_number(comparison.point_estimate)}",
        "",
        f"{comparison.confidence:.0%} paired interval: {interval}",
        "",
        f"Complete pairs: {comparison.n_pairs}; missing pairs: {comparison.missing_pairs}.",
        "",
    ]
    if any(item.kind is SamplerKind.ORACLE for item in result.observations):
        lines.extend(
            [
                "> The exact-state arm is a methodological control. It is not a hardware "
                "quantum-advantage result.",
                "",
            ]
        )
    lines.extend(
        [
            "## Observations and resources",
            "",
            "| Problem | Source | Split | Seed | Shots | Dimension | Absolute error (Ha) | "
            "Sector retention | CPU seconds | QPU seconds |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in result.observations:
        lines.append(
            f"| {item.problem_id} | {item.source_id} | {item.split.value} | {item.seed} "
            f"| {item.shots} | {item.subspace_dimension} | {_number(item.absolute_error)} "
            f"| {_number(item.sector_retention)} | {_number(item.ledger.cpu_seconds)} "
            f"| {_number(item.ledger.qpu_seconds)} |"
        )
    lines.extend(
        [
            "",
            "Unmeasured resources are reported as `n/a`, never as zero.",
            "",
        ]
    )
    return "\n".join(lines)


def write_audit_figures(result: AuditResult, output_dir: Path) -> tuple[Path, ...]:
    """Write the primary convergence figure in both PNG and PDF formats."""

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for item in result.observations:
        grouped[(item.source_id, item.subspace_dimension)].append(item.absolute_error)
    figure, (axis, effect_axis) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    source_ids = sorted({source_id for source_id, _ in grouped})
    for source_id in source_ids:
        dimensions = sorted(
            dimension
            for grouped_source, dimension in grouped
            if grouped_source == source_id
        )
        means = [
            float(np.mean(grouped[(source_id, dimension)])) for dimension in dimensions
        ]
        axis.plot(
            dimensions,
            means,
            marker="o",
            label=source_id,
        )
    axis.set_xscale("log")
    axis.set_yscale("symlog", linthresh=1.0e-12)
    axis.set_xlabel("Retained determinant dimension")
    axis.set_ylabel("Pooled mean absolute energy error (Ha)")
    axis.set_title("Methodological pilot convergence")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()

    comparison = result.comparison
    if (
        comparison.point_estimate is not None
        and comparison.interval_low is not None
        and comparison.interval_high is not None
    ):
        effect_axis.errorbar(
            [comparison.point_estimate],
            [0],
            xerr=[
                [comparison.point_estimate - comparison.interval_low],
                [comparison.interval_high - comparison.point_estimate],
            ],
            fmt="o",
            capsize=5,
        )
    effect_axis.axvline(0.0, color="black", linewidth=1, linestyle="--")
    effect_axis.set_yticks([0], ["seed-clustered effect"])
    effect_axis.set_xlabel("Oracle − classical AUC (Ha)")
    effect_axis.set_title(f"Primary decision: {comparison.status.value}")
    effect_axis.grid(True, axis="x", alpha=0.25)
    figure.tight_layout()
    paths = (
        output_dir / "sampler-audit-convergence.png",
        output_dir / "sampler-audit-convergence.pdf",
    )
    for path in paths:
        figure.savefig(path, dpi=300)
    plt.close(figure)
    return paths


__all__ = ["build_audit_markdown", "write_audit_figures"]

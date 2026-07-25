"""Paper-oriented tables and figures derived from audit bundles."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

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
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for item in result.observations:
        grouped[item.source_id].append((item.subspace_dimension, item.absolute_error))
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for source_id, points in sorted(grouped.items()):
        points.sort()
        axis.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            label=source_id,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Retained determinant dimension")
    axis.set_ylabel("Absolute energy error (Ha)")
    axis.set_title("Sampler audit convergence")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
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

"""Human-readable, paper-oriented views of audit evidence."""

from __future__ import annotations

from pathlib import Path

from test_audit_artifacts import _result

from qdr_compute_engine.audit.report import build_audit_markdown, write_audit_figures


def test_report_exposes_decision_interval_and_unknown_costs() -> None:
    markdown = build_audit_markdown(_result())

    assert "quantum_benefit" in markdown
    assert "[-0.003, -0.001]" in markdown
    assert "QPU seconds" in markdown
    assert "n/a" in markdown
    assert "methodological control" in markdown


def test_write_figures_emits_png_and_pdf(tmp_path: Path) -> None:
    paths = write_audit_figures(_result(), tmp_path)

    assert {path.suffix for path in paths} == {".png", ".pdf"}
    assert all(path.stat().st_size > 0 for path in paths)

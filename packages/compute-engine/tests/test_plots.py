"""Figures are drawn from records, so they can be tested without PySCF.

Every test here builds ``BenchmarkRecord`` instances by hand and asserts on what the
plotting layer does with them -- which figures it produces, which it skips for lack of
data, and that a saved run round-trips back into records. None of it needs a chemistry
backend, so it runs on every platform; it needs only matplotlib, without which the whole
module is skipped.
"""

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from qdr_compute_engine.benchmarks.plots import (
    active_space_convergence_figure,
    convergence_figure,
    dissociation_error_figure,
    dissociation_figure,
    error_decomposition_figure,
    load_records,
    resource_figure,
    write_figures,
)
from qdr_compute_engine.benchmarks.report import write_results
from qdr_compute_engine.benchmarks.runner import BenchmarkRecord


def _record(
    system: str,
    active_space_label: str = "full",
    *,
    n_qubits: int = 8,
    e_hf: float = -75.0,
    e_vqe: float = -75.01,
    e_casci: float | None = -75.011,
    e_fci_full: float | None = -75.02,
    energy_history: tuple[float, ...] = (),
) -> BenchmarkRecord:
    """A fully-populated record; the error terms are derived to stay self-consistent."""
    error_quantum_path = None if e_casci is None else e_vqe - e_casci
    error_active_space = None if (e_casci is None or e_fci_full is None) else e_casci - e_fci_full
    error_total = None if e_fci_full is None else e_vqe - e_fci_full
    return BenchmarkRecord(
        system=system,
        basis="sto-3g",
        active_space_label=active_space_label,
        n_qubits=n_qubits,
        n_parameters=3,
        n_gates=10,
        n_cnot=4,
        n_hamiltonian_terms=15,
        e_hf=e_hf,
        e_vqe=e_vqe,
        e_casci=e_casci,
        e_fci_full=e_fci_full,
        error_quantum_path=error_quantum_path,
        error_active_space=error_active_space,
        error_total=error_total,
        optimizer="L-BFGS-B",
        used_analytic_gradient=True,
        converged=True,
        n_iterations=12,
        n_energy_evaluations=40,
        seconds_integrals=0.1,
        seconds_vqe=0.5,
        seconds_reference=0.2,
        warnings=[],
        energy_history=list(energy_history),
    )


def _scan_records() -> list[BenchmarkRecord]:
    return [
        _record(f"H2@{r:.3f}", n_qubits=4, e_hf=-1.10 + 0.01 * i, e_vqe=-1.13, e_fci_full=-1.13)
        for i, r in enumerate((0.5, 0.735, 1.5, 3.0))
    ]


def _standard_records() -> list[BenchmarkRecord]:
    # A short decreasing history that lands just above e_casci, so the convergence figure
    # has a residual curve to draw.
    history = (-74.98, -75.005, -75.0105, -75.011)
    return [
        _record("LiH", "CAS(2,5)", n_qubits=10, energy_history=history),
        _record("H2O", "CAS(4,4)", n_qubits=8, e_fci_full=-75.02, energy_history=history),
        _record(
            "H2O",
            "CAS(6,5)",
            n_qubits=10,
            e_casci=-75.017,
            e_fci_full=-75.02,
            energy_history=history,
        ),
        _record(
            "H2O",
            "CAS(8,6)",
            n_qubits=12,
            e_casci=-75.0198,
            e_fci_full=-75.02,
            energy_history=history,
        ),
    ]


def test_dissociation_figure_drawn_only_for_scan_records() -> None:
    assert dissociation_figure(_scan_records()) is not None
    assert dissociation_figure(_standard_records()) is None


def test_error_decomposition_needs_both_error_terms() -> None:
    assert error_decomposition_figure(_standard_records()) is not None
    # Without the full-space FCI reference the active-space term is unknown, so there is
    # nothing to decompose and the figure is skipped rather than drawn misleadingly.
    no_reference = [_record("LiH", "CAS(2,5)", e_fci_full=None)]
    assert error_decomposition_figure(no_reference) is None


def test_convergence_figure_needs_a_swept_system() -> None:
    # H2O appears in three active spaces here -> a convergence line exists.
    assert active_space_convergence_figure(_standard_records()) is not None
    # A single active space per system has nothing to converge.
    assert active_space_convergence_figure([_record("LiH", "CAS(2,5)")]) is None


def test_dissociation_error_figure_needs_a_scan_with_fci() -> None:
    assert dissociation_error_figure(_scan_records()) is not None
    assert dissociation_error_figure(_standard_records()) is None


def test_convergence_figure_needs_an_optimiser_history() -> None:
    assert convergence_figure(_standard_records()) is not None
    # Records without a recorded history (e.g. produced before it was plumbed through).
    no_history = [_record("LiH", "CAS(2,5)")]
    assert convergence_figure(no_history) is None


def test_resource_figure_needs_standard_cases() -> None:
    assert resource_figure(_standard_records()) is not None
    # A scan alone has no standard cases to chart.
    assert resource_figure(_scan_records()) is None


def test_write_figures_emits_a_png_per_available_figure(tmp_path: Path) -> None:
    records = [*_scan_records(), *_standard_records()]
    paths = write_figures(records, tmp_path, stem="run")

    names = {p.name for p in paths}
    assert names == {
        "run_dissociation.png",
        "run_dissociation_error.png",
        "run_error_decomposition.png",
        "run_active_space_convergence.png",
        "run_convergence.png",
        "run_resources.png",
    }
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_write_figures_skips_figures_without_data(tmp_path: Path) -> None:
    # Only a scan: the standard-case figures have nothing to draw and must not be written.
    paths = write_figures(_scan_records(), tmp_path, stem="run")
    assert {p.name for p in paths} == {"run_dissociation.png", "run_dissociation_error.png"}


def test_load_records_round_trips_a_saved_run(tmp_path: Path) -> None:
    """Figures must be reconstructable from a saved JSON without recomputing anything."""
    original = _standard_records()
    json_path, _ = write_results(original, tmp_path, stem="run")

    reloaded = load_records(json_path)

    assert [r.system for r in reloaded] == [r.system for r in original]
    assert [r.error_total for r in reloaded] == [r.error_total for r in original]
    # The reloaded records still drive the plotting layer.
    assert active_space_convergence_figure(reloaded) is not None

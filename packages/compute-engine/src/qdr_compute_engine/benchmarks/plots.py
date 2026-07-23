"""Render benchmark records as publication-quality figures.

The JSON and Markdown that :mod:`qdr_compute_engine.benchmarks.report` writes are the
artifact; these figures are a *view* of the same records, for a write-up or a slide.
Three figures, each answering a question a reviewer actually asks:

``dissociation``
    E vs bond length for the H2 scan, with HF, VQE and FCI overlaid. The one figure that
    shows the quantum path tracking the exact curve into the stretched-bond region where
    restricted Hartree-Fock breaks -- i.e. that it reproduces the *physics*, not just the
    reference near equilibrium.
``error_decomposition``
    Per case, the quantum-path error and the active-space error side by side, in
    kcal/mol, against the chemical-accuracy line. This is the honest-error-budget figure:
    it separates "error the quantum algorithm introduced" from "error I chose by
    truncating the orbital space", so neither can hide inside a flattering total.
``active_space_convergence``
    Total error vs active-space size for any system swept over more than one active space
    (H2O here). The systematic-improvability story: the error shrinks as orbitals are
    added back.

matplotlib is imported lazily and only inside the functions that draw, with the headless
``Agg`` backend forced before ``pyplot`` loads. Nothing in the engine imports this
module, and importing it does not require matplotlib -- so the package stays importable
where matplotlib is absent, exactly as the runner stays importable where PySCF is absent.
"""

import json
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qdr_compute_engine.benchmarks.runner import BenchmarkRecord, hartree_to_kcal
from qdr_core.limits import CHEMICAL_ACCURACY_HA, HARTREE_TO_KCAL_PER_MOL

if TYPE_CHECKING:
    from matplotlib.figure import Figure

#: Chemical accuracy expressed in kcal/mol -- the line every error figure is read against.
_CHEMICAL_ACCURACY_KCAL: float = CHEMICAL_ACCURACY_HA * HARTREE_TO_KCAL_PER_MOL

#: A display floor for log-scale error bars. A variational quantum-path error can be
#: ~1e-6 kcal/mol, which is below any plottable range; it is clipped to this for the bar
#: height and annotated with its true value so nothing is silently rounded to zero.
_LOG_FLOOR_KCAL = 1e-4


def _configure_matplotlib() -> Any:
    """Import pyplot with a headless backend. Lazy, so the module imports without it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _is_scan_record(record: BenchmarkRecord) -> bool:
    """H2 dissociation-scan cases are named ``H2@<r>``; everything else is a standard case."""
    return record.system.startswith("H2@")


def load_records(json_path: Path) -> list[BenchmarkRecord]:
    """Reconstruct records from a ``report.write_results`` JSON file.

    This is what lets figures be regenerated from a saved run without recomputing
    anything -- e.g. downloading ``validation.json`` from a VM and plotting locally.
    ``within_chemical_accuracy`` is a computed property, not a constructor field, so it is
    dropped from each payload before the record is rebuilt.
    """
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    field_names = {f.name for f in fields(BenchmarkRecord)}
    records = []
    for entry in payload["records"]:
        records.append(BenchmarkRecord(**{k: v for k, v in entry.items() if k in field_names}))
    return records


def dissociation_figure(records: list[BenchmarkRecord]) -> "Figure | None":
    """E vs bond length for the H2 scan: HF, VQE, FCI overlaid. ``None`` if no scan ran."""
    scan = sorted(
        (r for r in records if _is_scan_record(r)),
        key=lambda r: float(r.system.split("@")[1]),
    )
    if not scan:
        return None

    plt = _configure_matplotlib()

    distances = [float(r.system.split("@")[1]) for r in scan]
    e_hf = [r.e_hf for r in scan]
    e_vqe = [r.e_vqe for r in scan]
    e_fci = [r.e_fci_full for r in scan]

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot(distances, e_hf, "s--", color="#888888", label="Hartree-Fock", markersize=4)
    ax.plot(distances, e_fci, "-", color="#1f77b4", label="FCI (exact)", linewidth=2)
    ax.plot(distances, e_vqe, "o", color="#d62728", label="VQE / UCCSD", markersize=5)

    ax.set_xlabel("H-H bond length (Angstrom)")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title("H$_2$ dissociation, STO-3G: VQE tracks FCI where HF fails")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def error_decomposition_figure(records: list[BenchmarkRecord]) -> "Figure | None":
    """Quantum-path vs active-space error per case, in kcal/mol, on a log scale.

    Only standard cases with both error terms quantified are shown; scan cases and cases
    run without the full-space FCI reference are skipped. ``None`` if nothing qualifies.
    """
    cases = [
        r
        for r in records
        if not _is_scan_record(r)
        and r.error_quantum_path is not None
        and r.error_active_space is not None
    ]
    if not cases:
        return None

    plt = _configure_matplotlib()

    labels = [f"{r.system}\n{r.active_space_label}" for r in cases]
    quantum = [abs(hartree_to_kcal(r.error_quantum_path) or 0.0) for r in cases]
    active = [abs(hartree_to_kcal(r.error_active_space) or 0.0) for r in cases]

    def _clip(value: float) -> float:
        return max(value, _LOG_FLOOR_KCAL)

    import numpy as np

    x = np.arange(len(cases))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(6.0, 1.6 * len(cases)), 4.5))
    bars_q = ax.bar(
        x - width / 2, [_clip(v) for v in quantum], width, color="#d62728", label="quantum-path"
    )
    bars_a = ax.bar(
        x + width / 2, [_clip(v) for v in active], width, color="#1f77b4", label="active-space"
    )

    # Annotate each bar with its true value, so a bar clipped to the floor still reports
    # the number it stands for rather than reading as "zero".
    for bar, value in [*zip(bars_q, quantum, strict=True), *zip(bars_a, active, strict=True)]:
        ax.annotate(
            f"{value:.1e}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )

    ax.axhline(
        _CHEMICAL_ACCURACY_KCAL,
        color="green",
        linestyle="--",
        linewidth=1,
        label=f"chemical accuracy ({_CHEMICAL_ACCURACY_KCAL:.2f} kcal/mol)",
    )
    ax.set_yscale("log")
    # Headroom above the tallest bar so the vertical annotations clear the title/legend.
    ax.set_ylim(top=max(_clip(v) for v in (*quantum, *active, _CHEMICAL_ACCURACY_KCAL)) * 4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("|error| (kcal/mol)")
    ax.set_title("Error budget: quantum-path vs active-space truncation")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def active_space_convergence_figure(records: list[BenchmarkRecord]) -> "Figure | None":
    """Total error vs active-space size, one line per system swept over >1 active space.

    ``None`` if no system was run in more than one active space with a total error.
    """
    by_system: dict[str, list[BenchmarkRecord]] = {}
    for record in records:
        if _is_scan_record(record) or record.error_total is None:
            continue
        by_system.setdefault(record.system, []).append(record)

    swept = {name: recs for name, recs in by_system.items() if len(recs) > 1}
    if not swept:
        return None

    plt = _configure_matplotlib()

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for name, recs in sorted(swept.items()):
        ordered = sorted(recs, key=lambda r: r.n_qubits)
        qubits = [r.n_qubits for r in ordered]
        errors = [abs(hartree_to_kcal(r.error_total) or 0.0) for r in ordered]
        ax.plot(qubits, errors, "o-", label=name, markersize=5)

    ax.axhline(
        _CHEMICAL_ACCURACY_KCAL,
        color="green",
        linestyle="--",
        linewidth=1,
        label=f"chemical accuracy ({_CHEMICAL_ACCURACY_KCAL:.2f} kcal/mol)",
    )
    ax.set_yscale("log")
    ax.set_xlabel("active-space size (qubits)")
    ax.set_ylabel("|total error| (kcal/mol)")
    ax.set_title("Active-space convergence: error falls as orbitals are restored")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


#: The figures, keyed by the filename suffix they are written under.
_FIGURES = {
    "dissociation": dissociation_figure,
    "error_decomposition": error_decomposition_figure,
    "active_space_convergence": active_space_convergence_figure,
}


def write_figures(
    records: list[BenchmarkRecord],
    output_dir: Path,
    *,
    stem: str = "validation",
    dpi: int = 300,
) -> list[Path]:
    """Write every figure that has data as ``<stem>_<name>.png``; return the paths.

    A figure with nothing to draw (no scan, no swept active space) is skipped rather than
    written empty, so the set of files produced reflects the set of questions the run can
    actually answer.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _configure_matplotlib()

    written = []
    for name, builder in _FIGURES.items():
        figure = builder(records)
        if figure is None:
            continue
        path = output_dir / f"{stem}_{name}.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written

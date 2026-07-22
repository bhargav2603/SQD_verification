"""Turn benchmark records into the tables that go in a write-up.

Two outputs, deliberately:

``JSON``
    Every field, full precision, plus the environment the numbers were measured on.
    This is the artifact; the tables are a view of it.
``Markdown``
    Human-readable, rounded, for reading and pasting.

Rounding happens only in the Markdown. Energies are printed to 8 decimal places (well
below the ~1e-6 Ha scale that matters) and errors in both Hartree and kcal/mol, because
chemical accuracy is defined in kcal/mol and a reader should not have to convert.
"""

import json
from pathlib import Path
from typing import Any

from qdr_compute_engine.benchmarks.runner import (
    BenchmarkRecord,
    environment_metadata,
    hartree_to_kcal,
)
from qdr_core.limits import CHEMICAL_ACCURACY_HA


def _format_energy(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.8f}"


def _format_error(value: float | None) -> str:
    if value is None:
        return "n/a"
    kcal = hartree_to_kcal(value)
    assert kcal is not None
    return f"{value:+.2e} ({kcal:+.3f})"


def energy_table(records: list[BenchmarkRecord]) -> str:
    """Energies and the error decomposition. The core table."""
    header = (
        "| System | Basis | Active space | E_HF (Ha) | E_VQE (Ha) | E_CASCI (Ha) | "
        "E_FCI full (Ha) | quantum-path err Ha (kcal/mol) | active-space err Ha (kcal/mol) | "
        "total err Ha (kcal/mol) | < 1 kcal/mol |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for record in records:
        rows.append(
            f"| {record.system} | {record.basis} | {record.active_space_label} "
            f"| {_format_energy(record.e_hf)} "
            f"| {_format_energy(record.e_vqe)} "
            f"| {_format_energy(record.e_casci)} "
            f"| {_format_energy(record.e_fci_full)} "
            f"| {_format_error(record.error_quantum_path)} "
            f"| {_format_error(record.error_active_space)} "
            f"| {_format_error(record.error_total)} "
            f"| {'yes' if record.within_chemical_accuracy else 'no'} |"
        )
    return header + "\n".join(rows) + "\n"


def resource_table(records: list[BenchmarkRecord]) -> str:
    """Qubits, circuit size and wall-clock. What the result actually cost."""
    header = (
        "| System | Active space | Qubits | Params | Gates | CNOTs | H terms | "
        "Optimizer | Analytic grad | Iters | Energy evals | Integrals (s) | VQE (s) | "
        "Reference (s) |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for record in records:
        rows.append(
            f"| {record.system} | {record.active_space_label} | {record.n_qubits} "
            f"| {record.n_parameters} | {record.n_gates} | {record.n_cnot} "
            f"| {record.n_hamiltonian_terms} | {record.optimizer} "
            f"| {'yes' if record.used_analytic_gradient else 'no'} "
            f"| {record.n_iterations} | {record.n_energy_evaluations} "
            f"| {record.seconds_integrals:.2f} | {record.seconds_vqe:.2f} "
            f"| {record.seconds_reference:.2f} |"
        )
    return header + "\n".join(rows) + "\n"


def warnings_section(records: list[BenchmarkRecord]) -> str:
    """Every warning, verbatim. A run with warnings is not a clean run."""
    flagged = [r for r in records if r.warnings]
    if not flagged:
        return "No warnings were raised.\n"

    lines = ["The following cases raised warnings and must not be reported as clean:\n"]
    for record in flagged:
        lines.append(f"**{record.system} / {record.active_space_label}**")
        for warning in record.warnings:
            lines.append(f"  - {warning}")
        lines.append("")
    return "\n".join(lines) + "\n"


def dissociation_table(records: list[BenchmarkRecord]) -> str:
    """The H2 scan, ordered by bond length."""
    scan = sorted(
        (r for r in records if r.system.startswith("H2@")),
        key=lambda r: float(r.system.split("@")[1]),
    )
    if not scan:
        return ""

    header = (
        "| r (A) | E_HF (Ha) | E_VQE (Ha) | E_FCI (Ha) | E_VQE - E_FCI (Ha) | "
        "correlation energy (Ha) |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = []
    for record in scan:
        distance = float(record.system.split("@")[1])
        correlation = None if record.e_fci_full is None else record.e_fci_full - record.e_hf
        rows.append(
            f"| {distance:.3f} | {_format_energy(record.e_hf)} "
            f"| {_format_energy(record.e_vqe)} | {_format_energy(record.e_fci_full)} "
            f"| {_format_energy(record.error_total)} | {_format_energy(correlation)} |"
        )
    return header + "\n".join(rows) + "\n"


def build_markdown(records: list[BenchmarkRecord], environment: dict[str, Any]) -> str:
    n_clean = sum(1 for r in records if not r.warnings)
    n_accurate = sum(1 for r in records if r.within_chemical_accuracy)

    sections = [
        "# Engine validation results",
        "",
        "Every reference energy in these tables is **computed at run time** by PySCF in "
        "the same basis and active space, not copied from a paper. `E_CASCI` is exact "
        "within the active space; `E_FCI full` is exact within the full orbital space.",
        "",
        "The error is split because the two halves have different causes and different fixes:",
        "",
        "- **quantum-path error** = `E_VQE - E_CASCI` — ansatz, Trotter and optimiser. "
        "Non-negative by the variational principle; a negative value is a bug.",
        "- **active-space error** = `E_CASCI - E_FCI` — the cost of truncating the "
        "orbital space. Usually the larger term, and invisible in a bare VQE-vs-FCI "
        "comparison.",
        "",
        f"Chemical accuracy threshold: {CHEMICAL_ACCURACY_HA:.1e} Ha (1 kcal/mol), "
        "applied to the **total** error.",
        "",
        f"**{len(records)} cases run, {n_clean} without warnings, "
        f"{n_accurate} within chemical accuracy on total error.**",
        "",
        "## Energies and error decomposition",
        "",
        energy_table(records),
        "## Resources and timings",
        "",
        resource_table(records),
    ]

    scan = dissociation_table(records)
    if scan:
        sections.extend(
            [
                "## H2 dissociation curve",
                "",
                "Minimal-basis UCCSD has a single double excitation, so the first-order "
                "Trotter factorisation is exact and any deviation from FCI is an "
                "implementation error rather than a method error. Restricted "
                "Hartree-Fock degrades badly beyond ~1.5 A while FCI does not, so the "
                "growing correlation energy is the point of the scan.",
                "",
                scan,
            ]
        )

    sections.extend(
        [
            "## Warnings",
            "",
            warnings_section(records),
            "## Environment",
            "",
            "```json",
            json.dumps(environment, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(sections)


def write_results(
    records: list[BenchmarkRecord],
    output_dir: Path,
    *,
    stem: str = "validation",
) -> tuple[Path, Path]:
    """Write ``<stem>.json`` and ``<stem>.md``; return both paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = environment_metadata()

    json_path = output_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(
            {
                "environment": environment,
                "chemical_accuracy_ha": CHEMICAL_ACCURACY_HA,
                "records": [r.to_dict() for r in records],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    markdown_path = output_dir / f"{stem}.md"
    markdown_path.write_text(build_markdown(records, environment), encoding="utf-8")

    return json_path, markdown_path

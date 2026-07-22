"""Validation-suite tests that do not need PySCF.

The suite itself runs on a remote Linux host, so the cost of a mistake in it is a failed
overnight sweep. Everything that can be checked without PySCF is checked here: the system
definitions are physically valid, the active spaces are consistent with their basis, the
report renders, and the CLI fails cleanly rather than crashing when PySCF is absent.
"""

import json
from pathlib import Path

import pytest

from qdr_compute_engine.benchmarks import systems
from qdr_compute_engine.benchmarks.__main__ import _parse_args, main
from qdr_compute_engine.benchmarks.report import (
    build_markdown,
    dissociation_table,
    energy_table,
    resource_table,
    warnings_section,
    write_results,
)
from qdr_compute_engine.benchmarks.runner import BenchmarkRecord
from qdr_compute_engine.geometry.pose_builder import (
    count_electrons,
    require_closed_shell,
    validate_geometry,
)

# ---------------------------------------------------------------------------------
# System definitions
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("system", systems.all_systems(), ids=lambda s: s.name)
def test_every_system_is_a_valid_geometry(system) -> None:
    validate_geometry(system.geometry)


@pytest.mark.parametrize("system", systems.all_systems(), ids=lambda s: s.name)
def test_every_system_is_closed_shell(system) -> None:
    """The engine rejects open-shell input, so no benchmark may smuggle one in."""
    require_closed_shell(system.geometry)


@pytest.mark.parametrize("system", systems.all_systems(), ids=lambda s: s.name)
def test_every_system_records_its_geometry_source(system) -> None:
    """A geometry with no provenance cannot be reproduced or checked."""
    assert system.geometry_source.strip()


@pytest.mark.parametrize("system", systems.STANDARD_SYSTEMS, ids=lambda s: s.name)
def test_active_spaces_fit_the_electron_count(system) -> None:
    n_electrons = count_electrons(system.geometry)
    for spec in system.active_spaces:
        if spec.n_active_electrons is not None:
            assert spec.n_active_electrons <= n_electrons
            assert spec.n_active_electrons % 2 == 0, "closed shell needs paired electrons"
        if spec.n_active_electrons is not None and spec.n_active_orbitals is not None:
            assert spec.n_active_electrons <= 2 * spec.n_active_orbitals


@pytest.mark.parametrize("system", systems.STANDARD_SYSTEMS, ids=lambda s: s.name)
def test_active_spaces_stay_within_the_qubit_cap(system) -> None:
    from qdr_core.limits import MAX_QUBITS, qubits_for_orbitals

    for spec in system.active_spaces:
        if spec.n_active_orbitals is not None:
            assert qubits_for_orbitals(spec.n_active_orbitals) <= MAX_QUBITS


def test_dissociation_curve_spans_the_requested_range() -> None:
    lengths = systems.H2_SCAN_BOND_LENGTHS
    assert min(lengths) == pytest.approx(0.50)
    assert max(lengths) == pytest.approx(3.00)
    assert list(lengths) == sorted(lengths), "scan must be ordered"
    assert len(set(lengths)) == len(lengths), "no duplicate bond lengths"


def test_equilibrium_bond_length_is_in_the_scan() -> None:
    """0.735 A is the geometry the H2 golden file and the stage-0 gate use."""
    assert 0.735 in systems.H2_SCAN_BOND_LENGTHS


def test_by_name_round_trips() -> None:
    assert systems.by_name("LiH").name == "LiH"
    assert systems.by_name("H2@0.735").name == "H2@0.735"


def test_by_name_rejects_unknown_with_a_useful_message() -> None:
    with pytest.raises(KeyError, match="available"):
        systems.by_name("caffeine")


def test_system_names_are_unique() -> None:
    names = [s.name for s in systems.all_systems()]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------


def _record(**overrides) -> BenchmarkRecord:
    defaults = dict(
        system="LiH",
        basis="sto-3g",
        active_space_label="CAS(2,5)",
        n_qubits=10,
        n_parameters=54,
        n_gates=6590,
        n_cnot=3440,
        n_hamiltonian_terms=876,
        e_hf=-7.8633,
        e_vqe=-7.8823,
        e_casci=-7.8823,
        e_fci_full=-7.8824,
        error_quantum_path=1.2e-9,
        error_active_space=1.0e-4,
        error_total=1.0e-4,
        optimizer="L-BFGS-B",
        used_analytic_gradient=True,
        converged=True,
        n_iterations=41,
        n_energy_evaluations=44,
        seconds_integrals=0.3,
        seconds_vqe=9.1,
        seconds_reference=0.5,
    )
    defaults.update(overrides)
    return BenchmarkRecord(**defaults)  # type: ignore[arg-type]


def test_energy_table_renders_all_records() -> None:
    table = energy_table([_record(), _record(system="BeH2")])
    assert table.count("\n") >= 4
    assert "LiH" in table and "BeH2" in table


def test_resource_table_reports_qubits_and_gates() -> None:
    table = resource_table([_record()])
    assert "10" in table and "3440" in table and "L-BFGS-B" in table


def test_missing_references_render_as_not_available_not_zero() -> None:
    """An unknown error term must never be printed as 0.0."""
    table = energy_table([_record(e_fci_full=None, error_total=None, error_active_space=None)])
    assert "n/a" in table
    assert "0.00e+00" not in table


def test_chemical_accuracy_uses_total_error() -> None:
    """A tiny quantum-path error must not mark a case accurate when the total is large."""
    record = _record(error_quantum_path=1e-12, error_total=5e-2)
    assert not record.within_chemical_accuracy

    assert _record(error_total=1e-4).within_chemical_accuracy


def test_warnings_section_lists_every_warning() -> None:
    section = warnings_section([_record(warnings=["variational principle violated"])])
    assert "variational principle violated" in section
    assert "not be reported as clean" in section


def test_warnings_section_when_clean() -> None:
    assert "No warnings" in warnings_section([_record()])


def test_dissociation_table_orders_by_bond_length() -> None:
    records = [
        _record(system="H2@1.500", active_space_label="full"),
        _record(system="H2@0.500", active_space_label="full"),
        _record(system="H2@0.900", active_space_label="full"),
    ]
    table = dissociation_table(records)
    positions = [table.index(f"| {d:.3f} |") for d in (0.5, 0.9, 1.5)]
    assert positions == sorted(positions)


def test_dissociation_table_empty_without_scan_records() -> None:
    assert dissociation_table([_record()]) == ""


def test_markdown_states_the_error_decomposition() -> None:
    markdown = build_markdown([_record()], {"python": "3.13"})
    assert "quantum-path error" in markdown
    assert "active-space error" in markdown
    assert "variational principle" in markdown


def test_write_results_emits_both_artifacts(tmp_path: Path) -> None:
    json_path, markdown_path = write_results([_record()], tmp_path, stem="run1")

    assert json_path.exists() and markdown_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["records"][0]["system"] == "LiH"
    assert "environment" in payload
    assert payload["records"][0]["within_chemical_accuracy"] is True


def test_json_preserves_full_precision(tmp_path: Path) -> None:
    """The Markdown rounds; the JSON must not."""
    record = _record(e_vqe=-7.882362839012345)
    json_path, _ = write_results([record], tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["records"][0]["e_vqe"] == pytest.approx(-7.882362839012345, abs=0)


# ---------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------


def test_cli_defaults() -> None:
    args = _parse_args([])
    assert args.suite == "standard"
    assert args.optimizer is None  # driver chooses L-BFGS-B + adjoint


def test_cli_accepts_every_suite() -> None:
    from qdr_compute_engine.benchmarks.__main__ import SUITES

    for suite in SUITES:
        assert _parse_args(["--suite", suite]).suite == suite
        assert SUITES[suite](), f"suite {suite} is empty"


def test_cli_repeatable_system_flag() -> None:
    args = _parse_args(["--system", "LiH", "--system", "H2O"])
    assert args.system == ["LiH", "H2O"]


def test_cli_rejects_unknown_suite() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["--suite", "nonsense"])


def test_cli_exits_cleanly_without_pyscf(monkeypatch, capsys) -> None:
    """Without PySCF the suite must explain itself, not raise."""
    import qdr_compute_engine.integrals.pyscf_integrals as integrals_module

    monkeypatch.setattr(integrals_module, "pyscf_available", lambda: False)
    exit_code = main(["--suite", "standard"])

    assert exit_code == 2
    assert "PySCF is not installed" in capsys.readouterr().err

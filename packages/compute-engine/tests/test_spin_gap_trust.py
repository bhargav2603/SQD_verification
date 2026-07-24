"""The Colab spin-gap campaign must fail the known Fe false-positive case."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from qdr_compute_engine.validation.trust import (
    EvidenceStatus,
    FailureMode,
    Verdict,
    evaluate_trust,
)


def _load_runner() -> ModuleType:
    path = Path(__file__).parents[1] / "notebooks" / "spin_gap_colab.py"
    spec = importlib.util.spec_from_file_location("spin_gap_colab", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_shipped_fe_minimal_model_is_an_explicit_untrusted_control() -> None:
    """Would fail if the old QDR==CASCI gate could still approve the Fe model."""
    runner = _load_runner()

    report = evaluate_trust(runner.build_model_evidence(runner.FE_OCTAHEDRAL))

    assert report.verdict is Verdict.UNTRUSTED
    assert FailureMode.MINIMAL_BASIS in report.failure_modes
    assert FailureMode.MISSING_REQUIRED_ORBITALS in report.failure_modes
    assert "negative control" in runner.FE_OCTAHEDRAL.name.casefold()


def test_r5_production_case_is_nonminimal_and_contains_ligand_valence_targets() -> None:
    runner = _load_runner()

    evidence = runner.build_model_evidence(runner.FES_VERTICAL)
    by_code = {item.code: item for item in evidence}

    assert by_code["basis_representation"].status.value == "PASS"
    assert by_code["required_orbitals"].status.value == "PASS"
    assert runner.FES_VERTICAL.basis.casefold() != "sto-3g"


def test_campaign_evidence_checks_active_dynamic_and_independent_ladders() -> None:
    runner = _load_runner()
    results = [
        {"n_qubits": 10, "gap_kcal": {"casscf": 10.0, "nevpt2": 9.0}},
        {"n_qubits": 16, "gap_kcal": {"casscf": 7.0, "nevpt2": 6.2}},
        {
            "n_qubits": 18,
            "basis": "def2-svp",
            "gap_kcal": {"casscf": 6.4, "nevpt2": 6.0, "ccsd_t": 6.5},
        },
        {
            "n_qubits": 18,
            "basis": "def2-tzvp",
            "gap_kcal": {"casscf": 6.3, "nevpt2": 6.1, "ccsd_t": 6.5},
        },
    ]

    evidence = {item.code: item for item in runner.build_campaign_evidence(results)}

    assert evidence["active_space_ladder"].status is EvidenceStatus.PASS
    assert evidence["dynamic_correlation"].status is EvidenceStatus.PASS
    assert evidence["basis_convergence"].status is EvidenceStatus.PASS
    assert evidence["independent_method_agreement"].status is EvidenceStatus.PASS


def test_campaign_keeps_failed_operational_evidence_from_every_rung() -> None:
    runner = _load_runner()
    results = [
        {
            "trust_report": {
                "evidence": [
                    {
                        "code": "scf_stability_low",
                        "status": "FAIL",
                        "failure_mode": "SCF_UNSTABLE",
                        "message": "unstable small-space reference",
                    }
                ]
            }
        },
        {
            "trust_report": {
                "evidence": [
                    {
                        "code": "scf_stability_low",
                        "status": "PASS",
                        "failure_mode": "SCF_UNSTABLE",
                    }
                ]
            }
        },
    ]

    evidence = runner.build_campaign_member_evidence(results)

    assert evidence[0].code == "rung_1_scf_stability_low"
    assert evidence[0].status is EvidenceStatus.FAIL


def test_main_emits_failure_audit_json_before_reraising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    writes: list[tuple[str, object]] = []

    def fail_case(case: object) -> dict:
        raise RuntimeError("synthetic chemistry failure")

    monkeypatch.setattr(runner, "run_case", fail_case)
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "_write_strict_json",
        lambda path, payload: writes.append((path.name, payload)),
    )

    with pytest.raises(RuntimeError, match="synthetic chemistry failure"):
        runner.main()

    name, payload = writes[-1]
    assert name == "spin_gap_run.json"
    assert payload["status"] == "FAILED"
    assert payload["failure"]["verdict"] == "UNTRUSTED"

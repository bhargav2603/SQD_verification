"""End-to-end methodological pilot for the locked sampler-audit protocol."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from qdr_compute_engine.audit.artifacts import load_audit_bundle
from qdr_compute_engine.audit.h2_pilot import h2_sto3g_integrals
from qdr_compute_engine.audit.__main__ import main
from qdr_compute_engine.audit.protocol import load_protocol, run_protocol
from qdr_core.errors import ValidationError

PROTOCOL_PATH = (
    Path(__file__).resolve().parents[3]
    / "research"
    / "sampler-audit"
    / "protocol-v2.json"
)


def test_audit_h2_integrals_match_the_independent_golden_reference(
    golden_h2: dict[str, Any],
) -> None:
    integrals = h2_sto3g_integrals(0.735)

    assert integrals.e_nuc == pytest.approx(float(golden_h2["e_nuc"]), abs=1.0e-12)
    assert integrals.e_hf == pytest.approx(float(golden_h2["e_hf"]), abs=1.0e-10)
    np.testing.assert_allclose(integrals.h1, golden_h2["h1"], atol=1.0e-10)
    np.testing.assert_allclose(integrals.h2, golden_h2["h2"], atol=1.0e-10)


def test_protocol_loader_refuses_unknown_fields(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["unregistered_metric"] = "quietly-switch-the-outcome"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="unknown"):
        load_protocol(path)


def test_cli_returns_nonzero_for_an_invalid_protocol(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")

    assert main([str(invalid), "--output", str(tmp_path / "results")]) == 2
    assert "invalid protocol" in capsys.readouterr().err.lower()


def test_h2_pilot_produces_a_paired_decision_and_verifiable_bundle(
    tmp_path: Path,
) -> None:
    protocol = replace(
        load_protocol(PROTOCOL_PATH),
        shot_budgets=(64,),
        seeds=(0, 1),
        bootstrap_resamples=200,
        minimum_pairs=2,
    )

    run = run_protocol(protocol, output_dir=tmp_path)
    loaded = load_audit_bundle(run.bundle.manifest)

    assert run.result.comparison.n_pairs == len(protocol.seeds)
    assert loaded.comparison == run.result.comparison
    assert run.bundle.manifest_hash
    assert run.scientific_hash
    assert (run.bundle.manifest.parent / "report.md").is_file()
    assert (
        len(loaded.raw_samples)
        == (
            len(protocol.development_bond_lengths_angstrom)
            + len(protocol.held_out_bond_lengths_angstrom)
        )
        * len(protocol.shot_budgets)
        * len(protocol.seeds)
        * 3
    )

# Chemistry Trust Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add chemistry-aware TRUSTED/UNTRUSTED/UNDECIDABLE verdicts, instrument SQD convergence, and make the transition-metal Colab runner an auditable convergence campaign.

**Architecture:** A pure evidence-ledger evaluator owns verdict semantics. Scientific adapters produce evidence but cannot set verdicts. SQD gains a nested-subspace campaign helper, Stage 9 embeds the report in normal pipeline output, and the Colab runner assembles model, active-space, solver, and post-correlation evidence.

**Tech Stack:** Python 3.12, NumPy, SciPy, OpenFermion, PySCF on Linux/Colab, pytest, JSON.

## Global Constraints

- Preserve all existing uncommitted user changes.
- Run GitNexus upstream impact analysis before editing every existing symbol.
- A same-Hamiltonian solver comparison may satisfy only `ENGINE_CROSSCHECK`.
- Definite failures produce `UNTRUSTED`; missing required evidence produces `UNDECIDABLE`.
- `TRUSTED` requires every required diagnostic to pass.
- JSON output must contain finite, deterministic, machine-readable values.
- PySCF imports remain lazy so pure tests run on Windows.

---

### Task 1: Trust Evidence Model and Evaluator

**Files:**
- Create: `packages/compute-engine/src/qdr_compute_engine/validation/trust.py`
- Create: `packages/compute-engine/tests/test_trust.py`
- Modify: `packages/compute-engine/src/qdr_compute_engine/validation/__init__.py`

**Interfaces:**
- Produces: `Verdict`, `EvidenceStatus`, `FailureMode`, `Evidence`, `TrustReport`,
  `evaluate_trust(evidence: Iterable[Evidence]) -> TrustReport`.
- Consumes: no PySCF or solver implementation.

- [ ] **Step 1: Write failing verdict tests**

```python
def test_failure_dominates_missing_evidence() -> None:
    report = evaluate_trust([
        Evidence("basis", EvidenceStatus.FAIL, FailureMode.MINIMAL_BASIS, required=True),
        Evidence("ladder", EvidenceStatus.MISSING, FailureMode.ACTIVE_SPACE_NOT_CONVERGED, required=True),
    ])
    assert report.verdict is Verdict.UNTRUSTED
    assert report.failure_modes == (FailureMode.MINIMAL_BASIS,)

def test_missing_required_evidence_is_undecidable() -> None:
    report = evaluate_trust([
        Evidence("engine", EvidenceStatus.PASS, FailureMode.ENGINE_DISAGREEMENT, required=True),
        Evidence("ladder", EvidenceStatus.MISSING, FailureMode.ACTIVE_SPACE_NOT_CONVERGED, required=True),
    ])
    assert report.verdict is Verdict.UNDECIDABLE
```

- [ ] **Step 2: Run `pytest packages/compute-engine/tests/test_trust.py -q` and verify import failure**
- [ ] **Step 3: Implement frozen enums/dataclasses, deterministic ordering, finite-value validation, and JSON serialization**
- [ ] **Step 4: Run the trust tests and verify all pass**

### Task 2: Scientific Diagnostic Builders

**Files:**
- Create: `packages/compute-engine/src/qdr_compute_engine/validation/chemistry.py`
- Create: `packages/compute-engine/tests/test_chemistry_validation.py`

**Interfaces:**
- Consumes: scalar/list metadata from SCF, active-space, spin, embedding, and method runs.
- Produces: `basis_evidence`, `required_orbital_evidence`, `spin_evidence`,
  `convergence_evidence`, `agreement_evidence`, `geometry_evidence`,
  `environment_evidence`, `embedding_evidence`.

- [ ] **Step 1: Write failing tests for STO-3G transition-metal rejection, missing ligand shells, spin contamination, converged ladders, and absent ladders**
- [ ] **Step 2: Run the focused test and verify failures name the missing builders**
- [ ] **Step 3: Implement dimensioned threshold comparisons and conservative basis-family classification**
- [ ] **Step 4: Verify the focused tests pass**

### Task 3: SQD Convergence Campaign

**Files:**
- Modify: `packages/compute-engine/src/qdr_compute_engine/solvers/sqd.py`
- Modify: `packages/compute-engine/tests/test_sqd.py`

**Interfaces:**
- Produces: `SQDStep`, `SQDCampaign`, and
  `run_sqd_campaign(hamiltonian, n_alpha, n_beta, configurations, dimensions, reference_energy=None)`.
- Consumes: existing `solve_sqd`; dimensions must be strictly increasing and configuration
  sets nested.

- [ ] **Step 1: Add failing tests showing campaign energies are non-increasing, dimensions are nested, final increments are reported, and invalid ladders are rejected**
- [ ] **Step 2: Run focused SQD tests and verify the new API is absent**
- [ ] **Step 3: Run GitNexus impact on `solve_sqd`, report its direct caller and affected solve flows, then implement the campaign without changing `solve_sqd` semantics**
- [ ] **Step 4: Run all SQD tests**

### Task 4: Pipeline Trust Report

**Files:**
- Modify: `packages/compute-engine/src/qdr_compute_engine/typedefs/reporting.py`
- Modify: `packages/compute-engine/src/qdr_compute_engine/types.py`
- Modify: `packages/compute-engine/src/qdr_compute_engine/pipeline/stage_09_validator.py`
- Modify: `packages/compute-engine/tests/test_pipeline.py`

**Interfaces:**
- `ValidationReport.trust: TrustReport | None`.
- Stage 9 consumes optional `config["trust_evidence"]`; absent evidence yields
  `UNDECIDABLE`, never `TRUSTED`.

- [ ] **Step 1: Add failing serialization and Stage 9 fail-closed tests**
- [ ] **Step 2: Run focused pipeline tests and confirm expected failures**
- [ ] **Step 3: Run GitNexus impact for `ValidationReport` and `stage_09_validator.run`, report risk, then add the backward-compatible field and evaluator call**
- [ ] **Step 4: Run pipeline, reporting, and error-budget tests**

### Task 5: PySCF Evidence and Post-Active Correlation

**Files:**
- Modify: `packages/compute-engine/src/qdr_compute_engine/active_space/avas_selection.py`
- Create: `packages/compute-engine/src/qdr_compute_engine/validation/pyscf_diagnostics.py`
- Create: `packages/compute-engine/tests/test_pyscf_diagnostics.py`

**Interfaces:**
- AVAS result records whether and how a cap changed the selection.
- `natural_occupation_diagnostics` returns active occupations and boundary risk.
- `solve_nevpt2` returns CASCI/CASSCF reference energy, correction, total, convergence, and
  spin metadata using PySCF's strongly contracted NEVPT2 implementation.

- [ ] **Step 1: Add PySCF-marked failing tests on O2 for finite occupations and finite NEVPT2 correction, plus a pure test that a cap-forced selection is never silently trusted**
- [ ] **Step 2: Run tests locally and confirm pure failures/PySCF skips**
- [ ] **Step 3: Run GitNexus impact on `AvasResult` and `avas_active_integrals`, report risk, then implement lazy-PySCF diagnostics**
- [ ] **Step 4: Run the PySCF tests in the available Linux/CI environment when available and retain explicit skip evidence otherwise**

### Task 6: Colab R4/R5 Campaign and Negative Control

**Files:**
- Modify: `packages/compute-engine/notebooks/spin_gap_colab.py`
- Create: `packages/compute-engine/tests/test_spin_gap_trust.py`
- Modify: `packages/compute-engine/README.md`

**Interfaces:**
- `run_case` returns raw energies plus `trust_report`.
- Production execution writes `spin_gap.json` even for `UNTRUSTED` or `UNDECIDABLE`.
- The legacy Fe/STO-3G/Fe-3d case is explicitly named as a negative control.

- [ ] **Step 1: Add failing tests that load the notebook module and prove the negative control cannot emit `TRUSTED`, while same-Hamiltonian agreement remains present**
- [ ] **Step 2: Run focused tests and confirm the old gate produces the wrong trust behavior**
- [ ] **Step 3: Run GitNexus impact on `_solve_state`, `run_case`, and `main`; report blast radius, then wire the evidence ledger, SQD campaign, model labels, and JSON output**
- [ ] **Step 4: Update Colab commands, dependency versions, runtime expectations, and report interpretation**
- [ ] **Step 5: Run focused notebook tests**

### Task 7: Full Verification and Change Audit

**Files:**
- Verify all files above; no new production interface.

- [ ] **Step 1: Run `pytest packages/compute-engine/tests -q`**
- [ ] **Step 2: Run Ruff on the modified Python files**
- [ ] **Step 3: Run mypy on `packages/compute-engine/src/qdr_compute_engine`**
- [ ] **Step 4: Run the pure negative-control report generation and validate its JSON with Python's strict JSON decoder**
- [ ] **Step 5: Run GitNexus `detect_changes(scope="compare", base_ref="main")`; inspect every affected process**
- [ ] **Step 6: Re-read the design acceptance list and report any PySCF/Colab-only verification that could not run locally**


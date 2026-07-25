# Quantum Sampler Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, cost-aware experiment framework that measures the marginal value of a quantum determinant sampler in SQD while holding the chemistry problem and classical post-processing fixed.

**Architecture:** Add an isolated `qdr_compute_engine.audit` subsystem. Samplers emit validated raw determinant draws and a resource ledger; a frozen postprocessor consumes those draws; the runner executes paired, seeded comparisons; analysis computes predeclared estimands and confidence intervals; reporting writes content-addressed, paper-ready artifacts. Qiskit SQD and IBM Runtime remain optional adapters so the core audit and its tests run without cloud credentials or PySCF.

**Tech Stack:** Python 3.12, NumPy, SciPy, existing QDR SQD/provenance modules, pytest, Ruff, mypy; optional Qiskit 2.x, `qiskit-addon-sqd` 0.12.x, PySCF, and IBM Runtime on the Linux execution environment.

## Global Constraints

- Use test-driven development: every production behavior is preceded by a test that fails for the intended reason.
- Freeze Hamiltonian, electron sector, subspace ladder, recovery/postprocessing configuration, seeds, and analysis rule across sampler arms.
- Keep raw samples; never clean, deduplicate, or postselect inside a sampler.
- Report sample-count-matched and end-to-end resource views separately; never collapse heterogeneous costs into an invented scalar score.
- A quantum-benefit claim requires a paired confidence interval excluding zero on the preregistered primary estimand; otherwise report `NO_EVIDENCE` or `INCONCLUSIVE`.
- Hardware access is optional and credential-free imports must remain safe; absent Qiskit, PySCF, or IBM credentials is a named capability refusal.
- Existing scientific pipeline behavior must remain unchanged.
- Before editing an existing symbol, run GitNexus upstream impact analysis and warn before HIGH or CRITICAL changes.
- Before every commit, run GitNexus `detect_changes`; before completion compare the branch against `main`.

---

### Task 1: Immutable audit contracts and validation

**Files:**
- Create: `packages/compute-engine/src/qdr_compute_engine/audit/__init__.py`
- Create: `packages/compute-engine/src/qdr_compute_engine/audit/models.py`
- Test: `packages/compute-engine/tests/test_audit_models.py`

**Interfaces:**
- Produces: `SamplerKind`, `DatasetSplit`, `AuditBudget`, `ResourceLedger`, `AuditProblem`, `SampleBatch`, `PostprocessResult`, `AuditObservation`.
- `AuditProblem` carries a `QubitHamiltonian`, `(n_alpha, n_beta)`, exact/reference energy, split, and immutable descriptors.
- `SampleBatch` carries raw `np.int64` configurations, requested/returned shots, seed, source identity, ledger, and metadata.

- [ ] **Step 1: Write model tests**

```python
def test_sample_batch_preserves_raw_duplicates_and_wrong_sector_values() -> None:
    raw = np.array([12, 12, 0, 3], dtype=np.int64)
    batch = SampleBatch(
        source_id="recorded",
        kind=SamplerKind.HARDWARE_QUANTUM,
        configurations=raw,
        n_qubits=4,
        shots_requested=4,
        seed=7,
    )
    assert batch.configurations.tolist() == [12, 12, 0, 3]
    assert batch.shots_returned == 4


def test_resource_ledger_rejects_negative_measurements() -> None:
    with pytest.raises(ValidationError, match="cpu_seconds"):
        ResourceLedger(cpu_seconds=-0.1)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest packages/compute-engine/tests/test_audit_models.py -v`

Expected: collection fails because `qdr_compute_engine.audit.models` does not exist.

- [ ] **Step 3: Implement frozen, JSON-safe contracts**

```python
class SamplerKind(StrEnum):
    ORACLE = "oracle"
    IDEAL_QUANTUM = "ideal_quantum"
    NOISY_QUANTUM = "noisy_quantum"
    HARDWARE_QUANTUM = "hardware_quantum"
    CLASSICAL_SELECTED = "classical_selected"
    CLASSICAL_EXCITATION = "classical_excitation"
    NEGATIVE_CONTROL = "negative_control"


@dataclass(frozen=True)
class ResourceLedger:
    circuit_evaluations: int | None = None
    qpu_seconds: float | None = None
    cpu_seconds: float | None = None
    wall_seconds: float | None = None
    peak_memory_bytes: int | None = None
```

Validate finite non-negative measurements, positive budgets, configuration bounds `[0, 2**n_qubits)`, exact shot accounting, reference-energy finiteness, and valid particle sectors. Convert arrays and mappings defensively so callers cannot mutate recorded evidence.

- [ ] **Step 4: Run model tests and the existing SQD tests**

Run: `.venv\Scripts\python.exe -m pytest packages/compute-engine/tests/test_audit_models.py packages/compute-engine/tests/test_sqd.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

Commit: `feat(audit): add immutable experiment contracts`

---

### Task 2: Deterministic sampler arms and recorded-sample ingestion

**Files:**
- Create: `packages/compute-engine/src/qdr_compute_engine/audit/samplers.py`
- Test: `packages/compute-engine/tests/test_audit_samplers.py`

**Interfaces:**
- Consumes: `AuditProblem`, `AuditBudget`, `SampleBatch`.
- Produces: `DeterminantSampler` protocol plus `ExactStateSampler`, `UniformSectorSampler`, `HartreeFockExcitationSampler`, and `RecordedSampleSampler`.
- Every sampler implements `sample(problem: AuditProblem, budget: AuditBudget, seed: int) -> SampleBatch`.

- [ ] **Step 1: Write deterministic and distribution tests**

```python
def test_exact_state_sampler_replays_the_same_draw_for_the_same_seed(h2_problem) -> None:
    sampler = ExactStateSampler(h2_exact_state)
    first = sampler.sample(h2_problem, AuditBudget(shots=128, subspace_dimensions=(1, 2)), 11)
    second = sampler.sample(h2_problem, AuditBudget(shots=128, subspace_dimensions=(1, 2)), 11)
    assert first.configurations.tolist() == second.configurations.tolist()


def test_uniform_sector_sampler_never_emits_the_wrong_particle_sector(h2_problem) -> None:
    batch = UniformSectorSampler().sample(
        h2_problem, AuditBudget(shots=256, subspace_dimensions=(1, 2, 4)), 5
    )
    assert set(batch.configurations).issubset({3, 6, 9, 12})
```

Also test that recorded samples retain duplicates and wrong-sector outcomes, the excitation sampler includes the Hartree–Fock determinant, invalid statevector dimensions refuse, and every returned ledger records CPU and wall time.

- [ ] **Step 2: Run the tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest packages/compute-engine/tests/test_audit_samplers.py -v`

Expected: collection fails because `audit.samplers` does not exist.

- [ ] **Step 3: Implement the samplers**

Use the existing qubit convention: qubit `q` occupies amplitude-index bit `n_qubits - 1 - q`; even qubits are alpha and odd qubits beta. `HartreeFockExcitationSampler` constructs the reference occupation, enumerates symmetry-valid single/double replacements, and samples from explicit user-supplied non-negative weights. `RecordedSampleSampler` accepts only an immutable raw integer array and provenance metadata; it does no scientific processing.

- [ ] **Step 4: Verify focused and regression tests**

Run: `.venv\Scripts\python.exe -m pytest packages/compute-engine/tests/test_audit_samplers.py packages/compute-engine/tests/test_sqd.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

Commit: `feat(audit): add deterministic sampler baselines`

---

### Task 3: Frozen QDR postprocessor and paired experiment runner

**Files:**
- Create: `packages/compute-engine/src/qdr_compute_engine/audit/postprocessing.py`
- Create: `packages/compute-engine/src/qdr_compute_engine/audit/runner.py`
- Test: `packages/compute-engine/tests/test_audit_runner.py`

**Interfaces:**
- Produces: `AuditPostprocessor` protocol, `QDRSubspacePostprocessor`, `run_audit_cell`, and `run_paired_audit`.
- `QDRSubspacePostprocessor.process(problem, batch, budget) -> tuple[PostprocessResult, ...]` delegates to `run_sqd_campaign`.
- `run_paired_audit` uses identical `(problem, budget, seed)` cells for every sampler and returns observations in deterministic order.

- [ ] **Step 1: Write freeze and pairing tests**

```python
def test_runner_passes_identical_problem_budget_and_seed_to_every_sampler(h2_problem) -> None:
    observations = run_paired_audit(
        problems=(h2_problem,),
        samplers=(uniform_sampler, exact_sampler),
        budgets=(AuditBudget(shots=128, subspace_dimensions=(1, 2, 4)),),
        seeds=(3, 7),
        postprocessor=QDRSubspacePostprocessor(),
    )
    assert {(o.source_id, o.seed) for o in observations} == {
        ("uniform-sector", 3), ("uniform-sector", 7),
        ("exact-state", 3), ("exact-state", 7),
    }
```

Also assert the runner rejects duplicate source ids, unordered/non-nested dimension ladders, a postprocessor that changes the Hamiltonian identity, and energy below the known reference.

- [ ] **Step 2: Run the tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest packages/compute-engine/tests/test_audit_runner.py -v`

Expected: collection fails because the runner and postprocessor do not exist.

- [ ] **Step 3: Implement postprocessing and orchestration**

Hash the normalized Hamiltonian/problem specification once per problem. Measure sampling, postprocessing, and total wall times independently. Preserve sampler-provided resource fields, adding measured CPU/wall fields without overwriting hardware-reported QPU time. Store every SQD ladder step, sector-retention fraction, retained determinant identities, absolute energy error, and chemical-accuracy indicator.

- [ ] **Step 4: Verify focused and regression tests**

Run: `.venv\Scripts\python.exe -m pytest packages/compute-engine/tests/test_audit_runner.py packages/compute-engine/tests/test_sqd.py packages/qdr-core/tests/test_provenance.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

Commit: `feat(audit): run frozen paired sampler experiments`

---

### Task 4: Preregistered metrics, uncertainty, and decision rules

**Files:**
- Create: `packages/compute-engine/src/qdr_compute_engine/audit/analysis.py`
- Test: `packages/compute-engine/tests/test_audit_analysis.py`

**Interfaces:**
- Produces: `ComparisonStatus`, `SamplerComparison`, `log_budget_auc`, `paired_bootstrap_comparison`, and `compare_against_best_classical`.
- Primary estimand: paired difference `quantum AUC - classical AUC` for absolute energy error versus retained-subspace dimension on a logarithmic dimension axis; lower is better.
- Decision: `QUANTUM_BENEFIT` only when the two-sided 95% paired-bootstrap interval is below zero; `NO_EVIDENCE` when it contains zero; `CLASSICAL_BENEFIT` when it is above zero; `INCONCLUSIVE` for incomplete pairings or fewer than the configured minimum paired cells.

- [ ] **Step 1: Write literal numerical tests**

```python
def test_log_budget_auc_matches_hand_integrated_two_point_curve() -> None:
    assert log_budget_auc((1, 4), (0.010, 0.002)) == pytest.approx(0.006)


def test_paired_comparison_detects_a_consistent_lower_quantum_auc() -> None:
    comparison = paired_bootstrap_comparison(
        quantum=np.array([0.2, 0.3, 0.4]),
        classical=np.array([0.8, 0.9, 1.0]),
        confidence=0.95,
        n_resamples=5_000,
        seed=17,
        min_pairs=3,
    )
    assert comparison.status is ComparisonStatus.QUANTUM_BENEFIT
    assert comparison.interval_high < 0.0
```

Add tests for missing pairs, NaN/inf refusal, deterministic bootstrap output, classical benefit, ties, best-classical selection using development data only, and no silent metric switching.

- [ ] **Step 2: Run the tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest packages/compute-engine/tests/test_audit_analysis.py -v`

Expected: collection fails because `audit.analysis` does not exist.

- [ ] **Step 3: Implement analysis**

Integrate with the trapezoid rule after sorting and validating strictly increasing positive dimensions; normalize by log-axis width so systems with different ladders are comparable. Pair on `(problem_id, shots, seed)`. Bootstrap paired cells, never individual observations. Choose the best classical source exclusively on `DatasetSplit.DEVELOPMENT`; evaluate the locked source on `DatasetSplit.HELD_OUT`.

- [ ] **Step 4: Verify the analysis suite**

Run: `.venv\Scripts\python.exe -m pytest packages/compute-engine/tests/test_audit_analysis.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

Commit: `feat(audit): add preregistered paired analysis`

---

### Task 5: Qiskit SQD compatibility adapter

**Files:**
- Modify: `pyproject.toml`
- Modify: `packages/compute-engine/pyproject.toml`
- Create: `packages/compute-engine/src/qdr_compute_engine/audit/qiskit_sqd.py`
- Test: `packages/compute-engine/tests/test_audit_qiskit_sqd.py`

**Interfaces:**
- Produces: `FermionicAuditProblem`, `QiskitSQDPostprocessor`, `qiskit_sqd_available`, and `integers_to_qiskit_bit_array`.
- The adapter calls `qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian(hcore, eri, bit_array, samples_per_batch=..., norb=..., nelec=(n_alpha, n_beta), num_batches=..., max_iterations=..., seed=...)`.
- Qiskit result energy receives only the explicitly stored constant/nuclear energy shift.

- [ ] **Step 1: Add contract tests before dependencies**

```python
def test_integer_conversion_pins_qiskit_spin_block_order() -> None:
    bit_array = integers_to_qiskit_bit_array(np.array([12, 3]), n_qubits=4)
    assert bit_array.get_bitstrings() == ["1100", "0011"]


def test_adapter_refuses_when_optional_stack_is_absent(monkeypatch) -> None:
    monkeypatch.setattr(qiskit_module, "_load_qiskit_sqd", lambda: None)
    with pytest.raises(BackendUnavailable, match="qiskit-addon-sqd"):
        QiskitSQDPostprocessor().process(problem, batch, budget)
```

Also test integral shapes, even qubit count, electron counts, energy-shift application, deterministic seed forwarding, and translation of result/subspace histories using a complete fake result object at the adapter boundary.

- [ ] **Step 2: Run the tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest packages/compute-engine/tests/test_audit_qiskit_sqd.py -v`

Expected: collection fails because the adapter does not exist.

- [ ] **Step 3: Add optional dependency groups**

Add a root `audit-qiskit` dependency group and matching compute-engine optional extra with `qiskit>=2.4,<3`, `qiskit-addon-sqd>=0.12.1,<0.13`, and `ffsim`/PySCF where supported. Keep imports lazy and Windows-safe. Regenerate `uv.lock`.

- [ ] **Step 4: Implement the adapter against official 0.12.x API**

Use the addon’s `diagonalize_fermionic_hamiltonian` as the canonical full-recovery path. Preserve the raw sample artifact separately from Qiskit’s recovered configurations. Record addon version, solver settings, number of recovery iterations, batch results, occupancies, and convergence status.

- [ ] **Step 5: Verify local refusal tests and Linux integration marker**

Run: `.venv\Scripts\python.exe -m pytest packages/compute-engine/tests/test_audit_qiskit_sqd.py -q`

Expected on Windows without the optional stack: contract tests pass and the real integration test is skipped with an explicit reason. Expected on Linux with `uv sync --group chem --group audit-qiskit`: all adapter tests pass.

- [ ] **Step 6: Commit**

Commit: `feat(audit): integrate canonical Qiskit SQD`

---

### Task 6: Immutable artifacts, reports, and command-line execution

**Files:**
- Create: `packages/compute-engine/src/qdr_compute_engine/audit/artifacts.py`
- Create: `packages/compute-engine/src/qdr_compute_engine/audit/report.py`
- Create: `packages/compute-engine/src/qdr_compute_engine/audit/__main__.py`
- Test: `packages/compute-engine/tests/test_audit_artifacts.py`
- Test: `packages/compute-engine/tests/test_audit_report.py`

**Interfaces:**
- Produces: `build_audit_manifest`, `write_audit_bundle`, `load_audit_bundle`, `build_audit_markdown`, `write_audit_figures`, and CLI `python -m qdr_compute_engine.audit`.
- Bundle contains protocol version, immutable problem hashes, sampler metadata, raw draws, observations, comparison decisions, environment versions, Git commit, and artifact hashes.

- [ ] **Step 1: Write round-trip and tamper-detection tests**

```python
def test_bundle_round_trip_preserves_raw_samples_and_full_precision(tmp_path) -> None:
    paths = write_audit_bundle(audit_result, tmp_path)
    loaded = load_audit_bundle(paths.manifest)
    assert loaded.observations == audit_result.observations
    assert loaded.raw_samples["exact-state/seed-7"].tolist() == [12, 12, 3]


def test_bundle_rejects_a_modified_sample_file(tmp_path) -> None:
    paths = write_audit_bundle(audit_result, tmp_path)
    paths.samples.write_bytes(b"tampered")
    with pytest.raises(ValidationError, match="digest"):
        load_audit_bundle(paths.manifest)
```

Also test deterministic bundle hashing, machine-readable comparison status, report warnings for insufficient pairs, cost tables with `n/a` rather than zero for unmeasured resources, and CLI non-zero exit on invalid protocol.

- [ ] **Step 2: Run the tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest packages/compute-engine/tests/test_audit_artifacts.py packages/compute-engine/tests/test_audit_report.py -v`

Expected: collection fails because artifact/report modules do not exist.

- [ ] **Step 3: Implement artifact and report generation**

Use existing canonical JSON and content hashing. Store raw integer samples in compressed NumPy format and hash the exact bytes. Generate Markdown tables for energy error, sector retention, determinant discovery, wall/CPU/QPU resources, and paired confidence intervals. Produce vector-friendly PDF and PNG figures for convergence curves and paired effect intervals.

- [ ] **Step 4: Implement CLI protocol parsing**

Accept a JSON protocol path and output directory. Refuse unknown fields, duplicate sampler ids, missing held-out split, unseeded stochastic arms, and attempts to overwrite a bundle whose manifest hash differs. Print written artifact paths and comparison status.

- [ ] **Step 5: Verify report and CLI tests**

Run: `.venv\Scripts\python.exe -m pytest packages/compute-engine/tests/test_audit_artifacts.py packages/compute-engine/tests/test_audit_report.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

Commit: `feat(audit): emit reproducible paper artifacts`

---

### Task 7: Reproducible H2 pilot and research protocol

**Files:**
- Create: `research/sampler-audit/protocol-v1.json`
- Create: `research/sampler-audit/README.md`
- Create: `research/sampler-audit/preregistration.md`
- Create: `packages/compute-engine/tests/test_audit_pilot.py`
- Modify: `README.md`

**Interfaces:**
- Protocol fixes development/held-out H2 bond lengths, sampler arms, shot budgets, dimension ladder, 30 seeds, primary estimand, confidence level, and minimum-pair rule.
- Pilot command writes only to `research/sampler-audit/results/<run-hash>/`; generated result bundles remain ignored until a reviewed release artifact is intentionally committed.

- [ ] **Step 1: Write an end-to-end pilot test**

```python
def test_h2_pilot_produces_paired_decision_and_verifiable_bundle(tmp_path) -> None:
    result = run_protocol(load_protocol(PROTOCOL_PATH), output_dir=tmp_path)
    assert result.primary_comparison.status in set(ComparisonStatus)
    assert result.n_pairs == 30 * len(result.held_out_problem_ids)
    assert load_audit_bundle(result.bundle.manifest).manifest_hash == result.manifest_hash
```

- [ ] **Step 2: Run the pilot test and verify RED**

Run: `.venv\Scripts\python.exe -m pytest packages/compute-engine/tests/test_audit_pilot.py -v`

Expected: fails because the protocol and end-to-end loader do not exist.

- [ ] **Step 3: Add the locked protocol**

Development instances: H2 at 0.735 Å and 1.50 Å. Held-out instances: H2 at 1.00 Å and 2.50 Å. Arms: exact-state oracle, Hartree–Fock excitation baseline, and uniform-sector negative control. Budgets: 128, 512, and 2048 shots; nested retained dimensions `(1, 2, 4)`; seeds `0..29`. Primary comparison uses only held-out cells and compares the designated quantum/oracle arm against the classical arm; the oracle result is explicitly labeled methodological validation, not hardware quantum advantage.

- [ ] **Step 4: Document scientific interpretation**

State that the H2 pilot validates causal isolation, statistics, and artifacts only. It cannot establish useful quantum advantage because exact sampling is classically trivial and the sector has four determinants. Define the promotion gate: proceed to N2/FeS and free QPU only after deterministic replay, independent Qiskit agreement, and complete cost records pass.

- [ ] **Step 5: Run the pilot and verify its bundle**

Run: `.venv\Scripts\python.exe -m qdr_compute_engine.audit research/sampler-audit/protocol-v1.json --output research/sampler-audit/results`

Expected: exit 0, content-addressed bundle written, all hashes verify, report clearly labels the result as a pilot.

- [ ] **Step 6: Commit source protocol and documentation**

Commit: `feat(audit): add reproducible sampler audit pilot`

---

### Task 8: Full quality and scope verification

**Files:**
- Modify only files required to resolve verified failures.

- [ ] **Step 1: Run all tests**

Run: `.venv\Scripts\python.exe -m pytest -q -rs`

Expected on Windows: all executable tests pass; only explicitly marked PySCF/Linux/Qiskit integration tests skip.

- [ ] **Step 2: Run static analysis**

Run: `.venv\Scripts\python.exe -m mypy packages/qdr-core/src packages/compute-engine/src`

Run: `.venv\Scripts\python.exe -m ruff check .`

Run: `.venv\Scripts\python.exe -m ruff format --check .`

Expected: exit 0 for every command.

- [ ] **Step 3: Verify deterministic replay**

Run the pilot twice into separate temporary directories and compare manifest hashes, excluding only explicitly recorded wall-clock measurements. Scientific samples, energies, comparisons, and problem hashes must match bit-for-bit.

- [ ] **Step 4: Inspect branch impact**

Run GitNexus `detect_changes(scope="compare", base_ref="main", worktree=<worktree>)`. Review every changed symbol and affected execution flow; investigate any unexpected module.

- [ ] **Step 5: Review the evidence boundary**

Confirm the report never calls the exact-state H2 pilot “quantum advantage,” absent resource fields are `null`/`n/a`, held-out systems were not used for baseline selection, raw draws are recoverable, and every decision maps to the preregistered rule.

- [ ] **Step 6: Commit verified corrections**

Commit: `test(audit): verify research evidence pipeline`


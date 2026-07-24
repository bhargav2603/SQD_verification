# Chemistry Trust Pipeline Design

## Objective

Build an end-to-end, Colab-oriented embedded active-space/SQD pipeline that preserves R4
(SQD/QSCI) and R5 (transition-metal spin gaps) while making the principal output a
machine-checkable scientific verdict:

```text
TRUSTED | UNTRUSTED | UNDECIDABLE
```

The verdict must identify every failed or missing evidence item. Agreement between two
solvers using the same active-space Hamiltonian is evidence only for engine correctness;
it is never evidence that the Hamiltonian represents the chemistry.

## Decision Model

The implementation uses a fail-closed evidence ledger rather than a weighted confidence
score. A weighted score could let several cheap numerical checks conceal one fatal chemical
defect.

- `UNTRUSTED`: at least one required diagnostic has observed a definite failure.
- `UNDECIDABLE`: no required diagnostic failed, but at least one required diagnostic lacks
  enough evidence.
- `TRUSTED`: every required diagnostic passed.

Each diagnostic carries a stable code, status, measured value, threshold, units, explanation,
and provenance metadata. JSON output is deterministic and contains no NaN or infinity.

## Required Evidence Axes

### 1. Numerical and solver integrity

- SCF convergence and, when available, internal stability.
- Requested electron number and spin sector.
- Spin purity: `|<S^2> - S(S+1)|` within a declared tolerance.
- Variationality of every SQD projected solve.
- Nested SQD subspaces whose energies are non-increasing.
- Final SQD increment, batch spread, and in-sector sample retention.
- Same-Hamiltonian QDR/CASCI agreement, explicitly labeled `engine_only`.

### 2. Active-space adequacy

- Required chemical valence shells are declared for the scientific question.
- AVAS selection records target labels, overlap threshold, selected electron/orbital counts,
  and whether a qubit cap forced the threshold upward.
- A cap may not silently delete a required shell. If the requested chemistry cannot fit, the
  result is `UNTRUSTED`, not a smaller "best effort" calculation.
- A nested active-space ladder must show the target observable (for R5, the spin gap)
  stabilizing within the configured tolerance.
- Natural-orbital occupations at the active-space boundary are checked: materially
  non-integer excluded orbitals mean the active space is incomplete.

### 3. Model adequacy

- Transition-metal quantitative claims cannot use a minimal basis.
- A spin gap must use the same electron count, basis family, relativistic convention,
  Hamiltonian construction, and comparable orbital spaces for both states.
- State-specific optimized or cited experimental geometries are required for an adiabatic
  gap. An illustrative common geometry can produce only an `UNTRUSTED` adiabatic claim or
  an explicitly labeled vertical/model result.
- Charged or environmentally sensitive targets must declare the environment model. Missing
  solvation for a requested solution-phase claim is a definite model mismatch.

### 4. Embedding adequacy

- The full-system recovery limit must reproduce the unembedded one-electron operator.
- The localized-orbital partition must be stable when the population threshold is perturbed.
- The target observable must be stable when the WF region is enlarged.
- If embedding is requested but only AO-level artifacts exist and no total-energy-consistent
  embedded Hamiltonian was solved, the verdict is `UNDECIDABLE`.

### 5. Correlation and cross-method evidence

- Quantitative transition-metal gaps require a post-active-space dynamic-correlation method
  such as strongly contracted NEVPT2, tailored CC, or a documented equivalent.
- The post-active correction must be converged with respect to the active-space ladder.
- At least two scientifically distinct estimates must agree within the declared tolerance.
  CASCI and QDR exact diagonalization of the same Hamiltonian count as one estimate.
- Literature values may be used as external validation only when geometry, charge,
  environment, basis/relativistic convention, and observable definition match.

## R4: Instrumented SQD

`solve_sqd` remains the projection/diagonalization primitive. A campaign wrapper will run
nested determinant subspaces, retain their actual configuration sets, and report:

- requested and retained configurations;
- full sector size and coverage;
- shot count and sector retention;
- energy per nested dimension;
- energy increments and monotonicity;
- replicated/batch energy mean and standard deviation;
- exactness flag and reference deviation when an exact reference is feasible.

The trust layer refuses to call a sampled answer converged merely because the same
determinants were repeatedly sampled.

## R5: Spin-Gap Campaign

The Colab runner will stop presenting `[FeF6]4-`/STO-3G/Fe-3d-only as a quantitative result.
It will:

1. Keep O2 as a fast open-shell engine sanity case.
2. Preserve the inadequate Fe configuration as a negative control that must emit
   `UNTRUSTED` with `MINIMAL_BASIS` and `MISSING_REQUIRED_ORBITALS`.
3. Define production cases with nonminimal basis sets, metal and ligand valence targets,
   an active-space ladder, SQD convergence evidence, CASCI where feasible, and NEVPT2
   post-processing.
4. Label vertical and adiabatic gaps distinctly.
5. Write a self-contained JSON report containing raw state energies, corrections,
   diagnostics, verdict, versions, and input configuration.

No code can promise an accurate transition-metal number from an arbitrary input. The
deliverable instead makes accuracy conditional on explicit evidence and refuses a
quantitative claim when the requested calculation is beyond the available active-space,
post-processing, geometry, or embedding evidence.

## Integration

- New pure trust types and evaluator live under `qdr_compute_engine.validation`.
- Stage 9 adds a trust report without removing the existing error budget.
- SQD outcome metadata is extended compatibly; existing fields keep their meanings.
- The Colab script uses public compute-engine functions rather than duplicating verdict
  rules.
- PySCF-dependent diagnostics are isolated behind lazy imports and marked for Linux CI.
- Pure evaluator and SQD campaign tests run on every platform.

## Testing and Acceptance

- A synthetic all-pass evidence set returns `TRUSTED`.
- A same-Hamiltonian cross-check plus Fe/STO-3G/3d-only evidence returns `UNTRUSTED` with
  the chemistry failures, proving the original false-positive gate is closed.
- Missing active-space convergence returns `UNDECIDABLE`.
- Definite failure dominates missing evidence.
- Nested SQD energies are variational and non-increasing; a fabricated increase is rejected.
- Existing SQD, open-shell, AVAS, embedding, and pipeline tests remain green.
- Linux/PySCF tests verify natural occupations, spin purity, NEVPT2 plumbing, and the
  negative-control verdict.
- The Colab entry point always writes JSON, including when the verdict is not `TRUSTED`.
- GitNexus `detect_changes` reports only the validation, SQD, pipeline-reporting, and
  spin-gap flows expected by this design.

## Literature Basis

- David et al., arXiv:2607.16389: open-shell SQD, S-CORE sector recovery, spin constraints,
  active-space ladders up to 50 qubits, SCF stability, batch diagnostics, and solvent
  self-consistency.
- Do et al., arXiv:2606.06015: projection WF-in-DFT, subsystem-growth convergence, and
  post-active-space TCC/NEVPT2/AC0; reliability depends on subsystem definition.
- Sundar et al., arXiv:2601.10813: chemistry-dependent algorithmic failure modes and
  problem-adapted validation.
- Reinholdt et al., JCTC 2025, DOI 10.1021/acs.jctc.5c00375: QSCI/SQD sampling saturation
  and determinant-selection inefficiency.
- Sayfutyarova et al., JCTC 2017, DOI 10.1021/acs.jctc.7b00128: AVAS construction and its
  explicitly non-universal scope.


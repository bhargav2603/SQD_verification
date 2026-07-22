# Remote execution

All chemistry runs on a Linux host. This document is the whole setup.

## Why remote

PySCF publishes manylinux and macOS wheels only. On Windows `uv pip install pyscf`
resolves to the **source distribution** and the build fails at CMake — verified, not
assumed. Since stage 3 (integrals) and every classical reference energy go through
PySCF, no molecule beyond the committed H₂ golden file can be computed on Windows.

The engine itself stays importable everywhere: PySCF is confined to
[`integrals/pyscf_integrals.py`](../packages/compute-engine/src/qdr_compute_engine/integrals/pyscf_integrals.py)
and [`solvers/classical_baseline.py`](../packages/compute-engine/src/qdr_compute_engine/solvers/classical_baseline.py),
and imported lazily inside functions. Stages 4–9 are exercised on any platform because
the golden file stores *integrals*, not only energies.

## Setup

```bash
git clone <repo> && cd Project_QDR
bash tools/remote/setup.sh
```

The script installs `uv`, syncs dependencies, then **verifies** rather than assumes:
every scientific import resolves, PySCF reproduces the committed H₂ FCI reference to
1e-8 Ha, and no test skips. It exits non-zero if any of that fails.

### The one command that matters

```bash
uv sync --group chem      # correct
uv sync --extra chem      # WRONG — silently installs nothing
```

`--extra` resolves extras against the root `[project]` table. This workspace root is
*virtual* and has no `[project]`, so `--extra chem` reports "no changes" and installs
no PySCF. The platform marker hides the mistake on Windows, where PySCF would be
excluded anyway. `chem` is therefore also declared as a root **dependency group**,
which is unambiguous everywhere.

## Sizing the VM

Cost is set by two things that both scale as `2ⁿ`, and the second one binds first:

| | memory |
|---|---|
| statevector | `2ⁿ × 16 bytes` |
| sparse Hamiltonian | `n_terms × 2ⁿ × 16 bytes` |

The Hamiltonian dominates. Measured: a 16-qubit case needed **5.66 GiB for the sparse
operator alone** and died there, not in the simulator.

| Active space | Qubits | Practical RAM |
|---|---|---|
| ≤ CAS(6,6) | ≤ 12 | 4 GB |
| CAS(8,7) | 14 | 8–16 GB |
| CAS(8,8) | 16 | 32 GB, and see the note below |

**Recommended: 16 GB, 8 vCPU.** That covers everything through 14 qubits comfortably.

Going past 14 qubits wants a matrix-free Hamiltonian (apply Pauli strings directly to
the statevector instead of materialising the operator) — ~1 MB instead of ~6 GB, at
the cost of a slower `H|ψ⟩`. Not yet implemented; it is the next bottleneck.

Cores help less than expected: the statevector kernel is memory-bandwidth-bound, and a
single VQE run is sequential. Parallelism belongs *across* cases — the FMO workload is
many small independent solves, not one large one.

## Running the validation suite

```bash
uv run python -m qdr_compute_engine.benchmarks --suite h2-scan  --out results/
uv run python -m qdr_compute_engine.benchmarks --suite standard --out results/
uv run python -m qdr_compute_engine.benchmarks --suite all      --out results/
```

| Flag | Purpose |
|---|---|
| `--suite {h2-scan,standard,all}` | which systems |
| `--system NAME` | one system; repeatable; overrides `--suite` |
| `--optimizer COBYLA` | force the derivative-free path (default: L-BFGS-B + adjoint gradients) |
| `--no-full-fci` | skip the full-space reference; the active-space error is then reported **unknown**, never zero |
| `--fail-fast` | stop at the first failure instead of recording and continuing |

Long runs should be detached:

```bash
nohup uv run python -m qdr_compute_engine.benchmarks --suite all \
    --out results/ > results/run.log 2>&1 &
tail -f results/run.log
```

Results are persisted after **every** case, so a sweep that dies at case 19 of 21 still
leaves 18 usable results.

## Outputs

`results/validation.json` — every field at full precision, plus the environment the
numbers were measured on. This is the artifact; the tables are a view of it.

`results/validation.md` — rendered tables: energies with the error decomposition,
resources and timings, the H₂ dissociation curve, and every warning verbatim.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | every case completed with no warnings |
| 1 | completed, but some case warned or failed — **not a clean run** |
| 2 | environment unusable (PySCF missing) |

A run that warns exits non-zero deliberately. A variational-principle violation or a
disagreement between the two reference paths is a bug, and it should not be possible to
report such a sweep as green.

## Reading the results

Three energies are computed per case, and the error is split rather than reported as one
number:

```
E_VQE  - E_CASCI    ansatz + Trotter + optimiser   (the quantum path)
E_CASCI - E_FCI     active-space truncation        (usually the larger term)
```

A bare "VQE vs FCI" comparison hides which of the two is responsible. The first term is
non-negative by the variational principle — a negative value is a bug and is flagged,
not smoothed over.

No reference energy is hard-coded. Every one is computed at run time by PySCF in the same
basis and active space, so it can be checked and cannot drift when a geometry changes.

## Limitations, stated plainly

- **Closed-shell only.** Multiplicity ≠ 1 and odd electron counts are rejected at stage 1
  and again at the RHF call. `scf.RHF(mol)` silently returns ROHF for `spin != 0`, and
  nothing downstream would catch it.
- **STO-3G by default.** Adequate for validating against FCI in the same basis. **Not**
  adequate for interaction or binding energies — minimal-basis BSSE exceeds the
  interaction itself, and counterpoise correction is not yet implemented.
- **No FMO assembly.** Stage 2 fragments; stages 3–9 then compute the supersystem. The
  pipeline warns about this rather than labelling a supersystem energy as an FMO result.
- **Practical ceiling is 14 qubits**, not the `MAX_QUBITS = 40` in `qdr_core.limits`
  (a memory bound that assumes nothing about wall-clock) nor the statevector driver's
  self-declared 24.

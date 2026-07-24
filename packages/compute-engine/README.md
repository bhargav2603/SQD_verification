# qdr-compute-engine

The science. A 9-stage pipeline and the `SolverBackend` driver seam.

Everything here is pure computation: no database, no queue, no HTTP.

> **Scope, stated up front.** There are two traversals of the same stages:
> `pipeline.run()` computes the **supersystem**, and `pipeline.fmo_runner.run()` computes
> the **two-body fragment expansion** — one solve per monomer and per dimer, assembled via
> `E = Σ E_I + Σ_{I<J} (E_IJ − E_I − E_J)`.
>
> The expansion has **no electrostatic embedding**. Genuine FMO polarises each fragment in
> the field of the others; this does not, which makes it FMO2-without-ESP, equivalently
> MBE2. Every result records `scheme="fmo2-no-esp"` and carries the caveat as a warning, so
> a number from here cannot be mistaken for a fully embedded FMO energy.
>
> `services/compute-worker` and the rest of the control plane are **empty scaffolding**
> (0-byte placeholder files). The engine is driven directly, via
> `python -m qdr_compute_engine.benchmarks` or the two `run()` entry points.

## The pipeline

```
stage 0  fci_gate               correctness gate — NOT part of the chain
────────────────────────────────────────────────────────────────────────
stage 1  geometry_pose          parse / validate / centre
stage 2  fmo_fragmenter         bond perception → monomers + dimer pairs
stage 3  integrals              PySCF RHF → h1, h2, E_nuc        [needs PySCF]
stage 4  active_space_selector  CAS(n,m) window + the ≤40-qubit cap
stage 5  fermionic_hamiltonian  spin-orbital second quantisation
stage 6  qubit_mapper           Jordan-Wigner
stage 7  ansatz                 UCCSD → gate IR
stage 8  solver                 dispatch through the driver seam
stage 9  validator              ΔE_bind + error budget
```

Each stage is a plain `(StageContext) -> StageContext` function. The context is immutable;
a stage returns a new one. `pipeline.py` is an ordered list and a timing loop, nothing more.

Stage 0 sits outside the chain deliberately: it is a check on *the engine*, not a step in
a user's calculation.

## Two architectural decisions worth knowing

**The gate IR.** `types.Circuit` is an ordered list of `Gate`s. Ansätze emit it, drivers
consume it, and no vendor circuit type appears anywhere in the pipeline. This is what
makes ADR-0003's "one interface" claim true rather than aspirational — see
[`drivers/README.md`](src/qdr_compute_engine/drivers/README.md).

**Qubit ordering.** Spin orbital `2p` is α and `2p+1` is β; qubit `q` is spin orbital `q`;
amplitude index bit `n-1-q` is qubit `q`. This is OpenFermion's convention, adopted
wholesale rather than translated. A mismatch between our simulator and
`get_sparse_operator` would produce a *believable but wrong* energy with no symptom, so
`tests/test_qubit_mapper.py` pins it against hand-built states.

## The H₂ gate

The engine's central correctness claim: for a system whose exact answer is known, the
quantum path reproduces FCI to chemical accuracy.

```bash
uv run pytest packages/compute-engine/tests/test_stage_00_fci_gate.py -v
```

Current numbers for H₂/STO-3G at 0.735 Å (`tests/golden/h2_fci_reference.json`):

| Quantity | Value (Ha) |
|---|---|
| E_nuc | 0.719968994426 |
| E_HF | −1.116998990392 |
| E_FCI | −1.137306028320 |
| E_corr | −0.020307037928 |

Circuit at θ=0 reproduces E_HF to ~1e-16, and VQE reaches E_FCI to ~1e-15.

## Where this runs

**Chemistry runs on a Linux host** — see [`docs/REMOTE_EXECUTION.md`](../../docs/REMOTE_EXECUTION.md).

PySCF publishes manylinux and macOS wheels only. On Windows `uv pip install pyscf`
resolves to the sdist and the build fails at CMake (verified, not assumed). PySCF is
therefore confined to two modules — `integrals/pyscf_integrals.py` and
`solvers/classical_baseline.py` — and imported lazily inside functions, so the package
stays importable everywhere.

The golden file stores the **integrals**, not just the energies. That is what lets stages
5–9 run on any platform: tests load `h1`/`h2`/`e_nuc` from JSON instead of computing them.
Only stages 0 and 3 need PySCF, and they are marked `requires_pyscf` and auto-skipped when
it is absent.

```bash
uv sync                  # Windows: no PySCF, 316 tests run, 5 skip
uv sync --group chem     # Linux/macOS: full suite, nothing skips
uv run pytest packages/compute-engine -q -rs
```

`--group chem`, **not** `--extra chem`: extras resolve against the root `[project]`
table, and this workspace root is virtual, so `--extra chem` silently installs nothing.

The golden numbers are **not** taken on faith. They are computed analytically by
`tests/reference/sto3g_h2.py` (closed-form Gaussian integrals; H₂'s RHF orbitals are fixed
by symmetry, so there is no SCF), and CI recomputes the same quantities with PySCF and
asserts agreement. Two independent derivations, cross-checked.

## What is implemented

| Area | Implemented | Registered only |
|---|---|---|
| Mappings | Jordan-Wigner | parity, Bravyi-Kitaev |
| Ansätze | UCCSD | hardware-efficient, ADAPT |
| Backends | statevector, classical | sampled, QPU |
| Fragmentation | molecular, explicit cuts | hydrogen caps for covalent cuts |
| Fragment expansion | FMO2 / MBE2, monomers + dimers, binding energy | electrostatic embedding (ESP), 3-body terms |
| Gradients | adjoint (statevector) | parameter-shift, sampled |
| Reference chemistry | FCI (PySCF) | CCSD, counterpoise correction |
| Spin | closed-shell singlet only | open-shell (UHF/ROHF) — **hard-rejected**, not approximated |

"Registered only" means a real interface, a registry entry, honest capability flags, and a
`NotImplementedError` / `BackendUnavailable` on use. Each module's docstring says what
specifically is missing and why it did not ship on the first pass. Nothing here silently
returns a plausible number it cannot justify.

## Scientific trust verdicts

Stage 9 now emits a strict, machine-readable `TRUSTED`, `UNTRUSTED`, or
`UNDECIDABLE` verdict. Quantitative and embedded-chemistry profiles require
evidence for basis and active-space convergence, orbital content and nesting,
spin, dynamic correlation, genuinely independent methods, sampling provenance,
geometry/environment, and embedding recovery. Missing evidence cannot silently
pass, and agreement between two solvers for the same Hamiltonian is classified
as an engine check rather than independent chemical validation.

`notebooks/spin_gap_colab.py` is the Colab entry point. It retains the
Fe-3d-only/STO-3G octahedral complex as an explicitly untrusted negative
control, and runs the FeS active-space/basis/dynamic-correlation/SQD campaign as
an instrumented production case. Every run writes `spin_gap_run.json`, including
failed runs. A converged number remains `UNDECIDABLE` until all evidence required
by its selected trust profile has actually been measured.

## Known limitations

- **Projection embedding is not yet a total-energy solver.** Stage 3b validates
  the partition and embedding operators, but it does not yet transform a
  complete embedded Hamiltonian into the active-space energy pipeline. The
  embedded trust profile therefore requires separate partition-recovery and
  stability evidence and fails closed when it is absent.

- **No electrostatic embedding in the fragment expansion.** `fmo_runner` computes FMO2 /
  MBE2 — isolated monomers and dimers, no mutual polarisation. For hydrogen-bonded systems
  the missing embedding is on the same order as the pair interaction itself. Results carry
  `scheme="fmo2-no-esp"` and say so.
- **No counterpoise correction, and STO-3G by default.** Interaction and binding energies
  are therefore **not** quantitatively meaningful: minimal-basis BSSE exceeds the
  interaction being measured. The expansion runs end to end; the number it produces is not
  yet publishable chemistry.
- **Covalent cuts leave radicals.** `explicit_cuts` severs bonds without capping the
  dangling valence, so the resulting open-shell fragment is *rejected* by the closed-shell
  guard rather than silently computed.
- **Three-body and higher terms are not computed.** The two-body expansion is exact only
  when they vanish; runs with more than two fragments warn.
- **Active-space truncation error is not quantified.** It is usually the largest term in
  the budget; the validator records it as unknown rather than as zero, and warns. Truncating
  per fragment also breaks error cancellation in the pair term, which warns separately.

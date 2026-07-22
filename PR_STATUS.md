# PR_STATUS

Where the project actually is, as of **2026-07-22**. Branch `feat/compute-engine-core`.

Written to be read before touching anything. Claims here are things that have been run,
not things that are intended.

---

## One-paragraph summary

The **compute engine is real and verified**; everything else in the repo is empty
scaffolding. The quantum path (integrals → active space → Hamiltonian → Jordan-Wigner →
UCCSD → VQE) reproduces FCI for H₂, the two-body fragment expansion (FMO2) runs end to
end, and the statevector backend has an in-place kernel with analytic adjoint gradients.
**But exactly one real molecule has ever been computed** — H₂ at 0.735 Å — because PySCF
cannot install on Windows and the Linux VM does not exist yet. Everything else is written,
tested where testable, and unrun.

---

## Where the code is

| Path | State |
|---|---|
| `packages/compute-engine/` | **Real.** The engine. ~7,500 lines incl. tests. |
| `packages/qdr-core/` | **Real.** Errors, IDs, limits, provenance primitives. |
| `apps/api/`, `apps/web/`, `services/*` | **Empty.** 0-byte placeholder files. Not started. |
| `docs/adr/` | ADR-0003 and ADR-0006 written; **0000-0002, 0004, 0005, 0007 still empty.** |

Inside the engine:

- `pipeline/pipeline.py` — the supersystem chain, stages 1→9
- `pipeline/fmo_runner.py` — the fragment expansion (monomers + dimers → assembly)
- `drivers/statevector/` — `kernel.py` (fast, in-place), `adjoint.py` (gradients)
- `benchmarks/` — H₂ scan, LiH, BeH₂, H₂O + runner + report; **CLI entry point**
- `typedefs/` — the type definitions; import from `types.py`, never from here

## Verified state

```
406 passed, 12 skipped     # the 12 skips all need PySCF
ruff  clean (check + format)
mypy  clean, 87 source files
coverage 86%               # CI gate at 80%
```

**Nothing is committed.** The whole engine sits uncommitted on top of one commit named
"file strucutre". This is the single largest risk in the repo.

## What has actually been computed

- **H₂ / STO-3G at 0.735 Å** — the only real molecule. Golden integrals computed
  analytically (no PySCF), cross-checked against PySCF in CI.
- **Synthetic integrals, CAS(2,2)→CAS(8,8)** — timing only; energies meaningless.
- **Block-diagonal H₂×2** — validated the FMO wiring via a zero-interaction invariant.

Not yet run: the H₂ dissociation curve (18 points), LiH, BeH₂, H₂O, the water dimer. All
defined, all requiring the VM.

## Measured facts worth knowing

| | |
|---|---|
| Practical qubit ceiling | **14** — not the `MAX_QUBITS = 40` policy cap |
| What binds first | The **sparse Hamiltonian**, not the statevector: 5.66 GiB at 16 qubits |
| Kernel speedup | 1.7× (10q) → 4.2× (14q) |
| Adjoint gradient speedup | 36× (54 params) → 127× (204 params) |
| Recommended VM | 16 GB RAM, 8 vCPU |

## Known limitations — stated, not hidden

- **Closed-shell only.** Multiplicity ≠ 1 and odd electron counts are hard-rejected at
  stage 1 and again at the RHF call. No radicals, no metals, no triplets.
- **FMO has no electrostatic embedding.** It is FMO2-without-ESP ≡ MBE2. Results carry
  `scheme="fmo2-no-esp"`.
- **No counterpoise correction; STO-3G default.** Interaction and binding energies are
  therefore **not quantitatively meaningful** — minimal-basis BSSE exceeds the interaction
  being measured. The plumbing is correct; the chemistry is not yet publishable.
- **Covalent cuts leave radicals** — rejected rather than capped.
- **No 3-body terms.** Irrelevant for dimers; matters for clusters and per-residue work.
- **`provenance/artifact_writer.py` is unreachable.** Implemented, tested, called by
  nothing.
- **The engine assumes energy-ordered MO integrals** and does not enforce it. Feeding it
  fragment-ordered integrals produces a plausible wrong energy. (Cost me a real debugging
  session; PySCF always returns energy-ordered, so production is unaffected.)

## Next favorable steps, in order

1. **Commit and open a PR.** ~7,500 lines are unversioned. CI now triggers on `feat/**`,
   so this is also what makes the H₂ gate gate anything for the first time.
2. **Stand up the Linux VM.** `bash tools/remote/setup.sh`. Nothing scientific proceeds
   without it. Use `uv sync --group chem`, *not* `--extra chem` (the latter silently
   installs nothing — the workspace root is virtual).
3. **Run the validation suite.** `python -m qdr_compute_engine.benchmarks --suite all`.
   This turns one data point into the tables: H₂ curve, LiH, BeH₂, H₂O, with the error
   split into quantum-path vs active-space contributions.
4. **Add a CCSD baseline.** `solvers/classical_baseline.solve_ccsd` currently raises. Every
   reviewer asks "does VQE beat classical CCSD in the same active space?" — be able to
   answer, including if the answer is no.
5. **Counterpoise + a real basis (6-31G* floor).** The gate on every binding-energy claim.
6. **Electrostatic embedding.** What makes FMO2 behave like real FMO. Higher value than
   3-body terms.
7. **Matrix-free Hamiltonian.** Removes the 14-qubit wall; ~1 MB instead of ~6 GB.

Not on this list, deliberately: the control plane, the API, the web app. They are not
scientific blockers.

## Open engineering debt

- 6 of 8 ADRs still empty while live code cites them
- `apps/`, `services/` 0-byte trees retained by request; README says so explicitly
- No parallelism across FMO subsystems (the workload is embarrassingly parallel)
- Active-space selection is a contiguous orbital-index window; monomer and dimer windows
  are not consistent, so truncation error does not cancel in pair interactions
  (cf. arXiv 2402.17993)

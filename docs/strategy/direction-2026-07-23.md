# QDR — Direction Design Doc

**Date:** 2026-07-23 · **Branch:** `feat/compute-engine-core` · **Session:** YC office hours (startup + research)
**Status:** Recommended direction, pending your go/no-go on the beachhead.

This doc is a design artifact, not code. It records where QDR is, what the session
established, the market/science reality from the literature, and the next concrete moves.

---

## 1. Where QDR actually is

The compute engine is real and verified. Everything else (control plane, API, web,
6 of 8 ADRs) was empty scaffolding and has since been deleted from the tree.

- 9-stage pipeline + FMO2/MBE2 fragment expansion. Reproduces FCI for H2/STO-3G to ~1e-15.
- ~7,500 lines, 406 tests passing, mypy clean, 86% coverage.
- **Exactly one real molecule has ever been computed** (H2 at 0.735 A). PySCF won't
  install on Windows; the Linux VM doesn't exist yet.
- Practical qubit ceiling: **14** (not the 40 policy cap). The sparse Hamiltonian binds
  first: 5.66 GiB at 16 qubits.
- Closed-shell only. No metals, no radicals, no triplets (hard-rejected). No electrostatic
  embedding (FMO2-without-ESP). STO-3G default, no counterpoise. Plumbing correct;
  chemistry not yet publishable.
- **Biggest standing risk:** ~7,500 lines are uncommitted on top of one commit. Fix this
  independent of everything below.

## 2. The thesis (from the session)

Target **protein-ligand interactions** in pharma/materials. Compute the *targeted section*
(binding pocket) more accurately than classical, instead of the whole system. This is the
fragment/embedding play, and it is the most credible commercial thesis in quantum chemistry.

## 3. The one filter that governs everything

**Quantum only earns money where classical fails, and classical fails only on
strongly-correlated (multireference) systems.** The literature is unanimous on where
advantage can live: transition-metal active sites, heavy-metal complexes, multireference
excited states. Not organic drug pockets (kinases, GPCRs, proteases) — those are
closed-shell, weakly correlated, and classical DFT/CCSD/FMO already handles them faster
than any near-term quantum device.

Run drug-target classes through the filter:

| Target class | Correlated? | Classical status | Opening |
|---|---|---|---|
| Kinases, GPCRs, proteases (organic pockets) | No | DFT/classical FMO already good | None |
| Zn metalloenzymes (carbonic anhydrase, HDAC, MMPs) | Borderline | DFT shaky on Zn | Thin |
| Fe/Cu/Mn metalloenzymes (P450, NOS, tyrosinase) | **Yes** | DFT unreliable, multireference | **The wedge** |
| Photochem / excited states / conical intersections | **Yes** | Multireference, brutal | Real, not protein-ligand |

## 4. The finding that reframes the startup

**The current engine is aimed away from the only market where it can win.** Closed-shell +
no metals + no embedding + VQE points straight at the regime that's already solved
classically. Reaching the metalloenzyme wedge is a science project, not a config change.

### What the literature research established (2026-07-23)

1. **Filter confirmed by the field.** Advantage zone = strongly-correlated subsystems only.
   Sources: DDDT review 2026; MDPI review 2025 (PMC12249871).
2. **Demand is real and already captured by well-funded players.** Boehringer Ingelheim +
   PsiQuantum on metalloenzyme electronic structure for drug metabolism. Qubit
   Pharmaceuticals + Singapore CQT (2-year, 2026). Pasqal + Qubit Pharma. **Buyers partner
   with fault-tolerant-hardware companies and funded startups, not solo builders.** This is
   why cold outreach failed — there is no procurement door for one person with a simulator.
3. **The canonical target is fault-tolerant / 2035+.** Cytochrome P450 Compound I:
   ~4,900 logical qubits, ~1e9 Toffoli gates, 73 hours (Goings et al., PNAS 2022). Full
   P450 is a "mid-to-late 2030s application." QDR's wall is 14 qubits. Gap: ~3 orders of
   magnitude.

**Conclusion:** "Run a metalloenzyme and beat classical" is not a solo, near-term,
simulator project. As the *only* lane, there is no startup here — only a good paper.

## 5. The live lane QDR is already 60% inside: embedding + SQD

The field's near-term answer to the qubit gap is **quantum embedding** — solve the
strongly-correlated fragment on the quantum core, recover the rest classically. Embedding
collapses qubit counts:

- DMET-ESVQE: 68 qubits down to **16**, accuracy comparable to CCSD.
- (6e,6o) active space → **~10 qubits** after symmetry reduction.
- Frontier method (Nov 2025): **DMET + sample-based quantum diagonalization (SQD)** on
  "ligand-like molecules" (arXiv 2511.22158). Projection-based DFT embedding for NISQ
  (arXiv 2602.01994).

QDR already has: FMO fragmentation, a clean driver seam, active-space selection, a
Hamiltonian builder, an honest validator/error budget. **The two missing pieces are exactly
the frontier:**

1. **Real embedding** (DMET or projection-based DFT embedding) to replace FMO2-without-ESP.
   The missing ESP term is not a limitation — it is the entire near-term game.
2. **SQD/QSCI solver** alongside VQE. The field is leaving VQE (barren plateaus, doesn't
   scale) for sample-based diagonalization on real systems.

**Reachability meets:** matrix-free Hamiltonian (roadmap; ~1 MB vs ~6 GB) lifts the wall to
~24 qubits; embedding brings real fragments to 10-16. A drug-relevant, strongly-correlated
fragment is within simulator reach once embedding + matrix-free land.

**Honesty line:** on a classical simulator you are simulating the quantum computer, so you
will never "beat classical" there. The near-term deliverable is *faithfully reproducing the
frontier method on a drug-relevant fragment with honest error budgets, reproducibly.* That
is publishable and reputation-building. Advantage comes later, on fault-tolerant hardware,
and not solo.

## 6. Premises (session-confirmed framing)

1. Quantum earns money only on strongly-correlated systems. **Agreed.**
2. The current engine is aimed away from that beachhead; reaching it is science, not config. **Agreed.**
3. The binding constraint is market/science truth, not more code. **Agreed** (validate first).

## 7. Recommended direction — Embedding-first research beachhead

A solo builder's only currency in this field is published, reproducible credibility. The
startup, if it ever exists, is downstream of that, not parallel to it. So:

1. **Commit the repo first.** ~7,500 lines unversioned is the top risk. Commit + open a PR
   so CI and the H2 gate mean something. (Independent of everything else. Do it now.)
2. **Add real embedding.** Start with projection-based DFT embedding (simpler bath) or DMET,
   replacing FMO2-without-ESP. This is the missing physics *and* the frontier.
3. **Add an SQD/QSCI solver** through the existing driver seam, alongside VQE.
4. **Land the matrix-free Hamiltonian** to reach ~24 qubits.
5. **Produce ONE reproducible benchmark:** a documented, strongly-correlated, drug-relevant
   fragment (<=~16-20 qubits) where classical single-reference fails, computed with
   embedding+SQD vs classical CCSD in the same active space, with honest error budgets.
6. **Publish it** (arXiv + OSS repo + a clear README benchmark). This is the customer
   discovery a solo builder without a network can actually do — it brings the customers in.

### Alternatives considered

- **A — Pure customer discovery (interviews).** Rejected as primary: outreach already failed;
  no network into pharma. Revisit *after* a published benchmark gives you a reason to be
  answered.
- **B — Discovery + reachability spike.** Folded into the recommendation (steps 2-5 are the
  spike; the "discovery" is publishing).
- **C — Sell the diagnosis (reliability triage).** Reframe toward a tool that flags which of
  a pipeline's targets sit in the classically-unreliable regime, built on QDR's honesty
  tooling. **Keep as a live hypothesis** — it may be a stronger near-term product than the
  compute itself, and it needs no "beat classical" claim. Test it against the same
  literature and any practitioner you reach.

## 8. The assignment (concrete, solo-doable)

Do these in order. None require anyone to answer an email.

1. **This week:** commit the engine and open the PR. Remove the single largest risk.
2. **Read 3 papers, closely:** arXiv 2511.22158 (DMET-SQD, ligand-like), arXiv 2602.01994
   (projection DFT embedding for NISQ), Goings PNAS 2022 (P450 resource estimate). Write a
   one-page note: which embedding flavor fits QDR's driver seam, and what SQD needs.
3. **Find ONE target fragment:** a documented strongly-correlated, drug-relevant active site
   whose correlated fragment is <=~20 qubits and where classical single-reference is shown to
   fail. If it exists, it is your first benchmark and your entire pitch. If you genuinely
   can't find one at <=20 qubits, that is itself the answer, and the honest move is Fork A
   (research artifact) or Approach C (reliability triage).

## 9. Open questions for next session

- Which embedding flavor: DMET vs projection-based DFT embedding, given the driver seam?
- Is the reliability-triage reframe (Approach C) actually a stronger near-term product?
- Does any <=20-qubit drug-relevant fragment survive the "classical actually fails" test?

## CEO Review Outcome (2026-07-23, /plan-ceo-review, HOLD SCOPE)

### Reframe forced by the research
- **First-mover is gone.** IBM + RIKEN published SQD on iron-sulfur metalloenzyme
  active-site clusters ([2Fe-2S], [4Fe-4S]) up to 77 qubits in Science Advances, June 2025
  (Robledo-Moreno et al., eadu9991). QunaSys, PsiQuantum+Boehringer, Qubit Pharma are years
  and orders-of-magnitude of hardware ahead. Do not pitch "first mover."
- **Capability-lead on flagship metalloenzyme physics is unwinnable for a small simulator
  team.** The honestly-hard fragments are 40-72 qubits (above our ~24-qubit matrix-free
  ceiling); everything that fits <=24 qubits is classically solved. There is no "gradually
  impressive" middle rung on a simulator.
- **The moat is rigor, automation, and reproducibility**, not raw capability on hero systems.
  Never present a simulator result as "beating classical" — on a simulator you are simulating
  the quantum computer. Overclaiming is instant credibility death in front of experts.

### Team decision: step-by-step but fast, small molecules to big
Execute as an incremental ladder where **each rung hardens the differentiated
automation + rigor pipeline** (fragment -> active-space -> embedding -> solver, with honest
error budgets). No rung is wasted; each leaves behind a validated pipeline stage.

### Locked plan of record: rungs 1-4 (rung 5 = gated go/no-go; rung 6 = hardware ceiling)

Every rung carries a **known-answer correctness gate** (first-class scope): the rung is not
done until it reproduces a known-correct reference to chemical accuracy. This extends the
existing H2 FCI gate discipline to all new work.

| Rung | Build | Correctness gate (known-answer) | Effort |
|---|---|---|---|
| 1 | Stand up Linux VM (PySCF); run LiH, BeH2, H2O, H2 dissociation on the existing runner; add an automated FCI-reproduction test. **CCSD deferred** (see eng review) | Automated test: quantum-path error < 1e-6 + zero cross-check warnings + converged, on LiH/BeH2/H2O | ~1 team-wk |
| 2 | Replace FMO2-no-ESP with projection-based DFT embedding; add counterpoise + 6-31G* floor; validate on water dimer | Embedded energy vs full non-fragmented calc of same dimer + counterpoise | ~2 team-wk |
| 3 | Matrix-free Hamiltonian -> break 14q wall to ~24q | Bit-for-bit identical to sparse path on shared cases | ~1 team-wk |
| 4 | SQD/QSCI solver via existing driver seam, alongside VQE (NOISELESS sim) | SQD reproduces FCI/CCSD on rung 1-2 known systems | ~2 team-wk |
| 4.5 | Same SQD circuits on a NOISY simulator (Qiskit Aer noise model / IBM fake backends); optional one free-tier hardware sample | Configuration recovery survives realistic noise on a known system | ~1 team-wk |

- **Rung 5 (deferred, separate go/no-go after rung 4 proves the pipeline):** lift closed-shell
  hard-reject, add UHF/ROHF, target a single transition-metal active-site fragment <=24 qubits,
  run embedded via SQD, cross-check vs published DMRG-NEVPT2. This is the credibility artifact.
- **Rung 6 (out of scope, honest ceiling):** [2Fe-2S]-class (~40+ qubits) needs cloud hardware
  (IBM/Quantinuum) and enters IBM's published territory. Not a simulator project.

### Solver framing (post outside-voice + competitive scan)
- SQD stays in scope, framed as **quantum-ready capability, validated to match classical** —
  never "advantage," never "beat classical." Overclaiming is instant credibility death.
- Noise is sequenced, not skipped: noiseless sim (correctness) -> noisy sim (realism) ->
  hardware (proof). Same circuits, backend swap via the driver seam.
- Competitive reality (be honest with anyone you pitch): the method (QSCI) is QunaSys's
  open-sourced IP (`quri-parts-qsci`); DMET embedding is a shipped feature in Quantinuum's
  InQuanto; the live near-term frontier is the noise/hardware layer (Algorithmiq's TEM,
  IBM/RIKEN SQD on 77 qubits, Quantinuum's 200-qubit Ru-catalyst claim). A noiseless-sim
  stack duplicates free tools. Differentiation, if any, is the automation + rigor +
  reliability layer that none of them sell — keep Approach C alive as the real product hypothesis.

### Review verdict
- Mode: HOLD SCOPE. Findings accepted: per-rung correctness gates; rung 4.5 noise validation.
- Reversibility 5/5 (research roadmap). Architecture OK (driver seam absorbs SQD + noisy/hardware
  backends; embedding reuses fragmenter). Sole domain failure mode: the silent believable-wrong
  number — managed by the correctness gates.
- Open strategic risk (not resolved, flagged): the in-scope regime is classically solved, so this
  is capability-building, not near-term scientific value. Justified only as quantum-readiness.
- Deferred TODO (P3): parallelize FMO subsystems (embarrassingly parallel, currently serial).

## Rung 1 — Eng Review Outcome (2026-07-23, /plan-eng-review)

**Key finding:** rung 1 is ~90% already implemented. VM provisioning (`tools/remote/setup.sh`),
the benchmark runner with its built-in FCI cross-check (`benchmarks/runner.py`), and the CLI
all exist. The correctness-gate machinery is already there.

**Decisions:**
1. **CCSD deferred out of rung 1.** FCI is the exact gate and already implemented; CCSD is not
   used by the gate (the runner uses `solve_fci` for both references). Literature: PIs report
   FCI/CASCI/HCI/DMRG as references, not CCSD; CCSD(T) fails for the multireference targets QDR
   aims at (rung 5), so it is the wrong long-term baseline. When a classical baseline is needed
   (rung 4+), build a pluggable classical-reference seam prioritizing DMRG/SHCI, implemented via
   the PySCF mean-field object (never bare integrals). `solve_ccsd` stays an honest
   `NotImplementedError` stub.
2. **Add an automated FCI-reproduction test (P1, in scope).** A `requires_pyscf` test runs
   `run_case` for LiH/BeH2/H2O and asserts quantum-path error < 1e-6, zero cross-check
   disagreement warnings, and converged — turning the correctness gate from a human-read report
   into a CI failure. (`test_benchmarks.py` currently tests only plumbing with hardcoded
   energies; nothing runs the real computation.)

**Rung 1 final scope:** provision the VM; run the standard + h2-scan suites (CAS-truncated
cases first, full-space cases last so a memory blowup on H2O does not kill the sweep); add the
automated FCI-reproduction test; capture results. No new solver code.

**NOT in rung 1:** CCSD (deferred to the classical-reference seam at rung 4+); DMRG/SHCI (rung
4+); any embedding/SQD work (later rungs).

**Deferred TODOs:** (P2) pluggable classical-reference seam (FCI now, DMRG/SHCI later) via the
mean-field object; (P3, rung-5) `runner.py:169` catches `(MemoryError, ValueError,
NotImplementedError)` but `solve_fci` raises `ValidationError` for odd electrons — verify
`ValidationError` subclasses `ValueError`, or an open-shell active space will crash instead of
warn.

## Sources

- Goings et al., *Reliably assessing the electronic structure of cytochrome P450...*, PNAS 2022 — https://www.pnas.org/doi/10.1073/pnas.2203533119
- *Quantum Computing and Quantum Technologies in Drug Discovery...*, DDDT 2026 — https://www.tandfonline.com/doi/full/10.2147/DDDT.S590730
- *Quantum Mechanics in Drug Discovery: A Comprehensive Review*, MDPI/IJMS 2025 — https://pmc.ncbi.nlm.nih.gov/articles/PMC12249871/
- *Quantum Simulation of Ligand-like Molecules through SQD in a DMET Framework*, arXiv 2511.22158 — https://arxiv.org/html/2511.22158
- *Scalable Quantum-Classical DFT Embedding for NISQ Molecular Simulation*, arXiv 2602.01994 — https://arxiv.org/html/2602.01994
- Qubit Pharmaceuticals + Singapore CQT, 2026 — https://thequantuminsider.com/2026/04/30/qubit-pharmaceuticals-to-collaborate-with-the-centre-for-quantum-technologies-in-singapore-to-advance-quantum-algorithms-for-drug-discovery/
- QunaSys QSCI / QURI Parts (quri-parts-qsci) — https://github.com/QunaSys/quri-parts-qsci
- Quantinuum InQuanto DMET embedding — https://docs.quantinuum.com/inquanto/manual/dmet.html
- Robledo-Moreno et al., SQD on iron-sulfur clusters, Sci. Adv. 2025 — https://www.science.org/doi/pdf/10.1126/sciadv.adu9991
- HI-VQE vs CASCI/HCI/DMRG references, arXiv 2601.06935 — https://arxiv.org/pdf/2601.06935
- Active-space selection impact on VQE for drug discovery, arXiv 2512.18203 — https://arxiv.org/pdf/2512.18203
- SHCI near-exact energies, arXiv 1809.04600 — https://arxiv.org/pdf/1809.04600

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 1 | issues_open | HOLD SCOPE; 3 accepted findings; 1 open strategic risk (Approach C vs method-replication) |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | clean | Scope: rung 1. 2 findings, both resolved: CCSD deferred; automated FCI-reproduction test added |
| Outside Voice | Claude subagent | Independent 2nd opinion | 1 | absorbed | Ran in CEO review; rung 1 is a narrow subset, no fresh run (codex auth failing) |

- **CROSS-MODEL:** CEO-review outside voice challenged the whole quantum-vs-classical thesis; resolved as deliberate quantum-readiness. No new eng-review tension (rung 1 scope is uncontested).
- **VERDICT:** CEO + ENG CLEARED for rung 1 — ready to implement rung 1.

**UNRESOLVED DECISIONS:**
- Approach C (automation/reliability layer) vs method-replication as the primary product — flagged at CEO review, revisit once rungs 1-2 are running.

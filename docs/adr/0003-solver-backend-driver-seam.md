# ADR-0003: The SolverBackend driver seam

- **Status:** Accepted
- **Date:** 2026-07-22 (written retroactively; the decision predates the record)

## Context

A VQE engine has to talk to several kinds of solver: an exact statevector simulator, a
shot-based sampler, a real QPU behind a vendor SDK, and classical references such as FCI.
Each vendor ships its own circuit type, its own parameter-binding convention, and its own
result object.

The obvious approach — let each part of the pipeline build the circuit type its target
needs — fails in a specific and expensive way. Vendor circuit types leak upward into the
ansatz code, the ansatz grows per-backend branches, and the claim "the same physics ran on
both backends" stops being checkable, because the two paths no longer share a
representation. A discrepancy between simulator and hardware then cannot be attributed.

## Decision

**One interface, one circuit representation.**

1. `drivers/solver_backend.SolverBackend` is the only interface a backend implements:
   `capabilities` and `solve(SolveRequest) -> SolveResult`.
2. Ansätze emit `types.Circuit` — an ordered list of `types.Gate` over a fixed gate set.
   No vendor circuit type appears anywhere in the pipeline.
3. A vendor SDK import is legal in **exactly one place**: the driver that talks to that
   vendor.
4. Backends declare `BackendCapabilities` honestly (`max_qubits`, `is_exact`,
   `supports_shots`, `requires_ansatz`). `check_supported` refuses a request the backend
   cannot service rather than silently degrading it.
5. A registered-but-unimplemented backend raises `BackendUnavailable`. It never returns an
   approximate answer while presenting it as the requested one.

## Consequences

**Good.** The gate IR is what makes "same physics, different estimator" true rather than
aspirational: `tests/test_drivers_contract.py` runs one contract suite against every
registered backend. The statevector driver's adjoint gradient works on the IR directly, so
it needed no ansatz changes. Adding a backend touches one file.

**Costs.** The IR must be translated at each vendor boundary — real work per driver, and a
real place for endianness or basis-ordering bugs. Mitigated by pinning the qubit-ordering
convention in `tests/test_qubit_mapper.py` against hand-built states.

The gate set is deliberately small (`H, X, Y, Z, S, SDG, RX, RY, RZ, CX, CZ`). An ansatz
needing a gate outside it must decompose, or the set must be extended in one place.

**Currently implemented:** `statevector` (exact, with adjoint gradients) and `classical`
(FCI via PySCF). `sampled` and `qpu` are registered and raise.

## Alternatives rejected

*Use a vendor SDK's circuit type directly (Qiskit/Cirq).* Fastest to start, and it couples
the entire pipeline to one vendor's release cadence and conventions. The seam is worth
more than the saved translation code.

*No seam — one backend.* The engine's central correctness argument is that the exact
backend validates physics a sampled or hardware backend then reproduces. That argument
needs at least two backends and a shared representation to be meaningful.

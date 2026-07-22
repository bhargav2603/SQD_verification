# Adding a SolverBackend

The driver seam (ADR-0003) exists so that a new way of computing a ground-state energy —
a different simulator, a vendor QPU, a new classical method — plugs in **without touching
the pipeline, the mappers, the ansätze, or the validator**.

That property is not automatic. It holds only while implementations follow the two rules
below, and the contract tests enforce them.

## The two rules

**1. Consume the IR, never a vendor type.**
Your `solve()` takes a [`SolveRequest`](../types.py) carrying a `Circuit` — an ordered
list of `Gate`s. If you need a Qiskit/Cirq/Braket circuit, transpile it *inside your
driver module*. That module is the only place a vendor import is legal. A vendor type in
a shared signature re-couples the whole engine to that vendor.

**2. Fail loudly.**
If you cannot serve a request, raise `BackendUnavailable`. Never return an approximate
answer while presenting it as the requested one. These energies flow into binding-energy
differences where five significant figures cancel — a plausible wrong number is
undetectable downstream, and strictly worse than an error.

## Checklist

1. **Create the module** at `drivers/<name>/<name>_driver.py`.

2. **Subclass `SolverBackend`** and set a unique class-level `name`. It is persisted on
   run records, so treat it as a public identifier: stable, and never reused.

3. **Declare `capabilities`** honestly — `BackendCapabilities` in
   [`capabilities.py`](capabilities.py):
   - `max_qubits` — your real limit. It may be *below* the platform cap and often should
     be (the statevector driver caps at 24 because 2²⁴ complex128 is already 256 MiB).
     It may never be above; the constructor rejects that.
   - `is_exact`, `supports_shots` — how the energy is produced. Getting this wrong makes
     the error budget lie.
   - `requires_ansatz`, `requires_integrals` — which payload you need. The base class
     enforces these for you.
   - `is_implemented=False` + `unavailable_reason` while you are still building. A
     registered-but-honest driver is fine; a registered-but-lying one is not.

4. **Call `self.check_supported(request)` first** in `solve()`. It enforces the qubit cap,
   the payload requirements and the ansatz/Hamiltonian width match — uniformly, including
   for drivers written by people who never read this file.

5. **Register it** with `@register_backend`, and add the import to
   [`__init__.py`](__init__.py). The imports there look unused; they are not. Dropping one
   silently removes the backend from the registry.

6. **Return a complete `SolveResult`.** Set `converged` truthfully, and `std_error` if and
   only if your estimate is stochastic. Put the optimiser energy history in
   `metadata["history"]` so the stage-9 error budget can estimate the optimiser residual.

7. **Run the contract tests.** `tests/test_drivers_contract.py` is parametrized over the
   registry, so your backend is tested the moment it is registered — you do not add tests
   to get coverage, you add tests to cover behaviour specific to your backend.

```bash
uv run pytest packages/compute-engine/tests/test_drivers_contract.py -q
```

8. **Prove it against a known answer.** Every backend that claims to compute a ground
   state should be checked against the H₂/FCI reference the same way the statevector
   driver is — see `tests/test_stage_00_fci_gate.py`. Agreement with the other backends
   on a system with an exact answer is the only evidence that counts.

## Current backends

| Name | Status | Notes |
|---|---|---|
| `statevector` | implemented | Exact ⟨ψ\|H\|ψ⟩, ≤24 qubits. The correctness reference. |
| `classical` | implemented | FCI via PySCF, ≤16 qubits. Needs integrals, not a circuit. |
| `sampled` | registered | Needs commuting-set grouping + shot allocation before it can report a truthful error bar. |
| `qpu` | registered | Needs a vendor transpiler here and brokering via `services/qpu-broker`. |

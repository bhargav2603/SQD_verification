# ADR-0006: The 40-qubit cap, and the two smaller limits that actually bind

- **Status:** Accepted, with a correction (see *Revision*)
- **Date:** 2026-07-22 (written retroactively; revised the same day with measured data)

## Context

A dense statevector of `n` qubits is `2ⁿ` complex128 amplitudes. Growth is such that a
job one qubit too large does not run slowly — it exhausts memory and the worker is
OOM-killed, taking any co-scheduled work with it. That failure has to be prevented before
a job is scheduled, never discovered at runtime.

## Decision

`qdr_core.limits.MAX_QUBITS = 40` is a **scheduling policy**: the largest register the
platform will admit. `2⁴⁰ × 16 bytes` is 16 TiB, so the cap is what stops a request that
could never succeed from reaching a queue.

It is enforced at every layer that can refuse, because the layers protect different
things:

| Enforcement point | Protects |
|---|---|
| control plane (pre-submit) | the user, with a fast error |
| dispatcher (admission) | the queue |
| `active_space_selector` (engine) | the worker process |

Defence in depth is warranted when the failure mode is an OOM kill rather than an error
response.

## Revision: 40 was never an achievable size

The original record implied 40 qubits was a target. Measurement says otherwise, and the
correction matters more than the original decision.

Two costs scale as `2ⁿ`, and **the second dominates**:

| | memory |
|---|---|
| statevector | `2ⁿ × 16` bytes |
| sparse Hamiltonian | `n_terms × 2ⁿ × 16` bytes |

Measured: a 16-qubit case required **5.66 GiB for the sparse operator alone** and failed
there — not in the simulator. Wall-clock for a single energy evaluation, measured:

| Active space | Qubits | Gates | 1 energy eval |
|---|---|---|---|
| CAS(6,6) | 12 | 16,338 | 0.32 s |
| CAS(8,7) | 14 | 31,416 | 1.39 s |
| CAS(8,8) | 16 | 60,424 | sparse-H OOM |

So `qdr_core.limits` now carries three constants rather than one, and they are not
interchangeable:

```
MAX_QUBITS                    40   what the platform will schedule   (policy)
MAX_DENSE_STATEVECTOR_QUBITS  24   what one worker will allocate     (memory)
MAX_EXACT_REFERENCE_QUBITS    14   where the sparse Hamiltonian wins (measured)
```

They previously drifted: the statevector driver hard-coded 24, and two differently-named
constants for the exact-reference limit — `EXACT_REFERENCE_QUBIT_LIMIT = 16` in stage 9 and
`EXACT_QUBIT_REFERENCE_LIMIT = 14` in the benchmark runner — disagreed about the same
quantity. All are now imported from one place, and
`packages/qdr-core/tests/test_qubit_ladder.py` fails the build if a new one is pasted in.

## Consequences

Anything quoting a qubit ceiling must import one of the three, and must pick the right
one. Documentation that says "40 qubits" without qualification is wrong: **the practical
ceiling for a completed VQE run is 14.**

Raising `MAX_EXACT_REFERENCE_QUBITS` requires a matrix-free Hamiltonian application —
applying Pauli strings directly to the statevector, ~1 MB instead of ~6 GB, at the cost of
a slower `H|ψ⟩` — not a larger machine. That is the identified next bottleneck.

## Alternatives rejected

*Make the cap configurable.* It is an architectural invariant that three subsystems must
agree on. A tunable that can differ between control plane and worker reintroduces exactly
the drift this ADR exists to prevent.

*Enforce only at the engine.* By then the job has been queued, scheduled and dispatched;
the cheapest refusal is the earliest one.

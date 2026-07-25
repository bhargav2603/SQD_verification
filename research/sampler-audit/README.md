# QDR sampler audit

This directory contains the locked, credential-free H₂/STO-3G methodological pilot
for testing whether the determinant sample source changes SQD quality when the
Hamiltonian, shot count, retained-dimension ladder, recovery, diagonalization, seeds,
and decision rule are held fixed.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m qdr_compute_engine.audit `
  research/sampler-audit/protocol-v1.json `
  --output research/sampler-audit/results
```

The command creates `results/<scientific-hash>/` with raw draws, full-precision
observations, a tamper-evident manifest, a Markdown report, and PNG/PDF figures.
The result directory is ignored by Git until an evidence bundle has been independently
reviewed and intentionally released.

This pilot answers only whether the experiment machinery isolates a sampler effect.
H₂/STO-3G has four determinants in the target sector, and the exact-state sampler is
classically trivial. Therefore even a `quantum_benefit` decision here is a successful
positive-control result, not useful quantum advantage.

Promotion beyond the pilot requires:

1. deterministic replay of scientific hashes;
2. independent agreement with Qiskit SQD on Linux;
3. complete sampling, recovery, diagonalization, wall, CPU, and QPU cost records;
4. larger development and untouched held-out molecular systems, beginning with N₂ and
   an iron-sulfur active-space case;
5. genuine noisy-simulator and hardware samples acquired under the same locked protocol.

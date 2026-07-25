# Preregistration: determinant-sampler causal audit, protocol v1

## Question

Holding the molecular Hamiltonian and every classical processing choice fixed, does the
designated quantum-like determinant distribution produce lower SQD energy error than a
locked classical proposal at matched raw shot count?

## Frozen design

- Development H₂ bond lengths: 0.735 Å and 1.50 Å.
- Held-out H₂ bond lengths: 1.00 Å and 2.50 Å.
- Arms: exact-state oracle; Hartree–Fock excitation proposal; uniform-sector negative
  control.
- Shots per cell: 128, 512, and 2048.
- Retained dimensions: 1, 2, and 4.
- Seeds: integers 0 through 29.
- Recovery: observed in-sector determinants ranked by frequency, ties by determinant
  index, followed by unobserved sector determinants in ascending index order. This exact
  rule is applied to all arms.
- Classical selection: choose the lowest mean primary AUC using development systems
  only. Do not inspect held-out scores during selection.

## Primary estimand and decision

For every `(held-out problem, shots, seed)` cell, integrate absolute energy error against
the logarithm of retained dimension using the normalized trapezoid rule. The paired
effect is exact-state-oracle AUC minus locked-classical AUC.

Bootstrap complete paired cells 10,000 times. Report:

- `quantum_benefit` only when the two-sided 95% interval is entirely below zero;
- `classical_benefit` only when it is entirely above zero;
- `no_evidence` when it includes zero;
- `inconclusive` for missing cells or fewer than 30 complete pairs.

The metric and decision rule must not be changed after results are visible.

## Evidence boundary

The exact-state arm is an oracle positive control produced by exact classical
diagonalization. It is not a circuit simulation, noisy quantum run, or QPU run. H₂ is
small enough to solve exactly, so this protocol can validate causal isolation,
statistics, provenance, and reporting—but it cannot support a useful quantum-advantage
claim.

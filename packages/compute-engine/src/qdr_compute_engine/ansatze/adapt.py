"""ADAPT-VQE ansatz.

Grows the ansatz operator by operator, each iteration adding whichever pool operator has
the largest energy gradient. Produces far shallower circuits than UCCSD for the same
accuracy, at the cost of a gradient measurement over the whole pool per iteration.

Registered but not implemented. Unlike UCCSD, ADAPT's circuit is not a pure function of
the system -- it depends on gradients measured during construction, so the *builder*
needs backend access. That inverts the current stage 7 -> stage 8 dependency and is a
structural change to the pipeline, not just a new module.
"""

from qdr_compute_engine.types import Circuit

ANSATZ_NAME = "adapt"


def build_adapt_ansatz(
    n_electrons: int,
    n_spin_orbitals: int,
    *,
    max_operators: int = 20,
    gradient_threshold: float = 1e-3,
) -> Circuit:
    raise NotImplementedError(
        "ADAPT-VQE is not implemented yet; use 'uccsd'. It requires backend access "
        "during ansatz construction, which stage 7 does not currently have."
    )

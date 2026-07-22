"""Stage 9 support: error-budget accounting.

A VQE energy without an error budget is not a scientific result -- it is a number. This
module names the error sources the engine can actually quantify and refuses to invent the
ones it cannot.

Sources currently accounted for:

``ansatz``
    How far the variational answer sits above the true ground state of the same qubit
    Hamiltonian. Computable exactly for small systems by diagonalising that Hamiltonian,
    and it is a strict lower bound on the ansatz error by the variational principle.

``optimizer``
    Residual from incomplete convergence, estimated from the tail of the energy history.

``shot_noise``
    Standard error reported by a sampling backend; zero for exact backends.

``active_space``
    Correlation energy discarded by truncating the orbital space. This is usually the
    *largest* term, and the engine cannot compute it without a larger reference
    calculation -- so it must be supplied explicitly. It is recorded as ``None`` rather
    than ``0.0`` when unknown, because a missing term silently treated as zero would make
    the total look far better than it is.
"""

from qdr_compute_engine.types import ErrorBudget, SolveResult
from qdr_core.limits import CHEMICAL_ACCURACY_HA

#: Budget entries the engine knows how to reason about.
KNOWN_SOURCES = ("ansatz", "optimizer", "shot_noise", "active_space")


def optimizer_residual(history: list[float], window: int = 5) -> float:
    """Estimate residual optimisation error from the spread of the final energies.

    A flat tail means the optimiser has stopped moving; a wide one means it stopped early.
    This is a heuristic, and is labelled as such in the report.
    """
    if len(history) < 2:
        return 0.0
    tail = history[-window:]
    return float(max(tail) - min(tail))


def build_error_budget(
    result: SolveResult,
    *,
    exact_ground_state: float | None = None,
    active_space_error: float | None = None,
) -> tuple[ErrorBudget, tuple[str, ...]]:
    """Assemble the budget for one solve, plus any warnings it raises."""
    budget = ErrorBudget()
    warnings: list[str] = []

    if exact_ground_state is not None:
        ansatz_error = result.energy - exact_ground_state
        if ansatz_error < -1e-9:
            # Below the true ground state of the same Hamiltonian is variationally
            # impossible; something upstream disagrees about which operator is in play.
            warnings.append(
                f"energy {result.energy:.9f} lies {abs(ansatz_error):.2e} Ha below the "
                "exact ground state of the same Hamiltonian, which violates the "
                "variational principle"
            )
        budget = budget.with_contribution("ansatz", abs(ansatz_error))
    else:
        warnings.append("ansatz error not quantified: no exact reference was computed")

    history = result.metadata.get("history")
    if isinstance(history, list) and history:
        budget = budget.with_contribution("optimizer", optimizer_residual(history))

    if result.std_error is not None:
        budget = budget.with_contribution("shot_noise", result.std_error)

    if active_space_error is not None:
        budget = budget.with_contribution("active_space", active_space_error)
    else:
        warnings.append(
            "active-space truncation error not quantified; the reported total is a "
            "lower bound on the true error, not an estimate of it"
        )

    if not result.converged:
        warnings.append("optimizer did not report convergence")

    return budget, tuple(warnings)


def within_chemical_accuracy(budget: ErrorBudget) -> bool:
    """True when the quantified error total is below 1 kcal/mol."""
    return budget.total < CHEMICAL_ACCURACY_HA

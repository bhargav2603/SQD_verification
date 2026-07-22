"""Validation and error-budget reporting types.

What stage 9 produces.

Re-exported by ``qdr_compute_engine.types``, which remains the import site for the
whole engine -- these modules exist to keep each file small, not to add a new path.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ErrorBudget:
    """Named error contributions in Hartree, summed in quadrature.

    Quadrature rather than a linear sum: these are independent error sources, so adding
    them linearly would systematically overstate the total.
    """

    contributions: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return float(np.sqrt(sum(v**2 for v in self.contributions.values())))

    def with_contribution(self, name: str, value: float) -> "ErrorBudget":
        if value < 0:
            raise ValueError(f"error contribution {name!r} must be non-negative, got {value}")
        return ErrorBudget({**self.contributions, name: value})


@dataclass(frozen=True, slots=True)
class ValidationReport:
    binding_energy_ha: float
    binding_energy_kcal_per_mol: float
    error_budget: ErrorBudget
    within_chemical_accuracy: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "binding_energy_ha": self.binding_energy_ha,
            "binding_energy_kcal_per_mol": self.binding_energy_kcal_per_mol,
            "error_budget": dict(self.error_budget.contributions),
            "total_error_ha": self.error_budget.total,
            "within_chemical_accuracy": self.within_chemical_accuracy,
            "warnings": list(self.warnings),
        }

"""Solver request and result types.

The SolverBackend interface's inputs and outputs (ADR-0003).

Re-exported by ``qdr_compute_engine.types``, which remains the import site for the
whole engine -- these modules exist to keep each file small, not to add a new path.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass
from qdr_compute_engine.typedefs.circuit import Circuit
from qdr_compute_engine.typedefs.electronic import (
    ActiveSpace,
    IntegralSet,
    QubitHamiltonian,
)
from qdr_compute_engine.typedefs.geometry import Geometry  # noqa: F401
from qdr_core.errors import ValidationError

# --------------------------------------------------------------------------------------
# Solver request / result
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False, slots=True)
class SolveRequest:
    """Everything a backend needs to produce a ground-state energy.

    Both a qubit-level payload (``hamiltonian`` + ``ansatz``) and the orbital-basis
    ``integrals`` are carried, because the two families of backend need different things:
    a variational simulator consumes the circuit, while a classical backend runs CCSD/FCI
    directly on the integrals. A backend declares which it needs via its capabilities and
    raises ``BackendUnavailable`` if the request cannot serve it.
    """

    hamiltonian: QubitHamiltonian
    active_space: ActiveSpace
    ansatz: Circuit | None = None
    initial_params: NDArray[np.float64] | None = None
    integrals: IntegralSet | None = None
    shots: int | None = None
    seed: int = 0
    max_iterations: int = 1000
    #: How many configurations a subspace-diagonalisation backend (SQD) may keep. ``None``
    #: means "no truncation": take the whole sector, which turns SQD into exact CASCI and is
    #: what the equals-FCI test pins. It is a *requested* ceiling, not a promise -- the sector
    #: is often smaller, and a backend reports the dimension it actually used in its metadata.
    #: Carried on the request rather than in ``options`` because it is the one knob that
    #: decides whether a returned energy is exact or merely a variational upper bound, and a
    #: typo in an untyped options key would silently answer the wrong question.
    subspace_dimension: int | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.subspace_dimension is not None and self.subspace_dimension < 1:
            raise ValidationError(
                f"subspace_dimension must be >= 1 when given, got {self.subspace_dimension}",
                subspace_dimension=self.subspace_dimension,
            )

    @property
    def n_qubits(self) -> int:
        return self.hamiltonian.n_qubits


@dataclass(frozen=True, eq=False, slots=True)
class EnergyEstimate:
    """A single energy evaluation. ``std_error`` is None for exact (shot-free) backends."""

    value: float
    std_error: float | None = None
    shots: int | None = None


@dataclass(frozen=True, eq=False, slots=True)
class SolveResult:
    energy: float
    backend: str
    method: str
    converged: bool
    n_iterations: int = 0
    n_energy_evaluations: int = 0
    optimal_params: NDArray[np.float64] | None = None
    std_error: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

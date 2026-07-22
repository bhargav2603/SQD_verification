"""Electronic-structure types.

Integrals, active spaces, and the fermionic/qubit Hamiltonians.

Re-exported by ``qdr_compute_engine.types``, which remains the import site for the
whole engine -- these modules exist to keep each file small, not to add a new path.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from qdr_core.limits import qubits_for_orbitals

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openfermion import FermionOperator, InteractionOperator, QubitOperator
from qdr_compute_engine.typedefs.geometry import Geometry  # noqa: F401

# --------------------------------------------------------------------------------------
# Electronic structure
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False, slots=True)
class IntegralSet:
    """Molecular-orbital-basis integrals.

    ``h2`` is stored in **physicist notation** ``<pq|rs>``, matching what OpenFermion's
    ``InteractionOperator`` expects. PySCF emits chemist notation ``(pq|rs)``; the
    conversion happens once, in ``integrals/pyscf_integrals.py``, and is asserted there.
    Getting this backwards yields a Hamiltonian that is subtly wrong but still Hermitian,
    so it will not announce itself -- only the FCI comparison catches it.
    """

    h1: NDArray[np.float64]
    h2: NDArray[np.float64]
    e_nuc: float
    n_electrons: int
    e_hf: float | None = None
    orbital_energies: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if self.h1.ndim != 2 or self.h1.shape[0] != self.h1.shape[1]:
            raise ValueError(f"h1 must be square 2-D, got shape {self.h1.shape}")
        n = self.h1.shape[0]
        if self.h2.shape != (n, n, n, n):
            raise ValueError(f"h2 must have shape {(n, n, n, n)}, got {self.h2.shape}")
        if self.n_electrons < 0:
            raise ValueError(f"n_electrons must be non-negative, got {self.n_electrons}")

    @property
    def n_orbitals(self) -> int:
        return int(self.h1.shape[0])

    def to_dict(self) -> dict[str, Any]:
        """JSON-native form (used by the golden reference and input hashing)."""
        payload: dict[str, Any] = {
            "h1": self.h1.tolist(),
            "h2": self.h2.tolist(),
            "e_nuc": float(self.e_nuc),
            "n_electrons": int(self.n_electrons),
        }
        if self.e_hf is not None:
            payload["e_hf"] = float(self.e_hf)
        return payload


@dataclass(frozen=True, slots=True)
class ActiveSpace:
    """A selected set of spatial orbitals and the electrons correlated within them."""

    n_active_orbitals: int
    n_active_electrons: int
    orbital_indices: tuple[int, ...]
    n_frozen_core_orbitals: int = 0

    def __post_init__(self) -> None:
        if len(self.orbital_indices) != self.n_active_orbitals:
            raise ValueError(
                f"orbital_indices has {len(self.orbital_indices)} entries but "
                f"n_active_orbitals is {self.n_active_orbitals}"
            )
        if self.n_active_electrons > 2 * self.n_active_orbitals:
            raise ValueError(
                f"{self.n_active_electrons} electrons cannot occupy "
                f"{self.n_active_orbitals} spatial orbitals"
            )

    @property
    def n_qubits(self) -> int:
        """Qubit count under a one-qubit-per-spin-orbital mapping."""
        return qubits_for_orbitals(self.n_active_orbitals)

    @property
    def n_spin_orbitals(self) -> int:
        return 2 * self.n_active_orbitals


@dataclass(frozen=True, eq=False, slots=True)
class FermionicHamiltonian:
    operator: "InteractionOperator | FermionOperator"
    n_spin_orbitals: int
    constant: float = 0.0


@dataclass(frozen=True, eq=False, slots=True)
class QubitHamiltonian:
    operator: "QubitOperator"
    n_qubits: int
    mapping: str

    @property
    def n_terms(self) -> int:
        return len(self.operator.terms)

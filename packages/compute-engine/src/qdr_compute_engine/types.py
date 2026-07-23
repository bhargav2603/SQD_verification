"""Engine-wide types.

The important architectural content here is the **circuit IR** (:class:`Gate`,
:class:`Circuit`). ADR-0003 makes ``drivers/solver_backend.py`` the one interface a new
backend implements, and that only holds if no vendor circuit type ever appears in the
pipeline. So ansaetze emit this IR, drivers consume it, and a vendor SDK import is legal
in exactly one place: the driver that talks to that vendor.

Qubit ordering convention
-------------------------
Qubit ``q`` is spin orbital ``q``. Spin orbitals interleave spatial orbitals as
``2*p`` (alpha) and ``2*p + 1`` (beta), which is OpenFermion's convention -- we adopt it
wholesale rather than translating, because a mismatch between our statevector indexing
and OpenFermion's ``get_sparse_operator`` produces a plausible-looking but wrong energy.
:func:`statevector_index_order` documents the resulting amplitude layout, and
``tests/test_qubit_mapper.py`` pins it.

The definitions themselves live in ``qdr_compute_engine.typedefs.*``, split by concern so
each file stays small enough to read in one sitting. **This module remains the import site
for the entire engine** -- nothing imports from ``typedefs`` directly, so the split is an
organisational change with no call-site churn and no second way to spell an import.
"""

from qdr_compute_engine.typedefs.circuit import (
    GATE_ARITY,
    ROTATION_GATES,
    Circuit,
    Gate,
    GateName,
    statevector_index_order,
)
from qdr_compute_engine.typedefs.electronic import (
    ActiveSpace,
    FermionicHamiltonian,
    IntegralSet,
    QubitHamiltonian,
)
from qdr_compute_engine.typedefs.embedding import EmbeddingResult
from qdr_compute_engine.typedefs.fragments import Fragment, FragmentationResult
from qdr_compute_engine.typedefs.geometry import (
    BOHR_IN_ANGSTROM,
    Atom,
    Geometry,
    LengthUnit,
)
from qdr_compute_engine.typedefs.reporting import ErrorBudget, ValidationReport
from qdr_compute_engine.typedefs.solve import EnergyEstimate, SolveRequest, SolveResult

__all__ = [
    "BOHR_IN_ANGSTROM",
    "GATE_ARITY",
    "ROTATION_GATES",
    "ActiveSpace",
    "Atom",
    "Circuit",
    "EmbeddingResult",
    "EnergyEstimate",
    "ErrorBudget",
    "FermionicHamiltonian",
    "Fragment",
    "FragmentationResult",
    "Gate",
    "GateName",
    "Geometry",
    "IntegralSet",
    "LengthUnit",
    "QubitHamiltonian",
    "SolveRequest",
    "SolveResult",
    "ValidationReport",
    "statevector_index_order",
]

"""Stage 6 alternative: parity mapping.

Stores cumulative parity rather than occupation, so the Z string that JW needs for
antisymmetry becomes local -- at the cost of a non-local X string when updating
occupation. Its practical draw is the two-qubit reduction: with fixed particle number and
Sz, the qubits encoding the two conserved parities are separable and can be tapered off,
turning an n-qubit problem into n-2.

Registered but not implemented. That tapering is exactly where a subtle bug produces a
believable-but-wrong energy, so it lands with its own correctness test against the same
FCI reference rather than riding in on JW's.
"""

from qdr_compute_engine.types import FermionicHamiltonian, QubitHamiltonian

MAPPING_NAME = "parity"


def map_to_qubits(hamiltonian: FermionicHamiltonian) -> QubitHamiltonian:
    raise NotImplementedError(
        "the parity mapping is not implemented yet; use 'jordan_wigner'. "
        "Implementing it requires the two-qubit reduction and a dedicated FCI check."
    )

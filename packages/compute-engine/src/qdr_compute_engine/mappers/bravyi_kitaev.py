"""Stage 6 alternative: Bravyi-Kitaev mapping.

Balances occupation and parity information across a binary tree, giving O(log n) Pauli
weight per fermionic operator instead of JW's O(n). Fewer two-qubit gates per term makes
it the better choice on hardware, where circuit depth dominates the error budget.

Registered but not implemented. OpenFermion provides the transform, so the work is small,
but it ships with its own H2/FCI check -- a mapping that agrees with JW on term count
while disagreeing on the spectrum is a real failure mode.
"""

from qdr_compute_engine.types import FermionicHamiltonian, QubitHamiltonian

MAPPING_NAME = "bravyi_kitaev"


def map_to_qubits(hamiltonian: FermionicHamiltonian) -> QubitHamiltonian:
    raise NotImplementedError(
        "the Bravyi-Kitaev mapping is not implemented yet; use 'jordan_wigner'."
    )

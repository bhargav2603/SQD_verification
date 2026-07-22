"""Stage 6: Jordan-Wigner mapping.

JW assigns one qubit per spin orbital and encodes occupation directly in the computational
basis (|1> occupied, |0> empty), restoring fermionic antisymmetry with a string of Z
operators on all lower-indexed qubits:

    a+_j = (X_j - iY_j)/2  (x)  Z_0 ... Z_{j-1}

That Z string is the trade-off: locality is poor (a single excitation touches O(n)
qubits) but the occupation encoding is transparent, which makes the Hartree-Fock
reference a simple product state and keeps particle-number arguments easy to check.
Parity and Bravyi-Kitaev trade that transparency for shorter strings.

The transform itself delegates to OpenFermion. Re-implementing a well-tested
transformation would add risk without adding capability; this module's job is to own the
qubit-count bookkeeping and the platform's error taxonomy around it.
"""

from openfermion import QubitOperator
from openfermion import jordan_wigner as _openfermion_jordan_wigner

from qdr_compute_engine.types import FermionicHamiltonian, QubitHamiltonian
from qdr_core.errors import QubitCapExceeded
from qdr_core.limits import MAX_QUBITS

MAPPING_NAME = "jordan_wigner"


def map_to_qubits(hamiltonian: FermionicHamiltonian) -> QubitHamiltonian:
    """Map a fermionic Hamiltonian to qubits under Jordan-Wigner."""
    n_qubits = hamiltonian.n_spin_orbitals
    if n_qubits > MAX_QUBITS:
        raise QubitCapExceeded(required_qubits=n_qubits, max_qubits=MAX_QUBITS)

    qubit_operator = _openfermion_jordan_wigner(hamiltonian.operator)
    qubit_operator.compress()

    return QubitHamiltonian(
        operator=qubit_operator,
        n_qubits=n_qubits,
        mapping=MAPPING_NAME,
    )


def creation_operator(index: int) -> QubitOperator:
    """JW image of the creation operator a+_index, for tests and pedagogy."""
    z_string = " ".join(f"Z{i}" for i in range(index))
    prefix = f"{z_string} " if z_string else ""
    return QubitOperator(f"{prefix}X{index}", 0.5) + QubitOperator(f"{prefix}Y{index}", -0.5j)


def annihilation_operator(index: int) -> QubitOperator:
    """JW image of the annihilation operator a_index."""
    z_string = " ".join(f"Z{i}" for i in range(index))
    prefix = f"{z_string} " if z_string else ""
    return QubitOperator(f"{prefix}X{index}", 0.5) + QubitOperator(f"{prefix}Y{index}", 0.5j)

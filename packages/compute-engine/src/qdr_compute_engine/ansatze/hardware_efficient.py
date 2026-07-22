"""Hardware-efficient ansatz.

Layers of single-qubit rotations and native entanglers, chosen for shallow depth on real
devices rather than for chemical structure.

Registered but not implemented. The circuit itself is easy; what makes it more than a
weekend's work is that HEA does not conserve particle number, so the optimiser can and
does wander into states with the wrong electron count and return an energy that is not
the molecule's ground state at all. Shipping it responsibly means adding a particle-number
penalty or post-selection plus a test that catches symmetry breaking -- which is why it
does not ride in on UCCSD's correctness argument.
"""

from qdr_compute_engine.types import Circuit

ANSATZ_NAME = "hardware_efficient"


def build_hardware_efficient_ansatz(
    n_electrons: int,
    n_spin_orbitals: int,
    *,
    n_layers: int = 2,
) -> Circuit:
    raise NotImplementedError(
        "the hardware-efficient ansatz is not implemented yet; use 'uccsd'. "
        "It requires particle-number penalty or post-selection to be trustworthy."
    )

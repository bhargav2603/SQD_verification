"""Regenerate ``tests/golden/h2_fci_reference.json``.

Run with:  uv run python packages/compute-engine/tests/reference/generate_golden.py

The FCI energy is obtained by exact diagonalisation of the qubit Hamiltonian built from
these integrals. For H2/STO-3G that is exact, not approximate: the 4-qubit Hilbert space
*is* the full determinant space, so its lowest eigenvalue is the FCI energy by definition.
CI re-derives the same quantity through PySCF and asserts the two agree.
"""

import json
import sys
from pathlib import Path

# Make `reference.*` importable when run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdr_compute_engine.drivers.statevector.statevector_driver import (
    exact_ground_state_energy,
)
from qdr_compute_engine.hamiltonian.fermionic_builder import (
    build_fermionic_hamiltonian,
)
from qdr_compute_engine.mappers.jordan_wigner import map_to_qubits
from qdr_compute_engine.types import IntegralSet
from reference.sto3g_h2 import chemist_to_physicist, h2_reference

BOND_LENGTH_ANGSTROM = 0.735
GOLDEN_PATH = Path(__file__).resolve().parents[1] / "golden" / "h2_fci_reference.json"


def build_payload(bond_length: float = BOND_LENGTH_ANGSTROM) -> dict[str, object]:
    reference = h2_reference(bond_length)
    h2_physicist = chemist_to_physicist(reference["eri_mo_chemist"])

    integrals = IntegralSet(
        h1=reference["h1_mo"],
        h2=h2_physicist,
        e_nuc=reference["e_nuc"],
        n_electrons=2,
        e_hf=reference["e_hf"],
    )
    qubit_hamiltonian = map_to_qubits(build_fermionic_hamiltonian(integrals))
    e_fci = exact_ground_state_energy(qubit_hamiltonian)

    return {
        "_comment": (
            "H2/STO-3G reference. Integrals computed analytically by "
            "tests/reference/sto3g_h2.py; e_fci is the lowest eigenvalue of the 4-qubit "
            "Hamiltonian, which is exact for this system. CI recomputes all of it with "
            "PySCF and asserts agreement."
        ),
        "molecule": "H2",
        "basis": "sto-3g",
        "bond_length_angstrom": bond_length,
        "charge": 0,
        "spin_multiplicity": 1,
        "n_electrons": 2,
        "n_orbitals": 2,
        "n_qubits": 4,
        "h2_notation": "physicist <pq|rs>",
        "e_nuc": reference["e_nuc"],
        "e_hf": reference["e_hf"],
        "e_fci": float(e_fci),
        "correlation_energy": float(e_fci) - reference["e_hf"],
        "h1": reference["h1_mo"].tolist(),
        "h2": h2_physicist.tolist(),
    }


def main() -> None:
    payload = build_payload()
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {GOLDEN_PATH}")
    print(f"  E_nuc = {payload['e_nuc']:.12f}")
    print(f"  E_HF  = {payload['e_hf']:.12f}")
    print(f"  E_FCI = {payload['e_fci']:.12f}")
    print(f"  E_corr= {payload['correlation_energy']:.12f}")


if __name__ == "__main__":
    main()

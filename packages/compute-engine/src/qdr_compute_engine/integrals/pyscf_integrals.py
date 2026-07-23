"""Stage 3: molecular integrals via PySCF.

**The only module in the pipeline that may import PySCF.** PySCF publishes no Windows
wheels, so the import is deferred into the functions rather than done at module scope --
that keeps this module importable everywhere, lets the rest of the engine be developed
and tested on any platform, and turns a missing dependency into one clear error instead
of an import failure that takes the whole package down.

Notation
--------
PySCF returns two-electron integrals in **chemist** notation ``(pq|rs)``; the engine and
OpenFermion both work in **physicist** notation ``<pq|rs>``. The single conversion lives
in :func:`chemist_to_physicist`, and :func:`physicist_to_chemist` inverts it for handing
integrals back to PySCF's FCI solver. Getting this wrong yields a Hermitian Hamiltonian
with a plausible spectrum that is not the molecule's, so both directions are covered by
a round-trip test.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from qdr_compute_engine.geometry.pose_builder import to_pyscf_atom_spec
from qdr_compute_engine.types import Geometry, IntegralSet
from qdr_core.errors import ValidationError

DEFAULT_BASIS = "sto-3g"


def pyscf_available() -> bool:
    """True when PySCF can be imported in this environment."""
    try:
        import pyscf  # noqa: F401
    except ImportError:
        return False
    return True


def _require_pyscf() -> Any:
    try:
        import pyscf
    except ImportError as exc:
        raise ValidationError(
            "PySCF is required for integral evaluation but is not installed. "
            "It publishes no Windows wheels; install the 'chem' extra on Linux/macOS "
            "(`uv sync --extra chem`) or run this stage in the container image.",
            missing_dependency="pyscf",
        ) from exc
    return pyscf


def chemist_to_physicist(eri: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert ``(pq|rs)`` to ``<pq|rs>``.

    ``<pq|rs> = (ps|qr)``, i.e. ``transpose(0, 2, 3, 1)`` -- the same convention
    OpenFermion's own PySCF adapter uses.
    """
    return np.asarray(eri.transpose(0, 2, 3, 1), order="C")


def physicist_to_chemist(eri: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse of :func:`chemist_to_physicist`."""
    return np.asarray(eri.transpose(0, 3, 1, 2), order="C")


def compute_integrals(
    geometry: Geometry,
    *,
    basis: str = DEFAULT_BASIS,
    verbose: int = 0,
) -> IntegralSet:
    """Run the reference SCF and return MO-basis integrals for ``geometry``.

    Closed-shell geometries use RHF, exactly as before. Open-shell geometries
    (``spin_multiplicity > 1``) use **ROHF**, which -- unlike UHF -- keeps a *single* set of
    MO coefficients, so the resulting integrals are spin-free and feed the same
    ``(n_alpha, n_beta)`` FCI/CASCI machinery a spin gap needs. UHF is deliberately not used
    here: its two coefficient sets would make the MO-basis two-electron integrals ambiguous.
    """
    pyscf = _require_pyscf()
    from pyscf import ao2mo, scf

    molecule = pyscf.gto.Mole()
    molecule.atom = to_pyscf_atom_spec(geometry)
    molecule.basis = basis
    molecule.charge = geometry.charge
    molecule.spin = geometry.spin_multiplicity - 1  # PySCF counts unpaired electrons
    molecule.unit = "Angstrom"
    molecule.verbose = verbose
    molecule.build()

    is_closed_shell = molecule.spin == 0
    mean_field = scf.RHF(molecule) if is_closed_shell else scf.ROHF(molecule)
    e_hf = float(mean_field.kernel())
    if not mean_field.converged:
        raise ValidationError(
            "SCF did not converge; the geometry or charge/spin state is likely unphysical",
            e_hf=e_hf,
            reference="RHF" if is_closed_shell else "ROHF",
        )

    mo_coefficients = mean_field.mo_coeff
    n_orbitals = int(mo_coefficients.shape[1])

    # One-electron integrals: AO core Hamiltonian transformed into the MO basis.
    h_core_ao = mean_field.get_hcore()
    h1_mo = mo_coefficients.T @ h_core_ao @ mo_coefficients

    # Two-electron integrals, restored from PySCF's packed symmetric storage.
    eri_mo_chemist = ao2mo.restore(1, ao2mo.kernel(molecule, mo_coefficients), n_orbitals)

    return IntegralSet(
        h1=np.ascontiguousarray(h1_mo),
        h2=chemist_to_physicist(np.ascontiguousarray(eri_mo_chemist)),
        e_nuc=float(molecule.energy_nuc()),
        n_electrons=int(molecule.nelectron),
        e_hf=e_hf,
        orbital_energies=np.ascontiguousarray(mean_field.mo_energy),
    )

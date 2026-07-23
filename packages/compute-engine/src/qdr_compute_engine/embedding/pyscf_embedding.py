"""PySCF adapter for projection-based WF-in-DFT embedding.

This is the *glue* layer: it runs the mean-field calculation and produces the matrices the
backend-free core (:mod:`qdr_compute_engine.embedding.projection`) combines. Like
``integrals/pyscf_integrals``, it is the only place in the embedding path that imports
PySCF, and the import is deferred into the function so the module stays importable on
Windows (PySCF ships no Windows wheel).

Scope of this slice (verified on Colab/Linux, marked ``requires_pyscf``)
-----------------------------------------------------------------------
Everything up to and including the embedded one-electron operator in the AO basis:

  1. full-system reference SCF (RKS environment, ``environment_functional``),
  2. Pipek-Mezey localization of the occupied orbitals,
  3. Mulliken active-atom population per localized orbital,
  4. the active/environment partition (via the tested pure core),
  5. densities gamma_tot / gamma_A / gamma_env and potentials g[gamma] = get_veff,
  6. ``h_emb = h_core + (g[gamma_tot] - g[gamma_A]) + mu * P_env`` in the AO basis.

**Deliberately NOT in this slice:** the transform of ``h_emb`` + two-electron integrals
into the embedded active-orbital basis and the environment-energy bookkeeping that keeps
total energies correct. That bookkeeping is the frozen-core-fold class of subtlety (the
engine has already been bitten there once) and must reuse the bug-fixed convention and pass
the separated-fragment recovery gate on real hardware before any number is trusted. Until
then :func:`build_embedding_artifacts` returns the AO-level pieces + partition, and the
pipeline stage refuses to fabricate embedded integrals.

The operator-level recovery property IS checked here: with the active region set to the
whole molecule the environment is empty, so ``P_env`` is zero and ``g[gamma_tot]`` equals
``g[gamma_A]`` -- ``h_emb`` must equal the bare ``h_core`` to machine precision.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qdr_compute_engine.embedding.projection import (
    PartitionResult,
    embed_one_body_operator,
    levelshift_projector,
    partition_orbitals_by_population,
)
from qdr_compute_engine.geometry.pose_builder import to_pyscf_atom_spec
from qdr_compute_engine.integrals.pyscf_integrals import _require_pyscf
from qdr_compute_engine.types import EmbeddingResult, Geometry
from qdr_core.errors import ValidationError

DEFAULT_LEVEL_SHIFT = 1.0e6
DEFAULT_ENVIRONMENT_FUNCTIONAL = "b3lyp"

#: An orbital counts as "on the active atoms" when this much of its Mulliken population
#: sits there. 0.5 is the standard midpoint cut; boundary orbitals are logged, not hidden.
DEFAULT_POPULATION_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class EmbeddingArtifacts:
    """The AO-basis pieces of a projection embedding, before the active-space transform.

    ``h_emb`` is the embedded one-electron operator; ``partition`` records which localized
    occupied orbitals are active vs environment; ``result`` is the provenance object that
    flows onto the pipeline context. These are exactly the inputs the (not-yet-implemented)
    active-space transform will consume.
    """

    h_emb: NDArray[np.float64]
    overlap: NDArray[np.float64]
    core_hamiltonian: NDArray[np.float64]
    environment_potential: NDArray[np.float64]
    partition: PartitionResult
    result: EmbeddingResult


def build_embedding_artifacts(
    geometry: Geometry,
    *,
    active_atom_indices: tuple[int, ...],
    basis: str = "sto-3g",
    level_shift: float = DEFAULT_LEVEL_SHIFT,
    environment_functional: str = DEFAULT_ENVIRONMENT_FUNCTIONAL,
    population_threshold: float = DEFAULT_POPULATION_THRESHOLD,
    verbose: int = 0,
) -> EmbeddingArtifacts:
    """Run the reference SCF and assemble the AO-basis embedded operator.

    ``active_atom_indices`` names the atoms whose chemistry is treated at the WF level; the
    rest form the DFT environment. Passing *every* atom yields an empty environment, which
    is the recovery limit the gate checks.
    """
    if not active_atom_indices:
        raise ValidationError(
            "active_atom_indices is empty; an embedding with no active atoms has no "
            "active region to solve",
        )
    n_atoms = geometry.n_atoms
    out_of_range = [i for i in active_atom_indices if i < 0 or i >= n_atoms]
    if out_of_range:
        raise ValidationError(
            f"active_atom_indices {out_of_range} are outside the molecule's {n_atoms} atoms",
            n_atoms=n_atoms,
        )

    pyscf = _require_pyscf()
    from pyscf import dft, lo

    molecule = pyscf.gto.Mole()
    molecule.atom = to_pyscf_atom_spec(geometry)
    molecule.basis = basis
    molecule.charge = geometry.charge
    molecule.spin = geometry.spin_multiplicity - 1
    molecule.unit = "Angstrom"
    molecule.verbose = verbose
    molecule.build()

    mean_field = dft.RKS(molecule)
    mean_field.xc = environment_functional
    mean_field.kernel()
    if not mean_field.converged:
        raise ValidationError(
            "reference RKS did not converge; the geometry, charge, or spin state is "
            "likely unphysical",
            functional=environment_functional,
        )

    overlap = np.ascontiguousarray(molecule.intor("int1e_ovlp"))
    core_hamiltonian = np.ascontiguousarray(mean_field.get_hcore())

    # Occupied localized orbitals (Pipek-Mezey), then Mulliken active-atom population each.
    mo_coeff = mean_field.mo_coeff
    occupied_mask = mean_field.mo_occ > 0
    occupied = mo_coeff[:, occupied_mask]
    localized = lo.PM(molecule, occupied).kernel()

    active_ao = _active_atom_ao_mask(molecule, active_atom_indices)
    populations = _mulliken_active_population(localized, overlap, active_ao)
    partition = partition_orbitals_by_population(
        populations, threshold=population_threshold
    )

    # AO densities: total, active (active-occupied localized orbitals), environment.
    density_total = np.ascontiguousarray(mean_field.make_rdm1())
    active_occ = localized[:, list(partition.active_indices)]
    density_active = (
        2.0 * active_occ @ active_occ.T
    )  # restricted: 2 electrons per orbital
    density_environment = density_total - density_active

    potential_total = np.ascontiguousarray(mean_field.get_veff(molecule, density_total))
    potential_active = np.ascontiguousarray(
        mean_field.get_veff(molecule, density_active)
    )
    projector = levelshift_projector(overlap, density_environment)

    h_emb = embed_one_body_operator(
        core_hamiltonian=core_hamiltonian,
        potential_total=potential_total,
        potential_active=potential_active,
        projector=projector,
        level_shift=level_shift,
    )

    result = EmbeddingResult(
        scheme="projection-wf-in-dft",
        active_atom_indices=tuple(sorted(active_atom_indices)),
        warnings=partition.warnings,
        metadata={
            "environment_functional": environment_functional,
            "level_shift": level_shift,
            "population_threshold": population_threshold,
            "n_active_occupied": len(partition.active_indices),
            "n_environment_occupied": len(partition.environment_indices),
        },
    )

    return EmbeddingArtifacts(
        h_emb=np.ascontiguousarray(h_emb),
        overlap=overlap,
        core_hamiltonian=core_hamiltonian,
        environment_potential=np.ascontiguousarray(potential_total - potential_active),
        partition=partition,
        result=result,
    )


def _active_atom_ao_mask(
    molecule: Any, active_atom_indices: tuple[int, ...]
) -> NDArray[np.bool_]:
    """Boolean mask over AOs: True where the AO is centred on an active atom."""
    n_ao = molecule.nao_nr()
    mask = np.zeros(n_ao, dtype=bool)
    slices = molecule.aoslice_by_atom()
    active = set(active_atom_indices)
    for atom_index, (_, _, ao_start, ao_stop) in enumerate(slices):
        if atom_index in active:
            mask[ao_start:ao_stop] = True
    return mask


def _mulliken_active_population(
    orbitals: NDArray[np.float64],
    overlap: NDArray[np.float64],
    active_ao_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Per-orbital Mulliken population summed over the active-atom AOs.

    For a normalized orbital the total over all AOs is 1, so the active-atom sum is the
    fraction of the orbital that lives on the active region.
    """
    # gross population per AO per orbital: C * (S C), then keep active-atom AO rows.
    gross = orbitals * (overlap @ orbitals)
    return np.asarray(gross[active_ao_mask, :].sum(axis=0), dtype=float)

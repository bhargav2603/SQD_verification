"""Transition-metal spin-gap runner (Colab / Linux, needs PySCF).

What this produces
------------------
For a molecule and two spin states (low-spin vs high-spin) it computes the spin gap
``dE = E(high-spin) - E(low-spin)`` through an evidence-producing campaign:

  * **DFT (B3LYP)** -- the method that is *supposed* to be wrong for these systems; the
    reason spin gaps are the validation wedge in the first place.
  * **CASCI reference (PySCF)** -- exact diagonalisation in the chosen active space, the
    trustworthy classical answer for that active space.
  * **QDR engine (matrix-free, rung 3)** -- the same active-space Hamiltonian solved through
    this engine's own fermion -> qubit -> matrix-free path, in the correct ``(n_alpha,
    n_beta)`` sector (rung-1 open-shell support). This is the number QDR "owns".
  * **SQD/QSCI test subject** -- nested sampled determinant subspaces with variational and
    same-active-space reference diagnostics.
  * **CASSCF/NEVPT2** -- orbital relaxation plus external-space dynamic correlation.
  * **CCSD(T)** -- an independently parameterized comparison, admitted only when it
    converges and clears an open-shell T1 diagnostic.

The QDR/CASCI equality check remains a load-bearing **engine gate**, but it is explicitly
not chemistry validation. The final machine-readable verdict also requires representation,
required orbital roles, SCF/spin stability, active-space and basis ladders, SQD convergence,
dynamic-correlation convergence, and independent-method agreement. A fast open-shell
**sanity molecule (O2)** runs first, and the old Fe/STO-3G/3d-only setup is retained as a
negative control that must emit ``UNTRUSTED``.

How to run in Colab
-------------------
    !pip install -q pyscf matplotlib
    !git clone <this repo> && cd <repo>/packages/compute-engine && pip install -e . ../qdr-core
    !python notebooks/spin_gap_colab.py

Everything here is deliberately explicit (geometries inline, active spaces named) so the
recipe is auditable. The idealised octahedral metal geometries are *illustrative* -- an
adiabatic gap for publication needs a geometry optimised per spin state; swap in your own
coordinates and the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from qdr_compute_engine.active_space.active_space_selector import (
    restrict_integrals,
    select_active_space,
)
from qdr_compute_engine.active_space.avas_selection import (
    DEFAULT_AVAS_THRESHOLD,
    avas_active_integrals,
)
from qdr_compute_engine.hamiltonian.fermionic_builder import build_fermionic_hamiltonian
from qdr_compute_engine.hamiltonian.matrix_free import (
    ground_state_energy_at_spin_sector_matrix_free,
)
from qdr_compute_engine.hamiltonian.spin_penalty import spin_penalized_hamiltonian
from qdr_compute_engine.integrals.pyscf_integrals import compute_integrals
from qdr_compute_engine.mappers.jordan_wigner import map_to_qubits
from qdr_compute_engine.solvers.classical_baseline import (
    alpha_beta_from_spin,
    solve_fci,
)
from qdr_compute_engine.solvers.sqd import (
    run_sqd_campaign,
    sector_configurations,
    solve_sqd,
)
from qdr_compute_engine.types import Atom, Geometry, LengthUnit
from qdr_compute_engine.validation.chemistry import (
    agreement_evidence,
    avas_selection_evidence,
    basis_convergence_evidence,
    basis_evidence,
    convergence_evidence,
    coupled_cluster_reference_evidence,
    environment_evidence,
    geometry_evidence,
    nested_model_convergence_evidence,
    required_orbital_evidence,
    ScientificEstimate,
    spin_evidence,
)
from qdr_compute_engine.validation.pyscf_diagnostics import (
    check_scf_stability,
    run_casscf_nevpt2,
    run_ccsd_t,
)
from qdr_compute_engine.validation.trust import (
    ENGINE_SANITY_PROFILE,
    Evidence,
    EvidenceStatus,
    FailureMode,
    QUANTITATIVE_CHEMISTRY_PROFILE,
    TrustProfile,
    evidence_from_dict,
    evaluate_trust,
)
from qdr_compute_engine.types import QubitHamiltonian
from qdr_core.limits import HARTREE_TO_KCAL_PER_MOL, MAX_EXACT_REFERENCE_QUBITS

#: QDR-vs-PySCF agreement demanded before any gap is reported (Hartree).
CROSS_CHECK_TOLERANCE = 1e-6

RUN_FAILURE_PROFILE = TrustProfile(
    name="run_failure",
    required_codes={"run_failure": FailureMode.INVALID_RESULT},
)


@dataclass(frozen=True)
class SpinState:
    label: str
    spin_multiplicity: int


@dataclass(frozen=True)
class SpinGapCase:
    name: str
    atoms: tuple[Atom, ...]
    charge: int
    basis: str
    low_spin: SpinState
    high_spin: SpinState
    n_active_electrons: int
    n_active_orbitals: int
    #: AVAS target valence orbitals, e.g. ``("Fe 3d",)``. When set, the active space is
    #: chosen by AVAS (the 3d shell is guaranteed active) and ``n_active_*`` above are
    #: ignored -- AVAS decides the size. When ``None``, the index-window selector is used.
    ao_labels: tuple[str, ...] | None = None
    #: AVAS overlap threshold. Lower keeps more (weaker-overlap) orbitals active -- richer
    #: physics but more qubits; raise it to shrink the space when it overruns the exact-
    #: reference ceiling. ``None`` uses the AVAS default (0.2).
    avas_threshold: float | None = None
    #: Published complete-basis reference, kcal/mol (high - low). Context only.
    literature_cbs_kcal: float | None = None
    #: Scientific claim metadata. These fields drive the trust gate; they are not captions.
    transition_metal: bool = False
    quantitative: bool = False
    required_ao_groups: tuple[tuple[str, tuple[str, ...]], ...] = ()
    geometry_kind: str = "illustrative"
    observable_kind: str = "vertical"
    target_environment: str = "gas"
    environment_model: str | None = None
    #: External-space and independent-method calculations are opt-in because they dominate
    #: runtime. Quantitative campaign members enable NEVPT2; the largest enables CCSD(T).
    run_nevpt2: bool = False
    run_ccsd_t: bool = False
    ccsd_frozen_orbitals: int | None = None


@dataclass
class StateResult:
    label: str
    spin_multiplicity: int
    n_alpha: int
    n_beta: int
    n_qubits: int
    e_casci: float
    e_qdr: float
    spin_squared: float
    multiplicity: float
    e_dft: float
    #: AVAS overlap threshold actually used (auto-raised to fit the qubit budget); None for
    #: the index-window path.
    avas_threshold: float | None = None
    avas_requested_threshold: float | None = None
    crosscheck_error: float = 0.0
    sqd: dict[str, object] | None = None
    casscf_converged: bool | None = None
    e_casscf: float | None = None
    e_nevpt2_correction: float | None = None
    e_nevpt2: float | None = None
    casscf_spin_squared: float | None = None
    ccsd_converged: bool | None = None
    e_ccsd: float | None = None
    triples_correction: float | None = None
    e_ccsd_t: float | None = None
    t1_diagnostic: float | None = None
    ccsd_frozen_orbitals: int | None = None
    scf_stable: bool | None = None


def _octahedral(metal: str, ligand_atom: str, bond_length: float) -> list[Atom]:
    """A metal atom at the origin with six identical single-atom ligands along the axes.

    A single-atom ligand model (e.g. bare F or Cl donors) keeps the demonstration geometry
    unambiguous. For molecular ligands (H2O, NCH) substitute real coordinates.
    """
    directions = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]
    atoms = [Atom(metal, (0.0, 0.0, 0.0))]
    atoms += [
        Atom(ligand_atom, (d[0] * bond_length, d[1] * bond_length, d[2] * bond_length))
        for d in directions
    ]
    return atoms


def _dft_energy(case: SpinGapCase, state: SpinState) -> float:
    """B3LYP total energy for one spin state -- the 'DFT gets this wrong' baseline."""
    from pyscf import dft, gto

    molecule = gto.Mole()
    molecule.atom = [(a.symbol, a.position) for a in case.atoms]
    molecule.basis = case.basis
    molecule.charge = case.charge
    molecule.spin = state.spin_multiplicity - 1
    molecule.unit = "Angstrom"
    molecule.verbose = 0
    molecule.build()

    mean_field = dft.RKS(molecule) if molecule.spin == 0 else dft.ROKS(molecule)
    mean_field.xc = "b3lyp"
    mean_field.max_cycle = 200
    energy = float(mean_field.kernel())
    if not mean_field.converged:
        # Charged transition-metal centres (e.g. [FeF6]4-) stall first-order DIIS; hand the
        # stalled density to the second-order (Newton) solver, matching the HF path.
        mean_field = mean_field.newton()
        energy = float(mean_field.kernel(mean_field.mo_coeff, mean_field.mo_occ))
    if not mean_field.converged:
        raise RuntimeError(f"B3LYP did not converge for {case.name} {state.label}")
    return energy


def _run_sqd_test_subject(
    hamiltonian: QubitHamiltonian,
    n_alpha: int,
    n_beta: int,
    reference_energy: float,
    *,
    seed: int = 271828,
    shots: int = 4096,
    n_batches: int = 3,
) -> dict[str, object]:
    """Best-case sampled SQD campaign, instrumented against the exact small-space answer.

    This is intentionally a test subject rather than independent chemical evidence: samples
    come from the exact sector eigenvector so the experiment isolates determinant-selection
    and projected-diagonalization behavior from state-preparation and hardware noise.
    """
    exact, coefficients = solve_sqd(
        hamiltonian,
        n_alpha,
        n_beta,
        reference_energy=reference_energy,
    )
    sector = sector_configurations(
        hamiltonian.n_qubits,
        n_alpha,
        n_beta,
    )
    probabilities = np.abs(coefficients) ** 2
    probabilities = probabilities / probabilities.sum()
    batches = [
        np.random.default_rng(seed + batch).choice(
            sector,
            size=shots,
            p=probabilities,
        )
        for batch in range(n_batches)
    ]
    minimum_distinct = min(int(np.unique(samples).size) for samples in batches)
    dimensions = tuple(
        sorted(
            {
                min(target, minimum_distinct)
                for target in (8, 32, 128, 512, minimum_distinct)
            }
        )
    )
    campaigns = [
        run_sqd_campaign(
            hamiltonian,
            n_alpha,
            n_beta,
            configurations=samples,
            dimensions=dimensions,
            reference_energy=reference_energy,
        )
        for samples in batches
    ]
    campaign = campaigns[0]
    batch_energies = [item.steps[-1].energy for item in campaigns]
    batch_spread = max(batch_energies) - min(batch_energies)
    return {
        "sampling_source": "exact-sector-wavefunction-test-subject",
        "shots_per_batch": shots,
        "n_batches": n_batches,
        "sector_dimension": campaign.sector_dimension,
        "sector_retention": min(item.sector_retention for item in campaigns),
        "batch_final_energies": batch_energies,
        "batch_energy_spread": batch_spread,
        "exact_sector_energy": exact.energy,
        "final_reference_deviation": campaign.reference_deviation,
        "final_energy_change": campaign.final_energy_change,
        "steps": [
            {
                "requested_dimension": step.requested_dimension,
                "subspace_dimension": step.subspace_dimension,
                "energy": step.energy,
                "energy_change": step.energy_change,
                "is_exact": step.is_exact,
                "configurations": list(step.configurations),
            }
            for step in campaign.steps
        ],
    }


def _solve_state(case: SpinGapCase, state: SpinState) -> StateResult:
    """Active-space CASCI (PySCF) and the QDR engine, cross-checked, for one spin state."""
    geometry = Geometry(
        atoms=case.atoms,
        charge=case.charge,
        spin_multiplicity=state.spin_multiplicity,
        units=LengthUnit.ANGSTROM,
    )
    avas = None
    if case.ao_labels is not None:
        # AVAS: the named valence shells (e.g. Fe 3d + F 2p) are active by construction. The
        # exact CASCI/QDR cross-check materialises a state vector per sector, so it is only
        # feasible up to MAX_EXACT_REFERENCE_QUBITS; hand AVAS that ceiling as an orbital
        # budget so it auto-raises its threshold to pack the largest covalent space that fits.
        avas = avas_active_integrals(
            geometry,
            ao_labels=case.ao_labels,
            basis=case.basis,
            threshold=(
                DEFAULT_AVAS_THRESHOLD
                if case.avas_threshold is None
                else case.avas_threshold
            ),
            max_active_orbitals=MAX_EXACT_REFERENCE_QUBITS // 2,
        )
        restricted, active_space = avas.integrals, avas.active_space
        avas_threshold_used: float | None = avas.threshold
        avas_requested_threshold: float | None = avas.requested_threshold
    else:
        integrals = compute_integrals(geometry, basis=case.basis)
        active_space = select_active_space(
            integrals,
            n_active_orbitals=case.n_active_orbitals,
            n_active_electrons=case.n_active_electrons,
        )
        restricted = restrict_integrals(integrals, active_space)
        avas_threshold_used = None
        avas_requested_threshold = None

    n_alpha, n_beta = alpha_beta_from_spin(restricted.n_electrons, state.spin_multiplicity)

    # (1) Classical reference: PySCF CASCI in this active space and spin sector.
    reference = solve_fci(restricted, spin_multiplicity=state.spin_multiplicity)

    # (2) QDR's own engine: fermion -> qubit -> matrix-free, same active space + sector.
    # restricted.e_nuc (nuclear repulsion + frozen-core shift) is already folded into the
    # InteractionOperator's constant by build_fermionic_hamiltonian and carried into the
    # qubit Hamiltonian's identity term by Jordan-Wigner, so the matrix-free eigenvalue is
    # already the *total* energy. Adding e_nuc here would double-count it -- the golden H2
    # sector test (test_open_shell.test_h2_singlet_sector_is_the_fci_energy) pins this.
    # The (n_alpha, n_beta) sector fixes only S_z, so a requested singlet would otherwise
    # collapse onto a lower triplet's M_s=0 component; the S^2 penalty pins the actual spin
    # without shifting its energy (see hamiltonian/spin_penalty.py), exactly as fix_spin_
    # does for the CASCI reference above -- which is what keeps the cross-check honest.
    hamiltonian = map_to_qubits(build_fermionic_hamiltonian(restricted))
    hamiltonian = spin_penalized_hamiltonian(hamiltonian, spin_multiplicity=state.spin_multiplicity)
    e_qdr = ground_state_energy_at_spin_sector_matrix_free(hamiltonian, n_alpha, n_beta)
    crosscheck_error = abs(e_qdr - reference.energy)

    # The gate: a disagreement means the engine is wrong; do not report a gap built on it.
    if crosscheck_error > CROSS_CHECK_TOLERANCE:
        raise AssertionError(
            f"cross-check FAILED for {case.name} {state.label}: "
            f"QDR {e_qdr:.9f} vs CASCI {reference.energy:.9f} Ha "
            f"(|dE| = {crosscheck_error:.2e}); refusing to report a gap"
        )

    sqd = _run_sqd_test_subject(hamiltonian, n_alpha, n_beta, reference.energy)
    multireference = None
    coupled_cluster = None
    scf_stable = None
    if avas is not None and avas.mean_field is not None:
        scf_stable = check_scf_stability(avas.mean_field)
    if case.run_nevpt2:
        if avas is None:
            raise ValueError("NEVPT2 requires an AVAS case so the full PySCF work objects exist")
        multireference = run_casscf_nevpt2(
            avas,
            spin_multiplicity=state.spin_multiplicity,
        )
    if case.run_ccsd_t:
        if avas is None or avas.mean_field is None:
            raise ValueError("CCSD(T) requires an AVAS case with a retained mean-field object")
        coupled_cluster = run_ccsd_t(
            avas.mean_field,
            frozen_orbitals=case.ccsd_frozen_orbitals,
        )

    return StateResult(
        label=state.label,
        spin_multiplicity=state.spin_multiplicity,
        n_alpha=n_alpha,
        n_beta=n_beta,
        n_qubits=active_space.n_qubits,
        e_casci=reference.energy,
        e_qdr=e_qdr,
        spin_squared=float(reference.spin_squared or 0.0),
        multiplicity=float(reference.multiplicity or 0.0),
        e_dft=_dft_energy(case, state),
        avas_threshold=avas_threshold_used,
        avas_requested_threshold=avas_requested_threshold,
        crosscheck_error=crosscheck_error,
        sqd=sqd,
        casscf_converged=(
            None if multireference is None else multireference.converged
        ),
        e_casscf=None if multireference is None else multireference.e_casscf,
        e_nevpt2_correction=(
            None if multireference is None else multireference.e_nevpt2_correction
        ),
        e_nevpt2=None if multireference is None else multireference.e_nevpt2,
        casscf_spin_squared=(
            None if multireference is None else multireference.spin_squared
        ),
        ccsd_converged=(
            None if coupled_cluster is None else coupled_cluster.converged
        ),
        e_ccsd=None if coupled_cluster is None else coupled_cluster.e_ccsd,
        triples_correction=(
            None if coupled_cluster is None else coupled_cluster.triples_correction
        ),
        e_ccsd_t=None if coupled_cluster is None else coupled_cluster.e_ccsd_t,
        t1_diagnostic=(
            None if coupled_cluster is None else coupled_cluster.t1_diagnostic
        ),
        ccsd_frozen_orbitals=(
            None if coupled_cluster is None else coupled_cluster.frozen_orbitals
        ),
        scf_stable=scf_stable,
    )


def build_model_evidence(case: SpinGapCase) -> list[Evidence]:
    """Evidence available from the declared model before an expensive calculation."""
    selected_labels = case.ao_labels or ()
    evidence = [
        basis_evidence(
            case.basis,
            transition_metal=case.transition_metal,
            quantitative=case.quantitative,
        ),
        required_orbital_evidence(
            selected_labels=selected_labels,
            required_groups=dict(case.required_ao_groups),
        ),
        geometry_evidence(
            geometry_kind=case.geometry_kind,
            observable_kind=case.observable_kind,
        ),
        environment_evidence(
            target_environment=case.target_environment,
            model=case.environment_model,
        ),
    ]
    if case.quantitative:
        evidence.append(
            Evidence(
                code="active_space_ladder",
                status=EvidenceStatus.MISSING,
                failure_mode=FailureMode.ACTIVE_SPACE_NOT_CONVERGED,
                message="this individual case contains one active space; campaign aggregation is required",
            )
        )
        evidence.extend(
            [
                Evidence(
                    code="active_space_boundary",
                    status=EvidenceStatus.MISSING,
                    failure_mode=FailureMode.ACTIVE_SPACE_BOUNDARY_LEAKAGE,
                    message="excluded-orbital natural occupations were not measured",
                ),
                Evidence(
                    code="orbital_space_nesting",
                    status=EvidenceStatus.MISSING,
                    failure_mode=FailureMode.ORBITAL_SPACE_NOT_NESTED,
                    message="orbital counts alone do not prove AVAS subspace containment",
                ),
                Evidence(
                    code="sampling_provenance",
                    status=EvidenceStatus.MISSING,
                    failure_mode=FailureMode.SAMPLING_PROVENANCE_MISSING,
                    message=(
                        "SQD samples come from the exact sector wavefunction test subject, "
                        "not an independently prepared circuit or hardware sampler"
                    ),
                ),
            ]
        )
        evidence.append(
            Evidence(
                code="basis_convergence",
                status=EvidenceStatus.MISSING,
                failure_mode=FailureMode.BASIS_NOT_CONVERGED,
                message="this individual case contains one AO basis; campaign aggregation is required",
            )
        )
        if not case.run_nevpt2:
            evidence.append(
                Evidence(
                    code="dynamic_correlation",
                    status=EvidenceStatus.MISSING,
                    failure_mode=FailureMode.DYNAMIC_CORRELATION_MISSING,
                    message="CASCI/SQD alone does not recover correlation outside the active space",
                )
            )
        if not (case.run_nevpt2 and case.run_ccsd_t):
            evidence.append(
                Evidence(
                    code="independent_method_agreement",
                    status=EvidenceStatus.MISSING,
                    failure_mode=FailureMode.METHODS_DISAGREE,
                    message="QDR and CASCI share one Hamiltonian and count as one estimate",
                )
            )
    return evidence


def _result_evidence(case: SpinGapCase, low: StateResult, high: StateResult) -> list[Evidence]:
    crosscheck_error = max(low.crosscheck_error, high.crosscheck_error)
    evidence = [
        Evidence(
            code="engine_crosscheck",
            status=(
                EvidenceStatus.PASS
                if crosscheck_error <= CROSS_CHECK_TOLERANCE
                else EvidenceStatus.FAIL
            ),
            failure_mode=FailureMode.ENGINE_DISAGREEMENT,
            value=crosscheck_error,
            threshold=CROSS_CHECK_TOLERANCE,
            units="Ha",
            message="QDR and PySCF CASCI agreement for the same active-space Hamiltonian",
            metadata={"scope": "engine_only"},
        ),
        spin_evidence(
            code="spin_purity_low",
            spin_squared=low.spin_squared,
            spin_multiplicity=low.spin_multiplicity,
        ),
        spin_evidence(
            code="spin_purity_high",
            spin_squared=high.spin_squared,
            spin_multiplicity=high.spin_multiplicity,
        ),
    ]
    if case.quantitative:
        for label, state in (("low", low), ("high", high)):
            evidence.append(
                Evidence(
                    code=f"scf_stability_{label}",
                    status=(
                        EvidenceStatus.MISSING
                        if state.scf_stable is None
                        else (
                            EvidenceStatus.PASS
                            if state.scf_stable
                            else EvidenceStatus.FAIL
                        )
                    ),
                    failure_mode=FailureMode.SCF_UNSTABLE,
                    value=state.scf_stable,
                    message="internal RHF/ROHF orbital-Hessian stability analysis",
                )
            )
    for label, state in (("low", low), ("high", high)):
        sqd = state.sqd or {}
        raw_steps = sqd.get("steps", [])
        steps = raw_steps if isinstance(raw_steps, list) else []
        energies = [
            float(step["energy"])
            for step in steps
            if isinstance(step, dict) and "energy" in step
        ]
        evidence.append(
            convergence_evidence(
                code=f"sqd_convergence_{label}",
                values=energies,
                tolerance=1.6e-3,
                failure_mode=FailureMode.SQD_NOT_CONVERGED,
                units="Ha",
            )
        )
        raw_deviation = sqd.get("final_reference_deviation")
        deviation = (
            float(raw_deviation)
            if isinstance(raw_deviation, (int, float))
            else None
        )
        evidence.append(
            Evidence(
                code=f"sqd_reference_{label}",
                status=(
                    EvidenceStatus.MISSING
                    if deviation is None
                    else (
                        EvidenceStatus.PASS
                        if deviation <= 1.6e-3
                        else EvidenceStatus.FAIL
                    )
                ),
                failure_mode=FailureMode.SQD_NOT_CONVERGED,
                value=deviation,
                threshold=1.6e-3,
                units="Ha",
                message="sample-selected SQD deviation from the exact active-space reference",
                metadata={"sampling_source": sqd.get("sampling_source")},
            )
        )
        raw_retention = sqd.get("sector_retention")
        retention = (
            float(raw_retention)
            if isinstance(raw_retention, (int, float))
            else None
        )
        evidence.append(
            Evidence(
                code=f"sqd_sector_retention_{label}",
                status=(
                    EvidenceStatus.MISSING
                    if retention is None
                    else (
                        EvidenceStatus.PASS
                        if retention >= 0.9
                        else EvidenceStatus.FAIL
                    )
                ),
                failure_mode=FailureMode.LOW_SECTOR_RETENTION,
                value=retention,
                threshold=0.9,
                units="fraction",
                message="fraction of raw SQD samples in the requested electron/spin sector",
            )
        )
        raw_spread = sqd.get("batch_energy_spread")
        spread = (
            float(raw_spread)
            if isinstance(raw_spread, (int, float))
            else None
        )
        evidence.append(
            Evidence(
                code=f"sqd_reproducibility_{label}",
                status=(
                    EvidenceStatus.MISSING
                    if spread is None
                    else (
                        EvidenceStatus.PASS
                        if spread <= 1.6e-3
                        else EvidenceStatus.FAIL
                    )
                ),
                failure_mode=FailureMode.SAMPLING_NOT_REPRODUCIBLE,
                value=spread,
                threshold=1.6e-3,
                units="Ha",
                message="spread of final SQD energies across independent sample batches",
            )
        )
    if case.ao_labels is not None:
        requested = (
            low.avas_requested_threshold
            if low.avas_requested_threshold is not None
            else (case.avas_threshold or 0.2)
        )
        used = max(low.avas_threshold or requested, high.avas_threshold or requested)
        evidence.append(
            avas_selection_evidence(
                requested_threshold=requested,
                used_threshold=used,
                max_active_orbitals=MAX_EXACT_REFERENCE_QUBITS // 2,
            )
        )
    if case.run_nevpt2:
        converged = bool(low.casscf_converged and high.casscf_converged)
        evidence.append(
            Evidence(
                code="dynamic_correlation",
                status=EvidenceStatus.PASS if converged else EvidenceStatus.FAIL,
                failure_mode=FailureMode.DYNAMIC_CORRELATION_NOT_CONVERGED,
                message=(
                    "both state-specific CASSCF references converged and NEVPT2 was evaluated"
                    if converged
                    else "CASSCF/NEVPT2 failed for at least one spin state"
                ),
            )
        )
        evidence.extend(
            [
                spin_evidence(
                    code="casscf_spin_purity_low",
                    spin_squared=low.casscf_spin_squared,
                    spin_multiplicity=low.spin_multiplicity,
                ),
                spin_evidence(
                    code="casscf_spin_purity_high",
                    spin_squared=high.casscf_spin_squared,
                    spin_multiplicity=high.spin_multiplicity,
                ),
            ]
        )
    if case.run_ccsd_t:
        t1_values = [
            value
            for value in (low.t1_diagnostic, high.t1_diagnostic)
            if value is not None
        ]
        evidence.append(
            coupled_cluster_reference_evidence(
                converged=bool(low.ccsd_converged and high.ccsd_converged),
                t1_diagnostic=max(t1_values) if len(t1_values) == 2 else None,
                open_shell=(
                    low.spin_multiplicity > 1 or high.spin_multiplicity > 1
                ),
            )
        )
        method_values: dict[str, ScientificEstimate] = {}
        if low.e_nevpt2 is not None and high.e_nevpt2 is not None:
            method_values["casscf_nevpt2"] = ScientificEstimate(
                (high.e_nevpt2 - low.e_nevpt2) * HARTREE_TO_KCAL_PER_MOL,
                independence_group="multireference_perturbation",
            )
        if low.e_ccsd_t is not None and high.e_ccsd_t is not None:
            method_values["ccsd_t"] = ScientificEstimate(
                (high.e_ccsd_t - low.e_ccsd_t) * HARTREE_TO_KCAL_PER_MOL,
                independence_group="single_reference_coupled_cluster",
            )
        evidence.append(
            agreement_evidence(
                method_values,
                tolerance=1.0,
                units="kcal/mol",
            )
        )
    return evidence


def run_case(case: SpinGapCase) -> dict:
    """Run both spin states and assemble the gap every way."""
    low = _solve_state(case, case.low_spin)
    high = _solve_state(case, case.high_spin)

    to_kcal = HARTREE_TO_KCAL_PER_MOL
    gap_qdr = (high.e_qdr - low.e_qdr) * to_kcal
    gap_casci = (high.e_casci - low.e_casci) * to_kcal
    gap_dft = (high.e_dft - low.e_dft) * to_kcal
    gap_values = {"dft": gap_dft, "casci": gap_casci, "qdr": gap_qdr}
    if low.e_casscf is not None and high.e_casscf is not None:
        gap_values["casscf"] = (high.e_casscf - low.e_casscf) * to_kcal
    if low.e_nevpt2 is not None and high.e_nevpt2 is not None:
        gap_values["nevpt2"] = (high.e_nevpt2 - low.e_nevpt2) * to_kcal
    if low.e_ccsd_t is not None and high.e_ccsd_t is not None:
        gap_values["ccsd_t"] = (high.e_ccsd_t - low.e_ccsd_t) * to_kcal

    # Report the active space actually solved: for an AVAS case that is what AVAS chose, not
    # the declared n_active_* (which AVAS ignores), so the printout never overstates the model.
    active_orbitals = low.n_qubits // 2
    active_electrons = low.n_alpha + low.n_beta
    if case.ao_labels is not None:
        selection = f" via AVAS {case.ao_labels} @ threshold {low.avas_threshold:.2f}"
    else:
        selection = ""
    print(f"\n=== {case.name} ===")
    print(
        f"active space: ({active_electrons}e, {active_orbitals}o) "
        f"-> {low.n_qubits} qubits{selection}"
    )
    for state in (low, high):
        print(
            f"  {state.label:>10}: E(CASCI)={state.e_casci:.6f}  "
            f"E(QDR)={state.e_qdr:.6f}  <S^2>={state.spin_squared:.3f}  "
            f"2S+1={state.multiplicity:.2f}  [cross-check OK]"
        )
    print("  spin gap (high - low):")
    print(f"    DFT B3LYP     : {gap_dft:+8.2f} kcal/mol")
    print(f"    CASCI (PySCF) : {gap_casci:+8.2f} kcal/mol")
    print(f"    QDR engine    : {gap_qdr:+8.2f} kcal/mol")
    if "casscf" in gap_values:
        print(f"    CASSCF        : {gap_values['casscf']:+8.2f} kcal/mol")
    if "nevpt2" in gap_values:
        print(f"    CASSCF/NEVPT2 : {gap_values['nevpt2']:+8.2f} kcal/mol")
    if "ccsd_t" in gap_values:
        print(f"    CCSD(T)       : {gap_values['ccsd_t']:+8.2f} kcal/mol")
    if case.literature_cbs_kcal is not None:
        print(
            f"    literature CBS: {case.literature_cbs_kcal:+8.2f} kcal/mol "
            "(context only; needs post-processing to match)"
            )

    trust = evaluate_trust(
        [*build_model_evidence(case), *_result_evidence(case, low, high)],
        profile=(
            QUANTITATIVE_CHEMISTRY_PROFILE
            if case.quantitative
            else ENGINE_SANITY_PROFILE
        ),
    )
    modes = ", ".join(mode.value for mode in (*trust.failure_modes, *trust.missing_modes))
    print(f"  trust verdict    : {trust.verdict.value}" + (f" [{modes}]" if modes else ""))

    return {
        "name": case.name,
        "input": {
            "atoms_angstrom": [
                {"symbol": atom.symbol, "position": list(atom.position)}
                for atom in case.atoms
            ],
            "charge": case.charge,
            "low_spin": asdict(case.low_spin),
            "high_spin": asdict(case.high_spin),
            "basis": case.basis,
            "ao_labels": list(case.ao_labels or ()),
            "avas_threshold": case.avas_threshold,
            "observable_kind": case.observable_kind,
            "geometry_kind": case.geometry_kind,
            "target_environment": case.target_environment,
            "environment_model": case.environment_model,
            "ccsd_frozen_orbitals": case.ccsd_frozen_orbitals,
        },
        "basis": case.basis,
        "active_space": [active_electrons, active_orbitals],
        "n_qubits": low.n_qubits,
        "states": {"low": asdict(low), "high": asdict(high)},
        "gap_kcal": gap_values,
        "literature_cbs_kcal": case.literature_cbs_kcal,
        "trust_report": trust.to_dict(),
    }


def build_campaign_evidence(results: list[dict]) -> list[Evidence]:
    """Build chemistry evidence across a strictly nested active-space campaign."""
    model_sizes = [int(result["n_qubits"]) // 2 for result in results]
    active_stop = len(results)
    if len(model_sizes) >= 2 and model_sizes[-1] == model_sizes[-2]:
        active_stop -= 1
    active_results = results[:active_stop]
    active_sizes = model_sizes[:active_stop]
    nevpt2_gaps = [
        float(result["gap_kcal"]["nevpt2"])
        for result in active_results
        if "nevpt2" in result["gap_kcal"]
    ]
    dynamic_contributions = [
        float(result["gap_kcal"]["nevpt2"]) - float(result["gap_kcal"]["casscf"])
        for result in active_results
        if "nevpt2" in result["gap_kcal"] and "casscf" in result["gap_kcal"]
    ]
    basis_gaps = [
        float(result["gap_kcal"]["nevpt2"])
        for result in results[-2:]
        if (
            len(results) >= 2
            and model_sizes[-1] == model_sizes[-2]
            and "nevpt2" in result["gap_kcal"]
        )
    ]
    final_methods = {
        method: ScientificEstimate(
            float(results[-1]["gap_kcal"][method]),
            independence_group=group,
        )
        for method, group in (
            ("nevpt2", "multireference_perturbation"),
            ("ccsd_t", "single_reference_coupled_cluster"),
        )
        if method in results[-1]["gap_kcal"]
    }
    return [
        nested_model_convergence_evidence(
            code="active_space_ladder",
            model_sizes=active_sizes,
            values=nevpt2_gaps,
            tolerance=1.0,
            failure_mode=FailureMode.ACTIVE_SPACE_NOT_CONVERGED,
            units="kcal/mol",
        ),
        nested_model_convergence_evidence(
            code="dynamic_correlation",
            model_sizes=active_sizes,
            values=dynamic_contributions,
            tolerance=1.0,
            failure_mode=FailureMode.DYNAMIC_CORRELATION_NOT_CONVERGED,
            units="kcal/mol",
        ),
        basis_convergence_evidence(
            observable_values_kcal=basis_gaps,
            tolerance_kcal=1.0,
        ),
        agreement_evidence(
            final_methods,
            tolerance=1.0,
            units="kcal/mol",
        ),
    ]


def build_campaign_member_evidence(results: list[dict]) -> list[Evidence]:
    """Carry solver/reference health from every rung into the aggregate verdict."""
    exact_codes = {"engine_crosscheck", "avas_cap", "dynamic_correlation"}
    code_prefixes = (
        "spin_purity_",
        "casscf_spin_purity_",
        "scf_stability_",
        "sqd_convergence_",
        "sqd_reference_",
        "sqd_sector_retention_",
        "sqd_reproducibility_",
    )
    evidence: list[Evidence] = []
    for index, result in enumerate(results, start=1):
        for payload in result["trust_report"]["evidence"]:
            code = str(payload["code"])
            if code not in exact_codes and not code.startswith(code_prefixes):
                continue
            prefixed = dict(payload)
            prefixed["code"] = f"rung_{index}_{code}"
            evidence.append(evidence_from_dict(prefixed))
    return evidence


def run_active_space_campaign(cases: tuple[SpinGapCase, ...]) -> dict:
    """Run nested models and emit one verdict for the largest, converged model."""
    if len(cases) < 2:
        raise ValueError("an active-space campaign needs at least two nested cases")
    results = [run_case(case) for case in cases]
    replaced = {
        "active_space_ladder",
        "basis_convergence",
        "dynamic_correlation",
        "independent_method_agreement",
    }
    final_evidence = [
        evidence_from_dict(payload)
        for payload in results[-1]["trust_report"]["evidence"]
        if payload["code"] not in replaced
    ]
    trust = evaluate_trust(
        [
            *final_evidence,
            *build_campaign_member_evidence(results),
            *build_campaign_evidence(results),
        ],
        profile=QUANTITATIVE_CHEMISTRY_PROFILE,
    )
    modes = ", ".join(mode.value for mode in (*trust.failure_modes, *trust.missing_modes))
    print(
        f"\n=== FeS campaign verdict: {trust.verdict.value}"
        + (f" [{modes}]" if modes else "")
        + " ==="
    )
    return {
        "name": "FeS nested active-space reliability campaign",
        "members": results,
        "trust_report": trust.to_dict(),
    }


def plot_gaps(results: list[dict], output_dir: Path) -> None:
    """Bar chart of the spin gap by method, one group per molecule."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["dft", "casci", "qdr"]
    labels = ["DFT B3LYP", "CASCI (ref)", "QDR engine"]
    x = np.arange(len(results))
    width = 0.25

    fig, ax = plt.subplots(figsize=(1.8 * len(results) + 3, 5))
    for offset, (method, label) in enumerate(zip(methods, labels, strict=True)):
        values = [r["gap_kcal"][method] for r in results]
        ax.bar(x + (offset - 1) * width, values, width, label=label)

    for index, r in enumerate(results):
        cbs = r.get("literature_cbs_kcal")
        if cbs is not None:
            ax.plot(
                [index - 1.6 * width, index + 1.6 * width],
                [cbs, cbs],
                "k--",
                linewidth=1.2,
                label="literature CBS" if index == 0 else None,
            )

    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in results], rotation=15, ha="right")
    ax.set_ylabel("spin gap  E(high) - E(low)  [kcal/mol]")
    ax.set_title("Transition-metal spin gap: DFT vs active-space reference vs QDR engine")
    ax.legend()
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "spin_gap.png"
    fig.savefig(path, dpi=300)
    print(f"\nwrote {path}")


def _dependency_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "scipy", "openfermion", "pyscf"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "absent"
    return versions


def _write_strict_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


#: A fast open-shell sanity check: O2's ground state is the triplet, so the gap must be
#: NEGATIVE (triplet below singlet). If this comes out positive, the open-shell machinery
#: is wrong and nothing below is trustworthy.
O2 = SpinGapCase(
    name="O2 (sanity: triplet ground state)",
    atoms=(Atom("O", (0.0, 0.0, 0.0)), Atom("O", (0.0, 0.0, 1.208))),
    charge=0,
    basis="sto-3g",
    low_spin=SpinState("singlet", 1),
    high_spin=SpinState("triplet", 3),
    n_active_electrons=8,
    n_active_orbitals=6,
    geometry_kind="experimental",
    observable_kind="vertical",
)

#: The metal target. [FeF6]^4- as a single-atom-ligand octahedral model keeps the geometry
#: unambiguous while exercising the full open-shell + active-space + matrix-free path on a
#: real 3d^6 Fe(II) centre (low-spin singlet vs high-spin quintet).
#:
#: A bare Fe-3d CAS(6,5) in STO-3G gives a *qualitatively wrong* gap (low-spin ground state,
#: hundreds of kcal/mol) because it omits the ligand field: the 3d shell alone has no way to
#: feel the F donors. Two changes fix the physics of the selection:
#:   * ``def2-svp`` -- a real (double-zeta) basis, so the ligand field exists at the orbital
#:     level instead of the toy STO-3G one; and
#:   * ``("Fe 3d", "F 2p")`` -- the metal d shell *and* the fluorine 2p donors, so Fe-F
#:     covalency (the actual origin of the ligand-field splitting) is inside the active space.
#: Active space = the Fe 3d shell (CAS(6,5) = 10 qubits), which fits the exact CASCI/QDR
#: cross-check comfortably. Note what does *not* work here: adding ``"F 2p"`` for Fe-F
#: covalency pulls in all six fluorines' 2p shells (>20 orbitals), and no AVAS threshold can
#: shrink it -- the non-bonding F lone pairs are essentially pure atomic 2p, so their overlap
#: with the target is ~1 and they are always selected. Covalency is therefore irreducibly too
#: large for an exact reference; it belongs in the WF-in-DFT embedding path (dynamic
#: correlation from the environment), not a bigger CAS. This case keeps the engine honest end
#: to end; the *physics* of the Fe gap is the embedding story, not this minimal CAS.
FE_OCTAHEDRAL = SpinGapCase(
    name="[FeF6]4- NEGATIVE CONTROL (known inadequate model)",
    atoms=tuple(_octahedral("Fe", "F", bond_length=2.00)),
    charge=-4,
    basis="sto-3g",
    low_spin=SpinState("singlet", 1),
    high_spin=SpinState("quintet", 5),
    n_active_electrons=6,
    n_active_orbitals=5,
    ao_labels=("Fe 3d",),
    transition_metal=True,
    quantitative=True,
    required_ao_groups=(
        ("metal_valence", ("Fe 3d",)),
        ("ligand_donor", ("F 2p",)),
    ),
    geometry_kind="illustrative",
    observable_kind="adiabatic",
)


#: R5 production-sized test subject.  FeS is a chemically decisive high-spin benchmark
#: used in arXiv:2601.10813.  The 2.017 Angstrom bond length is the experimental equilibrium
#: distance of the X 5-Delta ground state; the reported quantity here is explicitly a
#: *vertical* quintet-triplet separation at that common geometry.  The campaign uses
#: def2-SVP for active-space growth and repeats the largest space in def2-TZVP; unlike the
#: ANO-RCC-MB model used for small algorithm tests, neither is a minimal AO basis.
FES_D_ONLY = SpinGapCase(
    name="FeS ladder 1: Fe 3d",
    atoms=(Atom("Fe", (0.0, 0.0, 0.0)), Atom("S", (0.0, 0.0, 2.017))),
    charge=0,
    basis="def2-svp",
    low_spin=SpinState("triplet", 3),
    high_spin=SpinState("quintet", 5),
    n_active_electrons=6,
    n_active_orbitals=5,
    ao_labels=("Fe 3d",),
    transition_metal=True,
    quantitative=True,
    required_ao_groups=(
        ("metal_valence", ("Fe 3d",)),
        ("ligand_valence", ("S 3p",)),
    ),
    geometry_kind="experimental",
    observable_kind="vertical",
    run_nevpt2=True,
)

FES_DP = SpinGapCase(
    name="FeS ladder 2: Fe 3d + S 3p",
    atoms=FES_D_ONLY.atoms,
    charge=0,
    basis="def2-svp",
    low_spin=FES_D_ONLY.low_spin,
    high_spin=FES_D_ONLY.high_spin,
    n_active_electrons=6,
    n_active_orbitals=8,
    ao_labels=("Fe 3d", "S 3p"),
    transition_metal=True,
    quantitative=True,
    required_ao_groups=FES_D_ONLY.required_ao_groups,
    geometry_kind="experimental",
    observable_kind="vertical",
    run_nevpt2=True,
)

FES_DPS_SMALL_BASIS = SpinGapCase(
    name="FeS ladder 3: Fe 3d + Fe 4s + S 3p / def2-SVP",
    atoms=FES_D_ONLY.atoms,
    charge=0,
    basis="def2-svp",
    low_spin=FES_D_ONLY.low_spin,
    high_spin=FES_D_ONLY.high_spin,
    n_active_electrons=6,
    n_active_orbitals=9,
    ao_labels=("Fe 3d", "Fe 4s", "S 3p"),
    transition_metal=True,
    quantitative=True,
    required_ao_groups=(
        ("metal_valence", ("Fe 3d",)),
        ("ligand_valence", ("S 3p",)),
    ),
    geometry_kind="experimental",
    observable_kind="vertical",
    run_nevpt2=True,
)

FES_VERTICAL = SpinGapCase(
    name="FeS ladder 4: Fe 3d + Fe 4s + S 3p / def2-TZVP",
    atoms=FES_D_ONLY.atoms,
    charge=0,
    basis="def2-tzvp",
    low_spin=FES_D_ONLY.low_spin,
    high_spin=FES_D_ONLY.high_spin,
    n_active_electrons=6,
    n_active_orbitals=9,
    ao_labels=("Fe 3d", "Fe 4s", "S 3p"),
    transition_metal=True,
    quantitative=True,
    required_ao_groups=FES_DPS_SMALL_BASIS.required_ao_groups,
    geometry_kind="experimental",
    observable_kind="vertical",
    run_nevpt2=True,
    run_ccsd_t=True,
    ccsd_frozen_orbitals=14,
)


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    fes_campaign: dict | None = None
    run_report: dict[str, object] = {
        "schema_version": 1,
        "status": "RUNNING",
        "versions": _dependency_versions(),
        "results": results,
        "campaign": None,
        "failure": None,
    }
    try:
        results.append(run_case(O2))
        results.append(run_case(FE_OCTAHEDRAL))
        fes_campaign = run_active_space_campaign(
            (FES_D_ONLY, FES_DP, FES_DPS_SMALL_BASIS, FES_VERTICAL)
        )
        results.extend(fes_campaign["members"])

        # The sanity gate: O2's triplet must be below its singlet.
        o2 = results[0]
        assert o2["gap_kcal"]["qdr"] < 0.0, (
            f"SANITY FAILED: O2 triplet is not below its singlet "
            f"(gap {o2['gap_kcal']['qdr']:+.2f} kcal/mol); open-shell path is wrong"
        )
        print("\nsanity OK: O2 triplet is the ground state, QDR == CASCI on every state.")

        run_report["status"] = "COMPLETED"
        run_report["campaign"] = fes_campaign
        _write_strict_json(output_dir / "spin_gap.json", results)
        _write_strict_json(output_dir / "spin_gap_campaign.json", fes_campaign)
        print(f"wrote {output_dir / 'spin_gap.json'}")
        print(f"wrote {output_dir / 'spin_gap_campaign.json'}")
        plot_gaps([results[0], results[1], results[-1]], output_dir)
    except Exception as exc:
        failure = Evidence(
            code="run_failure",
            status=EvidenceStatus.FAIL,
            failure_mode=FailureMode.INVALID_RESULT,
            value={"exception_type": type(exc).__name__},
            message=str(exc),
        )
        run_report["status"] = "FAILED"
        run_report["campaign"] = fes_campaign
        run_report["failure"] = evaluate_trust(
            [failure],
            profile=RUN_FAILURE_PROFILE,
        ).to_dict()
        raise
    finally:
        _write_strict_json(output_dir / "spin_gap_run.json", run_report)
        print(f"wrote {output_dir / 'spin_gap_run.json'}")


if __name__ == "__main__":
    main()

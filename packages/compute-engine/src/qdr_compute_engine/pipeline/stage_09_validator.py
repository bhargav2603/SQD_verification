"""Stage 9 -- binding energy and error budget.

Computes the exact ground state of the *same* qubit Hamiltonian when the register is
small enough, which turns "how good is the ansatz?" from a guess into a measurement. That
comparison is bounded by the variational principle, so a result below it is proof of a
bug rather than a good day.
"""

from qdr_compute_engine.drivers.statevector.statevector_driver import (
    exact_ground_state_energy,
)
from qdr_compute_engine.pipeline.stage_context import StageContext
from qdr_compute_engine.types import ValidationReport
from qdr_compute_engine.validation.binding_energy import (
    binding_energy,
    hartree_to_kcal_per_mol,
)
from qdr_compute_engine.validation.error_budget import (
    build_error_budget,
    within_chemical_accuracy,
)
from qdr_compute_engine.validation.trust import (
    Evidence,
    QUANTITATIVE_CHEMISTRY_PROFILE,
    TRUST_PROFILES,
    evidence_from_dict,
    evaluate_trust,
)
from qdr_core.errors import ValidationError
from qdr_core.limits import MAX_EXACT_REFERENCE_QUBITS

STAGE_NAME = "09_validator"


def run(context: StageContext) -> StageContext:
    config = context.config
    result = context.require_solve_result()
    qubit_hamiltonian = context.require_qubit_hamiltonian()

    exact_reference: float | None = None
    if qubit_hamiltonian.n_qubits <= MAX_EXACT_REFERENCE_QUBITS:
        exact_reference = exact_ground_state_energy(qubit_hamiltonian)

    budget, warnings = build_error_budget(
        result,
        exact_ground_state=exact_reference,
        active_space_error=config.get("active_space_error"),
    )

    # Binding energy needs the two isolated-fragment references, which only a campaign
    # that has computed them can supply. Absent them, report the total energy and say so
    # rather than inventing a difference.
    e_protein = config.get("reference_energy_protein")
    e_ligand = config.get("reference_energy_ligand")
    if e_protein is not None and e_ligand is not None:
        delta_e = binding_energy(result.energy, float(e_protein), float(e_ligand))
    else:
        delta_e = result.energy
        warnings = (
            *warnings,
            "no isolated-fragment reference energies supplied; reporting the total "
            "energy rather than a binding energy",
        )

    raw_evidence = config.get("trust_evidence", ())
    if not isinstance(raw_evidence, (list, tuple)):
        raise ValidationError(
            "config['trust_evidence'] must be a list of evidence mappings",
            evidence_type=type(raw_evidence).__name__,
        )
    evidence: list[Evidence] = []
    for item in raw_evidence:
        if isinstance(item, Evidence):
            evidence.append(item)
        elif isinstance(item, dict):
            evidence.append(evidence_from_dict(item))
        else:
            raise ValidationError(
                "each trust evidence item must be an Evidence or mapping",
                evidence_type=type(item).__name__,
            )
    profile_name = str(
        config.get("trust_profile", QUANTITATIVE_CHEMISTRY_PROFILE.name)
    )
    profile = TRUST_PROFILES.get(profile_name)
    if profile is None:
        raise ValidationError(
            f"unknown trust profile {profile_name!r}",
            profile=profile_name,
            available_profiles=sorted(TRUST_PROFILES),
        )
    trust = evaluate_trust(evidence, profile=profile)

    report = ValidationReport(
        binding_energy_ha=delta_e,
        binding_energy_kcal_per_mol=hartree_to_kcal_per_mol(delta_e),
        error_budget=budget,
        within_chemical_accuracy=within_chemical_accuracy(budget),
        warnings=warnings,
        trust=trust,
    )

    return context.evolve(validation=report)

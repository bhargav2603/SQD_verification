"""Chemistry diagnostics that same-Hamiltonian solver checks cannot answer."""

import pytest

from qdr_compute_engine.validation.chemistry import (
    agreement_evidence,
    avas_selection_evidence,
    basis_convergence_evidence,
    basis_evidence,
    convergence_evidence,
    coupled_cluster_reference_evidence,
    embedding_validation_evidence,
    environment_evidence,
    geometry_evidence,
    nested_model_convergence_evidence,
    required_orbital_evidence,
    ScientificEstimate,
    spin_evidence,
)
from qdr_compute_engine.validation.trust import EvidenceStatus, FailureMode


@pytest.mark.parametrize("basis", ["sto-3g", "STO-6G", "minao", "ANO-RCC-MB"])
def test_minimal_basis_fails_a_transition_metal_quantitative_claim(basis: str) -> None:
    """Would fail if a converged minimal-basis Fe calculation were accepted as chemistry."""
    evidence = basis_evidence(basis, transition_metal=True, quantitative=True)

    assert evidence.status is EvidenceStatus.FAIL
    assert evidence.failure_mode is FailureMode.MINIMAL_BASIS


def test_nonminimal_basis_passes_representation_floor_but_not_convergence() -> None:
    evidence = basis_evidence("def2-tzvp", transition_metal=True, quantitative=True)

    assert evidence.status is EvidenceStatus.PASS
    assert evidence.metadata["scope"] == "representation_floor_only"


def test_missing_ligand_donor_shell_fails_required_orbital_check() -> None:
    """Would fail if Fe 3d alone were treated as a complete ligand-field active space."""
    evidence = required_orbital_evidence(
        selected_labels=("Fe 3d",),
        required_groups={"metal_valence": ("Fe 3d",), "ligand_donor": ("F 2p",)},
    )

    assert evidence.status is EvidenceStatus.FAIL
    assert evidence.failure_mode is FailureMode.MISSING_REQUIRED_ORBITALS
    assert evidence.metadata["missing_groups"] == ["ligand_donor"]


def test_required_orbital_groups_all_present_pass() -> None:
    evidence = required_orbital_evidence(
        selected_labels=("Fe 3d", "F 2p"),
        required_groups={"metal_valence": ("Fe 3d",), "ligand_donor": ("F 2p",)},
    )

    assert evidence.status is EvidenceStatus.PASS


def test_qubit_cap_forcing_a_higher_avas_threshold_is_not_silently_trusted() -> None:
    """Would fail if fitting the hardware budget were mistaken for chemical adequacy."""
    evidence = avas_selection_evidence(
        requested_threshold=0.2,
        used_threshold=0.85,
        max_active_orbitals=7,
    )

    assert evidence.status is EvidenceStatus.FAIL
    assert evidence.failure_mode is FailureMode.ACTIVE_SPACE_CAP_FORCED


def test_spin_contamination_is_measured_against_requested_multiplicity() -> None:
    evidence = spin_evidence(spin_squared=1.8, spin_multiplicity=1, tolerance=1.0e-2)

    assert evidence.status is EvidenceStatus.FAIL
    assert evidence.failure_mode is FailureMode.SPIN_CONTAMINATION
    assert evidence.value == pytest.approx(1.8)
    assert evidence.metadata["target_s_squared"] == pytest.approx(0.0)


def test_one_active_space_is_missing_convergence_evidence() -> None:
    evidence = convergence_evidence(
        code="active_space_ladder",
        values=[-15.2],
        tolerance=1.0,
        failure_mode=FailureMode.ACTIVE_SPACE_NOT_CONVERGED,
        units="kcal/mol",
    )

    assert evidence.status is EvidenceStatus.MISSING


def test_active_space_ladder_passes_only_when_last_change_is_small() -> None:
    converged = convergence_evidence(
        code="active_space_ladder",
        values=[-20.0, -15.4, -15.0],
        tolerance=0.5,
        failure_mode=FailureMode.ACTIVE_SPACE_NOT_CONVERGED,
        units="kcal/mol",
    )
    unconverged = convergence_evidence(
        code="active_space_ladder",
        values=[-20.0, -15.4, -13.0],
        tolerance=0.5,
        failure_mode=FailureMode.ACTIVE_SPACE_NOT_CONVERGED,
        units="kcal/mol",
    )

    assert converged.status is EvidenceStatus.PASS
    assert converged.value == pytest.approx(0.4)
    assert unconverged.status is EvidenceStatus.FAIL
    assert unconverged.value == pytest.approx(2.4)


def test_active_space_ladder_requires_strictly_growing_models() -> None:
    evidence = nested_model_convergence_evidence(
        code="active_space_ladder",
        model_sizes=[8, 8],
        values=[-15.0, -15.0],
        tolerance=0.5,
        failure_mode=FailureMode.ACTIVE_SPACE_NOT_CONVERGED,
        units="kcal/mol",
    )

    assert evidence.status is EvidenceStatus.MISSING
    assert evidence.metadata["model_sizes"] == [8, 8]


def test_active_space_ladder_uses_the_last_two_distinct_sizes() -> None:
    evidence = nested_model_convergence_evidence(
        code="active_space_ladder",
        model_sizes=[6, 8, 10],
        values=[-20.0, -15.4, -15.0],
        tolerance=0.5,
        failure_mode=FailureMode.ACTIVE_SPACE_NOT_CONVERGED,
        units="kcal/mol",
    )

    assert evidence.status is EvidenceStatus.PASS
    assert evidence.value == pytest.approx(0.4)


def test_basis_ladder_compares_the_observable_not_total_energies() -> None:
    evidence = basis_convergence_evidence(
        observable_values_kcal=[8.0, 6.2, 6.0],
        tolerance_kcal=0.5,
    )

    assert evidence.status is EvidenceStatus.PASS
    assert evidence.value == pytest.approx(0.2)


def test_two_methods_must_agree_within_the_declared_tolerance() -> None:
    evidence = agreement_evidence(
        {
            "nevpt2": ScientificEstimate(6.0, independence_group="multireference_pt"),
            "tailored_cc": ScientificEstimate(6.6, independence_group="coupled_cluster"),
        },
        tolerance=1.0,
        units="kcal/mol",
    )

    assert evidence.status is EvidenceStatus.PASS
    assert evidence.value == pytest.approx(0.6)


def test_one_method_is_not_independent_agreement() -> None:
    evidence = agreement_evidence(
        {
            "qdr_and_casci_same_hamiltonian": ScientificEstimate(
                6.0,
                independence_group="active_hamiltonian_abc",
            )
        },
        tolerance=1.0,
        units="kcal/mol",
    )

    assert evidence.status is EvidenceStatus.MISSING
    assert evidence.failure_mode is FailureMode.METHODS_DISAGREE


def test_same_hamiltonian_solvers_are_not_independent_methods() -> None:
    evidence = agreement_evidence(
        {
            "qdr": ScientificEstimate(6.0, independence_group="hamiltonian_abc"),
            "casci": ScientificEstimate(6.0, independence_group="hamiltonian_abc"),
        },
        tolerance=1.0,
        units="kcal/mol",
    )

    assert evidence.status is EvidenceStatus.MISSING


def test_illustrative_common_geometry_fails_an_adiabatic_gap_claim() -> None:
    evidence = geometry_evidence(geometry_kind="illustrative", observable_kind="adiabatic")

    assert evidence.status is EvidenceStatus.FAIL
    assert evidence.failure_mode is FailureMode.GEOMETRY_INADEQUATE


def test_common_geometry_is_valid_for_an_explicit_vertical_gap() -> None:
    evidence = geometry_evidence(geometry_kind="common", observable_kind="vertical")

    assert evidence.status is EvidenceStatus.PASS


def test_solution_phase_claim_without_environment_model_fails() -> None:
    evidence = environment_evidence(target_environment="solution", model=None)

    assert evidence.status is EvidenceStatus.FAIL
    assert evidence.failure_mode is FailureMode.ENVIRONMENT_MISMATCH


def test_large_open_shell_t1_diagnostic_rejects_ccsd_t_as_crosscheck() -> None:
    evidence = coupled_cluster_reference_evidence(
        converged=True,
        t1_diagnostic=0.08,
        open_shell=True,
    )

    assert evidence.status is EvidenceStatus.FAIL
    assert evidence.failure_mode is FailureMode.SINGLE_REFERENCE_BREAKDOWN


def test_embedding_needs_recovery_and_partition_stability() -> None:
    evidence = embedding_validation_evidence(
        recovery_error_hartree=2.0e-4,
        recovery_tolerance_hartree=1.0e-6,
        partition_jaccard=0.98,
        minimum_partition_jaccard=0.95,
    )

    assert evidence.status is EvidenceStatus.FAIL
    assert evidence.failure_mode is FailureMode.EMBEDDING_NOT_VALIDATED


def test_missing_embedding_diagnostics_is_undecidable_not_a_pass() -> None:
    evidence = embedding_validation_evidence(
        recovery_error_hartree=None,
        recovery_tolerance_hartree=1.0e-6,
        partition_jaccard=None,
        minimum_partition_jaccard=0.95,
    )

    assert evidence.status is EvidenceStatus.MISSING


def test_solution_model_does_not_satisfy_a_crystal_environment_claim() -> None:
    evidence = environment_evidence(target_environment="crystal", model="pcm-water")

    assert evidence.status is EvidenceStatus.FAIL

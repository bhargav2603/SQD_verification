"""Pure scientific diagnostics for active-space and spin-gap calculations.

These builders convert measured facts into :class:`Evidence`.  They do not combine
evidence or know the final verdict; that separation prevents a local diagnostic from
quietly promoting the whole calculation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from qdr_compute_engine.validation.trust import (
    Evidence,
    EvidenceStatus,
    FailureMode,
)
from qdr_core.errors import ValidationError


@dataclass(frozen=True, slots=True)
class ScientificEstimate:
    """An observable estimate with enough provenance to test independence."""

    value: float
    independence_group: str
    hamiltonian_hash: str | None = None
    scope: str = "chemistry"


def _positive_tolerance(value: float, name: str) -> float:
    tolerance = float(value)
    if tolerance <= 0.0:
        raise ValidationError(f"{name} must be positive, got {tolerance}", value=tolerance)
    return tolerance


def _is_minimal_basis(basis: str) -> bool:
    normalized = basis.strip().lower().replace("_", "-")
    return (
        normalized in {"minao", "ano-rcc-mb"}
        or re.fullmatch(r"sto-\d+g", normalized) is not None
    )


def basis_evidence(
    basis: str,
    *,
    transition_metal: bool,
    quantitative: bool,
) -> Evidence:
    """Check the representation floor, not basis-set convergence."""
    minimal = _is_minimal_basis(basis)
    failed = bool(transition_metal and quantitative and minimal)
    return Evidence(
        code="basis_representation",
        status=EvidenceStatus.FAIL if failed else EvidenceStatus.PASS,
        failure_mode=FailureMode.MINIMAL_BASIS,
        value=basis,
        message=(
            "a minimal basis cannot support a quantitative transition-metal claim"
            if failed
            else "basis clears the minimal representation floor; convergence is separate"
        ),
        metadata={
            "transition_metal": transition_metal,
            "quantitative": quantitative,
            "minimal": minimal,
            "scope": "representation_floor_only",
        },
    )


def required_orbital_evidence(
    *,
    selected_labels: Sequence[str],
    required_groups: Mapping[str, Sequence[str]],
) -> Evidence:
    """Require at least one declared AO label from every chemical role."""
    selected_normalized = {label.strip().casefold() for label in selected_labels}
    missing_groups = [
        group
        for group, alternatives in required_groups.items()
        if not any(label.strip().casefold() in selected_normalized for label in alternatives)
    ]
    failed = bool(missing_groups)
    return Evidence(
        code="required_orbitals",
        status=EvidenceStatus.FAIL if failed else EvidenceStatus.PASS,
        failure_mode=FailureMode.MISSING_REQUIRED_ORBITALS,
        value=list(selected_labels),
        message=(
            f"active-space targets omit required chemical roles: {', '.join(missing_groups)}"
            if failed
            else "every declared chemical orbital role is represented"
        ),
        metadata={
            "required_groups": {
                str(group): list(alternatives)
                for group, alternatives in sorted(required_groups.items())
            },
            "missing_groups": missing_groups,
        },
    )


def avas_selection_evidence(
    *,
    requested_threshold: float,
    used_threshold: float,
    max_active_orbitals: int | None,
) -> Evidence:
    """Expose hardware-forced AVAS shrinkage as a model failure."""
    forced = used_threshold > requested_threshold + 1.0e-12
    return Evidence(
        code="avas_cap",
        status=EvidenceStatus.FAIL if forced else EvidenceStatus.PASS,
        failure_mode=FailureMode.ACTIVE_SPACE_CAP_FORCED,
        value=used_threshold,
        threshold=requested_threshold,
        message=(
            "the qubit cap forced AVAS to raise its overlap cutoff and shrink the target space"
            if forced
            else "the requested AVAS threshold fit without cap-driven shrinkage"
        ),
        metadata={"max_active_orbitals": max_active_orbitals},
    )


def spin_evidence(
    *,
    spin_squared: float | None,
    spin_multiplicity: int,
    tolerance: float = 5.0e-2,
    code: str = "spin_purity",
) -> Evidence:
    """Compare measured ``<S^2>`` with the requested spin eigenvalue."""
    tolerance = _positive_tolerance(tolerance, "spin tolerance")
    spin = (spin_multiplicity - 1) / 2.0
    target = spin * (spin + 1.0)
    if spin_squared is None:
        return Evidence(
            code=code,
            status=EvidenceStatus.MISSING,
            failure_mode=FailureMode.SPIN_CONTAMINATION,
            threshold=tolerance,
            units="S^2",
            message="the solver did not report <S^2>",
            metadata={"target_s_squared": target, "multiplicity": spin_multiplicity},
        )
    deviation = abs(float(spin_squared) - target)
    return Evidence(
        code=code,
        status=EvidenceStatus.PASS if deviation <= tolerance else EvidenceStatus.FAIL,
        failure_mode=FailureMode.SPIN_CONTAMINATION,
        value=float(spin_squared),
        threshold=tolerance,
        units="S^2",
        message=f"|<S^2>-S(S+1)| = {deviation:.3e}",
        metadata={
            "target_s_squared": target,
            "deviation": deviation,
            "multiplicity": spin_multiplicity,
        },
    )


def convergence_evidence(
    *,
    code: str,
    values: Sequence[float],
    tolerance: float,
    failure_mode: FailureMode,
    units: str,
    required: bool = True,
) -> Evidence:
    """Check stabilization of the target observable over the final ladder step."""
    tolerance = _positive_tolerance(tolerance, f"{code} tolerance")
    measured = [float(value) for value in values]
    if len(measured) < 2:
        return Evidence(
            code=code,
            status=EvidenceStatus.MISSING,
            failure_mode=failure_mode,
            required=required,
            threshold=tolerance,
            units=units,
            message="at least two nested model sizes are required to assess convergence",
            metadata={"values": measured, "n_points": len(measured)},
        )
    final_change = abs(measured[-1] - measured[-2])
    return Evidence(
        code=code,
        status=EvidenceStatus.PASS if final_change <= tolerance else EvidenceStatus.FAIL,
        failure_mode=failure_mode,
        required=required,
        value=final_change,
        threshold=tolerance,
        units=units,
        message=f"final observable change is {final_change:.6g} {units}",
        metadata={"values": measured, "n_points": len(measured)},
    )


def nested_model_convergence_evidence(
    *,
    code: str,
    model_sizes: Sequence[int],
    values: Sequence[float],
    tolerance: float,
    failure_mode: FailureMode,
    units: str,
    required: bool = True,
) -> Evidence:
    """Require a genuinely growing model ladder before testing observable stability."""
    sizes = [int(size) for size in model_sizes]
    measured = [float(value) for value in values]
    valid_sizes = (
        len(sizes) == len(measured)
        and len(sizes) >= 2
        and all(right > left for left, right in zip(sizes, sizes[1:]))
    )
    if not valid_sizes:
        return Evidence(
            code=code,
            status=EvidenceStatus.MISSING,
            failure_mode=failure_mode,
            required=required,
            threshold=_positive_tolerance(tolerance, f"{code} tolerance"),
            units=units,
            message="convergence needs matched values from at least two strictly growing models",
            metadata={"model_sizes": sizes, "values": measured},
        )
    result = convergence_evidence(
        code=code,
        values=measured,
        tolerance=tolerance,
        failure_mode=failure_mode,
        units=units,
        required=required,
    )
    return Evidence(
        code=result.code,
        status=result.status,
        failure_mode=result.failure_mode,
        required=result.required,
        value=result.value,
        threshold=result.threshold,
        units=result.units,
        message=result.message,
        metadata={**result.metadata, "model_sizes": sizes},
    )


def basis_convergence_evidence(
    *,
    observable_values_kcal: Sequence[float],
    tolerance_kcal: float = 1.0,
) -> Evidence:
    return convergence_evidence(
        code="basis_convergence",
        values=observable_values_kcal,
        tolerance=tolerance_kcal,
        failure_mode=FailureMode.BASIS_NOT_CONVERGED,
        units="kcal/mol",
    )


def agreement_evidence(
    method_values: Mapping[str, ScientificEstimate],
    *,
    tolerance: float,
    units: str,
) -> Evidence:
    """Require two distinct scientific estimates and compare their full spread."""
    tolerance = _positive_tolerance(tolerance, "method agreement tolerance")
    invalid = [
        str(name)
        for name, estimate in method_values.items()
        if not isinstance(estimate, ScientificEstimate)
    ]
    estimates = {
        str(name): estimate
        for name, estimate in method_values.items()
        if isinstance(estimate, ScientificEstimate)
        and estimate.scope.strip().casefold() != "engine_only"
    }
    groups = {estimate.independence_group for estimate in estimates.values()}
    hashes = {
        estimate.hamiltonian_hash
        for estimate in estimates.values()
        if estimate.hamiltonian_hash is not None
    }
    same_declared_hamiltonian = (
        len(estimates) >= 2
        and len(hashes) == 1
        and all(estimate.hamiltonian_hash is not None for estimate in estimates.values())
    )
    if len(estimates) < 2 or len(groups) < 2 or same_declared_hamiltonian or invalid:
        return Evidence(
            code="independent_method_agreement",
            status=EvidenceStatus.MISSING,
            failure_mode=FailureMode.METHODS_DISAGREE,
            threshold=tolerance,
            units=units,
            message="fewer than two provenance-distinct chemistry estimates were supplied",
            metadata={
                "methods": {
                    name: {
                        "value": estimate.value,
                        "independence_group": estimate.independence_group,
                        "hamiltonian_hash": estimate.hamiltonian_hash,
                        "scope": estimate.scope,
                    }
                    for name, estimate in estimates.items()
                },
                "invalid_or_unproven_methods": invalid,
                "independence_groups": sorted(groups),
                "same_declared_hamiltonian": same_declared_hamiltonian,
            },
        )
    values = {name: float(estimate.value) for name, estimate in estimates.items()}
    spread = max(values.values()) - min(values.values())
    return Evidence(
        code="independent_method_agreement",
        status=EvidenceStatus.PASS if spread <= tolerance else EvidenceStatus.FAIL,
        failure_mode=FailureMode.METHODS_DISAGREE,
        value=spread,
        threshold=tolerance,
        units=units,
        message=f"method spread is {spread:.6g} {units}",
        metadata={"methods": values},
    )


def coupled_cluster_reference_evidence(
    *,
    converged: bool,
    t1_diagnostic: float | None,
    open_shell: bool,
) -> Evidence:
    """Qualify CCSD(T) before treating it as an independent chemistry estimate."""
    threshold = 0.044 if open_shell else 0.02
    if not converged:
        return Evidence(
            code="coupled_cluster_reference",
            status=EvidenceStatus.FAIL,
            failure_mode=FailureMode.COUPLED_CLUSTER_NOT_CONVERGED,
            value=t1_diagnostic,
            threshold=threshold,
            message="CCSD amplitudes did not converge; CCSD(T) cannot be used as a cross-check",
            metadata={"open_shell": open_shell},
        )
    if t1_diagnostic is None:
        return Evidence(
            code="coupled_cluster_reference",
            status=EvidenceStatus.MISSING,
            failure_mode=FailureMode.SINGLE_REFERENCE_BREAKDOWN,
            threshold=threshold,
            message="CCSD converged but no T1 diagnostic was reported",
            metadata={"open_shell": open_shell},
        )
    failed = float(t1_diagnostic) > threshold
    return Evidence(
        code="coupled_cluster_reference",
        status=EvidenceStatus.FAIL if failed else EvidenceStatus.PASS,
        failure_mode=FailureMode.SINGLE_REFERENCE_BREAKDOWN,
        value=float(t1_diagnostic),
        threshold=threshold,
        message=(
            "T1 diagnostic is too large to use single-reference CCSD(T) as validation"
            if failed
            else "CCSD(T) reference clears the T1 diagnostic screen"
        ),
        metadata={"open_shell": open_shell},
    )


def embedding_validation_evidence(
    *,
    recovery_error_hartree: float | None,
    recovery_tolerance_hartree: float,
    partition_jaccard: float | None,
    minimum_partition_jaccard: float,
) -> Evidence:
    """Require both a recovery limit and stability to a partition perturbation."""
    recovery_tolerance = _positive_tolerance(
        recovery_tolerance_hartree,
        "embedding recovery tolerance",
    )
    minimum_jaccard = float(minimum_partition_jaccard)
    if not 0.0 <= minimum_jaccard <= 1.0:
        raise ValidationError(
            "minimum partition Jaccard must lie in [0, 1]",
            value=minimum_jaccard,
        )
    if recovery_error_hartree is None or partition_jaccard is None:
        return Evidence(
            code="embedding_validation",
            status=EvidenceStatus.MISSING,
            failure_mode=FailureMode.EMBEDDING_NOT_VALIDATED,
            threshold={
                "recovery_error_hartree": recovery_tolerance,
                "partition_jaccard": minimum_jaccard,
            },
            message="embedding recovery and partition-perturbation diagnostics are both required",
        )
    recovery_error = abs(float(recovery_error_hartree))
    jaccard = float(partition_jaccard)
    recovery_failed = recovery_error > recovery_tolerance
    partition_failed = jaccard < minimum_jaccard
    return Evidence(
        code="embedding_validation",
        status=(
            EvidenceStatus.FAIL
            if recovery_failed or partition_failed
            else EvidenceStatus.PASS
        ),
        failure_mode=(
            FailureMode.EMBEDDING_NOT_VALIDATED
            if recovery_failed
            else FailureMode.EMBEDDING_PARTITION_UNSTABLE
        ),
        value={
            "recovery_error_hartree": recovery_error,
            "partition_jaccard": jaccard,
        },
        threshold={
            "recovery_error_hartree": recovery_tolerance,
            "partition_jaccard": minimum_jaccard,
        },
        message=(
            "embedding recovery or partition stability failed"
            if recovery_failed or partition_failed
            else "embedding passes the recovery and partition-stability gates"
        ),
    )


def geometry_evidence(*, geometry_kind: str, observable_kind: str) -> Evidence:
    """Validate geometry provenance against vertical or adiabatic gap semantics."""
    geometry = geometry_kind.strip().casefold()
    observable = observable_kind.strip().casefold()
    valid = (
        geometry in {"optimized_each_state", "experimental"}
        if observable == "adiabatic"
        else geometry in {"common", "optimized_each_state", "experimental"}
    )
    return Evidence(
        code="geometry_model",
        status=EvidenceStatus.PASS if valid else EvidenceStatus.FAIL,
        failure_mode=FailureMode.GEOMETRY_INADEQUATE,
        value=geometry_kind,
        message=(
            f"{geometry_kind} geometry is adequate for an explicit {observable_kind} gap"
            if valid
            else f"{geometry_kind} geometry cannot support the claimed {observable_kind} gap"
        ),
        metadata={"observable_kind": observable_kind},
    )


def environment_evidence(*, target_environment: str, model: str | None) -> Evidence:
    """Reject a mismatch between the claimed and computed environment."""
    target = target_environment.strip().casefold()
    model_normalized = None if model is None else model.strip().casefold()
    if target in {"gas", "gas_phase", "vacuum"}:
        valid = model_normalized in {None, "", "gas", "gas_phase", "vacuum"}
    elif target in {"solution", "solvent"}:
        valid = model_normalized is not None and any(
            token in model_normalized
            for token in ("solution", "solvent", "pcm", "smd", "cosmo")
        )
    elif target in {"crystal", "solid", "periodic"}:
        valid = model_normalized is not None and any(
            token in model_normalized
            for token in ("crystal", "solid", "periodic", "pbc")
        )
    elif target in {"protein", "enzyme"}:
        valid = model_normalized is not None and any(
            token in model_normalized
            for token in ("protein", "enzyme", "qmmm", "qm/mm")
        )
    else:
        valid = model_normalized == target
    return Evidence(
        code="environment_model",
        status=EvidenceStatus.PASS if valid else EvidenceStatus.FAIL,
        failure_mode=FailureMode.ENVIRONMENT_MISMATCH,
        value=model,
        message=(
            "environment model matches the target observable"
            if valid
            else f"target environment {target_environment!r} is inconsistent with model {model!r}"
        ),
        metadata={"target_environment": target_environment},
    )

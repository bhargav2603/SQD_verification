"""Drive the fragment expansion: one subsystem solve per monomer and per dimer.

This is what makes stage 2's fragmentation load-bearing. Previously the fragmenter ran,
its result was recorded, and stages 3-9 then computed the supersystem regardless -- the
pipeline said so in a warning rather than pretending otherwise. This module closes that
gap.

Design
------
Each subsystem is solved by running ``SUBSYSTEM_STAGES`` (integrals -> active space ->
Hamiltonian -> mapping -> ansatz -> solve) on that fragment's own geometry. The fragment
path therefore executes *the same stage functions* as the supersystem path, which is the
property that matters: a discrepancy between a one-fragment FMO run and a plain
supersystem run would be a bug in this module, not a difference in physics.

Failure ordering
----------------
Every subsystem is planned, charge-checked and cap-checked **before** the first SCF runs.
An oversized fragment or a charge assignment that does not sum correctly is a mistake in
the input, and finding it after forty minutes of dimer calculations helps nobody.

What this does not do
---------------------
No electrostatic embedding. See ``fragmentation/fmo_assembly.py`` -- the scheme is
recorded on every result and the caveat is attached as a warning, so a number from here
cannot be mistaken for a fully embedded FMO energy.
"""

import logging
import time
from typing import Any

from qdr_compute_engine.fragmentation.fmo_assembly import (
    FMOResult,
    FragmentEnergy,
    SubsystemPlan,
    assemble,
    build_plans,
    resolve_fragment_charges,
    scheme_warnings,
)
from qdr_compute_engine.geometry.pose_builder import count_electrons, require_closed_shell
from qdr_compute_engine.pipeline import stage_01_geometry_pose, stage_02_fmo_fragmenter
from qdr_compute_engine.pipeline.pipeline import (
    SUBSYSTEM_STAGES,
    build_context,
    run_pipeline,
)
from qdr_compute_engine.pipeline.stage_context import StageContext
from qdr_core.errors import ValidationError
from qdr_core.ids import new_run_id
from qdr_core.limits import MAX_QUBITS, qubits_for_orbitals

logger = logging.getLogger(__name__)

#: Config keys that describe the *parent* system and must not reach a subsystem run.
#: ``charge`` and ``spin_multiplicity`` are the dangerous ones: a fragment carries its own,
#: and inheriting the parent's would silently change its electron count.
_PARENT_ONLY_CONFIG_KEYS = frozenset(
    {
        "charge",
        "spin_multiplicity",
        "xyz",
        "pdb",
        "fragmentation_scheme",
        "cut_bonds",
        "dimer_cutoff_angstrom",
        "fragment_charges",
        "fmo",
    }
)


def subsystem_config(parent_config: dict[str, Any]) -> dict[str, Any]:
    """Strip parent-system keys so a fragment cannot inherit the wrong charge."""
    return {k: v for k, v in parent_config.items() if k not in _PARENT_ONLY_CONFIG_KEYS}


def _estimate_orbitals(plan: SubsystemPlan, basis: str) -> int | None:
    """Minimal-basis orbital count, for a pre-flight cap check.

    Only exact for STO-3G, which is why an unknown basis returns ``None`` and the check
    is skipped rather than guessed -- a wrong estimate that rejects a valid fragment is
    worse than no estimate at all. The engine's real cap enforcement still runs inside
    stage 4 for every fragment.
    """
    if not basis.lower().replace("_", "-").startswith("sto-3g"):
        return None

    # STO-3G: 1 orbital per H/He, 5 for Li-Ne, 9 for Na-Ar.
    per_element = 0
    for atom in plan.geometry.atoms:
        from qdr_compute_engine.geometry.pose_builder import atomic_number

        z = atomic_number(atom.symbol)
        if z <= 2:
            per_element += 1
        elif z <= 10:
            per_element += 5
        elif z <= 18:
            per_element += 9
        else:
            return None
    return per_element


def preflight(plans: list[SubsystemPlan], basis: str) -> None:
    """Reject impossible subsystems before any expensive work begins."""
    for plan in plans:
        # Closed shell, per fragment. A covalent cut leaves a radical, and this is where
        # that shows up as an explicit refusal rather than a wrong reference state.
        try:
            require_closed_shell(plan.geometry)
        except ValidationError as exc:
            raise ValidationError(
                f"fragment {plan.label!r} ({'+'.join(plan.fragment_ids)}) is not "
                f"closed-shell: {exc.message} If this fragment came from a covalent cut, "
                "the dangling valence needs capping -- which is not implemented.",
                fragment_ids=list(plan.fragment_ids),
                fragment_label=plan.label,
                **exc.details,
            ) from exc

        n_orbitals = _estimate_orbitals(plan, basis)
        if n_orbitals is not None and qubits_for_orbitals(n_orbitals) > MAX_QUBITS:
            raise ValidationError(
                f"fragment {plan.label!r} needs about "
                f"{qubits_for_orbitals(n_orbitals)} qubits in {basis}, over the "
                f"{MAX_QUBITS}-qubit cap; fragment more finely or reduce the active space",
                fragment_ids=list(plan.fragment_ids),
                estimated_qubits=qubits_for_orbitals(n_orbitals),
                max_qubits=MAX_QUBITS,
            )


def solve_subsystem(
    plan: SubsystemPlan,
    config: dict[str, Any],
    *,
    seed: int = 0,
) -> FragmentEnergy:
    """Run stages 3-8 on one fragment and return its energy."""
    started = time.perf_counter()

    context = StageContext(
        run_id=new_run_id(),
        config=config,
        seed=seed,
        geometry=plan.geometry,
    )
    context = run_pipeline(context, SUBSYSTEM_STAGES)

    result = context.require_solve_result()
    hamiltonian = context.require_qubit_hamiltonian()
    ansatz = context.ansatz

    return FragmentEnergy(
        fragment_ids=plan.fragment_ids,
        energy=result.energy,
        n_atoms=plan.geometry.n_atoms,
        n_electrons=count_electrons(plan.geometry),
        charge=plan.charge,
        n_qubits=hamiltonian.n_qubits,
        n_parameters=ansatz.n_params if ansatz is not None else 0,
        seconds=time.perf_counter() - started,
        converged=result.converged,
        label=plan.label,
    )


def run_fmo(context: StageContext) -> tuple[StageContext, FMOResult]:
    """Run the full two-body expansion for the fragmented system on ``context``.

    Expects stages 1 and 2 to have run, so the context carries a validated geometry and
    a fragmentation. Returns the updated context (timings and warnings accumulated) and
    the assembled result.
    """
    parent_geometry = context.require_geometry()
    fragmentation = context.require_fragmentation()
    config = context.config
    basis = str(config.get("basis", "sto-3g"))

    charges = resolve_fragment_charges(
        fragmentation,
        total_charge=parent_geometry.charge,
        requested=config.get("fragment_charges"),
    )

    monomer_plans, dimer_plans = build_plans(parent_geometry, fragmentation, charges)
    all_plans = [*monomer_plans, *dimer_plans]

    preflight(all_plans, basis)

    child_config = subsystem_config(config)
    truncated = (
        child_config.get("n_active_orbitals") is not None
        or child_config.get("n_active_electrons") is not None
    )

    logger.info(
        "FMO2: %d monomers, %d dimers (run %s)",
        len(monomer_plans),
        len(dimer_plans),
        context.run_id,
    )

    fragments: list[FragmentEnergy] = []
    for index, plan in enumerate(all_plans, start=1):
        label = "+".join(plan.fragment_ids)
        logger.info("FMO2 subsystem %d/%d: %s", index, len(all_plans), label)
        try:
            fragments.append(solve_subsystem(plan, child_config, seed=context.seed))
        except Exception as exc:
            # A partial expansion is not a smaller answer, it is a wrong one: dropping a
            # monomer removes its energy from every pair term that references it.
            raise ValidationError(
                f"FMO subsystem {label} ({plan.label!r}) failed: {exc}. The expansion "
                "cannot be assembled from a subset of its terms.",
                fragment_ids=list(plan.fragment_ids),
                fragment_label=plan.label,
            ) from exc

    result = assemble(
        tuple(fragments),
        requested_pairs=fragmentation.dimer_pairs,
        warnings=scheme_warnings(
            fragmentation,
            active_space_truncated=truncated,
            cutoff_applied=config.get("dimer_cutoff_angstrom") is not None,
        ),
    )

    for message in result.warnings:
        context = context.with_warning(f"FMO: {message}")
    for fragment in fragments:
        if not fragment.converged:
            context = context.with_warning(
                f"FMO: subsystem {fragment.key} did not report convergence"
            )

    total_seconds = sum(f.seconds for f in fragments)
    context = context.with_timing("fmo_subsystems", total_seconds * 1000.0)

    logger.info(
        "FMO2 total: %.10f Ha from %d subsystems in %.1f s",
        result.total_energy,
        len(fragments),
        total_seconds,
    )

    return context, result


#: Stages 1 and 2: prepare the parent geometry and fragment it. Everything after that is
#: per-subsystem, so the parent chain stops here.
PARENT_STAGES = (
    (stage_01_geometry_pose.STAGE_NAME, stage_01_geometry_pose.run),
    (stage_02_fmo_fragmenter.STAGE_NAME, stage_02_fmo_fragmenter.run),
)


def run(
    config: dict[str, Any],
    *,
    seed: int = 0,
    **initial_fields: Any,
) -> tuple[StageContext, FMOResult]:
    """Full FMO2 run from a config: prepare, fragment, solve every subsystem, assemble.

    The counterpart to ``pipeline.run`` for fragmented systems. Both accept the same
    config; this one additionally honours ``fragment_charges``, ``cut_bonds`` and
    ``dimer_cutoff_angstrom``.

        context, result = fmo_runner.run(
            {"basis": "sto-3g", "fragmentation_scheme": "molecular"},
            geometry=water_dimer,
        )
        result.binding_energy()   # for a two-fragment system

    A single-fragment system is not an error: the expansion degenerates to one monomer
    and reproduces the supersystem energy exactly, which is the cheapest available check
    that the fragment path and the supersystem path agree.
    """
    context = build_context(config, seed=seed, **initial_fields)
    context = run_pipeline(context, PARENT_STAGES)
    return run_fmo(context)

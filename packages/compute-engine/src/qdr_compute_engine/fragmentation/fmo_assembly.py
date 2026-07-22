"""Fragment bookkeeping for the many-body expansion, and the assembly itself.

What this implements, precisely
-------------------------------
The two-body expansion of the total energy over fragments:

    E = sum_I E_I + sum_{I<J} (E_IJ - E_I - E_J)

Each ``E_I`` and ``E_IJ`` is a full, independent electronic-structure calculation on that
fragment's own geometry, in its own basis, solved by whichever backend the run selects.

**This is FMO2 without the electrostatic embedding**, which is the same thing as the
two-body many-body expansion (MBE2). Genuine FMO iterates each monomer's SCF in the
electrostatic potential of every other fragment, and that embedding is most of why FMO
works well for polar and charged systems. It is *not* implemented here. The distinction is
material -- for hydrogen-bonded systems the embedding contributes on the same order as the
pair interaction itself -- so the assembly reports which scheme produced a number rather
than letting a reader assume the stronger one.

Two consequences follow, and both are surfaced as warnings rather than buried:

  * a pair skipped by the distance cutoff has its interaction **dropped entirely**. Real
    FMO would approximate it electrostatically; without ESP there is nothing to fall back
    on, so the cutoff is a truncation, not an approximation.
  * truncating the active space per fragment breaks error cancellation in the pair term,
    because a dimer's orbital window is not the union of its monomers' windows. The full
    space is therefore the default, and truncation warns.

Fragment charges
----------------
A fragment's charge is a property of the fragmentation scheme, not something derivable
from the geometry -- ``Geometry.subset`` says as much and refuses to guess. Charges are
therefore taken from configuration, defaulting to neutral, and validated to sum to the
parent's total charge. That sum rule is the one consistency check available without
chemical perception, and getting it wrong silently changes the electron count of the whole
calculation.
"""

from dataclasses import dataclass

from qdr_compute_engine.types import Fragment, FragmentationResult, Geometry
from qdr_core.errors import ValidationError

#: Name of the scheme this module implements, recorded in results so a reader is never
#: left to assume electrostatic embedding was involved.
SCHEME = "fmo2-no-esp"


@dataclass(frozen=True)
class FragmentEnergy:
    """One completed subsystem calculation."""

    #: ``("F0",)`` for a monomer, ``("F0", "F1")`` for a dimer.
    fragment_ids: tuple[str, ...]
    energy: float
    n_atoms: int
    n_electrons: int
    charge: int
    n_qubits: int
    n_parameters: int
    seconds: float
    converged: bool
    label: str = ""

    @property
    def is_monomer(self) -> bool:
        return len(self.fragment_ids) == 1

    @property
    def key(self) -> str:
        return "+".join(self.fragment_ids)


@dataclass(frozen=True)
class FMOResult:
    """The assembled expansion, with every term kept separately.

    The pair interactions are retained rather than only their sum: for a two-fragment
    system the single pair interaction *is* the binding energy, and for larger systems
    the distribution of pair terms is what shows whether the two-body truncation is
    defensible.
    """

    total_energy: float
    monomer_energies: dict[str, float]
    dimer_energies: dict[tuple[str, str], float]
    interaction_energies: dict[tuple[str, str], float]
    fragments: tuple[FragmentEnergy, ...]
    scheme: str = SCHEME
    skipped_pairs: tuple[tuple[str, str], ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def n_monomers(self) -> int:
        return len(self.monomer_energies)

    @property
    def sum_of_monomers(self) -> float:
        return sum(self.monomer_energies.values())

    @property
    def total_interaction(self) -> float:
        """Everything the pair expansion adds on top of isolated fragments."""
        return sum(self.interaction_energies.values())

    def binding_energy(self) -> float:
        """The interaction energy for a two-fragment system.

        Defined only for exactly two monomers, where ``E_AB - E_A - E_B`` is unambiguous.
        With three or more fragments "binding" needs a stated partition of which
        fragments form each side, so this refuses rather than picking one.
        """
        if len(self.monomer_energies) != 2:
            raise ValidationError(
                "binding energy is only well defined for two fragments; this system has "
                f"{len(self.monomer_energies)}. Use interaction_energies, or define the "
                "two sides explicitly.",
                n_monomers=len(self.monomer_energies),
            )
        if len(self.interaction_energies) != 1:
            raise ValidationError(
                "the two fragments have no computed pair interaction; the distance "
                "cutoff excluded it, so no binding energy was calculated",
                skipped_pairs=[list(p) for p in self.skipped_pairs],
            )
        return next(iter(self.interaction_energies.values()))


def resolve_fragment_charges(
    fragmentation: FragmentationResult,
    total_charge: int,
    requested: dict[str, int] | None = None,
) -> dict[str, int]:
    """Assign a charge to every monomer and check the assignment adds up.

    Unlisted fragments default to neutral. The sum rule is enforced because a fragment
    charge that does not add up changes the electron count of the whole calculation, and
    nothing downstream would notice.
    """
    charges = {fragment.fragment_id: 0 for fragment in fragmentation.monomers}

    if requested:
        known = set(charges)
        unknown = sorted(set(requested) - known)
        if unknown:
            raise ValidationError(
                f"fragment_charges names unknown fragments: {unknown}",
                unknown_fragments=unknown,
                known_fragments=sorted(known),
            )
        charges.update({key: int(value) for key, value in requested.items()})

    assigned = sum(charges.values())
    if assigned != total_charge:
        raise ValidationError(
            f"fragment charges sum to {assigned} but the system carries {total_charge}; "
            "supply 'fragment_charges' so every fragment's charge is stated explicitly",
            assigned_total=assigned,
            system_charge=total_charge,
            fragment_charges=dict(charges),
        )
    return charges


def monomer_geometry(parent: Geometry, fragment: Fragment, charge: int) -> Geometry:
    """The sub-geometry for one fragment, at the parent's coordinates.

    Coordinates are taken unchanged: the fragment must sit where it sits in the
    supersystem, or the pair interactions describe a different arrangement of matter.
    """
    geometry = parent.subset(fragment.atom_indices)
    return Geometry(
        atoms=geometry.atoms,
        charge=charge,
        spin_multiplicity=1,
        units=geometry.units,
    )


def dimer_geometry(
    parent: Geometry,
    left: Fragment,
    right: Fragment,
    charge_left: int,
    charge_right: int,
) -> Geometry:
    """The sub-geometry for a fragment pair.

    Atom indices are sorted so the dimer's atom ordering is a deterministic function of
    the parent's, which keeps the input hash stable across runs.
    """
    overlap = set(left.atom_indices) & set(right.atom_indices)
    if overlap:
        raise ValidationError(
            f"fragments {left.fragment_id} and {right.fragment_id} share atoms "
            f"{sorted(overlap)}; the expansion requires disjoint fragments",
            shared_atoms=sorted(overlap),
        )

    combined = tuple(sorted((*left.atom_indices, *right.atom_indices)))
    geometry = parent.subset(combined)
    return Geometry(
        atoms=geometry.atoms,
        charge=charge_left + charge_right,
        spin_multiplicity=1,
        units=geometry.units,
    )


def validate_fragmentation(fragmentation: FragmentationResult, parent: Geometry) -> None:
    """Every atom must belong to exactly one fragment.

    A partition that drops an atom silently computes a different molecule; one that
    double-counts an atom double-counts its electrons. Neither shows up in the energy as
    anything other than a plausible wrong number.
    """
    if not fragmentation.monomers:
        raise ValidationError("fragmentation produced no monomers")

    seen: dict[int, str] = {}
    for fragment in fragmentation.monomers:
        for index in fragment.atom_indices:
            if index in seen:
                raise ValidationError(
                    f"atom {index} appears in both {seen[index]} and "
                    f"{fragment.fragment_id}; fragments must be disjoint",
                    atom_index=index,
                )
            seen[index] = fragment.fragment_id

    missing = sorted(set(range(parent.n_atoms)) - set(seen))
    if missing:
        raise ValidationError(
            f"atoms {missing} belong to no fragment; the partition is incomplete",
            unassigned_atoms=missing,
        )


def assemble(
    fragments: tuple[FragmentEnergy, ...],
    *,
    requested_pairs: tuple[tuple[str, str], ...],
    warnings: tuple[str, ...] = (),
) -> FMOResult:
    """Combine completed subsystem energies into the two-body expansion.

    Delegates the total to ``validation.binding_energy.total_fmo2_energy`` -- that
    function owns the expansion formula and is tested independently, so the arithmetic
    lives in exactly one place.
    """
    from qdr_compute_engine.validation.binding_energy import (
        interaction_energy,
        total_fmo2_energy,
    )

    monomer_energies = {
        entry.fragment_ids[0]: entry.energy for entry in fragments if entry.is_monomer
    }
    dimer_energies = {
        (entry.fragment_ids[0], entry.fragment_ids[1]): entry.energy
        for entry in fragments
        if not entry.is_monomer
    }

    if not monomer_energies:
        raise ValidationError("assembly requires at least one monomer energy")

    interactions = {
        pair: interaction_energy(energy, monomer_energies[pair[0]], monomer_energies[pair[1]])
        for pair, energy in dimer_energies.items()
    }

    computed = set(dimer_energies)
    skipped = tuple(pair for pair in requested_pairs if pair not in computed)

    return FMOResult(
        total_energy=total_fmo2_energy(monomer_energies, dimer_energies),
        monomer_energies=monomer_energies,
        dimer_energies=dimer_energies,
        interaction_energies=interactions,
        fragments=fragments,
        skipped_pairs=skipped,
        warnings=warnings,
    )


@dataclass
class SubsystemPlan:
    """Everything needed to run one subsystem, resolved before any of them start.

    Building the whole plan up front means a bad charge assignment or an oversized
    fragment fails before the first expensive SCF, rather than an hour into a sweep.
    """

    fragment_ids: tuple[str, ...]
    geometry: Geometry
    label: str
    charge: int


def build_plans(
    parent: Geometry,
    fragmentation: FragmentationResult,
    charges: dict[str, int],
) -> tuple[list[SubsystemPlan], list[SubsystemPlan]]:
    """Resolve every monomer and dimer subsystem. Returns ``(monomers, dimers)``."""
    validate_fragmentation(fragmentation, parent)

    monomer_plans = [
        SubsystemPlan(
            fragment_ids=(fragment.fragment_id,),
            geometry=monomer_geometry(parent, fragment, charges[fragment.fragment_id]),
            label=fragment.label or fragment.fragment_id,
            charge=charges[fragment.fragment_id],
        )
        for fragment in fragmentation.monomers
    ]

    dimer_plans = []
    for left_id, right_id in fragmentation.dimer_pairs:
        left = fragmentation.by_id(left_id)
        right = fragmentation.by_id(right_id)
        dimer_plans.append(
            SubsystemPlan(
                fragment_ids=(left_id, right_id),
                geometry=dimer_geometry(parent, left, right, charges[left_id], charges[right_id]),
                label=f"{left.label or left_id}+{right.label or right_id}",
                charge=charges[left_id] + charges[right_id],
            )
        )

    return monomer_plans, dimer_plans


def scheme_warnings(
    fragmentation: FragmentationResult,
    *,
    active_space_truncated: bool,
    cutoff_applied: bool,
) -> tuple[str, ...]:
    """The caveats that apply to any number this module produces."""
    messages: list[str] = [
        "energies come from the two-body expansion without electrostatic embedding "
        "(FMO2 without ESP, equivalently MBE2); genuine FMO polarises each fragment in "
        "the field of the others, which is not implemented"
    ]

    n_pairs_possible = len(fragmentation.monomers) * (len(fragmentation.monomers) - 1) // 2
    if cutoff_applied and len(fragmentation.dimer_pairs) < n_pairs_possible:
        messages.append(
            f"{n_pairs_possible - len(fragmentation.dimer_pairs)} of {n_pairs_possible} "
            "fragment pairs were excluded by the distance cutoff; without electrostatic "
            "embedding their interaction is dropped entirely rather than approximated"
        )

    if len(fragmentation.monomers) > 2:
        messages.append(
            "three-body and higher terms are not computed; the two-body expansion is "
            "exact only when they vanish"
        )

    if active_space_truncated:
        messages.append(
            "the active space is truncated, and a dimer's orbital window is not the "
            "union of its monomers' windows -- so truncation error does not cancel in "
            "the pair interaction and the interaction energies carry it"
        )

    return tuple(messages)

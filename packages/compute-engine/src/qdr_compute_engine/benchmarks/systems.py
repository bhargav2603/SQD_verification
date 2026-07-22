"""Benchmark systems for engine validation.

Scope
-----
These are the small, closed-shell molecules whose exact answer is computable, so that
"is the quantum path correct?" is a measurement rather than an opinion. They are *not*
chemistry results in themselves -- reproducing FCI for LiH is a precondition for
trusting the engine, not a finding.

On reference values
-------------------
No literature energy is hard-coded here. Every reference is **computed** at run time by
PySCF's FCI solver in the same basis and the same active space as the VQE run. Two
reasons, and both matter for a result anyone is expected to believe:

  * a copied number cannot be checked, and drifts silently when a geometry or basis
    changes;
  * PySCF's FCI is a genuinely independent implementation of the same physics, so the
    comparison is a cross-check between two derivations rather than a tautology.

Geometries below are standard equilibrium structures, recorded with their provenance.
Comparability to a published table depends on matching that paper's geometry and basis
exactly -- but the *internal* claim (VQE reproduces FCI in the same space) is independent
of which geometry is chosen, because both sides use the same one.

On active spaces
----------------
Each system lists more than one active space on purpose. Comparing them is how the
active-space truncation error is measured rather than assumed -- see
``validation/error_budget.py``, which records that term as ``None`` when nothing has
quantified it.
"""

from dataclasses import dataclass, field

from qdr_compute_engine.types import Atom, Geometry, LengthUnit


@dataclass(frozen=True)
class ActiveSpaceSpec:
    """One active space to run for a system.

    ``None`` for either field means "use the full space", which is exact but only
    tractable for the smallest systems.
    """

    n_active_electrons: int | None = None
    n_active_orbitals: int | None = None
    label: str = "full"

    @property
    def is_full_space(self) -> bool:
        return self.n_active_electrons is None and self.n_active_orbitals is None


@dataclass(frozen=True)
class BenchmarkSystem:
    """A molecule, a basis, and the active spaces to sweep."""

    name: str
    geometry: Geometry
    basis: str
    active_spaces: tuple[ActiveSpaceSpec, ...]
    #: Where the geometry came from. Recorded so a reader can reproduce or replace it.
    geometry_source: str
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


def _geometry(
    atoms: list[tuple[str, tuple[float, float, float]]],
    charge: int = 0,
    multiplicity: int = 1,
) -> Geometry:
    return Geometry(
        atoms=tuple(Atom(symbol, position) for symbol, position in atoms),
        charge=charge,
        spin_multiplicity=multiplicity,
        units=LengthUnit.ANGSTROM,
    )


# --------------------------------------------------------------------------------------
# H2 -- the correctness gate, and the dissociation curve
# --------------------------------------------------------------------------------------


def h2_at(bond_length_angstrom: float) -> BenchmarkSystem:
    """H2 at an arbitrary separation.

    The dissociation curve is the single most informative validation this engine can run.
    Near equilibrium almost any method looks correct; past ~1.5 A the restricted
    Hartree-Fock reference degrades badly while FCI does not, so the curve separates
    "reproduces the reference" from "reproduces the physics". UCCSD in a minimal basis has
    exactly one double excitation, which makes the first-order Trotter factorisation exact
    -- so any deviation from FCI along this curve is an implementation error, not a method
    error.
    """
    return BenchmarkSystem(
        name=f"H2@{bond_length_angstrom:.3f}",
        geometry=_geometry([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, bond_length_angstrom))]),
        basis="sto-3g",
        active_spaces=(ActiveSpaceSpec(label="full"),),
        geometry_source="scan parameter",
        notes="minimal-basis H2; UCCSD is exact here, so deviation from FCI is a bug",
        tags=("dissociation", "gate"),
    )


#: Bond lengths for the dissociation scan, in Angstrom. Dense near equilibrium where the
#: curvature is, sparser in the dissociated tail where it flattens.
H2_SCAN_BOND_LENGTHS: tuple[float, ...] = (
    0.50,
    0.60,
    0.70,
    0.735,
    0.80,
    0.90,
    1.00,
    1.10,
    1.20,
    1.35,
    1.50,
    1.70,
    1.90,
    2.10,
    2.30,
    2.50,
    2.75,
    3.00,
)


def h2_dissociation_curve() -> tuple[BenchmarkSystem, ...]:
    """The full 0.5 - 3.0 A scan."""
    return tuple(h2_at(r) for r in H2_SCAN_BOND_LENGTHS)


# --------------------------------------------------------------------------------------
# The standard small-molecule benchmarks
# --------------------------------------------------------------------------------------

LIH = BenchmarkSystem(
    name="LiH",
    geometry=_geometry([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.5949))]),
    basis="sto-3g",
    active_spaces=(
        ActiveSpaceSpec(2, 5, "CAS(2,5)"),
        ActiveSpaceSpec(label="full"),
    ),
    geometry_source="experimental equilibrium bond length r_e = 1.5949 A",
    notes="STO-3G gives 6 spatial orbitals / 4 electrons; the full space is 12 qubits",
    tags=("standard",),
)

BEH2 = BenchmarkSystem(
    name="BeH2",
    geometry=_geometry(
        [
            ("Be", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 1.3264)),
            ("H", (0.0, 0.0, -1.3264)),
        ]
    ),
    basis="sto-3g",
    active_spaces=(
        ActiveSpaceSpec(4, 6, "CAS(4,6)"),
        ActiveSpaceSpec(label="full"),
    ),
    geometry_source="linear D_inf_h equilibrium structure, Be-H = 1.3264 A",
    notes="STO-3G gives 7 spatial orbitals / 6 electrons; the full space is 14 qubits",
    tags=("standard",),
)

H2O = BenchmarkSystem(
    name="H2O",
    geometry=_geometry(
        [
            # C2v, experimental equilibrium: r(OH) = 0.9572 A, angle(HOH) = 104.52 deg.
            ("O", (0.000000, 0.000000, 0.117300)),
            ("H", (0.000000, 0.757200, -0.469200)),
            ("H", (0.000000, -0.757200, -0.469200)),
        ]
    ),
    basis="sto-3g",
    active_spaces=(
        ActiveSpaceSpec(4, 4, "CAS(4,4)"),
        ActiveSpaceSpec(6, 5, "CAS(6,5)"),
        ActiveSpaceSpec(8, 6, "CAS(8,6)"),
    ),
    geometry_source="experimental equilibrium: r(OH) = 0.9572 A, HOH = 104.52 deg",
    notes=(
        "STO-3G gives 7 spatial orbitals / 10 electrons; the full space is 14 qubits. "
        "The CAS sweep is what measures the active-space truncation error."
    ),
    tags=("standard",),
)


#: The Tier-0 validation set: everything needed to claim the engine is correct.
STANDARD_SYSTEMS: tuple[BenchmarkSystem, ...] = (LIH, BEH2, H2O)


def all_systems() -> tuple[BenchmarkSystem, ...]:
    """Dissociation curve plus the standard set."""
    return (*h2_dissociation_curve(), *STANDARD_SYSTEMS)


def by_name(name: str) -> BenchmarkSystem:
    """Look up one system, for running a single case."""
    for system in all_systems():
        if system.name == name:
            return system
    raise KeyError(
        f"unknown benchmark system {name!r}; available: {sorted(s.name for s in all_systems())}"
    )

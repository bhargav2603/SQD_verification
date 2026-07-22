"""Stage 9 support: binding energy.

    dE_bind = E_complex - (E_protein + E_ligand)

Negative means bound. The quantity is a *difference of large numbers*: each total energy
is order 10^3 Hartree for a real fragment while the binding energy is order 10^-2, so
five or six significant figures cancel. Two consequences follow, and both are enforced
here rather than left to callers:

  * every term must come from the same basis, active-space scheme and method, or the
    systematic errors stop cancelling and the difference is meaningless;
  * absolute-energy agreement is the wrong success criterion -- what matters is that the
    *errors* cancel, which is why the error budget tracks difference-level quantities.
"""

from qdr_core.errors import ValidationError
from qdr_core.limits import HARTREE_TO_KCAL_PER_MOL


def hartree_to_kcal_per_mol(energy_ha: float) -> float:
    return energy_ha * HARTREE_TO_KCAL_PER_MOL


def kcal_per_mol_to_hartree(energy_kcal: float) -> float:
    return energy_kcal / HARTREE_TO_KCAL_PER_MOL


def binding_energy(
    e_complex: float,
    e_protein: float,
    e_ligand: float,
) -> float:
    """Binding energy in Hartree. Negative indicates a bound complex."""
    return e_complex - (e_protein + e_ligand)


def interaction_energy(
    e_dimer: float,
    e_monomer_a: float,
    e_monomer_b: float,
) -> float:
    """FMO2 pair interaction energy for one dimer."""
    return e_dimer - (e_monomer_a + e_monomer_b)


def total_fmo2_energy(
    monomer_energies: dict[str, float],
    dimer_energies: dict[tuple[str, str], float],
) -> float:
    """Assemble the FMO2 total energy.

        E = sum_I E_I + sum_{I<J} (E_IJ - E_I - E_J)

    The pair term is the *interaction*, not the dimer total: adding dimer totals directly
    would count every monomer once per pair it appears in.
    """
    if not monomer_energies:
        raise ValidationError("FMO2 assembly requires at least one monomer energy")

    total = sum(monomer_energies.values())

    for (left, right), e_dimer in dimer_energies.items():
        try:
            e_left = monomer_energies[left]
            e_right = monomer_energies[right]
        except KeyError as exc:
            raise ValidationError(
                f"dimer ({left}, {right}) references an unknown monomer {exc.args[0]!r}",
                dimer=[left, right],
                known_monomers=sorted(monomer_energies),
            ) from None
        total += interaction_energy(e_dimer, e_left, e_right)

    return float(total)

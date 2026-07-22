"""Fermion-to-qubit mappings, selectable by name from the run config."""

from collections.abc import Callable

from qdr_compute_engine.mappers import bravyi_kitaev, jordan_wigner, parity
from qdr_compute_engine.types import FermionicHamiltonian, QubitHamiltonian
from qdr_core.errors import ValidationError

MapperFn = Callable[[FermionicHamiltonian], QubitHamiltonian]

MAPPERS: dict[str, MapperFn] = {
    jordan_wigner.MAPPING_NAME: jordan_wigner.map_to_qubits,
    parity.MAPPING_NAME: parity.map_to_qubits,
    bravyi_kitaev.MAPPING_NAME: bravyi_kitaev.map_to_qubits,
}

#: Mappings that are actually usable today. The control plane reads this to decide what
#: to offer in the config editor, so an unimplemented mapping is never selectable.
IMPLEMENTED_MAPPERS: frozenset[str] = frozenset({jordan_wigner.MAPPING_NAME})

DEFAULT_MAPPER = jordan_wigner.MAPPING_NAME


def get_mapper(name: str) -> MapperFn:
    try:
        return MAPPERS[name]
    except KeyError:
        raise ValidationError(
            f"unknown qubit mapping {name!r}",
            requested=name,
            available=sorted(MAPPERS),
        ) from None


__all__ = [
    "DEFAULT_MAPPER",
    "IMPLEMENTED_MAPPERS",
    "MAPPERS",
    "bravyi_kitaev",
    "get_mapper",
    "jordan_wigner",
    "parity",
]

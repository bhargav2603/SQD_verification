"""Variational ansaetze, selectable by name from the run config."""

from collections.abc import Callable

from qdr_compute_engine.ansatze import adapt, hardware_efficient, uccsd
from qdr_compute_engine.types import Circuit
from qdr_core.errors import ValidationError

AnsatzBuilder = Callable[..., Circuit]

ANSATZE: dict[str, AnsatzBuilder] = {
    uccsd.ANSATZ_NAME: uccsd.build_uccsd_ansatz,
    hardware_efficient.ANSATZ_NAME: hardware_efficient.build_hardware_efficient_ansatz,
    adapt.ANSATZ_NAME: adapt.build_adapt_ansatz,
}

#: Ansaetze that are usable today; the control plane offers only these.
IMPLEMENTED_ANSATZE: frozenset[str] = frozenset({uccsd.ANSATZ_NAME})

DEFAULT_ANSATZ = uccsd.ANSATZ_NAME


def get_ansatz_builder(name: str) -> AnsatzBuilder:
    try:
        return ANSATZE[name]
    except KeyError:
        raise ValidationError(
            f"unknown ansatz {name!r}",
            requested=name,
            available=sorted(ANSATZE),
        ) from None


__all__ = [
    "ANSATZE",
    "DEFAULT_ANSATZ",
    "IMPLEMENTED_ANSATZE",
    "adapt",
    "get_ansatz_builder",
    "hardware_efficient",
    "uccsd",
]

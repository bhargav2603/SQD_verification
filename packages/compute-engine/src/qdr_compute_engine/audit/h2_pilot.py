"""Analytic H₂/STO-3G inputs for the credential-free methodological pilot.

This deliberately narrow implementation evaluates closed-form s-Gaussian integrals for
two hydrogen atoms. It is independently checked against the repository's golden data and
PySCF CI; it is not a general chemistry-integral backend.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

from qdr_compute_engine.types import IntegralSet
from qdr_core.errors import ValidationError

BOHR_IN_ANGSTROM = 0.529177210903
STO3G_H_EXPONENTS = np.array([3.42525091, 0.62391373, 0.16885540])
STO3G_H_COEFFICIENTS = np.array([0.15432897, 0.53532814, 0.44463454])


def _boys_f0(value: float) -> float:
    if value < 1.0e-12:
        return 1.0
    return 0.5 * math.sqrt(math.pi / value) * erf(math.sqrt(value))


def _primitive_normalisation(alpha: float) -> float:
    return float((2.0 * alpha / math.pi) ** 0.75)


def _contracted_basis(
    centres: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> tuple[tuple[tuple[float, float, NDArray[np.float64]], ...], ...]:
    return tuple(
        tuple(
            (
                float(alpha),
                float(coefficient * _primitive_normalisation(float(alpha))),
                centre,
            )
            for alpha, coefficient in zip(
                STO3G_H_EXPONENTS,
                STO3G_H_COEFFICIENTS,
                strict=True,
            )
        )
        for centre in centres
    )


def h2_sto3g_integrals(bond_length_angstrom: float) -> IntegralSet:
    """Return analytic minimal-basis H₂ integrals in engine physicist notation."""

    if (
        isinstance(bond_length_angstrom, bool)
        or not math.isfinite(bond_length_angstrom)
        or bond_length_angstrom <= 0.0
    ):
        raise ValidationError(
            "H2 bond length must be finite and positive",
            bond_length_angstrom=bond_length_angstrom,
        )

    distance_bohr = bond_length_angstrom / BOHR_IN_ANGSTROM
    centres = (
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, distance_bohr]),
    )
    basis = _contracted_basis(centres)
    overlap = np.zeros((2, 2))
    kinetic = np.zeros((2, 2))
    nuclear = np.zeros((2, 2))

    for i in range(2):
        for j in range(2):
            for alpha, coefficient_a, centre_a in basis[i]:
                for beta, coefficient_b, centre_b in basis[j]:
                    exponent_sum = alpha + beta
                    reduced = alpha * beta / exponent_sum
                    separation_squared = float(
                        np.dot(centre_a - centre_b, centre_a - centre_b)
                    )
                    gaussian_product = math.exp(-reduced * separation_squared)
                    product_centre = (alpha * centre_a + beta * centre_b) / exponent_sum
                    overlap_primitive = (
                        math.pi / exponent_sum
                    ) ** 1.5 * gaussian_product
                    coefficient = coefficient_a * coefficient_b
                    overlap[i, j] += coefficient * overlap_primitive
                    kinetic[i, j] += (
                        coefficient
                        * reduced
                        * (3.0 - 2.0 * reduced * separation_squared)
                        * overlap_primitive
                    )
                    for nucleus in centres:
                        nucleus_distance_squared = float(
                            np.dot(
                                product_centre - nucleus,
                                product_centre - nucleus,
                            )
                        )
                        nuclear[i, j] += (
                            coefficient
                            * (-2.0 * math.pi / exponent_sum)
                            * gaussian_product
                            * _boys_f0(exponent_sum * nucleus_distance_squared)
                        )

    eri_atomic = np.zeros((2, 2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for m in range(2):
                    total = 0.0
                    for alpha, coefficient_a, centre_a in basis[i]:
                        for beta, coefficient_b, centre_b in basis[j]:
                            exponent_p = alpha + beta
                            centre_p = (alpha * centre_a + beta * centre_b) / exponent_p
                            factor_ab = math.exp(
                                -alpha
                                * beta
                                / exponent_p
                                * float(
                                    np.dot(
                                        centre_a - centre_b,
                                        centre_a - centre_b,
                                    )
                                )
                            )
                            for gamma, coefficient_c, centre_c in basis[k]:
                                for delta, coefficient_d, centre_d in basis[m]:
                                    exponent_q = gamma + delta
                                    centre_q = (
                                        gamma * centre_c + delta * centre_d
                                    ) / exponent_q
                                    factor_cd = math.exp(
                                        -gamma
                                        * delta
                                        / exponent_q
                                        * float(
                                            np.dot(
                                                centre_c - centre_d,
                                                centre_c - centre_d,
                                            )
                                        )
                                    )
                                    reduced_pq = (
                                        exponent_p
                                        * exponent_q
                                        / (exponent_p + exponent_q)
                                    )
                                    centre_distance_squared = float(
                                        np.dot(
                                            centre_p - centre_q,
                                            centre_p - centre_q,
                                        )
                                    )
                                    total += (
                                        coefficient_a
                                        * coefficient_b
                                        * coefficient_c
                                        * coefficient_d
                                        * 2.0
                                        * math.pi**2.5
                                        / (
                                            exponent_p
                                            * exponent_q
                                            * math.sqrt(exponent_p + exponent_q)
                                        )
                                        * factor_ab
                                        * factor_cd
                                        * _boys_f0(reduced_pq * centre_distance_squared)
                                    )
                    eri_atomic[i, j, k, m] = total

    h_core = kinetic + nuclear
    overlap_12 = overlap[0, 1]
    coefficients = np.zeros((2, 2))
    coefficients[:, 0] = np.array([1.0, 1.0]) / math.sqrt(2.0 * (1.0 + overlap_12))
    coefficients[:, 1] = np.array([1.0, -1.0]) / math.sqrt(2.0 * (1.0 - overlap_12))
    h1_mo = coefficients.T @ h_core @ coefficients
    eri_mo_chemist = np.einsum(
        "pi,qj,pqrs,rk,sl->ijkl",
        coefficients,
        coefficients,
        eri_atomic,
        coefficients,
        coefficients,
        optimize=True,
    )
    h2_physicist = np.asarray(
        eri_mo_chemist.transpose(0, 2, 3, 1),
        order="C",
    )
    nuclear_repulsion = 1.0 / distance_bohr
    hartree_fock_energy = (
        2.0 * h1_mo[0, 0] + eri_mo_chemist[0, 0, 0, 0] + nuclear_repulsion
    )
    return IntegralSet(
        h1=np.asarray(h1_mo, dtype=np.float64),
        h2=np.asarray(h2_physicist, dtype=np.float64),
        e_nuc=float(nuclear_repulsion),
        n_electrons=2,
        e_hf=float(hartree_fock_energy),
    )


__all__ = ["h2_sto3g_integrals"]

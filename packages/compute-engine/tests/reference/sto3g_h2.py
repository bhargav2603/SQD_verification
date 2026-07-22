"""Analytic H2/STO-3G integrals -- an independent reference, computed without PySCF.

Why this exists
---------------
The golden file needs integrals that can be trusted. Two options were available: copy
numbers from a paper, or compute them. Copying makes the golden file an article of faith
and cannot be checked on a machine without PySCF. Computing them here gives a second,
independent implementation, so CI comparing this against PySCF is a real cross-check
between two derivations rather than a tautology.

It is tractable because the problem is small and symmetric:

  * STO-3G hydrogen is three s-type Gaussian primitives, and all one- and two-electron
    integrals over s-type Gaussians have closed forms involving only the Boys function
    F0, which is an error function.
  * For a homonuclear diatomic in a minimal basis the RHF orbitals are fixed by symmetry
    -- sigma_g = (chi_1 + chi_2)/sqrt(2(1+S)) and sigma_u = (chi_1 - chi_2)/sqrt(2(1-S)).
    There is no SCF to converge, so no iteration and no convergence threshold.

This module is test-support code, not part of the shipped engine: production integrals
come from PySCF. It is deliberately limited to s-type basis functions on two centres and
will not generalise -- that is the trade that keeps it short enough to verify by reading.
"""

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

BOHR_IN_ANGSTROM = 0.529177210903

#: STO-3G hydrogen 1s (Hehre, Stewart & Pople 1969), already scaled for zeta = 1.24.
STO3G_H_EXPONENTS = np.array([3.42525091, 0.62391373, 0.16885540])
STO3G_H_COEFFICIENTS = np.array([0.15432897, 0.53532814, 0.44463454])


def boys_f0(t: float) -> float:
    """Boys function of order zero.

    F0(t) = integral_0^1 exp(-t u^2) du = sqrt(pi/4t) erf(sqrt(t)), with the removable
    singularity at t = 0 handled explicitly (the limit is 1).
    """
    if t < 1e-12:
        return 1.0
    return 0.5 * math.sqrt(math.pi / t) * erf(math.sqrt(t))


def primitive_normalisation(alpha: float) -> float:
    """Norm of an unnormalised s-type primitive exp(-alpha r^2)."""
    return float((2.0 * alpha / math.pi) ** 0.75)


def _contracted_basis(centres: list[NDArray[np.float64]]) -> list[list[tuple[float, float, Any]]]:
    return [
        [
            (alpha, coefficient * primitive_normalisation(alpha), centre)
            for alpha, coefficient in zip(STO3G_H_EXPONENTS, STO3G_H_COEFFICIENTS, strict=True)
        ]
        for centre in centres
    ]


def h2_reference(bond_length_angstrom: float) -> dict[str, Any]:
    """Compute the full H2/STO-3G reference at a given bond length."""
    r_bohr = bond_length_angstrom / BOHR_IN_ANGSTROM
    centres = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, r_bohr])]
    charges = [1.0, 1.0]
    basis = _contracted_basis(centres)
    n = len(basis)

    overlap = np.zeros((n, n))
    kinetic = np.zeros((n, n))
    nuclear = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            for a, ca, centre_a in basis[i]:
                for b, cb, centre_b in basis[j]:
                    p = a + b
                    mu = a * b / p
                    ab2 = float(np.dot(centre_a - centre_b, centre_a - centre_b))
                    k = math.exp(-mu * ab2)
                    product_centre = (a * centre_a + b * centre_b) / p

                    s = (math.pi / p) ** 1.5 * k
                    overlap[i, j] += ca * cb * s
                    kinetic[i, j] += ca * cb * mu * (3.0 - 2.0 * mu * ab2) * s

                    for charge, nucleus in zip(charges, centres, strict=True):
                        pc2 = float(np.dot(product_centre - nucleus, product_centre - nucleus))
                        nuclear[i, j] += (
                            ca * cb * (-2.0 * math.pi / p) * charge * k * boys_f0(p * pc2)
                        )

    eri = np.zeros((n, n, n, n))  # chemist notation (ij|kl)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for m in range(n):
                    total = 0.0
                    for a, ca, ca_centre in basis[i]:
                        for b, cb, cb_centre in basis[j]:
                            p = a + b
                            centre_p = (a * ca_centre + b * cb_centre) / p
                            k_ab = math.exp(
                                -a
                                * b
                                / p
                                * float(np.dot(ca_centre - cb_centre, ca_centre - cb_centre))
                            )
                            for c, cc, cc_centre in basis[k]:
                                for d, cd, cd_centre in basis[m]:
                                    q = c + d
                                    centre_q = (c * cc_centre + d * cd_centre) / q
                                    k_cd = math.exp(
                                        -c
                                        * d
                                        / q
                                        * float(
                                            np.dot(cc_centre - cd_centre, cc_centre - cd_centre)
                                        )
                                    )
                                    rho = p * q / (p + q)
                                    pq2 = float(np.dot(centre_p - centre_q, centre_p - centre_q))
                                    total += (
                                        ca
                                        * cb
                                        * cc
                                        * cd
                                        * 2.0
                                        * math.pi**2.5
                                        / (p * q * math.sqrt(p + q))
                                        * k_ab
                                        * k_cd
                                        * boys_f0(rho * pq2)
                                    )
                    eri[i, j, k, m] = total

    h_core = kinetic + nuclear

    # Symmetry-adapted RHF orbitals; no SCF needed for a homonuclear minimal basis.
    s12 = overlap[0, 1]
    mo_coefficients = np.zeros((2, 2))
    mo_coefficients[:, 0] = np.array([1.0, 1.0]) / math.sqrt(2.0 * (1.0 + s12))
    mo_coefficients[:, 1] = np.array([1.0, -1.0]) / math.sqrt(2.0 * (1.0 - s12))

    h1_mo = mo_coefficients.T @ h_core @ mo_coefficients
    eri_mo = np.einsum(
        "pi,qj,pqrs,rk,sl->ijkl",
        mo_coefficients,
        mo_coefficients,
        eri,
        mo_coefficients,
        mo_coefficients,
        optimize=True,
    )

    e_nuc = 1.0 / r_bohr
    # Closed-shell RHF with a single doubly occupied orbital: E = 2h_00 + (00|00) + E_nuc.
    e_hf = 2.0 * h1_mo[0, 0] + eri_mo[0, 0, 0, 0] + e_nuc

    return {
        "bond_length_angstrom": bond_length_angstrom,
        "r_bohr": r_bohr,
        "overlap": overlap,
        "h1_mo": h1_mo,
        "eri_mo_chemist": eri_mo,
        "e_nuc": float(e_nuc),
        "e_hf": float(e_hf),
        "mo_coefficients": mo_coefficients,
    }


def chemist_to_physicist(eri: NDArray[np.float64]) -> NDArray[np.float64]:
    """Local copy of the notation change, so this reference stays independent of the
    engine module it is used to check."""
    return np.asarray(eri.transpose(0, 2, 3, 1), order="C")

"""Unit tests for PySCF orchestration without requiring PySCF on Windows."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from qdr_compute_engine.validation.pyscf_diagnostics import (
    check_scf_stability,
    run_casscf_nevpt2,
    run_ccsd_t,
)


def _avas_fixture() -> SimpleNamespace:
    return SimpleNamespace(
        active_space=SimpleNamespace(n_active_orbitals=4, n_active_electrons=4),
        mean_field=object(),
        mo_coeff=np.eye(4),
    )


def test_casscf_nevpt2_uses_avas_orbitals_and_target_spin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeFCI:
        @staticmethod
        def spin_square(ci: object, ncas: int, nelecas: tuple[int, int]) -> tuple[float, float]:
            calls["spin_square"] = (ci, ncas, nelecas)
            return 2.0, 3.0

    class FakeCASSCF:
        converged = True
        e_tot = -10.5
        ci = object()
        fcisolver = FakeFCI()

        def __init__(self, mean_field: object, ncas: int, nelecas: tuple[int, int]) -> None:
            calls["constructor"] = (mean_field, ncas, nelecas)
            self.ncas = ncas
            self.nelecas = nelecas

        def fix_spin_(self, *, shift: float, ss: float) -> None:
            calls["fix_spin"] = (shift, ss)

        def kernel(self, mo_coeff: np.ndarray) -> tuple[float]:
            calls["mo_coeff"] = mo_coeff
            return (self.e_tot,)

    class FakeNEVPT:
        def __init__(self, casscf: FakeCASSCF) -> None:
            calls["nevpt_casscf"] = casscf

        def kernel(self) -> float:
            return -0.2

    pyscf = ModuleType("pyscf")
    pyscf.mcscf = SimpleNamespace(CASSCF=FakeCASSCF)  # type: ignore[attr-defined]
    pyscf.mrpt = SimpleNamespace(NEVPT=FakeNEVPT)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyscf", pyscf)

    result = run_casscf_nevpt2(_avas_fixture(), spin_multiplicity=3)

    assert calls["constructor"][1:] == (4, (3, 1))  # type: ignore[index]
    assert calls["fix_spin"] == (1.0, 2.0)
    assert calls["mo_coeff"] is _avas_fixture().mo_coeff or np.array_equal(
        calls["mo_coeff"], np.eye(4)
    )
    assert result.converged
    assert result.e_casscf == pytest.approx(-10.5)
    assert result.e_nevpt2_correction == pytest.approx(-0.2)
    assert result.e_nevpt2 == pytest.approx(-10.7)
    assert result.spin_squared == pytest.approx(2.0)


def test_unconverged_casscf_does_not_run_nevpt2(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCASSCF:
        converged = False
        e_tot = -10.0
        ci = None

        def __init__(self, mean_field: object, ncas: int, nelecas: tuple[int, int]) -> None:
            self.ncas = ncas
            self.nelecas = nelecas

        def fix_spin_(self, *, shift: float, ss: float) -> None:
            pass

        def kernel(self, mo_coeff: np.ndarray) -> tuple[float]:
            return (self.e_tot,)

    class ForbiddenNEVPT:
        def __init__(self, casscf: FakeCASSCF) -> None:
            raise AssertionError("NEVPT2 must not run on an unconverged CASSCF reference")

    pyscf = ModuleType("pyscf")
    pyscf.mcscf = SimpleNamespace(CASSCF=FakeCASSCF)  # type: ignore[attr-defined]
    pyscf.mrpt = SimpleNamespace(NEVPT=ForbiddenNEVPT)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyscf", pyscf)

    result = run_casscf_nevpt2(_avas_fixture(), spin_multiplicity=1)

    assert not result.converged
    assert result.e_nevpt2 is None


def test_open_shell_ccsd_t_reports_convergence_triples_and_t1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCCSD:
        converged = True
        e_tot = -20.0

        def __init__(self, mean_field: object, frozen: int | None = None) -> None:
            self.mean_field = mean_field
            self.frozen = frozen

        def kernel(self) -> tuple[float]:
            return (-0.3,)

        def ccsd_t(self) -> float:
            return -0.04

        def get_t1_diagnostic(self) -> float:
            return 0.031

    pyscf = ModuleType("pyscf")
    pyscf.cc = SimpleNamespace(CCSD=FakeCCSD)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyscf", pyscf)

    result = run_ccsd_t(object(), frozen_orbitals=14)

    assert result.converged
    assert result.e_ccsd == pytest.approx(-20.0)
    assert result.triples_correction == pytest.approx(-0.04)
    assert result.e_ccsd_t == pytest.approx(-20.04)
    assert result.t1_diagnostic == pytest.approx(0.031)


def test_scf_stability_extracts_internal_status_without_external_rohf_status() -> None:
    class FakeROHF:
        @staticmethod
        def stability(
            *,
            internal: bool,
            external: bool,
            return_status: bool,
        ) -> tuple[object, None, bool, None]:
            assert internal and not external and return_status
            return object(), None, False, None

    assert check_scf_stability(FakeROHF()) is False

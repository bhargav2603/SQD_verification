"""Engine test fixtures and the PySCF skip mechanism.

PySCF publishes no Windows wheels, so tests that need it are marked ``requires_pyscf``
and skipped automatically when it cannot be imported. The skip is decided by *importing*
rather than by checking the platform: what matters is whether the dependency is actually
there, and that keeps the same rule working in the Linux CI container where it is.

Everything that does not need PySCF is expected to run everywhere. That is the point of
the golden file carrying integrals rather than only energies -- stages 5 to 9 are fully
exercised on any platform.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

TESTS_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = TESTS_DIR / "golden" / "h2_fci_reference.json"

# Make `reference.*` importable regardless of where pytest is invoked from.
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from qdr_compute_engine.types import IntegralSet  # noqa: E402


def pyscf_is_available() -> bool:
    try:
        import pyscf  # noqa: F401
    except ImportError:
        return False
    return True


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip PySCF-dependent tests when PySCF is not installed."""
    if pyscf_is_available():
        return
    skip_marker = pytest.mark.skip(
        reason="PySCF is not installed (it publishes no Windows wheels); this test runs in Linux CI"
    )
    for item in items:
        if "requires_pyscf" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def golden_h2() -> dict[str, Any]:
    """The H2/STO-3G golden reference payload."""
    if not GOLDEN_PATH.exists():
        pytest.fail(
            f"golden reference missing at {GOLDEN_PATH}; "
            "regenerate with tests/reference/generate_golden.py"
        )
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def h2_integrals(golden_h2: dict[str, Any]) -> IntegralSet:
    """H2/STO-3G integrals from the golden file -- no PySCF required."""
    return IntegralSet(
        h1=np.array(golden_h2["h1"], dtype=float),
        h2=np.array(golden_h2["h2"], dtype=float),
        e_nuc=float(golden_h2["e_nuc"]),
        n_electrons=int(golden_h2["n_electrons"]),
        e_hf=float(golden_h2["e_hf"]),
    )


@pytest.fixture
def h2_geometry() -> Any:
    """H2 at the golden file's bond length."""
    from qdr_compute_engine.types import Atom, Geometry, LengthUnit

    return Geometry(
        atoms=(
            Atom("H", (0.0, 0.0, 0.0)),
            Atom("H", (0.0, 0.0, 0.735)),
        ),
        charge=0,
        spin_multiplicity=1,
        units=LengthUnit.ANGSTROM,
    )

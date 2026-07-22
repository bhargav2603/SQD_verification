"""Capture the software environment into the run manifest.

A quantum-chemistry result is only reproducible alongside the versions that produced it:
a PySCF or SciPy upgrade can move an energy in the last digits, and without a recorded
version an unexplained difference between two runs is unattributable. Recording the
Python version and platform matters for the same reason -- BLAS differences across
platforms are a real source of last-digit drift.
"""

import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

#: Packages whose version can move a computed energy.
TRACKED_PACKAGES = (
    "qdr-core",
    "qdr-compute-engine",
    "numpy",
    "scipy",
    "openfermion",
    "pyscf",
)


def package_version(name: str) -> str | None:
    """Installed version of ``name``, or None when it is absent."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def capture_version_pins() -> dict[str, str]:
    """Snapshot the versions of everything that can affect a result.

    Absent optional packages are recorded as ``"absent"`` rather than omitted, so a
    manifest distinguishes "PySCF was not installed" from "we forgot to look".
    """
    pins: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for package in TRACKED_PACKAGES:
        pins[package] = package_version(package) or "absent"
    return pins


def version_record() -> dict[str, Any]:
    """The version block written into the run manifest."""
    return {"versions": capture_version_pins()}

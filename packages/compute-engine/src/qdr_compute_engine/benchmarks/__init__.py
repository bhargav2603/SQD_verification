"""Engine validation suite.

Runs benchmark systems against exact references and reports *where* the error comes
from, rather than only how large it is. Requires PySCF, so it runs on a Linux/macOS
host -- see ``docs/REMOTE_EXECUTION.md``.

Nothing in the engine imports this package, so the engine stays importable where PySCF
is absent.
"""

from qdr_compute_engine.benchmarks.systems import (
    BEH2,
    H2O,
    LIH,
    STANDARD_SYSTEMS,
    ActiveSpaceSpec,
    BenchmarkSystem,
    all_systems,
    by_name,
    h2_at,
    h2_dissociation_curve,
)

__all__ = [
    "BEH2",
    "H2O",
    "LIH",
    "STANDARD_SYSTEMS",
    "ActiveSpaceSpec",
    "BenchmarkSystem",
    "all_systems",
    "by_name",
    "h2_at",
    "h2_dissociation_curve",
]

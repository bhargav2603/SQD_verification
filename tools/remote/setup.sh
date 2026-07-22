#!/usr/bin/env bash
# Provision a Linux VM to run the QDR validation suite.
#
# Every step that can fail silently is verified explicitly. The failure this guards
# against is a green-looking environment where PySCF did not install: the test suite
# would then *skip* its way to a pass and the validation sweep would refuse to run,
# which is a confusing way to lose an afternoon.
#
# Usage:  bash tools/remote/setup.sh
# Idempotent: safe to re-run.

set -euo pipefail

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

say "Host"
uname -a
echo "cores:  $(nproc 2>/dev/null || echo unknown)"
echo "memory: $(free -h 2>/dev/null | awk '/^Mem:/{print $2}' || echo unknown)"

# The statevector and the sparse Hamiltonian both scale as 2^n. At 16 qubits the
# sparse operator alone measured 5.7 GiB, and it is the binding constraint before the
# statevector is. Warn early rather than at hour three of a sweep.
if command -v free >/dev/null 2>&1; then
  mem_gb="$(free -g | awk '/^Mem:/{print $2}')"
  if [ "${mem_gb:-0}" -lt 8 ]; then
    printf '\033[33mWARNING: %s GiB RAM. 14-qubit cases need several GiB for the sparse\n' "$mem_gb"
    printf 'Hamiltonian; 16-qubit cases need ~6 GiB for that alone.\033[0m\n'
  fi
fi

if [ "$(uname -s)" = "Darwin" ] || [ "$(uname -s)" = "Linux" ]; then
  :
else
  die "PySCF requires Linux or macOS. Detected: $(uname -s)"
fi

say "uv"
if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv is still not on PATH; add \$HOME/.local/bin"
uv --version

say "Dependencies (including the chemistry group)"
# --group, not --extra: see the comment in the root pyproject.toml.
uv sync --group chem

say "Verifying the scientific stack actually imports"
uv run python - <<'PYTHON'
import sys

failures = []
for name in ("numpy", "scipy", "openfermion", "pyscf"):
    try:
        module = __import__(name)
        print(f"  {name:<14} {getattr(module, '__version__', 'unknown')}")
    except ImportError as exc:
        failures.append(f"{name}: {exc}")

if failures:
    print("\nFAILED to import:", file=sys.stderr)
    for failure in failures:
        print(f"  {failure}", file=sys.stderr)
    sys.exit(1)
PYTHON

say "Smoke test: RHF + FCI on H2 through PySCF directly"
uv run python - <<'PYTHON'
from pyscf import fci, gto, scf

mol = gto.M(atom="H 0 0 0; H 0 0 0.735", basis="sto-3g", verbose=0)
mf = scf.RHF(mol).run()
e_fci = fci.FCI(mf).kernel()[0]
print(f"  E_HF  = {mf.e_tot:.10f} Ha")
print(f"  E_FCI = {e_fci:.10f} Ha")

# The engine's committed golden reference. A mismatch here means the chemistry stack
# disagrees with the value the whole test suite is pinned to.
expected = -1.137306028320
assert abs(e_fci - expected) < 1e-8, f"E_FCI {e_fci} != golden {expected}"
print("  matches the committed golden reference")
PYTHON

say "Test suite (nothing may skip on this host)"
uv run pytest packages/compute-engine packages/qdr-core -q -rs 2>&1 | tail -20

say "Confirming no test was skipped"
uv run pytest packages/compute-engine -q --no-header -rs 2>&1 | tee /tmp/qdr-result.txt >/dev/null
if grep -qE "[0-9]+ skipped" /tmp/qdr-result.txt; then
  grep -E "SKIPPED" /tmp/qdr-result.txt || true
  die "tests were skipped on a host where every dependency is installed"
fi
echo "  nothing skipped"

say "Ready"
cat <<'EOF'
Run the validation suite:

  uv run python -m qdr_compute_engine.benchmarks --suite h2-scan  --out results/
  uv run python -m qdr_compute_engine.benchmarks --suite standard --out results/
  uv run python -m qdr_compute_engine.benchmarks --suite all      --out results/

For a long sweep, detach it so an SSH drop does not kill the run:

  nohup uv run python -m qdr_compute_engine.benchmarks --suite all \
      --out results/ > results/run.log 2>&1 &
  tail -f results/run.log

Results are written after every case, so a crash never discards completed work.
EOF

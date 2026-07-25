"""Command-line entry point for the locked sampler-audit protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from qdr_compute_engine.audit.protocol import load_protocol, run_protocol
from qdr_core.errors import ValidationError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m qdr_compute_engine.audit",
        description="Run a strict, paired determinant-sampler audit protocol.",
    )
    parser.add_argument("protocol", type=Path, help="locked JSON protocol")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="parent directory for the content-addressed evidence bundle",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        protocol = load_protocol(args.protocol)
        run = run_protocol(protocol, output_dir=args.output)
    except ValidationError as exc:
        print(f"invalid protocol or evidence: {exc}", file=sys.stderr)
        return 2

    print(f"decision: {run.result.comparison.status.value}")
    print(f"scientific hash: {run.scientific_hash}")
    print(f"manifest hash: {run.bundle.manifest_hash}")
    print(f"bundle: {run.bundle.manifest.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

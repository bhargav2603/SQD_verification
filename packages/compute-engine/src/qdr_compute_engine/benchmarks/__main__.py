"""CLI for the validation suite. Runs on the remote host; needs PySCF.

    python -m qdr_compute_engine.benchmarks --suite standard --out results/
    python -m qdr_compute_engine.benchmarks --suite h2-scan --out results/
    python -m qdr_compute_engine.benchmarks --suite all --optimizer COBYLA --out results/

Results are written incrementally: a case that finishes is persisted before the next one
starts, so a long sweep that dies at case 19 of 21 still leaves 18 usable results rather
than nothing. Long unattended runs on a VM are exactly where that matters.
"""

import argparse
import sys
import traceback
from pathlib import Path

from qdr_compute_engine.benchmarks import systems
from qdr_compute_engine.benchmarks.report import write_results
from qdr_compute_engine.benchmarks.runner import BenchmarkRecord, run_case

SUITES = {
    "h2-scan": lambda: list(systems.h2_dissociation_curve()),
    "standard": lambda: list(systems.STANDARD_SYSTEMS),
    "all": lambda: list(systems.all_systems()),
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qdr-benchmarks",
        description="Validate the engine against exact references and report the "
        "error decomposition.",
    )
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        default="standard",
        help="which set of systems to run (default: standard)",
    )
    parser.add_argument(
        "--system",
        action="append",
        default=None,
        help="run a single named system; repeatable. Overrides --suite.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results"),
        help="output directory (default: results/)",
    )
    parser.add_argument(
        "--stem", default="validation", help="output filename stem (default: validation)"
    )
    parser.add_argument(
        "--optimizer",
        default=None,
        help="force an optimizer (e.g. COBYLA). Default lets the driver choose "
        "L-BFGS-B with analytic adjoint gradients.",
    )
    parser.add_argument("--max-iterations", type=int, default=2000, help="optimizer iteration cap")
    parser.add_argument(
        "--no-full-fci",
        action="store_true",
        help="skip the full-space FCI reference (faster, but the active-space error "
        "then cannot be quantified and is reported as unknown rather than zero)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop at the first failing case instead of recording it and continuing",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="skip the PNG figures (JSON and Markdown are always written)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        from qdr_compute_engine.integrals.pyscf_integrals import pyscf_available

        if not pyscf_available():
            print(
                "ERROR: PySCF is not installed. The validation suite computes its own "
                "reference energies and cannot run without it.\n"
                "  Linux/macOS:  uv sync --extra chem\n"
                "  See docs/REMOTE_EXECUTION.md for the remote-VM setup.",
                file=sys.stderr,
            )
            return 2
    except ImportError:  # pragma: no cover - defensive
        print("ERROR: the engine package is not importable", file=sys.stderr)
        return 2

    if args.system:
        selected = [systems.by_name(name) for name in args.system]
    else:
        selected = SUITES[args.suite]()

    cases = [(system, spec) for system in selected for spec in system.active_spaces]
    print(f"Running {len(cases)} case(s) from {len(selected)} system(s).", flush=True)

    records: list[BenchmarkRecord] = []
    failures: list[tuple[str, str]] = []

    for index, (system, spec) in enumerate(cases, start=1):
        label = f"{system.name} / {spec.label}"
        print(f"[{index}/{len(cases)}] {label} ...", end=" ", flush=True)
        try:
            record = run_case(
                system,
                spec,
                optimizer=args.optimizer,
                max_iterations=args.max_iterations,
                compute_full_fci=not args.no_full_fci,
            )
        except Exception as exc:
            print(f"FAILED ({type(exc).__name__}: {exc})", flush=True)
            failures.append((label, traceback.format_exc()))
            if args.fail_fast:
                break
            continue

        records.append(record)
        total = "n/a" if record.error_total is None else f"{record.error_total:+.2e} Ha"
        print(
            f"{record.n_qubits}q  E_VQE={record.e_vqe:.8f}  total_err={total}  "
            f"{record.seconds_vqe:.1f}s" + ("  [WARNINGS]" if record.warnings else ""),
            flush=True,
        )

        # Persist after every case, so a crash later does not discard what already ran.
        write_results(records, args.out, stem=args.stem)

    if not records:
        print("\nNo cases completed successfully.", file=sys.stderr)
        for label, tb in failures:
            print(f"\n--- {label} ---\n{tb}", file=sys.stderr)
        return 1

    json_path, markdown_path = write_results(records, args.out, stem=args.stem)
    print(f"\nWrote {json_path}\nWrote {markdown_path}")

    if not args.no_plots:
        try:
            from qdr_compute_engine.benchmarks.plots import write_figures

            for figure_path in write_figures(records, args.out, stem=args.stem):
                print(f"Wrote {figure_path}")
        except ImportError:
            print(
                "NOTE: matplotlib is not installed, so no figures were drawn. "
                "Install it (`uv sync --group chem`) or plot later from "
                f"{json_path} with qdr_compute_engine.benchmarks.plots.load_records.",
                file=sys.stderr,
            )

    n_warned = sum(1 for r in records if r.warnings)
    print(f"\n{len(records)} case(s) completed, {n_warned} with warnings, {len(failures)} failed.")
    for label, tb in failures:
        print(f"\n--- {label} ---\n{tb}", file=sys.stderr)

    # A sweep with warnings or failures is not a clean run, and the exit code says so.
    return 0 if (n_warned == 0 and not failures) else 1


if __name__ == "__main__":
    raise SystemExit(main())

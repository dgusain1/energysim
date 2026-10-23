# -*- coding: utf-8 -*-
"""
Command-Line Interface (CLI) runner for energysim.

Usage
-----
energysim run scenario.yaml [--out DIR] [--to-csv] [--dashboard] [--dashboard-path P]
              [--offline] [--no-pbar] [--safe] [--validate-only]
energysim validate scenario.yaml [--safe] [-v]
energysim units [--system default|si] [--set quantity=unit ...] [--json]

``--out DIR`` writes the run to ``DIR/<YYYYmmdd-HHMMSS>_<hash8>/`` (Parquet or
CSV per simulator, ``meta.json``, ``scenario.yaml``).  ``--to-csv`` also
exports CSV files (into ``<run dir>/csv``; implies ``--out runs`` when no
``--out`` is given).  ``--dashboard`` writes the HTML dashboard and opens it.

``-v`` / ``--verbose`` (``run`` and ``validate``) also prints the unit system
(name and hash) and every connection's unit conversion.  ``energysim units``
prints the default unit per quantity of a unit system.

``--safe`` refuses simulator types that execute arbitrary code (external,
script, matlab, powerfactory), dotted-import model references and pickle
model files. Use it for scenario files from untrusted sources.
energysim --version
"""

from __future__ import annotations

import argparse
import sys
import os
from .world import world
from .base import ValidationError
from . import __version__


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="energysim",
        description="energysim: Co-simulation platform for multi-energy systems.",
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"energysim {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser(
        "run",
        help="Execute a simulation from a YAML scenario file.",
    )
    run_parser.add_argument(
        "scenario",
        help="Path to scenario YAML file.",
    )
    run_parser.add_argument(
        "--out",
        default=None,
        metavar="DIR",
        help="Write the run to DIR/<timestamp>_<hash>/ (results, meta.json, scenario.yaml).",
    )
    run_parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Write the interactive HTML dashboard and open it in the browser.",
    )
    run_parser.add_argument(
        "--dashboard-path",
        default=None,
        help="Path of the HTML dashboard file (written even without --dashboard).",
    )
    run_parser.add_argument(
        "--offline",
        action="store_true",
        help="Embed plotly.js in the dashboard (needs the 'plotly' package).",
    )
    run_parser.add_argument(
        "--to-csv",
        action="store_true",
        help="Export results as CSV files into <run dir>/csv (implies --out runs).",
    )
    run_parser.add_argument(
        "--no-pbar",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    run_parser.add_argument(
        "--safe",
        action="store_true",
        help="Safe mode: refuse code-executing simulator types, dotted imports and pickle models.",
    )
    run_parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run semantic pre-flight validation without executing simulation.",
    )

    # validate command
    val_parser = subparsers.add_parser(
        "validate",
        help="Perform pre-flight semantic and graph validation on a scenario file.",
    )
    val_parser.add_argument(
        "scenario",
        help="Path to scenario YAML file.",
    )
    val_parser.add_argument(
        "--safe",
        action="store_true",
        help="Safe mode: refuse code-executing simulator types, dotted imports and pickle models.",
    )

    for p in (run_parser, val_parser):
        p.add_argument(
            "-v", "--verbose",
            action="store_true",
            help="Print the unit system (name + hash) and every connection's unit conversion.",
        )

    # units command
    units_parser = subparsers.add_parser(
        "units",
        help="Print the default unit per quantity of a unit system.",
    )
    units_parser.add_argument(
        "--system",
        default="default",
        choices=["default", "si"],
        help="Unit system preset (default: the energysim default system).",
    )
    units_parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="QUANTITY=UNIT",
        help="Override the default unit of a quantity (repeatable), e.g. --set active_power=MW.",
    )
    units_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the unit system as JSON (name, hash, defaults, table).",
    )

    return parser


def _units_command(args) -> int:
    import json
    from .units import UnitSystem
    try:
        system = UnitSystem.coerce(args.system)
        overrides = {}
        for item in args.set:
            if "=" not in item:
                raise ValueError(f"--set expects QUANTITY=UNIT, got {item!r}")
            q, u = item.split("=", 1)
            overrides[q.strip()] = u.strip()
        if overrides:
            system = system.with_overrides(overrides)
    except (ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        data = system.to_dict()
        data["table"] = system.table()
        print(json.dumps(data, indent=2))
    else:
        print(system.format_table())
    return 0


def _print_units(w) -> None:
    report = w.validate(raise_on_error=False)
    print(report.format_units())


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "units":
        return _units_command(args)

    if args.command == "validate":
        scenario_path = os.path.abspath(args.scenario)
        if not os.path.exists(scenario_path):
            print(f"Error: Scenario file not found: {scenario_path}", file=sys.stderr)
            return 1
        try:
            w = world.from_yaml(scenario_path, safe=args.safe)
            report = w.validate(raise_on_error=False)
            print(report)
            if args.verbose:
                print(report.format_units())
            return 0 if report.is_valid else 1
        except Exception as e:
            print(f"Validation failed with error: {e}", file=sys.stderr)
            return 1

    elif args.command == "run":
        scenario_path = os.path.abspath(args.scenario)
        if not os.path.exists(scenario_path):
            print(f"Error: Scenario file not found: {scenario_path}", file=sys.stderr)
            return 1

        try:
            w = world.from_yaml(scenario_path, safe=args.safe)

            # Pre-flight validate
            report = w.validate(raise_on_error=False)
            if args.verbose:
                print(report.format_units())
            if report.warnings:
                print("Validation Warnings:")
                for warn in report.warnings:
                    print(f"  [WARN] {warn}")

            if not report.is_valid:
                print("Validation Errors:", file=sys.stderr)
                for err in report.errors:
                    print(f"  [ERROR] {err}", file=sys.stderr)
                return 1

            if args.validate_only:
                print("Scenario pre-flight validation PASSED.")
                return 0

            out_dir = args.out
            if args.to_csv and not out_dir:
                out_dir = w.out_dir or "runs"
            print(f"Starting simulation: {scenario_path}")
            res = w.simulate(pbar=not args.no_pbar, record_all=False, validate=False,
                             out_dir=out_dir)
            print("Simulation finished successfully.")
            if res.run_dir:
                print(f"Run directory: {res.run_dir}")
            print(res.summary().to_string())

            if args.to_csv:
                csv_dir = res.to_csv(os.path.join(res.run_dir, "csv"))
                print(f"CSV files: {csv_dir}")
            if args.dashboard or args.dashboard_path:
                path = res.dashboard(path=args.dashboard_path, open=args.dashboard,
                                     offline=args.offline)
                print(f"Dashboard: {path}")
            return 0
        except ValidationError as ve:
            print(f"Pre-flight Validation Error:\n{ve}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Simulation execution error: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

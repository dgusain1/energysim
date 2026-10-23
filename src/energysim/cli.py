# -*- coding: utf-8 -*-
"""
Command-Line Interface (CLI) runner for energysim.

Usage
-----
energysim run scenario.yaml [--dashboard] [--to-csv] [--no-pbar] [--safe]
energysim validate scenario.yaml [--safe]

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
        "--dashboard",
        action="store_true",
        help="Open interactive HTML dashboard after simulation finishes.",
    )
    run_parser.add_argument(
        "--dashboard-path",
        default=None,
        help="Path to save HTML dashboard file.",
    )
    run_parser.add_argument(
        "--to-csv",
        action="store_true",
        help="Export simulation results to CSV files.",
    )
    run_parser.add_argument(
        "--res-filename",
        default=None,
        help="Override results HDF5 storage filename.",
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "validate":
        scenario_path = os.path.abspath(args.scenario)
        if not os.path.exists(scenario_path):
            print(f"Error: Scenario file not found: {scenario_path}", file=sys.stderr)
            return 1
        try:
            w = world.from_yaml(scenario_path, safe=args.safe)
            report = w.validate(raise_on_error=False)
            print(report)
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
            if args.res_filename:
                w.file_name = args.res_filename

            # Pre-flight validate
            report = w.validate(raise_on_error=False)
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

            print(f"Starting simulation: {scenario_path}")
            w.simulate(pbar=not args.no_pbar, record_all=False, validate=False)
            print("Simulation finished successfully.")

            if args.to_csv or args.dashboard:
                w.results(
                    to_csv=args.to_csv,
                    dashboard=args.dashboard,
                    dashboard_path=args.dashboard_path,
                )
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

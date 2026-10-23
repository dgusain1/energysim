#!/usr/bin/env python
"""Merge ``changes/*.<type>.md`` fragments into CHANGELOG.txt (dependency-free, towncrier-like).

Fragment file name: ``<anything>.<type>.md`` with type in added|changed|fixed|removed|docs|deprecated.
The file content is one or more Markdown bullet lines (a bare sentence is turned into a bullet).

Usage::

    python scripts/build_changelog.py --version 3.0.0            # insert section, delete fragments
    python scripts/build_changelog.py --version 3.0.0 --draft    # print the section only
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHANGES = os.path.join(ROOT, "changes")
CHANGELOG = os.path.join(ROOT, "CHANGELOG.txt")
TYPES = [("added", "Added"), ("changed", "Changed"), ("deprecated", "Deprecated"),
         ("removed", "Removed"), ("fixed", "Fixed"), ("docs", "Documentation")]


def collect(changes_dir=CHANGES):
    found = {t: [] for t, _ in TYPES}
    paths = []
    for name in sorted(os.listdir(changes_dir)):
        parts = name.split(".")
        if name.lower() == "readme.md" or len(parts) < 3 or parts[-1] != "md":
            continue
        kind = parts[-2]
        if kind not in found:
            raise SystemExit(f"changes/{name}: unknown type '{kind}' (use one of {', '.join(found)})")
        with open(os.path.join(changes_dir, name), encoding="utf8") as fh:
            text = fh.read().strip()
        if not text:
            continue
        lines = text.splitlines()
        if not lines[0].lstrip().startswith(("-", "*")):
            lines[0] = "- " + lines[0]
        found[kind].append("\n".join(lines))
        paths.append(os.path.join(changes_dir, name))
    return found, paths


def render(version: str, found: dict, date: str | None = None) -> str:
    out = [f"## {version} ({date or datetime.date.today().isoformat()})"]
    for kind, title in TYPES:
        if found[kind]:
            out.append(f"- **{title}**:")
            for entry in found[kind]:
                out.extend("  " + ln if ln.strip() else ln for ln in entry.splitlines())
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", required=True)
    ap.add_argument("--draft", action="store_true", help="print the new section; change nothing")
    args = ap.parse_args(argv)
    found, paths = collect()
    if not paths:
        print("no change fragments found in changes/", file=sys.stderr)
        return 1
    section = render(args.version, found)
    if args.draft:
        print(section)
        return 0
    with open(CHANGELOG, encoding="utf8", newline="") as fh:
        text = fh.read().replace("\r\n", "\n")
    head, sep, rest = text.partition("\n## ")
    new = head + "\n\n" + section + ("\n## " + rest if sep else "")
    with open(CHANGELOG, "w", encoding="utf8", newline="\n") as fh:
        fh.write(new)
    for p in paths:
        os.remove(p)
    print(f"merged {len(paths)} fragment(s) into CHANGELOG.txt as {args.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

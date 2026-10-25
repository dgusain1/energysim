# -*- coding: utf-8 -*-
"""
Build ``examples/notebooks/*.ipynb`` from the percent-format sources in ``scripts/notebook_src/``.

Why percent format?  A notebook source is a plain ``.py`` file: it diffs cleanly, highlights in any
editor, and can be syntax-checked without a kernel (``python scripts/build_notebooks.py --check``).

Format (the same as jupytext's ``py:percent``)::

    # %% [markdown]
    # # Title                      <- every line of a markdown cell starts with "# " (or is a bare "#")
    # %%
    code = "everything else is a code cell"
    #@include snippets/x.py        <- replaced by the contents of that file (relative to this script's
                                      ``notebook_src`` directory) so shared code stays single-sourced
                                      while every notebook remains self-contained

The first ``# %% [markdown]`` cell becomes the notebook title.  Notebooks are written *unexecuted*
(no outputs); the slow test ``tests/docs/test_notebooks.py`` runs every notebook top to bottom.

Usage::

    python scripts/build_notebooks.py            # build every notebook_src/*.py
    python scripts/build_notebooks.py 12 14      # only those whose file name starts with 12 / 14
    python scripts/build_notebooks.py --check    # parse every code cell, write nothing
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "notebook_src"
OUT_DIR = HERE.parent / "examples" / "notebooks"

_CELL = re.compile(r"^# %%(?P<rest>.*)$")
_INCLUDE = re.compile(r"^#@include\s+(?P<path>\S+)\s*$")

METADATA = {
    "language_info": {
        "name": "python", "version": "3.11", "mimetype": "text/x-python",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "pygments_lexer": "ipython3", "nbconvert_exporter": "python", "file_extension": ".py",
    },
    "kernelspec": {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"},
}


def _lines(text: str) -> List[str]:
    out = text.strip("\n").split("\n")
    return [ln + "\n" for ln in out[:-1]] + [out[-1]]


def _expand_includes(code: str, origin: str) -> str:
    out = []
    for line in code.split("\n"):
        m = _INCLUDE.match(line)
        if not m:
            out.append(line)
            continue
        target = (SRC_DIR / m.group("path")).resolve()
        if not target.is_file():
            raise FileNotFoundError(f"{origin}: #@include target not found: {m.group('path')}")
        out.append(target.read_text(encoding="utf-8").strip("\n"))
    return "\n".join(out)


def parse_source(path: Path) -> List[Tuple[str, str]]:
    """Return ``[(kind, text), ...]`` with kind in {"markdown", "code"}."""
    cells: List[Tuple[str, str]] = []
    kind, buf = None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if not body.strip():
            return
        if kind == "markdown":
            md = []
            for ln in body.split("\n"):
                if ln.startswith("# "):
                    md.append(ln[2:])
                elif ln == "#":
                    md.append("")
                else:
                    raise ValueError(f"{path.name}: markdown line must start with '# ': {ln!r}")
            cells.append(("markdown", "\n".join(md)))
        else:
            cells.append(("code", _expand_includes(body, path.name)))

    for raw in path.read_text(encoding="utf-8").split("\n"):
        m = _CELL.match(raw)
        if m:
            flush()
            kind = "markdown" if "[markdown]" in m.group("rest") else "code"
            buf = []
        elif kind is not None:
            buf.append(raw)
        elif raw.strip() and not raw.startswith("#"):
            raise ValueError(f"{path.name}: text before the first '# %%' marker: {raw!r}")
    flush()
    return cells


def to_notebook(cells: List[Tuple[str, str]]) -> dict:
    nb_cells = []
    for kind, text in cells:
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(text)})
        else:
            nb_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                             "outputs": [], "source": _lines(text)})
    return {"cells": nb_cells, "metadata": METADATA, "nbformat": 4, "nbformat_minor": 4}


def check_cells(name: str, cells: List[Tuple[str, str]]) -> List[str]:
    """Syntax-check every code cell (no execution)."""
    problems = []
    for i, (kind, text) in enumerate(cells):
        if kind != "code":
            continue
        try:
            ast.parse(text)
        except SyntaxError as exc:
            problems.append(f"{name}: code cell {i}: {exc.msg} (line {exc.lineno}): {(exc.text or '').strip()}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("prefixes", nargs="*", help="only build sources whose name starts with one of these")
    ap.add_argument("--check", action="store_true", help="syntax-check only, write nothing")
    args = ap.parse_args(argv)

    sources = sorted(SRC_DIR.glob("[0-9][0-9]_*.py"))
    if args.prefixes:
        sources = [s for s in sources if any(s.name.startswith(p) for p in args.prefixes)]
    if not sources:
        print("no notebook sources found", file=sys.stderr)
        return 1

    problems: List[str] = []
    for src in sources:
        cells = parse_source(src)
        problems += check_cells(src.name, cells)
        if args.check:
            print(f"checked {src.name}: {len(cells)} cells")
            continue
        out = OUT_DIR / (src.stem + ".ipynb")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(to_notebook(cells), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {out.relative_to(HERE.parent)} ({len(cells)} cells)")
    for p in problems:
        print("SYNTAX:", p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Reference-value registry for the validation suite.

The single source of truth for reference values, tolerances and citations is
``validation/data/validation_cases.csv`` (columns: case, test, quantity,
reference_value, source, tolerance, notes).  Tests look their tolerance (and,
for constants, their reference value) up here, so the published validation
report (docs/validation.md) always matches what is actually asserted.

``tolerance`` is ``rel:<x>`` (relative) or ``abs:<x>`` (absolute).
``reference_value`` is a number, or text such as ``closed form`` when the
reference is computed inside the test.
"""
from __future__ import annotations

import csv
import functools
import math
import os
from typing import NamedTuple, Optional

DATA_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "validation", "data", "validation_cases.csv")
)


class Ref(NamedTuple):
    case: str
    test: str
    quantity: str
    reference_value: str
    source: str
    tolerance: str
    notes: str

    @property
    def value(self) -> float:
        return float(self.reference_value)

    @property
    def kind(self) -> str:
        return self.tolerance.split(":", 1)[0]

    @property
    def tol(self) -> float:
        return float(self.tolerance.split(":", 1)[1])


@functools.lru_cache(maxsize=1)
def _table():
    with open(DATA_FILE, newline="", encoding="utf-8") as fh:
        rows = [Ref(**{k: (v or "").strip() for k, v in r.items()}) for r in csv.DictReader(fh)]
    return {(r.case, r.quantity): r for r in rows}


def rows():
    return list(_table().values())


def ref(case: str, quantity: str) -> Ref:
    try:
        return _table()[(case, quantity)]
    except KeyError:
        raise KeyError(f"No entry ({case!r}, {quantity!r}) in {DATA_FILE}") from None


def assert_ref(actual, case: str, quantity: str, expected: Optional[float] = None):
    """Assert ``actual`` matches the reference (CSV value, or ``expected``) within the CSV tolerance."""
    r = ref(case, quantity)
    exp = r.value if expected is None else float(expected)
    actual = float(actual)
    if r.kind == "rel":
        ok = abs(actual - exp) <= r.tol * abs(exp) + 1e-300
    elif r.kind == "abs":
        ok = abs(actual - exp) <= r.tol
    else:  # pragma: no cover
        raise ValueError(f"bad tolerance {r.tolerance!r}")
    assert ok and math.isfinite(actual), (
        f"{case}/{quantity}: got {actual!r}, expected {exp!r} (tolerance {r.tolerance}; source: {r.source})"
    )

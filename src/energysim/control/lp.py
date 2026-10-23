# -*- coding: utf-8 -*-
"""
Tiny sparse (MI)LP builder on top of :func:`scipy.optimize.milp` (HiGHS).

Only what the control layer needs: named variable groups with bounds,
integrality and linear cost, and two-sided linear constraint rows that are
added vectorised over a horizon.  scipy is imported lazily so that
``import energysim.control`` works without it.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[float, Sequence[float], np.ndarray]

#: Solver status codes reported by controllers (``solver_status`` output).
STATUS_OPTIMAL = 0
STATUS_LIMIT = 1          # iteration / time limit reached
STATUS_INFEASIBLE = 2
STATUS_UNBOUNDED = 3
STATUS_ERROR = 4          # numerical / other solver failure
STATUS_NOT_SOLVED = -1    # no decision taken yet

STATUS_TEXT = {
    STATUS_NOT_SOLVED: "not solved",
    STATUS_OPTIMAL: "optimal",
    STATUS_LIMIT: "iteration or time limit",
    STATUS_INFEASIBLE: "infeasible",
    STATUS_UNBOUNDED: "unbounded",
    STATUS_ERROR: "solver error",
}


def require_scipy():
    """Import and return ``scipy.optimize``; raise a helpful ImportError otherwise."""
    try:
        import scipy.optimize as so  # noqa: F401
        from scipy.optimize import milp  # noqa: F401  (scipy >= 1.9)
    except ImportError as exc:  # pragma: no cover - exercised only without scipy
        raise ImportError(
            "energysim.control needs scipy >= 1.9 (HiGHS via scipy.optimize.milp). "
            "Install it with: pip install scipy  (or pip install energysim[control])"
        ) from exc
    return so


@dataclass
class LPResult:
    """Outcome of :meth:`LinearProblem.solve`."""
    status: int
    x: Optional[np.ndarray]
    objective: float
    message: str = ""
    solve_time_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OPTIMAL and self.x is not None


class LinearProblem:
    """Incrementally built ``min c'x  s.t.  lo <= A x <= hi,  lb <= x <= ub``."""

    def __init__(self) -> None:
        self._lb: List[np.ndarray] = []
        self._ub: List[np.ndarray] = []
        self._cost: List[np.ndarray] = []
        self._int: List[np.ndarray] = []
        self.n = 0
        self.groups: Dict[str, np.ndarray] = {}
        self._rows: List[np.ndarray] = []
        self._cols: List[np.ndarray] = []
        self._vals: List[np.ndarray] = []
        self._lo: List[np.ndarray] = []
        self._hi: List[np.ndarray] = []
        self.m = 0
        self._extra_cost: Dict[int, float] = {}
        self.constant = 0.0

    # ------------------------------------------------------------------
    def add_variables(self, name: str, n: int, lb: ArrayLike = 0.0, ub: ArrayLike = np.inf,
                      cost: ArrayLike = 0.0, integer: bool = False) -> np.ndarray:
        """Add ``n`` variables and return their column indices."""
        if name in self.groups:
            raise ValueError(f"Variable group '{name}' already exists.")
        n = int(n)
        idx = np.arange(self.n, self.n + n)
        self._lb.append(np.broadcast_to(np.asarray(lb, dtype=float), (n,)).copy())
        self._ub.append(np.broadcast_to(np.asarray(ub, dtype=float), (n,)).copy())
        self._cost.append(np.broadcast_to(np.asarray(cost, dtype=float), (n,)).copy())
        self._int.append(np.full(n, 1 if integer else 0, dtype=np.uint8))
        self.n += n
        self.groups[name] = idx
        return idx

    def add_cost(self, idx: Iterable[int], coef: ArrayLike) -> None:
        """Add ``coef`` to the cost of the variables ``idx`` (broadcast)."""
        idx = np.atleast_1d(np.asarray(idx, dtype=int))
        coef = np.broadcast_to(np.asarray(coef, dtype=float), idx.shape)
        for i, c in zip(idx, coef):
            self._extra_cost[int(i)] = self._extra_cost.get(int(i), 0.0) + float(c)

    def add_rows(self, terms: Sequence[Tuple[np.ndarray, ArrayLike]], lo: ArrayLike, hi: ArrayLike,
                 n: Optional[int] = None) -> np.ndarray:
        """Add ``n`` rows ``lo[k] <= sum_j coef_j[k] * x[idx_j[k]] <= hi[k]``.

        ``terms`` is a list of ``(idx, coef)`` with ``idx`` an int array of length ``n``
        (one column per row) or a scalar column used in every row.
        """
        if n is None:
            n = max((np.size(i) for i, _ in terms), default=0)
            if n == 0:
                n = np.size(lo)
        rows = np.arange(self.m, self.m + n)
        for idx, coef in terms:
            idx = np.broadcast_to(np.asarray(idx, dtype=int), (n,))
            coef = np.broadcast_to(np.asarray(coef, dtype=float), (n,))
            mask = coef != 0.0
            self._rows.append(rows[mask])
            self._cols.append(idx[mask])
            self._vals.append(coef[mask])
        self._lo.append(np.broadcast_to(np.asarray(lo, dtype=float), (n,)).copy())
        self._hi.append(np.broadcast_to(np.asarray(hi, dtype=float), (n,)).copy())
        self.m += n
        return rows

    def add_eq(self, terms, rhs, n=None):
        return self.add_rows(terms, rhs, rhs, n)

    def add_le(self, terms, rhs, n=None):
        return self.add_rows(terms, -np.inf, rhs, n)

    def add_ge(self, terms, rhs, n=None):
        return self.add_rows(terms, rhs, np.inf, n)

    # ------------------------------------------------------------------
    @property
    def has_integers(self) -> bool:
        return any(a.any() for a in self._int)

    def cost_vector(self) -> np.ndarray:
        c = np.concatenate(self._cost) if self._cost else np.zeros(0)
        for i, v in self._extra_cost.items():
            c[i] += v
        return c

    def solve(self, time_limit: Optional[float] = None, mip_rel_gap: Optional[float] = None,
              relax_integers: bool = False) -> LPResult:
        """Solve with HiGHS through :func:`scipy.optimize.milp`."""
        so = require_scipy()
        from scipy.sparse import coo_matrix

        c = self.cost_vector()
        lb = np.concatenate(self._lb) if self._lb else np.zeros(0)
        ub = np.concatenate(self._ub) if self._ub else np.zeros(0)
        integrality = np.concatenate(self._int) if self._int else np.zeros(0, dtype=np.uint8)
        if relax_integers:
            integrality = np.zeros_like(integrality)
        constraints = None
        if self.m:
            A = coo_matrix(
                (np.concatenate(self._vals) if self._vals else np.zeros(0),
                 (np.concatenate(self._rows) if self._rows else np.zeros(0, int),
                  np.concatenate(self._cols) if self._cols else np.zeros(0, int))),
                shape=(self.m, self.n),
            ).tocsr()
            constraints = so.LinearConstraint(A, np.concatenate(self._lo), np.concatenate(self._hi))
        options = {"disp": False}
        if time_limit is not None:
            options["time_limit"] = float(time_limit)
        if mip_rel_gap is not None and integrality.any():
            options["mip_rel_gap"] = float(mip_rel_gap)

        t0 = _time.perf_counter()
        try:
            res = so.milp(c, integrality=integrality, bounds=so.Bounds(lb, ub),
                          constraints=constraints, options=options)
        except Exception as exc:  # malformed problem or solver crash
            return LPResult(STATUS_ERROR, None, float("nan"), f"{type(exc).__name__}: {exc}",
                            _time.perf_counter() - t0)
        elapsed = _time.perf_counter() - t0
        status = int(res.status) if res.status in (0, 1, 2, 3) else STATUS_ERROR
        x = None if res.x is None else np.asarray(res.x, dtype=float)
        obj = float(res.fun) + self.constant if res.fun is not None else float("nan")
        return LPResult(status, x, obj, str(res.message), elapsed)

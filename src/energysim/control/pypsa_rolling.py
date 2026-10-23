# -*- coding: utf-8 -*-
"""
Rolling-horizon PyPSA optimisation as a controller.

At every decision the controller maps the decision time to a snapshot of a
multi-period PyPSA network, writes the measurements into the network
(e.g. the measured state of charge into ``state_of_charge_initial``), runs
``n.optimize(snapshots=window)`` over the next ``window`` snapshots (linopy
+ HiGHS by default) and outputs the dispatch of the *first* snapshot only.

Requirements: ``pypsa`` with ``linopy`` and a solver (``highspy``).  They
are imported lazily; a clear ``ImportError`` is raised when missing.

Mappings use ``(component, attribute, element[, scale])`` tuples, where
``component`` is a PyPSA class name (``'StorageUnit'``, ``'Generator'``,
``'Load'``, ``'Link'``...).  For measurements a *static* attribute is set
(``value * scale``); a *time-varying* attribute is written to the first
snapshot of the window.  For set-points the time-varying result of the
first window snapshot is read (``value * scale``).

Notes
-----
* Set ``cyclic_state_of_charge=False`` on storage whose SoC is measured
  (the controller does this for every storage unit named in a
  ``state_of_charge_initial`` measurement).
* Snapshot spacing defaults to the network's snapshot frequency (or
  ``snapshot_seconds``); decision time ``t`` maps to snapshot
  ``start_snapshot + round((t - start_time) / snapshot_seconds)``.
* The PyPSA network is deep-copied at construction and restored at every
  ``init()`` so runs are repeatable.
"""

from __future__ import annotations

import copy as _copy
import math
import time as _time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..base import SimulatorVariableError
from .base import Controller, FALLBACK_NONE, FALLBACK_PREVIOUS_PLAN
from .lp import STATUS_ERROR, STATUS_INFEASIBLE, STATUS_LIMIT, STATUS_OPTIMAL

_LIST_NAMES = {
    "storage_units": "StorageUnit", "generators": "Generator", "loads": "Load", "links": "Link",
    "stores": "Store", "lines": "Line", "buses": "Bus",
}


def _require_pypsa():
    try:
        import pypsa  # noqa: F401
        import linopy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PyPSARollingHorizonController needs pypsa and linopy (and a solver such as highspy). "
            "Install with: pip install pypsa highspy  (or pip install energysim[pypsa])"
        ) from exc
    return pypsa


def _static(n, comp):
    f = getattr(n, "static", None) or getattr(n, "df")
    return f(comp)


def _dynamic(n, comp):
    f = getattr(n, "dynamic", None) or getattr(n, "pnl")
    return f(comp)


class PyPSARollingHorizonController(Controller):
    """Receding-horizon controller around ``pypsa.Network.optimize``.

    Parameters
    ----------
    network : pypsa.Network
        Multi-period network with time series over (at least) the simulated period.
    window : int
        Number of snapshots optimised at each decision.
    setpoints : dict[str, tuple]
        ``{output_name: (component, attribute, element[, scale])}`` read from the
        first window snapshot (e.g. ``{'bess.P_cmd': ('StorageUnit', 'p', 'bess', -1.0)}``).
    measurements : dict[str, tuple], optional
        ``{input_name: (component, attribute, element[, scale])}`` written before
        each solve (e.g. ``{'bess.SoC': ('StorageUnit', 'state_of_charge_initial', 'bess', 4.0)}``).
    snapshot_seconds : float, optional
        Snapshot spacing in seconds (default: inferred from a DatetimeIndex, else 3600).
    """

    def __init__(self, network: Any, window: int, setpoints: Mapping[str, Sequence],
                 measurements: Optional[Mapping[str, Sequence]] = None, snapshot_seconds: Optional[float] = None,
                 start_snapshot: int = 0, solver_name: str = "highs", solver_options: Optional[Mapping] = None,
                 copy: bool = True, name: str = "pypsa_mpc", **kwargs):
        _require_pypsa()
        if int(window) < 1:
            raise ValueError("window must be >= 1.")
        self.network = _copy.deepcopy(network) if copy else network
        self.window = int(window)
        self.solver_name = solver_name
        self.solver_options = dict(solver_options or {})
        self.start_snapshot = int(start_snapshot)
        self._sp_map = {k: self._norm(v, "setpoint", k) for k, v in setpoints.items()}
        self._meas_map = {k: self._norm(v, "measurement", k) for k, v in (measurements or {}).items()}
        if snapshot_seconds is None:
            snapshot_seconds = self._infer_spacing()
        self.snapshot_seconds = float(snapshot_seconds)
        for _, (comp, attr, el, _) in self._meas_map.items():
            if attr == "state_of_charge_initial" and "cyclic_state_of_charge" in _static(self.network, comp):
                _static(self.network, comp).at[el, "cyclic_state_of_charge"] = False
        self._network0 = _copy.deepcopy(self.network)
        kwargs.setdefault("step_size", self.snapshot_seconds)
        super().__init__(name=name, measurement_inputs=list(self._meas_map), setpoint_outputs=list(self._sp_map),
                         **kwargs)

    # ------------------------------------------------------------------
    def _norm(self, spec, kind, key) -> Tuple[str, str, str, float]:
        if not isinstance(spec, (tuple, list)) or len(spec) not in (3, 4):
            raise ValueError(f"{kind} '{key}' must be (component, attribute, element[, scale]).")
        comp, attr, el = spec[0], spec[1], spec[2]
        scale = float(spec[3]) if len(spec) == 4 else 1.0
        comp = _LIST_NAMES.get(comp, comp)
        try:
            static = _static(self.network, comp)
        except Exception:
            raise SimulatorVariableError(f"{kind} '{key}': unknown PyPSA component '{comp}'.") from None
        if el not in static.index:
            raise SimulatorVariableError(f"{kind} '{key}': no {comp} named '{el}'.")
        if kind == "setpoint" and attr not in _dynamic(self.network, comp):
            raise SimulatorVariableError(f"setpoint '{key}': '{attr}' is not a time-varying output of {comp}.")
        if kind == "measurement" and attr not in static.columns and attr not in _dynamic(self.network, comp):
            raise SimulatorVariableError(f"measurement '{key}': {comp} has no attribute '{attr}'.")
        return comp, attr, el, scale

    def _infer_spacing(self) -> float:
        snaps = self.network.snapshots
        if len(snaps) > 1:
            try:
                return float((snaps[1] - snaps[0]).total_seconds())
            except AttributeError:
                return float(snaps[1] - snaps[0]) * 3600.0
        return 3600.0

    def reset_controller(self) -> None:
        self.network = _copy.deepcopy(self._network0)
        self.last_error = None

    def snapshot_index(self, t: float) -> int:
        return self.start_snapshot + int(round((float(t) - self.start_time) / self.snapshot_seconds))

    # ------------------------------------------------------------------
    def decide(self, t, measurements, forecasts):
        n = self.network
        k = self.snapshot_index(t)
        snaps = n.snapshots
        if not 0 <= k < len(snaps):
            self.last_error = f"time {t} maps to snapshot {k}, outside the network's {len(snaps)} snapshots"
            return {"solver_status": STATUS_ERROR, "fallback": FALLBACK_PREVIOUS_PLAN}
        window = snaps[k:k + self.window]
        for key, (comp, attr, el, scale) in self._meas_map.items():
            val = measurements.get(key)
            if val is None or not math.isfinite(val):
                continue
            static = _static(n, comp)
            if attr in static.columns:
                static.at[el, attr] = val * scale
            else:
                _dynamic(n, comp)[attr].loc[window[0], el] = val * scale
        t0 = _time.perf_counter()
        try:
            status, condition = n.optimize(snapshots=window, solver_name=self.solver_name,
                                           log_to_console=False, **self.solver_options)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return {"solver_status": STATUS_ERROR, "solve_time_s": _time.perf_counter() - t0,
                    "fallback": FALLBACK_PREVIOUS_PLAN}
        elapsed = _time.perf_counter() - t0
        if status != "ok":
            cond = str(condition).lower()
            code = STATUS_INFEASIBLE if "infeasible" in cond else (STATUS_LIMIT if "limit" in cond else STATUS_ERROR)
            self.last_error = f"{status}: {condition}"
            return {"solver_status": code, "solve_time_s": elapsed, "fallback": FALLBACK_PREVIOUS_PLAN}
        out: Dict[str, Any] = {}
        for key, (comp, attr, el, scale) in self._sp_map.items():
            out[key] = float(_dynamic(n, comp)[attr].loc[window[0], el]) * scale
        out.update({"solver_status": STATUS_OPTIMAL, "solve_time_s": elapsed,
                    "objective": float(getattr(n, "objective", float("nan"))), "fallback": FALLBACK_NONE})
        return out

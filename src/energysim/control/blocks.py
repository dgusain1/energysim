# -*- coding: utf-8 -*-
"""
LP / MILP building blocks for optimisation-based controllers.

A *block* adds decision variables, linear constraints and cost terms for one
asset over a horizon of ``N`` intervals of length ``dt`` and injects its
flows into per-carrier energy balances (supply positive, consumption
negative).  :class:`~energysim.control.MPCController` assembles the blocks
into one problem and adds the balance rows ``sum(flows) = 0`` per carrier
and interval.

Units
-----
* power-like flows: MW for ``electricity`` / ``heat`` / ``gas`` / ``cooling``,
  kg/h for ``hydrogen`` (see :data:`CARRIER_UNITS`);
* stored energy: MWh (kg for hydrogen); state of charge as a fraction 0..1;
* prices: EUR per MWh (EUR per kg for hydrogen), demand charges EUR/MW,
  ramp limits MW per hour, self-discharge fraction per hour;
* ``dt`` is given in seconds and converted with ``dt_h = dt / 3600``.

Time-varying parameters (prices, demand, COP, availability, limits) accept a
number, an array (at least ``N`` long), a forecast key (``str``, looked up in
the controller's forecasts, else used as a persistence forecast of the
measurement input of that name), a :class:`~energysim.control.Forecast` or a
callable ``f(times) -> array``.

:func:`block_from_component` maps energysim model components to blocks.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .forecast import Forecast
from .lp import LinearProblem

CARRIER_UNITS = {
    "electricity": "MW", "heat": "MW", "gas": "MW", "cooling": "MW", "hydrogen": "kg/h",
}
INF = float("inf")


def carrier_unit(carrier: str) -> str:
    return CARRIER_UNITS.get(carrier, "MW")


# ---------------------------------------------------------------------------
# Planning context
# ---------------------------------------------------------------------------

class PlanContext:
    """Shared state while the blocks build one horizon problem."""

    def __init__(self, lp: LinearProblem, times: np.ndarray, dt: float,
                 forecasts: Optional[Mapping[str, np.ndarray]] = None,
                 measurements: Optional[Mapping[str, float]] = None, co2_price: float = 0.0):
        self.lp = lp
        self.times = np.asarray(times, dtype=float)
        self.N = int(self.times.size)
        self.dt = float(dt)
        self.dt_h = self.dt / 3600.0
        self.forecasts = dict(forecasts or {})
        self.measurements = dict(measurements or {})
        self.co2_price = float(co2_price)
        self.balance_terms: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
        self.balance_const: Dict[str, np.ndarray] = {}
        self.emission_terms: List[Tuple[np.ndarray, np.ndarray]] = []

    def series(self, spec: Any, label: str = "value") -> np.ndarray:
        """Resolve a time-varying parameter to an array of length ``N``."""
        N = self.N
        if spec is None:
            return np.zeros(N)
        if isinstance(spec, str):
            if spec in self.forecasts:
                arr = np.asarray(self.forecasts[spec], dtype=float)
            elif spec in self.measurements:
                arr = np.full(N, float(self.measurements[spec]))
            else:
                raise KeyError(
                    f"'{label}' refers to '{spec}', which is neither a forecast "
                    f"({sorted(self.forecasts)}) nor an input ({sorted(self.measurements)})."
                )
        elif isinstance(spec, Forecast):
            arr = np.asarray(spec(float(self.times[0]), self.times, self.measurements), dtype=float)
        elif callable(spec):
            arr = np.asarray(spec(self.times), dtype=float)
        else:
            arr = np.asarray(spec, dtype=float)
        if arr.ndim == 0:
            return np.full(N, float(arr))
        arr = arr.ravel()
        if arr.size < N:
            raise ValueError(f"'{label}' has {arr.size} values but the horizon needs {N}.")
        arr = arr[:N]
        if np.isnan(arr).any():
            raise ValueError(f"'{label}' contains NaN values.")
        return arr

    def flow(self, carrier: str, idx: np.ndarray, coef: Any = 1.0) -> None:
        """Add ``coef * x[idx]`` (supply positive) to the carrier balance."""
        coef = np.broadcast_to(np.asarray(coef, dtype=float), (self.N,)).copy()
        self.balance_terms.setdefault(carrier, []).append((np.asarray(idx, dtype=int), coef))

    def const_flow(self, carrier: str, values: Any) -> None:
        """Add a fixed flow (e.g. ``-demand``) to the carrier balance."""
        values = np.broadcast_to(np.asarray(values, dtype=float), (self.N,))
        self.balance_terms.setdefault(carrier, [])
        self.balance_const[carrier] = self.balance_const.get(carrier, np.zeros(self.N)) + values

    def cost(self, idx: np.ndarray, coef: Any) -> None:
        self.lp.add_cost(idx, coef)

    def emit(self, idx: np.ndarray, t_per_unit: Any) -> None:
        """Register CO2 emissions in tonnes per unit of ``x[idx]`` (priced at ``co2_price``)."""
        coef = np.broadcast_to(np.asarray(t_per_unit, dtype=float), np.shape(idx)).copy()
        if not np.any(coef):
            return
        self.emission_terms.append((np.asarray(idx, dtype=int), coef))
        if self.co2_price:
            self.lp.add_cost(idx, coef * self.co2_price)


# ---------------------------------------------------------------------------
# Block base class
# ---------------------------------------------------------------------------

class Block:
    """Base class: one asset in the optimisation problem.

    Subclasses implement :meth:`build` and :meth:`extract`; stateful blocks
    also :meth:`initial_state`, :meth:`state_from_measurements` and
    :meth:`predict`.  Variable names exposed by the controller are prefixed
    with ``"<block name>."``.
    """

    def __init__(self, name: str):
        if not name or "." in name:
            raise ValueError(f"Block name must be a non-empty string without '.', got {name!r}.")
        self.name = name
        self._idx: Dict[str, np.ndarray] = {}

    # --- interface -------------------------------------------------------
    @property
    def outputs(self) -> Dict[str, str]:
        """``{output_variable: unit}`` (set-points) exposed by the controller."""
        return {}

    @property
    def measurements(self) -> Dict[str, str]:
        """``{input_variable: unit}`` the block reads (e.g. ``SoC``)."""
        return {}

    def initial_state(self) -> Dict[str, float]:
        return {}

    def initial_measurements(self) -> Dict[str, float]:
        return {}

    def state_from_measurements(self, state: Dict[str, float], measurements: Mapping[str, float],
                                fresh: set) -> Dict[str, float]:
        return dict(state)

    def predict(self, state: Dict[str, float], setpoints: Mapping[str, float], seconds: float) -> Dict[str, float]:
        """State after applying ``setpoints`` for ``seconds`` (delay compensation)."""
        return dict(state)

    def build(self, ctx: PlanContext, state: Dict[str, float]) -> None:
        raise NotImplementedError

    def extract(self, x: np.ndarray, ctx: PlanContext) -> Dict[str, np.ndarray]:
        """Planned trajectories ``{output_variable: array(N)}`` from the solution vector."""
        return {}

    # --- helpers ---------------------------------------------------------
    def key(self, var: str) -> str:
        return f"{self.name}.{var}"

    def _var(self, ctx: PlanContext, tag: str, n: int, lb=0.0, ub=INF, cost=0.0, integer=False):
        idx = ctx.lp.add_variables(f"{self.name}.{tag}", n, lb, ub, cost, integer)
        self._idx[tag] = idx
        return idx

    def _ramp_rows(self, ctx: PlanContext, p: np.ndarray, p_prev: Optional[float], ramp: float) -> None:
        lim = ramp * ctx.dt_h
        if ctx.N > 1:
            ctx.lp.add_rows([(p[1:], 1.0), (p[:-1], -1.0)], -lim, lim)
        if p_prev is not None and math.isfinite(p_prev):
            ctx.lp.add_rows([(p[:1], 1.0)], p_prev - lim, p_prev + lim)

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

class StorageBlock(Block):
    """Generic storage (battery, thermal tank, hydrogen tank ...).

    Dynamics (``e`` stored energy, ``dt_h`` hours)::

        e[k+1] = (1 - self_discharge * dt_h) * e[k] + eta_c * pc[k] * dt_h - pd[k] / eta_d * dt_h

    Parameters
    ----------
    capacity : float
        Usable capacity in MWh (kg for hydrogen).
    p_charge_max, p_discharge_max : float
        Power limits (MW; kg/h for hydrogen).
    eta_c, eta_d : float
        Charge / discharge efficiencies.
    soc_min, soc_max, soc0 : float
        SoC window and initial SoC (fractions).
    self_discharge : float
        Fraction of the stored energy lost per hour.
    setpoint : str, default 'P_cmd'
        Name of the set-point output; ``charge_positive`` gives its sign.
    soc_measurement : str or None, default 'SoC'
        Name of the measured SoC input (fraction); ``None`` = open loop.
    soc_terminal : float, optional
        Minimum SoC at the end of the horizon.
    terminal_value : float, optional
        Value (EUR/MWh) of energy left at the end of the horizon.
    cycle_cost : float, default 0
        Throughput cost (EUR/MWh charged + discharged).
    use_binary : bool, default False
        Forbid simultaneous charge and discharge with a binary (MILP).
    """

    def __init__(self, name: str, capacity: float, p_charge_max: float, p_discharge_max: Optional[float] = None,
                 eta_c: float = 1.0, eta_d: float = 1.0, soc_min: float = 0.0, soc_max: float = 1.0,
                 soc0: float = 0.5, self_discharge: float = 0.0, carrier: str = "electricity",
                 setpoint: str = "P_cmd", charge_positive: bool = True, soc_measurement: Optional[str] = "SoC",
                 soc_terminal: Optional[float] = None, terminal_value: Optional[float] = None,
                 cycle_cost: float = 0.0, use_binary: bool = False):
        super().__init__(name)
        if capacity <= 0:
            raise ValueError(f"{name}: capacity must be > 0.")
        if not (0 < eta_c <= 1 and 0 < eta_d <= 1):
            raise ValueError(f"{name}: efficiencies must be in (0, 1].")
        if not (0 <= soc_min <= soc_max <= 1):
            raise ValueError(f"{name}: need 0 <= soc_min <= soc_max <= 1.")
        self.capacity = float(capacity)
        self.p_charge_max = float(p_charge_max)
        self.p_discharge_max = float(p_charge_max if p_discharge_max is None else p_discharge_max)
        self.eta_c, self.eta_d = float(eta_c), float(eta_d)
        self.soc_min, self.soc_max, self.soc0 = float(soc_min), float(soc_max), float(soc0)
        self.self_discharge = float(self_discharge)
        self.carrier = carrier
        self.setpoint = setpoint
        self.charge_positive = bool(charge_positive)
        self.soc_measurement = soc_measurement
        self.soc_terminal = soc_terminal
        self.terminal_value = terminal_value
        self.cycle_cost = float(cycle_cost)
        self.use_binary = bool(use_binary)

    @property
    def outputs(self):
        return {self.setpoint: carrier_unit(self.carrier)}

    @property
    def measurements(self):
        return {self.soc_measurement: "1"} if self.soc_measurement else {}

    def initial_state(self):
        return {"soc": self.soc0}

    def initial_measurements(self):
        return {self.soc_measurement: self.soc0} if self.soc_measurement else {}

    def state_from_measurements(self, state, measurements, fresh):
        state = dict(state)
        if self.soc_measurement:
            k = self.key(self.soc_measurement)
            if k in fresh and k in measurements and math.isfinite(measurements[k]):
                state["soc"] = float(measurements[k])
        return state

    def _split(self, p: float) -> Tuple[float, float]:
        p = p if self.charge_positive else -p
        return max(p, 0.0), max(-p, 0.0)

    def predict(self, state, setpoints, seconds):
        pc, pd = self._split(float(setpoints.get(self.key(self.setpoint), 0.0)))
        pc, pd = min(pc, self.p_charge_max), min(pd, self.p_discharge_max)
        h = seconds / 3600.0
        e = state["soc"] * self.capacity
        e = e * (1.0 - self.self_discharge * h) + self.eta_c * pc * h - pd / self.eta_d * h
        lo = min(self.soc_min, state["soc"]) * self.capacity
        hi = max(self.soc_max, state["soc"]) * self.capacity
        return {"soc": float(np.clip(e, lo, hi) / self.capacity)}

    def build(self, ctx, state):
        N, dt_h, C = ctx.N, ctx.dt_h, self.capacity
        e0 = float(state.get("soc", self.soc0)) * C
        decay = 1.0 - self.self_discharge * dt_h
        k = np.arange(1, N + 1)
        lb = np.minimum(self.soc_min * C, e0 * decay ** k)
        ub = np.full(N, max(self.soc_max * C, e0))
        pc = self._var(ctx, "pc", N, 0.0, self.p_charge_max, self.cycle_cost * dt_h)
        pd = self._var(ctx, "pd", N, 0.0, self.p_discharge_max, self.cycle_cost * dt_h)
        e = self._var(ctx, "e", N + 1, np.r_[e0, lb], np.r_[e0, ub])
        ctx.lp.add_eq([(e[1:], 1.0), (e[:-1], -decay), (pc, -self.eta_c * dt_h), (pd, dt_h / self.eta_d)], 0.0)
        if self.soc_terminal is not None:
            ctx.lp.add_ge([(e[N:], 1.0)], min(float(self.soc_terminal) * C, ub[-1]))
        if self.terminal_value:
            ctx.cost(e[N:], -float(self.terminal_value))
        if self.use_binary:
            z = self._var(ctx, "z", N, 0.0, 1.0, integer=True)
            ctx.lp.add_le([(pc, 1.0), (z, -self.p_charge_max)], 0.0)
            ctx.lp.add_le([(pd, 1.0), (z, self.p_discharge_max)], self.p_discharge_max)
        ctx.flow(self.carrier, pd, 1.0)
        ctx.flow(self.carrier, pc, -1.0)

    def extract(self, x, ctx):
        pc, pd, e = (x[self._idx[t]] for t in ("pc", "pd", "e"))
        p = pc - pd if self.charge_positive else pd - pc
        return {self.setpoint: p, "SoC_plan": e[1:] / self.capacity}


# ---------------------------------------------------------------------------
# Conversion assets
# ---------------------------------------------------------------------------

def _upper_hull(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Concave (upper) envelope of points sorted by x."""
    pts = sorted((float(a), float(b)) for a, b in points)
    hull: List[Tuple[float, float]] = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:  # hull[-1] below chord
                hull.pop()
            else:
                break
        if hull and abs(hull[-1][0] - p[0]) < 1e-12:
            hull[-1] = (p[0], max(hull[-1][1], p[1]))
        else:
            hull.append(p)
    return hull


class ElectrolyzerBlock(Block):
    """Electrolyser: electricity -> hydrogen with a piecewise-linear production curve.

    ``P = p_min * on + sum(seg_j)``, ``H2 = h(p_min) * on + sum(slope_j * seg_j)``
    with ``0 <= seg_j <= width_j * on``.  The curve is replaced by its concave
    envelope (exact for concave curves, slightly optimistic otherwise).  With
    ``use_binary=True`` ``on`` is binary and the minimum turndown ``p_min`` is
    enforced exactly (MILP); otherwise ``on`` is relaxed to [0, 1] (LP).

    Parameters
    ----------
    p_max : float
        Rated electrical power (MW).
    p_min : float
        Minimum turndown (MW).
    curve : sequence of (P_MW, H2_kg_per_h), optional
        Production curve between ``p_min`` and ``p_max``.  Default: constant
        ``spec_consumption`` (kWh/kg).
    """

    def __init__(self, name: str, p_max: float, p_min: float = 0.0,
                 curve: Optional[Sequence[Tuple[float, float]]] = None, spec_consumption: float = 55.0,
                 use_binary: bool = False, setpoint: str = "P_cmd", carrier_in: str = "electricity",
                 carrier_out: str = "hydrogen", standby_cost: float = 0.0):
        super().__init__(name)
        if not (0 <= p_min <= p_max) or p_max <= 0:
            raise ValueError(f"{name}: need 0 <= p_min <= p_max and p_max > 0.")
        self.p_max, self.p_min = float(p_max), float(p_min)
        if curve is None:
            curve = [(p, p * 1000.0 / spec_consumption) for p in (self.p_min, self.p_max)]
        pts = [(p, h) for p, h in curve if self.p_min - 1e-9 <= p <= self.p_max + 1e-9]
        if len(pts) < 2 and self.p_min < self.p_max:
            raise ValueError(f"{name}: the curve needs at least two points within [p_min, p_max].")
        hull = _upper_hull(pts) if pts else [(self.p_min, 0.0)]
        if abs(hull[0][0] - self.p_min) > 1e-9:
            hull.insert(0, (self.p_min, float(np.interp(self.p_min, *zip(*hull)))))
        self.h_min = hull[0][1]
        self.widths = np.array([b[0] - a[0] for a, b in zip(hull[:-1], hull[1:])])
        self.slopes = np.array([(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(hull[:-1], hull[1:])])
        self.hull = hull
        self.use_binary = bool(use_binary)
        self.setpoint = setpoint
        self.carrier_in, self.carrier_out = carrier_in, carrier_out
        self.standby_cost = float(standby_cost)

    @property
    def outputs(self):
        return {self.setpoint: carrier_unit(self.carrier_in)}

    def build(self, ctx, state):
        N = ctx.N
        on = self._var(ctx, "on", N, 0.0, 1.0, self.standby_cost * ctx.dt_h, integer=self.use_binary)
        p_terms = [(on, self.p_min)] if self.p_min else []
        h_terms = [(on, self.h_min)] if self.h_min else []
        for j, (w, s) in enumerate(zip(self.widths, self.slopes)):
            seg = self._var(ctx, f"seg{j}", N, 0.0, w)
            ctx.lp.add_le([(seg, 1.0), (on, -w)], 0.0)
            p_terms.append((seg, 1.0))
            h_terms.append((seg, s))
        for idx, c in p_terms:
            ctx.flow(self.carrier_in, idx, -c)
        for idx, c in h_terms:
            ctx.flow(self.carrier_out, idx, c)
        self._p_terms, self._h_terms = p_terms, h_terms

    def extract(self, x, ctx):
        p = sum(x[i] * c for i, c in self._p_terms) if self._p_terms else np.zeros(ctx.N)
        h = sum(x[i] * c for i, c in self._h_terms) if self._h_terms else np.zeros(ctx.N)
        return {self.setpoint: np.asarray(p, dtype=float), "H2_plan": np.asarray(h, dtype=float)}


class HeatPumpBlock(Block):
    """Heat pump: ``P_elec = Q_heat / COP``; COP fixed or time-varying (forecast)."""

    def __init__(self, name: str, q_max: float, cop: Any = 3.0, q_min: float = 0.0,
                 setpoint: str = "Q_heat_cmd_MW", carrier_in: str = "electricity", carrier_out: str = "heat"):
        super().__init__(name)
        self.q_max, self.q_min, self.cop = float(q_max), float(q_min), cop
        self.setpoint = setpoint
        self.carrier_in, self.carrier_out = carrier_in, carrier_out

    @property
    def outputs(self):
        return {self.setpoint: carrier_unit(self.carrier_out)}

    def build(self, ctx, state):
        cop = ctx.series(self.cop, f"{self.name}.cop")
        if np.any(cop <= 0):
            raise ValueError(f"{self.name}: COP must be > 0.")
        q = self._var(ctx, "q", ctx.N, self.q_min, self.q_max)
        ctx.flow(self.carrier_out, q, 1.0)
        ctx.flow(self.carrier_in, q, -1.0 / cop)

    def extract(self, x, ctx):
        return {self.setpoint: x[self._idx["q"]]}


class BoilerBlock(Block):
    """Boiler: ``heat = efficiency * fuel``.

    ``fuel`` is a carrier name whose balance supplies the input (e.g.
    ``'electricity'`` for an electric boiler) or ``None`` for fuel bought at
    ``fuel_price`` (EUR/MWh fuel).  ``co2_intensity`` is t CO2 per MWh fuel.
    The set-point is the fuel / electric input power.
    """

    def __init__(self, name: str, p_max: float, efficiency: float = 1.0, fuel: Optional[str] = "electricity",
                 fuel_price: Any = 0.0, co2_intensity: float = 0.0, ramp: Optional[float] = None,
                 setpoint: str = "P_cmd", carrier_out: str = "heat"):
        super().__init__(name)
        self.p_max, self.efficiency = float(p_max), float(efficiency)
        self.fuel, self.fuel_price, self.co2_intensity = fuel, fuel_price, float(co2_intensity)
        self.ramp = None if ramp is None else float(ramp)
        self.setpoint, self.carrier_out = setpoint, carrier_out

    @property
    def outputs(self):
        return {self.setpoint: "MW"}

    def initial_state(self):
        return {"p_prev": float("nan")}

    def predict(self, state, setpoints, seconds):
        return {"p_prev": float(setpoints.get(self.key(self.setpoint), 0.0))}

    def build(self, ctx, state):
        p = self._var(ctx, "p", ctx.N, 0.0, self.p_max)
        price = ctx.series(self.fuel_price, f"{self.name}.fuel_price")
        if np.any(price):
            ctx.cost(p, price * ctx.dt_h)
        ctx.emit(p, self.co2_intensity * ctx.dt_h)
        if self.fuel:
            ctx.flow(self.fuel, p, -1.0)
        ctx.flow(self.carrier_out, p, self.efficiency)
        if self.ramp is not None:
            self._ramp_rows(ctx, p, state.get("p_prev"), self.ramp)

    def extract(self, x, ctx):
        return {self.setpoint: x[self._idx["p"]]}


class GeneratorBlock(Block):
    """Dispatchable generator (optionally CHP) with unit-commitment relaxation.

    ``p_min * on <= p <= p_max * on``; cost ``cost * p + no_load_cost * on``
    (EUR/MWh, EUR/h); CO2 ``co2_intensity * p + no_load_co2 * on`` (t/MWh,
    t/h); ramp limit in MW/h; recoverable heat ``<= heat_ratio * p``.  ``on``
    is binary with ``use_binary=True`` (MILP) else relaxed to [0, 1].
    """

    def __init__(self, name: str, p_max: float, p_min: float = 0.0, cost: Any = 0.0, no_load_cost: float = 0.0,
                 ramp: Optional[float] = None, co2_intensity: float = 0.0, no_load_co2: float = 0.0,
                 heat_ratio: float = 0.0, carrier: str = "electricity", heat_carrier: str = "heat",
                 use_binary: bool = False, setpoint: str = "P_cmd"):
        super().__init__(name)
        self.p_max, self.p_min = float(p_max), float(p_min)
        self.cost_spec, self.no_load_cost = cost, float(no_load_cost)
        self.ramp = None if ramp is None else float(ramp)
        self.co2_intensity, self.no_load_co2 = float(co2_intensity), float(no_load_co2)
        self.heat_ratio = float(heat_ratio)
        self.carrier, self.heat_carrier = carrier, heat_carrier
        self.use_binary = bool(use_binary)
        self.setpoint = setpoint

    @property
    def outputs(self):
        return {self.setpoint: carrier_unit(self.carrier)}

    def initial_state(self):
        return {"p_prev": float("nan")}

    def predict(self, state, setpoints, seconds):
        return {"p_prev": float(setpoints.get(self.key(self.setpoint), 0.0))}

    def build(self, ctx, state):
        N, dt_h = ctx.N, ctx.dt_h
        cost = ctx.series(self.cost_spec, f"{self.name}.cost")
        p = self._var(ctx, "p", N, 0.0, self.p_max, cost * dt_h)
        on = self._var(ctx, "on", N, 0.0, 1.0, self.no_load_cost * dt_h, integer=self.use_binary)
        ctx.lp.add_le([(p, 1.0), (on, -self.p_max)], 0.0)
        if self.p_min:
            ctx.lp.add_ge([(p, 1.0), (on, -self.p_min)], 0.0)
        ctx.emit(p, self.co2_intensity * dt_h)
        ctx.emit(on, self.no_load_co2 * dt_h)
        ctx.flow(self.carrier, p, 1.0)
        if self.heat_ratio > 0:
            q = self._var(ctx, "q", N, 0.0)
            ctx.lp.add_le([(q, 1.0), (p, -self.heat_ratio)], 0.0)
            ctx.flow(self.heat_carrier, q, 1.0)
        if self.ramp is not None:
            self._ramp_rows(ctx, p, state.get("p_prev"), self.ramp)

    def extract(self, x, ctx):
        return {self.setpoint: x[self._idx["p"]]}


# ---------------------------------------------------------------------------
# Network-side blocks
# ---------------------------------------------------------------------------

class GridBlock(Block):
    """Market / grid connection: import and export at (time-varying) prices.

    ``peak_cost`` (EUR/MW) prices the maximum import over the horizon (demand
    charge).  ``co2_intensity`` (t/MWh) applies to imports.  Make sure
    ``price_export <= price_import`` or limit the exchange, otherwise the
    problem is unbounded.  Works for any carrier (e.g. a hydrogen offtaker
    with ``carrier='hydrogen', import_limit=0, price_export=4.0``).
    """

    def __init__(self, name: str = "grid", import_limit: Any = INF, export_limit: Any = INF,
                 price_import: Any = 0.0, price_export: Any = 0.0, peak_cost: float = 0.0,
                 co2_intensity: Any = 0.0, carrier: str = "electricity"):
        super().__init__(name)
        self.import_limit, self.export_limit = import_limit, export_limit
        self.price_import, self.price_export = price_import, price_export
        self.peak_cost = float(peak_cost)
        self.co2_intensity = co2_intensity
        self.carrier = carrier

    @property
    def outputs(self):
        u = carrier_unit(self.carrier)
        return {"P_import": u, "P_export": u}

    def build(self, ctx, state):
        N, dt_h = ctx.N, ctx.dt_h
        imp_lim = ctx.series(self.import_limit, f"{self.name}.import_limit")
        exp_lim = ctx.series(self.export_limit, f"{self.name}.export_limit")
        pi = ctx.series(self.price_import, f"{self.name}.price_import")
        pe = ctx.series(self.price_export, f"{self.name}.price_export")
        imp = self._var(ctx, "imp", N, 0.0, imp_lim, pi * dt_h)
        exp = self._var(ctx, "exp", N, 0.0, exp_lim, -pe * dt_h)
        ctx.emit(imp, ctx.series(self.co2_intensity, f"{self.name}.co2_intensity") * dt_h)
        ctx.flow(self.carrier, imp, 1.0)
        ctx.flow(self.carrier, exp, -1.0)
        if self.peak_cost:
            pk = self._var(ctx, "peak", 1, 0.0, INF, self.peak_cost)
            ctx.lp.add_le([(imp, 1.0), (np.full(N, pk[0]), -1.0)], 0.0)

    def extract(self, x, ctx):
        return {"P_import": x[self._idx["imp"]], "P_export": x[self._idx["exp"]]}


class LoadBlock(Block):
    """Demand (forecast) with an optional shiftable share.

    The fixed part ``(1 - f) * d`` must be served when it occurs; the
    shiftable part ``f * d`` may be served at any interval of the horizon
    (``0 <= s <= shift_max``, total energy conserved within the horizon).
    The output ``P_set`` is the planned load (a set-point for flexible loads).
    """

    def __init__(self, name: str, demand: Any, carrier: str = "electricity", shiftable_fraction: float = 0.0,
                 shift_max: Optional[float] = None):
        super().__init__(name)
        if not 0.0 <= shiftable_fraction <= 1.0:
            raise ValueError(f"{name}: shiftable_fraction must be in [0, 1].")
        self.demand, self.carrier = demand, carrier
        self.shiftable_fraction = float(shiftable_fraction)
        self.shift_max = shift_max

    @property
    def outputs(self):
        return {"P_set": carrier_unit(self.carrier)}

    def build(self, ctx, state):
        d = ctx.series(self.demand, f"{self.name}.demand")
        self._d = d
        f = self.shiftable_fraction
        ctx.const_flow(self.carrier, -(1.0 - f) * d)
        if f > 0:
            smax = self.shift_max if self.shift_max is not None else 2.0 * f * float(np.max(np.abs(d)) or 0.0)
            s = self._var(ctx, "s", ctx.N, 0.0, smax)
            total = f * float(d.sum())
            ctx.lp.add_rows([(s[i:i + 1], 1.0) for i in range(ctx.N)], total, total, n=1)
            ctx.flow(self.carrier, s, -1.0)

    def extract(self, x, ctx):
        base = (1.0 - self.shiftable_fraction) * self._d
        if "s" in self._idx:
            base = base + x[self._idx["s"]]
        return {"P_set": base}


class RenewableBlock(Block):
    """Variable renewable source: ``0 <= p <= available`` (``== available`` if not curtailable)."""

    def __init__(self, name: str, available: Any, curtailable: bool = True, carrier: str = "electricity",
                 cost: Any = 0.0):
        super().__init__(name)
        self.available, self.curtailable = available, bool(curtailable)
        self.carrier, self.cost_spec = carrier, cost

    @property
    def outputs(self):
        return {"P_set": carrier_unit(self.carrier)}

    def build(self, ctx, state):
        av = np.maximum(ctx.series(self.available, f"{self.name}.available"), 0.0)
        cost = ctx.series(self.cost_spec, f"{self.name}.cost")
        p = self._var(ctx, "p", ctx.N, 0.0 if self.curtailable else av, av, cost * ctx.dt_h)
        ctx.flow(self.carrier, p, 1.0)

    def extract(self, x, ctx):
        return {"P_set": x[self._idx["p"]]}


# ---------------------------------------------------------------------------
# Component -> block registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Callable[..., Block]] = {}


def register_block(class_name: str):
    """Decorator registering ``factory(component, name, **overrides) -> Block`` for a component class name."""
    def deco(factory):
        _REGISTRY[class_name] = factory
        return factory
    return deco


def block_from_component(component: Any, name: Optional[str] = None, **overrides) -> Block:
    """Build the LP block for an energysim model component.

    Parameters are read from the component (ratings, efficiency curves,
    initial SoC).  ``overrides`` replace derived block arguments (e.g.
    ``soc_terminal=0.5``) or feed factory options (``fuel_price``,
    ``co2_per_mwh_fuel`` for thermal generators, ``use_binary``...).
    """
    if isinstance(component, Block):
        return component
    for cls in type(component).__mro__:
        factory = _REGISTRY.get(cls.__name__)
        if factory is not None:
            return factory(component, name or getattr(component, "name", None) or cls.__name__.lower(), **overrides)
    raise TypeError(
        f"No LP block registered for {type(component).__name__}. Known: {sorted(_REGISTRY)}. "
        f"Pass a Block instance or use register_block()."
    )


def _merge(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(defaults)
    out.update(overrides)
    return out


def _safe_name(name: str) -> str:
    return str(name).replace(".", "_").replace(" ", "_")


@register_block("BatteryBESS")
def _bess_block(c, name, **kw):
    eta_c, eta_d = _sample_bess_efficiency(c)
    return StorageBlock(_safe_name(name), **_merge(dict(
        capacity=c.capacity_mwh * getattr(c, "soh", 1.0), p_charge_max=c.p_max_mw, p_discharge_max=c.p_max_mw,
        eta_c=eta_c, eta_d=eta_d, soc_min=c.soc_min, soc_max=c.soc_max, soc0=c.soc,
        setpoint="P_cmd", charge_positive=True, soc_measurement="SoC"), kw))


def _sample_bess_efficiency(c, load: float = 0.5, seconds: float = 600.0) -> Tuple[float, float]:
    """Effective AC->stored (charge) and stored->AC (discharge) efficiencies at ``load`` x rating."""
    effs = []
    for sign in (1.0, -1.0):
        m = copy.deepcopy(c)
        m.soc = 0.5 * (m.soc_min + m.soc_max)
        soc_before = m.soc
        p = sign * load * m.p_max_mw
        m._inputs_val["P_cmd"] = p
        m.step_model(seconds, 0.0)
        stored = (m.soc - soc_before) * m.capacity_mwh * getattr(m, "soh", 1.0)
        ac = m.p_actual * seconds / 3600.0
        if abs(stored) < 1e-12 or abs(ac) < 1e-12:
            effs.append(0.95)
        else:
            effs.append(stored / ac if sign > 0 else ac / stored)
    return float(min(max(effs[0], 0.5), 1.0)), float(min(max(effs[1], 0.5), 1.0))


@register_block("FlowBattery")
def _flow_block(c, name, **kw):
    eta = math.sqrt(c.stack_eff)
    return StorageBlock(_safe_name(name), **_merge(dict(
        capacity=c.tank_capacity_mwh, p_charge_max=c.stack_p_max_mw, eta_c=eta, eta_d=eta,
        soc_min=c.soc_min, soc_max=c.soc_max, soc0=c.soc, setpoint="P_cmd"), kw))


@register_block("Flywheel")
def _flywheel_block(c, name, **kw):
    eta = math.sqrt(c.eff)
    return StorageBlock(_safe_name(name), **_merge(dict(
        capacity=c.capacity_kwh / 1000.0, p_charge_max=c.p_max, eta_c=eta, eta_d=eta,
        soc_min=c.soc_min, soc_max=c.soc_max, soc0=c.soc, self_discharge=c.self_discharge_rate * 3600.0,
        setpoint="P_cmd"), kw))


@register_block("ThermalStorageTank")
def _tes_block(c, name, **kw):
    return StorageBlock(_safe_name(name), **_merge(dict(
        capacity=c.capacity_mwh, p_charge_max=c.max_flow, carrier="heat", soc_min=0.0, soc_max=1.0,
        soc0=c.soc, self_discharge=c.loss_rate_s * 3600.0, setpoint="Q_cmd_MW"), kw))


def _sample_h2_curve(c, p_min: float, p_max: float, n: int = 8) -> List[Tuple[float, float]]:
    pts = []
    for p in np.linspace(p_min, p_max, n):
        m = copy.deepcopy(c)
        if hasattr(m, "ramp_rate_pct_per_s"):
            m.ramp_rate_pct_per_s = None
        m._inputs_val["P_cmd"] = float(p) * (1.0 + 1e-9)
        m.step_model(1.0, 0.0)
        pts.append((float(p), float(m.m_dot_h2_kg_hr)))
    return pts


@register_block("PEMElectrolyzer")
def _pem_block(c, name, **kw):
    p_min = c.min_turndown * c.p_nominal_mw
    curve = _sample_h2_curve(c, p_min, c.p_nominal_mw)
    return ElectrolyzerBlock(_safe_name(name), **_merge(dict(p_max=c.p_nominal_mw, p_min=p_min, curve=curve), kw))


@register_block("AlkalineElectrolyzer")
def _ael_block(c, name, **kw):
    p_min = c.min_turndown * c.p_nominal_mw
    curve = _sample_h2_curve(c, p_min, c.p_nominal_mw)
    return ElectrolyzerBlock(_safe_name(name), **_merge(dict(p_max=c.p_nominal_mw, p_min=p_min, curve=curve), kw))


@register_block("IndustrialHeatPump")
def _hp_block(c, name, **kw):
    t_snk = c.t_sink_nom + 273.15
    t_src = c.t_source_nom + 273.15
    cop = max(1.2, min(7.0, c.eta_carnot * t_snk / max(5.0, t_snk - t_src)))
    return HeatPumpBlock(_safe_name(name), **_merge(dict(q_max=c.q_max, cop=cop), kw))


@register_block("ElectricBoiler")
def _eboiler_block(c, name, **kw):
    return BoilerBlock(_safe_name(name), **_merge(dict(
        p_max=c.p_max, efficiency=c.eff, fuel="electricity", ramp=c.ramp_rate * 60.0), kw))


@register_block("GasTurbine_OCGT")
def _ocgt_block(c, name, fuel_price: float = 30.0, co2_per_mwh_fuel: float = 0.202, **kw):
    """Relaxed LP: fuel(P) ~ a * on + b * P fitted between MSG and rating; start-up dynamics ignored."""
    def fuel_mw(p):
        return p * c.hr_full * float(c.hr_curve(p / c.p_rated_mw)) / 3600.0
    p0, p1 = c.msg_mw, c.p_rated_mw
    b = (fuel_mw(p1) - fuel_mw(p0)) / (p1 - p0) if p1 > p0 else fuel_mw(p1) / p1
    a = max(0.0, fuel_mw(p0) - b * p0)
    ramp = getattr(c.sm, "ramp_up_rate_min", None)
    return GeneratorBlock(_safe_name(name), **_merge(dict(
        p_max=c.p_rated_mw, p_min=c.msg_mw, cost=fuel_price * b, no_load_cost=fuel_price * a,
        co2_intensity=co2_per_mwh_fuel * b, no_load_co2=co2_per_mwh_fuel * a,
        ramp=None if ramp is None else ramp * 60.0), kw))


@register_block("ReciprocatingGenset")
def _genset_block(c, name, fuel_price: float = 30.0, co2_per_mwh_fuel: float = 0.202, **kw):
    return GeneratorBlock(_safe_name(name), **_merge(dict(
        p_max=c.p_rated, cost=fuel_price / c.eff_elec, co2_intensity=co2_per_mwh_fuel / c.eff_elec,
        heat_ratio=c.eff_th / c.eff_elec), kw))


__all__ = [
    "CARRIER_UNITS", "PlanContext", "Block", "StorageBlock", "ElectrolyzerBlock", "HeatPumpBlock",
    "BoilerBlock", "GeneratorBlock", "GridBlock", "LoadBlock", "RenewableBlock",
    "register_block", "block_from_component",
]

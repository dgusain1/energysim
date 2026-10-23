# -*- coding: utf-8 -*-
"""
Receding-horizon model predictive controller built from LP blocks.

At every decision the controller

1. reads the measured states (e.g. ``bess.SoC``); unmeasured states are
   carried over from its own previous prediction (open loop),
2. predicts the state at the actuation time over the actuation delay with
   the set-points currently applied (delay compensation, see
   :mod:`energysim.control.base`),
3. evaluates the forecasts over ``horizon`` intervals of ``dt`` seconds,
4. builds one LP (MILP if any block uses binaries) with per-carrier energy
   balances and minimises energy cost + demand charges + CO2 cost +
   unserved-energy penalty, solved by HiGHS (``scipy.optimize.milp``),
5. applies only the first interval of the plan.

If the solve fails the controller falls back to the previous plan (shifted
to the current time) or to a rule, and reports it via ``solver_status`` /
``fallback``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from .base import Controller, FALLBACK_NONE, FALLBACK_PREVIOUS_PLAN, FALLBACK_RULE
from .blocks import Block, GridBlock, LoadBlock, PlanContext, block_from_component, carrier_unit
from .lp import LinearProblem, STATUS_OPTIMAL, STATUS_TEXT


@dataclass
class MPCPlan:
    """One solved horizon problem."""
    times: np.ndarray
    status: int
    objective: float
    values: Dict[str, np.ndarray] = field(default_factory=dict)
    solve_time_s: float = 0.0
    message: str = ""
    emissions_t: float = float("nan")

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OPTIMAL

    def __getitem__(self, key: str) -> np.ndarray:
        return self.values[key]


class MPCController(Controller):
    """Economic MPC over a set of :class:`~energysim.control.blocks.Block` objects.

    Parameters
    ----------
    blocks : list[Block]
        Assets of the problem.  Outputs are named ``"<block>.<var>"``
        (e.g. ``bess.P_cmd``), measured states likewise (``bess.SoC``).
    forecasts : dict[str, Forecast]
        Providers referenced by name in block parameters (``'price'``, ``'load'``...).
    horizon : int, default 24
        Number of intervals.
    dt : float, default 900
        Interval length in seconds.
    co2_price : float, default 0
        EUR per tonne CO2.
    unserved_penalty : float or None, default 1e4
        EUR per MWh of unserved demand per carrier; ``None`` makes the
        balances hard (the problem can then be infeasible).
    spill_penalty : float or None
        If set, surplus may be spilled at this cost (EUR/MWh).
    extra_inputs : list[str]
        Additional inputs (e.g. a connected ``price`` signal usable as a
        persistence forecast by name).
    fallback : 'hold' | 'zero' | callable
        Rule used when the solve fails and no previous plan covers the
        current time: keep the previous set-points, use zeros, or call
        ``fallback(t, measurements, forecasts) -> dict``.
    horizon_end : float, optional
        Absolute time beyond which the horizon is not extended (shrinking horizon).
    delay_compensation : bool, default True
        Predict the state at the actuation time (see module docstring).
    time_limit, mip_rel_gap : float, optional
        HiGHS options.
    """

    def __init__(self, blocks: Sequence[Block], forecasts: Optional[Mapping[str, Any]] = None,
                 horizon: int = 24, dt: float = 900.0, co2_price: float = 0.0,
                 unserved_penalty: Optional[float] = 1e4, spill_penalty: Optional[float] = None,
                 extra_inputs: Sequence[str] = (), fallback: Union[str, Callable] = "hold",
                 horizon_end: Optional[float] = None, delay_compensation: bool = True,
                 time_limit: Optional[float] = None, mip_rel_gap: Optional[float] = 1e-6,
                 name: str = "mpc", decide_at_init: bool = True, **kwargs):
        blocks = list(blocks)
        names = [b.name for b in blocks]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise ValueError(f"Duplicate block names: {sorted(dup)}")
        if int(horizon) < 1:
            raise ValueError("horizon must be >= 1.")
        if dt <= 0:
            raise ValueError("dt must be > 0.")
        if not (fallback in ("hold", "zero") or callable(fallback)):
            raise ValueError("fallback must be 'hold', 'zero' or a callable.")
        self.blocks = blocks
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.co2_price = float(co2_price)
        self.unserved_penalty = unserved_penalty
        self.spill_penalty = spill_penalty
        self.fallback_rule = fallback
        self.horizon_end = horizon_end
        self.delay_compensation = bool(delay_compensation)
        self.time_limit = time_limit
        self.mip_rel_gap = mip_rel_gap

        setpoints, meas, units, init_meas = [], [], {}, {}
        for b in blocks:
            for var, unit in b.outputs.items():
                setpoints.append(b.key(var))
                units[b.key(var)] = unit
            for var, unit in b.measurements.items():
                meas.append(b.key(var))
                units[b.key(var)] = unit
            init_meas.update({b.key(k): v for k, v in b.initial_measurements().items()})
        meas.extend(extra_inputs)
        kwargs.setdefault("step_size", self.dt)
        super().__init__(name=name, measurement_inputs=meas, setpoint_outputs=setpoints, forecasts=forecasts,
                         initial_measurements=init_meas, units=units, decide_at_init=decide_at_init, **kwargs)
        self.last_plan: Optional[MPCPlan] = None

    # ------------------------------------------------------------------
    @classmethod
    def from_components(cls, components: Mapping[str, Any], forecasts: Optional[Mapping[str, Any]] = None,
                        prices: Any = None, loads: Optional[Mapping[str, Any]] = None,
                        grid: Union[None, bool, Mapping[str, Any], GridBlock] = None,
                        horizon: int = 24, dt: float = 900.0,
                        block_options: Optional[Mapping[str, Mapping[str, Any]]] = None, **kwargs) -> "MPCController":
        """Build an MPC from energysim model components.

        ``components`` maps block names to model instances (or Blocks);
        ``loads`` maps load names to demand specs (forecast key, array,
        Forecast) or LoadBlocks; ``prices`` is the grid import price spec.
        ``grid`` may be a dict of :class:`GridBlock` arguments, a GridBlock,
        or ``False`` (islanded).  ``block_options`` passes per-component
        overrides to :func:`block_from_component`.
        """
        block_options = dict(block_options or {})
        blocks: List[Block] = [
            block_from_component(comp, name=name, **dict(block_options.get(name, {})))
            for name, comp in components.items()
        ]
        for lname, spec in (loads or {}).items():
            blocks.append(spec if isinstance(spec, Block) else LoadBlock(lname, spec))
        if isinstance(grid, GridBlock):
            blocks.append(grid)
        elif grid is not False and (grid is not None or prices is not None):
            opts = dict(grid or {})
            if prices is not None:
                opts.setdefault("price_import", prices)
            blocks.append(GridBlock(opts.pop("name", "grid"), **opts))
        return cls(blocks, forecasts=forecasts, horizon=horizon, dt=dt, **kwargs)

    def block(self, name: str) -> Block:
        for b in self.blocks:
            if b.name == name:
                return b
        raise KeyError(f"No block named '{name}'.")

    # ------------------------------------------------------------------
    def reset_controller(self) -> None:
        self._state = {b.name: b.initial_state() for b in self.blocks}
        self.last_plan = None
        self.plans: List[MPCPlan] = []

    def horizon_times(self, t_act: float) -> np.ndarray:
        n = self.horizon
        if self.horizon_end is not None:
            n = max(1, min(n, int(math.ceil((self.horizon_end - t_act) / self.dt - 1e-9))))
        return float(t_act) + self.dt * np.arange(n)

    # ------------------------------------------------------------------
    def solve_horizon(self, times: Sequence[float], states: Mapping[str, Dict[str, float]],
                      forecasts: Optional[Mapping[str, np.ndarray]] = None,
                      measurements: Optional[Mapping[str, float]] = None) -> MPCPlan:
        """Build and solve the problem over ``times`` from the given block states."""
        times = np.asarray(times, dtype=float)
        lp = LinearProblem()
        ctx = PlanContext(lp, times, self.dt, forecasts, measurements, self.co2_price)
        for b in self.blocks:
            b.build(ctx, dict(states.get(b.name, b.initial_state())))
        N, dt_h = ctx.N, ctx.dt_h
        for carrier in sorted(ctx.balance_terms):
            terms = list(ctx.balance_terms[carrier])
            if self.unserved_penalty is not None:
                u = lp.add_variables(f"unserved.{carrier}", N, 0.0, np.inf, float(self.unserved_penalty) * dt_h)
                terms.append((u, np.ones(N)))
            if self.spill_penalty is not None:
                s = lp.add_variables(f"spill.{carrier}", N, 0.0, np.inf, float(self.spill_penalty) * dt_h)
                terms.append((s, -np.ones(N)))
            const = ctx.balance_const.get(carrier, np.zeros(N))
            if not terms:
                if np.any(np.abs(const) > 1e-12):
                    return MPCPlan(times, 2, float("nan"), message=f"carrier '{carrier}' has no flexibility")
                continue
            lp.add_eq(terms, -const, n=N)
        res = lp.solve(time_limit=self.time_limit, mip_rel_gap=self.mip_rel_gap)
        plan = MPCPlan(times, res.status, res.objective, solve_time_s=res.solve_time_s, message=res.message)
        if res.ok:
            x = res.x
            for b in self.blocks:
                for var, arr in b.extract(x, ctx).items():
                    plan.values[b.key(var)] = np.asarray(arr, dtype=float)
            for carrier in ctx.balance_terms:
                key = f"unserved.{carrier}"
                if key in lp.groups:
                    plan.values[key] = x[lp.groups[key]]
            plan.emissions_t = float(sum(float(np.dot(x[i], c)) for i, c in ctx.emission_terms))
        return plan

    def plan_open_loop(self, t0: float, n_steps: Optional[int] = None,
                       measurements: Optional[Mapping[str, float]] = None) -> MPCPlan:
        """Solve one problem from ``t0`` over ``n_steps`` (default ``horizon``) from the initial
        (or given measured) states, without changing the controller's run state."""
        n = self.horizon if n_steps is None else int(n_steps)
        times = float(t0) + self.dt * np.arange(n)
        meas = dict(self.initial_measurements)
        meas.update(measurements or {})
        fresh = set(meas)
        states = {b.name: b.state_from_measurements(b.initial_state(), meas, fresh) for b in self.blocks}
        fc = {k: np.asarray(f(t0, times, meas), dtype=float) for k, f in self.forecasts.items()}
        return self.solve_horizon(times, states, fc, meas)

    # ------------------------------------------------------------------
    def decide(self, t, measurements, forecasts):
        lag = float(t) - float(self.measurement_time)
        states = {}
        for b in self.blocks:
            st = b.state_from_measurements(self._state[b.name], measurements, self.fresh_measurements)
            if self.delay_compensation:
                st = b.predict(st, self.applied_setpoints, max(lag, 0.0))
            states[b.name] = st
        times = self.horizon_times(t)
        plan = self.solve_horizon(times, states, forecasts, measurements)
        self.plans.append(plan)
        out: Dict[str, Any] = {"solver_status": plan.status, "solve_time_s": plan.solve_time_s,
                               "objective": plan.objective, "fallback": FALLBACK_NONE}
        if plan.ok:
            sps = {k: float(plan.values[k][0]) for k in self._setpoint_names if k in plan.values}
            self.last_plan = plan
        else:
            sps, code = self._fallback(t, measurements, forecasts)
            out["fallback"] = code
        out.update(sps)
        # Carry the state estimate to the actuation time with the new set-points.
        new_sp = dict(self.applied_setpoints)
        new_sp.update(sps)
        self._state = {b.name: b.predict(states[b.name], new_sp, 0.0) for b in self.blocks}
        return out

    def _fallback(self, t, measurements, forecasts):
        prev = self.last_plan
        if prev is not None and prev.times.size:
            j = int(round((t - prev.times[0]) / self.dt))
            if 0 <= j < prev.times.size:
                return ({k: float(prev.values[k][j]) for k in self._setpoint_names if k in prev.values},
                        FALLBACK_PREVIOUS_PLAN)
        rule = self.fallback_rule
        if callable(rule):
            return dict(rule(t, measurements, forecasts) or {}), FALLBACK_RULE
        if rule == "zero":
            return {k: 0.0 for k in self._setpoint_names}, FALLBACK_RULE
        return {}, FALLBACK_RULE

    def status_text(self) -> str:
        return STATUS_TEXT.get(int(self._diag["solver_status"]), "unknown")

# -*- coding: utf-8 -*-
"""
Controller base class: a controller is just another simulator.

At every macro step the world hands the controller its *measurement*
inputs (connected from plant outputs or signals); the controller reads its
forecasts, solves a decision problem in :meth:`Controller.decide` and
exposes *set-points* as outputs, which the world connects to plant inputs.

Timing (actuation delay)
------------------------
energysim exchanges values at macro-step boundaries and then advances all
simulators (Jacobi), so a feedback loop plant -> controller -> plant always
contains one macro step of delay: the set-point computed while the world
advances ``[t, t + h]`` from measurements taken at ``t`` reaches the plant at
``t + h``.  The controller therefore calls ``decide(t_act, ...)`` with
``t_act = t + h`` (the time at which the decision takes effect) and exposes
``measurement_time`` (``t``) and ``applied_setpoints`` (what the plant runs
over ``[t, t + h]``) so that model-based controllers can predict the state
at ``t_act`` (delay compensation, see :class:`~energysim.control.MPCController`).
With ``actuation_delay=False`` ``t_act == t``.

If ``initial_measurements`` are given (or ``decide_at_init=True``), a first
decision is taken in :meth:`init` so that the plant receives a meaningful
set-point for the very first interval.
"""

from __future__ import annotations

import math
import numbers
import time as _time
from abc import abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..base import SimulatorAdapter, SimulatorVariableError
from .forecast import Forecast, as_forecast
from .lp import STATUS_NOT_SOLVED, STATUS_OPTIMAL

#: Diagnostic outputs every controller exposes.
DIAGNOSTIC_OUTPUTS = ("solver_status", "solve_time_s", "objective", "fallback")

#: ``fallback`` output codes.
FALLBACK_NONE = 0
FALLBACK_PREVIOUS_PLAN = 1
FALLBACK_RULE = 2


class Controller(SimulatorAdapter):
    """Base class of controllers (optimisers, MPC, OPF, rule-based ...).

    Parameters
    ----------
    name : str
        Controller name (used in messages).
    measurement_inputs : list[str]
        Names of the inputs (measurements / external signals).
    setpoint_outputs : list[str]
        Names of the set-point outputs.  The diagnostic outputs
        ``solver_status`` (0 = optimal, 1 = limit, 2 = infeasible,
        3 = unbounded, 4 = error, -1 = not solved yet), ``solve_time_s``,
        ``objective`` and ``fallback`` (0 none, 1 previous plan, 2 rule) are
        always added.
    forecasts : dict[str, Forecast], optional
        Forecast providers evaluated at every decision over
        :meth:`horizon_times` and handed to :meth:`decide`.
    initial_setpoints : dict, optional
        Output values before the first decision (default 0.0).
    initial_measurements : dict, optional
        Measurement values assumed before the first exchange (default 0.0).
    units : dict, optional
        ``{variable: unit}`` for inputs / outputs (used by world unit conversion).
    decide_at_init : bool, optional
        Take a decision in :meth:`init` (default: True when
        ``initial_measurements`` were given).
    actuation_delay : bool, default True
        See the module docstring.
    step_size : float, default 900
        Decision interval used by :meth:`step` (``advance`` decides once per call).
    """

    def __init__(self, name: str = "controller", measurement_inputs: Sequence[str] = (),
                 setpoint_outputs: Sequence[str] = (), forecasts: Optional[Mapping[str, Any]] = None,
                 initial_setpoints: Optional[Mapping[str, float]] = None,
                 initial_measurements: Optional[Mapping[str, float]] = None,
                 units: Optional[Mapping[str, str]] = None, decide_at_init: Optional[bool] = None,
                 actuation_delay: bool = True, step_size: float = 900.0):
        self.name = name
        meas = [str(m) for m in measurement_inputs]
        sps = [str(s) for s in setpoint_outputs]
        for label, names in (("measurement_inputs", meas), ("setpoint_outputs", sps)):
            dup = {n for n in names if names.count(n) > 1}
            if dup:
                raise ValueError(f"{name}: duplicate names in {label}: {sorted(dup)}")
        clash = (set(meas) & set(sps)) | (set(meas + sps) & set(DIAGNOSTIC_OUTPUTS))
        if clash:
            raise ValueError(f"{name}: names used more than once or reserved: {sorted(clash)}")
        self._measurement_names: List[str] = meas
        self._setpoint_names: List[str] = sps
        self._output_names: List[str] = sps + list(DIAGNOSTIC_OUTPUTS)
        # Public lists (the world may overwrite these; strictness uses the private ones).
        self.inputs = list(meas)
        self.outputs = list(self._output_names)

        self.forecasts: Dict[str, Forecast] = {k: as_forecast(v) for k, v in (forecasts or {}).items()}
        self._check_known(initial_setpoints or {}, sps, "initial_setpoints")
        self._check_known(initial_measurements or {}, meas, "initial_measurements")
        self.initial_setpoints = {s: 0.0 for s in sps}
        self.initial_setpoints.update({k: float(v) for k, v in (initial_setpoints or {}).items()})
        self.initial_measurements = {m: 0.0 for m in meas}
        self.initial_measurements.update({k: float(v) for k, v in (initial_measurements or {}).items()})
        self.units = dict(units or {})
        self.decide_at_init = (initial_measurements is not None) if decide_at_init is None else bool(decide_at_init)
        self.actuation_delay = bool(actuation_delay)
        self.step_size = float(step_size)
        self.start_time = 0.0
        self.history: List[Dict[str, Any]] = []
        self._reset_runtime()

    # ------------------------------------------------------------------
    def _check_known(self, mapping, allowed, label):
        unknown = [k for k in mapping if k not in allowed]
        if unknown:
            raise SimulatorVariableError(f"{self.name}: unknown names in {label}: {unknown}. Allowed: {allowed}")

    def _reset_runtime(self) -> None:
        self._meas: Dict[str, float] = dict(self.initial_measurements)
        self._fresh: set = set()
        self._setpoints: Dict[str, float] = dict(self.initial_setpoints)
        self._diag: Dict[str, float] = {
            "solver_status": float(STATUS_NOT_SOLVED), "solve_time_s": 0.0,
            "objective": float("nan"), "fallback": float(FALLBACK_NONE),
        }
        self.measurement_time = self.start_time
        self.applied_setpoints: Dict[str, float] = dict(self._setpoints)
        self.fresh_measurements: set = set()
        self.history = []

    # ------------------------------------------------------------------
    # Life cycle
    # ------------------------------------------------------------------
    def configure(self, start_time: float, stop_time: float, macro_step: float) -> None:
        self.start_time = float(start_time)

    def reset_controller(self) -> None:
        """Hook for subclasses: clear internal state (plans, estimates) at ``init()``."""

    def init(self) -> None:
        """Start a fresh run: outputs, measurements, history and forecasts are reset."""
        self._reset_runtime()
        for f in self.forecasts.values():
            f.reset()
        self.reset_controller()
        if self.decide_at_init:
            self._decide(self.start_time, self.start_time, fresh=set(self._measurement_names))

    # ------------------------------------------------------------------
    # Variable access (strict)
    # ------------------------------------------------------------------
    def set_value(self, parameters: Sequence[str], values: Sequence[Any]) -> None:
        for p, v in zip(parameters, values):
            if p not in self._measurement_names:
                raise SimulatorVariableError(
                    f"{self.name}: '{p}' is not a measurement input. Inputs: {self._measurement_names}."
                )
            if isinstance(v, bool) or not isinstance(v, numbers.Real):
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    raise SimulatorVariableError(f"{self.name}: input '{p}' needs a number, got {v!r}.") from None
            self._meas[p] = float(v)
            self._fresh.add(p)

    def get_value(self, parameters: Sequence[str], time: float = None) -> List[Any]:
        out = []
        for p in parameters:
            if p in self._setpoints:
                out.append(self._setpoints[p])
            elif p in self._diag:
                out.append(self._diag[p])
            elif p in self._meas:
                out.append(self._meas[p])
            else:
                raise SimulatorVariableError(
                    f"{self.name}: unknown variable '{p}'. Outputs: {self._output_names}; "
                    f"inputs: {self._measurement_names}."
                )
        return out

    def get_available_variables(self) -> Dict[str, List[str]]:
        return {"inputs": list(self._measurement_names), "outputs": list(self._output_names)}

    @property
    def setpoints(self) -> Dict[str, float]:
        """Current set-point outputs (copy)."""
        return dict(self._setpoints)

    @property
    def diagnostics(self) -> Dict[str, float]:
        return dict(self._diag)

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------
    def step(self, time: float) -> None:
        """Decide once at ``time`` (stand-alone use; delay = ``step_size``)."""
        lag = self.step_size if self.actuation_delay else 0.0
        self._decide(time, time + lag)

    def advance(self, start_time: float, stop_time: float) -> None:
        """One decision per world macro step (delay = macro step)."""
        t_act = stop_time if self.actuation_delay else start_time
        self._decide(start_time, t_act)

    def advance_with_recording(self, start_time, stop_time, outputs):
        row = [start_time] + (list(self.get_value(outputs, start_time)) if outputs else [])
        self.advance(start_time, stop_time)
        return [row]

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------
    def horizon_times(self, t_act: float) -> np.ndarray:
        """Start times of the forecast horizon intervals (default: just ``t_act``)."""
        return np.array([float(t_act)])

    def evaluate_forecasts(self, t_act: float, measurements: Mapping[str, float]) -> Dict[str, np.ndarray]:
        times = self.horizon_times(t_act)
        return {k: np.asarray(f(t_act, times, measurements), dtype=float) for k, f in self.forecasts.items()}

    @abstractmethod
    def decide(self, t: float, measurements: Dict[str, float], forecasts: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Return ``{setpoint: value}`` for the interval starting at ``t``.

        May also contain the diagnostic keys ``solver_status``,
        ``solve_time_s``, ``objective`` and ``fallback``.  Set-points that are
        omitted keep their previous value.  Unknown keys raise
        :class:`~energysim.SimulatorVariableError`.
        """

    def _decide(self, t_meas: float, t_act: float, fresh: Optional[set] = None) -> None:
        meas = dict(self._meas)
        self.measurement_time = float(t_meas)
        self.applied_setpoints = dict(self._setpoints)
        self.fresh_measurements = set(self._fresh if fresh is None else fresh)
        forecasts = self.evaluate_forecasts(t_act, meas)
        t0 = _time.perf_counter()
        result = self.decide(float(t_act), meas, forecasts) or {}
        elapsed = _time.perf_counter() - t0
        if not isinstance(result, Mapping):
            raise TypeError(f"{self.name}: decide() must return a dict, got {type(result).__name__}.")
        diag = {"solver_status": float(STATUS_OPTIMAL), "solve_time_s": elapsed,
                "objective": float("nan"), "fallback": float(FALLBACK_NONE)}
        for k, v in result.items():
            if k in self._setpoints:
                v = float(v)
                if not math.isfinite(v):
                    raise ValueError(f"{self.name}: decide() returned non-finite set-point {k}={v}.")
                self._setpoints[k] = v
            elif k in diag:
                diag[k] = float(v)
            else:
                raise SimulatorVariableError(
                    f"{self.name}: decide() returned unknown output '{k}'. Outputs: {self._output_names}."
                )
        self._diag = diag
        self._fresh = set()
        self.history.append({"time": float(t_meas), "t_act": float(t_act),
                             "measurements": meas, "setpoints": dict(self._setpoints), **diag})


class RuleController(Controller):
    """Controller defined by a plain function ``rule(t, measurements, forecasts) -> dict``.

    Handy as a baseline or as the fallback of an optimising controller.
    """

    def __init__(self, rule, measurement_inputs=(), setpoint_outputs=(), **kwargs):
        if not callable(rule):
            raise TypeError("RuleController needs a callable rule(t, measurements, forecasts).")
        self.rule = rule
        super().__init__(measurement_inputs=measurement_inputs, setpoint_outputs=setpoint_outputs, **kwargs)

    def decide(self, t, measurements, forecasts):
        return self.rule(t, measurements, forecasts)

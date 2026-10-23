# -*- coding: utf-8 -*-
"""
Base ModelComponent for all energysim physical and empirical models.

Inherits from SimulatorAdapter and provides:

* strict variable routing (``set_value`` accepts only declared inputs and
  registered parameters, ``get_value`` raises for unknown names),
* outputs that are valid immediately after ``init()`` (evaluated from the
  initial state without advancing it),
* re-runnable models: ``init()`` / ``reset()`` restore the initial instance
  state captured on the first ``init()``,
* self-description (``get_available_variables()`` / ``describe()``).

Rollback (``save_state`` / ``restore_state``) is intentionally not supported.
"""

from __future__ import annotations

import copy
import inspect
import numbers
from typing import Any, Dict, List, Optional, Sequence

from energysim.base import SimulatorAdapter, SimulatorVariableError

# Name of the instance attribute holding the initial-state snapshot.
_SNAPSHOT_ATTR = "_initial_snapshot"


def _is_shared(value: Any) -> bool:
    """Objects treated as immutable and shared (not deep-copied) in snapshots."""
    if inspect.isroutine(value):
        return True
    # Lookup curves / surfaces are immutable by convention.
    return type(value).__name__ in ("Curve1D", "Surface2D")


def _copy_dict(src: Dict[str, Any], exclude: Sequence[str] = ()) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in src.items():
        if k in exclude:
            continue
        out[k] = v if _is_shared(v) else copy.deepcopy(v)
    return out


class ModelComponent(SimulatorAdapter):
    """
    Ergonomic base class for physical and empirical energy system models.

    Parameters
    ----------
    name : str, optional
        Unique component name.
    inputs : list[str], optional
        Declared input parameter names.
    outputs : list[str], optional
        Declared output parameter names.
    step_size : float, default 1.0
        Simulation micro-step size in seconds.

    Notes
    -----
    Subclasses implement :meth:`step_model`, call :meth:`register_variables`
    (and optionally :meth:`register_parameters`) in ``__init__`` and may set
    the class attribute ``units`` mapping every input/output name to a unit
    string (``'MW'``, ``'degC'``, ``'1'`` for dimensionless, ...).

    A model should be *quiescent at dt = 0*: ``step_model(0.0, t)`` should not
    change state. This is how outputs are evaluated at ``init()``; as a safety
    net the framework restores all state after that evaluation anyway.
    """

    #: Optional ``{variable_name: unit}`` mapping for inputs and outputs.
    units: Dict[str, str] = {}

    def __init__(
        self,
        name: str = "",
        inputs: Optional[Sequence[str]] = None,
        outputs: Optional[Sequence[str]] = None,
        step_size: float = 1.0,
        **kwargs,
    ):
        self.name = name
        self.inputs = list(inputs) if inputs is not None else []
        self.outputs = list(outputs) if outputs is not None else []
        self.step_size = float(step_size)

        # Internal state dictionary for continuous and discrete variables
        self._states: Dict[str, Any] = {}
        self._inputs_val: Dict[str, Any] = {}
        self._outputs_val: Dict[str, Any] = {}
        self._parameters: List[str] = []
        self._initial_snapshot: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_variables(
        self,
        inputs: Optional[Sequence[str]] = None,
        outputs: Optional[Sequence[str]] = None,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register default inputs and outputs.

        ``defaults`` maps input names to their initial value; inputs without an
        entry fall back to an attribute of the same (or lower-case) name, else 0.0.
        """
        defaults = defaults or {}
        if inputs:
            for inp in inputs:
                if inp not in self.inputs:
                    self.inputs.append(inp)
                if inp in defaults:
                    default_val = defaults[inp]
                else:
                    default_val = getattr(self, inp, getattr(self, inp.lower(), 0.0))
                self._inputs_val.setdefault(inp, default_val)
        if outputs:
            for out in outputs:
                if out not in self.outputs:
                    self.outputs.append(out)
                if out not in self._inputs_val:   # an input that is also an output passes through
                    self._outputs_val.setdefault(out, 0.0)

    def register_parameters(self, *names: str) -> None:
        """Declare numeric instance attributes that ``set_value`` may overwrite."""
        for n in names:
            if not hasattr(self, n):
                raise AttributeError(f"{type(self).__name__} has no attribute '{n}' to register as parameter.")
            if n not in self._parameters:
                self._parameters.append(n)

    # ------------------------------------------------------------------
    # Life-cycle: init / reset
    # ------------------------------------------------------------------
    def _take_snapshot(self) -> Dict[str, Any]:
        return _copy_dict(self.__dict__, exclude=(_SNAPSHOT_ATTR,))

    def _restore_snapshot(self, snap: Dict[str, Any]) -> None:
        for k in [k for k in self.__dict__ if k not in snap and k != _SNAPSHOT_ATTR]:
            del self.__dict__[k]
        self.__dict__.update(_copy_dict(snap))

    def init(self) -> None:
        """
        Prepare the model for a run.

        The first call captures the initial instance state; every later call
        restores it, so a model can be run repeatedly from the same start.
        Outputs are then evaluated from the initial state (without advancing
        it) so that they are valid before the first step.
        """
        if self._initial_snapshot is None:
            self._initial_snapshot = self._take_snapshot()
        else:
            self._restore_snapshot(self._initial_snapshot)
        self._update_outputs()

    def reset(self) -> None:
        """Restore the initial state captured by the first ``init()``."""
        self.init()

    def _update_outputs(self) -> None:
        """
        Evaluate outputs from the current state without advancing it.

        Default: run ``step_model`` with ``dt = 0`` and restore every state
        variable afterwards, keeping only the freshly computed outputs.
        Override for models whose ``step_model`` is not quiescent at dt = 0.
        """
        before = self._take_snapshot()
        self.step_model(0.0, 0.0)
        outputs = copy.deepcopy(self._outputs_val)
        self._restore_snapshot(before)
        self._outputs_val = outputs

    # ------------------------------------------------------------------
    # Variable access
    # ------------------------------------------------------------------
    def set_value(self, parameters: Sequence[str], values: Sequence[Any]) -> None:
        """Receive inputs from the world, or overwrite registered parameters.

        Raises
        ------
        SimulatorVariableError
            For names that are neither declared inputs nor registered
            parameters, or for non-numeric values given to a parameter.
        """
        label = self.name or type(self).__name__
        for p, v in zip(parameters, values):
            if p in self.inputs:
                self._inputs_val[p] = v
            elif p in self._parameters:
                cur = getattr(self, p)
                if isinstance(cur, bool) or not isinstance(cur, numbers.Real):
                    raise SimulatorVariableError(f"{label}: parameter '{p}' is not numeric and cannot be set.")
                if isinstance(v, bool) or not isinstance(v, numbers.Real):
                    raise SimulatorVariableError(f"{label}: parameter '{p}' requires a numeric value, got {v!r}.")
                setattr(self, p, v)
            else:
                raise SimulatorVariableError(
                    f"{label}: '{p}' is not a declared input or parameter. "
                    f"Inputs: {self.inputs}; parameters: {self._parameters}."
                )

    def get_value(self, parameters: Sequence[str], time: float) -> List[Any]:
        """Query component output (or input/state/parameter) variables."""
        results = []
        for p in parameters:
            if p in self._outputs_val:
                val = self._outputs_val[p]
            elif p in self._inputs_val:
                # also covers an output name that is an input as well (pass-through)
                val = self._inputs_val[p]
            elif p in self.outputs:
                val = 0.0
            elif p in self._states:
                val = self._states[p]
            elif p in self._parameters:
                val = getattr(self, p)
            else:
                raise SimulatorVariableError(
                    f"{self.name or type(self).__name__}: unknown variable '{p}'. "
                    f"Outputs: {self.outputs}; inputs: {self.inputs}."
                )
            results.append(val)
        return results

    def get_available_variables(self) -> Dict[str, List[str]]:
        """Return ``{'inputs': [...], 'outputs': [...]}`` of registered variables."""
        return {"inputs": list(self.inputs), "outputs": list(self.outputs)}

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        """
        Machine-readable description of this model (feeds generated docs).

        Returns a dict with keys ``class``, ``doc``, ``inputs``, ``outputs``,
        ``parameters`` (constructor defaults) and ``units``.
        """
        cls = type(self)
        params: Dict[str, Any] = {}
        try:
            sig = inspect.signature(cls.__init__)
            for pname, p in sig.parameters.items():
                if pname in ("self", "name") or p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
                    continue
                if p.default is not inspect.Parameter.empty and p.default is not None:
                    params[pname] = p.default
        except (TypeError, ValueError):
            pass
        return {
            "class": cls.__name__,
            "doc": inspect.getdoc(cls) or "",
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "parameters": params,
            "units": {k: v for k, v in cls.units.items() if k in self.inputs or k in self.outputs},
        }

    @classmethod
    def describe_class(cls) -> Dict[str, Any]:
        """Describe the model class using a default-constructed instance."""
        return cls().describe()

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------
    def step(self, time: float) -> None:
        """Advance the model by one micro-step."""
        self.step_model(self.step_size, time)

    def advance(self, start_time: float, stop_time: float) -> None:
        """Advance model across interval [start_time, stop_time]."""
        t = start_time
        dt = self.step_size
        while t < stop_time - 1e-9:
            step_dt = min(dt, stop_time - t)
            self.step_model(step_dt, t)
            t += step_dt

    def step_partial(self, time: float, dt: float) -> None:
        """Shorter final micro-step (event-driven macro steps): ``step_model(dt, time)``."""
        self.step_model(dt, time)

    # ------------------------------------------------------------------
    # Event hints
    # ------------------------------------------------------------------
    def time_to_event(self) -> float:
        """Seconds until the next predictable internal event, from the current state and inputs.

        Override in models with timers or limits (start-up sequence finished,
        storage full/empty at the current power, ramp completed, ...).  The
        world uses it (via :meth:`next_event_time`) to end a macro step exactly
        at the event when ``t_macro_min < t_macro``.  Default: ``inf``.
        """
        return float("inf")

    def next_event_time(self, time: float) -> float:
        """``time + time_to_event()`` (``inf`` when no event is ahead)."""
        try:
            dt = float(self.time_to_event())
        except Exception:
            return float("inf")
        if not (dt > 0.0) or dt == float("inf") or dt != dt:
            return float("inf")
        return time + dt

    def step_model(self, dt: float, time: float) -> None:
        """
        Subclasses implement this method with their physical/empirical logic.
        """
        raise NotImplementedError("Subclasses must implement step_model(dt, time).")

    def cleanup(self) -> None:
        """Release any resources."""
        pass

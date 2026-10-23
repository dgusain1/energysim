# -*- coding: utf-8 -*-
"""
Base class for all energysim simulator adapters.

Every adapter (FMU CS/ME, pandapower, PyPSA, CSV, signal, external, …)
inherits from ``SimulatorAdapter`` and implements the required abstract
methods.  The ``world`` orchestrator interacts *only* through this
interface — no ``isinstance`` / ``sim_type`` branching needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class EnergysimError(Exception):
    """Base exception for all energysim errors."""


class SimulatorVariableError(EnergysimError):
    """A variable name was not found in a simulator."""


class SimulatorElementNotFoundError(EnergysimError):
    """An element / component was not found in a network model."""


class ConnectionSpecError(EnergysimError):
    """A connection specification is invalid."""


# Backwards-compatible alias (shadows the builtin only when imported explicitly).
ConnectionError = ConnectionSpecError


class ValidationError(EnergysimError):
    """A world pre-flight validation check failed."""


@dataclass
class ValidationReport:
    """Detailed summary of world.validate() execution."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    #: one line per connection: ``A.y [kW] -> B.u [MW]  x0.001  (source: ...)``
    conversions: List[str] = field(default_factory=list)
    #: ``"<name> [<hash>]"`` of the world's unit system
    unit_system: Optional[str] = None

    def __str__(self) -> str:
        lines = [f"ValidationReport: {'PASSED' if self.is_valid else 'FAILED'}"]
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"    - {e}")
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)

    def format_units(self) -> str:
        """Unit system and per-connection unit conversions (used by ``energysim validate -v``)."""
        lines = [f"Unit system: {self.unit_system or 'n/a'}",
                 f"Unit conversions ({len(self.conversions)}):"]
        lines.extend(f"    {c}" for c in self.conversions)
        return "\n".join(lines)


class ParsedConnection(tuple):
    """
    Parsed connection tuple (output, input) with scale, offset, and unit attributes.
    Supports unpacking as (output, input) for 100% backwards compatibility.

    ``scale`` / ``offset`` are the *effective* values applied during exchange
    (unit conversion followed by the user's scale/offset):
    ``value_in = (value_out * unit_scale + unit_offset) * user_scale + user_offset``.
    ``user_scale`` / ``user_offset`` keep what the user specified, so the
    connection can be serialised without applying unit conversion twice.

    Unit bookkeeping: ``from_unit`` / ``to_unit`` are the resolved units of
    the (first) source and the (first) receiver, ``from_source`` /
    ``to_source`` where they came from (``'declared'``, ``'component'``,
    ``'connection'``, ``'suffix'``, ``'default-by-quantity'``,
    ``'unspecified'`` or ``'manual'`` for a user scale without units).
    Values are always converted to the *receiver's* unit.  For fan-out
    connections whose receivers have different units ``target_conversions``
    holds one effective ``(scale, offset)`` per receiver; for fan-in
    connections whose sources have different units ``source_scales`` holds
    the factor converting each source into the first source's unit.
    """
    scale: float
    offset: float
    from_unit: Optional[str]
    to_unit: Optional[str]
    user_scale: float
    user_offset: float

    def __new__(cls, parsed_output: Any, parsed_input: Any, scale: float = 1.0, offset: float = 0.0,
                from_unit: Optional[str] = None, to_unit: Optional[str] = None,
                user_scale: Optional[float] = None, user_offset: Optional[float] = None,
                from_source: Optional[str] = None, to_source: Optional[str] = None,
                unit_scale: float = 1.0, unit_offset: float = 0.0,
                target_conversions: Optional[List[Any]] = None,
                source_scales: Optional[List[float]] = None,
                explicit_units: bool = False):
        instance = super().__new__(cls, (parsed_output, parsed_input))
        instance.scale = float(scale)
        instance.offset = float(offset)
        instance.from_unit = from_unit
        instance.to_unit = to_unit
        instance.user_scale = float(scale if user_scale is None else user_scale)
        instance.user_offset = float(offset if user_offset is None else user_offset)
        instance.from_source = from_source
        instance.to_source = to_source
        instance.unit_scale = float(unit_scale)
        instance.unit_offset = float(unit_offset)
        instance.target_conversions = target_conversions
        instance.source_scales = source_scales
        instance.explicit_units = bool(explicit_units)
        return instance

    @property
    def source(self) -> str:
        """``'<from_source> -> <to_source>'`` (or a single token when both agree)."""
        a, b = self.from_source or 'unspecified', self.to_source or 'unspecified'
        return a if a == b else f"{a} -> {b}"


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class SimulatorAdapter(ABC):
    """Uniform interface that every simulator adapter must satisfy."""

    # Subclasses *may* set this; the world can also override it.
    step_size: float = 1.0

    #: When True, the world applies ``options({'init': ...})`` /
    #: ``set_parameters`` values *before* calling :meth:`init` (needed by
    #: FMUs, whose start values must be set before initialisation mode is
    #: left).  When False (default) they are applied *after* :meth:`init`,
    #: which is what tools that load their model in ``init()`` (OpenDSS,
    #: PowerFactory, MATLAB, ...) need.
    SET_PARAMS_BEFORE_INIT: bool = False

    # ------------------------------------------------------------------
    # Life-cycle hooks (override when needed; defaults are no-ops)
    # ------------------------------------------------------------------

    def configure(self, start_time: float, stop_time: float, macro_step: float) -> None:
        """Receive the run's time frame before :meth:`init` is called.

        The world calls this on every adapter at the start of every run
        (before any parameters are applied and before ``init()``).
        Default is a no-op; adapters such as FMUs override it to set up
        experiment start/stop times.
        """

    def init(self) -> None:
        """Initialise the simulator (called once at the start of every run)."""

    def cleanup(self) -> None:
        """Release resources (called once after the last step)."""

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Apply a {name: value} dict of initial / start values.

        Default implementation delegates to ``set_value``; FMU adapters
        override this to call ``apply_start_values`` instead.
        """
        if params:
            names = list(params.keys())
            values = list(params.values())
            self.set_value(names, values)

    # ------------------------------------------------------------------
    # Abstract methods — every adapter MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def step(self, time: float) -> None:
        """Advance the internal model by *one* micro-step at ``time``.

        For simple adapters (CSV, signal, pandapower, …) this is a
        single evaluation / power-flow run.  For FMU CS adapters it
        wraps ``doStep``.  For FMU ME adapters it is one ODE-solver
        micro-step.
        """

    @abstractmethod
    def get_value(self, parameters: List[str], time: float) -> List[Any]:
        """Return a list of values corresponding to ``parameters``."""

    @abstractmethod
    def set_value(self, parameters: List[str], values: List[Any]) -> None:
        """Set a list of ``parameters`` to the given ``values``."""

    # ------------------------------------------------------------------
    # Default advance() — loops step() over [start_time, stop_time)
    # ------------------------------------------------------------------

    def micro_steps(self, start_time: float, stop_time: float) -> List[Tuple[float, float, bool]]:
        """Micro-step plan ``[(t, dt, full), ...]`` covering ``[start_time, stop_time)``.

        When the interval is an integer multiple of ``step_size`` (relative
        tolerance 1e-9) this is ``n`` full steps at ``start + i*step_size``.
        Otherwise (the world shortened the macro step for an event) the full
        steps that fit are followed by one shorter final step with
        ``full=False`` and ``dt = stop_time - t``.
        """
        h = self.step_size
        span = stop_time - start_time
        if span <= 0:
            return []
        n = round(span / h)
        if n >= 1 and abs(n * h - span) <= 1e-9 * max(1.0, abs(span)):
            return [(start_time + i * h, h, True) for i in range(n)]
        n_full = int(math.floor(span / h + 1e-9))
        plan = [(start_time + i * h, h, True) for i in range(n_full)]
        t = start_time + n_full * h
        rem = stop_time - t
        if rem > 1e-12 * max(1.0, abs(stop_time)):
            plan.append((t, rem, False))
        return plan

    def step_partial(self, time: float, dt: float) -> None:
        """Advance by a *shorter* micro-step ``dt < step_size`` starting at ``time``.

        Called for the final micro-step when a macro interval is not a
        multiple of ``step_size`` (event-driven macro steps).  The default
        calls :meth:`step` (a full step - adapters whose ``step`` has no
        notion of ``dt``, e.g. power flows, are unaffected); adapters that
        integrate in time should override it (``ModelComponent`` uses
        ``step_model(dt, time)``, FMU co-simulation ``doStep(time, dt)``).
        """
        self.step(time)

    def _do_micro_step(self, t: float, dt: float, full: bool) -> None:
        if full:
            self.step(t)
        else:
            self.step_partial(t, dt)

    def advance(self, start_time: float, stop_time: float) -> None:
        """Advance the simulator from *start_time* to *stop_time*.

        The default implementation loops ``step()`` at intervals of
        ``self.step_size`` and finishes with :meth:`step_partial` when the
        interval is not a multiple of the step size (see :meth:`micro_steps`).
        Adapters with an internal solver (e.g. FMU Model Exchange with CVode)
        should override this to hand the full interval to their solver in one call.
        """
        for t, dt, full in self.micro_steps(start_time, stop_time):
            self._do_micro_step(t, dt, full)

    # ------------------------------------------------------------------
    # advance_with_recording() — step + collect intermediate results
    # ------------------------------------------------------------------

    def advance_with_recording(self, start_time: float, stop_time: float,
                               outputs: List[str]) -> List[List[Any]]:
        """Advance while recording output values at every micro-step.

        Returns a list of ``[time, val1, val2, …]`` rows (one per micro-step,
        taken before it, including a shorter final step).  Adapters with
        internal solvers (e.g. FMU ME) should override this so that the
        solver's adaptive stepping is honoured.
        """
        rows: List[List[Any]] = []
        for t, dt, full in self.micro_steps(start_time, stop_time):
            if outputs:
                row = [t] + list(self.get_value(outputs, t))
            else:
                row = [t]
            rows.append(row)
            self._do_micro_step(t, dt, full)
        return rows

    # ------------------------------------------------------------------
    # Event / step-size hints for the event-driven macro step (optional)
    # ------------------------------------------------------------------

    def max_step(self, time: float) -> float:
        """Largest macro step (s) this simulator accepts from ``time`` (default ``inf``).

        The world takes the minimum over all simulators, clamped to
        ``[t_macro_min, t_macro]``; hints only matter when ``t_macro_min < t_macro``.
        """
        return math.inf

    def next_event_time(self, time: float) -> float:
        """Absolute time of the next known internal event after ``time`` (default ``inf``).

        Examples: a signal breakpoint, an FMU time event, a storage that will be
        full at the current power.  The world tries to end a macro step exactly
        there (within ``[t_macro_min, t_macro]``).
        """
        return math.inf

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_available_variables(self) -> Dict[str, List[str]]:
        """Return ``{'inputs': [...], 'outputs': [...]}`` of known variable names.

        Override in adapters that can introspect their variable set
        (FMU via modelDescription, pandapower via element tables, etc.).
        The default returns empty lists.
        """
        return {'inputs': [], 'outputs': []}


# ---------------------------------------------------------------------------
# SimEntry — typed replacement for the old simulator_dict list-of-lists
# ---------------------------------------------------------------------------

@dataclass
class SimEntry:
    """Holds everything the ``world`` needs to know about one simulator."""

    sim_type: str                       # 'fmu', 'powerflow', 'pypsa', 'opendss', 'csv', 'signal', 'external', 'script'
    adapter: SimulatorAdapter           # the adapter instance
    step_size: float                    # micro-step size
    outputs: List[str]                  # variable names to record
    inputs: List[str] = field(default_factory=list)
    variable_step: bool = False         # only relevant for CS FMUs today
    mode: Optional[str] = None          # 'ac' or 'dc' for powerflow adapters
    power_unit: str = 'kW'              # default physical unit for power in this simulator
    units: Dict[str, str] = field(default_factory=dict)  # per-variable unit overrides
    model: Any = None                   # model reference: in-memory object, file path, project name, or callable

    @property
    def sim_loc(self) -> Any:
        """Backward compatibility alias for model."""
        return self.model

    @sim_loc.setter
    def sim_loc(self, val: Any):
        self.model = val

    @property
    def pf_mode(self) -> Optional[str]:
        """Backward compatibility alias for mode."""
        return self.mode

    @pf_mode.setter
    def pf_mode(self, val: Optional[str]):
        self.mode = val

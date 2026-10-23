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
from typing import Any, Dict, List, Optional


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


class ParsedConnection(tuple):
    """
    Parsed connection tuple (output, input) with scale, offset, and unit attributes.
    Supports unpacking as (output, input) for 100% backwards compatibility.

    ``scale`` / ``offset`` are the *effective* values applied during exchange
    (unit conversion followed by the user's scale/offset):
    ``value_in = (value_out * unit_scale + unit_offset) * user_scale + user_offset``.
    ``user_scale`` / ``user_offset`` keep what the user specified, so the
    connection can be serialised without applying unit conversion twice.
    """
    scale: float
    offset: float
    from_unit: Optional[str]
    to_unit: Optional[str]
    user_scale: float
    user_offset: float

    def __new__(cls, parsed_output: Any, parsed_input: Any, scale: float = 1.0, offset: float = 0.0,
                from_unit: Optional[str] = None, to_unit: Optional[str] = None,
                user_scale: Optional[float] = None, user_offset: Optional[float] = None):
        instance = super().__new__(cls, (parsed_output, parsed_input))
        instance.scale = float(scale)
        instance.offset = float(offset)
        instance.from_unit = from_unit
        instance.to_unit = to_unit
        instance.user_scale = float(scale if user_scale is None else user_scale)
        instance.user_offset = float(offset if user_offset is None else user_offset)
        return instance


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

    def advance(self, start_time: float, stop_time: float) -> None:
        """Advance the simulator from *start_time* to *stop_time*.

        The default implementation loops ``step()`` at intervals of
        ``self.step_size``.  Adapters with an internal solver (e.g. FMU
        Model Exchange with CVode) should override this to hand the full
        interval to their solver in one call.
        """
        n_steps = round((stop_time - start_time) / self.step_size)
        for i in range(n_steps):
            t = start_time + i * self.step_size
            self.step(t)

    # ------------------------------------------------------------------
    # advance_with_recording() — step + collect intermediate results
    # ------------------------------------------------------------------

    def advance_with_recording(self, start_time: float, stop_time: float,
                               outputs: List[str]) -> List[List[Any]]:
        """Advance while recording output values at every micro-step.

        Returns a list of ``[time, val1, val2, …]`` rows.  The default
        implementation mirrors the old ``record_all`` manual loop.
        Adapters with internal solvers (e.g. FMU ME) should override
        this so that the solver's adaptive stepping is honoured.
        """
        rows: List[List[Any]] = []
        n_steps = round((stop_time - start_time) / self.step_size)
        for i in range(n_steps):
            t = start_time + i * self.step_size
            if outputs:
                row = [t] + list(self.get_value(outputs, t))
            else:
                row = [t]
            rows.append(row)
            self.step(t)
        return rows

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

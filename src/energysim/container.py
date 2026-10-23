# -*- coding: utf-8 -*-
"""
Vectorized Container Simulators for energysim.

Enables high-performance co-simulation of large fleets of Distributed Energy
Resources (DERs) — such as 1,000s of electric vehicles, residential heat pumps,
thermal building envelopes, or rooftop PV arrays — in C-speed vectorized NumPy
operations rather than millions of interpreted Python object dispatches.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from .base import SimulatorAdapter, SimulatorVariableError


class VectorizedContainerAdapter(SimulatorAdapter):
    """
    Base class for vectorized container simulators managing a fleet of N entities.

    Provides:
      - Automatic scalar broadcast for inputs (e.g. ambient temperature or irradiance applied to all N entities).
      - Array inputs for individual entity control (e.g. passing an N-length vector of setpoints).
      - Automatic aggregation metrics (``total_<var>``, ``sum_<var>``, ``mean_<var>``, ``min_<var>``,
        ``max_<var>``). An exact variable name always wins over a prefix interpretation.
      - Per-entity indexing syntax (``<var>[0]``, ``<var>[42]``); out-of-range or negative indices raise
        :class:`SimulatorVariableError`.
      - Full introspection of available inputs, outputs, and aggregations.
      - ``init()`` / ``reset()``: the first ``init()`` snapshots the (possibly user-initialised) states and
        inputs; every later ``init()`` restores that snapshot, so a simulation can be re-run.
        ``reset()`` restores the snapshot (or, if ``init()`` was never called, the registered initial values).
    """

    def __init__(
        self,
        name: str,
        size: int,
        step_size: float = 1.0,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
    ):
        self.name = name
        self.size = int(size)
        if self.size < 1:
            raise ValueError("Container size must be >= 1.")
        self.step_size = float(step_size)
        # Declared (recorded) variable name lists; the world may overwrite them.
        self.inputs: List[str] = list(inputs or [])
        self.outputs: List[str] = list(outputs or [])

        # Internal state/variable stores (mapping: var_name -> 1D numpy array of length `size`)
        self.states: Dict[str, np.ndarray] = {}
        self.input_values: Dict[str, np.ndarray] = {}
        self.aliases: Dict[str, str] = {}

        self._registered_states: Dict[str, np.ndarray] = {}
        self._registered_inputs: Dict[str, np.ndarray] = {}
        self._init_snapshot: Optional[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]] = None

    # -- construction helpers -------------------------------------------------

    def _as_vector(self, value: Union[float, np.ndarray], label: str = "value") -> np.ndarray:
        """Broadcast a scalar to length ``size`` or validate an array of that length (returns a copy)."""
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(self.size, float(arr), dtype=np.float64)
        arr = arr.flatten()
        if len(arr) != self.size:
            raise ValueError(
                f"{label} array length ({len(arr)}) does not match container size ({self.size})."
            )
        return arr.copy()

    def register_state(self, var_name: str, initial_values: Union[float, np.ndarray]) -> None:
        """Register an internal state vector for all N entities."""
        arr = self._as_vector(initial_values, f"Initial values of '{var_name}'")
        self.states[var_name] = arr
        self._registered_states[var_name] = arr.copy()

    def register_input(self, var_name: str, default_value: float = 0.0) -> None:
        """Register an input variable vector for all N entities."""
        arr = np.full(self.size, float(default_value), dtype=np.float64)
        self.input_values[var_name] = arr
        self._registered_inputs[var_name] = arr.copy()

    def register_alias(self, alias_name: str, target_name: str) -> None:
        """Register a user-facing variable alias pointing to a canonical state or input."""
        self.aliases[alias_name] = target_name

    # -- lifecycle ------------------------------------------------------------

    def _restore(self, states: Dict[str, np.ndarray], inputs: Dict[str, np.ndarray]) -> None:
        for k, v in states.items():
            self.states[k][:] = v
        for k, v in inputs.items():
            self.input_values[k][:] = v

    def init(self) -> None:
        """First call: snapshot the current states/inputs. Later calls: restore that snapshot."""
        if self._init_snapshot is None:
            self._init_snapshot = (
                {k: v.copy() for k, v in self.states.items()},
                {k: v.copy() for k, v in self.input_values.items()},
            )
        else:
            self._restore(*self._init_snapshot)

    def reset(self) -> None:
        """Restore the initial states/inputs (the ``init()`` snapshot, else the registered values)."""
        if self._init_snapshot is not None:
            self._restore(*self._init_snapshot)
        else:
            self._restore(self._registered_states, self._registered_inputs)

    # -- addressing -----------------------------------------------------------

    _INDEXED = re.compile(r"^(\w+)\[\s*(-?\d+)\s*\]$")
    _AGG_PREFIXES = {
        'total_': np.sum, 'sum_': np.sum, 'mean_': np.mean, 'min_': np.min, 'max_': np.max,
    }

    def _parse_var_target(self, var_name: str) -> Tuple[str, Optional[int]]:
        """Parse 'var' or 'var[123]' into (canonical_base_name, index)."""
        text = var_name.strip()
        match = self._INDEXED.match(text)
        if match:
            base, idx = match.group(1), int(match.group(2))
        else:
            base, idx = text, None
        return self.aliases.get(base, base), idx

    def _check_index(self, idx: int) -> None:
        if idx < 0 or idx >= self.size:
            raise SimulatorVariableError(
                f"Index {idx} out of range for container '{self.name}' of size {self.size}."
            )

    def _known(self, canonical: str) -> bool:
        return canonical in self.states or canonical in self.input_values

    def _variable_names(self) -> List[str]:
        return sorted(set(self.states) | set(self.input_values) | set(self.aliases))

    def set_value(self, parameters: List[str], values: Any) -> None:
        """
        Set input (or state) values for the container.

        Supports:
          - Scalar broadcast: setting 'temperature' with a scalar updates all N entities.
          - Array update: setting 'p_cmd' with an N-length array updates all N entities element-wise.
          - Indexed update: setting 'p_cmd[5]' updates only entity 5.

        Unknown variable names and out-of-range indices raise :class:`SimulatorVariableError`.
        """
        if not isinstance(values, (list, tuple)):
            val_list = [values] * len(parameters)
        else:
            val_list = list(values)
        if len(val_list) != len(parameters):
            raise ValueError(
                f"set_value received {len(parameters)} parameters but {len(val_list)} values."
            )

        for param, val in zip(parameters, val_list):
            base_name, idx = self._parse_var_target(param)
            if not self._known(base_name):
                raise SimulatorVariableError(
                    f"Variable '{param}' not found in container '{self.name}'. "
                    f"Available: {self._variable_names()}"
                )
            target = (self.input_values if base_name in self.input_values else self.states)[base_name]

            if idx is not None:
                self._check_index(idx)
                target[idx] = float(np.asarray(val, dtype=np.float64).flatten()[0])
                continue

            val_arr = np.asarray(val, dtype=np.float64).flatten()
            if len(val_arr) == 1:
                target[:] = val_arr[0]
            elif len(val_arr) == self.size:
                target[:] = val_arr
            else:
                raise ValueError(
                    f"Input '{param}' length ({len(val_arr)}) must be 1 (broadcast) "
                    f"or match container size ({self.size})."
                )

    def get_value(self, outputs: List[str], time: float = 0.0) -> List[Any]:
        """
        Retrieve values from the container.

        Supports:
          - Aggregations: 'total_<var>', 'sum_<var>', 'mean_<var>', 'min_<var>', 'max_<var>'
          - Indexed values: '<var>[5]'
          - Full vectors: '<var>' returns a copy of the 1D numpy array of length N.
        """
        result = []
        for out in outputs:
            text = out.strip()
            base, idx = self._parse_var_target(text)

            if idx is not None:                       # indexed value
                self._check_index(idx)
                result.append(float(self._get_base_array(base)[idx]))
                continue

            if self._known(base):                     # exact name wins over prefixes
                result.append(self._get_base_array(base).copy())
                continue

            for prefix, fn in self._AGG_PREFIXES.items():
                if text.startswith(prefix):
                    result.append(float(fn(self._get_base_array(text[len(prefix):]))))
                    break
            else:
                result.append(self._get_base_array(base))   # raises SimulatorVariableError
        return result

    def _get_base_array(self, base_name: str) -> np.ndarray:
        """Lookup state or input array by (possibly aliased) variable name."""
        canonical = self.aliases.get(base_name, base_name)
        if canonical in self.states:
            return self.states[canonical]
        if canonical in self.input_values:
            return self.input_values[canonical]
        raise SimulatorVariableError(
            f"Variable '{base_name}' not found in container '{self.name}'. "
            f"Available: {self._variable_names()}"
        )

    def step_partial(self, time: float, dt: float) -> None:
        """Shorter final micro-step: ``step(time, step_size=dt)``."""
        self.step(time, step_size=dt)

    def get_available_variables(self) -> Dict[str, List[str]]:
        """Return declared variable names including ``total_/mean_/min_/max_`` aggregations."""
        var_names = list(self.states.keys()) + list(self.input_values.keys()) + list(self.aliases.keys())
        out_vars = []
        for v in var_names:
            out_vars.extend([v, f"total_{v}", f"mean_{v}", f"min_{v}", f"max_{v}"])
        return {'inputs': list(var_names), 'outputs': out_vars}


# ---------------------------------------------------------------------------
# Pre-built Domain Container Simulators
# ---------------------------------------------------------------------------

class BatteryFleetAdapter(VectorizedContainerAdapter):
    """
    High-performance vectorized battery energy storage / EV fleet adapter.

    Simulates N battery units simultaneously using pure NumPy vector operations:
      - State of Charge (SoC) tracking: dSoC/dt = P * eta / Capacity (charging),
        P / eta / Capacity (discharging); ``P > 0`` charges, ``P < 0`` discharges (AC side)
      - Per-unit capacity and power limits
      - Charge / discharge efficiency (same ``efficiency`` each way)
      - Automatic power clipping when full or empty
      - ``loss_kw``: conversion loss power, so that per step
        ``P_ac * dt - d(stored energy) == loss_kw * dt`` exactly (energy conservation).

    A SoC outside ``[soc_min, soc_max]`` at the start (e.g. set by the user) is clamped into
    that range at the next step; energy conservation holds for in-range states. A step with
    ``step_size <= 0`` is a no-op.
    """

    def __init__(
        self,
        name: str = 'battery_fleet',
        size: int = 100,
        capacity_kwh: Union[float, np.ndarray] = 50.0,
        max_charge_kw: Union[float, np.ndarray] = 11.0,
        max_discharge_kw: Union[float, np.ndarray] = 11.0,
        initial_soc: Union[float, np.ndarray] = 0.5,
        soc_min: float = 0.05,
        soc_max: float = 0.95,
        efficiency: float = 0.95,
        step_size: float = 900.0,
        **kwargs
    ):
        super().__init__(name=name, size=size, step_size=step_size)

        eff = kwargs.get('eta_charge', kwargs.get('eta_discharge', efficiency))
        self.efficiency = float(eff)
        if not 0.0 < self.efficiency <= 1.0:
            raise ValueError("efficiency must be in (0, 1].")
        self.soc_min = float(kwargs.get('min_soc', soc_min))
        self.soc_max = float(kwargs.get('max_soc', soc_max))

        self.capacity = self._as_vector(kwargs.get('capacity', capacity_kwh), 'capacity')
        self.max_charge = self._as_vector(kwargs.get('max_charge', max_charge_kw), 'max_charge')
        self.max_discharge = self._as_vector(kwargs.get('max_discharge', max_discharge_kw), 'max_discharge')

        self.register_state('soc', kwargs.get('soc', initial_soc))
        self.register_state('actual_p_kw', 0.0)
        self.register_state('loss_kw', 0.0)
        self.register_input('p_cmd_kw', 0.0)

        # Aliases
        self.register_alias('p_cmd', 'p_cmd_kw')
        self.register_alias('p_req', 'p_cmd_kw')
        self.register_alias('power', 'actual_p_kw')
        self.register_alias('p_kw', 'actual_p_kw')
        self.register_alias('actual_p', 'actual_p_kw')
        self.register_alias('losses_kw', 'loss_kw')

    def step(self, time: float, step_size: Optional[float] = None) -> None:
        """Advance all N batteries by one micro-step."""
        effective_step = float(step_size) if step_size is not None else self.step_size
        if effective_step <= 0.0:
            return
        eff = self.efficiency
        soc = self.states['soc']
        dt_hours = effective_step / 3600.0

        # Enforce power limits
        p_req = np.clip(self.input_values['p_cmd_kw'], -self.max_discharge, self.max_charge)

        # Power that the SoC headroom still allows (AC side)
        headroom_charge_kwh = np.maximum(0.0, (self.soc_max - soc) * self.capacity)
        max_p_charge_kw = headroom_charge_kwh / dt_hours / eff
        headroom_discharge_kwh = np.maximum(0.0, (soc - self.soc_min) * self.capacity)
        max_p_discharge_kw = headroom_discharge_kwh / dt_hours * eff

        actual_p = np.where(
            p_req > 0.0,
            np.minimum(p_req, max_p_charge_kw),
            np.where(p_req < 0.0, np.maximum(p_req, -max_p_discharge_kw), 0.0),
        )

        # Energy into / out of the cells (kWh) and conversion loss (kW)
        charging = actual_p > 0.0
        delta_kwh = np.where(charging, actual_p * eff, actual_p / eff) * dt_hours
        loss_kw = np.where(charging, actual_p * (1.0 - eff), -actual_p * (1.0 / eff - 1.0))

        self.states['soc'][:] = np.clip(soc + delta_kwh / self.capacity, self.soc_min, self.soc_max)
        self.states['actual_p_kw'][:] = actual_p
        self.states['loss_kw'][:] = loss_kw


class BuildingFleetAdapter(VectorizedContainerAdapter):
    """
    Vectorized lumped thermal building envelope fleet (1R1C equivalent model).

    Simulates N residential buildings simultaneously (explicit Euler):
      dT_in/dt = (T_ambient - T_in) / (R * C) + (Q_heat + Q_solar) / C

    Explicit Euler is accurate/stable only for ``step_size << R*C`` (default 15 min vs 37.5 h).
    """

    def __init__(
        self,
        name: str = 'building_fleet',
        size: int = 100,
        c_th_kwh_per_k: Union[float, np.ndarray] = 15.0,     # Thermal capacitance (kWh/K)
        r_th_k_per_kw: Union[float, np.ndarray] = 2.5,       # Thermal resistance (K/kW)
        initial_t_indoor: Union[float, np.ndarray] = 20.0,   # Initial indoor temp (deg C)
        step_size: float = 900.0,
        **kwargs
    ):
        super().__init__(name=name, size=size, step_size=step_size)

        self.c_th = self._as_vector(kwargs.get('c_th', c_th_kwh_per_k), 'c_th')
        self.r_th = self._as_vector(kwargs.get('r_th', r_th_k_per_kw), 'r_th')

        t_in = kwargs.get('initial_t_in', kwargs.get('t_in', initial_t_indoor))
        self.register_state('t_indoor', t_in)
        self.register_input('t_ambient', 10.0)      # Outdoor temperature (deg C)
        self.register_input('q_heat_kw', 0.0)       # Heating power input (kW)
        self.register_input('q_solar_kw', 0.0)      # Solar thermal gains (kW)

        # Aliases
        self.register_alias('t_in', 't_indoor')
        self.register_alias('t_in_degc', 't_indoor')
        self.register_alias('t_indoor_degc', 't_indoor')
        self.register_alias('t_amb', 't_ambient')
        self.register_alias('t_amb_degc', 't_ambient')
        self.register_alias('t_ambient_degc', 't_ambient')
        self.register_alias('p_heat', 'q_heat_kw')
        self.register_alias('p_heat_kw', 'q_heat_kw')
        self.register_alias('q_heat', 'q_heat_kw')
        self.register_alias('q_solar', 'q_solar_kw')
        self.register_alias('p_solar_kw', 'q_solar_kw')

    def step(self, time: float, step_size: Optional[float] = None) -> None:
        """Advance thermal state of all N buildings."""
        effective_step = float(step_size) if step_size is not None else self.step_size
        t_in = self.states['t_indoor']
        dt_hours = effective_step / 3600.0

        q_loss = (t_in - self.input_values['t_ambient']) / self.r_th                    # kW
        q_net = self.input_values['q_heat_kw'] + self.input_values['q_solar_kw'] - q_loss
        self.states['t_indoor'][:] = t_in + (q_net / self.c_th) * dt_hours


class PVFleetAdapter(VectorizedContainerAdapter):
    """
    Vectorized Solar Photovoltaic (PV) fleet adapter.

    Simulates N solar rooftop systems with heterogeneous peak capacities:
      T_cell = T_amb + (G / 800) * 25
      P = P_peak * (G / 1000) * [1 + gamma * (T_cell - 25)] * PR     (clipped at >= 0)
    """

    def __init__(
        self,
        name: str = 'pv_fleet',
        size: int = 100,
        peak_kw: Union[float, np.ndarray] = 5.0,
        gamma_temp_coeff: float = -0.004,                    # -0.4% / deg C
        performance_ratio: float = 0.85,
        step_size: float = 900.0,
        **kwargs
    ):
        super().__init__(name=name, size=size, step_size=step_size)

        self.gamma = float(kwargs.get('gamma_p', kwargs.get('gamma', gamma_temp_coeff)))
        self.pr = float(kwargs.get('pr', performance_ratio))
        self.peak_kw = self._as_vector(kwargs.get('peak_power_kw', kwargs.get('p_peak', peak_kw)), 'peak_kw')

        self.register_state('p_kw', 0.0)
        self.register_input('irradiance_w_m2', 0.0)
        self.register_input('t_ambient', 25.0)

        # Aliases
        self.register_alias('p_pv', 'p_kw')
        self.register_alias('p_pv_kw', 'p_kw')
        self.register_alias('g_irr', 'irradiance_w_m2')
        self.register_alias('g_irr_w_m2', 'irradiance_w_m2')
        self.register_alias('irradiance', 'irradiance_w_m2')
        self.register_alias('t_amb', 't_ambient')
        self.register_alias('t_amb_degc', 't_ambient')

    def step(self, time: float, step_size: Optional[float] = None) -> None:
        """Compute active power generation for all N PV systems."""
        g = np.maximum(0.0, self.input_values['irradiance_w_m2'])
        t_cell = self.input_values['t_ambient'] + (g / 800.0) * 25.0
        p_out = self.peak_kw * (g / 1000.0) * (1.0 + self.gamma * (t_cell - 25.0)) * self.pr
        self.states['p_kw'][:] = np.maximum(0.0, p_out)


class VectorizedFunctionAdapter(VectorizedContainerAdapter):
    """
    User-defined vectorized container adapter using arbitrary NumPy functions.

    Allows users to pass a custom state update function:
      func(time, states, inputs, step_size) -> updated_states_dict
    """

    def __init__(
        self,
        name: str,
        size: int,
        update_func: Optional[Callable[[float, Dict[str, np.ndarray], Dict[str, np.ndarray], float], Dict[str, np.ndarray]]] = None,
        initial_states: Optional[Dict[str, Union[float, np.ndarray]]] = None,
        inputs: Optional[List[str]] = None,
        step_size: float = 1.0,
        **kwargs
    ):
        super().__init__(name=name, size=size, step_size=step_size)
        fn = update_func or kwargs.get('update_fn') or kwargs.get('func')
        if fn is None:
            raise ValueError("VectorizedFunctionAdapter requires an update_func callable.")
        self.update_func = fn

        init_st = initial_states or kwargs.get('states', {})
        for s_name, init_val in init_st.items():
            self.register_state(s_name, init_val)

        for inp in (inputs or kwargs.get('input_names') or []):
            self.register_input(inp, 0.0)

    def step(self, time: float, step_size: Optional[float] = None) -> None:
        """Call user-provided vectorized update function."""
        effective_step = float(step_size) if step_size is not None else self.step_size
        updated = self.update_func(time, self.states, self.input_values, effective_step)
        if isinstance(updated, dict):
            for k, arr in updated.items():
                if k in self.states:
                    self.states[k][:] = arr

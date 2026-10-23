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
      - Automatic aggregation metrics (e.g. 'total_<var>', 'mean_<var>', 'min_<var>', 'max_<var>').
      - Per-entity indexing syntax (e.g. '<var>[0]', '<var>[42]').
      - Full introspection of available inputs, outputs, and aggregations.
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
        self.step_size = float(step_size)
        self.inputs = list(inputs or [])
        self.outputs = list(outputs or [])

        # Internal state/variable stores (mapping: var_name -> 1D numpy array of length `size`)
        self.states: Dict[str, np.ndarray] = {}
        self.input_values: Dict[str, np.ndarray] = {}
        self.aliases: Dict[str, str] = {}

    def register_state(self, var_name: str, initial_values: Union[float, np.ndarray]) -> None:
        """Register an internal state vector for all N entities."""
        if isinstance(initial_values, (int, float, np.floating, np.integer)):
            arr = np.full(self.size, float(initial_values), dtype=np.float64)
        else:
            arr = np.asarray(initial_values, dtype=np.float64)
            if len(arr) != self.size:
                raise ValueError(
                    f"Initial values array length ({len(arr)}) does not match container size ({self.size})."
                )
        self.states[var_name] = arr

    def register_input(self, var_name: str, default_value: float = 0.0) -> None:
        """Register an input variable vector for all N entities."""
        self.input_values[var_name] = np.full(self.size, float(default_value), dtype=np.float64)

    def register_alias(self, alias_name: str, target_name: str) -> None:
        """Register a user-facing variable alias pointing to a canonical state or input."""
        self.aliases[alias_name] = target_name

    def _parse_var_target(self, var_name: str) -> Tuple[str, Optional[int]]:
        """Parse 'var' or 'var[123]' into (base_name, index)."""
        match = re.match(r"^(\w+)\[(\d+)\]$", var_name.strip())
        if match:
            base = match.group(1)
            base = self.aliases.get(base, base)
            return base, int(match.group(2))
        base = var_name.strip()
        return self.aliases.get(base, base), None

    def set_value(self, parameters: List[str], values: Any) -> None:
        """
        Set input values for the container.

        Supports:
          - Scalar broadcast: setting 'temperature' with a scalar updates all N entities.
          - Array update: setting 'p_cmd' with an N-length array updates all N entities element-wise.
          - Indexed update: setting 'p_cmd[5]' updates only entity 5.
        """
        if not isinstance(values, (list, tuple)):
            val_list = [values] * len(parameters)
        else:
            val_list = values

        for param, val in zip(parameters, val_list):
            base_name, idx = self._parse_var_target(param)

            if base_name not in self.input_values and base_name not in self.states:
                # Treat as dynamically registered input
                self.register_input(base_name, 0.0)

            target_store = self.input_values if base_name in self.input_values else self.states

            if idx is not None:
                if idx < 0 or idx >= self.size:
                    raise SimulatorVariableError(
                        f"Index {idx} out of range for container '{self.name}' of size {self.size}."
                    )
                target_store[base_name][idx] = float(val[0] if isinstance(val, (list, tuple)) else val)
            else:
                # Array or scalar
                if isinstance(val, (list, tuple, np.ndarray)):
                    val_arr = np.asarray(val, dtype=np.float64).flatten()
                    if len(val_arr) == 1:
                        target_store[base_name][:] = val_arr[0]
                    elif len(val_arr) == self.size:
                        target_store[base_name][:] = val_arr
                    else:
                        raise ValueError(
                            f"Input '{param}' length ({len(val_arr)}) must be 1 (broadcast) "
                            f"or match container size ({self.size})."
                        )
                else:
                    target_store[base_name][:] = float(val)

    def get_value(self, outputs: List[str], time: float = 0.0) -> List[Any]:
        """
        Retrieve values from the container.

        Supports:
          - Aggregations: 'total_<var>', 'mean_<var>', 'sum_<var>', 'min_<var>', 'max_<var>'
          - Indexed values: '<var>[5]'
          - Full vectors: '<var>' returns the 1D numpy array of length N.
        """
        result = []
        for out in outputs:
            out_str = out.strip()

            # 1. Check aggregation prefixes
            if out_str.startswith(('total_', 'sum_')):
                base = out_str.split('_', 1)[1]
                arr = self._get_base_array(base)
                result.append(float(np.sum(arr)))
                continue

            if out_str.startswith('mean_'):
                base = out_str.split('_', 1)[1]
                arr = self._get_base_array(base)
                result.append(float(np.mean(arr)))
                continue

            if out_str.startswith('min_'):
                base = out_str.split('_', 1)[1]
                arr = self._get_base_array(base)
                result.append(float(np.min(arr)))
                continue

            if out_str.startswith('max_'):
                base = out_str.split('_', 1)[1]
                arr = self._get_base_array(base)
                result.append(float(np.max(arr)))
                continue

            # 2. Check indexed output (e.g. 'soc[42]')
            base, idx = self._parse_var_target(out_str)
            if idx is not None:
                arr = self._get_base_array(base)
                if idx < 0 or idx >= self.size:
                    raise SimulatorVariableError(
                        f"Output index {idx} out of range for container '{self.name}' (size={self.size})."
                    )
                result.append(float(arr[idx]))
                continue

            # 3. Full vector
            arr = self._get_base_array(base)
            result.append(arr.copy())

        return result

    def _get_base_array(self, base_name: str) -> np.ndarray:
        """Lookup state or input array by base variable name."""
        canonical = self.aliases.get(base_name, base_name)
        if canonical in self.states:
            return self.states[canonical]
        if canonical in self.input_values:
            return self.input_values[canonical]
        raise SimulatorVariableError(
            f"Variable '{base_name}' not found in container '{self.name}'. "
            f"Available: {list(self.states.keys()) + list(self.input_values.keys()) + list(self.aliases.keys())}"
        )

    def get_available_variables(self) -> Dict[str, List[str]]:
        """Return declared variable names including aggregations."""
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
      - State of Charge (SoC) tracking: dSoC/dt = P * eta / Capacity
      - Per-unit capacity and power limits
      - Charge / discharge efficiency
      - Automatic power clipping when full or empty
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
        self.soc_min = float(kwargs.get('min_soc', soc_min))
        self.soc_max = float(kwargs.get('max_soc', soc_max))

        cap = kwargs.get('capacity', capacity_kwh)
        if isinstance(cap, (int, float)):
            self.capacity = np.full(self.size, float(cap), dtype=np.float64)
        else:
            self.capacity = np.asarray(cap, dtype=np.float64)

        mc = kwargs.get('max_charge', max_charge_kw)
        if isinstance(mc, (int, float)):
            self.max_charge = np.full(self.size, float(mc), dtype=np.float64)
        else:
            self.max_charge = np.asarray(mc, dtype=np.float64)

        md = kwargs.get('max_discharge', max_discharge_kw)
        if isinstance(md, (int, float)):
            self.max_discharge = np.full(self.size, float(md), dtype=np.float64)
        else:
            self.max_discharge = np.asarray(md, dtype=np.float64)

        init_soc = kwargs.get('soc', initial_soc)
        self.register_state('soc', init_soc)
        self.register_state('actual_p_kw', 0.0)
        self.register_input('p_cmd_kw', 0.0)

        # Aliases
        self.register_alias('p_cmd', 'p_cmd_kw')
        self.register_alias('p_req', 'p_cmd_kw')
        self.register_alias('power', 'actual_p_kw')
        self.register_alias('p_kw', 'actual_p_kw')
        self.register_alias('actual_p', 'actual_p_kw')

    def step(self, time: float, step_size: Optional[float] = None) -> None:
        """Advance all N batteries by one micro-step."""
        effective_step = float(step_size) if step_size is not None else self.step_size
        p_cmd = self.input_values['p_cmd_kw']
        soc = self.states['soc']
        dt_hours = effective_step / 3600.0

        # Enforce power limits
        p_req = np.clip(p_cmd, -self.max_discharge, self.max_charge)

        # Compute allowable power based on SoC headroom
        charging_mask = p_req > 0
        discharging_mask = p_req < 0

        # Maximum energy that can be absorbed (kWh)
        headroom_charge_kwh = np.maximum(0.0, (self.soc_max - soc) * self.capacity)
        max_p_charge_kw = (headroom_charge_kwh / dt_hours) / max(self.efficiency, 1e-6)

        # Maximum energy that can be delivered (kWh)
        headroom_discharge_kwh = np.maximum(0.0, (soc - self.soc_min) * self.capacity)
        max_p_discharge_kw = (headroom_discharge_kwh / dt_hours) * self.efficiency

        actual_p = np.zeros(self.size, dtype=np.float64)
        actual_p[charging_mask] = np.minimum(p_req[charging_mask], max_p_charge_kw[charging_mask])
        actual_p[discharging_mask] = np.maximum(p_req[discharging_mask], -max_p_discharge_kw[discharging_mask])

        # Integrate SoC
        delta_kwh = np.zeros(self.size, dtype=np.float64)
        delta_kwh[charging_mask] = actual_p[charging_mask] * self.efficiency * dt_hours
        delta_kwh[discharging_mask] = (actual_p[discharging_mask] / max(self.efficiency, 1e-6)) * dt_hours

        new_soc = soc + (delta_kwh / self.capacity)
        self.states['soc'][:] = np.clip(new_soc, self.soc_min, self.soc_max)
        self.states['actual_p_kw'][:] = actual_p


class BuildingFleetAdapter(VectorizedContainerAdapter):
    """
    Vectorized lumped thermal building envelope fleet (1R1C equivalent model).

    Simulates N residential buildings simultaneously:
      dT_in/dt = (T_ambient - T_in) / (R * C) + (Q_heat + Q_solar) / C
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

        c_th = kwargs.get('c_th', c_th_kwh_per_k)
        if isinstance(c_th, (int, float)):
            self.c_th = np.full(self.size, float(c_th), dtype=np.float64)
        else:
            self.c_th = np.asarray(c_th, dtype=np.float64)

        r_th = kwargs.get('r_th', r_th_k_per_kw)
        if isinstance(r_th, (int, float)):
            self.r_th = np.full(self.size, float(r_th), dtype=np.float64)
        else:
            self.r_th = np.asarray(r_th, dtype=np.float64)

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
        t_amb = self.input_values['t_ambient']
        q_heat = self.input_values['q_heat_kw']
        q_solar = self.input_values['q_solar_kw']
        dt_hours = effective_step / 3600.0

        # Heat loss through envelope (kW)
        q_loss = (t_in - t_amb) / self.r_th

        # Net thermal power into building (kW)
        q_net = q_heat + q_solar - q_loss

        # Temperature change
        dt_indoor = (q_net / self.c_th) * dt_hours
        self.states['t_indoor'][:] = t_in + dt_indoor


class PVFleetAdapter(VectorizedContainerAdapter):
    """
    Vectorized Solar Photovoltaic (PV) fleet adapter.

    Simulates N solar rooftop systems with heterogeneous peak capacities:
      P_dc = P_peak * (G / 1000) * [1 + gamma * (T_cell - 25)] * PR
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

        gamma = kwargs.get('gamma_p', kwargs.get('gamma', gamma_temp_coeff))
        self.gamma = float(gamma)
        pr = kwargs.get('pr', performance_ratio)
        self.pr = float(pr)

        p_peak = kwargs.get('peak_power_kw', kwargs.get('p_peak', peak_kw))
        if isinstance(p_peak, (int, float)):
            self.peak_kw = np.full(self.size, float(p_peak), dtype=np.float64)
        else:
            self.peak_kw = np.asarray(p_peak, dtype=np.float64)

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
        t_amb = self.input_values['t_ambient']

        # Approximate cell temperature: T_cell = T_amb + (G / 800) * 25
        t_cell = t_amb + (g / 800.0) * 25.0
        temp_correction = 1.0 + self.gamma * (t_cell - 25.0)

        # Output in kW
        p_out = self.peak_kw * (g / 1000.0) * temp_correction * self.pr
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

        inps = inputs or kwargs.get('input_names') or []
        for inp in inps:
            self.register_input(inp, 0.0)

    def step(self, time: float, step_size: Optional[float] = None) -> None:
        """Call user-provided vectorized update function."""
        effective_step = float(step_size) if step_size is not None else self.step_size
        updated = self.update_func(time, self.states, self.input_values, effective_step)
        if isinstance(updated, dict):
            for k, arr in updated.items():
                if k in self.states:
                    self.states[k][:] = arr

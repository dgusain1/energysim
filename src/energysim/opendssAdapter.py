# -*- coding: utf-8 -*-
"""
OpenDSS adapter for energysim using OpenDSSDirect.py.

Provides co-simulation capabilities for distribution systems modeled in
Electric Power Research Institute's (EPRI) OpenDSS.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from .base import SimulatorAdapter, SimulatorElementNotFoundError, EnergysimError


class opendss_adapter(SimulatorAdapter):
    """
    Adapter for OpenDSS distribution system simulation via OpenDSSDirect.py.

    Parameters
    ----------
    dss_name : str
        Unique identifier for the OpenDSS simulator.
    dss_location : str
        Path to the primary OpenDSS script (e.g. Master.dss).
    inputs : list of str, optional
        Input variables to receive from other simulators, format:
        ``'ElementType.ElementName.Property'`` (e.g. ``'Load.Load1.kW'``,
        ``'Generator.PV.kW'``, ``'Load.Load1.kvar'``).
    outputs : list of str, optional
        Output variables to record, format:
        ``'Bus.BusName.vm_pu'``, ``'Bus.BusName.kv'``, ``'Bus.BusName.vang'``,
        ``'Line.LineName.p_kw'``, ``'Circuit.losses_kw'``, ``'Circuit.total_power_kw'``.
    step_size : float, optional
        Micro time step size in seconds.
    """

    def __init__(
        self,
        dss_name: str,
        dss_location: Optional[str] = None,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        step_size: float = 900.0,
        mode: str = 'ac',
        pf: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.dss_name = dss_name
        target = model if model is not None else dss_location
        self.dss_location = os.path.abspath(target) if target else ""
        self.inputs = list(inputs or [])
        self.outputs = list(outputs or [])
        self.step_size = step_size
        
        raw_mode = (mode if (mode != 'ac' or pf is None) else pf) or 'ac'
        raw_mode = raw_mode.lower()
        if raw_mode in ('ac', 'pf', 'snapshot'):
            self.mode = 'ac'
        elif raw_mode in ('dc', 'dcpf', 'direct'):
            self.mode = 'dc'
        else:
            raise ValueError(
                f"Unsupported calculation mode '{raw_mode}'. "
                f"energysim only supports 'ac' and 'dc' power flow modes."
            )
        self.pf = self.mode

        self._dss = None
        self._input_cache: Dict[str, float] = {}

    def _get_dss(self):
        """Lazy loader for opendssdirect module."""
        if self._dss is None:
            try:
                import opendssdirect as dss
                self._dss = dss
            except ImportError as err:
                raise ImportError(
                    f"OpenDSS simulation requires 'opendssdirect.py'. "
                    f"Please install it using: pip install opendssdirect.py"
                ) from err
        return self._dss

    def init(self) -> None:
        """Initialize OpenDSS circuit and run initial solve."""
        dss = self._get_dss()
        if self.dss_location:
            if not os.path.exists(self.dss_location):
                raise FileNotFoundError(
                    f"OpenDSS model file not found: {self.dss_location}"
                )
            dss.run_command(f'Redirect "{self.dss_location}"')
        if self.mode == 'dc':
            dss.run_command("Set mode=Direct")
        else:
            dss.run_command("Set mode=Snapshot")
        dss.Solution.Solve()

    def set_value(self, parameters: List[str], values: List[Any]) -> None:
        """
        Apply inputs to OpenDSS components.

        Expected format: 'ElementType.ElementName.Property' (e.g. 'Load.Load1.kW')
        """
        dss = self._get_dss()
        for param, val in zip(parameters, values):
            self._input_cache[param] = float(val)
            parts = param.split('.')
            if len(parts) >= 3:
                ele_type, ele_name, prop = parts[0], parts[1], parts[2]
                cmd = f"edit {ele_type}.{ele_name} {prop}={val}"
                dss.run_command(cmd)
            elif len(parts) == 2:
                ele_name, prop = parts[0], parts[1]
                cmd = f"edit {ele_name} {prop}={val}"
                dss.run_command(cmd)
            else:
                raise SimulatorElementNotFoundError(
                    f"Malformed OpenDSS input identifier '{param}'. "
                    f"Expected 'Type.Name.Property' or 'Name.Property'."
                )

    def get_value(self, parameters: List[str], time: Optional[float] = None) -> List[float]:
        """
        Query values from the solved OpenDSS circuit.
        """
        dss = self._get_dss()
        results: List[float] = []

        for param in parameters:
            parts = param.split('.')
            target_type = parts[0].lower()

            if target_type == 'circuit':
                prop = parts[1].lower() if len(parts) > 1 else 'losses_kw'
                if prop in ('losses_kw', 'losses'):
                    losses = dss.Circuit.Losses()  # [watts, vars]
                    results.append(float(losses[0]) / 1000.0 if losses else 0.0)
                elif prop in ('losses_kvar',):
                    losses = dss.Circuit.Losses()
                    results.append(float(losses[1]) / 1000.0 if losses and len(losses) > 1 else 0.0)
                elif prop in ('total_power_kw', 'power_kw'):
                    total_p = dss.Circuit.TotalPower()
                    results.append(float(total_p[0]) if total_p else 0.0)
                else:
                    results.append(0.0)

            elif target_type == 'bus':
                if len(parts) < 3:
                    raise SimulatorElementNotFoundError(
                        f"Bus variable query '{param}' missing property. Expected 'Bus.<busname>.<prop>'."
                    )
                bus_name = parts[1]
                prop = parts[2].lower()
                dss.Circuit.SetActiveBus(bus_name)
                if prop in ('vm_pu', 'vmag_pu', 'v_pu'):
                    pu_v = dss.Bus.VMagAngle()
                    # returns [mag1, ang1, mag2, ang2, ...]
                    results.append(float(dss.Bus.PuVoltage()[0]) if dss.Bus.PuVoltage() else 1.0)
                elif prop in ('kv', 'vm_kv', 'v'):
                    kv_v = dss.Bus.kVBase()
                    results.append(float(kv_v))
                elif prop in ('vang', 'va_deg', 'va'):
                    ang = dss.Bus.VMagAngle()
                    results.append(float(ang[1]) if len(ang) > 1 else 0.0)
                else:
                    raise SimulatorElementNotFoundError(f"Unknown bus property '{prop}' in '{param}'.")

            elif target_type == 'line':
                if len(parts) < 3:
                    raise SimulatorElementNotFoundError(
                        f"Line variable query '{param}' missing property. Expected 'Line.<linename>.<prop>'."
                    )
                line_name = parts[1]
                prop = parts[2].lower()
                dss.Lines.Name(line_name)
                if prop in ('p_kw', 'p'):
                    powers = dss.CktElement.Powers()  # [p1, q1, p2, q2, ...]
                    results.append(float(powers[0]) if powers else 0.0)
                elif prop in ('q_kvar', 'q'):
                    powers = dss.CktElement.Powers()
                    results.append(float(powers[1]) if len(powers) > 1 else 0.0)
                else:
                    results.append(0.0)

            else:
                # Generic CktElement query
                if len(parts) >= 2:
                    dss.Circuit.SetActiveElement(f"{parts[0]}.{parts[1]}")
                    powers = dss.CktElement.Powers()
                    results.append(float(powers[0]) if powers else 0.0)
                else:
                    raise SimulatorElementNotFoundError(
                        f"Cannot resolve OpenDSS parameter query '{param}'."
                    )

        return results

    def step(self, time: float) -> None:
        """Advance OpenDSS solution by one time step."""
        dss = self._get_dss()
        dss.Solution.Solve()

    def cleanup(self) -> None:
        """Clear OpenDSS circuit resources."""
        if self._dss is not None:
            self._dss.run_command("Clear")
            self._dss = None

    def get_available_variables(self) -> Dict[str, List[str]]:
        return {'inputs': list(self.inputs), 'outputs': list(self.outputs)}

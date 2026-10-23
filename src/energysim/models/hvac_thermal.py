# -*- coding: utf-8 -*-
"""
Heating, Cooling, and Thermal Storage Models for energysim.

Includes:
- IndustrialHeatPump: Temperature-lift dependent COP heat pump.
- ElectricBoiler: Fast-acting electrode/resistance boiler for steam and district heat.
- ElectricChiller: Vapor-compression chiller with ambient-dependent COP.
- AbsorptionChiller: Thermal-driven absorption chiller utilizing waste heat/steam.
- ThermalStorageTank: Stratified sensible hot/cold water thermal energy storage.
"""

from __future__ import annotations
import math
from typing import Optional, Sequence
import numpy as np

from .common.base_component import ModelComponent
from .common.curve import Curve1D, Surface2D


class IndustrialHeatPump(ModelComponent):
    """
    Industrial Heat Pump with temperature-lift dependent COP.

    Parameters
    ----------
    q_max_thermal_mw : float
        Maximum thermal heating capacity in MW.
    carnot_efficiency : float, default 0.50
        Empirical factor of theoretical Carnot efficiency.
    t_source_nom_c : float, default 15.0
        Nominal source temperature in C (e.g. ambient air, river, waste water).
    t_sink_nom_c : float, default 75.0
        Nominal sink delivery temperature in C (district heat or process water).
    """

    def __init__(
        self,
        name: str = "HeatPump",
        q_max_thermal_mw: float = 10.0,
        carnot_efficiency: float = 0.50,
        t_source_nom_c: float = 15.0,
        t_sink_nom_c: float = 75.0,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.q_max = q_max_thermal_mw
        self.eta_carnot = carnot_efficiency
        self.t_source_nom = t_source_nom_c
        self.t_sink_nom = t_sink_nom_c

        self.q_actual_mw = 0.0
        self.p_elec_mw = 0.0
        self.cop = 3.5
        self.q_absorbed_source_mw = 0.0

        self.register_variables(
            inputs=["Q_heat_cmd_MW", "T_source", "T_sink"],
            outputs=["P_elec_MW", "Q_heat_MW", "COP", "Q_absorbed_source_MW"],
        )

    def step_model(self, dt: float, time: float) -> None:
        q_cmd = float(self._inputs_val.get("Q_heat_cmd_MW", 0.0))
        t_src = float(self._inputs_val.get("T_source", self.t_source_nom))
        t_snk = float(self._inputs_val.get("T_sink", self.t_sink_nom))

        self.q_actual_mw = min(self.q_max, max(0.0, q_cmd))

        if self.q_actual_mw > 0.01:
            # Carnot COP = T_sink / (T_sink - T_source) in Kelvin
            t_snk_k = t_snk + 273.15
            t_src_k = t_src + 273.15
            delta_t = max(5.0, t_snk_k - t_src_k)
            cop_carnot = t_snk_k / delta_t
            self.cop = max(1.2, min(7.0, self.eta_carnot * cop_carnot))

            self.p_elec_mw = self.q_actual_mw / self.cop
            self.q_absorbed_source_mw = self.q_actual_mw - self.p_elec_mw
        else:
            self.p_elec_mw = 0.0
            self.q_absorbed_source_mw = 0.0

        self._outputs_val["P_elec_MW"] = self.p_elec_mw
        self._outputs_val["Q_heat_MW"] = self.q_actual_mw
        self._outputs_val["COP"] = self.cop
        self._outputs_val["Q_absorbed_source_MW"] = self.q_absorbed_source_mw


class ElectricBoiler(ModelComponent):
    """
    Electrode / Electric Resistance Boiler for rapid steam generation.
    """

    def __init__(
        self,
        name: str = "ElectricBoiler",
        p_max_mw: float = 20.0,
        efficiency: float = 0.99,
        ramp_rate_mw_min: float = 40.0,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.p_max = p_max_mw
        self.eff = efficiency
        self.ramp_rate = ramp_rate_mw_min

        self.p_actual = 0.0
        self.q_thermal_mw = 0.0

        self.register_variables(
            inputs=["P_cmd"],
            outputs=["P_actual", "Q_thermal_MW"],
        )

    def step_model(self, dt: float, time: float) -> None:
        p_cmd = float(self._inputs_val.get("P_cmd", 0.0))
        target = min(self.p_max, max(0.0, p_cmd))

        max_delta = self.ramp_rate * (dt / 60.0)
        if abs(target - self.p_actual) <= max_delta:
            self.p_actual = target
        elif target > self.p_actual:
            self.p_actual += max_delta
        else:
            self.p_actual -= max_delta

        self.q_thermal_mw = self.p_actual * self.eff
        self._outputs_val["P_actual"] = self.p_actual
        self._outputs_val["Q_thermal_MW"] = self.q_thermal_mw


class ElectricChiller(ModelComponent):
    """
    Electric vapor-compression chiller.
    """

    def __init__(
        self,
        name: str = "ElectricChiller",
        q_cooling_max_mw: float = 5.0,
        cop_nominal: float = 5.5,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.q_max = q_cooling_max_mw
        self.cop_nom = cop_nominal

        self.q_cooling_actual_mw = 0.0
        self.p_elec_mw = 0.0
        self.cop = cop_nominal

        self.register_variables(
            inputs=["Q_cool_cmd_MW", "T_amb"],
            outputs=["P_elec_MW", "Q_cool_actual_MW", "COP"],
        )

    def step_model(self, dt: float, time: float) -> None:
        q_cmd = float(self._inputs_val.get("Q_cool_cmd_MW", 0.0))
        t_amb = float(self._inputs_val.get("T_amb", 25.0))

        self.q_cooling_actual_mw = min(self.q_max, max(0.0, q_cmd))

        if self.q_cooling_actual_mw > 0.01:
            # Ambient derating: COP degrades by ~1.5% per C above 25 C
            cop_adj = self.cop_nom * (1.0 - 0.015 * (t_amb - 25.0))
            self.cop = max(2.0, min(8.0, cop_adj))
            self.p_elec_mw = self.q_cooling_actual_mw / self.cop
        else:
            self.p_elec_mw = 0.0

        self._outputs_val["P_elec_MW"] = self.p_elec_mw
        self._outputs_val["Q_cool_actual_MW"] = self.q_cooling_actual_mw
        self._outputs_val["COP"] = self.cop


class AbsorptionChiller(ModelComponent):
    """
    Waste-heat / steam driven absorption chiller.
    Replaces electricity with thermal heat to produce chilled water.
    """

    def __init__(
        self,
        name: str = "AbsorptionChiller",
        q_cooling_max_mw: float = 5.0,
        cop_thermal: float = 0.85,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.q_max = q_cooling_max_mw
        self.cop_th = cop_thermal

        self.q_cooling_actual = 0.0
        self.q_heat_consumed = 0.0

        self.register_variables(
            inputs=["Q_cool_cmd_MW", "Q_waste_heat_available_MW"],
            outputs=["Q_cool_actual_MW", "Q_heat_consumed_MW"],
        )

    def step_model(self, dt: float, time: float) -> None:
        q_cmd = float(self._inputs_val.get("Q_cool_cmd_MW", 0.0))
        q_avail = float(self._inputs_val.get("Q_waste_heat_available_MW", 1e6))

        q_req_heat = q_cmd / self.cop_th
        heat_used = min(q_avail, q_req_heat)
        self.q_cooling_actual = min(self.q_max, heat_used * self.cop_th)
        self.q_heat_consumed = self.q_cooling_actual / self.cop_th

        self._outputs_val["Q_cool_actual_MW"] = self.q_cooling_actual
        self._outputs_val["Q_heat_consumed_MW"] = self.q_heat_consumed


class ThermalStorageTank(ModelComponent):
    """
    Stratified hot/cold water sensible Thermal Energy Storage (TES).
    """

    def __init__(
        self,
        name: str = "ThermalTank",
        capacity_mwh_th: float = 50.0,
        max_flow_rate_mw_th: float = 10.0,
        loss_pct_per_day: float = 2.0,
        soc_init: float = 0.5,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.capacity_mwh = capacity_mwh_th
        self.max_flow = max_flow_rate_mw_th
        self.loss_rate_s = (loss_pct_per_day / 100.0) / 86400.0

        self.soc = float(soc_init)
        self.q_actual_mw = 0.0

        self.register_variables(
            inputs=["Q_cmd_MW"],
            outputs=["Q_actual_MW", "SoC"],
        )

    def step_model(self, dt: float, time: float) -> None:
        q_cmd = float(self._inputs_val.get("Q_cmd_MW", 0.0))
        q_flow = max(-self.max_flow, min(self.max_flow, q_cmd))

        if q_flow > 0 and self.soc >= 0.99:
            q_flow = 0.0
        elif q_flow < 0 and self.soc <= 0.01:
            q_flow = 0.0

        d_soc = (q_flow * dt) / (self.capacity_mwh * 3600.0)
        d_soc -= self.loss_rate_s * self.soc * dt

        self.soc = max(0.0, min(1.0, self.soc + d_soc))
        self.q_actual_mw = q_flow

        self._outputs_val["Q_actual_MW"] = self.q_actual_mw
        self._outputs_val["SoC"] = self.soc

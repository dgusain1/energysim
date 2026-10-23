# -*- coding: utf-8 -*-
"""
Heating, Cooling, and Thermal Storage Models for energysim.

Includes:
- IndustrialHeatPump: Temperature-lift dependent COP heat pump.
- ElectricBoiler: Fast-acting electrode/resistance boiler for steam and district heat.
- ElectricChiller: Vapor-compression chiller with ambient-dependent COP.
- AbsorptionChiller: Thermal-driven absorption chiller utilizing waste heat/steam.
- ThermalStorageTank: Sensible hot/cold water thermal energy storage
  (single energy bucket with proportional standing loss; no stratification).
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

    COP = carnot_efficiency * T_sink / (T_sink - T_source) in Kelvin, with the
    lift floored at 5 K and COP clamped to [1.2, 7.0]. The COP is evaluated
    continuously from the current source/sink temperatures (also when idle).
    Capacity is not derated with temperature and there is no start-up dynamic.

    Validity: source -10..60 C, sink 30..120 C, lift 10..90 K.

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

    Inputs
    ------
    T_source : degC
        Dynamic evaporator heat source inlet temperature
    T_sink : degC
        Dynamic condenser delivery temperature

    Outputs
    -------
    Q_heat_MW : MW
        Useful thermal heat injected into heat network
    COP : 1
        Actual operating Coefficient of Performance
    """

    units = {
        "Q_heat_cmd_MW": "MW", "T_source": "degC", "T_sink": "degC",
        "P_elec_MW": "MW", "Q_heat_MW": "MW", "COP": "1", "Q_absorbed_source_MW": "MW",
    }

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
            defaults={"T_source": self.t_source_nom, "T_sink": self.t_sink_nom},
        )
        self.register_parameters("q_max", "eta_carnot")

    def step_model(self, dt: float, time: float) -> None:
        q_cmd = float(self._inputs_val.get("Q_heat_cmd_MW", 0.0))
        t_src = float(self._inputs_val.get("T_source", self.t_source_nom))
        t_snk = float(self._inputs_val.get("T_sink", self.t_sink_nom))

        self.q_actual_mw = min(self.q_max, max(0.0, q_cmd))

        # Carnot COP = T_sink / (T_sink - T_source) in Kelvin (valid also when idle)
        t_snk_k = t_snk + 273.15
        t_src_k = t_src + 273.15
        delta_t = max(5.0, t_snk_k - t_src_k)
        cop_carnot = t_snk_k / delta_t
        self.cop = max(1.2, min(7.0, self.eta_carnot * cop_carnot))

        if self.q_actual_mw > 0.01:
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

    Constant efficiency, ramp-limited electric input. Validity: 0 to p_max_mw.

    Parameters
    ----------
    name : str
        Unique component name identifier
    efficiency : float
        Conversion thermal efficiency

    Inputs
    ------
    P_cmd : MW
        Electrical consumption power request

    Outputs
    -------
    P_actual : MW
        Actual active power absorbed
    Q_thermal_MW : MW
        Useful steam or hot water thermal power delivered
    """

    units = {"P_cmd": "MW", "P_actual": "MW", "Q_thermal_MW": "MW"}

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

    COP degrades linearly with ambient temperature (1.5 % per K above 25 C) and
    is clamped to [2, 8]; it is evaluated continuously, also when idle.
    Validity: T_amb 5..45 C, constant-COP part-load behaviour.

    Parameters
    ----------
    name : str
        Unique component name identifier
    cop_nominal : float
        Design Coefficient of Performance
    """

    units = {
        "Q_cool_cmd_MW": "MW", "T_amb": "degC",
        "P_elec_MW": "MW", "Q_cool_actual_MW": "MW", "COP": "1",
    }

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
            defaults={"T_amb": 25.0},
        )

    def step_model(self, dt: float, time: float) -> None:
        q_cmd = float(self._inputs_val.get("Q_cool_cmd_MW", 0.0))
        t_amb = float(self._inputs_val.get("T_amb", 25.0))

        self.q_cooling_actual_mw = min(self.q_max, max(0.0, q_cmd))

        # Ambient derating: COP degrades by ~1.5% per C above 25 C
        cop_adj = self.cop_nom * (1.0 - 0.015 * (t_amb - 25.0))
        self.cop = max(2.0, min(8.0, cop_adj))
        if self.q_cooling_actual_mw > 0.01:
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

    Constant thermal COP; cooling is limited by capacity and by the available
    waste heat (heat consumed = cooling / COP). No electrical parasitics.

    Parameters
    ----------
    name : str
        Unique component name identifier

    Outputs
    -------
    Q_heat_consumed_MW : MW
        Waste heat consumed (Absorption Chiller)

    Notes
    -----
    Validity: constant thermal COP independent of source and sink temperatures; cooling between 0 and
    q_cooling_max_mw, limited by the available waste heat; steady state.
    """

    units = {
        "Q_cool_cmd_MW": "MW", "Q_waste_heat_available_MW": "MW",
        "Q_cool_actual_MW": "MW", "Q_heat_consumed_MW": "MW",
    }

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
            defaults={"Q_waste_heat_available_MW": 1e6},
        )

    def step_model(self, dt: float, time: float) -> None:
        q_cmd = max(0.0, float(self._inputs_val.get("Q_cool_cmd_MW", 0.0)))
        q_avail = max(0.0, float(self._inputs_val.get("Q_waste_heat_available_MW", 1e6)))

        q_req_heat = q_cmd / self.cop_th
        heat_used = min(q_avail, q_req_heat)
        self.q_cooling_actual = min(self.q_max, heat_used * self.cop_th)
        self.q_heat_consumed = self.q_cooling_actual / self.cop_th

        self._outputs_val["Q_cool_actual_MW"] = self.q_cooling_actual
        self._outputs_val["Q_heat_consumed_MW"] = self.q_heat_consumed


class ThermalStorageTank(ModelComponent):
    """
    Sensible hot/cold water Thermal Energy Storage (TES), energy-bucket model.

    The tank is a single energy store: SoC integrates the commanded thermal
    flow (positive = charging) minus a standing loss proportional to SoC. There
    is *no* stratification, thermocline, temperature or flow-temperature
    dependence. Requests are scaled so SoC stays in [0, 1], so stored energy
    is exactly the integral of ``Q_actual_MW - Q_loss_MW``.

    Validity: capacity-level studies at hours-to-days timescales.

    Parameters
    ----------
    name : str
        Unique component name identifier

    Outputs
    -------
    SoC : 1
        Thermal state of charge
    """

    units = {"Q_cmd_MW": "MW", "Q_actual_MW": "MW", "SoC": "1", "Q_loss_MW": "MW"}

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
        self.q_loss_mw = 0.0

        self.register_variables(
            inputs=["Q_cmd_MW"],
            outputs=["Q_actual_MW", "SoC", "Q_loss_MW"],
        )
        self.register_parameters("capacity_mwh", "max_flow")

    def step_model(self, dt: float, time: float) -> None:
        q_cmd = float(self._inputs_val.get("Q_cmd_MW", 0.0))
        q_flow = max(-self.max_flow, min(self.max_flow, q_cmd))

        e_per_mw = dt / (self.capacity_mwh * 3600.0)  # SoC change per MW over this step
        loss = self.loss_rate_s * self.soc * dt        # SoC lost to standing loss
        if e_per_mw > 0:
            # Scale the flow so that SoC remains within [0, 1].
            if q_flow > 0:
                q_flow = min(q_flow, max(0.0, (1.0 - self.soc + loss) / e_per_mw))
            elif q_flow < 0:
                q_flow = max(q_flow, min(0.0, (0.0 - self.soc + loss) / e_per_mw))

        soc_before = self.soc
        self.soc = max(0.0, min(1.0, self.soc + q_flow * e_per_mw - loss))
        self.q_actual_mw = q_flow
        self.q_loss_mw = self.loss_rate_s * soc_before * self.capacity_mwh * 3600.0

        self._outputs_val["Q_actual_MW"] = self.q_actual_mw
        self._outputs_val["SoC"] = self.soc
        self._outputs_val["Q_loss_MW"] = self.q_loss_mw

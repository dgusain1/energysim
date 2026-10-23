# -*- coding: utf-8 -*-
"""
Industrial Demand and Specialized Load Models for energysim.

Includes:
- DataCenterLoad: Hyperscale IT compute load with ambient-dependent PUE
  (Power Usage Effectiveness) and cooling power calculations.
- IndustrialProcessSteam: Industrial multi-pressure steam network balance.
"""

from __future__ import annotations
import math
from typing import Optional, Sequence

from .common.base_component import ModelComponent
from .common.curve import Curve1D


class DataCenterLoad(ModelComponent):
    """
    Hyperscale AI / Cloud Data Center Electrical & Thermal Load.

    Grid power = IT power x PUE(T_amb) with PUE from a lookup curve; 98 % of IT
    power is assumed to appear as recoverable waste heat. Quasi-static (no
    thermal inertia or workload dynamics). Validity: T_amb -10..40 C,
    0 <= IT <= it_capacity_mw.

    Parameters
    ----------
    it_capacity_mw : float
        Nominal IT compute server capacity in MW.
    base_pue : float, default 1.15
        Base Power Usage Effectiveness at mild ambient temperatures (<=15 C).
    pue_curve : Curve1D, optional
        PUE vs ambient dry-bulb temperature.

    Inputs
    ------
    T_amb : degC
        Ambient wet/dry-bulb temperature

    Outputs
    -------
    P_cooling_MW : MW
        Active power consumed solely by chillers and HVAC
    PUE : 1
        Operating Power Usage Effectiveness
    """

    units = {
        "P_it_cmd_MW": "MW", "T_amb": "degC",
        "P_total_grid_MW": "MW", "P_it_MW": "MW", "P_cooling_MW": "MW",
        "PUE": "1", "Q_waste_heat_MW": "MW",
    }

    def __init__(
        self,
        name: str = "DataCenter",
        it_capacity_mw: float = 100.0,
        base_pue: float = 1.15,
        pue_curve: Optional[Curve1D] = None,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.it_capacity_mw = it_capacity_mw
        self.base_pue = base_pue

        if pue_curve is None:
            t_pts = [-10.0, 10.0, 18.0, 25.0, 32.0, 40.0]
            pue_pts = [base_pue, base_pue, base_pue + 0.02, base_pue + 0.08, base_pue + 0.18, base_pue + 0.32]
            self.pue_curve = Curve1D(t_pts, pue_pts, min_val=1.05, max_val=1.8)
        else:
            self.pue_curve = pue_curve

        self.p_it_actual = 0.0
        self.pue = base_pue
        self.p_total_grid_mw = 0.0
        self.p_cooling_mw = 0.0
        self.q_waste_heat_mw = 0.0

        self.register_variables(
            inputs=["P_it_cmd_MW", "T_amb"],
            outputs=[
                "P_total_grid_MW", "P_it_MW", "P_cooling_MW",
                "PUE", "Q_waste_heat_MW",
            ],
            defaults={"P_it_cmd_MW": self.it_capacity_mw, "T_amb": 20.0},
        )

    def step_model(self, dt: float, time: float) -> None:
        it_cmd = float(self._inputs_val.get("P_it_cmd_MW", self.it_capacity_mw))
        t_amb = float(self._inputs_val.get("T_amb", 20.0))

        self.p_it_actual = min(self.it_capacity_mw, max(0.0, it_cmd))
        self.pue = float(self.pue_curve(t_amb))

        # Total grid power = IT * PUE
        self.p_total_grid_mw = self.p_it_actual * self.pue
        # Cooling + infrastructure power = Total - IT
        self.p_cooling_mw = max(0.0, self.p_total_grid_mw - self.p_it_actual)
        # Server electricity turns almost 100% into waste heat
        self.q_waste_heat_mw = self.p_it_actual * 0.98

        self._outputs_val["P_total_grid_MW"] = self.p_total_grid_mw
        self._outputs_val["P_it_MW"] = self.p_it_actual
        self._outputs_val["P_cooling_MW"] = self.p_cooling_mw
        self._outputs_val["PUE"] = self.pue
        self._outputs_val["Q_waste_heat_MW"] = self.q_waste_heat_mw


class IndustrialProcessSteam(ModelComponent):
    """
    Industrial process steam demand header for refineries and chemical plants.

    Quasi-static: demand = setpoint (>= 0), deficit = max(0, demand - supply),
    thermal demand = demand x enthalpy. No header pressure dynamics.

    Parameters
    ----------
    name : str
        Unique component name identifier
    nominal_demand_t_hr : float
        Nominal process steam demand

    Outputs
    -------
    steam_demand_t_hr : t/h
        Process steam demand
    steam_deficit_t_hr : t/h
        Unmet steam demand

    Notes
    -----
    Validity: quasi-static demand header; no pressure, temperature or condensate-return dynamics;
    constant specific enthalpy.
    """

    units = {
        "steam_supply_t_hr": "t/h", "demand_setpoint_t_hr": "t/h",
        "steam_demand_t_hr": "t/h", "steam_deficit_t_hr": "t/h", "Q_demand_MW_th": "MW",
    }

    def __init__(
        self,
        name: str = "SteamHeader",
        nominal_demand_t_hr: float = 100.0,
        enthalpy_kj_kg: float = 2800.0,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.nominal_demand = nominal_demand_t_hr
        self.h_steam = enthalpy_kj_kg

        self.demand_actual_t_hr = nominal_demand_t_hr
        self.deficit_t_hr = 0.0
        self.q_demand_mw_th = (nominal_demand_t_hr * 1000.0 * enthalpy_kj_kg) / 3.6e6

        self.register_variables(
            inputs=["steam_supply_t_hr", "demand_setpoint_t_hr"],
            outputs=["steam_demand_t_hr", "steam_deficit_t_hr", "Q_demand_MW_th"],
            defaults={"demand_setpoint_t_hr": self.nominal_demand},
        )

    def step_model(self, dt: float, time: float) -> None:
        supply = float(self._inputs_val.get("steam_supply_t_hr", 0.0))
        dem = float(self._inputs_val.get("demand_setpoint_t_hr", self.nominal_demand))

        self.demand_actual_t_hr = max(0.0, dem)
        self.deficit_t_hr = max(0.0, self.demand_actual_t_hr - supply)
        self.q_demand_mw_th = (self.demand_actual_t_hr * 1000.0 * self.h_steam) / 3.6e6

        self._outputs_val["steam_demand_t_hr"] = self.demand_actual_t_hr
        self._outputs_val["steam_deficit_t_hr"] = self.deficit_t_hr
        self._outputs_val["Q_demand_MW_th"] = self.q_demand_mw_th

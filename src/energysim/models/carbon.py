# -*- coding: utf-8 -*-
"""
Carbon Capture and Abatement Models for energysim.

Includes:
- CarbonCapture_Amine: Post-combustion amine solvent CO2 capture system.
  Calculates reboiler steam thermal demand and electrical auxiliary parasitics.
"""

from __future__ import annotations
import math
from typing import Optional, Sequence

from .common.base_component import ModelComponent


class CarbonCapture_Amine(ModelComponent):
    """
    Post-Combustion Chemical Absorption Amine CO2 Capture System.

    Parameters
    ----------
    capture_rate_nom : float, default 0.90
        Nominal CO2 capture efficiency (typically 90%).
    spec_steam_reboiler_gj_tco2 : float, default 3.0
        Thermal energy required in reboiler per ton of CO2 captured (GJ/tCO2).
    spec_elec_kwh_tco2 : float, default 120.0
        Electrical auxiliary consumption for blowers, solvent pumps, and compressors.
    """

    def __init__(
        self,
        name: str = "CarbonCapture",
        capture_rate_nom: float = 0.90,
        spec_steam_reboiler_gj_tco2: float = 3.0,
        spec_elec_kwh_tco2: float = 120.0,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.capture_rate = capture_rate_nom
        self.spec_steam_gj = spec_steam_reboiler_gj_tco2
        self.spec_elec_kwh = spec_elec_kwh_tco2

        self.m_dot_co2_captured_t_hr = 0.0
        self.p_elec_parasitic_mw = 0.0
        self.q_steam_reboiler_mw_th = 0.0

        self.register_variables(
            inputs=["m_dot_CO2_in_t_hr"],
            outputs=[
                "m_dot_CO2_captured_t_hr",
                "P_elec_parasitic_MW",
                "Q_steam_reboiler_MW_th",
            ],
        )

    def step_model(self, dt: float, time: float) -> None:
        co2_in = float(self._inputs_val.get("m_dot_CO2_in_t_hr", 0.0))

        if co2_in > 0.001:
            self.m_dot_co2_captured_t_hr = co2_in * self.capture_rate
            # Electrical parasitic power: (t/hr * kWh/t) / 1000 = MW
            self.p_elec_parasitic_mw = (self.m_dot_co2_captured_t_hr * self.spec_elec_kwh) / 1000.0
            # Reboiler thermal power: (t/hr * GJ/t) * (1000 MJ / 3600 s) = MW_th
            self.q_steam_reboiler_mw_th = (self.m_dot_co2_captured_t_hr * self.spec_steam_gj * 1000.0) / 3600.0
        else:
            self.m_dot_co2_captured_t_hr = 0.0
            self.p_elec_parasitic_mw = 0.0
            self.q_steam_reboiler_mw_th = 0.0

        self._outputs_val["m_dot_CO2_captured_t_hr"] = self.m_dot_co2_captured_t_hr
        self._outputs_val["P_elec_parasitic_MW"] = self.p_elec_parasitic_mw
        self._outputs_val["Q_steam_reboiler_MW_th"] = self.q_steam_reboiler_mw_th

# -*- coding: utf-8 -*-
"""
Renewable Generation Models for energysim.

Includes:
- PhysicsPV: Solar photovoltaic generator with empirical cell temperature,
  temperature derating, Sandia inverter efficiency, and curtailment.
- WindTurbine: Wind turbine with multi-point power curve, air density derating,
  and rotor inertia lag.
- SolarThermalCollector: Solar thermal collector with Hottel-Whillier-Bliss efficiency.
"""

from __future__ import annotations
import math
from typing import Optional, Sequence
import numpy as np

from .common.base_component import ModelComponent
from .common.curve import Curve1D


class PhysicsPV(ModelComponent):
    """
    Solar Photovoltaic system with irradiance, temperature derating, and inverter clipping.

    Parameters
    ----------
    p_dc_stc_mw : float
        DC array rated peak power at Standard Test Conditions (STC: 1000 W/m2, 25 C) in MW.
    inverter_p_ac_max_mw : float
        AC inverter maximum rated capacity in MW.
    temp_coeff_p : float, default -0.0038
        Power temperature coefficient in 1/K (default -0.38 %/C for crystalline silicon).
    noct_c : float, default 45.0
        Nominal Operating Cell Temperature in C (at 800 W/m2, 20 C ambient, 1 m/s wind).
    inverter_curve : Curve1D, optional
        Inverter efficiency vs normalized DC power.
    """

    def __init__(
        self,
        name: str = "SolarPV",
        p_dc_stc_mw: float = 1.0,
        inverter_p_ac_max_mw: float = 0.9,
        temp_coeff_p: float = -0.0038,
        noct_c: float = 45.0,
        inverter_curve: Optional[Curve1D] = None,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.p_dc_stc_mw = p_dc_stc_mw
        self.inverter_p_ac_max_mw = inverter_p_ac_max_mw
        self.temp_coeff_p = temp_coeff_p
        self.noct_c = noct_c

        if inverter_curve is None:
            load_pts = [0.0, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0, 1.2]
            eff_pts = [0.80, 0.89, 0.93, 0.96, 0.98, 0.982, 0.98, 0.975]
            self.inverter_curve = Curve1D(load_pts, eff_pts, min_val=0.5, max_val=0.99)
        else:
            self.inverter_curve = inverter_curve

        self.g_poa = 0.0          # W/m2 Plane-of-Array irradiance
        self.t_amb = 25.0         # deg C
        self.v_wind = 1.0         # m/s
        self.curtail_limit = 1e6  # MW
        self.curtail_limit_mw = 1e6
        self.p_dc = 0.0
        self.p_ac = 0.0
        self.t_cell_c = 25.0

        self.register_variables(
            inputs=["G_poa", "T_amb", "v_wind", "curtail_limit"],
            outputs=["P_ac", "P_dc", "T_cell"],
        )

    def step_model(self, dt: float, time: float) -> None:
        g_poa = float(self._inputs_val.get("G_poa", self.g_poa))
        t_amb = float(self._inputs_val.get("T_amb", self.t_amb))
        v_wind = float(self._inputs_val.get("v_wind", self.v_wind))
        curtail = float(self._inputs_val.get("curtail_limit", self.curtail_limit_mw))

        if g_poa <= 1.0:
            self.p_dc = 0.0
            self.p_ac = 0.0
            self.t_cell_c = t_amb
        else:
            # NOCT cell temperature with wind speed damping
            wind_factor = 1.0 / (1.0 + 0.05 * max(0.0, v_wind - 1.0))
            self.t_cell_c = t_amb + ((self.noct_c - 20.0) / 800.0) * g_poa * wind_factor

            # DC power calculation with temperature derating
            temp_derate = 1.0 + self.temp_coeff_p * (self.t_cell_c - 25.0)
            temp_derate = max(0.5, temp_derate)
            p_dc_unclipped = self.p_dc_stc_mw * (g_poa / 1000.0) * temp_derate

            # Inverter efficiency
            load_ratio = p_dc_unclipped / max(self.inverter_p_ac_max_mw, 1e-4)
            inv_eff = float(self.inverter_curve(load_ratio))

            p_ac_raw = p_dc_unclipped * inv_eff

            # Clipping to inverter max AC capacity and curtailment setpoint
            p_ac_clipped = min(p_ac_raw, self.inverter_p_ac_max_mw, curtail)
            self.p_ac = max(0.0, p_ac_clipped)
            self.p_dc = self.p_ac / max(inv_eff, 0.5)

        self._outputs_val["P_ac"] = self.p_ac
        self._outputs_val["P_dc"] = self.p_dc
        self._outputs_val["T_cell"] = self.t_cell_c


class WindTurbine(ModelComponent):
    """
    Wind Turbine with multi-point power curve, air density derating, and rotor inertia lag.

    Parameters
    ----------
    p_rated_mw : float
        Rated electrical output power in MW.
    cut_in_speed_ms : float, default 3.0
        Cut-in wind speed in m/s.
    rated_speed_ms : float, default 11.5
        Rated wind speed in m/s.
    cut_out_speed_ms : float, default 25.0
        Cut-out wind speed in m/s.
    rotor_time_constant_sec : float, default 4.0
        1st-order rotor inertia acceleration/pitch lag time constant.
    """

    def __init__(
        self,
        name: str = "WindTurbine",
        p_rated_mw: float = 3.0,
        cut_in_speed_ms: float = 3.0,
        rated_speed_ms: float = 11.5,
        cut_out_speed_ms: float = 25.0,
        rotor_time_constant_sec: float = 4.0,
        power_curve: Optional[Curve1D] = None,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.p_rated_mw = p_rated_mw
        self.cut_in = cut_in_speed_ms
        self.rated_speed = rated_speed_ms
        self.cut_out = cut_out_speed_ms
        self.tau = rotor_time_constant_sec

        if power_curve is None:
            # Standard cubic-transition power curve
            v_pts = [0.0, self.cut_in, 5.0, 7.0, 9.0, self.rated_speed, self.cut_out - 0.1, self.cut_out, 35.0]
            p_pts = [
                0.0,
                0.0,
                p_rated_mw * (5.0 / rated_speed_ms) ** 3,
                p_rated_mw * (7.0 / rated_speed_ms) ** 3,
                p_rated_mw * (9.0 / rated_speed_ms) ** 3,
                p_rated_mw,
                p_rated_mw,
                0.0,
                0.0,
            ]
            self.power_curve = Curve1D(v_pts, p_pts, min_val=0.0, max_val=p_rated_mw)
        else:
            self.power_curve = power_curve

        self.v_wind = 0.0
        self.t_amb = 15.0
        self.p_actual = 0.0

        self.register_variables(
            inputs=["v_wind", "T_amb"],
            outputs=["P_actual"],
        )

    def step_model(self, dt: float, time: float) -> None:
        v_wind = float(self._inputs_val.get("v_wind", self.v_wind))
        t_amb = float(self._inputs_val.get("T_amb", self.t_amb))

        # Air density correction: rho = p / (R * T)
        t_kelvin = t_amb + 273.15
        rho = 101325.0 / (287.05 * max(t_kelvin, 100.0))
        rho_0 = 1.225  # Standard sea-level density
        density_mult = rho / rho_0

        p_raw = float(self.power_curve(v_wind)) * density_mult
        p_target = min(self.p_rated_mw, max(0.0, p_raw))

        # Rotor inertia 1st-order filter
        alpha = dt / (self.tau + dt)
        self.p_actual += alpha * (p_target - self.p_actual)

        self._outputs_val["P_actual"] = self.p_actual


class SolarThermalCollector(ModelComponent):
    """
    Solar thermal collector with quadratic Hottel-Whillier-Bliss efficiency.
    """

    def __init__(
        self,
        name: str = "SolarThermal",
        area_m2: float = 1000.0,
        eta0: float = 0.80,
        a1_w_m2_k: float = 3.5,
        a2_w_m2_k2: float = 0.015,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.area_m2 = area_m2
        self.eta0 = eta0
        self.a1 = a1_w_m2_k
        self.a2 = a2_w_m2_k2

        self.g_solar = 0.0
        self.t_in_c = 40.0
        self.t_amb = 20.0
        self.q_thermal_mw = 0.0

        self.register_variables(
            inputs=["G_solar", "T_in", "T_amb"],
            outputs=["Q_thermal_MW"],
        )

    def step_model(self, dt: float, time: float) -> None:
        g = float(self._inputs_val.get("G_solar", self.g_solar))
        t_in = float(self._inputs_val.get("T_in", self.t_in_c))
        t_amb = float(self._inputs_val.get("T_amb", self.t_amb))

        if g > 10.0:
            delta_t = max(0.0, t_in - t_amb)
            eff = self.eta0 - self.a1 * (delta_t / g) - self.a2 * (delta_t**2 / g)
            eff = max(0.0, min(eff, self.eta0))
            q_w = self.area_m2 * g * eff
            self.q_thermal_mw = q_w / 1e6
        else:
            self.q_thermal_mw = 0.0

        self._outputs_val["Q_thermal_MW"] = self.q_thermal_mw

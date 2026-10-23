# -*- coding: utf-8 -*-
"""
Energy Storage Models for energysim.

Includes:
- BatteryBESS: Li-ion battery with state-dependent OCV, internal resistance,
  lumped 1-state thermal ODE, and calendar/cyclic SOH degradation.
- FlowBattery: Vanadium redox flow battery (VRFB) with decoupled power/energy
  and pump parasitics.
- Flywheel: Kinetic energy storage with speed curve and self-discharge.
"""

from __future__ import annotations
import math
from typing import Optional, Sequence
import numpy as np

from .common.base_component import ModelComponent
from .common.curve import Curve1D


class BatteryBESS(ModelComponent):
    """
    State-dependent Lithium-ion Battery Energy Storage System (BESS).

    Models:
    - OCV curve: Voc(SoC)
    - Internal resistance: Rin(SoC)
    - Inverter efficiency: eta(P/P_rated)
    - 1-state thermal balance: C_th * dT/dt = I^2 * R - (T - T_amb) / R_th
    - Calendar & cyclic degradation (SOH fade)

    Parameters
    ----------
    capacity_mwh : float
        Nominal energy capacity in MWh.
    p_max_mw : float
        Maximum charge/discharge power in MW.
    soc_init : float, default 0.5
        Initial state of charge [0.0, 1.0].
    soc_min : float, default 0.05
        Minimum allowable state of charge.
    soc_max : float, default 0.95
        Maximum allowable state of charge.
    nominal_voltage_v : float, default 800.0
        Pack nominal DC voltage in Volts.
    c_th_kj_k : float, default 50000.0
        Lumped thermal capacitance of pack in kJ/K.
    r_th_k_w : float, default 0.05
        Thermal resistance to ambient in K/W.
    t_init_c : float, default 25.0
        Initial battery temperature in degrees C.
    """

    def __init__(
        self,
        name: str = "BESS",
        capacity_mwh: float = 1.0,
        p_max_mw: float = 0.5,
        soc_init: float = 0.5,
        soc_min: float = 0.05,
        soc_max: float = 0.95,
        nominal_voltage_v: float = 800.0,
        c_th_kj_k: float = 50000.0,
        r_th_k_w: float = 0.05,
        t_init_c: float = 25.0,
        ocv_curve: Optional[Curve1D] = None,
        rin_curve: Optional[Curve1D] = None,
        inverter_curve: Optional[Curve1D] = None,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.capacity_mwh = capacity_mwh
        self.p_max_mw = p_max_mw
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.nominal_voltage_v = nominal_voltage_v
        self.c_th_kj_k = c_th_kj_k
        self.r_th_k_w = r_th_k_w

        # Default OCV curve (NMC / LFP hybrid characteristic)
        if ocv_curve is None:
            soc_pts = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
            voc_mult = [0.85, 0.92, 0.95, 0.98, 0.99, 1.00, 1.02, 1.05, 1.07, 1.10]
            self.ocv_curve = Curve1D(soc_pts, [nominal_voltage_v * m for m in voc_mult])
        else:
            self.ocv_curve = ocv_curve

        # Default internal resistance curve (higher at low SoC)
        if rin_curve is None:
            r_base = (nominal_voltage_v * 0.03) / (p_max_mw * 1e6 / nominal_voltage_v)  # ~3% voltage drop at full load
            soc_pts = [0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0]
            r_mult = [3.0, 1.8, 1.2, 1.0, 1.0, 1.2, 1.5]
            self.rin_curve = Curve1D(soc_pts, [r_base * m for m in r_mult], min_val=1e-5)
        else:
            self.rin_curve = rin_curve

        # Default inverter efficiency curve
        if inverter_curve is None:
            load_pts = [0.0, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0]
            eff_pts = [0.80, 0.88, 0.93, 0.96, 0.975, 0.97, 0.96]
            self.inverter_curve = Curve1D(load_pts, eff_pts, min_val=0.5, max_val=0.99)
        else:
            self.inverter_curve = inverter_curve

        # Continuous states
        self.soc = float(soc_init)
        self.t_batt_c = float(t_init_c)
        self.soh = 1.0

        # Inputs and outputs
        self.p_cmd = 0.0         # MW (positive = charge, negative = discharge)
        self.t_amb = 25.0        # deg C
        self.p_actual = 0.0      # MW (AC grid interface)
        self.p_dc = 0.0          # MW (DC cell level)
        self.v_terminal = nominal_voltage_v
        self.i_current = 0.0     # A
        self.q_heat_kw = 0.0     # kW thermal loss

        self.register_variables(
            inputs=["P_cmd", "T_amb"],
            outputs=["P_actual", "SoC", "T_batt", "SOH", "V_terminal", "I_current", "Q_heat_kW"],
        )

    def step_model(self, dt: float, time: float) -> None:
        p_cmd = float(self._inputs_val.get("P_cmd", self.p_cmd))
        t_amb = float(self._inputs_val.get("T_amb", self.t_amb))

        # Clamp command to rated power
        p_req = max(-self.p_max_mw, min(self.p_max_mw, p_cmd))

        # Inverter efficiency
        load_ratio = abs(p_req) / max(self.p_max_mw, 1e-6)
        inv_eff = float(self.inverter_curve(load_ratio))

        # Check SoC limits
        if p_req > 0 and self.soc >= self.soc_max:
            p_req = 0.0  # Cannot charge further
        elif p_req < 0 and self.soc <= self.soc_min:
            p_req = 0.0  # Cannot discharge further

        if p_req > 0:  # Charging: grid draws P_req, DC receives P_req * inv_eff
            p_dc = p_req * inv_eff
            self.p_actual = p_req
        elif p_req < 0:  # Discharging: DC delivers P_req / inv_eff, grid receives P_req
            p_dc = p_req / inv_eff
            self.p_actual = p_req
        else:
            p_dc = 0.0
            self.p_actual = 0.0

        # DC Electrical state: Voc and Rin
        voc = float(self.ocv_curve(self.soc))
        rin = float(self.rin_curve(self.soc))

        # P_dc = V_t * I = (Voc + I * Rin) * I -> Rin * I^2 + Voc * I - P_dc = 0
        p_dc_w = p_dc * 1e6
        disc = voc**2 + 4.0 * rin * p_dc_w
        if disc >= 0:
            i_cell = (-voc + math.sqrt(disc)) / (2.0 * rin) if rin > 0 else p_dc_w / voc
        else:
            i_cell = p_dc_w / voc

        v_t = voc + i_cell * rin
        self.v_terminal = v_t
        self.i_current = i_cell
        self.p_dc = p_dc

        # Energy update (SoC)
        effective_cap_mwh = self.capacity_mwh * self.soh
        d_soc = (p_dc * dt) / (effective_cap_mwh * 3600.0)
        self.soc = max(self.soc_min, min(self.soc_max, self.soc + d_soc))

        # Thermal ODE: C_th * dT/dt = Q_joule - (T - T_amb)/R_th
        q_joule_w = (i_cell ** 2) * rin
        self.q_heat_kw = q_joule_w / 1000.0
        q_cool_w = (self.t_batt_c - t_amb) / max(self.r_th_k_w, 1e-4)
        c_th_j_k = self.c_th_kj_k * 1000.0
        d_temp = ((q_joule_w - q_cool_w) / c_th_j_k) * dt
        self.t_batt_c += d_temp

        # SOH Degradation (Calendar + Cyclic)
        t_kelvin = self.t_batt_c + 273.15
        arrhenius = math.exp(-25000.0 / (8.314 * t_kelvin))
        d_soh_cal = 1e-9 * arrhenius * dt
        throughput_mwh = abs(p_dc) * (dt / 3600.0)
        d_soh_cyc = (throughput_mwh / (2.0 * effective_cap_mwh * 4000.0))  # ~4000 full cycles to EOL
        self.soh = max(0.5, self.soh - (d_soh_cal + d_soh_cyc))

        # Update output dict
        self._outputs_val["P_actual"] = self.p_actual
        self._outputs_val["SoC"] = self.soc
        self._outputs_val["T_batt"] = self.t_batt_c
        self._outputs_val["SOH"] = self.soh
        self._outputs_val["V_terminal"] = self.v_terminal
        self._outputs_val["I_current"] = self.i_current
        self._outputs_val["Q_heat_kW"] = self.q_heat_kw


class FlowBattery(ModelComponent):
    """
    Vanadium Redox Flow Battery (VRFB).
    Decoupled power (stack MW) and energy (tank MWh) with state-dependent pump losses.
    """

    def __init__(
        self,
        name: str = "VRFB",
        stack_p_max_mw: float = 1.0,
        tank_capacity_mwh: float = 6.0,
        soc_init: float = 0.5,
        stack_efficiency: float = 0.85,
        pump_base_loss_kw: float = 10.0,
        pump_var_loss_kw: float = 30.0,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.stack_p_max_mw = stack_p_max_mw
        self.tank_capacity_mwh = tank_capacity_mwh
        self.stack_eff = stack_efficiency
        self.pump_base_loss_kw = pump_base_loss_kw
        self.pump_var_loss_kw = pump_var_loss_kw

        self.soc = float(soc_init)
        self.p_actual = 0.0
        self.p_parasitic_kw = 0.0

        self.register_variables(
            inputs=["P_cmd"],
            outputs=["P_actual", "SoC", "P_parasitic_kW"],
        )

    def step_model(self, dt: float, time: float) -> None:
        p_cmd = float(self._inputs_val.get("P_cmd", 0.0))
        p_req = max(-self.stack_p_max_mw, min(self.stack_p_max_mw, p_cmd))

        # Flow pump parasitics
        load_ratio = abs(p_req) / max(self.stack_p_max_mw, 1e-6)
        if abs(p_req) > 1e-4:
            p_pump_kw = self.pump_base_loss_kw + self.pump_var_loss_kw * (load_ratio ** 2)
        else:
            p_pump_kw = 0.0
        self.p_parasitic_kw = p_pump_kw
        p_pump_mw = p_pump_kw / 1000.0

        if p_req > 0 and self.soc >= 0.98:
            p_req = 0.0
        elif p_req < 0 and self.soc <= 0.02:
            p_req = 0.0

        # Energy update with coulombic/voltage round-trip efficiency
        sqrt_eff = math.sqrt(self.stack_eff)
        if p_req > 0:
            d_soc = (p_req * sqrt_eff * dt) / (self.tank_capacity_mwh * 3600.0)
            self.p_actual = p_req + p_pump_mw
        elif p_req < 0:
            d_soc = (p_req / sqrt_eff * dt) / (self.tank_capacity_mwh * 3600.0)
            self.p_actual = p_req + p_pump_mw
        else:
            d_soc = 0.0
            self.p_actual = p_pump_mw

        self.soc = max(0.0, min(1.0, self.soc + d_soc))
        self._outputs_val["P_actual"] = self.p_actual
        self._outputs_val["SoC"] = self.soc
        self._outputs_val["P_parasitic_kW"] = self.p_parasitic_kw


class Flywheel(ModelComponent):
    """
    Flywheel kinetic energy storage system for high-frequency regulation.
    """

    def __init__(
        self,
        name: str = "Flywheel",
        power_max_mw: float = 0.5,
        energy_capacity_kwh: float = 125.0,
        soc_init: float = 0.5,
        roundtrip_eff: float = 0.90,
        self_discharge_pct_hr: float = 1.0,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.p_max = power_max_mw
        self.capacity_kwh = energy_capacity_kwh
        self.eff = roundtrip_eff
        self.self_discharge_rate = (self_discharge_pct_hr / 100.0) / 3600.0

        self.soc = float(soc_init)
        self.p_actual = 0.0

        self.register_variables(
            inputs=["P_cmd"],
            outputs=["P_actual", "SoC"],
        )

    def step_model(self, dt: float, time: float) -> None:
        p_cmd = float(self._inputs_val.get("P_cmd", 0.0))
        p_req = max(-self.p_max, min(self.p_max, p_cmd))

        if p_req > 0 and self.soc >= 0.99:
            p_req = 0.0
        elif p_req < 0 and self.soc <= 0.01:
            p_req = 0.0

        sqrt_eff = math.sqrt(self.eff)
        # Power in MW -> kWh/s = P_MW * 1000 / 3600
        p_kwh_s = (p_req * 1000.0) / 3600.0
        if p_req > 0:
            d_soc = (p_kwh_s * sqrt_eff * dt) / self.capacity_kwh
        else:
            d_soc = (p_kwh_s / sqrt_eff * dt) / self.capacity_kwh

        # Self-discharge
        d_soc -= self.self_discharge_rate * dt

        self.soc = max(0.0, min(1.0, self.soc + d_soc))
        self.p_actual = p_req
        self._outputs_val["P_actual"] = self.p_actual
        self._outputs_val["SoC"] = self.soc

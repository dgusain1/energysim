# -*- coding: utf-8 -*-
"""
Energy Storage Models for energysim.

Includes:
- BatteryBESS: Li-ion battery, Rint equivalent circuit (Voc(SoC) in series with
  Rin(SoC)), lumped 1-state thermal ODE, and calendar/cyclic SOH degradation.
- FlowBattery: Vanadium redox flow battery (VRFB) with decoupled power/energy
  and pump parasitics.
- Flywheel: Kinetic energy storage with round-trip efficiency and self-discharge.
"""

from __future__ import annotations
import math
from typing import Optional, Sequence
import numpy as np

from .common.base_component import ModelComponent
from .common.curve import Curve1D

_R_GAS = 8.314
_SECONDS_PER_YEAR = 365.25 * 86400.0


def _fit_to_bounds(p_req, soc, k_chg, k_dis, loss, lo, hi):
    """Scale a power request so that SoC stays inside [lo, hi].

    ``d_soc = k_chg * p - loss`` for p > 0 and ``k_dis * p - loss`` for p < 0
    (k_* > 0 are the SoC change per MW over the step, loss >= 0 is the self-loss).
    """
    if p_req > 0:
        p_hi = (hi - soc + loss) / k_chg if k_chg > 0 else p_req
        return max(0.0, min(p_req, p_hi))
    if p_req < 0:
        p_lo = (lo - soc + loss) / k_dis if k_dis > 0 else p_req
        return min(0.0, max(p_req, p_lo))
    return 0.0


class BatteryBESS(ModelComponent):
    """
    Lithium-ion Battery Energy Storage System (BESS), Rint equivalent circuit.

    The pack is a Rint model: an open-circuit voltage source Voc(SoC) in series
    with an internal resistance Rin(SoC) (no RC / Thevenin polarisation
    branches). Models:

    - OCV curve Voc(SoC) and internal resistance Rin(SoC)
    - inverter efficiency eta(P/P_rated)
    - SoC integrates the *stored chemical power* Voc*I, so joule heat I^2*Rin is
      accounted for: P_dc = Voc*I + I^2*Rin (energy in = stored + heat)
    - 1-state thermal balance: C_th * dT/dt = I^2 * Rin - (T - T_amb) / R_th
    - calendar fade (Arrhenius-scaled) and cyclic fade (full equivalent cycles)

    Parameters
    ----------
    capacity_mwh : float
        Nominal energy capacity in MWh.
    p_max_mw : float
        Maximum charge/discharge power in MW (AC side).
    soc_init : float, default 0.5
        Initial state of charge [0.0, 1.0].
    soc_min, soc_max : float
        Operating SoC window; requests are scaled so SoC never leaves it.
    nominal_voltage_v : float, default 800.0
        Pack nominal DC voltage in Volts.
    c_th_kj_k : float, default 50000.0
        Lumped thermal capacitance of pack in kJ/K.
    r_th_k_w : float, default 0.002
        Thermal resistance to ambient in K/W.
    t_init_c : float, default 25.0
        Initial battery temperature in degrees C.
    calendar_fade_pct_per_year_25c : float, default 2.0
        Calendar capacity fade per year at 25 C (Arrhenius scaled, Ea = 25 kJ/mol).
    cycle_life_to_eol : float, default 4000.0
        Full equivalent cycles (2 x nominal capacity of DC throughput) to reach ``eol_fade``.
    eol_fade : float, default 0.20
        Cyclic capacity fade at ``cycle_life_to_eol`` (0.20 -> 80 % SOH).

    Notes
    -----
    Validity: meaningful for roughly 0-45 C pack temperature, SoC between ~5 and
    95 %, and |P| <= p_max_mw. SOH is floored at 0.5. ``Q_heat_kW`` is DC joule
    heat only (inverter losses are not included).
    """

    units = {
        "P_cmd": "MW", "T_amb": "degC",
        "P_actual": "MW", "SoC": "1", "T_batt": "degC", "SOH": "1",
        "V_terminal": "V", "I_current": "A", "Q_heat_kW": "kW",
    }

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
        r_th_k_w: float = 0.002,
        t_init_c: float = 25.0,
        ocv_curve: Optional[Curve1D] = None,
        rin_curve: Optional[Curve1D] = None,
        inverter_curve: Optional[Curve1D] = None,
        step_size: float = 1.0,
        calendar_fade_pct_per_year_25c: float = 2.0,
        cycle_life_to_eol: float = 4000.0,
        eol_fade: float = 0.20,
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
        self.calendar_fade_pct_per_year_25c = calendar_fade_pct_per_year_25c
        self.cycle_life_to_eol = cycle_life_to_eol
        self.eol_fade = eol_fade

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
        self.p_dc = 0.0          # MW (DC terminal)
        self.v_terminal = float(self.ocv_curve(self.soc))
        self.i_current = 0.0     # A
        self.q_heat_kw = 0.0     # kW thermal loss

        self.register_variables(
            inputs=["P_cmd", "T_amb"],
            outputs=["P_actual", "SoC", "T_batt", "SOH", "V_terminal", "I_current", "Q_heat_kW"],
        )
        self.register_parameters("capacity_mwh", "p_max_mw", "soc_min", "soc_max", "c_th_kj_k", "r_th_k_w")

    def _solve_dc(self, p_req: float, voc: float, rin: float):
        """Return (p_dc_mw, current_a, inverter_eff) for an AC request in MW."""
        load_ratio = abs(p_req) / max(self.p_max_mw, 1e-6)
        inv_eff = float(self.inverter_curve(load_ratio))
        if p_req > 0:    # charging: DC receives P_req * eff
            p_dc_w = p_req * inv_eff * 1e6
        elif p_req < 0:  # discharging: DC delivers P_req / eff
            p_dc_w = p_req / inv_eff * 1e6
        else:
            return 0.0, 0.0, inv_eff
        # P_dc = V_t * I = (Voc + I * Rin) * I  ->  Rin * I^2 + Voc * I - P_dc = 0
        disc = voc ** 2 + 4.0 * rin * p_dc_w
        if disc >= 0.0:
            i_cell = (-voc + math.sqrt(disc)) / (2.0 * rin)
        else:
            # Beyond the maximum power transfer point: limit current there.
            i_cell = -voc / (2.0 * rin)
        p_dc_w = (voc + i_cell * rin) * i_cell
        return p_dc_w / 1e6, i_cell, inv_eff

    def step_model(self, dt: float, time: float) -> None:
        p_cmd = float(self._inputs_val.get("P_cmd", self.p_cmd))
        t_amb = float(self._inputs_val.get("T_amb", self.t_amb))

        p_req = max(-self.p_max_mw, min(self.p_max_mw, p_cmd))
        if p_req > 0 and self.soc >= self.soc_max:
            p_req = 0.0  # Cannot charge further
        elif p_req < 0 and self.soc <= self.soc_min:
            p_req = 0.0  # Cannot discharge further

        voc = float(self.ocv_curve(self.soc))
        rin = float(self.rin_curve(self.soc))
        cap_eff_mwh = self.capacity_mwh * self.soh
        soc_per_mw_s = 1.0 / (cap_eff_mwh * 3600.0)

        p_dc, i_cell, inv_eff = self._solve_dc(p_req, voc, rin)
        # Stored chemical power Voc*I (MW); SoC integrates this, not terminal power.
        d_soc = (voc * i_cell / 1e6) * dt * soc_per_mw_s

        # Scale the request so SoC stays inside the operating window.
        new_soc = self.soc + d_soc
        if dt > 0 and d_soc != 0.0 and (new_soc > self.soc_max or new_soc < self.soc_min):
            # Bisect on the scale factor (the DC solve is slightly nonlinear in P)
            # so SoC lands exactly on the window bound.
            bound = self.soc_max if new_soc > self.soc_max else self.soc_min
            p_full = p_req
            lo_f, hi_f = 0.0, 1.0
            for _ in range(50):
                mid = 0.5 * (lo_f + hi_f)
                _, i_mid, _ = self._solve_dc(p_full * mid, voc, rin)
                d_mid = (voc * i_mid / 1e6) * dt * soc_per_mw_s
                if abs(self.soc + d_mid - bound) < 1e-15:
                    lo_f = mid
                    break
                if (self.soc + d_mid > self.soc_max) or (self.soc + d_mid < self.soc_min):
                    hi_f = mid
                else:
                    lo_f = mid
            p_req = p_full * lo_f
            p_dc, i_cell, inv_eff = self._solve_dc(p_req, voc, rin)
            d_soc = (voc * i_cell / 1e6) * dt * soc_per_mw_s

        if p_req > 0:
            self.p_actual = p_req
        elif p_req < 0:
            # Equals the request unless limited at the max power transfer point.
            self.p_actual = max(p_req, p_dc * inv_eff)
        else:
            self.p_actual = 0.0

        self.v_terminal = voc + i_cell * rin
        self.i_current = i_cell
        self.p_dc = p_dc
        lo = min(self.soc_min, self.soc)
        hi = max(self.soc_max, self.soc)
        self.soc = max(lo, min(hi, self.soc + d_soc))

        # Thermal ODE: C_th * dT/dt = Q_joule - (T - T_amb)/R_th
        q_joule_w = (i_cell ** 2) * rin
        self.q_heat_kw = q_joule_w / 1000.0
        q_cool_w = (self.t_batt_c - t_amb) / max(self.r_th_k_w, 1e-6)
        c_th_j_k = self.c_th_kj_k * 1000.0
        self.t_batt_c += ((q_joule_w - q_cool_w) / c_th_j_k) * dt

        # SOH degradation: Arrhenius-scaled calendar fade + cyclic fade
        t_kelvin = self.t_batt_c + 273.15
        arrhenius = math.exp(-25000.0 / _R_GAS * (1.0 / t_kelvin - 1.0 / 298.15))
        d_soh_cal = (self.calendar_fade_pct_per_year_25c / 100.0) / _SECONDS_PER_YEAR * arrhenius * dt
        throughput_mwh = abs(p_dc) * (dt / 3600.0)
        fec = throughput_mwh / (2.0 * self.capacity_mwh)  # full equivalent cycles
        d_soh_cyc = self.eol_fade * fec / max(self.cycle_life_to_eol, 1e-9)
        self.soh = max(0.5, self.soh - (d_soh_cal + d_soh_cyc))

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

    Decoupled power (stack MW) and energy (tank MWh) with load-dependent pump
    parasitics. Round-trip stack efficiency is split symmetrically
    (sqrt(eta) on charge and discharge). SoC is limited to [0.02, 0.98] by
    scaling the request, so SoC integrates exactly what the stack exchanged.

    Notes
    -----
    Validity: an energy-bucket model; no electrolyte imbalance, crossover or
    temperature dependence. ``P_actual`` is the grid-side power including
    pump parasitics (positive = import).
    """

    units = {
        "P_cmd": "MW",
        "P_actual": "MW", "SoC": "1", "P_parasitic_kW": "kW",
    }

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
        self.soc_min = 0.02
        self.soc_max = 0.98

        self.soc = float(soc_init)
        self.p_actual = 0.0
        self.p_parasitic_kw = 0.0

        self.register_variables(
            inputs=["P_cmd"],
            outputs=["P_actual", "SoC", "P_parasitic_kW"],
        )
        self.register_parameters("stack_p_max_mw", "tank_capacity_mwh", "pump_base_loss_kw", "pump_var_loss_kw")

    def step_model(self, dt: float, time: float) -> None:
        p_cmd = float(self._inputs_val.get("P_cmd", 0.0))
        p_req = max(-self.stack_p_max_mw, min(self.stack_p_max_mw, p_cmd))

        sqrt_eff = math.sqrt(self.stack_eff)
        soc_per_mw_s = 1.0 / (self.tank_capacity_mwh * 3600.0)
        p_req = _fit_to_bounds(
            p_req, self.soc,
            k_chg=sqrt_eff * dt * soc_per_mw_s, k_dis=dt * soc_per_mw_s / sqrt_eff,
            loss=0.0, lo=self.soc_min, hi=self.soc_max,
        )

        # Flow pump parasitics (only while the stack is exchanging power)
        if abs(p_req) > 1e-4:
            load_ratio = abs(p_req) / max(self.stack_p_max_mw, 1e-6)
            p_pump_kw = self.pump_base_loss_kw + self.pump_var_loss_kw * (load_ratio ** 2)
        else:
            p_pump_kw = 0.0
        self.p_parasitic_kw = p_pump_kw
        p_pump_mw = p_pump_kw / 1000.0

        if p_req > 0:
            d_soc = p_req * sqrt_eff * dt * soc_per_mw_s
        elif p_req < 0:
            d_soc = p_req / sqrt_eff * dt * soc_per_mw_s
        else:
            d_soc = 0.0
        self.p_actual = p_req + p_pump_mw

        self.soc = max(min(self.soc_min, self.soc), min(max(self.soc_max, self.soc), self.soc + d_soc))
        self._outputs_val["P_actual"] = self.p_actual
        self._outputs_val["SoC"] = self.soc
        self._outputs_val["P_parasitic_kW"] = self.p_parasitic_kw


class Flywheel(ModelComponent):
    """
    Flywheel kinetic energy storage system for high-frequency regulation.

    Round-trip efficiency is split symmetrically (sqrt(eta) each way) and the
    rotor self-discharges proportionally to its SoC. SoC is limited to
    [0.01, 0.99] by scaling the request.

    Notes
    -----
    Validity: seconds-to-minutes duty cycles; no speed-dependent power limit
    or bearing/windage model beyond the lumped self-discharge.
    """

    units = {
        "P_cmd": "MW",
        "P_actual": "MW", "SoC": "1",
    }

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
        self.soc_min = 0.01
        self.soc_max = 0.99

        self.soc = float(soc_init)
        self.p_actual = 0.0

        self.register_variables(
            inputs=["P_cmd"],
            outputs=["P_actual", "SoC"],
        )

    def step_model(self, dt: float, time: float) -> None:
        p_cmd = float(self._inputs_val.get("P_cmd", 0.0))
        p_req = max(-self.p_max, min(self.p_max, p_cmd))

        sqrt_eff = math.sqrt(self.eff)
        # Power in MW -> kWh/s = P_MW * 1000 / 3600
        soc_per_mw_s = (1000.0 / 3600.0) / self.capacity_kwh
        loss = self.self_discharge_rate * self.soc * dt
        p_req = _fit_to_bounds(
            p_req, self.soc,
            k_chg=sqrt_eff * dt * soc_per_mw_s, k_dis=dt * soc_per_mw_s / sqrt_eff,
            loss=loss, lo=self.soc_min, hi=self.soc_max,
        )

        if p_req > 0:
            d_soc = p_req * sqrt_eff * dt * soc_per_mw_s
        elif p_req < 0:
            d_soc = p_req / sqrt_eff * dt * soc_per_mw_s
        else:
            d_soc = 0.0
        d_soc -= loss

        self.soc = max(0.0, min(1.0, self.soc + d_soc))
        self.p_actual = p_req
        self._outputs_val["P_actual"] = self.p_actual
        self._outputs_val["SoC"] = self.soc

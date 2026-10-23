# -*- coding: utf-8 -*-
"""
Thermal Turbomachinery, Engines and Cogeneration Models for energysim.

Includes:
- GasTurbine_OCGT: Peaking open-cycle gas turbine with ambient derating,
  part-load heat rate, Cold/Warm/Hot startup timers, min up/down time and
  ramp limits.
- CombinedCycle_CCGT: Combined cycle plant as a fixed GT/ST power split with
  ambient capacity derate, part-load heat rate, ramp limits, optional min
  up/down time and CO2 accounting (no HRSG / Rankine cycle model).
- ReciprocatingGenset: Fast-start gas/diesel engine generator with
  fixed-efficiency heat recovery.
- CHP_SteamTurbine: Industrial cogeneration steam turbine, simple correlation
  with a power-loss coefficient for extraction.
"""

from __future__ import annotations
import math
from typing import Optional, Sequence
import numpy as np

from .common.base_component import ModelComponent
from .common.curve import Curve1D
from .common.state_machine import OperatingStateMachine, OperatingState


class GasTurbine_OCGT(ModelComponent):
    """
    Open-Cycle Gas Turbine (OCGT) peaker plant.

    A unit-commitment style model: an operating state machine
    (OFF -> STARTING -> RUNNING -> OFF) with cold/warm/hot start durations,
    enforced minimum up- and down-time, ramp limits, an ambient-temperature
    capacity derate curve and a part-load heat-rate curve. Once STARTING the
    start always completes (command dips do not abort it) and the unit
    synchronises at minimum stable generation (MSG), then ramps toward the
    setpoint. While below ``min_uptime_sec`` a stop command holds the unit at
    MSG. A start command is ignored until ``min_downtime_sec`` has elapsed.

    Validity: T_amb -20..45 C, loads between MSG and rated; no emissions
    transients, no turbine inertia dynamics, no humidity or altitude effects.

    Parameters
    ----------
    p_rated_mw : float
        Nominal ISO rated electrical capacity in MW (at 15 C, sea level).
    heat_rate_full_load_kj_kwh : float, default 9800.0
        Full load heat rate in kJ/kWh (LHV).
    ramp_rate_mw_min : float, default 30.0
        Maximum ramp rate in MW/min.
    min_stable_gen_pct : float, default 30.0
        Minimum Stable Generation (MSG) limit in % of rated power.
    cold_start_duration_sec : float, default 1200.0
        Time to synchronize and reach MSG from cold state.
    warm_start_duration_sec : float, default 600.0
        Time to synchronize from warm state.
    hot_start_duration_sec : float, default 300.0
        Time to synchronize from hot state.
    min_uptime_sec : float, default 1800.0
        Minimum running time (after synchronisation) before a stop is allowed.
    min_downtime_sec : float, default 1800.0
        Minimum offline time before a restart is allowed.
    """

    units = {
        "P_cmd": "MW", "T_amb": "degC",
        "P_actual": "MW", "fuel_consumption_MMBtu_hr": "MMBtu/h", "Q_exhaust_MW": "MW",
        "state": "-",
    }

    def __init__(
        self,
        name: str = "OCGT",
        p_rated_mw: float = 100.0,
        heat_rate_full_load_kj_kwh: float = 9800.0,
        ramp_rate_mw_min: float = 30.0,
        min_stable_gen_pct: float = 30.0,
        cold_start_duration_sec: float = 1200.0,
        warm_start_duration_sec: float = 600.0,
        hot_start_duration_sec: float = 300.0,
        ambient_derate_curve: Optional[Curve1D] = None,
        heat_rate_curve: Optional[Curve1D] = None,
        step_size: float = 1.0,
        min_uptime_sec: float = 1800.0,
        min_downtime_sec: float = 1800.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.p_rated_mw = p_rated_mw
        self.hr_full = heat_rate_full_load_kj_kwh
        self.msg_mw = p_rated_mw * (min_stable_gen_pct / 100.0)

        # State machine
        self.sm = OperatingStateMachine(
            ramp_up_rate_min=ramp_rate_mw_min,
            ramp_down_rate_min=ramp_rate_mw_min,
            min_uptime_sec=min_uptime_sec,
            min_downtime_sec=min_downtime_sec,
            cold_start_duration_sec=cold_start_duration_sec,
            warm_start_duration_sec=warm_start_duration_sec,
            hot_start_duration_sec=hot_start_duration_sec,
        )

        # Ambient temperature capacity multiplier
        if ambient_derate_curve is None:
            t_pts = [-20.0, -10.0, 0.0, 15.0, 25.0, 35.0, 45.0]
            mult_pts = [1.12, 1.08, 1.05, 1.00, 0.95, 0.89, 0.82]
            self.amb_curve = Curve1D(t_pts, mult_pts, min_val=0.7, max_val=1.2)
        else:
            self.amb_curve = ambient_derate_curve

        # Part-load heat rate multiplier
        if heat_rate_curve is None:
            load_pts = [0.2, 0.3, 0.5, 0.75, 1.0]
            hr_mult = [1.45, 1.30, 1.15, 1.05, 1.00]
            self.hr_curve = Curve1D(load_pts, hr_mult, min_val=1.0, max_val=1.8)
        else:
            self.hr_curve = heat_rate_curve

        self.p_actual = 0.0
        self.fuel_mmbtu_hr = 0.0
        self.q_exhaust_mw = 0.0
        self.state_str = "OFF"

        self.register_variables(
            inputs=["P_cmd", "T_amb"],
            outputs=["P_actual", "fuel_consumption_MMBtu_hr", "Q_exhaust_MW", "state"],
            defaults={"T_amb": 15.0},
        )

    def step_model(self, dt: float, time: float) -> None:
        p_cmd = float(self._inputs_val.get("P_cmd", 0.0))
        t_amb = float(self._inputs_val.get("T_amb", 15.0))

        self.sm.update_time(dt)
        sm = self.sm
        cap_mult = float(self.amb_curve(t_amb))
        p_max_avail = self.p_rated_mw * cap_mult
        want_run = p_cmd >= self.msg_mw

        if sm.state == OperatingState.OFF:
            self.p_actual = 0.0
            self.fuel_mmbtu_hr = 0.0
            self.q_exhaust_mw = 0.0
            if want_run and sm.can_start():
                sm.transition(OperatingState.STARTING)
                sm.current_startup_category = sm.determine_startup_category()
                sm.startup_timer_sec = sm.get_startup_duration()

        if sm.state == OperatingState.STARTING:
            # Once started, the start sequence always completes.
            sm.startup_timer_sec -= dt
            # Startup fuel penalty: ~15% of rated fuel burn during purge/roll
            self.fuel_mmbtu_hr = (self.p_rated_mw * 1000.0 * self.hr_full * 0.15) / 1.055e6
            self.p_actual = 0.0
            self.q_exhaust_mw = 0.0
            if sm.startup_timer_sec <= 0:
                sm.transition(OperatingState.RUNNING)
                self.p_actual = min(self.msg_mw, p_max_avail)  # synchronise at MSG, ramp from here

        elif sm.state == OperatingState.RUNNING:
            if want_run:
                target = min(p_max_avail, p_cmd)
            elif not sm.can_stop():
                target = min(p_max_avail, self.msg_mw)  # hold at MSG until min uptime elapsed
            else:
                target = 0.0
            self.p_actual = sm.apply_ramp(self.p_actual, target, dt)
            if target <= 0.0 and self.p_actual <= 0.01:
                self.p_actual = 0.0
                sm.transition(OperatingState.OFF)
                self.fuel_mmbtu_hr = 0.0
                self.q_exhaust_mw = 0.0

        if sm.state == OperatingState.RUNNING:
            load_ratio = self.p_actual / self.p_rated_mw
            hr_mult = float(self.hr_curve(load_ratio))
            current_hr = self.hr_full * hr_mult
            # Fuel MMBtu/hr = P(MW) * 1000 kW * HR(kJ/kWh) / 1.055e6 kJ/MMBtu
            self.fuel_mmbtu_hr = (self.p_actual * 1000.0 * current_hr) / 1.055e6
            # Exhaust/loss thermal power = fuel input - electric output
            fuel_mw = self.fuel_mmbtu_hr * (1.055e6 / 3600.0) / 1000.0
            self.q_exhaust_mw = max(0.0, fuel_mw - self.p_actual)

        self.state_str = self.sm.state.value
        self._outputs_val["P_actual"] = self.p_actual
        self._outputs_val["fuel_consumption_MMBtu_hr"] = self.fuel_mmbtu_hr
        self._outputs_val["Q_exhaust_MW"] = self.q_exhaust_mw
        self._outputs_val["state"] = self.state_str


class CombinedCycle_CCGT(ModelComponent):
    """
    Combined Cycle Gas Turbine (CCGT), plant-level surrogate.

    The plant is a single dispatchable block: electrical output follows the
    setpoint subject to a ramp limit, an ambient-temperature capacity derate
    curve (``T_amb`` input) and minimum stable generation. Output is split into
    a *fixed* gas-turbine / steam-turbine share (``gt_ratio``); fuel use follows
    a part-load heat-rate curve; CO2 follows from fuel. There is no HRSG,
    Rankine cycle or steam-side dynamics. Minimum up/down time are enforced
    through the shared operating state machine (defaults 0 = disabled, i.e.
    ramp-only behaviour); there is no cold/warm/hot start sequence.

    Validity: T_amb -20..45 C, loads between MSG and rated.

    Parameters
    ----------
    p_rated_mw : float
        Total combined cycle plant rated capacity (GT + ST) in MW.
    gt_ratio : float, default 0.67
        Fraction of total capacity delivered by the gas turbine (typically 2/3).
    heat_rate_full_load_kj_kwh : float, default 6400.0
        Full load CCGT plant heat rate (kJ/kWh LHV, ~56.2% efficiency).
    min_stable_gen_pct : float, default 40.0
        Minimum stable generation limit (% of total plant capacity).
    ramp_rate_mw_min : float, default 15.0
        Maximum CCGT ramp rate in MW/min.
    min_uptime_sec, min_downtime_sec : float, default 0.0
        Minimum running / offline time (0 disables).
    ambient_derate_curve : Curve1D, optional
        Capacity multiplier vs ambient temperature in degC.
    """

    units = {
        "P_cmd": "MW", "T_amb": "degC",
        "P_actual": "MW", "P_gt": "MW", "P_st": "MW",
        "fuel_consumption_MMBtu_hr": "MMBtu/h", "CO2_emissions_t_hr": "t/h",
        "Q_condenser_MW": "MW", "state": "-",
    }

    def __init__(
        self,
        name: str = "CCGT",
        p_rated_mw: float = 400.0,
        gt_ratio: float = 0.67,
        heat_rate_full_load_kj_kwh: float = 6400.0,
        min_stable_gen_pct: float = 40.0,
        ramp_rate_mw_min: float = 15.0,
        step_size: float = 1.0,
        min_uptime_sec: float = 0.0,
        min_downtime_sec: float = 0.0,
        ambient_derate_curve: Optional[Curve1D] = None,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.p_rated_mw = p_rated_mw
        self.gt_ratio = gt_ratio
        self.st_ratio = 1.0 - gt_ratio
        self.hr_full = heat_rate_full_load_kj_kwh
        self.msg_mw = p_rated_mw * (min_stable_gen_pct / 100.0)
        self.ramp_rate = ramp_rate_mw_min
        self.sm = OperatingStateMachine(
            ramp_up_rate_min=ramp_rate_mw_min,
            ramp_down_rate_min=ramp_rate_mw_min,
            min_uptime_sec=min_uptime_sec,
            min_downtime_sec=min_downtime_sec,
        )

        # Ambient temperature capacity multiplier (steam cycle derates less than a GT alone)
        if ambient_derate_curve is None:
            t_pts = [-20.0, -10.0, 0.0, 15.0, 25.0, 35.0, 45.0]
            mult_pts = [1.06, 1.04, 1.02, 1.00, 0.97, 0.94, 0.90]
            self.amb_curve = Curve1D(t_pts, mult_pts, min_val=0.7, max_val=1.2)
        else:
            self.amb_curve = ambient_derate_curve

        # Heat rate multiplier curve
        load_pts = [0.35, 0.5, 0.7, 0.85, 1.0]
        hr_mult = [1.32, 1.18, 1.07, 1.02, 1.00]
        self.hr_curve = Curve1D(load_pts, hr_mult, min_val=1.0, max_val=1.5)

        self.p_actual = 0.0
        self.p_gt = 0.0
        self.p_st = 0.0
        self.fuel_mmbtu_hr = 0.0
        self.co2_emissions_t_hr = 0.0
        self.q_condenser_mw = 0.0

        self.register_variables(
            inputs=["P_cmd", "T_amb"],
            outputs=[
                "P_actual", "P_gt", "P_st",
                "fuel_consumption_MMBtu_hr", "CO2_emissions_t_hr", "Q_condenser_MW", "state",
            ],
            defaults={"T_amb": 15.0},
        )

    def step_model(self, dt: float, time: float) -> None:
        p_cmd = float(self._inputs_val.get("P_cmd", 0.0))
        t_amb = float(self._inputs_val.get("T_amb", 15.0))

        sm = self.sm
        sm.update_time(dt)
        p_avail = self.p_rated_mw * float(self.amb_curve(t_amb))
        want_run = p_cmd >= self.msg_mw

        if sm.state == OperatingState.OFF and want_run and sm.can_start():
            sm.transition(OperatingState.RUNNING)

        if sm.state == OperatingState.RUNNING:
            if want_run:
                target = min(p_avail, p_cmd)
            elif not sm.can_stop():
                target = min(p_avail, self.msg_mw)  # hold at MSG until min uptime elapsed
            else:
                target = 0.0
            self.p_actual = sm.apply_ramp(self.p_actual, target, dt)
            if target <= 0.0 and self.p_actual <= 0.01:
                self.p_actual = 0.0
                sm.transition(OperatingState.OFF)
        else:
            self.p_actual = 0.0

        if self.p_actual > 0.01:
            self.p_gt = self.p_actual * self.gt_ratio
            self.p_st = self.p_actual * self.st_ratio

            load_ratio = self.p_actual / self.p_rated_mw
            hr = self.hr_full * float(self.hr_curve(load_ratio))
            self.fuel_mmbtu_hr = (self.p_actual * 1000.0 * hr) / 1.055e6

            # CO2 emissions: ~56.1 kg CO2 / GJ = 0.0592 t CO2 / MMBtu for natural gas
            self.co2_emissions_t_hr = self.fuel_mmbtu_hr * 0.0592

            # Condenser cooling load: fuel energy - total electrical power
            fuel_mw = self.fuel_mmbtu_hr * (1.055e6 / 3600.0) / 1000.0
            self.q_condenser_mw = max(0.0, fuel_mw - self.p_actual - (fuel_mw * 0.12))  # ~12% stack loss
        else:
            self.p_gt = 0.0
            self.p_st = 0.0
            self.fuel_mmbtu_hr = 0.0
            self.co2_emissions_t_hr = 0.0
            self.q_condenser_mw = 0.0

        self._outputs_val["P_actual"] = self.p_actual
        self._outputs_val["P_gt"] = self.p_gt
        self._outputs_val["P_st"] = self.p_st
        self._outputs_val["fuel_consumption_MMBtu_hr"] = self.fuel_mmbtu_hr
        self._outputs_val["CO2_emissions_t_hr"] = self.co2_emissions_t_hr
        self._outputs_val["Q_condenser_MW"] = self.q_condenser_mw
        self._outputs_val["state"] = sm.state.value


class ReciprocatingGenset(ModelComponent):
    """
    Reciprocating Internal Combustion Engine (Genset) with thermal heat recovery.

    Fixed electrical and heat-recovery efficiencies (no part-load curve, no
    start-up delay or ramp limit). Validity: 0 to rated power; the sum of
    electrical and heat-recovery efficiency must not exceed 1.
    """

    units = {
        "P_cmd": "MW",
        "P_actual": "MW", "fuel_consumption_MMBtu_hr": "MMBtu/h", "Q_thermal_MW": "MW",
    }

    def __init__(
        self,
        name: str = "Genset",
        p_rated_mw: float = 10.0,
        electrical_efficiency: float = 0.44,
        heat_recovery_efficiency: float = 0.42,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.p_rated = p_rated_mw
        self.eff_elec = electrical_efficiency
        self.eff_th = heat_recovery_efficiency

        self.p_actual = 0.0
        self.fuel_mmbtu_hr = 0.0
        self.q_thermal_mw = 0.0

        self.register_variables(
            inputs=["P_cmd"],
            outputs=["P_actual", "fuel_consumption_MMBtu_hr", "Q_thermal_MW"],
        )

    def step_model(self, dt: float, time: float) -> None:
        p_cmd = float(self._inputs_val.get("P_cmd", 0.0))
        self.p_actual = min(self.p_rated, max(0.0, p_cmd))

        if self.p_actual > 0.01:
            fuel_input_mw = self.p_actual / self.eff_elec
            self.fuel_mmbtu_hr = fuel_input_mw * 3600.0 / 1055.06
            self.q_thermal_mw = fuel_input_mw * self.eff_th
        else:
            self.fuel_mmbtu_hr = 0.0
            self.q_thermal_mw = 0.0

        self._outputs_val["P_actual"] = self.p_actual
        self._outputs_val["fuel_consumption_MMBtu_hr"] = self.fuel_mmbtu_hr
        self._outputs_val["Q_thermal_MW"] = self.q_thermal_mw


class CHP_SteamTurbine(ModelComponent):
    """
    Combined Heat and Power (CHP) extraction steam turbine, simple correlation.

    This is a correlation, not a thermodynamic turbine model: base electrical
    power is taken as 1 MW per kg/s of live steam (capped at ``p_max_mw``),
    delivered heat is limited to twice the base power, and extraction reduces
    electrical output by ``beta`` MW per MW of heat extracted
    (``P = P_base - beta * Q``). No pressure levels, enthalpy drop, condensing
    tail or part-load efficiency are modelled.

    Validity: steam flow 0-``p_max_mw`` kg/s, heat demand <= 2 x base power.
    """

    units = {
        "steam_flow_kg_s": "kg/s", "heat_demand_MW": "MW",
        "P_actual": "MW", "Q_heat_delivered_MW": "MW",
    }

    def __init__(
        self,
        name: str = "CHP_Turbine",
        p_max_mw: float = 50.0,
        power_loss_coefficient_beta: float = 0.18,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.p_max = p_max_mw
        self.beta = power_loss_coefficient_beta

        self.p_actual = 0.0
        self.q_heat_mw = 0.0

        self.register_variables(
            inputs=["steam_flow_kg_s", "heat_demand_MW"],
            outputs=["P_actual", "Q_heat_delivered_MW"],
            defaults={"steam_flow_kg_s": 50.0},
        )

    def step_model(self, dt: float, time: float) -> None:
        steam_flow = max(0.0, float(self._inputs_val.get("steam_flow_kg_s", 50.0)))
        heat_dem = max(0.0, float(self._inputs_val.get("heat_demand_MW", 0.0)))

        # Max electrical power without extraction ~ 1 MW per kg/s high-pressure steam
        p_base = steam_flow * 1.0
        p_base = min(self.p_max, p_base)

        self.q_heat_mw = min(heat_dem, p_base * 2.0)
        # Power derating due to steam extraction: P = P_base - beta * Q_heat
        self.p_actual = max(0.0, p_base - self.beta * self.q_heat_mw)

        self._outputs_val["P_actual"] = self.p_actual
        self._outputs_val["Q_heat_delivered_MW"] = self.q_heat_mw

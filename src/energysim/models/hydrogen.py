# -*- coding: utf-8 -*-
"""
Hydrogen and Power-to-X (PtX) Models for energysim.

Includes:
- PEMElectrolyzer: PEM electrolyzer using an empirical lookup polarization curve
  (current density -> cell voltage), Faraday efficiency, stack thermal mass and water
  stoichiometry.
- AlkalineElectrolyzer: Liquid KOH electrolyzer, system-level specific consumption with a
  part-load penalty and a minimum-turndown trip.
- SOECElectrolyzer: High-temperature solid oxide steam electrolysis (specific-consumption
  model with steam enthalpy heat demand).
- H2Compressor: Multi-stage compressor with equal pressure ratios, isentropic efficiency
  and intercooling.
- H2StorageTank: Pressurized hydrogen buffer tank (simple linear-Z real gas) with
  capacity enforcement and rejected / unserved flow accounting.
- FuelCell_PEM: Rapid-dispatch PEM fuel cell generator.
- FuelCell_SOFC: Baseload high-efficiency solid oxide fuel cell with high-grade CHP heat.
- MethanationReactor: Sabatier PtG reactor converting H2 + CO2 into synthetic methane.
"""

from __future__ import annotations
import math
from typing import Optional, Sequence
import numpy as np

from .common.base_component import ModelComponent
from .common.curve import Curve1D


# Physical Constants
FARADAY_CONSTANT = 96485.3321  # C/mol (A*s/mol)
MOLAR_MASS_H2 = 2.01588e-3     # kg/mol
MOLAR_MASS_H2O = 18.01528e-3   # kg/mol
LHV_H2_MJ_KG = 120.0           # MJ/kg Lower Heating Value
HHV_H2_MJ_KG = 141.8           # MJ/kg Higher Heating Value
R_GAS = 8.314462               # J/(mol*K)
V_THERMONEUTRAL_HHV = 1.48     # V, thermoneutral cell voltage (HHV basis, liquid water)
MOLAR_MASS_CO2 = 44.0095e-3    # kg/mol
# Chosen so that CO2 + 4 H2 -> CH4 + 2 H2O closes the mass balance exactly.
MOLAR_MASS_CH4 = MOLAR_MASS_CO2 + 4.0 * MOLAR_MASS_H2 - 2.0 * MOLAR_MASS_H2O


def _ramp_toward(current: float, target: float, max_delta: float) -> float:
    """Move ``current`` toward ``target`` by at most ``max_delta`` (>= 0)."""
    if abs(target - current) <= max_delta:
        return target
    return current + math.copysign(max_delta, target - current)




class PEMElectrolyzer(ModelComponent):
    """
    Proton Exchange Membrane (PEM) Electrolyzer.

    Empirical model: the cell voltage is read from an **empirical lookup polarization
    curve** V(j) (current density j in A/cm2) -- it is *not* an Ulleberg-type
    (Nernst + activation + ohmic) model, and it has no temperature dependence. The
    default curve is illustrative, not fitted to any particular stack.

    Model
    -----
    - The stack has ``n_cells`` cells of ``cell_area_cm2`` each. For a commanded power P
      the per-cell power density P/(n_cells*area) is inverted through ``j*V(j)`` to obtain
      the *actual* current density j, then I = j*area.
    - ``p_nominal_mw`` is made consistent with ``n_cells`` and ``cell_area_cm2``: it defines
      the nominal current density ``j_nominal`` (attribute); ``ValueError`` is raised if the
      stack geometry cannot absorb ``p_nominal_mw`` within the polarization curve range.
    - Faraday efficiency eta_F(j / j_nominal) (drops at low load because of gas crossover);
      H2 [mol/s] = eta_F * n_cells * I / (2 F).
    - Water consumption is stoichiometric (~9 kg H2O per kg H2).
    - Waste heat is taken relative to the **thermoneutral voltage** (1.48 V, HHV basis):
      q = n_cells * I * (V_cell - 1.48), clipped to >= 0 (below 1.48 V the cell would
      absorb heat from the surroundings; that endothermic regime is not modelled).
      Gas-crossover losses are ignored in the heat term.
    - Lumped stack thermal mass: C_th dT/dt = Q_waste - Q_cooling.
    - Optional ramp limit ``ramp_rate_pct_per_s`` (percent of nominal power per second);
      ``None`` means unlimited. Below ``min_turndown`` the stack idles (zero production).

    A step with ``dt == 0`` leaves the dynamic state (power, stack temperature) unchanged
    and only refreshes the outputs.

    Parameters
    ----------
    name : str
        Unique component name identifier
    p_nominal_mw : float
        Nominal rated electrical power consumption

    Inputs
    ------
    P_cmd : MW
        Electrical power command allocated to the electrolyzer

    Outputs
    -------
    P_actual : MW
        Actual active power absorbed from the power network
    m_dot_H2_kg_hr : kg/h
        Mass flow rate of high-purity hydrogen produced
    Q_waste_MW : MW
        Thermal power dissipated into cooling water system

    Notes
    -----
    Validity: stack power between min_turndown * p_nominal_mw and p_nominal_mw (the polarization
    curve ends at 2.5 A/cm2); steady-state electrochemistry with no temperature dependence of the cell voltage
    and no degradation; dynamics are limited to the ramp limit and the lumped stack thermal mass.
    """

    units = {
        "P_cmd": "MW", "T_cooling_in": "degC",
        "P_actual": "MW", "m_dot_H2_kg_hr": "kg/h", "m_dot_H2O_kg_hr": "kg/h",
        "Q_waste_MW": "MW", "T_stack": "degC", "specific_consumption_kWh_kg": "kWh/kg",
        "I_stack_A": "A", "j_A_cm2": "A/cm2", "V_cell": "V", "eta_faraday": "1",
    }

    def __init__(
        self,
        name: str = "PEM_Electrolyzer",
        p_nominal_mw: float = 5.0,
        cell_area_cm2: float = 1000.0,
        n_cells: int = 1500,
        t_stack_nom_c: float = 70.0,
        min_turndown: float = 0.05,
        c_th_kj_k: float = 120000.0,
        ramp_rate_pct_per_s: Optional[float] = None,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.p_nominal_mw = p_nominal_mw
        self.cell_area_cm2 = cell_area_cm2
        self.n_cells = n_cells
        self.t_stack_nom_c = t_stack_nom_c
        self.min_turndown = min_turndown
        self.c_th_kj_k = c_th_kj_k
        self.ramp_rate_pct_per_s = ramp_rate_pct_per_s

        # Polarization curve: cell voltage vs current density (A/cm2). Empirical lookup.
        j_pts = [0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5]
        v_pts = [1.23, 1.55, 1.68, 1.78, 1.95, 2.10, 2.25, 2.40]
        self.polarization_curve = Curve1D(j_pts, v_pts, min_val=1.23, max_val=3.0)

        # Monotone table of areal power density w = j*V(j) [W/cm2] for inversion w -> j.
        self._j_grid = np.linspace(0.0, j_pts[-1], 2501)
        self._w_grid = self._j_grid * np.asarray(self.polarization_curve(self._j_grid))
        w_nom = (p_nominal_mw * 1e6) / (n_cells * cell_area_cm2)
        if w_nom > self._w_grid[-1]:
            raise ValueError(
                f"{name}: p_nominal_mw={p_nominal_mw} MW exceeds what {n_cells} cells x "
                f"{cell_area_cm2} cm2 can absorb ({self._w_grid[-1] * n_cells * cell_area_cm2 / 1e6:.2f} MW "
                f"at j={j_pts[-1]} A/cm2). Increase n_cells or cell_area_cm2."
            )
        self.j_nominal = float(np.interp(w_nom, self._w_grid, self._j_grid))

        # Faraday efficiency curve vs relative load j / j_nominal
        load_pts = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0, 1.2]
        faraday_pts = [0.0, 0.70, 0.85, 0.94, 0.97, 0.985, 0.99, 0.99]
        self.faraday_curve = Curve1D(load_pts, faraday_pts, min_val=0.0, max_val=0.995)

        # Dynamic states
        self.t_stack_c = float(t_stack_nom_c)
        self.p_actual_mw = 0.0
        self.m_dot_h2_kg_hr = 0.0
        self.m_dot_h2o_kg_hr = 0.0
        self.q_waste_mw = 0.0
        self.spec_consumption_kwh_kg = 0.0
        self.i_stack_a = 0.0
        self.j_a_cm2 = 0.0
        self.v_cell = 0.0
        self.eta_faraday = 0.0

        self.register_variables(
            inputs=["P_cmd", "T_cooling_in"],
            outputs=[
                "P_actual", "m_dot_H2_kg_hr", "m_dot_H2O_kg_hr",
                "Q_waste_MW", "T_stack", "specific_consumption_kWh_kg",
                "I_stack_A", "j_A_cm2", "V_cell", "eta_faraday",
            ],
        )
        self._outputs_val["T_stack"] = self.t_stack_c

    def step_model(self, dt: float, time: float) -> None:
        p_cmd = float(self._inputs_val.get("P_cmd", 0.0))
        t_cool_in = float(self._inputs_val.get("T_cooling_in", 25.0))

        target = 0.0
        if p_cmd >= self.min_turndown * self.p_nominal_mw:
            target = min(self.p_nominal_mw, p_cmd)
        if dt > 0.0:
            if self.ramp_rate_pct_per_s is None:
                self.p_actual_mw = target
            else:
                max_delta = self.ramp_rate_pct_per_s / 100.0 * self.p_nominal_mw * dt
                self.p_actual_mw = _ramp_toward(self.p_actual_mw, target, max_delta)

        if self.p_actual_mw <= 0.0:
            self.m_dot_h2_kg_hr = 0.0
            self.m_dot_h2o_kg_hr = 0.0
            self.q_waste_mw = 0.0
            self.spec_consumption_kwh_kg = 0.0
            self.i_stack_a = self.j_a_cm2 = self.v_cell = self.eta_faraday = 0.0
            # Thermal decay towards the coolant inlet temperature
            self.t_stack_c += ((t_cool_in - self.t_stack_c) / 3600.0) * dt
        else:
            w = (self.p_actual_mw * 1e6) / (self.n_cells * self.cell_area_cm2)
            j = float(np.interp(w, self._w_grid, self._j_grid))
            v_cell = float(self.polarization_curve(j))
            i_stack = j * self.cell_area_cm2
            eta_f = float(self.faraday_curve(j / self.j_nominal))

            # Faraday's law: mol/s = eta_F * n_cells * I / (2 F)
            n_dot_h2 = eta_f * self.n_cells * i_stack / (2.0 * FARADAY_CONSTANT)
            self.m_dot_h2_kg_hr = n_dot_h2 * MOLAR_MASS_H2 * 3600.0
            self.m_dot_h2o_kg_hr = n_dot_h2 * MOLAR_MASS_H2O * 3600.0
            self.spec_consumption_kwh_kg = (
                (self.p_actual_mw * 1000.0) / self.m_dot_h2_kg_hr if self.m_dot_h2_kg_hr > 0.01 else 0.0
            )
            self.j_a_cm2, self.v_cell, self.i_stack_a, self.eta_faraday = j, v_cell, i_stack, eta_f

            # Waste heat relative to the thermoneutral voltage (HHV basis)
            q_w = self.n_cells * i_stack * (v_cell - V_THERMONEUTRAL_HHV)
            self.q_waste_mw = max(0.0, q_w) / 1e6

            # Thermal balance: cooling loop regulates the stack towards t_stack_nom_c
            q_cooling_mw = max(0.0, self.q_waste_mw * 0.9 + 0.05 * (self.t_stack_c - self.t_stack_nom_c))
            net_heat_kj_s = (self.q_waste_mw - q_cooling_mw) * 1000.0
            self.t_stack_c += (net_heat_kj_s / self.c_th_kj_k) * dt

        self._outputs_val["P_actual"] = self.p_actual_mw
        self._outputs_val["m_dot_H2_kg_hr"] = self.m_dot_h2_kg_hr
        self._outputs_val["m_dot_H2O_kg_hr"] = self.m_dot_h2o_kg_hr
        self._outputs_val["Q_waste_MW"] = self.q_waste_mw
        self._outputs_val["T_stack"] = self.t_stack_c
        self._outputs_val["specific_consumption_kWh_kg"] = self.spec_consumption_kwh_kg
        self._outputs_val["I_stack_A"] = self.i_stack_a
        self._outputs_val["j_A_cm2"] = self.j_a_cm2
        self._outputs_val["V_cell"] = self.v_cell
        self._outputs_val["eta_faraday"] = self.eta_faraday


class AlkalineElectrolyzer(ModelComponent):
    """
    Alkaline Electrolyzer (AEL) with liquid KOH electrolyte.

    System-level specific-consumption model (no stack polarization). The specific
    consumption of the whole plant worsens at part load because balance-of-plant
    consumers (pumps, gas conditioning, rectifier) do not scale down with load:

        spec(load) = spec_nominal * (1 + part_load_penalty * (1 - load))

    ``part_load_penalty`` is a dimensionless empirical parameter (default 0.15, i.e. +15 %
    at zero load, linearly interpolated). It is a system-level approximation, not a
    stack-level effect (stack efficiency actually improves at low current density).

    Below ``min_turndown_pct`` of nominal the plant trips to STANDBY (gas-crossover
    safety). ``ramp_rate_pct_per_s`` (percent of nominal per second, ``None`` = unlimited)
    limits power changes. ``state`` is a string output: OFF (never stepped with dt>0),
    STANDBY or RUNNING. A step with ``dt == 0`` leaves the dynamic state unchanged.

    Parameters
    ----------
    name : str
        Unique component name identifier
    p_nominal_mw : float
        Nominal rated plant power
    min_turndown_pct : float
        Strict safety turndown limit (typically 20-25%)

    Inputs
    ------
    P_cmd : MW
        Power dispatch command

    Outputs
    -------
    P_actual : MW
        Actual active power absorbed
    m_dot_H2_kg_hr : kg/h
        Hydrogen mass production rate
    state : 1
        Operating status ('RUNNING', 'CUTOFF_TRIP', 'OFF')

    Notes
    -----
    Validity: between min_turndown_pct and 100 % of p_nominal_mw (below that the plant is in STANDBY);
    system-level steady-state specific consumption with no stack temperature, pressure or degradation model.
    """

    units = {
        "P_cmd": "MW", "P_actual": "MW", "m_dot_H2_kg_hr": "kg/h", "state": "1",
    }

    def __init__(
        self,
        name: str = "AEL_Electrolyzer",
        p_nominal_mw: float = 10.0,
        min_turndown_pct: float = 25.0,
        spec_consumption_kwh_kg: float = 53.0,
        part_load_penalty: float = 0.15,
        ramp_rate_pct_per_s: Optional[float] = None,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.p_nominal_mw = p_nominal_mw
        self.min_turndown = min_turndown_pct / 100.0
        self.spec_consumption_nominal = spec_consumption_kwh_kg
        self.part_load_penalty = part_load_penalty
        self.ramp_rate_pct_per_s = ramp_rate_pct_per_s

        self.p_actual = 0.0
        self.m_dot_h2_kg_hr = 0.0
        self.state = "OFF"

        self.register_variables(
            inputs=["P_cmd"],
            outputs=["P_actual", "m_dot_H2_kg_hr", "state"],
        )
        self._outputs_val["state"] = self.state

    def step_model(self, dt: float, time: float) -> None:
        p_cmd = float(self._inputs_val.get("P_cmd", 0.0))

        target = 0.0
        if p_cmd >= self.min_turndown * self.p_nominal_mw:
            target = min(self.p_nominal_mw, p_cmd)
        if dt > 0.0:
            if self.ramp_rate_pct_per_s is None:
                self.p_actual = target
            else:
                max_delta = self.ramp_rate_pct_per_s / 100.0 * self.p_nominal_mw * dt
                self.p_actual = _ramp_toward(self.p_actual, target, max_delta)
            self.state = "RUNNING" if self.p_actual > 0.0 else "STANDBY"

        if self.p_actual > 0.0:
            load_ratio = self.p_actual / self.p_nominal_mw
            spec_kwh = self.spec_consumption_nominal * (1.0 + self.part_load_penalty * (1.0 - load_ratio))
            self.m_dot_h2_kg_hr = (self.p_actual * 1000.0) / spec_kwh
        else:
            self.m_dot_h2_kg_hr = 0.0

        self._outputs_val["P_actual"] = self.p_actual
        self._outputs_val["m_dot_H2_kg_hr"] = self.m_dot_h2_kg_hr
        self._outputs_val["state"] = self.state


class SOECElectrolyzer(ModelComponent):
    """
    Solid Oxide Electrolyzer Cell (SOEC) for high-temperature steam electrolysis.

    Simple specific-consumption model: H2 = P / ``spec_elec_kwh_kg`` limited by the
    available steam (``spec_steam_kg_per_kg_h2`` kg steam per kg H2, stoichiometric
    minimum 8.94). There is no stack voltage / thermoneutral-heat model: the electricity
    figure assumes the heat for steam generation is supplied externally. That heat is
    reported as ``Q_steam_MW_th = m_steam * steam_enthalpy_kj_kg``, the thermal demand to
    turn feed water into steam at stack conditions (default 3900 kJ/kg: liquid water at
    ~25 degC to ~750 degC steam at ~1 bar; adjust to your steam source).

    ``operating_temp_c`` is only a validity guard (600-900 degC, ``ValueError`` otherwise);
    it does not enter the calculation. Stateless: ``dt`` has no effect.

    Parameters
    ----------
    name : str
        Unique component name identifier
    p_nominal_mw : float
        Nominal electrical stack power rating

    Inputs
    ------
    P_cmd : MW
        Active power setpoint

    Outputs
    -------
    P_actual : MW
        Actual electrical power consumed
    m_dot_H2_kg_hr : kg/h
        Hydrogen production rate

    Notes
    -----
    Validity: operating_temp_c between 600 and 900 degC (enforced at construction); steady state,
    constant specific electricity consumption independent of load and temperature; the heat for steam
    generation is supplied externally.
    """

    units = {
        "P_cmd": "MW", "steam_available_kg_hr": "kg/h",
        "P_actual": "MW", "m_dot_H2_kg_hr": "kg/h", "steam_consumed_kg_hr": "kg/h",
        "Q_steam_MW_th": "MW",
    }

    def __init__(
        self,
        name: str = "SOEC_Electrolyzer",
        p_nominal_mw: float = 2.0,
        operating_temp_c: float = 750.0,
        spec_elec_kwh_kg: float = 38.0,
        spec_steam_kg_per_kg_h2: float = 9.0,
        steam_enthalpy_kj_kg: float = 3900.0,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        if not 600.0 <= operating_temp_c <= 900.0:
            raise ValueError(
                f"{name}: operating_temp_c={operating_temp_c} outside the SOEC validity range 600-900 degC."
            )
        self.p_nominal_mw = p_nominal_mw
        self.temp_c = operating_temp_c
        self.spec_elec_kwh_kg = spec_elec_kwh_kg
        self.spec_steam_ratio = spec_steam_kg_per_kg_h2
        self.steam_enthalpy_kj_kg = steam_enthalpy_kj_kg

        self.p_actual = 0.0
        self.m_dot_h2_kg_hr = 0.0
        self.m_dot_steam_req_kg_hr = 0.0
        self.q_steam_mw_th = 0.0

        self.register_variables(
            inputs=["P_cmd", "steam_available_kg_hr"],
            outputs=["P_actual", "m_dot_H2_kg_hr", "steam_consumed_kg_hr", "Q_steam_MW_th"],
        )

    def step_model(self, dt: float, time: float) -> None:
        p_cmd = float(self._inputs_val.get("P_cmd", 0.0))
        steam_avail = float(self._inputs_val.get("steam_available_kg_hr", 1e6))

        if p_cmd <= 0.05 * self.p_nominal_mw or steam_avail <= 1.0:
            self.p_actual = 0.0
            self.m_dot_h2_kg_hr = 0.0
            self.m_dot_steam_req_kg_hr = 0.0
        else:
            p_cap = min(self.p_nominal_mw, p_cmd)
            h2_from_power = (p_cap * 1000.0) / self.spec_elec_kwh_kg
            h2_from_steam = steam_avail / self.spec_steam_ratio
            h2_prod = min(h2_from_power, h2_from_steam)
            self.m_dot_h2_kg_hr = h2_prod
            self.p_actual = (h2_prod * self.spec_elec_kwh_kg) / 1000.0
            self.m_dot_steam_req_kg_hr = h2_prod * self.spec_steam_ratio

        self.q_steam_mw_th = self.m_dot_steam_req_kg_hr / 3600.0 * self.steam_enthalpy_kj_kg / 1000.0

        self._outputs_val["P_actual"] = self.p_actual
        self._outputs_val["m_dot_H2_kg_hr"] = self.m_dot_h2_kg_hr
        self._outputs_val["steam_consumed_kg_hr"] = self.m_dot_steam_req_kg_hr
        self._outputs_val["Q_steam_MW_th"] = self.q_steam_mw_th


class H2Compressor(ModelComponent):
    """
    Multi-stage hydrogen compressor with intercooling.

    ``n_stages`` stages (default 3) share the overall pressure ratio equally,
    r = (p_discharge / p_suction)^(1/n_stages). Each stage draws gas at temperature
    ``t_in_c`` (first stage) or ``t_intercool_c`` (after intercooling) and needs the
    isentropic work (ideal gas, constant gamma = 1.41, no compressibility correction)

        w_stage = gamma/(gamma-1) * R_H2 * T_in * (r^((gamma-1)/gamma) - 1) / eta_isen.

    ``P_elec_kW`` is the total shaft/electric work (no mechanical/motor losses beyond
    ``isentropic_efficiency``). ``Q_intercool_kW`` is the heat removed in the
    ``n_stages - 1`` intercoolers (cooling gas from the stage outlet temperature to
    ``t_intercool_c``); the after-cooler is not counted. Stateless: ``dt`` has no effect.

    Parameters
    ----------
    name : str
        Unique component name identifier

    Inputs
    ------
    m_dot_in_kg_hr : kg/h
        Inlet mass flow rate of hydrogen to compress

    Outputs
    -------
    m_dot_out_kg_hr : kg/h
        High-pressure delivery flow rate

    Notes
    -----
    Validity: ideal-gas hydrogen (gamma = 1.41, no compressibility correction), equal stage pressure
    ratios, steady-state flow up to max_flow_kg_hr; no start-up or surge behaviour.
    """

    units = {
        "m_dot_in_kg_hr": "kg/h", "P_elec_kW": "kW", "m_dot_out_kg_hr": "kg/h",
        "Q_intercool_kW": "kW",
    }

    GAMMA = 1.41
    R_SPEC_H2 = 4124.0  # J/(kg*K)

    def __init__(
        self,
        name: str = "H2Compressor",
        max_flow_kg_hr: float = 500.0,
        p_suction_bar: float = 30.0,
        p_discharge_bar: float = 350.0,
        isentropic_efficiency: float = 0.70,
        n_stages: int = 3,
        t_in_c: float = 20.0,
        t_intercool_c: float = 30.0,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        if int(n_stages) < 1:
            raise ValueError("n_stages must be >= 1")
        self.max_flow_kg_hr = max_flow_kg_hr
        self.p_in = p_suction_bar
        self.p_out = p_discharge_bar
        self.eta_isen = isentropic_efficiency
        self.n_stages = int(n_stages)
        self.t_in_c = t_in_c
        self.t_intercool_c = t_intercool_c

        self.p_elec_kw = 0.0
        self.q_intercool_kw = 0.0
        self.m_dot_actual_kg_hr = 0.0

        self.register_variables(
            inputs=["m_dot_in_kg_hr"],
            outputs=["P_elec_kW", "m_dot_out_kg_hr", "Q_intercool_kW"],
        )

    def specific_work_kj_kg(self) -> tuple:
        """Return (total specific work, specific intercooler heat) in kJ/kg."""
        if self.p_out <= self.p_in:
            return 0.0, 0.0
        g = self.GAMMA
        cp = g / (g - 1.0) * self.R_SPEC_H2  # J/(kg K)
        r = (self.p_out / self.p_in) ** (1.0 / self.n_stages)
        x = r ** ((g - 1.0) / g) - 1.0
        tic = self.t_intercool_c + 273.15
        w = 0.0
        q = 0.0
        t_stage_in = self.t_in_c + 273.15
        for stage in range(self.n_stages):
            w += cp * t_stage_in * x / self.eta_isen
            if stage < self.n_stages - 1:
                t_out = t_stage_in * (1.0 + x / self.eta_isen)
                q += cp * max(0.0, t_out - tic)
                t_stage_in = tic
        return w / 1000.0, q / 1000.0

    def step_model(self, dt: float, time: float) -> None:
        m_in = float(self._inputs_val.get("m_dot_in_kg_hr", 0.0))
        m_flow = min(self.max_flow_kg_hr, max(0.0, m_in))
        self.m_dot_actual_kg_hr = m_flow

        if m_flow > 0.01:
            w_kj_kg, q_kj_kg = self.specific_work_kj_kg()
            self.p_elec_kw = m_flow / 3600.0 * w_kj_kg
            self.q_intercool_kw = m_flow / 3600.0 * q_kj_kg
        else:
            self.p_elec_kw = 0.0
            self.q_intercool_kw = 0.0

        self._outputs_val["P_elec_kW"] = self.p_elec_kw
        self._outputs_val["m_dot_out_kg_hr"] = self.m_dot_actual_kg_hr
        self._outputs_val["Q_intercool_kW"] = self.q_intercool_kw


class H2StorageTank(ModelComponent):
    """
    Pressurized hydrogen storage tank (single-node, linear compressibility Z = 1 + 0.0006 p[bar]).

    The stored mass is the state; the reported pressure is always derived from the mass
    (p V = Z n R T), so mass and pressure are consistent. Capacity is the mass at
    ``p_max_bar``: inflow beyond it is rejected (``m_dot_rejected_kg_hr``) and withdrawals
    beyond the available mass are not delivered (``m_dot_unserved_kg_hr``). Mass is never
    negative and pressure never exceeds ``p_max_bar``. Per step (kg/h):

        m_in - m_out - m_rejected + m_unserved == d(mass)/dt

    ``p_min_bar`` is only used to define SoC (it is not a withdrawal limit). A step with
    ``dt == 0`` leaves the mass unchanged; rejected / unserved then report the flow that
    could not be handled given the current fill level.

    Parameters
    ----------
    name : str
        Unique component name identifier
    volume_m3 : float
        Internal geometrical tank volume
    p_max_bar : float
        Maximum allowable working pressure (MAWP)
    p_min_bar : float
        Minimum residual suction pressure
    p_init_bar : float
        Initial storage tank pressure

    Inputs
    ------
    m_dot_in_kg_hr : kg/h
        Inflow mass flow rate entering the tank (from electrolyzer/compressor)
    m_dot_out_kg_hr : kg/h
        Outflow mass flow rate dispatched to industrial loads or fuel cells

    Outputs
    -------
    pressure_bar : bar
        Current internal tank pressure
    mass_stored_kg : kg
        Total mass of hydrogen currently stored in the vessel
    SoC : 1
        Fill level between p_min_bar (0) and p_max_bar (1)

    Notes
    -----
    Validity: single-node, isothermal tank (gas temperature fixed at t_gas_c) with the linear
    compressibility Z = 1 + 0.0006 p[bar]; intended for pressures up to about 350 bar at roughly ambient
    temperature; no thermal effects of filling or discharging.
    """

    units = {
        "m_dot_in_kg_hr": "kg/h", "m_dot_out_kg_hr": "kg/h",
        "pressure_bar": "bar", "mass_stored_kg": "kg", "SoC": "1",
        "m_dot_rejected_kg_hr": "kg/h", "m_dot_unserved_kg_hr": "kg/h",
    }

    _ALPHA = 0.0006 / 1e5  # 1/Pa

    def __init__(
        self,
        name: str = "H2Storage",
        volume_m3: float = 50.0,
        p_max_bar: float = 350.0,
        p_min_bar: float = 20.0,
        p_init_bar: float = 150.0,
        t_gas_c: float = 20.0,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.volume_m3 = volume_m3
        self.p_max = p_max_bar
        self.p_min = p_min_bar
        self.t_k = t_gas_c + 273.15

        self.mass_capacity_kg = self._mass_from_pressure(p_max_bar)
        self.mass_stored_kg = min(self._mass_from_pressure(p_init_bar), self.mass_capacity_kg)
        self.pressure_bar = self._pressure_from_mass(self.mass_stored_kg)
        self.soc = self._soc(self.pressure_bar)
        self.m_dot_rejected_kg_hr = 0.0
        self.m_dot_unserved_kg_hr = 0.0

        self.register_variables(
            inputs=["m_dot_in_kg_hr", "m_dot_out_kg_hr"],
            outputs=["pressure_bar", "mass_stored_kg", "SoC",
                     "m_dot_rejected_kg_hr", "m_dot_unserved_kg_hr"],
        )
        self._publish()

    def _mass_from_pressure(self, p_bar: float) -> float:
        p_pa = p_bar * 1e5
        n = p_pa * self.volume_m3 / ((1.0 + self._ALPHA * p_pa) * R_GAS * self.t_k)
        return n * MOLAR_MASS_H2

    def _pressure_from_mass(self, mass_kg: float) -> float:
        n = mass_kg / MOLAR_MASS_H2
        denom = self.volume_m3 - self._ALPHA * n * R_GAS * self.t_k
        if denom <= 1e-9:
            return self.p_max
        return min(self.p_max, max(0.0, (n * R_GAS * self.t_k) / denom / 1e5))

    def _soc(self, p_bar: float) -> float:
        return max(0.0, min(1.0, (p_bar - self.p_min) / max(self.p_max - self.p_min, 1.0)))

    def _publish(self) -> None:
        self._outputs_val["pressure_bar"] = self.pressure_bar
        self._outputs_val["mass_stored_kg"] = self.mass_stored_kg
        self._outputs_val["SoC"] = self.soc
        self._outputs_val["m_dot_rejected_kg_hr"] = self.m_dot_rejected_kg_hr
        self._outputs_val["m_dot_unserved_kg_hr"] = self.m_dot_unserved_kg_hr

    def step_model(self, dt: float, time: float) -> None:
        m_in = max(0.0, float(self._inputs_val.get("m_dot_in_kg_hr", 0.0)))
        m_out = max(0.0, float(self._inputs_val.get("m_dot_out_kg_hr", 0.0)))

        if dt > 0.0:
            h = dt / 3600.0
            new_mass = self.mass_stored_kg + (m_in - m_out) * h
            rejected = unserved = 0.0
            if new_mass > self.mass_capacity_kg:
                rejected = new_mass - self.mass_capacity_kg
                new_mass = self.mass_capacity_kg
            elif new_mass < 0.0:
                unserved = -new_mass
                new_mass = 0.0
            self.mass_stored_kg = new_mass
            self.m_dot_rejected_kg_hr = rejected / h
            self.m_dot_unserved_kg_hr = unserved / h
        else:
            full = self.mass_stored_kg >= self.mass_capacity_kg - 1e-9
            empty = self.mass_stored_kg <= 1e-9
            self.m_dot_rejected_kg_hr = max(0.0, m_in - m_out) if full else 0.0
            self.m_dot_unserved_kg_hr = max(0.0, m_out - m_in) if empty else 0.0

        self.pressure_bar = self._pressure_from_mass(self.mass_stored_kg)
        self.soc = self._soc(self.pressure_bar)
        self._publish()


class FuelCell_PEM(ModelComponent):
    """
    Rapid-dispatch PEM Fuel Cell generator (constant electrical efficiency).

    Fuel (H2, LHV) = electricity + waste heat; ``Q_waste_MW`` is all non-electric fuel power.
    Output power ramps toward the command at ``ramp_rate_mw_min``. When idle (zero power)
    no fuel is consumed and no heat is produced. A step with ``dt == 0`` leaves the power
    state unchanged.

    Parameters
    ----------
    name : str
        Unique component name identifier

    Inputs
    ------
    P_cmd : MW
        Active power dispatch setpoint

    Outputs
    -------
    P_actual : MW
        Net active power delivered to the electrical bus

    Notes
    -----
    Validity: constant electrical efficiency independent of load, power between 0 and p_nominal_mw,
    ramp limited to ramp_rate_mw_min; no degradation or stack-temperature model.
    """

    units = {
        "P_cmd": "MW", "P_actual": "MW", "m_dot_H2_kg_hr": "kg/h", "Q_waste_MW": "MW",
    }

    def __init__(
        self,
        name: str = "PEM_FuelCell",
        p_nominal_mw: float = 1.0,
        electrical_efficiency: float = 0.50,
        ramp_rate_mw_min: float = 2.0,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.p_nom = p_nominal_mw
        self.eff = electrical_efficiency
        self.ramp_rate = ramp_rate_mw_min

        self.p_actual = 0.0
        self.m_dot_h2_kg_hr = 0.0
        self.q_waste_mw = 0.0

        self.register_variables(
            inputs=["P_cmd"],
            outputs=["P_actual", "m_dot_H2_kg_hr", "Q_waste_MW"],
        )

    def step_model(self, dt: float, time: float) -> None:
        p_cmd = float(self._inputs_val.get("P_cmd", 0.0))
        target = min(self.p_nom, max(0.0, p_cmd))

        self.p_actual = _ramp_toward(self.p_actual, target, self.ramp_rate * (max(dt, 0.0) / 60.0))

        if self.p_actual > 0.001:
            lhv_kwh_kg = LHV_H2_MJ_KG / 3.6
            self.m_dot_h2_kg_hr = (self.p_actual * 1000.0) / (self.eff * lhv_kwh_kg)
            chemical_input_mw = (self.m_dot_h2_kg_hr / 3600.0) * LHV_H2_MJ_KG
            self.q_waste_mw = max(0.0, chemical_input_mw - self.p_actual)
        else:
            self.m_dot_h2_kg_hr = 0.0
            self.q_waste_mw = 0.0

        self._outputs_val["P_actual"] = self.p_actual
        self._outputs_val["m_dot_H2_kg_hr"] = self.m_dot_h2_kg_hr
        self._outputs_val["Q_waste_MW"] = self.q_waste_mw


class FuelCell_SOFC(ModelComponent):
    """
    Solid Oxide Fuel Cell (SOFC) for high-efficiency baseload and high-grade CHP heat.

    Constant efficiencies (no load dependence, no ramp limit). Fuel LHV power splits into
    electricity (``electrical_efficiency``), recovered CHP heat (``thermal_efficiency``)
    and unrecovered losses ``Q_loss_MW``; the sum of the two efficiencies must not
    exceed 1. Idle: no fuel, no heat. Stateless: ``dt`` has no effect.

    Parameters
    ----------
    name : str
        Unique component name identifier

    Inputs
    ------
    P_cmd : MW
        Active power dispatch setpoint

    Outputs
    -------
    P_actual : MW
        Net active power delivered to the electrical bus

    Notes
    -----
    Validity: constant efficiencies independent of load, power between 0 and p_nominal_mw; no ramp
    limit and no thermal-cycling or start-up model.
    """

    units = {
        "P_cmd": "MW", "P_actual": "MW", "m_dot_H2_kg_hr": "kg/h",
        "Q_chp_MW": "MW", "Q_loss_MW": "MW",
    }

    def __init__(
        self,
        name: str = "SOFC_FuelCell",
        p_nominal_mw: float = 1.0,
        electrical_efficiency: float = 0.60,
        thermal_efficiency: float = 0.25,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        if electrical_efficiency + thermal_efficiency > 1.0 + 1e-12:
            raise ValueError("electrical_efficiency + thermal_efficiency must not exceed 1.")
        self.p_nom = p_nominal_mw
        self.eff_elec = electrical_efficiency
        self.eff_th = thermal_efficiency

        self.p_actual = 0.0
        self.m_dot_h2_kg_hr = 0.0
        self.q_chp_mw = 0.0
        self.q_loss_mw = 0.0

        self.register_variables(
            inputs=["P_cmd"],
            outputs=["P_actual", "m_dot_H2_kg_hr", "Q_chp_MW", "Q_loss_MW"],
        )

    def step_model(self, dt: float, time: float) -> None:
        p_cmd = float(self._inputs_val.get("P_cmd", 0.0))
        self.p_actual = min(self.p_nom, max(0.0, p_cmd))

        if self.p_actual > 0.001:
            lhv_kwh_kg = LHV_H2_MJ_KG / 3.6
            self.m_dot_h2_kg_hr = (self.p_actual * 1000.0) / (self.eff_elec * lhv_kwh_kg)
            fuel_power_mw = (self.m_dot_h2_kg_hr / 3600.0) * LHV_H2_MJ_KG
            self.q_chp_mw = fuel_power_mw * self.eff_th
            self.q_loss_mw = fuel_power_mw - self.p_actual - self.q_chp_mw
        else:
            self.p_actual = 0.0
            self.m_dot_h2_kg_hr = 0.0
            self.q_chp_mw = 0.0
            self.q_loss_mw = 0.0

        self._outputs_val["P_actual"] = self.p_actual
        self._outputs_val["m_dot_H2_kg_hr"] = self.m_dot_h2_kg_hr
        self._outputs_val["Q_chp_MW"] = self.q_chp_mw
        self._outputs_val["Q_loss_MW"] = self.q_loss_mw


class MethanationReactor(ModelComponent):
    """
    Sabatier Power-to-Gas Methanation Reactor: CO2 + 4 H2 -> CH4 + 2 H2O.

    A fraction ``conversion_efficiency`` of the accepted H2 feed (capped at
    ``max_h2_feed_kg_hr``) reacts with stoichiometric CO2 (assumed available); the rest
    leaves unreacted (``m_dot_H2_unreacted_kg_hr``, no recycle). The water by-product is
    ``m_dot_H2O_kg_hr``. Mass balance (kg/h):

        H2_fed + CO2_consumed == CH4 + H2O + H2_unreacted

    Reaction heat is 165 kJ per mol CH4 (exothermic, at 25 degC). Stateless.

    Parameters
    ----------
    name : str
        Unique component name identifier
    conversion_efficiency : float
        Chemical conversion yield factor

    Outputs
    -------
    m_dot_CH4_kg_hr : kg/h
        Synthetic natural gas production mass flow rate

    Notes
    -----
    Validity: steady state with a constant conversion_efficiency independent of load, temperature and
    pressure; CO2 is assumed available; H2 feed up to max_h2_feed_kg_hr.
    """

    units = {
        "m_dot_H2_kg_hr": "kg/h",
        "m_dot_CH4_kg_hr": "kg/h", "m_dot_CO2_consumed_kg_hr": "kg/h",
        "Q_reaction_MW": "MW", "m_dot_H2_unreacted_kg_hr": "kg/h",
        "m_dot_H2O_kg_hr": "kg/h", "m_dot_H2_fed_kg_hr": "kg/h",
    }

    def __init__(
        self,
        name: str = "Methanation",
        max_h2_feed_kg_hr: float = 200.0,
        conversion_efficiency: float = 0.95,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.max_h2 = max_h2_feed_kg_hr
        self.eff = conversion_efficiency

        self.m_dot_ch4_kg_hr = 0.0
        self.m_dot_co2_kg_hr = 0.0
        self.q_reaction_mw = 0.0
        self.m_dot_h2_unreacted_kg_hr = 0.0
        self.m_dot_h2o_kg_hr = 0.0
        self.m_dot_h2_fed_kg_hr = 0.0

        self.register_variables(
            inputs=["m_dot_H2_kg_hr"],
            outputs=["m_dot_CH4_kg_hr", "m_dot_CO2_consumed_kg_hr", "Q_reaction_MW",
                     "m_dot_H2_unreacted_kg_hr", "m_dot_H2O_kg_hr", "m_dot_H2_fed_kg_hr"],
        )

    def step_model(self, dt: float, time: float) -> None:
        h2_in = float(self._inputs_val.get("m_dot_H2_kg_hr", 0.0))
        h2_feed = min(self.max_h2, max(0.0, h2_in))

        if h2_feed > 0.01:
            m_h2_react = h2_feed * self.eff
            mol_ch4_per_hr = (m_h2_react / MOLAR_MASS_H2) / 4.0
            self.m_dot_h2_fed_kg_hr = h2_feed
            self.m_dot_ch4_kg_hr = mol_ch4_per_hr * MOLAR_MASS_CH4
            self.m_dot_co2_kg_hr = mol_ch4_per_hr * MOLAR_MASS_CO2
            self.m_dot_h2o_kg_hr = 2.0 * mol_ch4_per_hr * MOLAR_MASS_H2O
            self.m_dot_h2_unreacted_kg_hr = h2_feed - m_h2_react
            self.q_reaction_mw = (mol_ch4_per_hr / 3600.0) * 165.0 / 1000.0
        else:
            self.m_dot_h2_fed_kg_hr = 0.0
            self.m_dot_ch4_kg_hr = 0.0
            self.m_dot_co2_kg_hr = 0.0
            self.m_dot_h2o_kg_hr = 0.0
            self.m_dot_h2_unreacted_kg_hr = 0.0
            self.q_reaction_mw = 0.0

        self._outputs_val["m_dot_CH4_kg_hr"] = self.m_dot_ch4_kg_hr
        self._outputs_val["m_dot_CO2_consumed_kg_hr"] = self.m_dot_co2_kg_hr
        self._outputs_val["Q_reaction_MW"] = self.q_reaction_mw
        self._outputs_val["m_dot_H2_unreacted_kg_hr"] = self.m_dot_h2_unreacted_kg_hr
        self._outputs_val["m_dot_H2O_kg_hr"] = self.m_dot_h2o_kg_hr
        self._outputs_val["m_dot_H2_fed_kg_hr"] = self.m_dot_h2_fed_kg_hr

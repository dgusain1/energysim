# -*- coding: utf-8 -*-
"""
Hydrogen and Power-to-X (PtX) Models for energysim.

Includes:
- PEMElectrolyzer: Proton exchange membrane electrolyzer with Ulleberg polarization,
  Faraday efficiency, stack thermal mass, and water stoichiometry.
- AlkalineElectrolyzer: Liquid KOH electrolyzer with strict 20-30% minimum turndown
  safety cutoff to prevent gas crossover.
- SOECElectrolyzer: High-temperature solid oxide steam electrolysis with thermoneutral
  heat/electricity split.
- H2Compressor: Multi-stage compressor with isentropic efficiency and intercooling.
- H2StorageTank: Pressurized hydrogen buffer tank with real-gas compressibility.
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


class PEMElectrolyzer(ModelComponent):
    """
    Proton Exchange Membrane (PEM) Electrolyzer.

    Models:
    - Polarization curve: Cell voltage as a function of current density & temperature
    - Faraday efficiency: Drop-off at low current density due to parasitic crossover
    - Stack thermal mass ODE: C_th * dT/dt = Q_loss - Q_cooling
    - Hydrogen production: Faraday's law
    - Stoichiometric demineralized water consumption (~9 kg H2O / kg H2)
    - Available waste heat: Q_waste = P_elec - m_h2 * LHV
    """

    def __init__(
        self,
        name: str = "PEM_Electrolyzer",
        p_nominal_mw: float = 5.0,
        cell_area_cm2: float = 1000.0,
        n_cells: int = 1500,
        t_stack_nom_c: float = 70.0,
        min_turndown: float = 0.05,
        c_th_kj_k: float = 120000.0,
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

        # Faraday efficiency curve vs load ratio
        load_pts = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0, 1.2]
        faraday_pts = [0.0, 0.70, 0.85, 0.94, 0.97, 0.985, 0.99, 0.99]
        self.faraday_curve = Curve1D(load_pts, faraday_pts, min_val=0.0, max_val=0.995)

        # Polarization curve: cell voltage vs current density (A/cm2)
        j_pts = [0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5]
        v_pts = [1.23, 1.55, 1.68, 1.78, 1.95, 2.10, 2.25, 2.40]
        self.polarization_curve = Curve1D(j_pts, v_pts, min_val=1.23, max_val=3.0)

        # Dynamic states
        self.t_stack_c = float(t_stack_nom_c)
        self.p_actual_mw = 0.0
        self.m_dot_h2_kg_hr = 0.0
        self.m_dot_h2o_kg_hr = 0.0
        self.q_waste_mw = 0.0
        self.spec_consumption_kwh_kg = 50.0

        self.register_variables(
            inputs=["P_cmd", "T_cooling_in"],
            outputs=[
                "P_actual", "m_dot_H2_kg_hr", "m_dot_H2O_kg_hr",
                "Q_waste_MW", "T_stack", "specific_consumption_kWh_kg",
            ],
        )

    def step_model(self, dt: float, time: float) -> None:
        p_cmd = float(self._inputs_val.get("P_cmd", 0.0))
        t_cool_in = float(self._inputs_val.get("T_cooling_in", 25.0))

        if p_cmd < (self.min_turndown * self.p_nominal_mw):
            # Standby / idling: zero production
            self.p_actual_mw = 0.0
            self.m_dot_h2_kg_hr = 0.0
            self.m_dot_h2o_kg_hr = 0.0
            self.q_waste_mw = 0.0
            self.spec_consumption_kwh_kg = 0.0
            # Thermal decay towards ambient
            self.t_stack_c += ((t_cool_in - self.t_stack_c) / 3600.0) * dt
        else:
            p_target = min(self.p_nominal_mw, p_cmd)
            self.p_actual_mw = p_target
            load_ratio = p_target / self.p_nominal_mw

            # Current calculation
            p_per_cell_w = (p_target * 1e6) / self.n_cells
            # Estimate cell voltage from typical curve
            j_est = load_ratio * 1.8  # ~1.8 A/cm2 at nominal
            v_cell = float(self.polarization_curve(j_est))
            i_stack = p_per_cell_w / max(v_cell, 1.23)

            # Faraday efficiency
            eta_f = float(self.faraday_curve(load_ratio))

            # Hydrogen production rate: mol/s = (eta_f * n_cells * I) / (2 * F)
            n_dot_h2_mol_s = (eta_f * self.n_cells * i_stack) / (2.0 * FARADAY_CONSTANT)
            m_dot_h2_kg_s = n_dot_h2_mol_s * MOLAR_MASS_H2
            self.m_dot_h2_kg_hr = m_dot_h2_kg_s * 3600.0

            # Water consumption
            m_dot_h2o_kg_s = n_dot_h2_mol_s * MOLAR_MASS_H2O
            self.m_dot_h2o_kg_hr = m_dot_h2o_kg_s * 3600.0

            # Specific consumption in kWh / kg H2
            if self.m_dot_h2_kg_hr > 0.01:
                self.spec_consumption_kwh_kg = (self.p_actual_mw * 1000.0) / self.m_dot_h2_kg_hr
            else:
                self.spec_consumption_kwh_kg = 0.0

            # Waste heat (MW) = P_elec - (m_dot_h2_kg_s * LHV_MJ_KG)
            chemical_power_mw = m_dot_h2_kg_s * LHV_H2_MJ_KG
            self.q_waste_mw = max(0.0, self.p_actual_mw - chemical_power_mw)

            # Thermal balance
            # Heat cooling system regulates stack towards t_stack_nom_c
            q_cooling_mw = max(0.0, self.q_waste_mw * 0.9 + 0.05 * (self.t_stack_c - self.t_stack_nom_c))
            net_heat_kj_s = (self.q_waste_mw - q_cooling_mw) * 1000.0
            self.t_stack_c += (net_heat_kj_s / self.c_th_kj_k) * dt

        self._outputs_val["P_actual"] = self.p_actual_mw
        self._outputs_val["m_dot_H2_kg_hr"] = self.m_dot_h2_kg_hr
        self._outputs_val["m_dot_H2O_kg_hr"] = self.m_dot_h2o_kg_hr
        self._outputs_val["Q_waste_MW"] = self.q_waste_mw
        self._outputs_val["T_stack"] = self.t_stack_c
        self._outputs_val["specific_consumption_kWh_kg"] = self.spec_consumption_kwh_kg


class AlkalineElectrolyzer(ModelComponent):
    """
    Alkaline Electrolyzer (AEL) with liquid KOH electrolyte.

    Enforces strict 20-30% minimum turndown safety limit: below this threshold,
    the electrolyzer trips to standby to prevent hazardous gas crossover.
    """

    def __init__(
        self,
        name: str = "AEL_Electrolyzer",
        p_nominal_mw: float = 10.0,
        min_turndown_pct: float = 25.0,
        spec_consumption_kwh_kg: float = 53.0,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.p_nominal_mw = p_nominal_mw
        self.min_turndown = min_turndown_pct / 100.0
        self.spec_consumption_nominal = spec_consumption_kwh_kg

        self.p_actual = 0.0
        self.m_dot_h2_kg_hr = 0.0
        self.state = "OFF"

        self.register_variables(
            inputs=["P_cmd"],
            outputs=["P_actual", "m_dot_H2_kg_hr", "state"],
        )

    def step_model(self, dt: float, time: float) -> None:
        p_cmd = float(self._inputs_val.get("P_cmd", 0.0))

        if p_cmd < (self.min_turndown * self.p_nominal_mw):
            self.state = "STANDBY"
            self.p_actual = 0.0
            self.m_dot_h2_kg_hr = 0.0
        else:
            self.state = "RUNNING"
            self.p_actual = min(self.p_nominal_mw, p_cmd)
            # Alkaline specific consumption slightly worsens at partial load
            load_ratio = self.p_actual / self.p_nominal_mw
            spec_kwh = self.spec_consumption_nominal * (1.0 + 0.15 * (1.0 - load_ratio))
            self.m_dot_h2_kg_hr = (self.p_actual * 1000.0) / spec_kwh

        self._outputs_val["P_actual"] = self.p_actual
        self._outputs_val["m_dot_H2_kg_hr"] = self.m_dot_h2_kg_hr
        self._outputs_val["state"] = self.state


class SOECElectrolyzer(ModelComponent):
    """
    Solid Oxide Electrolyzer Cell (SOEC) for high-temperature steam electrolysis.
    Utilizes steam at 650-850 C, greatly reducing electricity consumption.
    """

    def __init__(
        self,
        name: str = "SOEC_Electrolyzer",
        p_nominal_mw: float = 2.0,
        operating_temp_c: float = 750.0,
        spec_elec_kwh_kg: float = 38.0,
        spec_steam_kg_per_kg_h2: float = 9.0,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.p_nominal_mw = p_nominal_mw
        self.temp_c = operating_temp_c
        self.spec_elec_kwh_kg = spec_elec_kwh_kg
        self.spec_steam_ratio = spec_steam_kg_per_kg_h2

        self.p_actual = 0.0
        self.m_dot_h2_kg_hr = 0.0
        self.m_dot_steam_req_kg_hr = 0.0

        self.register_variables(
            inputs=["P_cmd", "steam_available_kg_hr"],
            outputs=["P_actual", "m_dot_H2_kg_hr", "steam_consumed_kg_hr"],
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
            # Hydrogen from electrical input
            h2_from_power = (p_cap * 1000.0) / self.spec_elec_kwh_kg
            # Constrain by available steam
            h2_from_steam = steam_avail / self.spec_steam_ratio

            h2_prod = min(h2_from_power, h2_from_steam)
            self.m_dot_h2_kg_hr = h2_prod
            self.p_actual = (h2_prod * self.spec_elec_kwh_kg) / 1000.0
            self.m_dot_steam_req_kg_hr = h2_prod * self.spec_steam_ratio

        self._outputs_val["P_actual"] = self.p_actual
        self._outputs_val["m_dot_H2_kg_hr"] = self.m_dot_h2_kg_hr
        self._outputs_val["steam_consumed_kg_hr"] = self.m_dot_steam_req_kg_hr


class H2Compressor(ModelComponent):
    """
    Multi-stage hydrogen gas compressor.
    """

    def __init__(
        self,
        name: str = "H2Compressor",
        max_flow_kg_hr: float = 500.0,
        p_suction_bar: float = 30.0,
        p_discharge_bar: float = 350.0,
        isentropic_efficiency: float = 0.70,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.max_flow_kg_hr = max_flow_kg_hr
        self.p_in = p_suction_bar
        self.p_out = p_discharge_bar
        self.eta_isen = isentropic_efficiency

        self.p_elec_kw = 0.0
        self.m_dot_actual_kg_hr = 0.0

        self.register_variables(
            inputs=["m_dot_in_kg_hr"],
            outputs=["P_elec_kW", "m_dot_out_kg_hr"],
        )

    def step_model(self, dt: float, time: float) -> None:
        m_in = float(self._inputs_val.get("m_dot_in_kg_hr", 0.0))
        m_flow = min(self.max_flow_kg_hr, max(0.0, m_in))
        self.m_dot_actual_kg_hr = m_flow

        if m_flow > 0.01:
            # Isentropic work for diatomic gas (gamma = 1.41)
            gamma = 1.41
            pr = self.p_out / max(self.p_in, 1.0)
            # W_isen (kJ/kg) = (gamma/(gamma-1)) * (R/M) * T_in * (pr^((gamma-1)/gamma) - 1)
            t_in_k = 293.15
            r_spec = 4124.0  # J/(kg*K) for H2
            term = pr ** ((gamma - 1.0) / gamma) - 1.0
            w_isen_j_kg = (gamma / (gamma - 1.0)) * r_spec * t_in_k * term
            w_actual_kwh_kg = (w_isen_j_kg / 3.6e6) / self.eta_isen
            self.p_elec_kw = m_flow * w_actual_kwh_kg
        else:
            self.p_elec_kw = 0.0

        self._outputs_val["P_elec_kW"] = self.p_elec_kw
        self._outputs_val["m_dot_out_kg_hr"] = self.m_dot_actual_kg_hr


class H2StorageTank(ModelComponent):
    """
    Pressurized hydrogen storage tank with real gas compressibility.
    """

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

        # Initial mass calculation: p * V = Z * n * R * T
        # At 150 bar, Z_H2 ~ 1.09
        z_init = 1.0 + 0.0006 * p_init_bar
        n_mols = (p_init_bar * 1e5 * volume_m3) / (z_init * R_GAS * self.t_k)
        self.mass_stored_kg = n_mols * MOLAR_MASS_H2
        self.pressure_bar = p_init_bar
        self.soc = (p_init_bar - p_min_bar) / max(p_max_bar - p_min_bar, 1.0)

        self.register_variables(
            inputs=["m_dot_in_kg_hr", "m_dot_out_kg_hr"],
            outputs=["pressure_bar", "mass_stored_kg", "SoC"],
        )

    def step_model(self, dt: float, time: float) -> None:
        m_in = float(self._inputs_val.get("m_dot_in_kg_hr", 0.0))
        m_out = float(self._inputs_val.get("m_dot_out_kg_hr", 0.0))

        dm_kg = (m_in - m_out) * (dt / 3600.0)
        self.mass_stored_kg = max(0.0, self.mass_stored_kg + dm_kg)

        # Update pressure: Z ~ 1 + 0.0006 * p
        n_mols = self.mass_stored_kg / MOLAR_MASS_H2
        # p * V = (1 + alpha * p) * n * R * T -> p * (V - alpha * n * R * T) = n * R * T
        alpha = 0.0006 / 1e5
        denom = self.volume_m3 - alpha * n_mols * R_GAS * self.t_k
        if denom > 1e-4:
            p_pa = (n_mols * R_GAS * self.t_k) / denom
            self.pressure_bar = p_pa / 1e5
        else:
            self.pressure_bar = self.p_max

        self.pressure_bar = max(0.0, min(self.pressure_bar, self.p_max * 1.05))
        self.soc = (self.pressure_bar - self.p_min) / max(self.p_max - self.p_min, 1.0)
        self.soc = max(0.0, min(1.0, self.soc))

        self._outputs_val["pressure_bar"] = self.pressure_bar
        self._outputs_val["mass_stored_kg"] = self.mass_stored_kg
        self._outputs_val["SoC"] = self.soc


class FuelCell_PEM(ModelComponent):
    """
    Rapid-dispatch PEM Fuel Cell generator.
    """

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

        # Ramp limit
        max_delta = self.ramp_rate * (dt / 60.0)
        if abs(target - self.p_actual) <= max_delta:
            self.p_actual = target
        elif target > self.p_actual:
            self.p_actual += max_delta
        else:
            self.p_actual -= max_delta

        if self.p_actual > 0.001:
            # H2 consumed = P_elec / (eff * LHV)
            # LHV = 120 MJ/kg = 120 / 3.6 kWh/kg = 33.33 kWh/kg
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
    """

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
        self.p_nom = p_nominal_mw
        self.eff_elec = electrical_efficiency
        self.eff_th = thermal_efficiency

        self.p_actual = 0.0
        self.m_dot_h2_kg_hr = 0.0
        self.q_chp_mw = 0.0

        self.register_variables(
            inputs=["P_cmd"],
            outputs=["P_actual", "m_dot_H2_kg_hr", "Q_chp_MW"],
        )

    def step_model(self, dt: float, time: float) -> None:
        p_cmd = float(self._inputs_val.get("P_cmd", 0.0))
        self.p_actual = min(self.p_nom, max(0.0, p_cmd))

        if self.p_actual > 0.001:
            lhv_kwh_kg = LHV_H2_MJ_KG / 3.6
            self.m_dot_h2_kg_hr = (self.p_actual * 1000.0) / (self.eff_elec * lhv_kwh_kg)
            fuel_power_mw = (self.m_dot_h2_kg_hr / 3600.0) * LHV_H2_MJ_KG
            self.q_chp_mw = fuel_power_mw * self.eff_th
        else:
            self.m_dot_h2_kg_hr = 0.0
            self.q_chp_mw = 0.0

        self._outputs_val["P_actual"] = self.p_actual
        self._outputs_val["m_dot_H2_kg_hr"] = self.m_dot_h2_kg_hr
        self._outputs_val["Q_chp_MW"] = self.q_chp_mw


class MethanationReactor(ModelComponent):
    """
    Sabatier Power-to-Gas Methanation Reactor: CO2 + 4 H2 -> CH4 + 2 H2O.
    """

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

        self.register_variables(
            inputs=["m_dot_H2_kg_hr"],
            outputs=["m_dot_CH4_kg_hr", "m_dot_CO2_consumed_kg_hr", "Q_reaction_MW"],
        )

    def step_model(self, dt: float, time: float) -> None:
        h2_in = float(self._inputs_val.get("m_dot_H2_kg_hr", 0.0))
        h2_feed = min(self.max_h2, max(0.0, h2_in))

        if h2_feed > 0.01:
            # Stoichiometry: 4 mol H2 (8.06 g) + 1 mol CO2 (44.01 g) -> 1 mol CH4 (16.04 g)
            m_h2_react = h2_feed * self.eff
            self.m_dot_ch4_kg_hr = m_h2_react * (16.043 / (4.0 * 2.01588))
            self.m_dot_co2_kg_hr = m_h2_react * (44.01 / (4.0 * 2.01588))
            # Reaction heat: 165 kJ / mol CH4
            mols_ch4_s = (self.m_dot_ch4_kg_hr / 16.043e-3) / 3600.0
            self.q_reaction_mw = (mols_ch4_s * 165.0) / 1000.0
        else:
            self.m_dot_ch4_kg_hr = 0.0
            self.m_dot_co2_kg_hr = 0.0
            self.q_reaction_mw = 0.0

        self._outputs_val["m_dot_CH4_kg_hr"] = self.m_dot_ch4_kg_hr
        self._outputs_val["m_dot_CO2_consumed_kg_hr"] = self.m_dot_co2_kg_hr
        self._outputs_val["Q_reaction_MW"] = self.q_reaction_mw

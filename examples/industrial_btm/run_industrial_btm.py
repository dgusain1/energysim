# -*- coding: utf-8 -*-
"""
Behind-The-Meter (BTM) Industrial Multi-Energy Simulation Example.

Simulates an industrial refinery / chemical complex with:
1. Solar PV Farm (PhysicsPV, 12 MWp)
2. Battery Storage (Tesla_Megapack_2XL, 3.9 MWh / 1.9 MW)
3. Combined Cycle Gas Turbine (GE_7HA_02 scaled to 40 MW for baseload/steam)
4. Green Hydrogen Facility (PEMElectrolyzer, 10 MW)
5. Pressurized H2 Storage (H2StorageTank, 350 bar)
6. Industrial Steam Header Demand (IndustrialProcessSteam, 40 t/hr)

Coupled through an energysim World orchestrator over a 24-hour cycle.
"""

import sys
import os
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from energysim import world
from energysim.models import catalog
from energysim.models.renewables import PhysicsPV
from energysim.models.hydrogen import PEMElectrolyzer, H2StorageTank
from energysim.models.thermal import CombinedCycle_CCGT
from energysim.models.loads import IndustrialProcessSteam


def run_simulation():
    print("=" * 70)
    print("energysim: Industrial BTM Multi-Energy System Co-Simulation")
    print("=" * 70)

    # 1. Initialize World: 24 hours (86,400s) with 15-minute macro exchange (900s)
    sim_world = world(
        start_time=0,
        stop_time=86400,
        t_macro=900,
        res_filename="btm_industrial_results.h5",
        logging=False,
    )

    # 2. Instantiate Physical Components
    # 12 MW Solar PV
    solar_pv = PhysicsPV(name="solar_pv", p_dc_stc_mw=12.0, inverter_p_ac_max_mw=10.0)
    # Tesla Megapack 2XL from equipment catalog
    bess = catalog.Tesla_Megapack_2XL(name="bess", soc_init=0.5)
    # 40 MW CCGT
    ccgt = CombinedCycle_CCGT(name="ccgt", p_rated_mw=40.0, min_stable_gen_pct=35.0)
    # 10 MW PEM Electrolyzer
    electrolyzer = PEMElectrolyzer(name="electrolyzer", p_nominal_mw=10.0)
    # 350-bar H2 Storage Tank (100 m3)
    h2_tank = H2StorageTank(name="h2_tank", volume_m3=100.0, p_max_bar=350.0, p_init_bar=120.0)
    # Industrial Steam Header (40 t/hr)
    steam_load = IndustrialProcessSteam(name="steam_load", nominal_demand_t_hr=40.0)

    # 3. Add Models to World
    sim_world.add_model("solar_pv", model=solar_pv, outputs=["P_ac", "T_cell"])
    sim_world.add_model("bess", model=bess, outputs=["P_actual", "SoC", "T_batt", "SOH"])
    sim_world.add_model("ccgt", model=ccgt, outputs=["P_actual", "fuel_consumption_MMBtu_hr", "CO2_emissions_t_hr"])
    sim_world.add_model("electrolyzer", model=electrolyzer, outputs=["P_actual", "m_dot_H2_kg_hr", "Q_waste_MW"])
    sim_world.add_model("h2_tank", model=h2_tank, outputs=["pressure_bar", "mass_stored_kg", "SoC"])
    sim_world.add_model("steam_load", model=steam_load, outputs=["steam_demand_t_hr", "steam_deficit_t_hr"])

    # 4. Diurnal Profiles (Solar Irradiance and Ambient Temperature)
    # Peak solar noon at t=43200s (12:00)
    def solar_profile(t):
        hr = (t / 3600.0) % 24.0
        if 6.0 <= hr <= 18.0:
            return 950.0 * np.sin(np.pi * (hr - 6.0) / 12.0)
        return 0.0

    def temp_profile(t):
        hr = (t / 3600.0) % 24.0
        return 20.0 + 10.0 * np.sin(np.pi * (hr - 9.0) / 12.0)

    # Dispatch: Follow solar with Electrolyzer, charge BESS on solar peak, dispatch CCGT at 25 MW baseload
    def h2_dispatch(t):
        hr = (t / 3600.0) % 24.0
        # Ramp electrolyzer during solar hours
        if 9.0 <= hr <= 16.0:
            return 8.0
        return 2.0  # Basal H2 generation

    def bess_dispatch(t):
        hr = (t / 3600.0) % 24.0
        if 11.0 <= hr <= 14.0:
            return 1.5   # Charge 1.5 MW on midday surplus
        elif 18.0 <= hr <= 21.0:
            return -1.5  # Discharge 1.5 MW for evening peak
        return 0.0

    sim_world.add_signal("irr_sig", signal=solar_profile)
    sim_world.add_signal("amb_temp_sig", signal=temp_profile)
    sim_world.add_signal("h2_sig", signal=h2_dispatch)
    sim_world.add_signal("bess_sig", signal=bess_dispatch)
    sim_world.add_signal("ccgt_sig", signal=lambda t: 25.0)  # 25 MW baseload

    # 5. Variable Connections
    sim_world.add_connections({
        "irr_sig.y": "solar_pv.G_poa",
        "amb_temp_sig.y": ("solar_pv.T_amb", "bess.T_amb", "ccgt.T_amb"),
        "h2_sig.y": "electrolyzer.P_cmd",
        "bess_sig.y": "bess.P_cmd",
        "ccgt_sig.y": "ccgt.P_cmd",
        # Route hydrogen produced directly into storage tank
        "electrolyzer.m_dot_H2_kg_hr": "h2_tank.m_dot_in_kg_hr",
    })

    # 6. Execute Co-Simulation
    print("\nRunning 24-hour Multi-Energy Co-Simulation...")
    sim_world.simulate(pbar=True, record_all=False)

    # 7. Analyze and Print Summary Report
    results = sim_world.results(to_csv=False)
    df_pv = results["solar_pv"]
    df_bess = results["bess"]
    df_ccgt = results["ccgt"]
    df_h2 = results["electrolyzer"]
    df_tank = results["h2_tank"]

    # At macro step (900s = 0.25h), energy = sum(P_MW) * 0.25h
    dt_hr = 900.0 / 3600.0
    total_pv_mwh = df_pv["P_ac"].sum() * dt_hr
    total_ccgt_mwh = df_ccgt["P_actual"].sum() * dt_hr
    total_h2_produced_kg = df_h2["m_dot_H2_kg_hr"].sum() * dt_hr
    total_co2_tonnes = df_ccgt["CO2_emissions_t_hr"].sum() * dt_hr

    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY REPORT (24-Hour BTM Operation)")
    print("=" * 70)
    print(f"Total Solar Energy Generated     : {total_pv_mwh:,.2f} MWh")
    print(f"Total CCGT Energy Generated      : {total_ccgt_mwh:,.2f} MWh")
    print(f"Total Green Hydrogen Produced    : {total_h2_produced_kg:,.2f} kg H2")
    print(f"H2 Storage Tank Ending Pressure  : {df_tank['pressure_bar'].iloc[-1]:,.1f} bar (started 120 bar)")
    print(f"BESS Final SoC                   : {df_bess['SoC'].iloc[-1]*100:.1f} %")
    print(f"Total CO2 Emissions from CCGT    : {total_co2_tonnes:,.2f} tonnes")
    print("=" * 70)
    print("Co-simulation completed successfully!\n")


if __name__ == "__main__":
    run_simulation()

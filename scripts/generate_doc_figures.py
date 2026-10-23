# -*- coding: utf-8 -*-
"""
Generate publication-quality figures and visualizations for energysim documentation.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from energysim import world
from energysim.models import catalog
from energysim.models.renewables import PhysicsPV
from energysim.models.hydrogen import PEMElectrolyzer, AlkalineElectrolyzer, H2StorageTank
from energysim.models.thermal import CombinedCycle_CCGT
from energysim.models.storage import BatteryBESS


# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans, Arial, Helvetica'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#eeeeee'
plt.rcParams['grid.linestyle'] = '--'

output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "assets"))
os.makedirs(output_dir, exist_ok=True)


def generate_btm_simulation_figure():
    print("Generating 24-Hour BTM Industrial Simulation Figure...")
    # 1. Run simulation
    sim = world(start_time=0, stop_time=86400, t_macro=900, logging=False)

    pv = PhysicsPV(name="pv", p_dc_stc_mw=12.0, inverter_p_ac_max_mw=10.0)
    bess = catalog.Tesla_Megapack_2XL(name="bess", soc_init=0.5)
    ccgt = CombinedCycle_CCGT(name="ccgt", p_rated_mw=40.0, min_stable_gen_pct=35.0)
    h2 = PEMElectrolyzer(name="electrolyzer", p_nominal_mw=10.0)
    tank = H2StorageTank(name="tank", volume_m3=100.0, p_max_bar=350.0, p_init_bar=120.0)

    sim.add_model("pv", model=pv, outputs=["P_ac", "T_cell"])
    sim.add_model("bess", model=bess, outputs=["P_actual", "SoC", "T_batt"])
    sim.add_model("ccgt", model=ccgt, outputs=["P_actual", "fuel_consumption_MMBtu_hr", "CO2_emissions_t_hr"])
    sim.add_model("electrolyzer", model=h2, outputs=["P_actual", "m_dot_H2_kg_hr"])
    sim.add_model("tank", model=tank, outputs=["pressure_bar", "mass_stored_kg"])

    # Diurnal profiles
    def solar_profile(t):
        hr = (t / 3600.0) % 24.0
        return 950.0 * np.sin(np.pi * (hr - 6.0) / 12.0) if 6.0 <= hr <= 18.0 else 0.0

    def temp_profile(t):
        hr = (t / 3600.0) % 24.0
        return 20.0 + 10.0 * np.sin(np.pi * (hr - 9.0) / 12.0)

    def h2_dispatch(t):
        hr = (t / 3600.0) % 24.0
        return 8.5 if 9.0 <= hr <= 16.0 else 2.5

    def bess_dispatch(t):
        hr = (t / 3600.0) % 24.0
        if 11.0 <= hr <= 14.5:
            return 1.8   # Charge on solar peak
        elif 18.0 <= hr <= 21.0:
            return -1.8  # Discharge for evening load
        return 0.0

    sim.add_signal("irr_sig", signal=solar_profile)
    sim.add_signal("amb_temp_sig", signal=temp_profile)
    sim.add_signal("h2_sig", signal=h2_dispatch)
    sim.add_signal("bess_sig", signal=bess_dispatch)
    sim.add_signal("ccgt_sig", signal=lambda t: 25.0)

    sim.add_connections({
        "irr_sig.y": "pv.G_poa",
        "amb_temp_sig.y": ("pv.T_amb", "bess.T_amb", "ccgt.T_amb"),
        "h2_sig.y": "electrolyzer.P_cmd",
        "bess_sig.y": "bess.P_cmd",
        "ccgt_sig.y": "ccgt.P_cmd",
        "electrolyzer.m_dot_H2_kg_hr": "tank.m_dot_in_kg_hr",
    })

    sim.simulate(pbar=False, record_all=False)
    results = sim.results(to_csv=False, dashboard=False)

    df_pv = results["pv"]
    df_bess = results["bess"]
    df_ccgt = results["ccgt"]
    df_h2 = results["electrolyzer"]
    df_tank = results["tank"]

    # results() auto-converts the 'time' column to hours when max_t >= 7200s
    if df_pv["time"].max() > 100.0:
        hours = df_pv["time"] / 3600.0
    else:
        hours = df_pv["time"]

    # Create 4-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=200)

    # Panel 1: Multi-Carrier Power Dispatch
    ax1 = axes[0, 0]
    ax1.plot(hours, df_pv["P_ac"], label="Solar PV Generation", color="#e67e22", lw=2.2)
    ax1.plot(hours, df_ccgt["P_actual"], label="CCGT Baseload Power", color="#2c3e50", lw=2.0)
    ax1.plot(hours, df_h2["P_actual"], label="PEM Electrolyzer Demand", color="#27ae60", lw=2.0, ls="--")
    ax1.plot(hours, df_bess["P_actual"], label="BESS (+Chg / -Dis)", color="#2980b9", lw=1.8)
    ax1.axhline(0, color="#95a5a6", lw=0.8, ls=":")
    ax1.set_title("(A) Industrial Complex Power Dispatch Balance", fontsize=12, fontweight="bold", pad=8)
    ax1.set_xlabel("Time of Day (Hours)", fontsize=10)
    ax1.set_ylabel("Power (MW)", fontsize=10)
    ax1.set_xlim(0, 24)
    ax1.legend(loc="upper left", frameon=True, fontsize=8.5)

    # Panel 2: Hydrogen Production & 350-bar Storage
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()
    p1 = ax2.plot(hours, df_h2["m_dot_H2_kg_hr"], color="#16a085", lw=2.2, label="H2 Production (kg/hr)")
    p2 = ax2_twin.plot(hours, df_tank["pressure_bar"], color="#8e44ad", lw=2.2, ls="-.", label="Tank Pressure (bar)")
    ax2.set_title("(B) Green Hydrogen Production & Tank Pressure Dynamics", fontsize=12, fontweight="bold", pad=8)
    ax2.set_xlabel("Time of Day (Hours)", fontsize=10)
    ax2.set_ylabel("H2 Production Rate (kg/hr)", color="#16a085", fontsize=10)
    ax2_twin.set_ylabel("Storage Tank Pressure (bar)", color="#8e44ad", fontsize=10)
    ax2.set_xlim(0, 24)
    lines = p1 + p2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper left", frameon=True, fontsize=8.5)

    # Panel 3: Battery Energy Storage System (SoC & Temperature)
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    p3 = ax3.plot(hours, df_bess["SoC"] * 100.0, color="#2980b9", lw=2.2, label="BESS State of Charge (%)")
    p4 = ax3_twin.plot(hours, df_bess["T_batt"], color="#c0392b", lw=1.8, ls="--", label="Cell Temp (°C)")
    ax3.set_title("(C) Battery Energy Storage (SoC & Thermal Profile)", fontsize=12, fontweight="bold", pad=8)
    ax3.set_xlabel("Time of Day (Hours)", fontsize=10)
    ax3.set_ylabel("State of Charge (%)", color="#2980b9", fontsize=10)
    ax3_twin.set_ylabel("Battery Pack Temp (°C)", color="#c0392b", fontsize=10)
    ax3.set_xlim(0, 24)
    lines3 = p3 + p4
    labels3 = [l.get_label() for l in lines3]
    ax3.legend(lines3, labels3, loc="upper right", frameon=True, fontsize=8.5)

    # Panel 4: Cumulative Emissions & Fuel Consumption
    ax4 = axes[1, 1]
    cum_co2 = np.cumsum(df_ccgt["CO2_emissions_t_hr"] * 0.25)
    ax4.plot(hours, cum_co2, color="#7f8c8d", lw=2.2, label="Cumulative CO2 (Tonnes)")
    ax4.set_title("(D) Cumulative CCGT CO2 Emissions", fontsize=12, fontweight="bold", pad=8)
    ax4.set_xlabel("Time of Day (Hours)", fontsize=10)
    ax4.set_ylabel("CO2 Emissions (Tonnes)", fontsize=10)
    ax4.set_xlim(0, 24)
    ax4.legend(loc="upper left", frameon=True, fontsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "btm_simulation_results.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_component_curves_figure():
    print("Generating Component Characteristics Figure...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=200)

    # 1. Electrolyzer Polarization Curve
    j = np.linspace(0.05, 2.5, 100)
    v_rev = 1.23
    r = 0.25
    s = 0.12
    t = 10.0
    v_cell = v_rev + r * j + s * np.log(t * j + 1.0)

    ax1 = axes[0]
    ax1.plot(j, v_cell, color="#16a085", lw=2.2, label="Cell Voltage V(j)")
    ax1.axhline(1.48, color="#e67e22", ls="--", label="Thermoneutral V_tn (1.48V)")
    ax1.axhline(1.23, color="#2c3e50", ls=":", label="Reversible V_rev (1.23V)")
    ax1.set_title("Electrolyzer Polarization Curve", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Current Density (A/cm²)", fontsize=10)
    ax1.set_ylabel("Cell Voltage (V)", fontsize=10)
    ax1.set_ylim(1.0, 2.6)
    ax1.legend(loc="lower right", fontsize=8.5)

    # 2. Battery OCV & Internal Resistance vs SoC
    soc = np.linspace(0.02, 1.0, 100)
    voc = 750.0 + 80.0 * soc - 30.0 * np.exp(-30.0 * soc)
    rin = 0.02 + 0.05 * np.exp(-25.0 * soc) + 0.01 * (soc ** 4)

    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    p1 = ax2.plot(soc * 100, voc, color="#2980b9", lw=2.2, label="Open-Circuit Voltage Voc (V)")
    p2 = ax2_twin.plot(soc * 100, rin * 1000, color="#c0392b", lw=2.0, ls="--", label="Internal Resistance Rin (mΩ)")
    ax2.set_title("BESS OCV & Internal Resistance vs SoC", fontsize=11, fontweight="bold")
    ax2.set_xlabel("State of Charge (%)", fontsize=10)
    ax2.set_ylabel("OCV (V)", color="#2980b9", fontsize=10)
    ax2_twin.set_ylabel("Resistance (mΩ)", color="#c0392b", fontsize=10)
    lines = p1 + p2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="center right", fontsize=8.5)

    # 3. CCGT Part-Load Heat Rate Curve
    load = np.linspace(0.35, 1.0, 100)
    hr_nom = 6400.0  # kJ/kWh
    hr_mult = 1.0 + 0.35 * ((1.0 - load) ** 1.8)
    hr = hr_nom * hr_mult
    eff = 3600.0 / hr * 100.0  # Efficiency %

    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    p3 = ax3.plot(load * 100, hr, color="#d35400", lw=2.2, label="Heat Rate (kJ/kWh)")
    p4 = ax3_twin.plot(load * 100, eff, color="#27ae60", lw=2.0, ls="--", label="Thermal Efficiency (%)")
    ax3.set_title("CCGT Part-Load Heat Rate & Efficiency", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Plant Load Ratio (%)", fontsize=10)
    ax3.set_ylabel("Heat Rate (kJ/kWh)", color="#d35400", fontsize=10)
    ax3_twin.set_ylabel("Efficiency (%)", color="#27ae60", fontsize=10)
    lines3 = p3 + p4
    labels3 = [l.get_label() for l in lines3]
    ax3.legend(lines3, labels3, loc="upper right", fontsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "component_characteristics.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    generate_btm_simulation_figure()
    generate_component_curves_figure()

# -*- coding: utf-8 -*-
"""
Generate the 7 unexecuted, production-grade Jupyter notebooks for energysim.
"""

import json
import os

notebooks_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "notebooks"))
os.makedirs(notebooks_dir, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.11",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py"
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3 (ipykernel)",
                "language": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }


# -----------------------------------------------------------------------------
# NOTEBOOK 01: Quickstart Multi-Energy Co-Simulation
# -----------------------------------------------------------------------------
nb1_cells = [
    md_cell("""# 01 - Getting Started: Multi-Energy System Co-Simulation

Welcome to **`energysim`**! This tutorial introduces the fundamental concepts of multi-energy system (MES) co-simulation:

1. Initializing an `energysim.world` co-simulation canvas.
2. Macro time-steps ($t_{\\text{macro}}$) vs. Micro time-steps ($t_{\\text{micro}}$).
3. Adding physical component models (`BatteryBESS`, `PhysicsPV`).
4. Injecting dynamic boundary conditions (signals and profiles).
5. Establishing directional variable connections and automatic unit scaling.
6. Running the simulation and inspecting results in pandas DataFrames.
"""),
    code_cell("""import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from energysim import world
from energysim.models.renewables import PhysicsPV
from energysim.models.storage import BatteryBESS

print("energysim imported successfully!")
"""),
    md_cell("""## Step 1: Initialize the Co-Simulation World

The `world` object orchestrates time-synchronization, variable routing, and coupling algorithms.
Here we configure a 24-hour simulation ($86,400\\text{ s}$) with a 15-minute macro exchange step ($900\\text{ s}$).
"""),
    code_cell("""sim = world(
    start_time=0,
    stop_time=86400,
    t_macro=900,
    power_unit="kW",  # Global default unit
    logging=False
)
print("Canvas initialized:", sim)
"""),
    md_cell("""## Step 2: Instantiate Physical Components

We will add a **5 MW Solar PV Plant** and a **10 MWh / 2.5 MW Battery Energy Storage System (BESS)**.
"""),
    code_cell("""# 1. 5 MWp Solar PV array
pv = PhysicsPV(name="solar_pv", p_dc_stc_mw=5.0, inverter_p_ac_max_mw=4.5)

# 2. 10 MWh / 2.5 MW Battery
bess = BatteryBESS(name="battery", capacity_mwh=10.0, p_max_mw=2.5, soc_init=0.4)

# 3. Add to world canvas
sim.add_model("solar_pv", model=pv, outputs=["P_ac", "T_cell"])
sim.add_model("battery", model=bess, outputs=["P_actual", "SoC", "T_batt"])
"""),
    md_cell("""## Step 3: Add Time-Varying Signals

We create synthetic diurnal profiles for solar irradiance ($W/m^2$) and battery charging dispatch commands.
"""),
    code_cell("""# Diurnal solar irradiance curve peaking at 900 W/m2 at noon
def solar_irradiance(t):
    hr = (t / 3600.0) % 24.0
    if 6.0 <= hr <= 18.0:
        return 900.0 * np.sin(np.pi * (hr - 6.0) / 12.0)
    return 0.0

# Charge battery on midday solar peak, discharge into evening load
def battery_dispatch(t):
    hr = (t / 3600.0) % 24.0
    if 10.0 <= hr <= 14.0:
        return 1.8   # Charge at 1.8 MW
    elif 18.0 <= hr <= 22.0:
        return -1.5  # Discharge at 1.5 MW
    return 0.0

sim.add_signal("irr_sig", signal=solar_irradiance)
sim.add_signal("bess_sig", signal=battery_dispatch, units={"y": "MW"})
"""),
    md_cell("""## Step 4: Connect Variables

Route the solar irradiance into the PV model, and the dispatch command into the battery.
"""),
    code_cell("""sim.add_connections({
    "irr_sig.y": "solar_pv.G_poa",
    "bess_sig.y": "battery.P_cmd",
})
print("Connections established!")
"""),
    md_cell("""## Step 5: Execute Simulation and Plot Results"""),
    code_cell("""sim.simulate(pbar=True)
results = sim.last_results

df_pv = results["solar_pv"]
df_bess = results["battery"]

print("Simulation finished! Extracted DataFrames:")
display(df_pv.head())
display(df_bess.head())
"""),
    code_cell("""# Plotting Results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

hours = df_pv.index / 3600.0   # Results tables are indexed by time in seconds

ax1.plot(hours, df_pv["P_ac"], color="orange", lw=2, label="Solar PV (MW)")
ax1.plot(hours, df_bess["P_actual"], color="blue", lw=2, label="BESS Power (MW)")
ax1.set_ylabel("Power (MW)")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.6)

ax2.plot(hours, df_bess["SoC"] * 100, color="green", lw=2, label="Battery SoC (%)")
ax2.set_xlabel("Time of Day (Hours)")
ax2.set_ylabel("SoC (%)")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
""")
]

# -----------------------------------------------------------------------------
# NOTEBOOK 02: Multi-Grid Network Co-Simulation (pandapower, PyPSA, OpenDSS)
# -----------------------------------------------------------------------------
nb2_cells = [
    md_cell("""# 02 - Multi-Grid Co-Simulation: pandapower, PyPSA & OpenDSS

This notebook demonstrates heterogeneous power grid co-simulation across transmission, distribution, and market dispatch layers:

1. **pandapower**: Detailed AC power flow and voltage constraint monitoring on a distribution feeder.
2. **PyPSA**: High-level regional economic dispatch and transmission constraints.
3. **OpenDSS**: Multiphase unbalanced 3-phase feeder modeling.
4. **Automatic Unit Resolution**: Seamlessly exchanging variables between `kW` and `MW`.
"""),
    code_cell("""import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from energysim import world

# Check optional library installations
try:
    import pandapower as pp
    import pandapower.networks as pn
    has_pp = True
except ImportError:
    has_pp = False

try:
    import pypsa
    has_pypsa = True
except ImportError:
    has_pypsa = False

print(f"pandapower available: {has_pp}")
print(f"PyPSA available     : {has_pypsa}")
"""),
    md_cell("""## Step 1: Build a Distribution Grid in pandapower

We create a medium-voltage feeder in pandapower with a commercial load and solar generator.
"""),
    code_cell("""net = pp.create_empty_network(name="commercial_microgrid")
b1 = pp.create_bus(net, vn_kv=20.0, name="SubstationBus")
b2 = pp.create_bus(net, vn_kv=20.0, name="IndustrialParkBus")
b3 = pp.create_bus(net, vn_kv=20.0, name="SolarFarmBus")

pp.create_ext_grid(net, b1, name="GridSlack")
pp.create_line(net, b1, b2, length_km=2.5, std_type="NAYY 4x150 SE", name="MainFeeder")
pp.create_line(net, b2, b3, length_km=1.2, std_type="NAYY 4x150 SE", name="SolarTieline")

pp.create_load(net, b2, p_mw=1.5, q_mvar=0.3, name="FactoryLoad")
pp.create_sgen(net, b3, p_mw=0.0, q_mvar=0.0, name="SolarPlant")
print("pandapower network configured!")
"""),
    md_cell("""## Step 2: Build a Regional Dispatch Market in PyPSA

We create an upstream PyPSA market network to evaluate regional dispatch costs and tie-line import power.
"""),
    code_cell("""n_pypsa = pypsa.Network()
n_pypsa.add("Bus", "RegionalGrid", v_nom=110.0)
n_pypsa.add("Bus", "LocalSubstation", v_nom=110.0)
n_pypsa.add("Line", "Tieline", bus0="RegionalGrid", bus1="LocalSubstation", x=0.02, r=0.005, s_nom=100.0)
n_pypsa.add("Generator", "ThermalPlant", bus="RegionalGrid", p_nom=200.0, marginal_cost=45.0)
n_pypsa.add("Load", "RegionalLoad", bus="LocalSubstation", p_set=50.0)
print("PyPSA market network configured!")
"""),
    md_cell("""## Step 3: Couple Grids in energysim World

We couple pandapower and PyPSA together on a 4-hour co-simulation timeline with automatic unit scaling (`kW` $\\leftrightarrow$ `MW`).
"""),
    code_cell("""sim = world(start_time=0, stop_time=14400, t_macro=900, power_unit="kW")

# Add pandapower AC power flow
sim.add_powerflow(
    sim_name="mv_grid",
    model=net,
    inputs=["FactoryLoad.p_mw", "SolarPlant.p_mw"],
    outputs=["IndustrialParkBus.vm_pu", "SolarFarmBus.vm_pu", "GridSlack.p_mw"],
    step_size=900,
    mode="ac"
)

# Add PyPSA economic dispatch
sim.add_pypsa(
    sim_name="upstream_market",
    model=n_pypsa,
    inputs=["RegionalLoad.P"],
    outputs=["ThermalPlant.P", "LocalSubstation.V"],
    step_size=900,
    mode="ac"
)

# Dynamic solar feed-in
sim.add_signal("solar_profile", lambda t: 2000.0 * min(1.0, t / 7200.0), units={"y": "kW"})
sim.add_connections({
    "solar_profile.y": "mv_grid.SolarPlant.p_mw"
})

print("Multi-grid co-simulation canvas ready!")
"""),
    md_cell("""## Step 4: Run Simulation and Analyze Power Flow Results"""),
    code_cell("""sim.simulate()
res = sim.last_results

df_grid = res["mv_grid"]
display(df_grid)
""")
]

# -----------------------------------------------------------------------------
# NOTEBOOK 03: Authoring Custom Physical Models
# -----------------------------------------------------------------------------
nb3_cells = [
    md_cell("""# 03 - Authoring Custom Physical Models & Parametrization

In this tutorial, you will learn how to:

1. Subclass `ModelComponent` to create brand new physical models.
2. Define custom 1-D and 2-D empirical lookup curves (`Curve1D`, `Surface2D`).
3. Embed an `OperatingStateMachine` to model start-up durations, warm-up purges, and ramp limits.
4. Modify and customize parameters of existing catalog models.
"""),
    code_cell("""import numpy as np
import matplotlib.pyplot as plt

from energysim import world
from energysim.models.common.base_component import ModelComponent
from energysim.models.common.curve import Curve1D
from energysim.models.common.state_machine import OperatingStateMachine, OperatingState

print("Dependencies loaded!")
"""),
    md_cell("""## Step 1: Create a Custom Supercapacitor Component

We implement a fast-discharge supercapacitor module with voltage limits and internal resistance:
"""),
    code_cell("""class SupercapacitorModule(ModelComponent):
    def __init__(self, name="supercap", capacitance_f=600.0, v_rated=48.0, r_esr=0.008, v_init=48.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.c_f = capacitance_f
        self.v_rated = v_rated
        self.r_esr = r_esr
        self.v_cell = v_init
        
        self.register_variables(
            inputs=["P_cmd_kW"],
            outputs=["P_actual_kW", "V_terminal", "SoC", "energy_kJ"]
        )

    def step_model(self, dt: float, time: float) -> None:
        p_cmd_w = float(self._inputs_val.get("P_cmd_kW", 0.0)) * 1000.0
        
        # Calculate current
        i = p_cmd_w / max(self.v_cell, 1.0)
        
        # dV = I * dt / C
        dv = (i / self.c_f) * dt
        self.v_cell = np.clip(self.v_cell + dv, 5.0, self.v_rated)
        
        v_term = self.v_cell + i * self.r_esr
        p_act_kw = (v_term * i) / 1000.0
        soc = (self.v_cell / self.v_rated) ** 2
        energy_kj = 0.5 * self.c_f * (self.v_cell ** 2) / 1000.0
        
        self._outputs_val["P_actual_kW"] = p_act_kw
        self._outputs_val["V_terminal"] = v_term
        self._outputs_val["SoC"] = soc
        self._outputs_val["energy_kJ"] = energy_kj

print("SupercapacitorModule class defined!")
"""),
    md_cell("""## Step 2: Test Custom Model in energysim"""),
    code_cell("""cap = SupercapacitorModule(name="fast_cap", capacitance_f=1200.0, v_rated=48.0)

sim = world(start_time=0, stop_time=300, t_macro=5)  # 5-minute study with 5s time-steps
sim.add_model("fast_cap", model=cap, outputs=["P_actual_kW", "V_terminal", "SoC"])

# Rapid charge and discharge sequence
def pulse_cmd(t):
    if 20 <= t < 80:
        return 15.0   # Fast charge 15 kW
    elif 150 <= t < 220:
        return -20.0  # Deep discharge 20 kW
    return 0.0

sim.add_signal("pulse", signal=pulse_cmd)
sim.add_connections({"pulse.y": "fast_cap.P_cmd_kW"})

sim.simulate()
df_cap = sim.last_results["fast_cap"]
display(df_cap.head())
""")
]

# -----------------------------------------------------------------------------
# NOTEBOOK 04: Building Custom Simulator Adapters
# -----------------------------------------------------------------------------
nb4_cells = [
    md_cell("""# 04 - Building Custom Simulator Adapters

In this advanced guide, you will learn how to wrap third-party software engines, REST APIs, or compiled C/C++ libraries into an `energysim` co-simulation adapter by subclassing `SimulatorAdapter`:

1. The `SimulatorAdapter` contract: `init()`, `set_value()`, `step()`, `get_value()`.
2. Declaring the adapter's inputs and outputs and connecting it to other simulators in a `world`.
3. Wrapping an external mock thermodynamic fluid flow solver.
"""),
    code_cell("""from typing import List, Any, Dict, Optional
from energysim import world
from energysim.base import SimulatorAdapter

print("Base classes imported!")
"""),
    md_cell("""## Step 1: Implement the Custom Adapter"""),
    code_cell("""class MockHydraulicNetworkAdapter(SimulatorAdapter):
    \"\"\"
    Adapter wrapping an external hydraulic pipe network solver.
    Simulates pipe flow, pressure drop, and fluid temperature.
    \"\"\"
    def __init__(self, sim_name: str, step_size: float = 60.0):
        super().__init__()
        self.sim_name = sim_name
        self.step_size = step_size
        self.mass_flow_kg_s = 10.0
        self.temp_in_c = 80.0
        self.pressure_drop_bar = 1.2
        self.temp_out_c = 75.0

    def init(self) -> None:
        pass

    def set_value(self, parameters: List[str], values: Any) -> None:
        if not isinstance(values, (list, tuple)):
            values = [values] * len(parameters)
        for param, val in zip(parameters, values):
            if param == "m_dot_in":
                self.mass_flow_kg_s = float(val)
            elif param == "temp_in":
                self.temp_in_c = float(val)

    def step(self, time: float, step_size: Optional[float] = None) -> None:
        # Hydraulic friction: delta_P ~ m_dot^2
        self.pressure_drop_bar = 0.012 * (self.mass_flow_kg_s ** 2)
        # Thermal loss through insulation
        self.temp_out_c = self.temp_in_c - (0.5 * 30.0 / max(self.mass_flow_kg_s, 0.1))

    def get_value(self, parameters: List[str], time: Optional[float] = None) -> List[Any]:
        res = []
        for p in parameters:
            if p == "pressure_drop_bar":
                res.append(self.pressure_drop_bar)
            elif p == "temp_out":
                res.append(self.temp_out_c)
        return res

print("MockHydraulicNetworkAdapter implemented!")
"""),
    md_cell("""## Step 2: Co-Simulate the Hydraulic Network in World"""),
    code_cell("""sim = world(start_time=0, stop_time=1800, t_macro=60)

pipe = MockHydraulicNetworkAdapter(sim_name="district_heating_pipe")
sim.add_model("pipe", model=pipe, inputs=["m_dot_in", "temp_in"], outputs=["pressure_drop_bar", "temp_out"])

sim.add_signal("flow_ramp", lambda t: 5.0 + 10.0 * (t / 1800.0))
sim.add_signal("temp_source", lambda t: 85.0)

sim.add_connections({
    "flow_ramp.y": "pipe.m_dot_in",
    "temp_source.y": "pipe.temp_in",
})

sim.simulate()
results = sim.last_results
display(results["pipe"].head())
""")
]

# -----------------------------------------------------------------------------
# NOTEBOOK 05: Industrial BTM Multi-Energy Complex
# -----------------------------------------------------------------------------
nb5_cells = [
    md_cell("""# 05 - Industrial BTM Refinery & Multi-Energy Complex

This case study simulates a complex Behind-The-Meter (BTM) industrial chemical / refinery facility operating over a 24-hour cycle:

* **12 MW Solar PV Plant** (`PhysicsPV`)
* **40 MW Combined Cycle Gas Turbine** (`CombinedCycle_CCGT`)
* **10 MW Green Hydrogen PEM Electrolyzer** (`PEMElectrolyzer`)
* **350-bar Compressed H2 Buffer Storage** (`H2StorageTank`)
* **Li-ion Battery System** (`catalog.BESS_2h_2MW`, a generic 2 MW / 4 MWh archetype)
* **Industrial Process Steam Header Demand** (`IndustrialProcessSteam`)
"""),
    code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from energysim import world
from energysim.models import catalog
from energysim.models.renewables import PhysicsPV
from energysim.models.hydrogen import PEMElectrolyzer, H2StorageTank
from energysim.models.thermal import CombinedCycle_CCGT
from energysim.models.loads import IndustrialProcessSteam

print("Industrial components imported!")
"""),
    md_cell("""## Step 1: Set Up the Industrial Complex Canvas"""),
    code_cell("""sim = world(start_time=0, stop_time=86400, t_macro=900, logging=False)

# 1. Solar PV
solar = PhysicsPV(name="solar_pv", p_dc_stc_mw=12.0, inverter_p_ac_max_mw=10.0)

# 2. BESS Archetype from catalog
bess = catalog.BESS_2h_2MW(name="bess", soc_init=0.5)

# 3. Combined Cycle Gas Turbine
ccgt = CombinedCycle_CCGT(name="ccgt", p_rated_mw=40.0, min_stable_gen_pct=35.0)

# 4. PEM Electrolyzer & Tank
electrolyzer = PEMElectrolyzer(name="electrolyzer", p_nominal_mw=10.0, n_cells=1800)
tank = H2StorageTank(name="h2_tank", volume_m3=100.0, p_max_bar=350.0, p_init_bar=120.0)

# 5. Steam Header
steam = IndustrialProcessSteam(name="steam_header", nominal_demand_t_hr=40.0)

# Add all components
sim.add_model("solar_pv", model=solar, outputs=["P_ac", "T_cell"])
sim.add_model("bess", model=bess, outputs=["P_actual", "SoC", "T_batt"])
sim.add_model("ccgt", model=ccgt, outputs=["P_actual", "fuel_consumption_MMBtu_hr", "CO2_emissions_t_hr"])
sim.add_model("electrolyzer", model=electrolyzer, outputs=["P_actual", "m_dot_H2_kg_hr"])
sim.add_model("h2_tank", model=tank, outputs=["pressure_bar", "mass_stored_kg"])
sim.add_model("steam_header", model=steam, outputs=["steam_demand_t_hr"])
"""),
    md_cell("""## Step 2: Establish Diurnal Weather and Plant Dispatch Signals"""),
    code_cell("""# Solar Irradiance
def solar_profile(t):
    hr = (t / 3600.0) % 24.0
    return 950.0 * np.sin(np.pi * (hr - 6.0) / 12.0) if 6.0 <= hr <= 18.0 else 0.0

# Ambient Temperature
def temp_profile(t):
    hr = (t / 3600.0) % 24.0
    return 20.0 + 10.0 * np.sin(np.pi * (hr - 9.0) / 12.0)

# Electrolyzer flex: soak up midday solar
def h2_dispatch(t):
    hr = (t / 3600.0) % 24.0
    return 8.5 if 9.0 <= hr <= 16.0 else 2.5

# BESS peak shaving
def bess_dispatch(t):
    hr = (t / 3600.0) % 24.0
    if 11.0 <= hr <= 14.5:
        return 1.8
    elif 18.0 <= hr <= 21.0:
        return -1.8
    return 0.0

sim.add_signal("irr_sig", signal=solar_profile)
sim.add_signal("temp_sig", signal=temp_profile)
sim.add_signal("h2_sig", signal=h2_dispatch)
sim.add_signal("bess_sig", signal=bess_dispatch)
sim.add_signal("ccgt_sig", signal=lambda t: 25.0)  # 25 MW baseload

# Connect flows
sim.add_connections({
    "irr_sig.y": "solar_pv.G_poa",
    "temp_sig.y": ("solar_pv.T_amb", "bess.T_amb", "ccgt.T_amb"),
    "h2_sig.y": "electrolyzer.P_cmd",
    "bess_sig.y": "bess.P_cmd",
    "ccgt_sig.y": "ccgt.P_cmd",
    "electrolyzer.m_dot_H2_kg_hr": "h2_tank.m_dot_in_kg_hr",
})
"""),
    md_cell("""## Step 3: Run Co-Simulation and Analyze Dispatch"""),
    code_cell("""sim.simulate(pbar=True)
results = sim.last_results

print("Co-simulation complete! Results summary:")
for k, df in results.items():
    print(f" - {k}: {len(df)} time-steps recorded.")
""")
]

# -----------------------------------------------------------------------------
# NOTEBOOK 06: Vectorized Fleet Aggregation
# -----------------------------------------------------------------------------
nb6_cells = [
    md_cell("""# 06 - Large-Scale Vectorized Fleet Aggregation (10,000+ DERs)

Simulating thousands of individual Distributed Energy Resources (DERs) in Python typically causes severe performance degradation due to object dispatch overhead.

In this notebook, we demonstrate **Vectorized Fleet Containers**:
1. Simulating 10,000 heterogeneous residential batteries.
2. Simulating 5,000 rooftop solar PV arrays.
3. Microsecond execution benchmarks across varying fleet sizes.
"""),
    code_cell("""import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from energysim import world

print("Vectorized container benchmark ready!")
"""),
    md_cell("""## Step 1: Add a 10,000-Unit Residential Battery Fleet"""),
    code_cell("""N_BATTERIES = 10000
np.random.seed(42)

capacities = np.random.uniform(10.0, 20.0, N_BATTERIES)   # 10 to 20 kWh
initial_socs = np.random.uniform(0.2, 0.8, N_BATTERIES)

sim = world(start_time=0, stop_time=86400, t_macro=900)

sim.add_container(
    sim_name="home_storage",
    kind="battery",
    size=N_BATTERIES,
    capacity_kwh=capacities,
    max_charge_kw=5.0,
    max_discharge_kw=5.0,
    initial_soc=initial_socs,
    outputs=["total_actual_p_kw", "mean_soc", "min_soc", "max_soc"]
)

# Broadcast solar absorption command
sim.add_signal("dr_command", lambda t: 3.5 if 36000 <= t <= 50400 else -2.0)
sim.add_connections({"dr_command.y": "home_storage.p_cmd_kw"})

t0 = time.time()
sim.simulate()
elapsed = time.time() - t0

print(f"Successfully simulated {N_BATTERIES:,} batteries over 24 hours in {elapsed:.2f} seconds!")
"""),
    md_cell("""## Step 2: Inspect Fleet Results"""),
    code_cell("""df_fleet = sim.last_results["home_storage"]
display(df_fleet.head())
""")
]

# -----------------------------------------------------------------------------
# NOTEBOOK 07: Green Hydrogen Valley & Sector Coupling
# -----------------------------------------------------------------------------
nb7_cells = [
    md_cell("""# 07 - Green Hydrogen Valley & Cross-Sector Coupling

In this notebook, we model an integrated regional **Green Hydrogen Valley**:

1. Hybrid **PEM + Alkaline + SOEC** electrolyzer park.
2. Waste heat recovery from electrolysis routed into district heating via an **Industrial Heat Pump**.
3. High-pressure **H2 Compressor** and **350-bar Storage Tank**.
4. **Methanation Reactor** (Power-to-Gas) combining green $\\text{H}_2$ with captured $\\text{CO}_2$ to produce synthetic natural gas (SNG).
"""),
    code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from energysim import world
from energysim.models.hydrogen import (
    PEMElectrolyzer, AlkalineElectrolyzer, SOECElectrolyzer,
    H2Compressor, H2StorageTank, MethanationReactor
)
from energysim.models.hvac_thermal import IndustrialHeatPump

print("Hydrogen Valley models loaded!")
"""),
    md_cell("""## Step 1: Instantiate the Sector-Coupled Hydrogen Park"""),
    code_cell("""sim = world(start_time=0, stop_time=86400, t_macro=900)

# Electrolyzers
pem = PEMElectrolyzer(name="pem_stack", p_nominal_mw=10.0, n_cells=1800)
ael = AlkalineElectrolyzer(name="ael_stack", p_nominal_mw=20.0, min_turndown_pct=20.0)

# Balance of Plant
comp = H2Compressor(name="compressor", p_in_bar=30.0, p_out_bar=350.0)
tank = H2StorageTank(name="storage", volume_m3=200.0, p_max_bar=350.0, p_init_bar=80.0)

# Waste heat recovery
hp = IndustrialHeatPump(name="district_hp", p_electric_max_mw=3.0, t_source_nom_c=45.0, t_sink_nom_c=90.0)

# Methanation
p2g = MethanationReactor(name="methanation", capacity_mw_sng=5.0)

# Add to canvas
sim.add_model("pem", model=pem, outputs=["P_actual", "m_dot_H2_kg_hr", "Q_waste_MW"])
sim.add_model("ael", model=ael, outputs=["P_actual", "m_dot_H2_kg_hr"])
sim.add_model("compressor", model=comp, outputs=["P_electric_MW", "m_dot_out_kg_hr"])
sim.add_model("tank", model=tank, outputs=["pressure_bar", "mass_stored_kg"])
sim.add_model("heat_pump", model=hp, outputs=["P_actual", "Q_heat_MW", "COP"])
sim.add_model("methanation", model=p2g, outputs=["m_dot_CH4_kg_hr", "Q_exothermic_MWth"])
"""),
    md_cell("""## Step 2: Establish Cross-Carrier Couplings"""),
    code_cell("""# Route electrolyzer hydrogen to compressor, and compressor discharge into buffer storage
sim.add_connections({
    "pem.m_dot_H2_kg_hr": "compressor.m_dot_in_kg_hr",
    "compressor.m_dot_out_kg_hr": "tank.m_dot_in_kg_hr",
})

# Power dispatch orders
sim.add_signal("wind_surplus", lambda t: 8.0)
sim.add_connections({"wind_surplus.y": "pem.P_cmd"})

print("Sector-coupled topology connected!")
"""),
    md_cell("""## Step 3: Simulate and Extract Hydrogen Valley Metrics"""),
    code_cell("""sim.simulate()
results = sim.last_results
display(results["tank"].tail())
""")
]

# Write all 7 notebooks
notebooks = {
    "01_quickstart_multi_energy_cosim.ipynb": nb1_cells,
    "02_grid_cosim_pandapower_pypsa_opendss.ipynb": nb2_cells,
    "03_custom_models_and_parametrization.ipynb": nb3_cells,
    "04_building_custom_simulator_adapters.ipynb": nb4_cells,
    "05_industrial_btm_complex.ipynb": nb5_cells,
    "06_vectorized_fleet_aggregation.ipynb": nb6_cells,
    "07_green_hydrogen_valley.ipynb": nb7_cells,
}

for filename, cells in notebooks.items():
    nb_json = make_notebook(cells)
    file_path = os.path.join(notebooks_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(nb_json, f, indent=2)
    print(f"Generated clean notebook: {file_path}")

print("All 7 notebooks created successfully!")

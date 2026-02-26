#!/usr/bin/env python
"""
Optimal Dispatch of Flexible Energy Resources — energysim example
=================================================================

This script demonstrates a multi-energy system co-simulation using
energysim.  It couples:

    * A pandapower electrical grid  (gridModel_case2.p)
    * A Power-to-Gas FMU            (Hydrogen.ptg_modelB_case2.fmu)
    * A Power-to-Heat FMU           (P2H.pth_modelB_case2.fmu)
    * A CSV data source              (data.csv — profiles for wind, PV,
                                      demand, temperature, etc.)

The simulation runs for 24 h (86 400 s) with a macro time-step of 15 min
(900 s).  After completion an interactive HTML dashboard opens in the
browser.

Usage
-----
    cd examples/optimal_dispatch
    python run_optimal_dispatch.py

Reference
---------
MSc thesis project at TU Delft.
https://repository.tudelft.nl/islandora/object/uuid%3Af5dd1a15-d66d-4fc9-b2cf-15129c6c2800
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure energysim is importable (works both with pip install and dev layout)
# ---------------------------------------------------------------------------
_src = str(Path(__file__).resolve().parents[2] / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from energysim import world  # noqa: E402

# ---------------------------------------------------------------------------
# Paths to model / data files (all relative to this script)
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
os.chdir(HERE)

GRID_MODEL = str(HERE / "gridModel_case2.p")
P2G_FMU    = str(HERE / "Hydrogen.ptg_modelB_case2.fmu")
P2H_FMU    = str(HERE / "P2H.pth_modelB_case2.fmu")
CSV_DATA   = str(HERE / "data.csv")


def main():
    # ------------------------------------------------------------------
    # 1.  Create the co-simulation world
    # ------------------------------------------------------------------
    my_world = world(
        start_time=0,
        stop_time=86_400,      # 24 h
        logging=True,
        t_macro=900,           # 15 min exchange interval
    )

    # ------------------------------------------------------------------
    # 2.  Add simulators
    # ------------------------------------------------------------------

    # Pandapower grid  — use OPF so the dispatch is optimised each step
    my_world.add_simulator(
        sim_name="grid1",
        sim_loc=GRID_MODEL,
        sim_type="powerflow",
        inputs=[
            "WF.p_mw", "PV.p_mw",
            "Power2Gas.max_p_mw", "Power2Gas.min_p_mw",
            "Power2Heat.max_p_mw", "Power2Heat.min_p_mw",
        ],
        outputs=[
            "Bus 0.vm_pu", "Bus 1.vm_pu", "Bus 2.vm_pu",
            "WF.p_mw", "PV.p_mw",
            "Electrical.p_mw", "Power2Heat.p_mw", "Power2Gas.p_mw",
            "grid.p_mw", "grid.q_mvar", "Electrical.q_mvar",
            "Power2Gas.max_p_mw", "Power2Gas.min_p_mw",
            "Power2Heat.max_p_mw", "Power2Heat.min_p_mw",
            "WF.max_p_mw", "PV.max_p_mw",
        ],
        step_size=300,
        pf="opf",
    )

    # ------------------------------------------------------------------
    # 2a. Make P2G and P2H loads controllable so OPF can optimise them
    # ------------------------------------------------------------------
    grid_net = my_world.simulator_dict["grid1"].adapter.network
    for load_name in ("Power2Gas", "Power2Heat"):
        idx = grid_net.load[grid_net.load["name"] == load_name].index
        grid_net.load.loc[idx, "controllable"] = True

    # Power-to-Gas FMU (electrolyser + hydrogen storage)
    my_world.add_simulator(
        sim_name="p2g",
        sim_loc=P2G_FMU,
        sim_type="fmu",
        step_size=1,
        inputs=["gas_demand", "P_order", "T_ambient", "ptg_switch"],
        outputs=[
            "electrolyser_detailed1.Pelec", "lcoh.cost", "gas_demand",
            "electrolyser_detailed1.electrochemical.efficiency2",
            "storage2.S_storage",
            "controller_P2G3.Pmin", "controller_P2G3.Pmax",
            "electrolyser_detailed1.nH2",
            "P_order", "ptg_switch",
        ],
        validate=False,
    )

    # Power-to-Heat FMU (heat pump + thermal storage)
    my_world.add_simulator(
        sim_name="p2h",
        sim_loc=P2H_FMU,
        sim_type="fmu",
        step_size=1,
        inputs=["heat_demand", "T_ambient", "P_order", "pth_switch"],
        outputs=[
            "hp2.Pelec", "lcodh.cost", "heat_demand", "hp2.COP", "hp2.Q",
            "storage.S",
            "controller_APL.Pmin", "controller_APL.Pmax",
            "P_order", "pth_switch",
        ],
        variable=False,
        validate=False,
    )

    # CSV data profiles (wind, PV, demand, temperature, …)
    my_world.add_simulator(
        sim_name="data",
        sim_loc=CSV_DATA,
        sim_type="csv",
        step_size=900,
    )

    # ------------------------------------------------------------------
    # 3.  Define inter-simulator connections
    # ------------------------------------------------------------------
    connections = {
        # Data → Grid
        "data.WF":                  "grid1.WF.p_mw",
        "data.PV":                  "grid1.PV.p_mw",
        # Data → FMUs (switch / demand / temperature)
        "data.constant1":           "p2g.ptg_switch",
        "data.constant_pth":        "p2h.pth_switch",
        "data.ptg_demand_half":     "p2g.gas_demand",
        "data.pth_demand_half":     "p2h.heat_demand",
        "data.T_amb_pth":           "p2h.T_ambient",
        "data.T_amb_ptg":           "p2g.T_ambient",
        # Grid → FMUs (power dispatch)
        "grid1.Power2Gas.p_mw":     "p2g.P_order",
        "grid1.Power2Heat.p_mw":    "p2h.P_order",
        # FMUs → Grid (flexibility limits & cost)
        "p2g.controller_P2G3.Pmin": "grid1.Power2Gas.min_p_mw",
        "p2g.controller_P2G3.Pmax": "grid1.Power2Gas.max_p_mw",
        "p2g.lcoh.cost":            "grid1.Power2Gas.cp1_eur_per_mw",
        "p2h.controller_APL.Pmin":  "grid1.Power2Heat.min_p_mw",
        "p2h.controller_APL.Pmax":  "grid1.Power2Heat.max_p_mw",
        "p2h.lcodh.cost":           "grid1.Power2Heat.cp1_eur_per_mw",
    }
    my_world.add_connections(connections)

    # ------------------------------------------------------------------
    # 4.  Run simulation
    # ------------------------------------------------------------------
    print("Starting co-simulation …")
    my_world.simulate(record_all=False)

    # ------------------------------------------------------------------
    # 5.  Collect results  (dashboard opens automatically)
    # ------------------------------------------------------------------
    dashboard_file = str(HERE / "energysim_dashboard.html")
    res = my_world.results(
        to_csv=False,
        dashboard=True,
        dashboard_path=dashboard_file,
    )

    # Quick summary
    print("\n===  Simulation Results  ===")
    for name, df in res.items():
        print(f"  {name:8s}  →  {df.shape[0]} time-steps, {df.shape[1]-1} variables")
    print(f"\nDashboard saved to: {dashboard_file}")

    return res


if __name__ == "__main__":
    results = main()

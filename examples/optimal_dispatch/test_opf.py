"""Quick OPF validation — runs the simulation and prints key result summaries."""
import sys, os
from pathlib import Path

_src = str(Path(__file__).resolve().parents[2] / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from energysim import world

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

GRID_MODEL = str(HERE / "gridModel_case2.p")
P2G_FMU    = str(HERE / "Hydrogen.ptg_modelB_case2.fmu")
P2H_FMU    = str(HERE / "P2H.pth_modelB_case2.fmu")
CSV_DATA   = str(HERE / "data.csv")

my_world = world(start_time=0, stop_time=86_400, logging=True, t_macro=900)

my_world.add_simulator(
    sim_name="grid1", sim_loc=GRID_MODEL, sim_type="powerflow",
    inputs=["WF.p_mw","PV.p_mw","Power2Gas.max_p_mw","Power2Gas.min_p_mw",
            "Power2Heat.max_p_mw","Power2Heat.min_p_mw"],
    outputs=["Power2Gas.p_mw","Power2Heat.p_mw","grid.p_mw","WF.p_mw","PV.p_mw",
             "Power2Gas.max_p_mw","Power2Gas.min_p_mw",
             "Power2Heat.max_p_mw","Power2Heat.min_p_mw"],
    step_size=300, pf="opf",
)
# Make P2G/P2H controllable for OPF
net = my_world.simulator_dict["grid1"].adapter.network
for ln in ("Power2Gas", "Power2Heat"):
    idx = net.load[net.load["name"] == ln].index
    net.load.loc[idx, "controllable"] = True

my_world.add_simulator(sim_name="p2g", sim_loc=P2G_FMU, sim_type="fmu", step_size=1,
    inputs=["gas_demand","P_order","T_ambient","ptg_switch"],
    outputs=["controller_P2G3.Pmin","controller_P2G3.Pmax","P_order","ptg_switch"],
    validate=False)

my_world.add_simulator(sim_name="p2h", sim_loc=P2H_FMU, sim_type="fmu", step_size=1,
    inputs=["heat_demand","T_ambient","P_order","pth_switch"],
    outputs=["controller_APL.Pmin","controller_APL.Pmax","P_order","pth_switch"],
    variable=False, validate=False)

my_world.add_simulator(sim_name="data", sim_loc=CSV_DATA, sim_type="csv", step_size=900)

connections = {
    "data.WF":"grid1.WF.p_mw", "data.PV":"grid1.PV.p_mw",
    "data.constant1":"p2g.ptg_switch", "data.constant_pth":"p2h.pth_switch",
    "data.ptg_demand_half":"p2g.gas_demand", "data.pth_demand_half":"p2h.heat_demand",
    "data.T_amb_pth":"p2h.T_ambient", "data.T_amb_ptg":"p2g.T_ambient",
    "grid1.Power2Gas.p_mw":"p2g.P_order", "grid1.Power2Heat.p_mw":"p2h.P_order",
    "p2g.controller_P2G3.Pmin":"grid1.Power2Gas.min_p_mw",
    "p2g.controller_P2G3.Pmax":"grid1.Power2Gas.max_p_mw",
    "p2h.controller_APL.Pmin":"grid1.Power2Heat.min_p_mw",
    "p2h.controller_APL.Pmax":"grid1.Power2Heat.max_p_mw",
    "p2g.lcoh.cost":"grid1.Power2Gas.cp1_eur_per_mw",
    "p2h.lcodh.cost":"grid1.Power2Heat.cp1_eur_per_mw",
}
my_world.add_connections(connections)

print("Running…")
my_world.simulate(pbar=True, record_all=False)

res = my_world.results(to_csv=False, dashboard=False)

print("\n=== GRID RESULTS (first 20 + last 5 rows) ===")
g = res["grid1"]
print(g.head(20).to_string())
print("...")
print(g.tail(5).to_string())

print("\n=== P2G RESULTS (first 10 + last 5 rows) ===")
print(res["p2g"].head(10).to_string())
print("...")
print(res["p2g"].tail(5).to_string())

print("\n=== P2H RESULTS (first 10 + last 5 rows) ===")
print(res["p2h"].head(10).to_string())
print("...")
print(res["p2h"].tail(5).to_string())

# Quick sanity: are Power2Gas/Power2Heat varying?
import numpy as np
p2g_vals = g["Power2Gas.p_mw"].values
p2h_vals = g["Power2Heat.p_mw"].values
grid_vals = g["grid.p_mw"].values

print(f"\n=== SANITY CHECK ===")
print(f"Power2Gas.p_mw  — min={np.min(p2g_vals):.3f}  max={np.max(p2g_vals):.3f}  std={np.std(p2g_vals):.3f}")
print(f"Power2Heat.p_mw — min={np.min(p2h_vals):.3f}  max={np.max(p2h_vals):.3f}  std={np.std(p2h_vals):.3f}")
print(f"grid.p_mw       — min={np.min(grid_vals):.3f}  max={np.max(grid_vals):.3f}  std={np.std(grid_vals):.3f}")

if np.std(p2g_vals) > 0.01 and np.std(p2h_vals) > 0.01:
    print("\n✓ Outputs are VARYING — OPF feedback loop is working!")
else:
    print("\n✗ WARNING: outputs are still (nearly) constant — check logic.")

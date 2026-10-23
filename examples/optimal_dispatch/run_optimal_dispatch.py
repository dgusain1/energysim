#!/usr/bin/env python
"""
Optimal Dispatch of Flexible Energy Resources — energysim example
=================================================================

A multi-energy system co-simulation in which an **optimal power flow
controller** dispatches two flexible loads every 15 minutes:

    * ``data``  — CSV profiles (wind, PV, flexibility limits, bid prices)
    * ``opf``   — :class:`energysim.control.OPFController`: pandapower OPF on
                  a copy of the grid model; outputs the optimal set-points
    * ``grid1`` — the pandapower grid (plant) running an AC power flow with
                  the dispatched set-points
    * ``p2g``   — Power-to-Gas: :class:`energysim.models.AlkalineElectrolyzer`
    * ``p2h``   — Power-to-Heat: :class:`energysim.models.ElectricBoiler`

In energysim < 3.0 the OPF was a power-flow *mode* (``pf='opf'``); in 3.0 it
is a controller, i.e. an ordinary simulator whose outputs are connected to
the plant inputs.  The original thesis study used Modelica FMUs for P2G and
P2H (see ``examples/Optimal Dispatch of FER``); here they are replaced by
energysim's built-in component models so the example runs without FMUs.

The script runs headless (no browser) for 24 h with a 15 min macro step.

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
import warnings
from pathlib import Path

_src = str(Path(__file__).resolve().parents[2] / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from energysim import world  # noqa: E402
from energysim.control import OPFController  # noqa: E402
from energysim.models import AlkalineElectrolyzer, ElectricBoiler  # noqa: E402

HERE = Path(__file__).resolve().parent
GRID_MODEL = str(HERE / "gridModel_case2.p")
CSV_DATA = str(HERE / "data.csv")

FLEX_LOADS = ("Power2Gas", "Power2Heat")


def build_world(stop_time=86_400, t_macro=900, res_filename=None):
    kwargs = dict(start_time=0, stop_time=stop_time, t_macro=t_macro, power_unit="MW")
    if res_filename is not None:
        kwargs["res_filename"] = res_filename
    w = world(**kwargs)

    # Profiles: wind / PV infeed, flexibility limits and bid prices (EUR/MWh).
    w.add_csv("data", CSV_DATA, step_size=t_macro,
              outputs=["WF", "PV", "ptg_pmin", "ptg_pmax", "ptg_cost", "pth_pmin", "pth_pmax", "pth_cost"])

    # OPF controller: inputs are the profiles, outputs the optimal dispatch.
    opf = OPFController(
        GRID_MODEL,
        inputs=["WF.p_mw", "PV.p_mw",
                "Power2Gas.min_p_mw", "Power2Gas.max_p_mw", "Power2Gas.cp1_eur_per_mw",
                "Power2Heat.min_p_mw", "Power2Heat.max_p_mw", "Power2Heat.cp1_eur_per_mw"],
        outputs=["Power2Gas.p_mw", "Power2Heat.p_mw", "grid.p_mw", "cost"],
        controllable=FLEX_LOADS,
        name="opf",
    )
    w.add_model("opf", opf, step_size=t_macro)

    # Plant: the real grid runs a plain AC power flow with the dispatched set-points.
    w.add_powerflow(
        "grid1", net_loc=GRID_MODEL,
        inputs=["WF.p_mw", "PV.p_mw", "Power2Gas.p_mw", "Power2Heat.p_mw"],
        outputs=["Bus 1.vm_pu", "Bus 2.vm_pu", "grid.p_mw", "Power2Gas.p_mw", "Power2Heat.p_mw"],
        step_size=t_macro,
    )

    # Flexible devices follow the OPF set-points.
    # (the electrolyser's string 'state' output is not recorded: the HDF recorder stores numbers only)
    w.add_model("p2g", AlkalineElectrolyzer(name="p2g", p_nominal_mw=50.0, min_turndown_pct=10.0), step_size=60,
                outputs=["P_actual", "m_dot_H2_kg_hr"])
    w.add_model("p2h", ElectricBoiler(name="p2h", p_max_mw=50.0), step_size=60)

    w.add_connections({
        "data.WF": ["opf.WF.p_mw", "grid1.WF.p_mw"],
        "data.PV": ["opf.PV.p_mw", "grid1.PV.p_mw"],
        "data.ptg_pmin": "opf.Power2Gas.min_p_mw",
        "data.ptg_pmax": "opf.Power2Gas.max_p_mw",
        "data.ptg_cost": "opf.Power2Gas.cp1_eur_per_mw",
        "data.pth_pmin": "opf.Power2Heat.min_p_mw",
        "data.pth_pmax": "opf.Power2Heat.max_p_mw",
        "data.pth_cost": "opf.Power2Heat.cp1_eur_per_mw",
        "opf.Power2Gas.p_mw": ["grid1.Power2Gas.p_mw", "p2g.P_cmd"],
        "opf.Power2Heat.p_mw": ["grid1.Power2Heat.p_mw", "p2h.P_cmd"],
    })
    return w, opf


def _collect_results(w, sim_return):
    """Return the run's results for both the 3.0 Results API and the legacy dict API.

    REVIEW AFTER RESULTS-API MERGE: once ``world.simulate()`` returns a Results
    object (``res['sim.var']``, ``res.to_frame()``) the two fallbacks below can go.
    """
    if hasattr(sim_return, "to_frame"):
        return sim_return
    res = getattr(w, "last_results", None)
    if res is not None:
        return res
    return w.results(dashboard=False)


def summarize(res, opf):
    print("\n===  Optimal dispatch summary  ===")
    if hasattr(res, "to_frame"):
        df = res.to_frame()
        print(f"  {df.shape[0]} time steps, {df.shape[1]} recorded variables")
    else:  # legacy dict {sim_name: DataFrame}
        for name, df in res.items():
            print(f"  {name:6s} -> {df.shape[0]} time-steps, {df.shape[1] - 1} variables")
    hist = opf.history
    ok = sum(1 for h in hist if h["solver_status"] == 0)
    print(f"  OPF solves: {len(hist)} ({ok} optimal), mean solve time "
          f"{sum(h['solve_time_s'] for h in hist) / max(len(hist), 1):.3f} s")
    p2g = [h["setpoints"]["Power2Gas.p_mw"] for h in hist]
    p2h = [h["setpoints"]["Power2Heat.p_mw"] for h in hist]
    print(f"  P2G dispatch: {min(p2g):.1f} .. {max(p2g):.1f} MW, P2H dispatch: {min(p2h):.1f} .. {max(p2h):.1f} MW")


def main(stop_time=86_400):
    os.chdir(HERE)
    w, opf = build_world(stop_time=stop_time)
    print("Starting co-simulation ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # algebraic-loop / unit notices
        ret = w.simulate(pbar=False)
    res = _collect_results(w, ret)
    summarize(res, opf)
    return res


if __name__ == "__main__":
    results = main()

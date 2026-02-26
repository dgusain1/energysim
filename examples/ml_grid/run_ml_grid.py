#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Distribution grid with ML-based load/generation forecasters.

Co-simulates four neural-network predictors (residential load, commercial load,
PV output, wind output) with a pandapower distribution grid via energysim.

  Feature data     CSV    900 s   weather + time-of-day features
  NN predictors    ext     60 s   small feedforward NNs (pure numpy)
  Power grid       PF     300 s   AC power flow via pandapower

24-hour weekday simulation with 900 s macro time-step.
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure energysim is importable from the dev source tree
# ---------------------------------------------------------------------------
_src = str(Path(__file__).resolve().parents[2] / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
os.chdir(HERE)

from energysim import world  # noqa: E402

# ---------------------------------------------------------------------------
# Ensure prerequisite data and trained models exist
# ---------------------------------------------------------------------------
if not (HERE / "features.csv").exists() or not (HERE / "distribution_grid.p").exists():
    import generate_data
    generate_data.main()

if not (HERE / "residential_load_weights.npz").exists():
    import train_models
    train_models.main()

SIM_DIR = str(HERE / "simulators")


def main():
    my_world = world(start_time=0, stop_time=86400, logging=True, t_macro=900)

    # ── Feature data  (time + weather, 15-min resolution) ────────
    my_world.add_simulator(
        sim_type="csv",
        sim_name="features",
        sim_loc=str(HERE / "features.csv"),
        outputs=["hour_sin", "hour_cos", "temperature",
                 "cloud_cover", "wind_speed", "is_weekend"],
        step_size=900,
    )

    # ── NN-based load / generation forecasters  (60 s step) ──────
    my_world.add_simulator(
        sim_type="external", sim_name="nn_residential",
        sim_loc=SIM_DIR,
        inputs=["hour_sin", "hour_cos", "temperature", "is_weekend"],
        outputs=["P_load"],
        step_size=60,
    )

    my_world.add_simulator(
        sim_type="external", sim_name="nn_commercial",
        sim_loc=SIM_DIR,
        inputs=["hour_sin", "hour_cos", "temperature", "is_weekend"],
        outputs=["P_load"],
        step_size=60,
    )

    my_world.add_simulator(
        sim_type="external", sim_name="nn_pv",
        sim_loc=SIM_DIR,
        inputs=["hour_sin", "hour_cos", "cloud_cover", "temperature"],
        outputs=["P_gen"],
        step_size=60,
    )

    my_world.add_simulator(
        sim_type="external", sim_name="nn_wind",
        sim_loc=SIM_DIR,
        inputs=["hour_sin", "hour_cos", "wind_speed", "temperature"],
        outputs=["P_gen"],
        step_size=60,
    )

    # ── Pandapower distribution grid  (AC PF every 5 min) ────────
    my_world.add_simulator(
        sim_type="powerflow", sim_name="grid",
        sim_loc=str(HERE / "distribution_grid.p"),
        inputs=[
            "Residential.p_mw", "Commercial.p_mw",
            "SolarPark.p_mw", "WindFarm.p_mw",
        ],
        outputs=[
            "HV_Bus.vm_pu", "MV_Bus1.vm_pu", "MV_Bus2.vm_pu",
            "MV_Bus3.vm_pu", "MV_Bus4.vm_pu",
            "Residential.p_mw", "Commercial.p_mw",
            "SolarPark.p_mw", "WindFarm.p_mw",
        ],
        step_size=300,
        pf="pf",
    )

    # ── Connections ───────────────────────────────────────────────
    connections = {
        # Features → all NN models
        "features.hour_sin": (
            "nn_residential.hour_sin", "nn_commercial.hour_sin",
            "nn_pv.hour_sin", "nn_wind.hour_sin",
        ),
        "features.hour_cos": (
            "nn_residential.hour_cos", "nn_commercial.hour_cos",
            "nn_pv.hour_cos", "nn_wind.hour_cos",
        ),
        "features.temperature": (
            "nn_residential.temperature", "nn_commercial.temperature",
            "nn_pv.temperature", "nn_wind.temperature",
        ),
        "features.is_weekend": (
            "nn_residential.is_weekend", "nn_commercial.is_weekend",
        ),
        "features.cloud_cover": "nn_pv.cloud_cover",
        "features.wind_speed": "nn_wind.wind_speed",

        # NN outputs → grid loads / sgens
        "nn_residential.P_load": "grid.Residential.p_mw",
        "nn_commercial.P_load":  "grid.Commercial.p_mw",
        "nn_pv.P_gen":           "grid.SolarPark.p_mw",
        "nn_wind.P_gen":         "grid.WindFarm.p_mw",
    }
    my_world.add_connections(connections)

    print("Starting ML grid co-simulation ...")
    my_world.simulate(pbar=True, record_all=False)

    results = my_world.results(
        to_csv=False,
        dashboard=True,
        dashboard_path=str(HERE / "ml_grid_dashboard.html"),
    )

    print("\n=== Simulation Complete ===")
    for name, df in results.items():
        print(f"  {name:15s} -> {df.shape[0]} steps, {df.shape[1] - 1} variables")

    return results


if __name__ == "__main__":
    results = main()

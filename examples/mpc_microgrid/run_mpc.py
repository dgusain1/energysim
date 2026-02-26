#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MPC-optimised residential microgrid co-simulation.

Demonstrates Model Predictive Control across **all five** energysim
simulator types:

  +-----------------+------------------+--------------------------------+
  | Component       | sim_type         | Description                    |
  +=================+==================+================================+
  | Weather / load  | csv              | T_ambient, solar, PV, load,    |
  |                 |                  | electricity price              |
  +-----------------+------------------+--------------------------------+
  | Temperature     | signal           | Time-varying comfort setpoint  |
  | setpoint        |                  | (night setback 17 °C → 21 °C)  |
  +-----------------+------------------+--------------------------------+
  | Battery (13.5   | matlab (Octave)  | Li-ion battery with persistent |
  | kWh)            |                  | SoC — battery_mpc.m            |
  +-----------------+------------------+--------------------------------+
  | Building        | external (Python)| Lumped-capacitance RC model     |
  |                 |                  | with solar / internal gains    |
  +-----------------+------------------+--------------------------------+
  | MPC controller  | external (Python)| Receding-horizon optimisation  |
  |                 |                  | (scipy SLSQP, 4 h horizon)    |
  +-----------------+------------------+--------------------------------+
  | LV grid         | powerflow        | 5-bus pandapower network       |
  +-----------------+------------------+--------------------------------+

Run
---
    python run_mpc.py

Requirements: Python ≥ 3.9, energysim, pandapower, scipy, oct2py + Octave
"""

import os
import sys
import math
from pathlib import Path

import numpy as np

# --- Ensure local energysim is importable ---
_src = str(Path(__file__).resolve().parents[2] / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from energysim import world  # noqa: E402

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

# ====================================================================
# 0.  Generate data files if missing
# ====================================================================

if not (HERE / "microgrid.p").exists() or not (HERE / "weather.csv").exists():
    print("Generating pandapower network and weather CSV ...")
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import generate_data
    generate_data.main()

SIM_DIR = str(HERE / "simulators")
MODELS  = HERE / "models"

# ====================================================================
# 1.  Comfort setpoint signal  (night setback)
# ====================================================================


def comfort_setpoint(time):
    """Comfort temperature setpoint with night setback.

    21 °C during 07:00 – 22:00, 17 °C overnight.
    Smooth ramp over 1 hour at the transitions.
    """
    hour = (time % 86400) / 3600.0
    if hour < 6.0:
        T = 17.0
    elif hour < 7.0:
        T = 17.0 + 4.0 * (hour - 6.0)      # ramp up
    elif hour < 21.0:
        T = 21.0
    elif hour < 22.0:
        T = 21.0 - 4.0 * (hour - 21.0)     # ramp down
    else:
        T = 17.0
    return [T]


# ====================================================================
# 2.  Build co-simulation world
# ====================================================================

def main():
    N_STEPS  = 96
    DT_MACRO = 900.0   # 15 minutes

    my_world = world(
        start_time=0,
        stop_time=N_STEPS * DT_MACRO,
        logging=True,
        t_macro=DT_MACRO,
    )

    # ── CSV: weather + load + price profiles ────────────────────
    my_world.add_simulator(
        sim_type='csv',
        sim_name='weather',
        sim_loc=str(HERE / 'weather.csv'),
        outputs=['T_ambient', 'solar', 'P_pv', 'P_load', 'elec_price'],
        step_size=DT_MACRO,
    )

    # ── Signal: comfort temperature setpoint ────────────────────
    my_world.add_signal(
        sim_name='setpoint',
        signal=comfort_setpoint,
        step_size=DT_MACRO,
    )

    # ── MATLAB/Octave: battery (persistent SoC) ────────────────
    my_world.add_simulator(
        sim_type='matlab',
        sim_name='battery_mpc',
        sim_loc=str(MODELS / 'battery_mpc.m'),
        inputs=['P_cmd'],
        outputs=['SoC', 'P_actual', 'E_available'],
        step_size=DT_MACRO,
        engine='auto',
    )

    # ── Python external: building thermal model ─────────────────
    my_world.add_simulator(
        sim_type='external',
        sim_name='building',
        sim_loc=SIM_DIR,
        inputs=['P_hp', 'T_ambient', 'solar'],
        outputs=['T_inside'],
        step_size=60,
    )

    # ── Python external: MPC controller ─────────────────────────
    my_world.add_simulator(
        sim_type='external',
        sim_name='mpc_controller',
        sim_loc=SIM_DIR,
        inputs=['SoC', 'T_inside', 'T_ambient', 'solar',
                'P_pv', 'P_load', 'elec_price', 'T_setpoint'],
        outputs=['P_batt_cmd', 'P_hp_cmd'],
        step_size=DT_MACRO,
    )

    # ── Pandapower: LV grid ─────────────────────────────────────
    my_world.add_simulator(
        sim_type='powerflow',
        sim_name='grid',
        sim_loc=str(HERE / 'microgrid.p'),
        inputs=['PV.p_mw', 'Battery.p_mw', 'HeatPump.p_mw', 'House.p_mw'],
        outputs=['Slack.vm_pu', 'BusPV.vm_pu', 'BusBatt.vm_pu',
                 'BusHP.vm_pu', 'BusLoad.vm_pu'],
        step_size=DT_MACRO,
        pf='pf',
    )

    # ================================================================
    # 3.  Connections
    # ================================================================
    connections = {
        # Weather → building, controller, grid
        'weather.T_ambient':  ('building.T_ambient',
                               'mpc_controller.T_ambient'),
        'weather.solar':      ('building.solar',
                               'mpc_controller.solar'),
        'weather.P_pv':       ('grid.PV.p_mw',
                               'mpc_controller.P_pv'),
        'weather.P_load':     ('grid.House.p_mw',
                               'mpc_controller.P_load'),
        'weather.elec_price': 'mpc_controller.elec_price',

        # Signal → controller
        'setpoint.y':         'mpc_controller.T_setpoint',

        # Controller → actuators
        'mpc_controller.P_batt_cmd': 'battery_mpc.P_cmd',
        'mpc_controller.P_hp_cmd':   ('building.P_hp',
                                      'grid.HeatPump.p_mw'),

        # Battery → grid + controller feedback
        'battery_mpc.P_actual':      'grid.Battery.p_mw',
        'battery_mpc.SoC':           'mpc_controller.SoC',

        # Building → controller feedback
        'building.T_inside':         'mpc_controller.T_inside',
    }
    my_world.add_connections(connections)

    # ================================================================
    # 4.  Run
    # ================================================================
    print("\n── Starting MPC microgrid co-simulation ──")
    my_world.simulate(pbar=True, record_all=False)

    # ================================================================
    # 5.  Results
    # ================================================================
    results = my_world.results(
        to_csv=False,
        dashboard=True,
        dashboard_path=str(HERE / 'mpc_dashboard.html'),
    )

    print("\n=== Simulation Complete ===")
    for name, df in results.items():
        print(f"  {name:20s} -> {df.shape[0]:3d} steps, "
              f"{df.shape[1] - 1} variables")

    return results


if __name__ == "__main__":
    results = main()

"""
MATLAB / Octave co-simulation example
======================================

This example couples three MATLAB/Octave models via energysim:

    heatpump.m     – Carnot-limited air-source heat pump
    battery.m      – Li-ion home battery (Tesla Powerwall-sized)
    thermal_mass.m – Single-zone lumped-capacitance building

A CSV file provides outdoor temperature and a fixed electrical load
profile, and a simple rule-based controller decides the heat-pump and
battery set-points each time-step.

Run
---
    python run_matlab_cosim.py

Requirements
------------
* Python >= 3.9
* energysim (installed from the ``src/`` folder)
* GNU Octave on PATH  -or-  MATLAB with the MATLAB Engine for Python

The script auto-detects whichever engine is available.
"""

import os
import sys
import math
import pathlib

import numpy as np
import pandas as pd

# ── Make sure the local energysim source is importable ──────────────
SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import energysim  # noqa: E402


# ====================================================================
# 1.  Generate a synthetic weather / load CSV  (24 h, 15-min steps)
# ====================================================================

def generate_weather_csv(csv_path: str, n_steps: int = 96,
                         dt: float = 900.0) -> None:
    """Create a simple sinusoidal outdoor-temperature + load profile."""
    times = np.arange(n_steps) * dt                      # [s]
    hours = times / 3600.0

    # Outdoor temperature: 2 °C at 05:00, 12 °C at 15:00
    T_ambient = 7.0 + 5.0 * np.sin(2.0 * math.pi *
                                     (hours - 5.0) / 24.0)

    # Base electrical load (kW) – domestic profile
    P_load = 1.5 + 1.0 * np.sin(2.0 * math.pi *
                                  (hours - 8.0) / 24.0)

    # Constant heat-pump command and battery idle
    P_hp = np.full_like(hours, 3.0)    # kW
    P_bat = np.full_like(hours, 0.0)   # kW  (idle)

    df = pd.DataFrame({
        "time":        times,
        "T_ambient":   np.round(T_ambient, 2),
        "P_load":      np.round(P_load, 2),
        "P_hp_cmd":    P_hp,
        "P_bat_cmd":   P_bat,
    })
    df.to_csv(csv_path, index=False)
    print(f"[weather] wrote {csv_path}  ({n_steps} rows)")


# ====================================================================
# 2.  Build the co-simulation world
# ====================================================================

def main() -> None:
    here     = pathlib.Path(__file__).resolve().parent
    models   = here / "models"
    csv_path = here / "weather.csv"

    # -- Synthetic weather file --
    n_steps  = 96          # 24 h at 15-min resolution
    dt_macro = 900.0       # 15 min in seconds
    generate_weather_csv(str(csv_path), n_steps=n_steps, dt=dt_macro)

    # -- Create world --
    sim = energysim.world(start_time=0, stop_time=n_steps * dt_macro,
                          logging=True, t_macro=dt_macro)

    # --- Weather / load (CSV) ---
    sim.add_simulator(
        sim_type="csv",
        sim_name="weather",
        sim_loc=str(csv_path),
        outputs=["T_ambient", "P_load", "P_hp_cmd", "P_bat_cmd"],
        step_size=dt_macro,
    )

    # --- Heat pump (MATLAB/Octave) ---
    sim.add_simulator(
        sim_type="matlab",
        sim_name="heatpump",
        sim_loc=str(models / "heatpump.m"),
        inputs=["P_electric", "T_source", "T_sink"],
        outputs=["Q_thermal", "COP"],
        step_size=dt_macro,
        engine="auto",                   # try MATLAB, fall back to Octave
    )

    # --- Battery (MATLAB/Octave) ---
    sim.add_simulator(
        sim_type="matlab",
        sim_name="battery",
        sim_loc=str(models / "battery.m"),
        inputs=["P_cmd"],
        outputs=["SoC", "P_actual"],
        step_size=dt_macro,
        engine="auto",
    )

    # --- Building thermal envelope (MATLAB/Octave) ---
    sim.add_simulator(
        sim_type="matlab",
        sim_name="thermal_mass",
        sim_loc=str(models / "thermal_mass.m"),
        inputs=["Q_heating", "T_ambient"],
        outputs=["T_inside"],
        step_size=dt_macro,
        engine="auto",
    )

    # ================================================================
    # 3.  Connections  (dict: "sender.var" → "receiver.var")
    # ================================================================
    connections = {
        # Weather → simulators
        'weather.T_ambient':  ('heatpump.T_source', 'thermal_mass.T_ambient'),
        'weather.P_hp_cmd':   'heatpump.P_electric',
        'weather.P_bat_cmd':  'battery.P_cmd',

        # Heat-pump → building
        'heatpump.Q_thermal': 'thermal_mass.Q_heating',

        # Building → heat-pump feedback (T_sink = indoor temperature)
        'thermal_mass.T_inside': 'heatpump.T_sink',
    }
    sim.add_connections(connections)

    # ================================================================
    # 4.  Run
    # ================================================================
    print("\n── Starting co-simulation ──")
    sim.simulate(pbar=True)

    # ================================================================
    # 5.  Results
    # ================================================================
    res = sim.results(
        to_csv=False,
        dashboard=True,
        dashboard_path=str(here / "matlab_cosim_dashboard.html"),
    )

    print("\n=== Simulation Complete ===")
    for name, df in res.items():
        print(f"  {name:15s} -> {df.shape[0]} steps, {df.shape[1] - 1} variables")

    print("── Done ──")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Microgrid co-simulation — behind-the-meter PV + battery + heat pump
   + thermal storage + greenhouse.

Demonstrates multi-energy (electrical + thermal) co-simulation combining
dynamic, quasi-steady-state, and steady-state models at different time scales:

  Battery        10 s   (fast dynamics)
  Heat pump      30 s   (fast dynamics)
  Thermal tank   60 s   (medium dynamics)
  Greenhouse     60 s   (medium dynamics)
  Controller    900 s   (quasi-steady-state)
  Power grid    300 s   (steady-state power flow)
  Weather CSV   900 s   (input profiles)

24-hour winter-day simulation with 900 s macro time-step.
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

from energysim import world  # noqa: E402

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

# Generate data files if they don't exist
if not (HERE / "weather_data.csv").exists() or not (HERE / "microgrid.p").exists():
    print("Generating data files ...")
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import generate_data
    generate_data.main()

SIM_DIR = str(HERE / "simulators")


def main():
    my_world = world(start_time=0, stop_time=86400, logging=True, t_macro=900)

    # ── CSV weather / demand profiles ─────────────────────────────
    my_world.add_simulator(
        sim_type='csv', sim_name='weather',
        sim_loc=str(HERE / 'weather_data.csv'),
        outputs=['T_ambient', 'solar_irradiance', 'elec_demand',
                 'heat_demand', 'elec_price', 'pv_power'],
        step_size=900,
    )

    # ── Pandapower microgrid  (AC power flow) ─────────────────────
    my_world.add_simulator(
        sim_type='powerflow', sim_name='grid',
        sim_loc=str(HERE / 'microgrid.p'),
        inputs=['PV.p_mw', 'Battery.p_mw', 'HeatPump.p_mw', 'House.p_mw'],
        outputs=[
            'Bus0.vm_pu', 'Bus1.vm_pu', 'Bus2.vm_pu',
            'PV.p_mw', 'Battery.p_mw',
            'HeatPump.p_mw', 'House.p_mw',
        ],
        step_size=300,
        pf='pf',
    )

    # ── External simulators  (dynamic models) ────────────────────
    my_world.add_simulator(
        sim_type='external', sim_name='battery',
        sim_loc=SIM_DIR,
        inputs=['P_cmd'],
        outputs=['SoC', 'P_actual'],
        step_size=10,
    )

    my_world.add_simulator(
        sim_type='external', sim_name='heatpump',
        sim_loc=SIM_DIR,
        inputs=['P_cmd', 'T_source', 'T_sink'],
        outputs=['Q_thermal', 'COP', 'P_elec'],
        step_size=30,
    )

    my_world.add_simulator(
        sim_type='external', sim_name='thermal_tank',
        sim_loc=SIM_DIR,
        inputs=['Q_in', 'Q_out', 'T_ambient'],
        outputs=['T_storage', 'Q_loss'],
        step_size=60,
    )

    my_world.add_simulator(
        sim_type='external', sim_name='greenhouse',
        sim_loc=SIM_DIR,
        inputs=['Q_heating', 'T_ambient', 'solar_irradiance'],
        outputs=['T_inside', 'Q_demand'],
        step_size=60,
    )

    my_world.add_simulator(
        sim_type='external', sim_name='controller',
        sim_loc=SIM_DIR,
        inputs=['SoC', 'T_storage', 'T_greenhouse', 'P_pv',
                'elec_price', 'heat_demand_signal', 'T_ambient'],
        outputs=['P_battery_cmd', 'P_hp_cmd'],
        step_size=900,
    )

    # ── Connections  (tuples = fan-out) ───────────────────────────
    connections = {
        # Weather → various simulators
        'weather.T_ambient':        ('heatpump.T_source', 'thermal_tank.T_ambient',
                                     'greenhouse.T_ambient', 'controller.T_ambient'),
        'weather.solar_irradiance': 'greenhouse.solar_irradiance',
        'weather.pv_power':         ('grid.PV.p_mw', 'controller.P_pv'),
        'weather.elec_demand':      'grid.House.p_mw',
        'weather.heat_demand':      'controller.heat_demand_signal',
        'weather.elec_price':       'controller.elec_price',

        # Controller → actuators
        'controller.P_battery_cmd': 'battery.P_cmd',
        'controller.P_hp_cmd':      'heatpump.P_cmd',

        # Battery → grid + controller feedback
        'battery.P_actual':         'grid.Battery.p_mw',
        'battery.SoC':              'controller.SoC',

        # Heat pump → thermal chain + grid
        'heatpump.Q_thermal':       'thermal_tank.Q_in',
        'heatpump.P_elec':          'grid.HeatPump.p_mw',

        # Thermal tank → feedback
        'thermal_tank.T_storage':   ('heatpump.T_sink', 'controller.T_storage'),

        # Greenhouse ↔ thermal loop
        'greenhouse.Q_demand':      'thermal_tank.Q_out',
        'greenhouse.T_inside':      'controller.T_greenhouse',
    }
    my_world.add_connections(connections)

    print("Starting microgrid co-simulation ...")
    my_world.simulate(pbar=True, record_all=False)

    results = my_world.results(
        to_csv=False,
        dashboard=True,
        dashboard_path=str(HERE / 'microgrid_dashboard.html'),
    )

    print("\n=== Simulation Complete ===")
    for name, df in results.items():
        print(f"  {name:15s} -> {df.shape[0]} steps, {df.shape[1] - 1} variables")

    return results


if __name__ == "__main__":
    results = main()

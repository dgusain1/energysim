#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""9-bus PowerFactory co-simulation — main runner.

Co-simulates a DIgSILENT PowerFactory 9-bus network with P2H and P2G
flexible loads, driven by renewable generation and a dispatch controller.

Architecture
────────────
  3 simulators:
    • weather      (CSV)          — wind, PV, price, demand profiles
    • controller   (external)     — dispatches P2H / P2G setpoints
    • grid         (powerfactory) — 9-bus load flow

  IEEE 9-bus style:
    Bus1 (slack/Gen1), Bus2 (Gen2), Bus3 (Gen3),
    Bus4–Bus9 (load/transmission buses).
    Loads: Load_Conv (~125 MW), Load_P2H (~50 MW), Load_P2G (~50 MW)
    Renewables: WindFarm, PVFarm (as generators or negative loads)

Usage
─────
  1. Set the configuration variables below (marked <-- CONFIGURE).
  2. Ensure PowerFactory is running or pf_path is correct.
  3. Run:  python run_pf_ninebus.py
"""

import sys
from pathlib import Path

# ── Ensure energysim is importable ──
_src = str(Path(__file__).resolve().parents[2] / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from energysim import world  # noqa: E402

# ====================================================================
#  CONFIGURATION — fill in your PowerFactory details
# ====================================================================

# PowerFactory project name (as it appears in the PF Data Manager)
PF_PROJECT_NAME = "IEEE_9Bus_P2X"                          # <-- CONFIGURE

# Path to the PowerFactory Python directory.
# Example: r"C:\DIgSILENT\PowerFactory 2024\Python\3.11"
# Set to None if 'powerfactory' is already importable (e.g. running
# inside PF's built-in Python interpreter).
PF_PATH = r"C:\DIgSILENT\PowerFactory 2024\Python\3.11"    # <-- CONFIGURE

# ── PowerFactory element names (must match your PF model) ──
# Loads
LOAD_P2H_NAME  = "Load_P2H"       # <-- CONFIGURE: P2H load element name
LOAD_P2G_NAME  = "Load_P2G"       # <-- CONFIGURE: P2G load element name
LOAD_CONV_NAME = "Load_Conv"      # <-- CONFIGURE: conventional load name

# Generators / renewable sources
WIND_NAME = "WindFarm"             # <-- CONFIGURE: wind generator name
PV_NAME   = "PVFarm"              # <-- CONFIGURE: PV generator name
GEN1_NAME = "Gen1"                # <-- CONFIGURE: slack generator name
GEN2_NAME = "Gen2"                # <-- CONFIGURE: generator 2 name
GEN3_NAME = "Gen3"                # <-- CONFIGURE: generator 3 name

# Buses to monitor voltages
BUS_NAMES = [
    "Bus1", "Bus2", "Bus3",        # <-- CONFIGURE: generator buses
    "Bus4", "Bus5", "Bus6",        # <-- CONFIGURE: transmission buses
    "Bus7", "Bus8", "Bus9",        # <-- CONFIGURE: load buses
]

# Bus where P2H and P2G are connected (for voltage feedback)
BUS_P2H = "Bus8"                   # <-- CONFIGURE
BUS_P2G = "Bus6"                   # <-- CONFIGURE

# ── PowerFactory attribute names ──
# Input attributes (written to element data model).
# Typically 'plini' for active power [MW] and 'qlini' for reactive.
# The pfAdapter auto-prefixes 'e:' if no prefix is given.
LOAD_P_ATTR   = "plini"           # <-- CONFIGURE: active power attribute
LOAD_Q_ATTR   = "qlini"           # <-- CONFIGURE: reactive power attribute
GEN_P_ATTR    = "pgini"           # <-- CONFIGURE: generator active power
# Output / result attributes (read after load flow).
# 'm:' prefix = measured / result variables in PF.
BUS_V_ATTR    = "m:u"             # <-- CONFIGURE: bus voltage result
LOAD_P_RES    = "m:P:bus1"        # <-- CONFIGURE: load active power result
GEN_P_RES     = "m:P:bus1"        # <-- CONFIGURE: generator power result

# ====================================================================

HERE     = Path(__file__).resolve().parent
SIM_DIR  = str(HERE / "simulators")
CSV_PATH = str(HERE / "weather.csv")


def run():
    """Set up and run the 9-bus PowerFactory co-simulation."""

    # ── Generate weather CSV if it doesn't exist ──
    if not Path(CSV_PATH).exists():
        print("Generating weather CSV data ...")
        from generate_data import main as gen_main
        gen_main()

    # ── Simulation parameters ──
    T_START   = 0
    T_STOP    = 86400       # 24 hours
    DT_MACRO  = 900         # 15-minute macro time step

    w = world(
        start_time=T_START,
        stop_time=T_STOP,
        logging=True,
        t_macro=DT_MACRO,
    )

    # ────────────────────────────────────────────────────────────
    #  1. Weather / renewable data (CSV)
    # ────────────────────────────────────────────────────────────
    w.add_simulator(
        sim_type='csv',
        sim_name='weather',
        sim_loc=CSV_PATH,
        outputs=[
            'wind_power',    # [MW] wind farm output
            'pv_power',      # [MW] PV farm output
            'elec_price',    # [EUR/MWh]
            'T_ambient',     # [°C]
            'p2h_demand',    # [MW] heat demand
            'p2g_demand',    # [MW] gas demand
        ],
        step_size=DT_MACRO,
    )

    # ────────────────────────────────────────────────────────────
    #  2. Dispatch controller (external Python)
    # ────────────────────────────────────────────────────────────
    w.add_simulator(
        sim_type='external',
        sim_name='dispatch_controller',
        sim_loc=SIM_DIR,
        inputs=[
            'wind_power',    # from weather CSV
            'pv_power',      # from weather CSV
            'elec_price',    # from weather CSV
            'p2h_demand',    # from weather CSV
            'p2g_demand',    # from weather CSV
            'bus_v_p2h',     # voltage feedback from PF grid
            'bus_v_p2g',     # voltage feedback from PF grid
        ],
        outputs=[
            'P2H_cmd',      # [MW] P2H active power command
            'P2G_cmd',      # [MW] P2G active power command
            'ren_surplus',   # [MW] renewable surplus (info)
        ],
        step_size=DT_MACRO,
    )

    # ────────────────────────────────────────────────────────────
    #  3. PowerFactory 9-bus grid
    # ────────────────────────────────────────────────────────────
    #
    # Inputs:  set active power of P2H, P2G loads and renewable gens
    # Outputs: read bus voltages + load/gen active power results
    #
    pf_inputs = [
        f'{LOAD_P2H_NAME}.{LOAD_P_ATTR}',    # P2H active power [MW]
        f'{LOAD_P2G_NAME}.{LOAD_P_ATTR}',    # P2G active power [MW]
        f'{WIND_NAME}.{GEN_P_ATTR}',          # Wind gen setpoint [MW]
        f'{PV_NAME}.{GEN_P_ATTR}',            # PV gen setpoint [MW]
    ]

    pf_outputs = (
        # Bus voltages
        [f'{bus}.{BUS_V_ATTR}' for bus in BUS_NAMES]
        # Load active power results
        + [f'{LOAD_P2H_NAME}.{LOAD_P_RES}',
           f'{LOAD_P2G_NAME}.{LOAD_P_RES}',
           f'{LOAD_CONV_NAME}.{LOAD_P_RES}']
        # Generator results
        + [f'{GEN1_NAME}.{GEN_P_RES}',
           f'{GEN2_NAME}.{GEN_P_RES}',
           f'{GEN3_NAME}.{GEN_P_RES}']
        # Renewable results
        + [f'{WIND_NAME}.{GEN_P_RES}',
           f'{PV_NAME}.{GEN_P_RES}']
    )

    w.add_simulator(
        sim_type='powerfactory',
        sim_name='grid',
        sim_loc='',                          # not used for PF
        inputs=pf_inputs,
        outputs=pf_outputs,
        step_size=DT_MACRO,
        pf='ldf',                            # load flow
        pf_path=PF_PATH,
    )

    # ────────────────────────────────────────────────────────────
    #  4. Connections
    # ────────────────────────────────────────────────────────────
    connections = {
        # Weather → controller + grid (fan-out via tuples)
        'weather.wind_power':  ('dispatch_controller.wind_power',
                                f'grid.{WIND_NAME}.{GEN_P_ATTR}'),
        'weather.pv_power':    ('dispatch_controller.pv_power',
                                f'grid.{PV_NAME}.{GEN_P_ATTR}'),
        'weather.elec_price':  'dispatch_controller.elec_price',
        'weather.p2h_demand':  'dispatch_controller.p2h_demand',
        'weather.p2g_demand':  'dispatch_controller.p2g_demand',

        # Controller → grid (set P2H / P2G load active power)
        f'dispatch_controller.P2H_cmd': f'grid.{LOAD_P2H_NAME}.{LOAD_P_ATTR}',
        f'dispatch_controller.P2G_cmd': f'grid.{LOAD_P2G_NAME}.{LOAD_P_ATTR}',

        # Grid → controller (voltage feedback)
        f'grid.{BUS_P2H}.{BUS_V_ATTR}': 'dispatch_controller.bus_v_p2h',
        f'grid.{BUS_P2G}.{BUS_V_ATTR}': 'dispatch_controller.bus_v_p2g',
    }

    w.add_connections(connections)

    # ────────────────────────────────────────────────────────────
    #  5. Initial values (optional)
    # ────────────────────────────────────────────────────────────
    initializations = {
        'grid': (
            [f'{LOAD_P2H_NAME}.{LOAD_P_ATTR}',
             f'{LOAD_P2G_NAME}.{LOAD_P_ATTR}'],
            [20.0, 15.0]   # start with moderate P2H / P2G load
        ),
    }
    w.options({'init': initializations})

    # ────────────────────────────────────────────────────────────
    #  6. Simulate
    # ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  9-Bus PowerFactory Co-Simulation")
    print("  Project: ", PF_PROJECT_NAME)
    print("  Duration: 24 h | Macro step: 15 min | Steps: 96")
    print("=" * 60)

    w.simulate(pbar=True, record_all=False)

    # ────────────────────────────────────────────────────────────
    #  7. Results
    # ────────────────────────────────────────────────────────────
    res = w.results(to_csv=False)

    print("\n── Result keys ──")
    for k in res:
        print(f"  {k}: {list(res[k].columns)}")

    # ── Quick plots (if matplotlib is available) ──
    try:
        import matplotlib.pyplot as plt

        time_h = res['grid']['time'] / 3600.0

        # --- Bus voltages ---
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        for bus in BUS_NAMES:
            col = f'{bus}.{BUS_V_ATTR}'
            if col in res['grid'].columns:
                ax1.plot(time_h, res['grid'][col], label=bus)
        ax1.set_xlabel('Time [h]')
        ax1.set_ylabel('Voltage [pu]')
        ax1.set_title('9-Bus Voltages')
        ax1.legend(ncol=3, fontsize=8)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()

        # --- P2H / P2G dispatch ---
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ctrl = res['dispatch_controller']
        ax2.plot(ctrl['time'] / 3600.0, ctrl['P2H_cmd'], label='P2H command')
        ax2.plot(ctrl['time'] / 3600.0, ctrl['P2G_cmd'], label='P2G command')
        ax2.plot(ctrl['time'] / 3600.0, ctrl['ren_surplus'],
                 '--', label='Renewable surplus', alpha=0.7)
        ax2.set_xlabel('Time [h]')
        ax2.set_ylabel('Power [MW]')
        ax2.set_title('Dispatch Controller — P2H / P2G Commands')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='k', linewidth=0.5)
        plt.tight_layout()

        # --- Renewable generation ---
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        wdata = res['weather']
        ax3.plot(wdata['time'] / 3600.0, wdata['wind_power'], label='Wind')
        ax3.plot(wdata['time'] / 3600.0, wdata['pv_power'], label='PV')
        ax3.fill_between(wdata['time'] / 3600.0, 0, wdata['wind_power'],
                         alpha=0.15)
        ax3.fill_between(wdata['time'] / 3600.0, 0, wdata['pv_power'],
                         alpha=0.15)
        ax3.set_xlabel('Time [h]')
        ax3.set_ylabel('Power [MW]')
        ax3.set_title('Renewable Generation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.show()

    except ImportError:
        print("\n(matplotlib not installed — skipping plots)")

    return res


if __name__ == '__main__':
    run()

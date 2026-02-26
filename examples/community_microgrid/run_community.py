#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Community microgrid co-simulation — main runner.

Topology
────────
  8 simulators:  weather (csv), community_battery (matlab/octave),
  household_2p_ev_pv (external), household_1p (external),
  mall (external), supermarket (external), dispatcher (external),
  grid (powerflow).

  Multi-physics models:
  • Electrochemical 2-RC NMC622 battery (Arrhenius kinetics, thermal, aging)
  • R-C thermal building envelopes (households & commercial)
  • PV with NOCT cell temperature + thermal derating
  • EV with electrochemical CC-CV charging + cold-weather penalty
  • Multi-zone HVAC with COP = f(ΔT)  (mall, supermarket)
  • Multi-temperature refrigeration (freezer / chiller) with defrost
  • MPC dispatcher with rolling 4-hour horizon

  24 hours simulated at 15-minute macro time step (96 steps).

Usage
─────
  python run_community.py
"""

import sys
from pathlib import Path

# ── Ensure energysim is importable ──
_root  = Path(__file__).resolve().parents[1]
_src   = str(_root / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from energysim import world

# ── Paths ──
HERE     = Path(__file__).resolve().parent
SIM_DIR  = str(HERE / "simulators")
MODEL_DIR = str(HERE / "models")
GRID_PATH = str(HERE / "community_grid.p")
CSV_PATH  = str(HERE / "weather.csv")

# ────────────────────────────────────────────────────────
#  Generate network + weather if they don't exist
# ────────────────────────────────────────────────────────
if not Path(GRID_PATH).exists() or not Path(CSV_PATH).exists():
    from generate_data import main as gen_main
    gen_main()


def run():
    """Set up and run the community microgrid co-simulation."""

    # ── Simulation parameters ──
    T_START = 0
    T_STOP  = 86400       # 24 hours
    DT      = 900         # 15 min macro step

    w = world(start_time=T_START, stop_time=T_STOP,
              logging=True, t_macro=DT)

    # ────────────────────────────────────────────────────
    #  1. Weather  (CSV source — T_ambient, solar, price)
    # ────────────────────────────────────────────────────
    w.add_simulator(
        sim_type='csv',
        sim_name='weather',
        sim_loc=CSV_PATH,
        inputs=[],
        outputs=['T_ambient', 'solar', 'elec_price'],
        step_size=DT,
    )

    # ────────────────────────────────────────────────────
    #  2. Community battery  (MATLAB / Octave)
    #     Electrochemical 2-RC NMC622 + thermal + aging
    # ────────────────────────────────────────────────────
    w.add_simulator(
        sim_type='matlab',
        sim_name='community_battery',
        sim_loc=str(Path(MODEL_DIR) / 'community_battery.m'),
        inputs=['P_cmd', 'T_ambient'],
        outputs=['SoC', 'P_actual', 'E_available',
                 'V_terminal', 'T_cell', 'R_internal', 'Q_capacity'],
        step_size=DT,
    )

    # ────────────────────────────────────────────────────
    #  3. Household — 2-person with EV + PV
    #     R-C thermal building, electrochemical EV, PV thermal derating
    # ────────────────────────────────────────────────────
    w.add_simulator(
        sim_type='external',
        sim_name='household_2p_ev_pv',
        sim_loc=SIM_DIR,
        inputs=['T_ambient', 'solar'],
        outputs=['P_net', 'P_pv', 'P_ev', 'T_indoor'],
        step_size=DT,
    )

    # ────────────────────────────────────────────────────
    #  4. Household — single person
    #     R-C thermal building
    # ────────────────────────────────────────────────────
    w.add_simulator(
        sim_type='external',
        sim_name='household_1p',
        sim_loc=SIM_DIR,
        inputs=['T_ambient', 'solar'],
        outputs=['P_net', 'T_indoor'],
        step_size=DT,
    )

    # ────────────────────────────────────────────────────
    #  5. Mall — multi-zone HVAC + elevators + EV chargers
    # ────────────────────────────────────────────────────
    w.add_simulator(
        sim_type='external',
        sim_name='mall',
        sim_loc=SIM_DIR,
        inputs=['T_ambient', 'solar', 'P_ev_limit'],
        outputs=['P_net', 'P_ev_mall', 'P_hvac_mall', 'T_zone_avg'],
        step_size=DT,
    )

    # ────────────────────────────────────────────────────
    #  6. Supermarket — multi-zone refrigeration + HVAC
    # ────────────────────────────────────────────────────
    w.add_simulator(
        sim_type='external',
        sim_name='supermarket',
        sim_loc=SIM_DIR,
        inputs=['T_ambient', 'solar'],
        outputs=['P_net', 'P_refrig', 'P_hvac_sup', 'T_store'],
        step_size=DT,
    )

    # ────────────────────────────────────────────────────
    #  7. MPC Dispatcher
    # ────────────────────────────────────────────────────
    w.add_simulator(
        sim_type='external',
        sim_name='dispatcher',
        sim_loc=SIM_DIR,
        inputs=['SoC', 'P_h1', 'P_h2', 'P_mall', 'P_super',
                'P_pv_h1', 'elec_price', 'solar'],
        outputs=['P_batt_cmd', 'P_ev_limit'],
        step_size=DT,
    )

    # ────────────────────────────────────────────────────
    #  8. Power-flow grid
    # ────────────────────────────────────────────────────
    w.add_simulator(
        sim_type='powerflow',
        sim_name='grid',
        sim_loc=GRID_PATH,
        inputs=[],
        outputs=[],
        step_size=DT,
    )

    # ────────────────────────────────────────────────────
    #  Connections
    # ────────────────────────────────────────────────────
    w.add_connections({
        # Weather → all agents + battery + dispatcher
        'weather.T_ambient': (
            'household_2p_ev_pv.T_ambient',
            'household_1p.T_ambient',
            'mall.T_ambient',
            'supermarket.T_ambient',
            'community_battery.T_ambient',
        ),
        'weather.solar': (
            'household_2p_ev_pv.solar',
            'household_1p.solar',
            'mall.solar',
            'supermarket.solar',
            'dispatcher.solar',
        ),
        'weather.elec_price': 'dispatcher.elec_price',

        # Agent loads → grid + dispatcher
        'household_2p_ev_pv.P_net': (
            'grid.House1.p_mw',
            'dispatcher.P_h1',
        ),
        'household_2p_ev_pv.P_pv': 'dispatcher.P_pv_h1',
        'household_1p.P_net': (
            'grid.House2.p_mw',
            'dispatcher.P_h2',
        ),
        'mall.P_net': (
            'grid.Mall.p_mw',
            'dispatcher.P_mall',
        ),
        'supermarket.P_net': (
            'grid.Supermarket.p_mw',
            'dispatcher.P_super',
        ),

        # Dispatcher → battery + mall EV
        'dispatcher.P_batt_cmd': 'community_battery.P_cmd',
        'dispatcher.P_ev_limit': 'mall.P_ev_limit',

        # Battery → dispatcher + grid
        'community_battery.SoC':      'dispatcher.SoC',
        'community_battery.P_actual': 'grid.Battery.p_mw',
    })

    # ────────────────────────────────────────────────────
    #  Run
    # ────────────────────────────────────────────────────
    print("=" * 60)
    print("  Community Microgrid — Multi-Physics Co-Simulation")
    print("=" * 60)
    print(f"  Duration : {T_STOP // 3600} hours")
    print(f"  Step     : {DT // 60} min")
    print(f"  Steps    : {T_STOP // DT}")
    print(f"  Models   : 8 (2 household + 2 commercial + battery + PV")
    print(f"             + MPC dispatcher + power-flow grid)")
    print("=" * 60)

    w.simulate(pbar=True)

    # ────────────────────────────────────────────────────
    #  Results
    # ────────────────────────────────────────────────────
    # Get results dict + open interactive dashboard
    res = w.results(to_csv=False, dashboard=True)

    # Also export to CSV
    try:
        w.results(to_csv=True, dashboard=False)
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("  Simulation complete.  Summary")
    print("=" * 60)
    if isinstance(res, dict):
        for sim_name, df in res.items():
            cols = [c for c in df.columns if c != 'time']
            print(f"  {sim_name:30s}  →  {', '.join(cols)}")
    else:
        print(f"  {res}")
    print("=" * 60)

    return res


if __name__ == "__main__":
    run()

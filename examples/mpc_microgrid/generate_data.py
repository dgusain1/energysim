#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the pandapower network and weather CSV for the MPC example.

Run this script once (or let run_mpc.py call it automatically).
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_src = str(Path(__file__).resolve().parents[2] / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


# ── Pandapower network ──────────────────────────────────────────────
def create_network(path: str) -> None:
    """Create a simple 5-bus LV microgrid and save as pickle."""
    import pandapower as pp

    net = pp.create_empty_network(name="MPC_Microgrid")

    # --- Buses (0.4 kV) ---
    bus_slack = pp.create_bus(net, vn_kv=0.4, name="Slack")
    bus_pv    = pp.create_bus(net, vn_kv=0.4, name="BusPV")
    bus_batt  = pp.create_bus(net, vn_kv=0.4, name="BusBatt")
    bus_hp    = pp.create_bus(net, vn_kv=0.4, name="BusHP")
    bus_load  = pp.create_bus(net, vn_kv=0.4, name="BusLoad")

    # --- External grid (slack) ---
    pp.create_ext_grid(net, bus=bus_slack, vm_pu=1.0, name="Grid")

    # --- Lines  (NAYY 4x150, standard LV cable) ---
    for from_bus, to_bus, length_km, name in [
        (bus_slack, bus_pv,   0.05, "Line_Slack_PV"),
        (bus_slack, bus_load, 0.08, "Line_Slack_Load"),
        (bus_pv,   bus_batt, 0.03, "Line_PV_Batt"),
        (bus_pv,   bus_hp,   0.04, "Line_PV_HP"),
    ]:
        pp.create_line(net, from_bus=from_bus, to_bus=to_bus,
                       length_km=length_km,
                       std_type="NAYY 4x150 SE", name=name)

    # --- Static generator (PV) — negative load convention ---
    pp.create_sgen(net, bus=bus_pv, p_mw=0.0, q_mvar=0.0, name="PV")

    # --- Loads ---
    pp.create_load(net, bus=bus_batt, p_mw=0.0, q_mvar=0.0, name="Battery")
    pp.create_load(net, bus=bus_hp,   p_mw=0.0, q_mvar=0.0, name="HeatPump")
    pp.create_load(net, bus=bus_load, p_mw=0.002, q_mvar=0.0, name="House")

    pp.runpp(net)  # validate
    pp.to_pickle(net, path)
    print(f"[network] saved {path}")


# ── Weather / price CSV ─────────────────────────────────────────────
def create_weather_csv(path: str, n_steps: int = 96,
                       dt: float = 900.0) -> None:
    """24h synthetic profiles at 15-min resolution."""
    t = np.arange(n_steps) * dt
    h = t / 3600.0

    # Outdoor temperature: min 0 °C at 05:00, max 10 °C at 15:00
    T_ambient = 5.0 + 5.0 * np.sin(2 * math.pi * (h - 5.0) / 24.0)

    # Solar irradiance  (W/m²) — bell-shaped during day
    solar = np.maximum(0.0, 800.0 * np.sin(math.pi * (h - 6.0) / 12.0))
    solar[h < 6.0] = 0.0
    solar[h > 18.0] = 0.0

    # PV output (MW) — 5 kWp panel
    P_pv = solar / 1000.0 * 0.005

    # Base household load (MW)
    P_load = 0.002 + 0.001 * np.sin(2 * math.pi * (h - 8.0) / 24.0)

    # Electricity price  (EUR/kWh) — high during evening peak
    price = 0.15 + 0.10 * np.sin(2 * math.pi * (h - 18.0) / 24.0)

    df = pd.DataFrame({
        "time":       t,
        "T_ambient":  np.round(T_ambient, 3),
        "solar":      np.round(solar, 1),
        "P_pv":       np.round(P_pv, 6),
        "P_load":     np.round(P_load, 6),
        "elec_price": np.round(price, 4),
    })
    df.to_csv(path, index=False)
    print(f"[weather] saved {path}  ({n_steps} rows)")


# ────────────────────────────────────────────────────────────────────
def main():
    here = Path(__file__).resolve().parent
    create_network(str(here / "microgrid.p"))
    create_weather_csv(str(here / "weather.csv"))


if __name__ == "__main__":
    main()

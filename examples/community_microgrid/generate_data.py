#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the pandapower community network and weather CSV.

Creates:
  - community_grid.p  : 7-bus LV network (slack + 4 agents + battery + EV)
  - weather.csv       : 24 h synthetic profiles (T_ambient, solar, elec_price)
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_src = str(Path(__file__).resolve().parents[2] / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


def create_network(path: str) -> None:
    """7-bus LV community microgrid."""
    import pandapower as pp

    net = pp.create_empty_network(name="CommunityMicrogrid")

    # --- Buses (0.4 kV LV feeder) ---
    b_slack = pp.create_bus(net, vn_kv=0.4, name="Slack")
    b_h1    = pp.create_bus(net, vn_kv=0.4, name="BusH1")
    b_h2    = pp.create_bus(net, vn_kv=0.4, name="BusH2")
    b_mall  = pp.create_bus(net, vn_kv=0.4, name="BusMall")
    b_super = pp.create_bus(net, vn_kv=0.4, name="BusSuper")
    b_batt  = pp.create_bus(net, vn_kv=0.4, name="BusBatt")

    # --- External grid ---
    pp.create_ext_grid(net, bus=b_slack, vm_pu=1.0, name="UtilityGrid")

    # --- Lines (NAYY 4x150 SE, typical LV cable) ---
    for fb, tb, length, name in [
        (b_slack, b_h1,    0.05, "Line_S_H1"),
        (b_slack, b_h2,    0.06, "Line_S_H2"),
        (b_slack, b_mall,  0.10, "Line_S_Mall"),
        (b_mall,  b_super, 0.04, "Line_Mall_Super"),
        (b_slack, b_batt,  0.03, "Line_S_Batt"),
    ]:
        pp.create_line(net, from_bus=fb, to_bus=tb,
                       length_km=length, std_type="NAYY 4x150 SE",
                       name=name)

    # --- Loads (initial zero — set by co-sim) ---
    pp.create_load(net, bus=b_h1,    p_mw=0.001, q_mvar=0.0, name="House1")
    pp.create_load(net, bus=b_h2,    p_mw=0.001, q_mvar=0.0, name="House2")
    pp.create_load(net, bus=b_mall,  p_mw=0.010, q_mvar=0.0, name="Mall")
    pp.create_load(net, bus=b_super, p_mw=0.005, q_mvar=0.0, name="Supermarket")
    pp.create_load(net, bus=b_batt,  p_mw=0.000, q_mvar=0.0, name="Battery")

    pp.runpp(net)
    pp.to_pickle(net, path)
    print(f"[network] saved {path}")


def create_weather_csv(path: str, n_steps: int = 96,
                       dt: float = 900.0) -> None:
    """24 h synthetic weather, solar irradiance, and price profile."""
    t = np.arange(n_steps) * dt
    h = t / 3600.0

    # Outdoor temperature (°C): min ~1 °C at 05:00, max ~11 °C at 15:00
    T_ambient = 6.0 + 5.0 * np.sin(2 * math.pi * (h - 5.0) / 24.0)

    # Global horizontal irradiance (W/m²) — bell-shaped 06:00–18:00
    solar = np.where(
        (h >= 6.0) & (h <= 18.0),
        850.0 * np.sin(math.pi * (h - 6.0) / 12.0),
        0.0,
    )

    # Electricity price (EUR/kWh):
    #   cheap overnight, afternoon shoulder, evening peak
    base = 0.15
    peak = 0.12 * np.sin(2 * math.pi * (h - 18.0) / 24.0)
    morning = 0.04 * np.where((h >= 7) & (h <= 9), 1.0, 0.0)
    price = base + peak + morning

    df = pd.DataFrame({
        "time":       t,
        "T_ambient":  np.round(T_ambient, 3),
        "solar":      np.round(solar, 1),
        "elec_price": np.round(price, 4),
    })
    df.to_csv(path, index=False)
    print(f"[weather] saved {path}  ({n_steps} rows)")


def main():
    here = Path(__file__).resolve().parent
    create_network(str(here / "community_grid.p"))
    create_weather_csv(str(here / "weather.csv"))


if __name__ == "__main__":
    main()

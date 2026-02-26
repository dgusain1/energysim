#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the weather / renewable CSV data for the 9-bus PF example.

Creates ``weather.csv`` with 24 hours of 15-minute data:
  - time       [s]
  - T_ambient   [°C]
  - wind_power  [MW]   (wind farm output, IEEE 9-bus scale)
  - pv_power    [MW]   (PV farm output)
  - elec_price  [EUR/MWh]
  - p2h_demand  [MW]   (heat demand signal for P2H)
  - p2g_demand  [MW]   (gas demand signal for P2G)

All values are realistic for a ~300 MW system (IEEE 9-bus total
generation ≈ 315 MW, total load ≈ 315 MW).
"""

import math
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    """Write weather.csv to the example directory."""
    dt = 900          # 15 min
    n_steps = 97      # 0 … 86400 s inclusive

    rows = []
    for i in range(n_steps):
        t = i * dt
        hour = t / 3600.0

        # ── Ambient temperature (sinusoidal, 8–22 °C) ──
        T_amb = 15.0 + 7.0 * math.sin(2 * math.pi * (hour - 6.0) / 24.0)

        # ── Wind farm output (0–100 MW, semi-random pattern) ──
        wind = (55.0
                + 30.0 * math.sin(2 * math.pi * hour / 12.0)
                + 15.0 * math.sin(2 * math.pi * hour / 5.3 + 1.2))
        wind = max(5.0, min(100.0, wind))

        # ── PV farm output (0–80 MW, bell curve 06–18 h) ──
        if 6.0 <= hour <= 18.0:
            pv = 80.0 * math.sin(math.pi * (hour - 6.0) / 12.0)
        else:
            pv = 0.0
        pv = max(0.0, pv)

        # ── Electricity price (EUR/MWh, diurnal shape) ──
        price = (45.0
                 + 25.0 * math.sin(2 * math.pi * (hour - 18.0) / 24.0)
                 + (10.0 if 7.0 <= hour <= 9.0 else 0.0)
                 + (8.0 if 17.0 <= hour <= 20.0 else 0.0))
        price = max(15.0, price)

        # ── P2H heat demand (MW, higher in morning/evening) ──
        p2h_demand = (20.0
                      + 15.0 * math.cos(2 * math.pi * (hour - 7.0) / 24.0)
                      + 5.0 * math.cos(2 * math.pi * (hour - 19.0) / 12.0))
        p2h_demand = max(5.0, min(50.0, p2h_demand))

        # ── P2G gas demand (MW, relatively flat with peak) ──
        p2g_demand = (15.0
                      + 10.0 * math.sin(2 * math.pi * (hour - 10.0) / 24.0))
        p2g_demand = max(5.0, min(45.0, p2g_demand))

        rows.append({
            'time':       t,
            'T_ambient':  round(T_amb, 2),
            'wind_power': round(wind, 3),
            'pv_power':   round(pv, 3),
            'elec_price': round(price, 2),
            'p2h_demand': round(p2h_demand, 3),
            'p2g_demand': round(p2g_demand, 3),
        })

    csv_path = HERE / 'weather.csv'
    fieldnames = list(rows[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[generate_data] Wrote {len(rows)} rows to {csv_path}")


if __name__ == '__main__':
    main()

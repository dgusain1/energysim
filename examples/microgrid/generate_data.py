#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate weather/demand CSV and pandapower network for the microgrid example.

Creates:
  - weather_data.csv   24 h winter-day profiles at 15-min resolution
  - microgrid.p        3-bus LV pandapower network (pickle)
"""

import numpy as np
import pandas as pd
from pathlib import Path


def _gauss(x, mu, sigma, amp):
    """Un-normalised Gaussian bell curve."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def main():
    HERE = Path(__file__).resolve().parent

    # ------------------------------------------------------------------
    # 1. Weather / demand / price CSV  (24 h, dt = 900 s)
    # ------------------------------------------------------------------
    times = np.arange(0, 86400 + 1, 900, dtype=float)
    hours = times / 3600.0

    # Ambient temperature  (winter day, °C)
    T_ambient = 3.0 + 4.0 * np.sin(2.0 * np.pi * (hours - 14.0) / 24.0)
    T_ambient = np.clip(T_ambient, -3.0, 10.0)

    # Solar irradiance  (W/m²)
    solar = np.where(
        (hours >= 6) & (hours <= 18),
        300.0 * np.sin(np.pi * (hours - 6.0) / 12.0),
        0.0,
    )
    solar = np.maximum(solar, 0.0)

    # PV power  (MW) — peak 8 kW
    pv_power = 0.008 * solar / 300.0

    # Electrical demand  (MW) — residential morning + evening peaks
    elec_demand = (
        0.002
        + _gauss(hours, 8.0, 1.5, 0.004)
        + _gauss(hours, 19.0, 2.0, 0.005)
    )

    # Heat demand reference signal  (MW, constant)
    heat_demand = np.full_like(times, 0.005)

    # Electricity price  (EUR/kWh)
    elec_price = (
        0.08
        + _gauss(hours, 8.0, 2.0, 0.17)
        + _gauss(hours, 18.0, 2.0, 0.20)
        - _gauss(hours, 13.0, 2.0, 0.06)
    )
    elec_price = np.clip(elec_price, 0.04, 0.35)

    df = pd.DataFrame({
        'time': times,
        'T_ambient': np.round(T_ambient, 3),
        'solar_irradiance': np.round(solar, 2),
        'elec_demand': np.round(elec_demand, 6),
        'heat_demand': np.round(heat_demand, 6),
        'elec_price': np.round(elec_price, 4),
        'pv_power': np.round(pv_power, 6),
    })
    csv_path = HERE / 'weather_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}  ({len(df)} rows)")

    # ------------------------------------------------------------------
    # 2. Pandapower microgrid network
    # ------------------------------------------------------------------
    import pandapower as pp

    net = pp.create_empty_network(name='microgrid')

    # Buses
    bus0 = pp.create_bus(net, vn_kv=20.0, name='Bus0')
    bus1 = pp.create_bus(net, vn_kv=0.4, name='Bus1')
    bus2 = pp.create_bus(net, vn_kv=0.4, name='Bus2')

    # External grid (slack on MV bus)
    pp.create_ext_grid(net, bus=bus0, name='grid')

    # MV / LV transformer
    pp.create_transformer(net, hv_bus=bus0, lv_bus=bus1,
                          std_type='0.25 MVA 20/0.4 kV', name='Trafo')

    # LV cable
    pp.create_line(net, from_bus=bus1, to_bus=bus2,
                   length_km=0.1, std_type='NAYY 4x50 SE', name='Cable')

    # PV as static generator on Bus 1
    pp.create_sgen(net, bus=bus1, p_mw=0.0, name='PV')

    # Loads
    pp.create_load(net, bus=bus1, p_mw=0.0, name='Battery')
    pp.create_load(net, bus=bus2, p_mw=0.003, name='House')
    pp.create_load(net, bus=bus2, p_mw=0.0, name='HeatPump')

    # Quick validation
    pp.runpp(net)
    print(f"Grid validated — converged, Vmin={net.res_bus.vm_pu.min():.4f} pu")

    pkl_path = HERE / 'microgrid.p'
    pp.to_pickle(net, str(pkl_path))
    print(f"Wrote {pkl_path}")


if __name__ == '__main__':
    main()

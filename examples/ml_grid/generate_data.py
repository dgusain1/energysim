#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate input data and pandapower network for the ML-grid example.

Creates:
  - features.csv          24 h of weather/time features at 15-min resolution
  - distribution_grid.p   5-bus 20 kV distribution network (pandapower pickle)
"""

import numpy as np
import pandas as pd
import pandapower as pp
from pathlib import Path

HERE = Path(__file__).resolve().parent
np.random.seed(0)


def create_features_csv():
    """Create features.csv with 24 h of time + weather data (dt = 900 s)."""
    time = np.arange(0, 86400 + 1, 900)
    hour = time / 3600.0

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    temperature = (8 + 5 * np.sin(2 * np.pi * (hour - 14) / 24)
                   + np.random.normal(0, 0.3, len(hour)))
    cloud_cover = np.clip(
        0.3 + 0.3 * np.sin(np.pi * hour / 12) + np.random.normal(0, 0.05, len(hour)), 0, 1)
    wind_speed = np.clip(
        5 + 3 * np.sin(2 * np.pi * hour / 24 + 1.5) + np.random.normal(0, 0.5, len(hour)), 0, 15)
    is_weekend = np.zeros_like(time, dtype=float)

    df = pd.DataFrame({
        "time": time,
        "hour_sin": np.round(hour_sin, 6),
        "hour_cos": np.round(hour_cos, 6),
        "temperature": np.round(temperature, 2),
        "cloud_cover": np.round(cloud_cover, 4),
        "wind_speed": np.round(wind_speed, 2),
        "is_weekend": is_weekend.astype(int),
    })
    path = HERE / "features.csv"
    df.to_csv(path, index=False)
    print(f"Created {path.name}  ({len(df)} rows)")
    return df


def create_distribution_grid():
    """Create a 5-bus 20 kV distribution grid and save as pickle."""
    net = pp.create_empty_network(name="ML_DistributionGrid")

    # Buses
    bus0 = pp.create_bus(net, vn_kv=110, name="HV_Bus")
    bus1 = pp.create_bus(net, vn_kv=20, name="MV_Bus1")
    bus2 = pp.create_bus(net, vn_kv=20, name="MV_Bus2")
    bus3 = pp.create_bus(net, vn_kv=20, name="MV_Bus3")
    bus4 = pp.create_bus(net, vn_kv=20, name="MV_Bus4")

    # External grid (slack) on HV bus
    pp.create_ext_grid(net, bus=bus0, vm_pu=1.02, name="Grid")

    # Transformer 110/20 kV
    pp.create_transformer(net, hv_bus=bus0, lv_bus=bus1,
                          std_type="40 MVA 110/20 kV", name="HV_MV_Trafo")

    # Lines (20 kV cable, 5 km each)
    line_type = "NA2XS2Y 1x95 RM/25 12/20 kV"
    pp.create_line(net, from_bus=bus1, to_bus=bus2, length_km=5,
                   std_type=line_type, name="Line_1_2")
    pp.create_line(net, from_bus=bus2, to_bus=bus3, length_km=5,
                   std_type=line_type, name="Line_2_3")
    pp.create_line(net, from_bus=bus1, to_bus=bus4, length_km=5,
                   std_type=line_type, name="Line_1_4")

    # Loads on Bus 3
    pp.create_load(net, bus=bus3, p_mw=0.002, q_mvar=0.0005, name="Residential")
    pp.create_load(net, bus=bus3, p_mw=0.003, q_mvar=0.001, name="Commercial")

    # Static generators (PV on Bus 2, Wind on Bus 4)
    pp.create_sgen(net, bus=bus2, p_mw=0.0, q_mvar=0.0, name="SolarPark")
    pp.create_sgen(net, bus=bus4, p_mw=0.0, q_mvar=0.0, name="WindFarm")

    # Quick validation
    pp.runpp(net)
    print(f"Grid validated — converged with Vmin={net.res_bus.vm_pu.min():.4f}, "
          f"Vmax={net.res_bus.vm_pu.max():.4f}")

    path = HERE / "distribution_grid.p"
    pp.to_pickle(net, str(path))
    print(f"Created {path.name}")
    return net


def main():
    create_features_csv()
    create_distribution_grid()
    print("\nData generation complete.")


if __name__ == "__main__":
    main()

# Microgrid Co-Simulation Example

Behind-the-meter microgrid coupling electrical and thermal domains across
five dynamic models, a pandapower AC power-flow grid, and CSV weather/demand
profiles — all orchestrated by energysim.

## Scenario

A 24-hour winter-day simulation of a prosumer microgrid with:

| Simulator       | Type       | Step Size | Description                                   |
|-----------------|------------|-----------|-----------------------------------------------|
| `weather`       | CSV        | 900 s     | Ambient temperature, solar irradiance, demand, PV output, electricity price |
| `grid`          | pandapower | 300 s     | 4-bus LV microgrid (PV, battery, heat pump, house load) |
| `battery`       | external   |  10 s     | Li-ion battery (13.5 kWh / 5 kW), Coulomb-counting SoC model |
| `heatpump`      | external   |  30 s     | Air-source heat pump, Carnot COP model (3 kW rated) |
| `thermal_tank`  | external   |  60 s     | 500 L hot-water storage tank, Euler integration |
| `greenhouse`    | external   |  60 s     | 200 m² greenhouse thermal envelope model |
| `controller`    | external   | 900 s     | Rule-based energy management system           |

This demonstrates **multi-rate** co-simulation (step sizes from 10 s to 900 s)
and **multi-energy** coupling (electrical ↔ thermal) with 16 signal connections
including fan-out (one output driving multiple inputs).

## Signal Flow

```
weather ──▸ T_ambient ──▸ heatpump, thermal_tank, greenhouse, controller
         ──▸ solar_irradiance ──▸ greenhouse
         ──▸ pv_power ──▸ grid, controller
         ──▸ elec_demand ──▸ grid
         ──▸ heat_demand ──▸ controller
         ──▸ elec_price ──▸ controller

controller ──▸ P_battery_cmd ──▸ battery
           ──▸ P_hp_cmd ──▸ heatpump

battery ──▸ P_actual ──▸ grid
        ──▸ SoC ──▸ controller

heatpump ──▸ Q_thermal ──▸ thermal_tank
         ──▸ P_elec ──▸ grid

thermal_tank ──▸ T_storage ──▸ heatpump (sink temp), controller

greenhouse ──▸ Q_demand ──▸ thermal_tank
           ──▸ T_inside ──▸ controller
```

## Files

| File                   | Purpose                                            |
|------------------------|----------------------------------------------------|
| `run_microgrid.py`     | Main co-simulation script                          |
| `generate_data.py`     | Creates `weather_data.csv` and `microgrid.p`       |
| `microgrid_cosim.ipynb`| Jupyter notebook with 4-subplot matplotlib analysis |
| `simulators/battery.py`| Li-ion battery model                               |
| `simulators/heatpump.py`| Air-source heat pump model                        |
| `simulators/thermal_tank.py`| Hot water storage tank model                  |
| `simulators/greenhouse.py`| Greenhouse thermal envelope model                |
| `simulators/controller.py`| Rule-based energy management system              |

## How to Run

```bash
# From the repository root
cd examples/microgrid
python run_microgrid.py
```

On the first run, `generate_data.py` is called automatically to create the
weather CSV and pandapower network file.

The simulation produces:

- **`es_res.h5`** — raw HDF5 results
- **`microgrid_dashboard.html`** — interactive Plotly dashboard (opens in browser)
- Console summary of per-simulator step counts and variable counts

Typical runtime: ~20 s for 96 macro steps (24 h at 900 s intervals).

## Requirements

- Python 3.10+
- `energysim`, `pandapower`, `numpy`, `pandas`
- For the notebook: `matplotlib`, `jupyter`

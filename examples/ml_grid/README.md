# ML-Based Distribution Grid Co-Simulation Example

Distribution-level power grid co-simulated with four neural-network-based
load and generation forecasters, all orchestrated by energysim.

## Scenario

A 24-hour weekday simulation of a 5-bus 20 kV distribution grid fed by
ML predictors:

| Simulator        | Type       | Step Size | Description                                    |
|------------------|------------|-----------|------------------------------------------------|
| `features`       | CSV        | 900 s     | Time-of-day (sin/cos), temperature, cloud cover, wind speed, weekend flag |
| `nn_residential` | external   |  60 s     | Feedforward NN [4→16→8→1] — residential load   |
| `nn_commercial`  | external   |  60 s     | Feedforward NN [4→16→8→1] — commercial load    |
| `nn_pv`          | external   |  60 s     | Feedforward NN [4→16→8→1] — PV generation      |
| `nn_wind`        | external   |  60 s     | Feedforward NN [4→16→8→1] — wind generation    |
| `grid`           | pandapower | 300 s     | 5-bus MV distribution grid (2 loads, 2 sgens)  |

This demonstrates how **machine-learning models** (pure numpy, no
framework dependency) can be embedded as external simulators and coupled
with physics-based power-flow through energysim's connection mechanism.

## Neural Network Architecture

Each NN is a small feedforward network trained on synthetic load/generation
profiles:

```
Input (4 features)
  │
  ▼
Dense 16, ReLU
  │
  ▼
Dense 8, ReLU
  │
  ▼
Dense 1, Linear → P_load or P_gen (MW)
```

Training uses MSE loss with Adam-style gradient descent (implemented in
pure numpy in `train_models.py`).  Weights are saved as `.npz` files and
loaded by each simulator at init time.

## Signal Flow

```
features ──▸ hour_sin, hour_cos ──▸ all 4 NN models
         ──▸ temperature ──▸ all 4 NN models
         ──▸ is_weekend ──▸ nn_residential, nn_commercial
         ──▸ cloud_cover ──▸ nn_pv
         ──▸ wind_speed ──▸ nn_wind

nn_residential ──▸ P_load ──▸ grid.Residential.p_mw
nn_commercial  ──▸ P_load ──▸ grid.Commercial.p_mw
nn_pv          ──▸ P_gen  ──▸ grid.SolarPark.p_mw
nn_wind        ──▸ P_gen  ──▸ grid.WindFarm.p_mw
```

## Files

| File                         | Purpose                                       |
|------------------------------|-----------------------------------------------|
| `run_ml_grid.py`             | Main co-simulation script                     |
| `generate_data.py`           | Creates `features.csv` and `distribution_grid.p` |
| `train_models.py`            | Trains 4 NN models, saves `.npz` weight files |
| `ml_grid_cosim.ipynb`        | Jupyter notebook with training + 4-subplot analysis |
| `simulators/nn_residential.py` | Residential load NN predictor               |
| `simulators/nn_commercial.py`  | Commercial load NN predictor                |
| `simulators/nn_pv.py`         | PV generation NN predictor                   |
| `simulators/nn_wind.py`       | Wind generation NN predictor                 |

## How to Run

```bash
# From the repository root
cd examples/ml_grid
python run_ml_grid.py
```

On the first run, `generate_data.py` and `train_models.py` are called
automatically to create the feature CSV, pandapower grid, and trained
weight files.

The simulation produces:

- **`es_res.h5`** — raw HDF5 results
- **`ml_grid_dashboard.html`** — interactive Plotly dashboard (opens in browser)
- Console summary of per-simulator step counts and variable counts

Typical runtime: ~23 s for 96 macro steps (24 h at 900 s intervals).

## Requirements

- Python 3.10+
- `energysim`, `pandapower`, `numpy`, `pandas`
- For the notebook: `matplotlib`, `jupyter`

# Interactive Jupyter Notebook Tutorials

`energysim` ships with a suite of 7 interactive Jupyter notebooks (`.ipynb`) located in [`examples/notebooks/`](https://github.com/dgusain1/energysim/tree/master/examples/notebooks), guiding you from first principles to advanced multi-carrier and multi-simulator system studies.

---

## Notebook Gallery Overview

| # | Notebook | Level | Focus & Skills Taught |
|---|---|---|---|
| **01** | [`01_quickstart_multi_energy_cosim.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/01_quickstart_multi_energy_cosim.ipynb) | **Beginner** | Core `world` orchestration, macro vs micro time-stepping, signals, BESS + PV co-simulation, DataFrame results extraction. |
| **02** | [`02_grid_cosim_pandapower_pypsa_opendss.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/02_grid_cosim_pandapower_pypsa_opendss.ipynb) | **Intermediate** | Multi-grid network coupling: pandapower AC distribution power flow, PyPSA regional dispatch, OpenDSS unbalanced circuits, automated `kW` $\leftrightarrow$ `MW` scaling. |
| **03** | [`03_custom_models_and_parametrization.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/03_custom_models_and_parametrization.ipynb) | **Intermediate** | Subclassing `ModelComponent`, building custom physical models (e.g. supercapacitors), using `Curve1D` / `Surface2D`, embedding `OperatingStateMachine`, parametrizing catalog models. |
| **04** | [`04_building_custom_simulator_adapters.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/04_building_custom_simulator_adapters.ipynb) | **Advanced** | Subclassing `SimulatorAdapter` to integrate external C-solvers, proprietary REST APIs, or hydraulic pipe networks; implementing rollback checkpoints with `save_state()` / `restore_state()`. |
| **05** | [`05_industrial_btm_complex.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/05_industrial_btm_complex.ipynb) | **Complex MES** | End-to-end behind-the-meter industrial refinery case study: 12 MW Solar PV + 40 MW CCGT + 10 MW PEM Electrolyzer + 350-bar H2 Tank + Tesla Megapack BESS + Industrial Steam Load. |
| **06** | [`06_vectorized_fleet_aggregation.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/06_vectorized_fleet_aggregation.ipynb) | **Performance** | High-performance fleet aggregation: simulating 10,000 residential batteries and 5,000 PV arrays in SIMD NumPy C-arrays with microsecond step benchmarks. |
| **07** | [`07_green_hydrogen_valley.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/07_green_hydrogen_valley.ipynb) | **Sector Coupling** | Integrated regional Green Hydrogen Valley: hybrid PEM + Alkaline + SOEC electrolyzers, waste heat recovery into district heat pump, H2 compression & storage, and Sabatier methanation (Power-to-Gas). |

---

## Running the Notebooks Locally

To run the interactive tutorials on your machine:

```bash
# Clone the repository
git clone https://github.com/dgusain1/energysim.git
cd energysim

# Install energysim with all optional simulator extensions
pip install -e .[all]
pip install jupyterlab matplotlib

# Launch JupyterLab
jupyter lab examples/notebooks/
```

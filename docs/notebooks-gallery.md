# Notebooks and examples

Twenty-one Jupyter notebooks (00-20) ship with the repository; see the
[notebook roadmap](notebooks-roadmap.md) for the assessment, the quality bar and the wave plan. Notebooks are authored as
percent-format sources in `scripts/notebook_src/` and built into `examples/notebooks/` with
`python scripts/build_notebooks.py` (use `--check` to syntax-check the sources without writing). Notebooks 01-07 were
produced by the legacy `scripts/generate_notebooks.py` and are being migrated to the new path. The notebooks are executed
top to bottom by the slow test suite (`pytest -m slow tests/docs/test_notebooks.py`); the example scripts are
smoke-tested by `tests/docs/test_examples_smoke.py`. They are tutorials, not validation: see [Validation](validation.md)
for the checked benchmarks.

## Notebooks (`examples/notebooks/`)

| # | Notebook | Level | What it shows | Extra dependencies |
|---|---|---|---|---|
| 00 | [`00_model_catalog_starter_guide.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/00_model_catalog_starter_guide.ipynb) | beginner | Overview of the physical model catalog: BESS, Flow Battery, Flywheel, PV, Wind, PEM Electrolyzer, Heat Pump. | matplotlib |
| 01 | [`01_quickstart_multi_energy_cosim.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/01_quickstart_multi_energy_cosim.ipynb) | beginner | The `world`, macro versus micro steps, signals, a PV plant with a battery, `Results` access and plots. | matplotlib |
| 02 | [`02_grid_cosim_pandapower_pypsa_opendss.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/02_grid_cosim_pandapower_pypsa_opendss.ipynb) | intermediate | A pandapower distribution feeder coupled with a PyPSA regional network, with kW / MW conversion between them. | pandapower, pypsa, matplotlib |
| 03 | [`03_custom_models_and_parametrization.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/03_custom_models_and_parametrization.ipynb) | intermediate | A custom `ModelComponent` (supercapacitor), `Curve1D` / `Surface2D` lookups, an `OperatingStateMachine`, re-parametrizing library models. See [Custom models](custom-models.md). | matplotlib |
| 04 | [`04_building_custom_simulator_adapters.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/04_building_custom_simulator_adapters.ipynb) | advanced | Subclassing `SimulatorAdapter` to wrap an external engine, here a hydraulic network. See [External simulators](external-simulators.md). | none |
| 05 | [`05_industrial_btm_complex.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/05_industrial_btm_complex.ipynb) | complex | A 24 h behind-the-meter industrial site: PV, battery, CCGT, PEM electrolyzer, hydrogen tank, process steam load. | matplotlib |
| 06 | [`06_vectorized_fleet_aggregation.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/06_vectorized_fleet_aggregation.ipynb) | performance | A 10 000-unit battery fleet advanced in one NumPy step and aggregated. See [Fleets](containers/overview.md). | matplotlib |
| 07 | [`07_green_hydrogen_valley.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/07_green_hydrogen_valley.ipynb) | sector coupling | PEM, alkaline and SOEC electrolyzers, waste heat into a heat pump, hydrogen compression and storage, methanation. | matplotlib |
| 08 | [`08_fleet_bess_physics.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/08_fleet_bess_physics.ipynb) | advanced | A battery fleet with OCV, internal resistance, thermal and SOH states, swept over C-rate and temperature, checked against the single-unit model. | none |
| 09 | [`09_fleet_ev_mobility.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/09_fleet_ev_mobility.ipynb) | advanced | A stochastic EV fleet with arrival and departure schedules, per-vehicle state of charge, smart versus uncontrolled charging. | none |
| 10 | [`10_fleet_heatpump_buildings.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/10_fleet_heatpump_buildings.ipynb) | advanced | A building fleet with heat pumps over a weather year, comfort-band violation counts, compared against the built-in model. | none |
| 11 | [`11_fleet_pv_smart_inverters.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/11_fleet_pv_smart_inverters.ipynb) | advanced | A PV fleet with volt-var and curtailment control, and the active / reactive power envelope against the inverter rating. | none |
| 12 | [`12_dso_feeder_with_fleets.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/12_dso_feeder_with_fleets.ipynb) | expert | A pandapower feeder hosting the fleets from 08-11, with voltage and loading checks and an `Ensemble` hosting-capacity search. | pandapower; networks for the sweep |
| 13 | [`13_district_heat_and_feeder.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/13_district_heat_and_feeder.ipynb) | expert | A district heat network, a building modelled in Octave and an OpenDSS feeder in one world, with energy balance closure across the three. | opendss, networks, Octave (guarded) |
| 14 | [`14_power_gas_cold_snap.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/14_power_gas_cold_snap.ipynb) | expert | A winter cold snap with gas-fired generation, a gas network with linepack, and unserved-gas, gas-shortfall-to-power, expected-energy-not-served and peak KPIs. | networks |
| 15 | [`15_curves_and_grid_solvers.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/15_curves_and_grid_solvers.ipynb) | expert | `Surface2D` operating maps (voltage against load and PV) from OpenDSS, PyPSA and Octave solvers on one grid, with a solver-agreement table. | opendss, pypsa, Octave (guarded) |
| 16 | [`16_fmu_with_builtin_physics.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/16_fmu_with_builtin_physics.ipynb) | expert | Repository FMUs coupled with built-in models, with explicit unit declarations (a power-to-heat FMU coupled with a built-in thermal store). | fmu (Windows-64 binaries, guarded) |
| 17 | [`17_gnu_octave_matlab_cosim.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/17_gnu_octave_matlab_cosim.ipynb) | intermediate | Setting up GNU Octave & MATLAB, function contract, and co-simulating `.m` models with energysim. | oct2py, GNU Octave or MATLAB |
| 18 | [`18_powerfactory_cosim_setup.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/18_powerfactory_cosim_setup.ipynb) | enterprise | DIgSILENT PowerFactory (2020+) integration guide, Python API matching, element variable syntax, and load flow co-simulation. | powerfactory (guarded/mocked) |
| 19 | [`19_online_fmus_and_fmi_standards.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/19_online_fmus_and_fmi_standards.ipynb) | intermediate | Downloading online reference FMUs from Modelica Cross-Check repositories, schema introspection, and co-simulation. | FMPy |
| 20 | [`20_simplified_vectorized_and_ensemble_modeling.ipynb`](https://github.com/dgusain1/energysim/blob/master/examples/notebooks/20_simplified_vectorized_and_ensemble_modeling.ipynb) | intermediate | Simplified high-performance modeling: 6-line vectorized DER fleets (`SimpleVectorizedModel`) and 1-line parameter sweeps (`Ensemble.sweep`). | matplotlib |

Running them locally:

```bash
git clone https://github.com/dgusain1/energysim.git
cd energysim
pip install -e ".[all]"          # the simulators and extras the notebooks use
pip install jupyterlab matplotlib
jupyter lab examples/notebooks/
```

Wave 1 notebooks are to be written to skip cells whose optional dependencies are missing and print a message; the
"Extra dependencies" column lists what each notebook needs. Notebook 02 guards its pandapower and PyPSA imports.

## Example scripts (`examples/`)

| Directory | Scenario |
|---|---|
| `microgrid/` | A 24 h prosumer microgrid: dynamic models plus a pandapower grid and CSV weather / demand. |
| `community_microgrid/` | Community microgrid with battery, building envelopes, PV and HVAC under a dispatcher. |
| `mpc_microgrid/` | A residential microgrid dispatched by model predictive control, using all the simulator kinds, including MATLAB / Octave. |
| `ml_grid/` | A 5-bus distribution grid with neural-network load and generation forecasters as simulators. |
| `matlab_cosim/` | Three `.m` function models (heat pump, battery, thermal mass); needs MATLAB or Octave. |
| `industrial_btm/` | The industrial site of notebook 05 as a script. |
| `optimal_dispatch/`, `Optimal Dispatch of FER/` | Dispatch of flexible resources (power-to-heat, power-to-gas) with an OPF controller / FMUs. |
| `pf_ninebus/` | A PowerFactory 9-bus system with flexible loads; needs a PowerFactory licence. |
| `scenarios/microgrid_scenario.yaml` | A minimal declarative scenario: `energysim run examples/scenarios/microgrid_scenario.yaml`. |

Each directory has its own README with the run command. The smoke test skips examples that need a proprietary engine
(PowerFactory, MATLAB / Octave) when it is not installed.

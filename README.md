# energysim

<p align="center">
  <img src="brand/energysim-full-light.svg#gh-light-mode-only" alt="energysim - Co-Simulation of Multi-Energy Systems" width="460">
  <img src="brand/energysim-full-dark.svg#gh-dark-mode-only" alt="energysim - Co-Simulation of Multi-Energy Systems" width="460">
</p>

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B%20%7C%203.14-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**Python-based co-simulation tool for multi-energy systems.**

Documentation: [energysim.readthedocs.io](https://energysim.readthedocs.io)

## What is energysim?

`energysim` is a high-performance Python framework for **multi-energy system (MES) modeling, sector coupling, and heterogeneous co-simulation**.

It bridges electrical power networks with **built-in physical models for green hydrogen electrolyzers, gas turbines, battery storage, and district heating**, alongside bridges for external simulators (pandapower, PyPSA, OpenDSS, PowerFactory, FMUs, MATLAB, Octave) through a single `world` orchestrator with automatic unit scaling and dependency resolution.

### Key Capabilities

<!-- features:start -->
<!-- Generated from features.yaml by scripts/generate_readme_features.py. Do not edit by hand. -->

**Simulator adapters**

| Capability | Status | Description |
|---|---|---|
| [CSV profile adapter](https://energysim.readthedocs.io/en/latest/add-simulator/) | stable | Replay time series from a CSV file as a simulator (zero-order hold, strict column checks). |
| [Signal generators](https://energysim.readthedocs.io/en/latest/add-simulator/) | stable | Python callables and safe expression strings as signal sources. |
| [Python script adapter](https://energysim.readthedocs.io/en/latest/add-simulator/) | stable | Wrap a user Python module as a simulator (disabled in safe mode). |
| [External simulator interface](https://energysim.readthedocs.io/en/latest/external-simulators/) | stable | Plug any object or factory implementing the adapter contract into a world. |
| [FMU co-simulation (FMI 1.0 / 2.0)](https://energysim.readthedocs.io/en/latest/add-simulator/) | stable | Run Co-Simulation FMUs through FMPy with start values, feed-through and re-runs. |
| [FMU model exchange](https://energysim.readthedocs.io/en/latest/add-simulator/) | stable | Integrate Model-Exchange FMUs with CVode, including time/state events. |
| [pandapower networks](https://energysim.readthedocs.io/en/latest/add-simulator/) | stable | AC/DC power flow and OPF through pandapower, validated against direct runpp. |
| [PyPSA networks](https://energysim.readthedocs.io/en/latest/add-simulator/) | stable | Single-snapshot online power flow / dispatch of PyPSA networks. |
| [OpenDSS distribution feeders](https://energysim.readthedocs.io/en/latest/add-simulator/) | stable | Unbalanced AC distribution feeders via OpenDSSDirect.py (DC mode rejected). |
| [DIgSILENT PowerFactory](https://energysim.readthedocs.io/en/latest/add-simulator/) | beta | AC / DC load flow through the PowerFactory Python API; CI uses a mock, a licensed run is manual. |
| [MATLAB / GNU Octave functions](https://energysim.readthedocs.io/en/latest/add-simulator/) | beta | Call .m functions through the MATLAB Engine or Octave (oct2py), auto-detected. |
| [Uniform adapter contract](https://energysim.readthedocs.io/en/latest/architecture/) | stable | Every adapter kind passes one contract suite and is interchangeable inside a world. |

**Coupling and numerics**

| Capability | Status | Description |
|---|---|---|
| [Jacobi coupling](https://energysim.readthedocs.io/en/latest/architecture/) | stable | Default parallel coupling; every stage sees the previous macro step's outputs. |
| [Gauss-Seidel coupling](https://energysim.readthedocs.io/en/latest/architecture/) | stable | Sequential coupling in topological order so downstream models see fresh upstream outputs. |
| [Algebraic-loop detection](https://energysim.readthedocs.io/en/latest/architecture/) | stable | Feedback loops are detected, warned about and resolved with a one-step delay. |
| [Input extrapolation](https://energysim.readthedocs.io/en/latest/architecture/) | stable | Zero-order hold or linear extrapolation of coupled inputs between macro steps. |
| [Events and variable macro step](https://energysim.readthedocs.io/en/latest/events/) | stable | Scheduled events, threshold triggers and adapter step hints; the macro step shrinks to land on events (t_macro_min). |

**Units**

| Capability | Status | Description |
|---|---|---|
| [Unit-aware connections](https://energysim.readthedocs.io/en/latest/units/) | stable | Declared or inferred units are converted automatically across connections; incompatible units raise. |
| [Unit policy](https://energysim.readthedocs.io/en/latest/units/) | stable | A variable with no determinable unit is never scaled (a warning by default, an error with unit_policy='strict'); a manual scale disables conversion. |

**Scenarios, validation and safety**

| Capability | Status | Description |
|---|---|---|
| [Pre-run validation](https://energysim.readthedocs.io/en/latest/yaml-cli/) | stable | world.validate() reports unknown names, unit clashes and step-size problems before running. |
| [Command-line interface](https://energysim.readthedocs.io/en/latest/yaml-cli/) | stable | `energysim validate` and `energysim run` with stable exit codes and --out / --safe flags. |
| [Safe mode](https://energysim.readthedocs.io/en/latest/safe-mode/) | stable | Refuses scripts, external code and dotted imports when running untrusted scenario files. |
| [Declarative YAML scenarios](https://energysim.readthedocs.io/en/latest/yaml-cli/) | stable | Describe a world in YAML, load it with world.from_yaml and round-trip it with to_yaml. |
| [Glass-box access](https://energysim.readthedocs.io/en/latest/architecture/) | stable | get_adapter / get_model expose the live objects inside a world. |

**Results and visualisation**

| Capability | Status | Description |
|---|---|---|
| [Results object](https://energysim.readthedocs.io/en/latest/results/) | stable | Typed per-simulator tables with units and run metadata; nothing is written to disk by default. |
| [Save and load formats](https://energysim.readthedocs.io/en/latest/results/) | stable | Parquet (CSV fallback), CSV and HDF5 export with a meta.json and reload via Results.load. |
| [KPIs and summaries](https://energysim.readthedocs.io/en/latest/results/) | stable | Energy integrals, time-weighted means and user-registered KPIs. |
| [Interactive HTML dashboard](https://energysim.readthedocs.io/en/latest/results/) | stable | Self-contained offline HTML dashboard generated from a Results object. |
| [Ensembles and parameter sweeps](https://energysim.readthedocs.io/en/latest/ensemble/) | stable | Run many scenario variants (grids, samplers, noise) and collect them in one Results object. |

**Built-in physical models**

| Capability | Status | Description |
|---|---|---|
| [Storage models](https://energysim.readthedocs.io/en/latest/components/bess/) | stable | BatteryBESS (Rint ECM + thermal + aging), FlowBattery (VRFB) and Flywheel. |
| [Renewable models](https://energysim.readthedocs.io/en/latest/components/pv/) | stable | PhysicsPV (NOCT cell temperature, inverter clipping), WindTurbine and SolarThermalCollector. |
| [Hydrogen and PtX models](https://energysim.readthedocs.io/en/latest/components/pem_electrolyzer/) | stable | PEM / alkaline / SOEC electrolyzers, H2 compressor and tank, PEM / SOFC fuel cells, methanation. |
| [Thermal generation models](https://energysim.readthedocs.io/en/latest/components/ocgt/) | stable | OCGT with start-up timers, CCGT (fixed GT/ST split, part-load heat rate), reciprocating genset and CHP steam turbine. |
| [Heating, cooling and thermal storage models](https://energysim.readthedocs.io/en/latest/components/heat_pump/) | stable | Industrial heat pump (Carnot-lift COP), electric boiler, electric / absorption chillers, thermal storage tank. |
| [Carbon capture](https://energysim.readthedocs.io/en/latest/components/carbon_capture/) | stable | Amine-based post-combustion CarbonCapture_Amine. |
| [Industrial and data-centre loads](https://energysim.readthedocs.io/en/latest/components/datacenter/) | stable | DataCenterLoad and IndustrialProcessSteam demand models. |
| [Model framework](https://energysim.readthedocs.io/en/latest/custom-models/) | stable | ModelComponent base with describe(), strict variable access, units on every variable and deterministic re-runs. |
| [Equipment archetypes](https://energysim.readthedocs.io/en/latest/components/catalog/) | stable | Generic documented archetypes (BESS_2h_2MW, PEM_5MW, ...) with provenance; brand-named aliases are deprecated. |

**Vectorized fleets**

| Capability | Status | Description |
|---|---|---|
| [Vectorized fleet containers](https://energysim.readthedocs.io/en/latest/containers/overview/) | stable | Battery, PV and building fleets of N units advanced in one NumPy step, equal to a loop of scalar models. |
| [Custom vectorized fleets](https://energysim.readthedocs.io/en/latest/containers/custom_fleet/) | stable | Subclass VectorizedContainerAdapter or pass an update function to VectorizedFunctionAdapter. |

**Energy networks**

| Capability | Status | Description |
|---|---|---|
| [Energy networks](https://energysim.readthedocs.io/en/latest/networks/) | stable | Producer / consumer / storage graphs with an LP balance, pipe limits, merit order and losses. |
| [Heat and gas networks](https://energysim.readthedocs.io/en/latest/networks/) | stable | HeatNetwork (transport delay, temperatures) and GasNetwork (MW_LHV / Nm3/h units, linepack). |

**Control and optimisation**

| Capability | Status | Description |
|---|---|---|
| [Economic MPC](https://energysim.readthedocs.io/en/latest/control/) | beta | Receding-horizon LP/MILP controller over library blocks (storage, electrolyzer, heat pump, ...). |
| [OPF and PyPSA controllers](https://energysim.readthedocs.io/en/latest/control/) | beta | pandapower OPF and PyPSA rolling-horizon controllers wrapped as energysim controllers. |
| [Forecast providers](https://energysim.readthedocs.io/en/latest/control/) | beta | Perfect, persistence, callable and seeded noisy forecasts for controllers. |

**Quality assurance**

| Capability | Status | Description |
|---|---|---|
| [Physical validation suite](https://energysim.readthedocs.io/en/latest/validation/) | stable | Analytic, textbook and cross-tool benchmarks with sourced tolerances, published in the docs. |
| [Clean-install packaging](https://energysim.readthedocs.io/en/latest/faq/) | stable | The wheel ships py.typed and installs and imports cleanly in a fresh venv. |

**Planned and removed**

| Capability | Status | Description |
|---|---|---|
| [Iterative (fixed-point) coupling](https://energysim.readthedocs.io/en/latest/architecture/) | removed | Removed in 3.0 because it relied on rollback that adapters never supported; world() now raises. |
| [State rollback API](https://energysim.readthedocs.io/en/latest/architecture/) | removed | save_state / restore_state were removed; no adapter implements rollback. |

Status legend: `stable` and `beta` features are covered by the test suite listed in `features.yaml`; `planned` features are not implemented yet; `removed` ones raise a clear error.
<!-- features:end -->

## Architecture

All simulator adapters inherit from the `SimulatorAdapter` abstract base
class (in `energysim.base`). The `world` orchestrator interacts *only*
through this interface:

| Method                              | Description                                       |
|-------------------------------------|---------------------------------------------------|
| `configure(start, stop, macro_step)` + `init()` | Called at the start of every run, before the first step. |
| `set_parameters(params)`            | Apply initial values / parameters (default: `set_value`). |
| `get_available_variables()`         | Introspection: `{"inputs": [...], "outputs": [...]}`. |
| `step(time)`                        | Advance the model by one micro-step.              |
| `advance(start, stop)`              | Advance from *start* to *stop* (loops `step()`).  |
| `get_value(parameters, time)`       | Query output variables.                           |
| `set_value(parameters, values)`     | Set input variables.                              |
| `step_partial(time, dt)`            | Shorter final micro-step when an event shortens a macro step. |
| `max_step(time)`, `next_event_time(time)` | Optional step-size hints for the variable macro step. |
| `cleanup()`                         | Release resources after the simulation.           |

Coupling is `'jacobi'` (default) or `'gauss-seidel'`; feedback loops are solved with a one-macro-step delay (reduce
`t_macro` or merge the coupled components). Fixed-point (iterative) coupling and state rollback were
removed in 3.0. See [Migrating to 3.0](https://energysim.readthedocs.io/en/latest/migration-3.0/).

## Installation

Install the core package:

```bash
pip install energysim   # or: uv add energysim
```

Only **numpy**, **pandas**, **tqdm**, **networkx** and **pyyaml** are required. Everything else is an
optional extra:

<!-- extras:start -->
<!-- Generated from pyproject.toml by scripts/generate_readme_features.py. Do not edit by hand. -->

With pip:

```bash
pip install energysim            # core: numpy, pandas, tqdm, networkx, pyyaml
pip install energysim[fmu]       # adds fmpy>=0.3.27,<0.4
pip install energysim[powerflow] # adds pandapower>=3.0,<4
pip install energysim[pypsa]     # adds pypsa>=0.30,<2
pip install energysim[opendss]   # adds opendssdirect.py>=0.9,<1
pip install energysim[plotting]  # adds matplotlib
pip install energysim[networks]  # adds scipy
pip install energysim[control]   # adds scipy>=1.9
pip install energysim[results]   # adds pyarrow
pip install energysim[hdf5]      # adds tables
pip install energysim[all]       # everything above
```

With uv:

```bash
uv add energysim
uv add energysim[fmu]       # adds fmpy>=0.3.27,<0.4
uv add energysim[powerflow] # adds pandapower>=3.0,<4
uv add energysim[pypsa]     # adds pypsa>=0.30,<2
uv add energysim[opendss]   # adds opendssdirect.py>=0.9,<1
uv add energysim[plotting]  # adds matplotlib
uv add energysim[networks]  # adds scipy
uv add energysim[control]   # adds scipy>=1.9
uv add energysim[results]   # adds pyarrow
uv add energysim[hdf5]      # adds tables
uv add energysim[all]       # everything above
```
<!-- extras:end -->

Adapters that are not pip extras:

- **PowerFactory**: requires DIgSILENT PowerFactory 2020+ with the Python API enabled. No extra pip package needed.
- **MATLAB**: requires the MATLAB Engine for Python.
- **Octave**: `pip install oct2py` / `uv add oct2py` + GNU Octave on PATH.

## CLI Runner & Declarative YAML

Run co-simulations directly from YAML without writing Python scripts:

```bash
# Validate scenario pre-flight
energysim validate scenario.yaml

# Run simulation and open interactive dashboard
energysim run scenario.yaml --dashboard
```

## Quick Start (Zero-Setup Minimal Example)

Run a complete multi-domain co-simulation in Python with zero external file dependencies:

```python
import pandapower as pp
from energysim import world

# 1. Create a minimal distribution network in memory
net = pp.create_empty_network(name='microgrid')
b1 = pp.create_bus(net, vn_kv=20.0, name='Bus1')
b2 = pp.create_bus(net, vn_kv=20.0, name='Bus2')
pp.create_ext_grid(net, b1, name='GridSlack')
pp.create_line(net, b1, b2, length_km=2.5, std_type='NAYY 4x150 SE', name='Feeder')
pp.create_load(net, b2, p_mw=1.2, q_mvar=0.3, name='Load1')

# 2. Initialize the co-simulation coordinator (macro step = 900s)
sim = world(start_time=0, stop_time=3600, t_macro=900, power_unit='kW')

# 3. Register the grid model and a dynamic controller signal
sim.add_powerflow('grid', net=net, inputs=['Load1.p_mw'], outputs=['Bus2.vm_pu'], step_size=900, mode='ac')
sim.add_signal('load_controller', lambda t: [800.0 + 200.0 * (t / 3600)], units={'y': 'kW'})

# 4. Connect signal to load (automatically converts 800 kW -> 0.8 MW)
sim.add_connections({'load_controller.y': 'grid.Load1.p_mw'})

# 5. Run co-simulation and view DataFrame results
results = sim.simulate()

print(results['grid.Bus2.vm_pu'])     # pandas Series indexed by time (s)
```

More workflows (each with runnable examples in the documentation):

- **Events and variable macro step**: `world.schedule(t, {'set': {...}})`, `world.add_trigger(...)`,
  `world(t_macro_min=...)` - [Events](https://energysim.readthedocs.io/en/latest/events/).
- **Ensembles and sweeps**: `Ensemble`, samplers, `noisy_signal`, `Sweep` -
  [Ensembles](https://energysim.readthedocs.io/en/latest/ensemble/).
- **Untrusted scenario files**: `world.from_yaml(path, safe=True)` / `energysim run --safe` -
  [Safe mode](https://energysim.readthedocs.io/en/latest/safe-mode/).
- **Units**: every connection is converted into the receiver's unit; unspecified units are never scaled
  (`unit_policy='strict'` makes them an error) - [Units](https://energysim.readthedocs.io/en/latest/units/).

## Parameters

<!-- params:start -->
<!-- Generated from the signatures by scripts/generate_readme_features.py. Do not edit by hand. -->

### `world()` parameters

| Parameter | Default | Description |
|---|---|---|
| `start_time` | `0` | Simulation start time in seconds. |
| `stop_time` | `1000` | Simulation end time in seconds; `stop_time - start_time` must be a multiple of `t_macro` (unless `t_macro_min < t_macro`). |
| `logging` | `False` | Print INFO progress messages of the `energysim` logger. |
| `t_macro` | `60` | Macro time-step (variable exchange interval); `0` means the LCM of all simulator step sizes. |
| `coupling` | `'jacobi'` | Coupling strategy: `'jacobi'` or `'gauss-seidel'`. |
| `extrapolation` | `'zero-order'` | Input extrapolation between macro steps: `'zero-order'` or `'linear'`. |
| `power_unit` | `None` | World default power unit used to resolve unit-less power variables. |
| `start_datetime` | `None` | Optional wall-clock start; results are then indexed by datetime. |
| `seed` | `None` | Optional random seed, recorded in the results metadata. |
| `out_dir` | `None` | If set, every run is written to `<out_dir>/<timestamp>_<hash>/`. |
| `unit_system` | `None` | `None`, `'default'`, `'si'`, an overrides dict or a `UnitSystem`. |
| `unit_policy` | `'lenient'` | `'lenient'` (default; unspecified units pass through with a warning) or `'strict'` (unspecified units on connections raise). |
| `t_macro_min` | `None` | Smallest macro step when adapters report step hints or events occur (default: `t_macro`, i.e. fixed step). |

### `add_simulator()` parameters

| Parameter | Default | Description |
|---|---|---|
| `sim_type` | `''` | `'fmu'`, `'powerflow'`, `'pypsa'`, `'opendss'`, `'csv'`, `'signal'`, `'external'`, `'powerfactory'`, `'matlab'`, `'script'`, `'model'` or a fleet kind (`'battery_fleet'`, ...). |
| `sim_name` | `''` | Unique identifier for this simulator. |
| `sim_loc` | `None` | Path / object of the model (alias of `model`; many other aliases such as `net`, `path` exist). |
| `inputs` | `None` | List of input variable names. |
| `outputs` | `None` | List of output variable names to record. |
| `step_size` | `1` | Micro time-step. |
| `mode` | `'ac'` | Power-flow mode: `'ac'` or `'dc'`. |
| `power_unit` | `None` | Power unit of this simulator's variables (default: the world's `power_unit`). |
| `units` | `None` | Optional `{variable: unit}` mapping that overrides inferred units. |
| `model` | `None` | Model object, file path or factory. |
| `allow_import` | `False` | Allow dotted-import model references (`'pandapower.networks.case14'`); refused in safe mode. |

Type-specific keyword arguments:

- **FMU**: `validate` (bool), `variable` (bool), `solver_name`
- **Power flow** (`pandapower`, `pypsa`, `opendss`): `mode` - `'ac'` or `'dc'`
- **CSV**: `delimiter` (str)
- **PowerFactory**: `mode` - `'ac'` or `'dc'` (load flow); `pf_path` (str)
- **MATLAB / Octave**: `engine` - `'auto'`, `'matlab'` or `'octave'`
<!-- params:end -->

## Results & Dashboard

`sim.simulate()` returns a `Results` object (also kept as
`sim.last_results`). Nothing is written to disk unless you ask for it:

```python
res = sim.simulate()                  # `sim` from the Quick Start above
res['grid']                   # DataFrame of one simulator (index: time)
res['grid.Bus2.vm_pu']        # Series of one variable
res.to_frame(resample='15min')  # wide frame, columns 'sim.var'
res.summary(); res.kpis()     # statistics, energy integrals (kWh), ...
res.save('out')               # Parquet (or CSV) + meta.json; Results.load('out')
res.dashboard('my_dashboard.html', open=True)
```

To keep every run on disk, create the world with `world(..., out_dir='runs')`; each run is then written to
`runs/<YYYYmmdd-HHMMSS>_<hash8>/`.

## Citing

Please cite the following paper if you use **energysim**:

> Gusain, D., Cvetkovic, M. & Palensky, P. (2019). *Energy flexibility
> analysis using FMUWorld.* 2019 IEEE Milan PowerTech.
> [doi:10.1109/PTC.2019.8810433](https://doi.org/10.1109/PTC.2019.8810433)

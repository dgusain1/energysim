# energysim

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**Python-based co-simulation tool for multi-energy systems.**

Documentation: [energysim.readthedocs.io](https://energysim.readthedocs.io)

## What is energysim?

`energysim` is a Python-based co-simulation tool designed to simplify
multi-energy co-simulations. It lets you couple heterogeneous energy-system
models — FMUs, power-flow networks, CSV data sources, PowerFactory models,
MATLAB/Octave functions, and custom Python simulators — through a single
`world` object that handles time progression, variable exchange, and results
collection automatically.

### Supported simulators

1. **Functional Mockup Units** (FMI 1.0 / 2.0, Co-Simulation & Model Exchange)
2. **Pandapower** networks — AC/DC power flow and OPF
3. **PyPSA** networks (experimental)
4. **CSV** data files with time-indexed columns
5. User-defined **external simulators** via a Python class
6. **DIgSILENT PowerFactory** models via the Python API
7. **MATLAB / GNU Octave** `.m` functions (auto-detects engine)
8. **Signal generators** (Python lambda / function)
9. Standalone **Python scripts**

## Architecture

All simulator adapters inherit from the `SimulatorAdapter` abstract base
class (in `energysim.base`). The `world` orchestrator interacts *only*
through this interface:

| Method                              | Description                                       |
|-------------------------------------|---------------------------------------------------|
| `init()`                            | Called once before the first time step.            |
| `step(time)`                        | Advance the model by one micro-step.              |
| `advance(start, stop)`              | Advance from *start* to *stop* (loops `step()`).  |
| `get_value(parameters, time)`       | Query output variables.                           |
| `set_value(parameters, values)`     | Set input variables.                              |
| `cleanup()`                         | Release resources after the simulation.           |
| `save_state()` / `restore_state()`  | State snapshot for iterative coupling.            |

## Installation

Install the core package:

```bash
pip install energysim   # or: uv add energysim
```

Only **numpy**, **pandas**, **tqdm**, **networkx**, and **PyTables** are
required. Simulator-specific packages are optional:

=== "pip"

    ```bash
    pip install energysim[fmu]         # adds FMPy
    pip install energysim[powerflow]   # adds pandapower
    pip install energysim[pypsa]       # adds PyPSA
    pip install energysim[plotting]    # adds matplotlib
    pip install energysim[all]         # all of the above
    ```

=== "uv"

    ```bash
    uv add energysim[fmu]         # adds FMPy
    uv add energysim[powerflow]   # adds pandapower
    uv add energysim[pypsa]       # adds PyPSA
    uv add energysim[plotting]    # adds matplotlib
    uv add energysim[all]         # all of the above
    ```

For the new adapters:

- **PowerFactory**: requires DIgSILENT PowerFactory 2020+ with the Python API enabled. No extra pip package needed.
- **MATLAB**: requires the MATLAB Engine for Python.
- **Octave**: `pip install oct2py` / `uv add oct2py` + GNU Octave on PATH.

## Quick Start

```python
from energysim import world

# 1. Create co-simulation world
my_world = world(
    start_time=0, stop_time=86400,
    logging=True, t_macro=900,
    coupling='jacobi',
    extrapolation='zero-order',
)

# 2. Add simulators
my_world.add_simulator(
    sim_type='fmu', sim_name='plant',
    sim_loc='plant.fmu',
    inputs=['power_cmd'], outputs=['temperature'],
    step_size=1, validate=False)

my_world.add_simulator(
    sim_type='powerflow', sim_name='grid',
    sim_loc='network.p',
    inputs=['Load1.p_mw'], outputs=['Bus1.vm_pu'],
    step_size=300, pf='opf')

my_world.add_simulator(
    sim_type='csv', sim_name='profiles',
    sim_loc='data.csv', step_size=900)

# 3. Connect simulators
connections = {
    'plant.temperature': 'grid.Load1.p_mw',
    'profiles.wind':     'plant.power_cmd',
}
my_world.add_connections(connections)

# 4. Run
my_world.simulate(pbar=True)

# 5. Results (opens interactive HTML dashboard)
results = my_world.results(to_csv=False, dashboard=True)
```

## Parameters

### `world()` parameters

| Parameter                | Default        | Description                                             |
|--------------------------|----------------|---------------------------------------------------------|
| `start_time`             | `0`            | Simulation start time in seconds.                       |
| `stop_time`              | `1000`         | Simulation end time in seconds.                         |
| `logging`                | `True`         | Print progress messages.                                |
| `t_macro`                | `60`           | Macro time-step (variable exchange interval).           |
| `coupling`               | `'jacobi'`     | Coupling strategy: `'jacobi'`, `'gauss-seidel'`, `'iterative'`. |
| `extrapolation`          | `'zero-order'` | Extrapolation method: `'zero-order'` or `'linear'`.    |
| `max_iterations`         | `10`           | Max iterations per macro step (iterative only).         |
| `convergence_tolerance`  | `1e-6`         | Convergence threshold (iterative only).                 |

### `add_simulator()` parameters

| Parameter    | Default  | Description                                          |
|--------------|----------|------------------------------------------------------|
| `sim_type`   | —        | `'fmu'`, `'powerflow'`, `'csv'`, `'external'`, `'powerfactory'`, `'matlab'`, `'script'`. |
| `sim_name`   | —        | Unique identifier for this simulator.                |
| `sim_loc`    | —        | Path to the model / data file.                       |
| `inputs`     | —        | List of input variable names.                        |
| `outputs`    | —        | List of output variable names to record.             |
| `step_size`  | `1e-3`   | Micro time-step.                                     |

Type-specific keyword arguments:

- **FMU**: `validate` (bool), `variable` (bool)
- **Powerflow**: `pf` — `'pf'`, `'dcpf'`, `'opf'`, or `'dcopf'`
- **CSV**: `delimiter` (str)
- **PowerFactory**: `pf` — `'ldf'`, `'shc'`, or `'rms'`; `pf_path` (str)
- **MATLAB/Octave**: `engine` — `'auto'`, `'matlab'`, or `'octave'`

## Results & Dashboard

`my_world.results()` returns a `dict[str, DataFrame]` keyed by simulator
name. It can also open an interactive HTML dashboard:

```python
results = my_world.results(
    to_csv=False,
    dashboard=True,
    dashboard_path='my_dashboard.html',
)
```

## Citing

Please cite the following paper if you use **energysim**:

> Gusain, D., Cvetkovic, M. & Palensky, P. (2019). *Energy flexibility
> analysis using FMUWorld.* 2019 IEEE Milan PowerTech.
> [doi:10.1109/PTC.2019.8810433](https://doi.org/10.1109/PTC.2019.8810433)

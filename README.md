# energysim

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B%20%7C%203.14-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**Python-based co-simulation tool for multi-energy systems.**

Documentation: [energysim.readthedocs.io](https://energysim.readthedocs.io)

## What is energysim?

`energysim` is a high-performance Python framework for **multi-energy system (MES) modeling, sector coupling, and heterogeneous co-simulation**.

It bridges electrical power networks with **built-in physical models for green hydrogen electrolyzers, gas turbines, battery storage, and district heating**, alongside bridges for external simulators (pandapower, PyPSA, OpenDSS, PowerFactory, FMUs, MATLAB, Octave) through a single `world` orchestrator with automatic unit scaling and dependency resolution.

### Key Capabilities

1. **Inbuilt Multi-Energy Physical Components**:
   - **Storage**: `BatteryBESS` (LFP/NMC ECM + thermal + aging), `FlowBattery` (VRFB), `Flywheel`.
   - **Renewables**: `PhysicsPV` (5-param + King NOCT), `WindTurbine`, `SolarThermalCollector`.
   - **Hydrogen & PtX**: `PEMElectrolyzer`, `AlkalineElectrolyzer` (20% turndown cutoff), `SOECElectrolyzer` (steam electrolysis), `H2Compressor`, `H2StorageTank` (350 bar real-gas), `FuelCell_PEM`, `FuelCell_SOFC`, `MethanationReactor`.
   - **Thermal & Turbomachinery**: `GasTurbine_OCGT` (Cold/Warm/Hot startup timers), `CombinedCycle_CCGT` (HRSG + Rankine steam), `ReciprocatingGenset`, `CHP_SteamTurbine`.
   - **Heating, Cooling & Carbon**: `IndustrialHeatPump` (Carnot lift COP), `ElectricBoiler`, `Electric/Absorption Chillers`, `CarbonCapture_Amine`.
   - **Equipment Catalog**: Ready-to-use archetypes from OEM datasheets (`catalog.Tesla_Megapack_2XL`, `catalog.Nel_A1000`, `catalog.GE_7HA_02`, `catalog.ITM_Power_Trident`, etc.).
2. **Heterogeneous Co-Simulation Adapters**:
   - **Pandapower** networks — AC/DC power flow and OPF
   - **PyPSA** networks (Pattern B single-snapshot online dispatch)
   - **OpenDSS** distribution networks via `OpenDSSDirect.py`
   - **Functional Mockup Units** (FMI 1.0 / 2.0 Co-Simulation & Model Exchange)
   - **DIgSILENT PowerFactory** models via Python API
   - **MATLAB / GNU Octave** `.m` functions (auto-detects engine)
   - **CSV profiles**, **signal generators**, and custom Python simulators.

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

Only **numpy**, **pandas**, **tqdm**, **tables**, **networkx**, and **pyyaml** are
required. Simulator-specific packages are optional:

=== "pip"

    ```bash
    pip install energysim[fmu]         # adds FMPy
    pip install energysim[powerflow]   # adds pandapower
    pip install energysim[pypsa]       # adds PyPSA
    pip install energysim[opendss]     # adds OpenDSSDirect.py
    pip install energysim[plotting]    # adds matplotlib
    pip install energysim[all]         # all of the above
    ```

=== "uv"

    ```bash
    uv add energysim[fmu]         # adds FMPy
    uv add energysim[powerflow]   # adds pandapower
    uv add energysim[pypsa]       # adds PyPSA
    uv add energysim[opendss]     # adds OpenDSSDirect.py
    uv add energysim[plotting]    # adds matplotlib
    uv add energysim[all]         # all of the above
    ```

For the new adapters:

- **OpenDSS**: `pip install energysim[opendss]` (cross-platform).
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
sim.add_signal('load_controller', lambda t: [800.0 + 200.0 * (t / 3600)], power_unit='kW')

# 4. Connect signal to load (automatically converts 800 kW -> 0.8 MW)
sim.add_connections({'load_controller.y': 'grid.Load1.p_mw'})

# 5. Run co-simulation and view DataFrame results
sim.simulate()
results = sim.results(dashboard=False)

print(results['grid'][['time', 'Bus2.vm_pu']])
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

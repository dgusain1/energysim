# Declarative YAML & CLI Runner

`energysim` supports declarative simulation specification via YAML or JSON. This enables reproducible, headless, cluster-friendly simulation execution without writing imperative Python code.

---

## Command-Line Interface (CLI)

The `energysim` package installs a native CLI entry point:

```bash
# Validate scenario pre-flight without running
energysim validate scenario.yaml

# Execute simulation
energysim run scenario.yaml

# Execute simulation with interactive dashboard and CSV results
energysim run scenario.yaml --dashboard --to-csv
```

### CLI Command Reference

| Flag / Option | Description |
| :--- | :--- |
| `scenario` | Positional argument: path to `.yaml` or `.yml` scenario file. |
| `--dashboard` | Opens the interactive HTML dashboard after completion. |
| `--dashboard-path <path>` | Specify custom HTML destination file path. |
| `--to-csv` | Exports time-series dataframes to CSV files. |
| `--res-filename <file>` | Overrides HDF5 results storage path. |
| `--no-pbar` | Disables console progress bar. |
| `--validate-only` | Performs semantic and graph validation checks, then exits (code `0` on success, `1` on error). |

---

## Scenario File Specification (`scenario.yaml`)

```yaml
world:
  start_time: 0
  stop_time: 86400              # Simulation time in seconds
  t_macro: 900                  # Macro synchronization time-step in seconds
  coupling: jacobi              # Options: 'jacobi', 'gauss-seidel', 'iterative'
  extrapolation: zero-order     # Options: 'zero-order', 'linear'
  power_unit: kW                # Global baseline power unit (default: 'kW')
  res_filename: results.h5
  max_iterations: 10            # Iterative coupling convergence iterations
  convergence_tolerance: 1.0e-6
  logging: true

simulators:
  - name: plant
    type: fmu                   # 'fmu', 'powerflow', 'pypsa', 'opendss', 'csv', 'signal', 'matlab'
    model: models/chp.fmu       # File path, in-memory object, or factory string
    power_unit: kW
    inputs: [gas_flow]
    outputs: [electric_power, heat_flow]
    step_size: 1.0
    options:
      validate: false

  - name: grid
    type: powerflow
    model: models/network.p     # Or Python dotted benchmark e.g. pandapower.networks.case14
    mode: ac                    # Standardized calculation mode: 'ac' or 'dc'
    inputs: [Gen1.p_mw]
    outputs: [Bus1.vm_pu]
    step_size: 300.0

  - name: solar_profile
    type: csv
    model: data/solar.csv
    step_size: 900.0
    options:
      delimiter: ","

connections:
  # Connection with automatic physical unit scaling (e.g. kW to MW):
  - from: plant.electric_power
    to: grid.Gen1.p_mw
    from_unit: kW
    to_unit: MW

  # Or explicit linear scaling and offset: y = x * scale + offset
  - from: plant.heat_flow
    to: grid.Load1.p_mw
    scale: 0.001
    offset: 0.0
```

---

## Python API for YAML

You can also load and export YAML scenarios directly in Python:

```python
from energysim import world

# Load and instantiate world from YAML file
my_world = world.from_yaml("scenario.yaml")

# Run pre-flight semantic validation
report = my_world.validate(raise_on_error=True)

# Run simulation
my_world.simulate()

# Export an existing world instance to YAML
my_world.to_yaml("exported_scenario.yaml")
```

---

## Pre-flight Semantic Validation

Before spending computational effort on complex simulations, `energysim` audits the simulation graph:

1. **Step-size Divisibility**: Emits warnings if simulator micro-step sizes do not divide the macro synchronization interval $t_{macro}$ evenly.
2. **Missing Simulator / Endpoint Detection**: Identifies misspelled or unregistered simulators before execution begins.
3. **Read-Only Simulator Protection**: Warns if data sources (e.g. CSV or signal generators) are targeted as inputs.
4. **Algebraic Feedback Loops**: Analyzes the dependency graph with Tarjan's Strongly Connected Components (SCC) algorithm. Advises on coupling strategy (e.g. recommending `iterative` mode for cyclic graphs).

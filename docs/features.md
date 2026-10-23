# Features

This page documents features beyond the basic workflow described on the
[home page](index.md).

---

## Coupling modes { #coupling-modes }

`energysim` supports three coupling strategies, selected via the
`coupling` parameter when creating the `world` object.

### Jacobi coupling (default)

```python
my_world = world(coupling='jacobi', ...)
```

All variables are exchanged first, then all simulators are stepped.
This introduces a **one-step delay** in feedback loops — each simulator
sees the *previous* macro-step's outputs from its dependencies.

**Best for:** loosely coupled systems, parallel-friendly setups.

### Gauss-Seidel coupling

```python
my_world = world(coupling='gauss-seidel', ...)
```

Simulators are stepped one at a time in **topological order**. After
each simulator steps, its outputs are immediately available to
downstream simulators. This eliminates the coupling delay for
feed-forward signal chains.

Within algebraic loops (cyclic dependencies), Gauss-Seidel falls back
to Jacobi coupling for the involved simulators.

**Best for:** predominantly feed-forward topologies, chains of sequential
models.

### Iterative coupling

```python
my_world = world(
    coupling='iterative',
    max_iterations=10,
    convergence_tolerance=1e-6,
)
```

Each macro step is repeated (iterated) until the exchanged variables
converge:

1. Save the state of all simulators.
2. Perform a Gauss-Seidel step.
3. Compare current outputs to the previous iteration.
4. If converged (max relative change < `convergence_tolerance`), accept the step.
5. Otherwise, restore the saved state and repeat (up to `max_iterations`).

!!! note
    Iterative coupling requires adapters that support `save_state()`
    and `restore_state()`. FMU Co-Simulation adapters implement this
    via `getFMUstate()` / `setFMUstate()` for FMI 2.0 FMUs with
    `canGetAndSetFMUstate=true`. External simulators can implement
    these methods by saving and restoring their internal state variables.

**Best for:** tightly coupled systems with algebraic loops (e.g. a
controller and plant with mutual feedback).

---

## Extrapolation

The `extrapolation` parameter controls how exchanged values are
estimated between macro time-steps:

- **`'zero-order'`** (default) — hold the last exchanged value constant (zero-order hold).
- **`'linear'`** — linearly extrapolate from the two most recent values.

```python
my_world = world(
    coupling='gauss-seidel',
    extrapolation='linear',
    t_macro=900,
)
```

Linear extrapolation can improve accuracy when variables change smoothly
between exchange points, but may cause instability if variables are noisy
or discontinuous.

---

## Connection validation

`add_connections()` automatically validates all connections:

- **Simulator existence** — raises `ConnectionError` if a simulator name in the connection string is not registered.
- **Variable name checking** — warns if a variable name is not found in the adapter's known variables (via `get_available_variables()`).
- **Read-only targets** — warns if a connection writes to a CSV or signal simulator, since those adapters discard incoming values.

Example of a validation error:

```python
my_world.add_connections({'sim1.y': 'nonexistent_sim.x'})
# → ConnectionError: Simulator 'nonexistent_sim' not found
```

Example of a validation warning:

```python
my_world.add_connections({'signal.y': 'csv_data.temperature'})
# → UserWarning: Simulator 'csv_data' is read-only (type='csv').
#   Connection to 'temperature' will be silently ignored.
```

### Pre-flight semantic validation

Before running a long simulation, you can audit the entire simulation topology using `my_world.validate()`:

```python
report = my_world.validate(raise_on_error=False)
if not report.is_valid:
    for err in report.errors:
        print(f"ERROR: {err}")
for warn in report.warnings:
    print(f"WARNING: {warn}")
```

Audits performed:
- **Step-size Divisibility**: Emits warnings if simulator micro-step sizes do not divide the macro synchronization interval `t_macro` evenly.
- **Unregistered Simulators & Read-Only Targets**: Detects unknown simulators or connections targeting read-only data sources.
- **Algebraic Loops & Coupling Strategy**: Analyzes strongly connected components (SCCs) and verifies that appropriate coupling (`iterative` or `jacobi`) is configured when cycles exist.

By default, `my_world.simulate(validate=True)` runs this check automatically at startup.

### Physical units abstraction & inline scaling

`energysim` features an engineering units abstraction layer with a default baseline unit of **`'kW'`**. It automatically resolves unit conversions across domains without requiring manual scaling factors.

#### Automatic unit conversion

When simulators have known or inferable units (for example, OpenDSS kW, Pandapower/PyPSA MW, and FMU W), `energysim` infers the physical dimension and scales values transparently during macro time-step exchanges:

- **Power**: `W`, `kW`, `MW`, `GW`
- **Reactive Power**: `var`, `kvar`, `mvar`
- **Voltage**: `V`, `kV`, `mV`
- **Energy**: `Wh`, `kWh`, `MWh`, `GWh`, `J`, `kJ`, `MJ`

#### Explicit connection units

You can explicitly declare source and target units on any connection:

```python
# Convert from kW (chp) to MW (grid) automatically
my_world.add_connections({
    'chp.power': {'to': 'grid.Gen1.p_mw', 'from_unit': 'kW', 'to_unit': 'MW'}
})
```

#### Simulator-level unit declaration

You can specify the default power unit or custom variable unit mappings per simulator:

```python
my_world.add_simulator(
    sim_type='fmu',
    sim_name='chp',
    sim_loc='chp.fmu',
    power_unit='kW',
    units={'heat_flow': 'kW', 'gas_consumption': 'm3/h'},
    outputs=['electric_power', 'heat_flow'],
)
```

#### Inline scaling and offsets ($y = x \cdot \text{scale} + \text{offset}$)

For custom unit systems or non-standard variables, pass explicit `scale` and `offset`:

```python
# Explicit custom scaling (bypasses automatic unit scaling)
my_world.add_connections({
    'chp.power_w': {'to': 'grid.Gen1.p_mw', 'scale': 1e-6, 'offset': 0.0}
})
```

---

## Dependency graph & algebraic loops

After connections are added, `energysim` builds a directed dependency
graph using NetworkX. Simulators are automatically sorted in
**topological execution order** so that upstream simulators are stepped
before downstream ones.

If the dependency graph contains **cycles** (algebraic loops),
`energysim`:

1. Warns about the loop, listing the involved simulators.
2. Uses **strongly-connected-component (SCC) condensation** to find a safe execution order.
3. Within each SCC, uses Jacobi coupling (regardless of the global coupling mode).

You can inspect the computed execution order:

```python
my_world.add_connections(connections)
# Prints: "Execution order: ['weather', 'pv', 'battery', 'grid']"
```

---

## Variable introspection

Some adapters support introspection of their available variables via
`get_available_variables()`. This is used during connection
validation and can also be called directly:

```python
entry = my_world.simulator_dict["grid"]
print(entry.adapter.get_available_variables())
# {'inputs': ['Gen1.p_mw', 'Load1.p_mw', ...],
#  'outputs': ['Bus1.vm_pu', 'Bus1.va_degree', ...]}
```

Currently implemented in:

- **ppAdapter** — returns element.variable names from gen, load, sgen, bus, and ext_grid tables.
- **csAdapter** — returns all variable names from the FMU's model description.
- **csv_adapter** — returns column names from the CSV file.
- **pypsa_adapter** — returns bus, generator, load, and branch variable names.
- **opendss_adapter** — returns bus, load, generator, line, and circuit metrics.
- **pyScriptAdapter** — returns configured inputs and outputs.

---

## Adding signals

The `add_signal()` method attaches user-defined time-varying (or
constant) signals to the co-simulation. This is useful when a simulator
input needs a constant set-point or a mathematical function of time:

```python
def my_signal(time):
    return [1]

my_world.add_signal(
    sim_name='constant_signal',
    signal=my_signal,
    step_size=1)
```

The return value **must** be a single-element list. More complex
signals are possible:

```python
import numpy as np

def sine_signal(time):
    return [np.sin(2 * np.pi * time / 3600)]

my_world.add_signal(sim_name='sine', signal=sine_signal, step_size=1)
```

In the connections dictionary:

```python
connections = {'constant_signal.y': 'sim1.input_variable1'}
my_world.add_connections(connections)
```

!!! note
    Signal simulators are **read-only**. If a connection writes to a
    signal simulator, `set_value()` emits a `UserWarning` and
    discards the value.

---

## Modify signals before exchange

Sometimes an output needs unit conversion before being passed to another
simulator. `energysim` supports this via the `modify_signal` option:

1. **Multiply by a constant:** `'sim1.var1': [x]` multiplies by *x*.
2. **Multiply and offset:** `'sim1.var1': [x1, x2]` computes `value * x1 + x2`.

Example — convert Watts to MW:

```python
modifications = {'chp.e_power': [1 / 1e6]}
options = {'modify_signal': modifications}
my_world.options(options)
```

---

## AC and DC Power Flow Modes

`energysim` focuses strictly on time-series dynamic co-simulation and standardizes power flow calculations across all supported power grid engines (Pandapower, PyPSA, OpenDSS, DIgSILENT PowerFactory) to two canonical modes:

- **`mode='ac'`** (default) — Full non-linear AC power flow (e.g. Newton-Raphson in Pandapower/PowerFactory, snapshot in OpenDSS, AC non-linear in PyPSA).
- **`mode='dc'`** — Linearized DC power flow for fast active-power approximations.

```python
# Configure DC power flow
my_world.add_simulator(
    sim_type='powerflow',
    sim_name='grid',
    sim_loc='network.p',
    inputs=['WF.p_mw'],
    outputs=['Bus1.vm_pu'],
    step_size=300,
    mode='dc',
)
```

!!! note "Scope of Co-Simulation"
    Higher-level static optimizations (such as OPF/LOPF) and specialized dynamic transient solvers (such as electromagnetic RMS or short-circuit calculations) are intentionally excluded from the core time-step synchronization loop. Co-simulation coordinates dynamic multi-energy exchanges; if optimization is required, external supervisory controllers or PyScript adapters should be coupled into the simulation loop.

---

## Validation of FMUs

By default, FMPy validates FMUs during initialisation. To skip
validation (useful for faster startup or FMUs that fail validation but
simulate correctly):

```python
my_world.add_simulator(
    sim_type='fmu', sim_name='plant',
    sim_loc='plant.fmu',
    inputs=['cmd'], outputs=['temp'],
    step_size=1, validate=False)
```

---

## Recording all time-steps

By default, `energysim` records output values once per macro
time-step. To capture values at every micro-step, use
`record_all=True`:

```python
my_world.options({'record_all': True})
```

This calls `adapter.advance_with_recording()` instead of
`adapter.advance()`. For most adapters, this records at every
`step_size` interval. For FMU Model Exchange, the ODE solver's
adaptive internal time points are used, providing higher-fidelity
traces without artificially constraining the solver.

---

## Sensitivity analysis

Parameter sweeps can be performed by updating the `init` option in a
loop:

```python
for v1, v2 in [(0.1, 0.2), (1, 2), (10, 20)]:
    inits = {
        'sim1': (['param_a'], [v1]),
        'sim2': (['param_b'], [v2]),
    }
    my_world.options({'init': inits})
    my_world.simulate(pbar=False)
    # extract and store results
```

A dedicated `Sweep` class is also available for automated parameter
sweeps with built-in plotting:

```python
from energysim import Sweep

sweep = Sweep(my_world, sensitivity_info, kind='single')
sweep.export_to_csv('sweep_results.csv')
```

---

## PowerFactory integration

DIgSILENT PowerFactory models (2020+) can be coupled into
co-simulations using the `'powerfactory'` sim type. The adapter
supports AC load flow (`mode='ac'`) and DC load flow (`mode='dc'`)
calculation modes (mapped to PowerFactory `ComLdf`).

Variable naming convention:

- **Inputs** — use the PowerFactory *input* attribute prefix `e:`, e.g. `Load.plini` (the adapter auto-prepends `e:` if missing).
- **Outputs** — use the *result* attribute prefix `m:`, e.g. `Bus1.m:u`, `Line1.m:loading`.

The element name before the dot must match the object's `loc_name`
in PowerFactory.

```python
my_world.add_simulator(
    sim_type='powerfactory',
    sim_name='IEEE13',
    sim_loc='IEEE13',
    inputs=['Load1.plini', 'Load2.plini'],
    outputs=['Bus1.m:u', 'Line1.m:loading'],
    step_size=900,
    mode='ac',
    pf_path=r'C:\DIgSILENT\PowerFactory 2024\Python\3.11')
```

!!! warning
    PowerFactory must be installed and licensed on the machine. The
    `pf_path` argument is only needed if the PowerFactory Python
    directory is not already on `sys.path`.

---

## MATLAB / Octave integration

MATLAB `.m` **functions** (not scripts) can be integrated via the
`'matlab'` sim type. The adapter auto-detects whether to use the
MATLAB Engine for Python or GNU Octave (via `oct2py`).

Every function must follow this contract:

```matlab
function [out1, out2] = my_model(in1, in2, time)
```

Rules:

- `time` is always the **last** input argument — `energysim` appends it automatically.
- The number of declared inputs (minus `time`) must match the `inputs` list.
- The number of declared outputs must match the `outputs` list.
- The file name (without `.m`) must match the function name.

Use **persistent** variables to keep state across time-steps (see the
battery and building examples in `examples/matlab_cosim/`).

```python
my_world.add_simulator(
    sim_type='matlab',
    sim_name='heatpump',
    sim_loc=r'models\heatpump.m',
    inputs=['P_electric', 'T_source', 'T_sink'],
    outputs=['Q_thermal', 'COP'],
    step_size=900,
    engine='auto')   # 'matlab', 'octave', or 'auto'
```

!!! note
    Install `oct2py` (`pip install oct2py` / `uv add oct2py`) and ensure
    GNU Octave is on PATH, or install the MATLAB Engine for Python from
    your MATLAB installation directory.

---

## Interactive dashboard

After simulation, `results()` can open an interactive HTML dashboard
in the browser:

```python
results = my_world.results(dashboard=True)
```

The dashboard features:

- **Variable browser** — tree view of all simulators and their variables.
- **Drag-and-drop** — drag variables onto subplots or click the **+** button.
- **Multiple subplots** — add, remove, and rename subplots.
- **Layout control** — switch between 1, 2, or 3 column layouts.
- **Auto-overview** — one-click plot of all variables grouped by simulator.
- **Export** — save the dashboard HTML file to share with others.

You can specify a custom path for the HTML file:

```python
results = my_world.results(
    dashboard=True,
    dashboard_path='my_dashboard.html')
```

!!! note
    The dashboard requires `plotly`. Install it with `pip install plotly`
    or `uv add plotly`.

---

## System topology plot

`energysim` can visualise the co-simulation topology as a network
graph. The `plot()` method reuses the internal dependency graph
(built during `add_connections()`):

```python
my_world.plot(plot_edge_labels=False, node_size=300, node_color='r')
```

If `add_connections()` has not been called yet, `plot()` will build
the graph on the fly from the current connections.

---

## Error handling

All simulator errors now raise proper Python exceptions instead of
calling `sys.exit()`:

| Exception                         | When raised                                                 |
|-----------------------------------|-------------------------------------------------------------|
| `ConnectionError`                 | Malformed connection string, missing simulator name         |
| `SimulatorVariableError`          | Invalid variable name or type in an adapter                 |
| `SimulatorElementNotFoundError`   | Named element not found in a network                        |
| `ValidationError`                 | World pre-flight semantic or graph check failure            |
| `EnergysimError`                  | General energysim error (base class)                        |

These can be caught and handled in user code:

```python
from energysim.base import SimulatorVariableError

try:
    my_world.simulate()
except SimulatorVariableError as e:
    print(f"Variable error: {e}")
```

---

## Accessing simulator internals ("Glass-Box" Co-Simulation)

`energysim` follows a "Glass-Box" co-simulation philosophy: while it handles the orchestration, the underlying domain-specific models remain directly accessible to users before, during, or after execution.

### Direct Glass-Box Accessors

Use `world.get_model()` and `world.get_adapter()`:

```python
# Access the native engine/network model directly
net = my_world.get_model("grid")           # Returns pandapowerNet or pypsa.Network
dss = my_world.get_model("distribution")   # Returns OpenDSSDirect module
app = my_world.get_model("powerfactory")    # Returns DIgSILENT Application object
df  = my_world.get_model("weather")         # Returns pandas DataFrame

# Access the simulator adapter wrapper
adapter = my_world.get_adapter("grid")
```

| Simulator Type | `world.get_model(name)` returns | Description |
| :--- | :--- | :--- |
| `'powerflow'` | `pandapowerNet` | Native Pandapower network structure |
| `'pypsa'` | `pypsa.Network` | Native PyPSA network object |
| `'opendss'` | `OpenDSSDirect` module | Direct OpenDSS interface handle |
| `'powerfactory'` | `Application` object | DIgSILENT PowerFactory root COM / Python object |
| `'csv'` | `pandas.DataFrame` | Loaded time-series profile data |
| `'fmu'` | `fmi2Component` | Active FMU simulation handle |
| `'matlab'` | `MatlabEngine` or `Oct2Py` | Underlying MATLAB / Octave engine session |
| `'script'` | Python module / callable | Executed script namespace |
| `'signal'` | `callable` | Signal generation function |
| `'container'` | `VectorizedContainerAdapter` | Underlying state/input NumPy dictionaries |

### Simulator Registry (`simulator_dict`)

Each registered simulator is also stored in `simulator_dict` as a `SimEntry` dataclass:

| Attribute         | Description                                              |
|-------------------|----------------------------------------------------------|
| `.adapter`        | The `SimulatorAdapter` instance.                         |
| `.outputs`        | List of output variable names.                           |
| `.inputs`         | List of input variable names (populated from connections).|
| `.step_size`      | Micro time-step.                                         |
| `.sim_type`       | `'fmu'`, `'powerflow'`, `'pypsa'`, `'opendss'`, `'container'`, etc. |
| `.mode`           | Standardized power-flow mode (`'ac'` or `'dc'`).         |
| `.power_unit`     | Default power unit for this simulator (e.g. `'kW'`, `'MW'`). |
| `.units`          | Mapping of variable names to specific physical units.    |
| `.variable_step`  | Whether variable stepping is enabled (FMU only).         |

---

## High-performance mass fleets (Vectorized Containers)

For simulating thousands of edge devices (such as 1,000 EVs or 500 heat pumps) without coordinator bottleneck or multi-process overhead, use `energysim`'s built-in **Vectorized Container Simulators**.

See the comprehensive [Vectorized Containers guide](containers/overview.md) for architecture, vector operations, and examples.


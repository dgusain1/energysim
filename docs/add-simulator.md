# Adding Simulators & Adapters

Simulators and models are added to the `world` co-simulation canvas using `add_simulator()`, or through specialized helper methods: `add_powerflow()`, `add_pypsa()`, `add_opendss()`, `add_model()`, `add_container()`, and `add_signal()`.

Ten simulator families are natively supported: **FMU (FMI 1.0/2.0)**, **pandapower (powerflow)**, **PyPSA**, **OpenDSS**, **CSV Data Feeds**, **External User Adapters**, **DIgSILENT PowerFactory**, **MATLAB/Octave**, **Script Functions**, and **Vectorized Container Fleets**.

---

## The Modern `model` vs `sim_loc` Interface

`energysim` features a unified model registration interface that supports both in-memory Python objects and filesystem paths:

* **`model`** (Recommended Modern Argument):
  Accepts in-memory model instances (such as a pandapower `net`, a PyPSA `Network`, a `ModelComponent` instance, or a Python callable) OR file path strings (`.fmu`, `.p`, `.dss`, `.csv`, `.m`).
* **`sim_loc`** (Backward-Compatible Alias):
  Accepts raw filesystem strings or directory paths. If both are passed, `model` takes precedence.

### Common Parameters Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sim_type` | `str` | - | Simulator type identifier (`'powerflow'`, `'pypsa'`, `'opendss'`, `'fmu'`, `'csv'`, `'external'`, `'powerfactory'`, `'matlab'`, `'script'`, `'container'`, or `'model'`). |
| `sim_name` | `str` | - | Unique identifier string used in connection mappings and results DataFrames. |
| `model` | `Any` | `None` | Model reference: in-memory network object, component instance, file path string, or callable factory. |
| `sim_loc` | `str` | `None` | Backward-compatible alias for `model` (raw string address of simulator file or directory). |
| `inputs` | `List[str]` | `[]` | List of settable variable names available to receive connections from other simulators or signals. |
| `outputs` | `List[str]` | `[]` | List of observable variable names to record in simulation results at each time-step. |
| `step_size` | `float` | `1.0` | Internal integration micro-step size in seconds. |
| `mode` | `str` | `'ac'` | Calculation mode for power flow engines (`'ac'` or `'dc'`). |
| `power_unit` | `str` | `'kW'` | Engineering unit for power exchanges (`'kW'`, `'MW'`, or `'W'`). |
| `units` | `Dict[str, str]` | `None` | Optional per-variable unit dictionary mapping (e.g. `{'P_ac': 'MW', 'T_cell': 'degC'}`). |

---

## Auto-Routing with `world.add_model()`

For physical component models inheriting from `ModelComponent` or `SimulatorAdapter` (e.g., BESS, Solar PV, PEM Electrolyzer, CCGT), you can use the direct `add_model()` convenience method:

```python
from energysim import world
from energysim.models.storage import BatteryBESS

bess = BatteryBESS(name="bess", capacity_mwh=4.0, p_max_mw=2.0)

sim = world(start_time=0, stop_time=3600, t_macro=900)
# Auto-routes model and detects inputs/outputs automatically
sim.add_model("bess", model=bess, outputs=["P_actual", "SoC"])
```

---

## FMU simulators

FMUs conforming to FMI 1.0 or 2.0 are supported. `energysim`
auto-detects whether the FMU is **Co-Simulation** or **Model Exchange**
and loads the appropriate adapter.

Additional keyword arguments:

| Parameter   | Default  | Description                                                              |
|-------------|----------|--------------------------------------------------------------------------|
| `validate`  | `True`   | Let FMPy validate the FMU before initialisation. Set to `False` for FMUs that fail validation but simulate correctly (common with OpenModelica exports). |
| `variable`  | `False`  | Enable variable-step integration (Co-Simulation only).                   |

```python
my_world.add_simulator(
    sim_type='fmu', sim_name='electrolyser',
    sim_loc='electrolyser.fmu',
    inputs=['P_order', 'T_ambient'],
    outputs=['H2_production', 'efficiency'],
    step_size=1,
    validate=False)
```

!!! note
    FMPy is only imported when the first FMU is added, so `energysim`
    works without FMPy installed as long as no FMU simulators are used.

### Variable naming convention (FMU)

FMU variables follow the Modelica dot-notation:

```
sim_name.Component.SubComponent.Variable
```

For example: `electrolyser.stack.voltage`.

---

## Powerflow simulators

Pandapower networks (saved as `.p` pickle files) can be added directly.

Additional keyword arguments:

| Parameter | Default | Description                                             |
|-----------|---------|---------------------------------------------------------|
| `mode`    | `'ac'`  | Power-flow mode: `'ac'` (AC power flow) or `'dc'` (DC power flow) |

```python
my_world.add_simulator(
    sim_type='powerflow', sim_name='grid',
    sim_loc='network.p',
    inputs=['WF.p_mw', 'Load1.p_mw'],
    outputs=['Bus 0.vm_pu', 'WF.p_mw', 'Load1.p_mw'],
    step_size=300,
    mode='ac')
```

!!! warning
    All load, generator, sgen, bus, and external-grid elements in the
    pandapower network **must have names**. `energysim` matches
    variables by element name, not by index. If a named element is not
    found, a `SimulatorElementNotFoundError` is raised. If a variable
    attribute is invalid, a `SimulatorVariableError` is raised.

The pandapower adapter implements `get_available_variables()` for
connection validation — see [Features](features.md).

### Variable naming convention (powerflow)

Variables are specified as `ElementName.variable`:

**Inputs** (values that can be set):

- Generators: `Gen1.p_mw`, `Gen1.q_mvar`
- Loads: `Load1.p_mw`, `Load1.q_mvar`
- Sgens: `PV.p_mw`, `WF.p_mw`
- Cost: `Load1.cp1_eur_per_mw` (for OPF)
- Controllability bounds: `Load1.max_p_mw`, `Load1.min_p_mw`

**Outputs** (values that can be recorded):

- Bus voltages: `Bus1.vm_pu`, `Bus1.va_degree`
- Generator results: `Gen1.p_mw`, `Gen1.q_mvar`
- Load results: `Load1.p_mw`, `Load1.q_mvar`
- Sgen results: `PV.p_mw`
- External grid: `grid.p_mw`, `grid.q_mvar`

!!! note
    For OPF, loads that should be dispatchable must have
    `controllable=True` set in the pandapower network. You can do
    this after adding the simulator:

### Standalone Runnable Example (pandapower AC Power Flow)

```python
import pandapower as pp
from energysim import world

# 1. Create a 3-bus network in memory
net = pp.create_empty_network(name="feeder")
b1 = pp.create_bus(net, vn_kv=20.0, name="Bus1")
b2 = pp.create_bus(net, vn_kv=20.0, name="Bus2")
b3 = pp.create_bus(net, vn_kv=20.0, name="Bus3")
pp.create_ext_grid(net, b1, name="SlackGrid")
pp.create_line(net, b1, b2, length_km=2.0, std_type="NAYY 4x150 SE", name="Line1")
pp.create_line(net, b2, b3, length_km=1.5, std_type="NAYY 4x150 SE", name="Line2")
pp.create_load(net, b2, p_mw=0.8, q_mvar=0.2, name="FactoryLoad")
pp.create_sgen(net, b3, p_mw=0.0, q_mvar=0.0, name="SolarPark")

# 2. Setup world canvas (kW basis with automatic MW scaling)
sim = world(start_time=0, stop_time=3600, t_macro=900, power_unit="kW")

# 3. Add simulator with in-memory net
sim.add_powerflow(
    sim_name="grid",
    model=net,
    inputs=["FactoryLoad.p_mw", "SolarPark.p_mw"],
    outputs=["Bus2.vm_pu", "Bus3.vm_pu", "Line1.loading_percent", "SlackGrid.p_mw"],
    step_size=900,
    mode="ac",
)

# 4. Inject dynamic time signals
sim.add_signal("solar_signal", lambda t: 1200.0 * (t / 3600.0), power_unit="kW")
sim.add_connections({"solar_signal.y": "grid.SolarPark.p_mw"})

# 5. Simulate
sim.simulate(dashboard=False)
results = sim.results(dashboard=False)
print(results["grid"].head())
```

---

## CSV simulators

CSV files provide time-indexed data profiles (e.g. weather, demand).

Additional keyword arguments:

| Parameter   | Default | Description        |
|-------------|---------|--------------------|
| `delimiter` | `','`   | Column delimiter.  |

Requirements:

- The CSV **must** have a `time` column.
- Time values must be at fixed intervals.

```python
my_world.add_simulator(
    sim_type='csv', sim_name='profiles',
    sim_loc='data.csv',
    step_size=900)
```

Variables are accessed by column name: `profiles.wind_power`,
`profiles.temperature`, etc.

!!! note
    CSV simulators are **read-only**. If a connection writes to a CSV
    simulator, `set_value()` emits a `UserWarning` and discards the
    value. The CSV adapter implements `get_available_variables()` to
    return all column names for connection validation.

---

## External simulators

User-defined Python simulators can be integrated by subclassing
`SimulatorAdapter` from `energysim.base`. See
[External Simulators](external-simulators.md) for the full guide.

```python
my_world.add_simulator(
    sim_type='external', sim_name='my_sim',
    sim_loc='/path/to/directory',
    inputs=['cmd'], outputs=['result'],
    step_size=60)
```

The `sim_name` must match the Python filename (without `.py`), and
the file must contain a class called `external_simulator`.

---

## PowerFactory simulators

DIgSILENT PowerFactory models can be integrated using the
`'powerfactory'` sim type. This requires PowerFactory 2020+ with
the Python API and a valid licence.

Additional keyword arguments:

| Parameter | Default  | Description                                                                              |
|-----------|----------|------------------------------------------------------------------------------------------|
| `mode`    | `'ac'`   | Calculation mode: `'ac'` (AC load flow, default) or `'dc'` (DC load flow)                |
| `pf_path` | `None`   | Path to the PowerFactory Python directory.                                               |

```python
# Generic registration using model='IEEE13' (PowerFactory project name)
my_world.add_simulator(
    sim_type='powerfactory',
    sim_name='grid_pf',
    model='IEEE13',
    inputs=['Load1.plini', 'Gen1.pgini'],
    outputs=['Bus1.m:u', 'Bus2.m:u', 'Line1.m:I:bus1'],
    step_size=900,
    mode='ac',
    pf_path=r'C:\DIgSILENT\PowerFactory 2024\Python\3.11')

# Or dedicated method:
my_world.add_powerfactory(
    sim_name='grid_pf',
    project_name='IEEE13',
    outputs=['Bus1.m:u'],
    step_size=900,
)
```

!!! tip "Project Activation"
    `model` (or `project_name`) specifies the PowerFactory **project name** to activate in the database. If omitted, `sim_name` is used by default.

### Variable naming convention (PowerFactory)

Variables are specified as `ElementName.attribute`:

**Inputs** (written via `SetAttribute`):

- Load active power: `Load1.plini` (auto-prefixed to `e:plini`)
- Generator active power: `Gen1.pgini`
- Any editable attribute: `Element.e:attributeName`

**Outputs** (read via `GetAttribute`):

- Bus voltage magnitude: `Bus1.m:u`
- Bus voltage angle: `Bus1.m:phiu`
- Line current: `Line1.m:I:bus1`
- Generator output: `Gen1.m:P:bus1`

Result attributes start with `m:`. For inputs, the `e:` prefix is
added automatically if omitted.

---

## MATLAB / Octave simulators

MATLAB `.m` **functions** (not scripts) can be integrated using the
`'matlab'` sim type. The adapter auto-detects whether to use the
MATLAB Engine for Python or GNU Octave (via `oct2py`).

Additional keyword arguments:

| Parameter | Default  | Description                                                      |
|-----------|----------|------------------------------------------------------------------|
| `engine`  | `'auto'` | Which backend: `'auto'` (try MATLAB first), `'matlab'`, `'octave'` |

```python
my_world.add_simulator(
    sim_type='matlab',
    sim_name='heatpump',
    sim_loc=r'/path/to/m_files',
    inputs=['P_cmd', 'T_source', 'T_sink'],
    outputs=['Q_thermal', 'COP'],
    step_size=30,
    engine='auto')
```

!!! warning
    `sim_name` must match the `.m` filename (without extension).
    The file must define a **function** (not a script).

### Writing .m functions for energysim

Your `.m` function must follow this signature:

```matlab
function [out1, out2, ...] = func_name(in1, in2, ..., time)
```

- **Input arguments** map to the `inputs` list **in order**, plus `time` (appended automatically by the adapter).
- **Output arguments** map to the `outputs` list **in order**.

The adapter validates at init time that:

1. The `.m` file is a function (not a script).
2. `nargin` matches `len(inputs) + 1` (the +1 is for `time`).
3. `nargout` matches `len(outputs)`.

#### Example: heatpump.m

```matlab
function [Q_thermal, COP] = heatpump(P_cmd, T_source, T_sink, time)
% HEATPUMP  Simple Carnot heat-pump model for energysim.
%
%   Inputs:
%     P_cmd    - commanded electrical power [kW]
%     T_source - source temperature [deg C]
%     T_sink   - sink temperature [deg C]
%     time     - simulation time [s] (unused here)
%
%   Outputs:
%     Q_thermal - thermal output [kW]
%     COP       - coefficient of performance [-]

    COP_carnot = (T_sink + 273.15) / max(T_sink - T_source, 0.1);
    COP = 0.45 * COP_carnot;
    Q_thermal = P_cmd * COP;
end
```

#### Example: battery.m

```matlab
function [SoC, P_actual] = battery(P_cmd, time)
% BATTERY  Simple Li-ion battery with Coulomb counting.

    persistent soc_state;
    if isempty(soc_state)
        soc_state = 0.5;  % start at 50%
    end

    capacity_kwh = 13.5;
    max_power = 5.0;
    dt = 30;

    P_actual = max(-max_power, min(max_power, P_cmd));
    soc_state = soc_state - (P_actual * dt / 3600) / capacity_kwh;
    soc_state = max(0.05, min(0.95, soc_state));
    SoC = soc_state;
end
```

!!! note
    Functions can use `persistent` variables to maintain state between
    steps — the MATLAB/Octave equivalent of Python instance variables.

---

## PyPSA Simulators (Pattern B Online Dispatch)

PyPSA networks (imported from CSV folders) can be simulated using `sim_type='pypsa'`.

```python
my_world.add_simulator(
    sim_type='pypsa',
    sim_name='transmission',
    sim_loc='/path/to/pypsa_csv_folder',
    inputs=['Generator1.P', 'Load1.P'],
    outputs=['Bus1.V', 'Generator1.P'],
    step_size=900,
    mode='ac'   # 'ac' for AC power flow, 'dc' for DC power flow (lpf)
)
```

!!! warning "Design Characteristic & Limitation (Pattern B)"
    The `energysim` PyPSA adapter operates strictly in **Pattern B (Single Snapshot Mode)**.
    At each co-simulation macro time step, the PyPSA network evaluates a single snapshot.
    Multi-period lookahead time series optimization (e.g. continuous multi-day optimization
    with inter-temporal storage state linkages) is solved sequentially snapshot-by-snapshot
    within the co-simulation loop rather than solving the entire multi-period time-horizon
    simultaneously.

Supported modes:
- `mode='ac'`: Non-linear AC power flow (`lpf()` + `pf()`).
- `mode='dc'`: Linear DC power flow (`lpf()`).

### Standalone Runnable Example (PyPSA In-Memory Dispatch)

```python
import pypsa
from energysim import world

# 1. Build a 2-bus PyPSA network programmatically
n = pypsa.Network()
n.add("Bus", "Bus0", v_nom=110.0)
n.add("Bus", "Bus1", v_nom=110.0)
n.add("Line", "Branch0", bus0="Bus0", bus1="Bus1", x=0.05, r=0.01, s_nom=200.0)
n.add("Generator", "Gen0", bus="Bus0", p_nom=150.0, marginal_cost=25.0)
n.add("Load", "Load1", bus="Bus1", p_set=50.0)

# 2. Add to world using modern 'model' parameter
sim = world(start_time=0, stop_time=3600, t_macro=900)
sim.add_pypsa(
    sim_name="market",
    model=n,
    inputs=["Load1.p_set"],
    outputs=["Gen0.p", "Bus1.v_mag_pu_set"],
    step_size=900,
    mode="ac",
)

# 3. Simulate
sim.simulate(dashboard=False)
results = sim.results(dashboard=False)
print(results["market"].head())
```

---

## OpenDSS Simulators

Distribution networks defined in EPRI OpenDSS scripts (`.dss`) can be simulated using `sim_type='opendss'`.
This uses `OpenDSSDirect.py` for cross-platform execution across Windows, Linux, and macOS without requiring legacy COM servers.

```python
my_world.add_simulator(
    sim_type='opendss',
    sim_name='feeder',
    model='Master.dss',
    inputs=['Load.Load1.kW', 'Generator.PV.kW', 'Load.Load1.kvar'],
    outputs=['Bus.B1.vm_pu', 'Bus.B1.kv', 'Line.Line1.p_kw', 'Circuit.losses_kw'],
    step_size=300,
    mode='ac'   # 'ac' for snapshot power flow, 'dc' for direct mode
)
```

### Standalone Runnable Example (OpenDSS Feeder)

```python
import tempfile
import os
from energysim import world

# 1. Generate a self-contained 3-bus test circuit
dss_content = \"\"\"Clear
New Circuit.TestFeeder basekv=11 pu=1.05 phases=3 bus1=SourceBus
New LineCode.336ACSR nphases=3 r1=0.198 x1=0.347 units=km
New Line.Line1 Bus1=SourceBus Bus2=LoadBus LineCode=336ACSR Length=1.5 Units=km
New Load.Factory Bus1=LoadBus Phases=3 kV=11 kW=600 kvar=150
Set VoltageBases = [11.0]
CalcVoltageBases
\"\"\"

dss_path = os.path.join(tempfile.gettempdir(), "test_circuit.dss")
with open(dss_path, "w") as f:
    f.write(dss_content)

# 2. Add to energysim world
sim = world(start_time=0, stop_time=3600, t_macro=900)
sim.add_opendss(
    sim_name="feeder",
    model=dss_path,
    inputs=["Load.Factory.kW"],
    outputs=["Bus.LoadBus.vm_pu", "Line.Line1.p_kw", "Circuit.losses_kw"],
    step_size=900,
)

# 3. Simulate
sim.simulate(dashboard=False)
results = sim.results(dashboard=False)
print(results["feeder"].head())
```

### Variable naming convention (OpenDSS)

- **Inputs**: `'ElementType.ElementName.Property'` (e.g. `'Load.Load1.kW'`, `'Generator.Gen1.kW'`, `'PVSystem.PV1.pmpp'`).
- **Outputs**:
  - Bus voltage: `'Bus.<busname>.vm_pu'`, `'Bus.<busname>.kv'`, `'Bus.<busname>.vang'`
  - Line power: `'Line.<linename>.p_kw'`, `'Line.<linename>.q_kvar'`
  - Circuit summary: `'Circuit.losses_kw'`, `'Circuit.total_power_kw'`

---

## Script simulators

Standalone Python functions can be plugged directly into the co-simulation using `sim_type='script'`:

```python
my_world.add_simulator(
    sim_type='script',
    sim_name='my_module.my_step_function',
    sim_loc='/path/to/script_directory',
    inputs=['in_power', 'in_temp'],
    outputs=['out_voltage', 'out_soc'],
    step_size=60
)
```

### Writing a script function

In your Python file (e.g. `my_module.py` located inside `sim_loc`):

```python
def my_step_function(inputs):
    """
    inputs: dict mapping input variable name -> current numeric value
    returns: list or tuple of output values matching the declared outputs order
    """
    p = inputs.get('in_power', 0.0)
    t = inputs.get('in_temp', 20.0)
    
    # Compute output values
    v = 230.0 - 0.05 * p
    soc = max(0.0, 1.0 - t / 100.0)
    
    return [v, soc]
```

---

## Physical Units & Automatic Engineering Conversion

Multi-energy co-simulations frequently exchange power variables between simulators that operate with different native engineering units (e.g. Modelica FMUs in Watts, OpenDSS in kilowatts, and Pandapower in megawatts).

`energysim` standardizes on **kW** (kilowatts) as its baseline unit, but allows users to specify custom units at the world, simulator, or connection level.

### Specifying units

1. **Simulation Canvas Default** (`world` level):
   ```python
   my_world = world(power_unit='kW', ...)  # Default is 'kW'
   ```

2. **Simulator Level**:
   ```python
   my_world.add_simulator(
       sim_type='csv',
       sim_name='solar',
       sim_loc='solar.csv',
       power_unit='kW',                     # Default for variables in this simulator
       units={'ghi': 'W/m2', 'p_solar': 'kW'}  # Per-variable overrides
   )
   ```

3. **Connection Level**:
   ```python
   # Explicit from_unit and to_unit:
   my_world.add_connections({
       'solar.p_solar': {'to': 'grid.Gen1.p_mw', 'from_unit': 'kW', 'to_unit': 'MW'}
   })
   ```

### Automatic Unit Resolution

When `from_unit` or `to_unit` are omitted, `energysim` automatically infers the physical unit from:
- Explicit per-variable annotations in `units`
- Variable naming conventions (`.p_mw` $\to$ MW, `.kW` $\to$ kW, `losses_kw` $\to$ kW, `.m:u` / `.vm_pu` $\to$ p.u.)
- Simulator native defaults (Pandapower $\to$ MW, OpenDSS $\to$ kW, FMU $\to$ W)

The conversion factor is automatically computed and applied during every variable exchange step without requiring manual scale factors.

---

## Glass-Box Co-Simulation (Accessing Simulator Internals)

Rather than treating simulators as opaque black boxes, `energysim` provides standardized public accessors to inspect or modify internal solver state and domain objects:

```python
# Access the underlying native simulation model
net = my_world.get_model("grid")       # returns pandapower network DataFrame collection
pypsa_net = my_world.get_model("pypsa") # returns PyPSA Network instance
dss = my_world.get_model("feeder")     # returns opendssdirect module handle
proj = my_world.get_model("grid_pf")   # returns active PowerFactory project
fmu = my_world.get_model("plant")      # returns FMU model instance

# Access the energysim SimulatorAdapter wrapper
adapter = my_world.get_adapter("grid")
```

This allows users to modify parameters midway through a scenario (e.g. line outages, tap changes, or custom control laws) and leverage vendor-specific visualization tools directly.

---

## Simulator Variable Interface & Plumbing Contract

| Simulator | Under the Hood | Variable Format | Native Power Unit | Read/Write Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **Pandapower** | Python API (`net` DataFrames) | `ElementName.attribute` (e.g. `PV.p_mw`, `Bus1.vm_pu`) | **MW / MVar** | Direct vector assignment to `net.load`, `net.gen`, `net.res_bus` |
| **OpenDSS** | `OpenDSSDirect.py` C-API | `Type.Name.Property` (e.g. `Load.L1.kW`, `Bus.B1.vm_pu`) | **kW / kVar** | Direct C-API setters via `opendssdirect` element navigation |
| **PyPSA** | Network component DataFrames | `Component.Attribute` (e.g. `Generator1.P`, `Bus1.V`) | **MW / MVar** | Vector assignment to `network.generators_t.p_set` at active snapshot |
| **PowerFactory** | C-Python Engine API | `Element.e:attr` / `Element.m:attr` | **MW / MVar** | `object.SetAttribute()` / `object.GetAttribute()` |
| **FMI (FMU)** | FMPy (C-DLL / Shared Lib) | Modelica dot path: `a.b.c` | **W / Var** (SI) | `fmi2SetReal()` / `fmi2GetReal()` |
| **MATLAB / Octave** | MATLAB Engine / `oct2py` | Positional function arguments | User-defined | Dynamic array marshaling |
| **Python Script** | Native function call | Arbitrary dict keys | User-defined | Keyword argument dictionary |
| **Vectorized Container** | Vectorized NumPy SIMD | `var[idx]`, `total_<var>`, `mean_<var>` | User-defined (`kW` default) | Direct NumPy array slices & in-place BLAS operations |

---

## Vectorized Container Simulators (Fleets)

When simulating mass fleets (e.g. 1,000 EVs, 500 heat pumps, or 200 PV arrays), use `world.add_fleet()` or `world.add_container()` instead of creating hundreds of individual simulators.

```python
# Add an EV battery fleet with 1,000 units
my_world.add_fleet(
    fleet_type='battery',
    name='ev_fleet',
    size=1000,
    capacity_kwh=60.0,
    max_charge_kw=11.0,
    initial_soc=0.3,
    step_size=900,
)

# Broadcast charging dispatch and connect aggregated fleet load to grid
my_world.add_connections({
    'dispatch.p_cmd': 'ev_fleet.p_cmd',
    'ev_fleet.total_actual_p_kw': 'grid.FleetLoad.p_mw',
})
```

See the full [Vectorized Containers guide](containers/overview.md) for detailed formulas, architecture, and end-to-end co-simulation examples.



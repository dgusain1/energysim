# FAQ and troubleshooting

## Running and results

### `AttributeError: 'world' object has no attribute 'results'`
`world.results()` was removed in 3.0. `simulate()` returns the `Results` object (also `world.last_results`):

```python
from energysim import world

w = world(start_time=0, stop_time=120, t_macro=60)
w.add_signal("s", "2 * t", step_size=60)
res = w.simulate(pbar=False)              # res = world.simulate(); there is no world.results()
print(res["s.y"].tolist() == w.last_results["s.y"].tolist())
```

See [Results](results.md) and [Migrating to 3.0](migration-3.0.md).

### `TypeError: world.__init__() got an unexpected keyword argument 'res_filename'`
No HDF5 file is written any more. Persist runs with `world(out_dir='runs')`, `energysim run ... --out runs`,
`res.save(path)` or `res.to_hdf(path)` (needs `pip install energysim[hdf5]`). A `res_filename:` key in a YAML scenario
raises a `ValueError` that says the same.

### `ValueError: coupling must be 'jacobi' or 'gauss-seidel'` / "iterative coupling was removed in 3.0"
Iterative coupling needed rollback (`save_state` / `restore_state`), which no adapter implements, so it was removed together
with `max_iterations` and `convergence_tolerance`. Feedback loops now carry a one-macro-step delay: reduce `t_macro` or
merge the coupled components. See [Architecture](architecture.md#algebraic-loops).

### `ValidationError: ... stop_time - start_time = 100 is not an integer multiple of the macro step 30`
For fixed steps the span must be a multiple of `t_macro`; the message suggests the nearest valid `stop_time`. Choose a
fitting `t_macro`, or set `t_macro_min` below `t_macro` (variable macro step), which lifts the restriction
([Events](events.md)).

### My results have one row per second (or per micro step) and are huge
`simulate()` defaults to `record_all=False`: one row per macro step (taken before the step). If you see micro-step
rows, `record_all=True` was passed or set via `options({'record_all': True})` / `world: {record_all: true}` /
`energysim run --record-all`. A final row at `stop_time` is always added.

### The first row is zero (or an old value), and values seem one step late
Rows are taken **before** stepping, so the row at `t` shows the state at the start of the step beginning at `t`. With
Jacobi coupling a receiver sees its sources' state from the start of the step, i.e. one macro step old; Gauss-Seidel
removes the delay for feed-forward chains. See [Architecture](architecture.md#coupling).

### Do I need `tables` / HDF5?
No. Core dependencies are numpy, pandas, tqdm, networkx and pyyaml. `pip install energysim[results]` adds pyarrow for
Parquet (without it `Results.save` and `out_dir` fall back to CSV); `energysim[hdf5]` adds pytables for `to_hdf` / `load`
of `.h5`.

### The dashboard does not open
`res.dashboard(path, open=True)` and `energysim run --dashboard` need `plotly` (`pip install plotly`);
`--offline` embeds plotly.js in the HTML. The file is written even if the browser cannot be opened.

## Units

### "unit of x is unspecified; the value is passed through unscaled"
The variable has no declared unit, no recognisable name suffix, and its name does not denote a known quantity, so
energysim refuses to guess. Declare it: `add_signal(..., units={'y': 'kW'})`, `add_*(..., units={'var': 'unit'})`,
`units = {...}` on a `ModelComponent`, or `from_unit` / `to_unit` on the connection. `world(unit_policy='strict')` turns
this warning into an error at `add_connections`; `energysim validate -v` lists every connection with its resolved units
and their sources.

### `ValueError: Unknown unit 'W/m2'` or "Incompatible unit conversion"
Unit names must be in the registry ([Units](units.md)); explicit units of different dimensions (`kW` to `degC`) raise. Some
model components still declare units that are not in the registry (irradiance `W/m2`, wind speed `m/s`, `MMBtu/h`): a
connection to or from them is passed through unscaled, with a warning.

### kW versus MW
Convert by declaring units, not by hand. pandapower, PyPSA and most built-in models work in MW, OpenDSS in kW, FMUs in
whatever their variables say:

```python
import pandapower as pp

from energysim import world

net = pp.create_empty_network()
b = pp.create_bus(net, vn_kv=20.0, name="B")
pp.create_ext_grid(net, b, name="Grid")
pp.create_load(net, b, p_mw=1.0, name="L")

w = world(start_time=0, stop_time=60, t_macro=60)
w.add_powerflow("grid", net=net, inputs=["L.p_mw"], outputs=["L.p_mw"], step_size=60)
w.add_signal("demand", 800.0, step_size=60, units={"y": "kW"})
w.add_connections({"demand.y": "grid.L.p_mw"})
print(w.simulate(pbar=False, record_all=False)["grid.L.p_mw"].iloc[1])     # 0.8: 800 kW became 0.8 MW
```

A `scale` on a connection without unit keys switches the automatic conversion off for that connection.

## Models

### Why does my alkaline electrolyzer produce 0 and sit in `STANDBY`?
It has a minimum turndown (`min_turndown_pct`): below that fraction of `p_nominal_mw` the stack switches to standby.
Command more power, lower the turndown, or use a PEM electrolyzer, which allows much deeper part load.

### Why is the PV output zero with plenty of irradiance?
Check the inputs actually reaching the model (`G_poa` in W/m2, `T_amb` in degC, `curtail_limit` in MW - default
unconstrained). A `curtail_limit` connected to a signal that evaluates to 0 clips the output to 0. Irradiance below
about 1 W/m2 is treated as night.

### Why does my gas turbine produce 0 MW for the first steps?
`GasTurbine_OCGT` and `CombinedCycle_CCGT` run an `OperatingStateMachine`: a start-up of `hot/warm/cold_start_duration_sec`
depending on the time offline (a fresh unit starts cold), then ramp limits and minimum up / down times. Shorten the
timers for studies that should not model the start-up (`cold_start_duration_sec=300`) and check `state` in the outputs.
With a variable macro step the start-up completion time is an [event hint](events.md).

### `SimulatorVariableError: 'x' is not a declared input or parameter`
Model components are strict: `set_value` accepts only declared inputs and registered parameters, `get_value` only
declared outputs, inputs and parameters, and the message lists the valid names. `component.describe()` prints them.

### Which equipment parameters are "real"?
None are manufacturer data. `energysim.models.catalog` holds generic archetypes with a provenance note
(`catalog.PROVENANCE`); the branded names of 2.x (`Tesla_Megapack_2XL`, `Nel_A1000`, `GE_7HA_02`, ...) are deprecated
aliases. Parametrize models with your own data, see [Custom models](custom-models.md).

## Connections and scenarios

### `ConnectionSpecError: Simulator 'x' referenced in connection ... does not exist`
Add the simulator before the connection; spell the name exactly as in `sim_name`; use `sim.variable` (the split is at the
first dot, the rest is the variable name, e.g. `grid.Bus2.vm_pu`). `ConnectionSpecError` is also importable as
`ConnectionError` for old code.

### `UnsafeExpressionError: Function call not allowed in signal expression`
Signal strings are no longer `eval`-ed. Use the safe subset (`sin cos tan exp log sqrt abs min max floor ceil`, `pi`, `e`,
`t`), a structured spec such as `{type: sine, amplitude: 2, period: 3600}`, or pass a Python callable
(`add_signal('s', lambda t: np.sin(t))`) in Python code. See [Safe mode](safe-mode.md#safe-expressions).

### `ValueError: ... dotted import ... allow_import`
`model: pandapower.networks.case14` would import and call code, so it needs `allow_import=True`
(`add_simulator(..., allow_import=True)`, `world.from_yaml(..., allow_import=True)`) and is never allowed with `safe=True`.

### `ValueError: ... is not allowed in safe mode`
The scenario uses a simulator type that executes code (`external`, `script`, `matlab`, `powerfactory`), a pickle model
(`.p`, `.pkl`) or a dotted import. Remove it, or load the file without `--safe` if you trust it.

### `type: model` does not work in my YAML
A scenario cannot instantiate a `ModelComponent` class by name. Load the scenario in Python and add models with
`add_model`, or put the class in a file and use `type: external, model: my_models.py:MyClass`.

### An algebraic loop warning appears
Two or more simulators feed each other. The loop is solved with a one-macro-step delay (also under Gauss-Seidel).
Reduce `t_macro` or merge the components into one simulator ([Architecture](architecture.md#algebraic-loops)).

### My event's `set` has no effect
A `set` on a **connected** input is overwritten by that step's exchange; scheduled events fire before the exchange. Set
unconnected inputs and parameters, or change the source of the connection. Values are written without unit conversion
([Events](events.md)).

### `EnsembleError: ... module level` when using `parallel='process'`
The factory and KPI callables must be importable (module-level functions); lambdas and closures cannot be sent to the
worker processes. Put them in a module, or use `parallel='serial'` ([Ensembles](ensemble.md)).

## Simulators

### `LoadflowNotConverged` (pandapower)
Usual causes: a bus without a path to an `ext_grid` (islanding), reactive load beyond what the feeder can carry, a
wrong unit (kW passed as MW). Run `pp.diagnostic(net)`, check `w.connection_units()`, or try `mode='dc'` for dispatch
studies.

### OpenDSS: "only one circuit can be active" / `mode='dc'` rejected
OpenDSSDirect.py holds one circuit per Python process, so only one `opendss` simulator can be initialised at a time;
OpenDSS has no DC power flow (`mode='dc'` raises). Time is not advanced inside OpenDSS: drive time-varying load through
inputs.

### PyPSA: a generator's `P` is `NaN`
With PyPSA 1.x a slack generator's active power is only written when it has a `p_set`; read the bus `P` instead or
give the slack a `p_set` (0 is fine). `mode='dc'` returns zero reactive quantities and `V = 1`.

### FMU: "The FMU contains no binary for your platform"
The FMU was exported for another operating system or bitness. Re-export it for your platform (`win64`, `linux64`,
`darwin64`).

### Octave: `nargin: invalid function name` / MATLAB function errors
The `.m` file must be a **function** `function [out1, ...] = name(in1, ..., time)` (not a script), its name must equal the file
name, and the number of declared inputs and outputs must match the `inputs` / `outputs` lists (`time` is appended
automatically). For Octave, GNU Octave must be on `PATH` and `oct2py` installed.

### `ImportError: fmpy is required` / pandapower / PyPSA missing
Optional dependencies are imported lazily when the first simulator of that kind is added. Install the matching extra
(`energysim[fmu]`, `[powerflow]`, `[pypsa]`, `[opendss]`).

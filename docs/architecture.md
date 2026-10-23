# Architecture

## Coordinator and adapters

`energysim` follows a **coordinator / adapter** pattern.

* The **coordinator** is the `world` class (`energysim.world`, composed from mixins in `energysim._world`). It owns
  the simulator registry, the connections, the unit resolution, the macro-step loop, events and the recording of
  results.
* Each simulator is wrapped in an **adapter**, a subclass of `SimulatorAdapter` (`energysim.base`). The world never
  branches on the simulator type; everything goes through the adapter contract. That is what makes an FMU, a
  pandapower network, a CSV profile, a built-in physical model and your own class interchangeable
  (`tests/test_adapter_contract.py` runs one contract suite over every adapter kind).

| Mixin (`energysim._world`) | Responsibility |
|---|---|
| `registry` | `add_signal`, `add_model`, `add_simulator`, `add_fmu`, `add_powerflow`, `add_pypsa`, `add_opendss`, `add_csv`, `add_script`, `add_external_simulator`, `add_container`, ...; `get_adapter`, `get_model` |
| `connections` | parse and validate connections, dependency graph, execution order |
| `unit_resolution` | per-variable unit resolution, `unit_table`, `connection_units` |
| `coupling` | Jacobi / Gauss-Seidel stepping, exchange, extrapolation |
| `events` | `schedule`, `add_trigger`, step-size planning |
| `runner` | run lifecycle, `simulate`, `options`, results assembly |
| `validation` | `validate()` |
| `scenario` | `to_dict`, `to_yaml`, `from_dict`, `from_yaml`, safe mode |

## The adapter contract

Required:

| Method | Description |
|---|---|
| `step(time)` | Advance the model by one micro step (`step_size` seconds) starting at `time`. |
| `get_value(parameters, time)` | Return a **list** with one value per requested name. |
| `set_value(parameters, values)` | Receive inputs (and parameters). |

Optional, with defaults:

| Method / attribute | Description |
|---|---|
| `configure(start_time, stop_time, macro_step)` | Called on every adapter at the start of every run, before parameters and `init()`. |
| `init()` | Prepare a run. Must (re)create all run state: a world can be simulated repeatedly. |
| `set_parameters(params)` | Apply `{name: value}` initial values (default: `set_value`). FMUs call `apply_start_values`. |
| `SET_PARAMS_BEFORE_INIT` | `True` if initial values must be applied **before** `init()` (FMUs). Default `False`: after `init()`. |
| `advance(start, stop)` | Advance over a macro interval. Default: loop `step()`; a final shorter piece is delegated to `step_partial`. Adapters with an internal solver override it. |
| `advance_with_recording(start, stop, outputs)` | Same, returning one row `[time, v1, ...]` per micro step (used when `record_all=True`). |
| `step_partial(time, dt)` | Shorter final micro step when an interval is not a multiple of `step_size`. Default: `step(time)`. |
| `micro_steps(start, stop)` | The plan `[(t, dt, full), ...]` used by the defaults. |
| `max_step(time)`, `next_event_time(time)` | Step-size hints for [variable macro steps](events.md). Default `inf`. |
| `get_available_variables()` | `{'inputs': [...], 'outputs': [...]}`; used by connection validation. |
| `cleanup()` | Release resources; always called, even after an error. |

Rollback (`save_state` / `restore_state`) is not part of the contract in 3.0.

Built-in adapters:

| Class | `sim_type` | Wraps |
|---|---|---|
| `FmuCsAdapter` / `FmuMeAdapter` | `'fmu'` | FMI 1.0 / 2.0 co-simulation / model exchange (FMPy; ME uses CVode or Euler) |
| `pp_adapter` | `'powerflow'` | pandapower network, AC or DC power flow |
| `pypsa_adapter` | `'pypsa'` | PyPSA network, one snapshot per step (AC `pf` or DC `lpf`) |
| `opendss_adapter` | `'opendss'` | OpenDSS circuit via OpenDSSDirect.py (AC snapshot only) |
| `pf_adapter` | `'powerfactory'` | DIgSILENT PowerFactory |
| `matlab_adapter` | `'matlab'` | MATLAB / Octave `.m` functions |
| `csv_simulator` | `'csv'` | CSV time series (read-only) |
| `signal_adapter` | `'signal'` | `f(t)` signal (read-only) |
| `py_adapter` | `'script'` | a function in a Python file |
| your class | `'external'` | any `SimulatorAdapter` |
| `ModelComponent` | `'model'` | built-in and custom physical models |
| `VectorizedContainerAdapter` | `'container'` | fleets of N units |

CSV and signal adapters are read-only: a connection that writes to them is reported by `validate()` and
`set_value()` warns and discards the value.

## Life of a run

`simulate()` does the following (`validate=True` runs `validate(raise_on_error=True)` first):

1. `init()` of the world: reset run state, compute the macro step (`t_macro`, or the LCM of all step sizes if
   `t_macro=0`), call `configure(start, stop, macro)` on every adapter, then for every adapter apply initial values
   from `options({'init': ...})` / `set_parameters` and call `init()` (in the order given by
   `SET_PARAMS_BEFORE_INIT`).
2. The macro-step loop (below).
3. One final row per simulator at `stop_time`.
4. `cleanup()` of every adapter (in a `finally`: also when a simulator raised). If the run fails, the partial
   `Results` (`meta['status'] == 'failed'`) is stored in `world.last_results` and the error is re-raised.
5. A `Results` object is returned (and kept as `world.last_results`).

The loop, for a step from `t` to `t_end` (`t_end = t + t_macro` unless [events or hints](events.md) shorten it):

```text
evaluate triggers; fire due scheduled events and triggered actions
plan the step (t_end)
couple:   Jacobi        - exchange all connections, then step every simulator
          Gauss-Seidel  - for each simulator in topological order: exchange its inputs, then step it
record:   one row per simulator before stepping (or one per micro step with record_all=True)
```

Execution order is a topological sort of the dependency graph built from the connections (loops: see below);
unconnected simulators carry no ordering constraint.

## Coupling

* **Jacobi** (default): all values are exchanged from the state at the start of the step, then everything advances.
  A simulator therefore sees its inputs from the *previous* macro step's end state: a one-step delay on every
  connection.
* **Gauss-Seidel**: simulators advance one at a time in topological order and pass their fresh outputs on, so a
  feed-forward chain has no delay.

The difference is visible on a chain `src -> dst` where `src.n` counts completed steps:

```python
from energysim import SimulatorAdapter, world


class Count(SimulatorAdapter):
    """n = number of completed micro steps."""

    def __init__(self):
        self.inputs, self.outputs = [], ["n"]
        self.n = 0

    def init(self):
        self.n = 0

    def step(self, time):
        self.n += 1

    def get_value(self, parameters, time):
        return [self.n for _ in parameters]

    def set_value(self, parameters, values):
        pass


class Follow(SimulatorAdapter):
    """y = the input u as received before the step."""

    def __init__(self):
        self.inputs, self.outputs = ["u"], ["y"]
        self.u = self.y = 0.0

    def init(self):
        self.u = self.y = 0.0

    def step(self, time):
        self.y = self.u

    def get_value(self, parameters, time):
        return [self.y for _ in parameters]

    def set_value(self, parameters, values):
        self.u = values[0]


for coupling in ("jacobi", "gauss-seidel"):
    w = world(start_time=0, stop_time=50, t_macro=10, coupling=coupling)
    w.add_model("src", Count(), step_size=10, units={"n": "1"})
    w.add_model("dst", Follow(), step_size=10, units={"u": "1"})
    w.add_connections({"src.n": "dst.u"})
    res = w.simulate(pbar=False, record_all=False)
    print(coupling, res["src.n"].tolist(), res["dst.y"].tolist())
# jacobi        [0, 1, 2, 3, 4, 5] [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]   <- one step behind
# gauss-seidel  [0, 1, 2, 3, 4, 5] [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
```

### Algebraic loops

If the connections form a cycle, `add_connections` warns (`UserWarning`, listing the simulators), the dependency graph is
condensed into strongly connected components for the execution order, and the simulators of a loop exchange their
inputs from the previous step (Jacobi) even under `coupling='gauss-seidel'`. The loop therefore carries a
**one-macro-step delay**, and `validate()` repeats the advice:

* reduce `t_macro` (the delay shrinks with it), or
* merge the coupled components into one simulator.

There is no iteration to convergence in 3.0 (it needed rollback, which no adapter implements).

```python
import warnings

from energysim import SimulatorAdapter, world


class Damped(SimulatorAdapter):
    def __init__(self):
        self.inputs, self.outputs = ["u"], ["y"]
        self.u = self.y = 0.0

    def init(self):
        self.u = self.y = 1.0

    def step(self, time):
        self.y = 0.5 * self.u + 1.0

    def get_value(self, parameters, time):
        return [self.y for _ in parameters]

    def set_value(self, parameters, values):
        self.u = values[0]


w = world(start_time=0, stop_time=50, t_macro=10)
w.add_model("a", Damped(), step_size=10, units={"u": "1", "y": "1"})
w.add_model("b", Damped(), step_size=10, units={"u": "1", "y": "1"})
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    w.add_connections({"a.y": "b.u", "b.y": "a.u"})
print(caught[0].message)                       # Algebraic loop detected involving simulators: ['a', 'b'] ...
print(w.validate().warnings[0][:60])           # Algebraic feedback loop detected among simulators ...
```

### Extrapolation

`world(extrapolation=...)` controls what a receiver gets between exchanges: `'zero-order'` (default) holds the last
exchanged value; `'linear'` extrapolates one macro step ahead from the two most recent exchanged values (zero-order
until two are known). Linear extrapolation helps smooth signals and can amplify noise or jumps.

## Recording

Every simulator contributes one table. By default (`record_all=False`) one row per **macro step**, taken before the
step; with `record_all=True` one row per micro step. A final row at `stop_time` is always added. Rows are buffered
in typed columns (float, int, bool, str) and, with `out_dir`, streamed to Parquet. See [Results](results.md).

## Registry and glass-box access

Every simulator is stored in `world.simulator_dict` as a `SimEntry`:

| Field | Meaning |
|---|---|
| `adapter` | the `SimulatorAdapter` instance |
| `sim_type` | `'fmu'`, `'powerflow'`, `'pypsa'`, `'opendss'`, `'csv'`, `'signal'`, `'script'`, `'external'`, `'powerfactory'`, `'matlab'`, `'model'`, `'container'` |
| `step_size`, `inputs`, `outputs` | micro step and variable lists |
| `mode` | `'ac'` / `'dc'` for power-flow adapters |
| `power_unit`, `units` | declared units, see [Units](units.md) |
| `model` | the model reference you passed (object, path, name) |

`world.get_adapter(name)` returns the adapter, `world.get_model(name)` the native object inside it:

| Type | `get_model` returns |
|---|---|
| `powerflow`, `pypsa` | the pandapower net / PyPSA `Network` |
| `opendss` | the OpenDSSDirect handle |
| `powerfactory` | the active project |
| `fmu` | the FMPy FMU object |
| `matlab` | the MATLAB / Octave engine |
| `script` | the loaded module |
| `csv` | the pandas DataFrame |
| `signal` | the signal callable |
| `model`, `external`, `container` | usually the adapter itself (or its `model` attribute if it has one) |

Because adapters are re-runnable (`pp_adapter` and `pypsa_adapter` keep a pristine copy and restore it on the next
`init()`; FMUs are re-instantiated), results stay inspectable after a run.

## Validation and errors

`validate()` checks that simulators exist, step sizes are positive and divide the macro step, the span is a multiple of
the macro step (fixed-step runs), connection endpoints exist, read-only simulators are not written to, variable names
are known to adapters that can introspect, loops, events/triggers reference known simulators and units
(`unit_policy='strict'` turns unspecified units into errors). It returns a `ValidationReport`
(`is_valid`, `errors`, `warnings`, `conversions`, `unit_system`).

Exceptions (`energysim.base`): `EnergysimError` (base), `SimulatorVariableError`, `SimulatorElementNotFoundError`,
`ConnectionSpecError` (alias `ConnectionError`), `ValidationError`.

## Dependencies and copies

Simulator packages (FMPy, pandapower, PyPSA, OpenDSSDirect.py, pyarrow, scipy, matplotlib, plotly) are imported lazily,
only when needed. Worlds, including their parsed connections, are deep-copyable and picklable; [`Sweep`](ensemble.md)
relies on this to give every run a pristine copy.

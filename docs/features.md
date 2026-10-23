# Features

A tour of what the engine does beyond "add simulators, connect, simulate". Each section links to the page with the
details. The machine-checked list of claims, with the tests that back them, is `features.yaml` in the repository
(and the table in the README).

## Coupling, time and events

| Feature | Summary | Details |
|---|---|---|
| Jacobi coupling (default) | all values are exchanged, then everything steps; one-step delay on every connection | [Architecture](architecture.md#coupling) |
| Gauss-Seidel coupling | simulators step in topological order and pass fresh outputs on | [Architecture](architecture.md#coupling) |
| Algebraic-loop handling | loops are detected, warned about and carry a one-macro-step delay; reduce `t_macro` or merge components | [Architecture](architecture.md#algebraic-loops) |
| Extrapolation | `'zero-order'` hold (default) or `'linear'` extrapolation between exchanges | [Architecture](architecture.md#extrapolation) |
| Variable macro step | `t_macro` is the maximum, `t_macro_min` the minimum step; hints from signals, CSV rows, FMUs and models shorten it | [Events](events.md) |
| Scheduled events and triggers | `world.schedule`, `world.add_trigger`, YAML `events:` / `triggers:`; no rollback | [Events](events.md) |
| Shorter final micro step | `step_partial(time, dt)` when an interval is not a multiple of `step_size` | [Events](events.md) |

!!! note "Removed in 3.0"
    `coupling='iterative'` (with `max_iterations`, `convergence_tolerance`) and `save_state()` / `restore_state()`
    need rollback, which no adapter implements, so they are gone. See [Migrating to 3.0](migration-3.0.md).

## Connections and validation

`add_connections` accepts a mapping `{'a.y': 'b.u'}` or a list of dictionaries with `from`, `to`, `scale`, `offset`,
`from_unit`, `to_unit`; a tuple of sources (fan-in) is summed; a list of targets (fan-out) receives the value. It
fails fast on broken specifications and warns on suspicious ones:

* an unknown simulator or a malformed `sim.var` string raises `ConnectionSpecError` (alias `ConnectionError`);
* a variable name unknown to an adapter that can list its variables gives a `UserWarning`;
* writing into a read-only simulator (CSV, signal) gives a `UserWarning`, and the value is discarded.

`world.validate()` audits the whole world before a long run (unknown endpoints, step sizes, loops, events, units);
`simulate()` calls it by default and raises `ValidationError` on errors.

```python
import warnings

from energysim import ConnectionSpecError, world
from energysim.models import BatteryBESS

w = world(start_time=0, stop_time=600, t_macro=60)
w.add_signal("cmd", 0.5, step_size=60, units={"y": "MW"})
w.add_model("bess", BatteryBESS(), step_size=60)

try:
    w.add_connections({"cmd.y": "ghost.P_cmd"})
except ConnectionSpecError as exc:
    print(exc)                                   # Simulator 'ghost' referenced in connection ... does not exist.

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    w.add_connections({"cmd.y": "bess.P_comand"})               # typo
print(caught[0].message)                                        # Variable 'P_comand' not found in simulator 'bess' ...

w.add_connections({"cmd.y": "bess.P_cmd"})
report = w.validate()
print(report.is_valid, report.errors)                           # True []
```

## Units

Every connected variable is resolved to a unit and values are converted into the receiver's unit. The default unit
system (`energysim-default`: kW, kvar, kV, kWh, degC, bar, kg/h, Nm3/h, ...) can be replaced by `'si'` or overridden per
quantity. Unspecified units are never scaled: a warning by default, an error with `unit_policy='strict'`. Incompatible
dimensions of explicit units (and unknown unit names) raise `ValueError`. A connection with a `scale` and no unit keys
bypasses conversion. See [Units](units.md) for the table, the resolution order and the inspection tools.

```python
from energysim import world
from energysim.models import BatteryBESS

w = world(start_time=0, stop_time=600, t_macro=60, unit_system={"active_power": "MW"})
w.add_signal("cmd", 500.0, step_size=60, units={"y": "kW"})
w.add_model("bess", BatteryBESS(), step_size=60)
w.add_connections({"cmd.y": "bess.P_cmd"})

print(w.connection_units()[["from_unit", "to_unit", "scale"]].to_dict("records"))     # kW -> MW, x0.001
print(w.unit_table().splitlines()[0])                                          # Unit system 'custom' [<hash>]
```

## Dependency graph

After connections are added, the world builds a directed graph (NetworkX), sorts the simulators topologically and
condenses strongly connected components for loops. `world.plot()` draws it (needs matplotlib); the order is logged at
INFO level (`world(logging=True)`).

```python
import matplotlib

matplotlib.use("Agg")

from energysim import world
from energysim.models import BatteryBESS, PhysicsPV

w = world(start_time=0, stop_time=600, t_macro=60)
w.add_signal("sun", 800.0, step_size=60)
w.add_model("pv", PhysicsPV(), step_size=60)
w.add_model("bess", BatteryBESS(), step_size=60)
w.add_connections({"sun.y": "pv.G_poa", "pv.P_ac": "bess.P_cmd"})
w.plot(plot_edge_labels=False)
print(list(w._dep_graph.edges))             # [('sun', 'pv'), ('pv', 'bess')]
```

## Recording

`simulate(record_all=True)` (the default) records every micro step through each adapter's
`advance_with_recording`, which for FMU model exchange honours the solver's own time points. `record_all=False`
records one row per macro step, before stepping. Either way each simulator gets a final row at `stop_time`.
`options({'record_all': False})` changes the default. See [Results](results.md).

## Introspection and glass-box access

`adapter.get_available_variables()` returns `{'inputs': [...], 'outputs': [...]}` for adapters that can introspect
(pandapower, PyPSA, OpenDSS, FMU, CSV, script, models, fleets). `ModelComponent.describe()` adds parameters and units.
`world.get_adapter(name)` and `world.get_model(name)` return the live adapter and native object (pandapower net, PyPSA
network, OpenDSS handle, FMU, DataFrame, ...); `world.simulator_dict` holds the `SimEntry` of each simulator.

```python
from energysim import world
from energysim.models import BatteryBESS

w = world(start_time=0, stop_time=600, t_macro=60)
w.add_model("bess", BatteryBESS(capacity_mwh=2.0), step_size=60)
print(w.get_adapter("bess").get_available_variables()["inputs"])        # ['P_cmd', 'T_amb']
print(w.simulator_dict["bess"].sim_type, w.get_model("bess").capacity_mwh)   # model 2.0
```

## What else is in the box

| Area | Where |
|---|---|
| Many simulator kinds behind one adapter contract: signal, CSV, script, built-in models, fleets, pandapower, PyPSA, OpenDSS, FMU, PowerFactory, MATLAB / Octave, your own class | [Simulator adapters](add-simulator.md), [External simulators](external-simulators.md) |
| Physical component library and generic equipment archetypes | [Components](physical-models.md), [Authoring models](custom-models.md) |
| Vectorized fleets (thousands of units in one NumPy step) | [Fleets](containers/overview.md) |
| Heat and gas network-lite models | [Networks](networks.md) |
| Economic MPC, pandapower OPF and PyPSA rolling-horizon controllers, forecast providers | [Control](control.md) |
| Typed `Results`, KPIs, Parquet / CSV / HDF5, interactive dashboard | [Results](results.md) |
| Monte-Carlo ensembles, samplers, noisy inputs, parameter sweeps | [Ensembles](ensemble.md) |
| YAML scenarios, validation, CLI | [YAML and CLI](yaml-cli.md) |
| Safe loading of untrusted scenarios | [Safe mode](safe-mode.md) |
| Analytic, textbook and cross-tool validation benchmarks | [Validation](validation.md) |

## Error handling

Simulator errors are exceptions, never `sys.exit()`. All derive from `EnergysimError` (`energysim.base`):

| Exception | Raised when |
|---|---|
| `ConnectionSpecError` (alias `ConnectionError`) | a connection string is malformed or names an unknown simulator |
| `SimulatorVariableError` | a variable name is unknown to an adapter or model |
| `SimulatorElementNotFoundError` | a named element is missing in a network |
| `ValidationError` | pre-flight validation failed |
| `EnergysimError` | base class |

When a run fails, `cleanup()` is still called on every adapter, the error is re-raised, and the partial results are in
`world.last_results` (`meta['status'] == 'failed'`).

```python
from energysim import SimulatorAdapter, world
from energysim.base import EnergysimError


class Broken(SimulatorAdapter):
    def __init__(self):
        self.inputs, self.outputs = [], ["y"]

    def step(self, time):
        if time >= 120:
            raise RuntimeError("solver diverged")

    def get_value(self, parameters, time):
        return [time for _ in parameters]

    def set_value(self, parameters, values):
        pass


w = world(start_time=0, stop_time=300, t_macro=60)
w.add_model("b", Broken(), step_size=60)
try:
    w.simulate(pbar=False, record_all=False)
except RuntimeError as exc:
    print(exc, w.last_results.meta["status"], list(w.last_results["b"].index))       # solver diverged failed [0.0, 60.0]
```

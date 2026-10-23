# Events and the variable macro step

By default `energysim` advances in fixed macro steps of `t_macro` seconds. For outages, set-point changes
and threshold reactions that rarely fall on the grid, 3.0 adds three tools that all work **without
rollback**:

| Tool | What it does |
|---|---|
| `world.schedule(t, action)` | Run an action at time `t`. The macro step is shortened so that a step boundary lands exactly on `t`. |
| `world.add_trigger(name, condition, action)` | Run an action after a condition on simulator variables changes at a step boundary. |
| `t_macro_min` + adapter hints | Let simulators ask for shorter steps (`max_step`, `next_event_time`, `ModelComponent.time_to_event`). `t_macro` is the maximum, `t_macro_min` the minimum step. |

With the defaults (`t_macro_min=None`, no events) a run is the fixed-step loop of 2.x, bit for bit; step hints are
not even queried.

!!! note "Recording"
    `simulate()` records every micro step by default (`record_all=True`). The examples below pass
    `record_all=False` so that one row corresponds to one macro step.

## Scheduled events

An action is either a declarative `{'set': {'sim.var': value}}` or a `callable(world)`. It runs at the
**start of the step that begins at `t`, before that step's exchange**. Consequences:

* A `set` on an input that is also **connected** is overwritten by the exchange of the same step. Use events
  for unconnected inputs and parameters.
* `set` writes the value as given (`set_value`): no unit conversion is applied.
* In fixed-step mode the step after the event re-aligns with the `start_time + k * t_macro` grid.
* Events at or after `stop_time` never fire; they are reported with `success: None`.

```python
from energysim import world
from energysim.models import BatteryBESS

w = world(start_time=0, stop_time=4 * 3600, t_macro=600)
w.add_model("bess", BatteryBESS(capacity_mwh=2.0, p_max_mw=1.0, soc_init=0.6), step_size=60)
w.options({"init": {"bess": (["P_cmd"], [-0.5])}})         # start by discharging at 0.5 MW

w.schedule(4300, {"set": {"bess.P_cmd": 0.0}}, name="stop_discharge")      # off the 600 s grid
w.schedule(7200, lambda world_: print("t = 2 h:", world_.get_adapter("bess").get_value(["SoC"], 7200)),
           name="log")

res = w.simulate(pbar=False, record_all=False)
times = list(res["bess"].index)
print(times[6:10])                      # [3600.0, 4200.0, 4300.0, 4800.0]: a boundary at 4300, then the grid again
print(res.meta["events"][0])
```

Each entry of `Results.meta['events']` has `time` (when it ran), `name`, `kind` (`'scheduled'` or
`'trigger'`), `success` and `error`; a failing action is recorded and logged, it does not abort the run.

## Threshold triggers

`add_trigger(name, condition, action, direction='rising')` evaluates `condition` at every step boundary.
`direction` is `'rising'` (False to True), `'falling'` or `'both'`. When the condition changes, the action runs at
the start of the next step; the event record carries `detected_time`, the `interval` of the step in which the
change happened and a note that **localisation is limited to one macro step**.

`condition` is either

* a **safe expression string** over `'sim.var'` names: `"bess.SoC < 0.45"`, `"grid.Bus1.vm_pu < 0.95 and t > 600"`
  (dotted names are variable references; quote names that are not attribute-like, `"'fmu.der(x)' > 0"`;
  `t` is the simulation time), or
* a `callable(values: dict) -> bool` together with `variables=['sim.var', ...]`.

```python
from energysim import world
from energysim.models import BatteryBESS

w = world(start_time=0, stop_time=4 * 3600, t_macro=600)
w.add_model("bess", BatteryBESS(capacity_mwh=2.0, p_max_mw=1.0, soc_init=0.6), step_size=60)
w.options({"init": {"bess": (["P_cmd"], [-0.5])}})

# when the state of charge falls below 45 %, switch to charging at 0.2 MW
w.add_trigger("soc_low", "bess.SoC < 0.45", {"set": {"bess.P_cmd": 0.2}})
# the same with a callable condition
w.add_trigger("soc_high", lambda v: v["bess.SoC"] > 0.55, {"set": {"bess.P_cmd": -0.5}},
              variables=["bess.SoC"], direction="rising")

res = w.simulate(pbar=False, record_all=False)
for ev in res.meta["events"]:
    print(ev["name"], ev["kind"], "detected at", ev["detected_time"], "interval", ev["interval"])
```

Trigger values are the raw values returned by the adapters (in the adapter's own units). `bisect=True` (locating
the crossing inside the step) would need rollback and raises `NotImplementedError`; reduce `t_macro` /
`t_macro_min` or provide a `next_event_time` hint instead.

## Variable macro step: `t_macro_min` and hints

With `t_macro_min < t_macro` the world asks every adapter for two hints before each step and takes the minimum
over all simulators, clamped to `[t_macro_min, t_macro]`:

| Hook (on `SimulatorAdapter`) | Meaning |
|---|---|
| `max_step(time)` | Largest macro step this simulator accepts from `time` (default `inf`). |
| `next_event_time(time)` | Absolute time of the next known internal event (default `inf`). The step ends exactly there. |

Built-in sources of hints:

| Source | Hint |
|---|---|
| `add_signal(..., breakpoints=[...])`, `step` / `ramp` / `piecewise` signal specs | breakpoints, `next_event_time` |
| `add_csv(..., align_to_rows=True)` | the next row time |
| FMU model exchange | the FMU's `nextEventTime` |
| `ModelComponent.time_to_event()` | seconds until a predictable event; the default `next_event_time` is `time + time_to_event()`. Implemented for the OCGT / CCGT start-up and minimum up/down timers, battery, flow battery, H2 tank and thermal-tank full / empty limits at the current power, and fuel-cell ramps. |

```python
from energysim import world

w = world(start_time=0, stop_time=3600, t_macro=600, t_macro_min=1)
w.add_signal("s", {"type": "step", "time": 1234, "before": 0, "after": 1}, step_size=1)
res = w.simulate(pbar=False, record_all=False)
print(list(res["s"].index))             # [0.0, 600.0, 1200.0, 1234.0, 1834.0, ...]
print(res.meta["n_macro_steps"], res.meta["t_macro_min"])
```

When the macro interval is not a multiple of a simulator's `step_size`, adapters do a **shorter final micro
step**: `SimulatorAdapter.step_partial(time, dt)` (default: `step(time)`; `ModelComponent` calls
`step_model(dt, time)`, FMU co-simulation uses `doStep(time, dt)`). A variable-step run also no longer needs
`stop_time - start_time` to be a multiple of `t_macro`.

Writing a hint for your own adapter or model:

```python
import math

from energysim import SimulatorAdapter, world


class Ramp(SimulatorAdapter):
    """y rises by 1 per second and saturates at 10 (the saturation time is known in advance)."""

    def __init__(self):
        self.step_size = 1.0
        self.inputs, self.outputs = [], ["y"]
        self.t = 0.0

    def init(self):
        self.t = 0.0

    def step(self, time):
        self.t = time + self.step_size

    def get_value(self, parameters, time):
        return [min(self.t, 10.0) for _ in parameters]

    def set_value(self, parameters, values):
        pass

    def next_event_time(self, time):                 # land a step exactly on the saturation point
        return 10.0 if time < 10.0 else math.inf

    def max_step(self, time):                        # and keep steps short while it is moving
        return 5.0 if time < 10.0 else math.inf


w = world(start_time=0, stop_time=60, t_macro=30, t_macro_min=1)
w.add_model("r", Ramp(), step_size=1)
print(list(w.simulate(pbar=False, record_all=False)["r"].index))     # [0.0, 5.0, 10.0, 40.0, 60.0]
```

For a `ModelComponent`, override `time_to_event()` instead:

```python
import math

from energysim import world
from energysim.models.common import ModelComponent


class Tank(ModelComponent):
    """Fills at `inflow` units per second up to `capacity`."""

    def __init__(self, capacity=100.0, inflow=2.0):
        super().__init__(name="tank", step_size=1.0)
        self.capacity, self.inflow, self.level = capacity, inflow, 0.0
        self.register_variables(outputs=["level"])

    def step_model(self, dt, time):
        self.level = min(self.capacity, self.level + self.inflow * dt)
        self._outputs_val["level"] = self.level

    def time_to_event(self):                         # seconds until full at the current inflow
        if self.level >= self.capacity or self.inflow <= 0:
            return math.inf
        return (self.capacity - self.level) / self.inflow


w = world(start_time=0, stop_time=300, t_macro=100, t_macro_min=1)
w.add_model("tank", Tank(), step_size=1)
res = w.simulate(pbar=False, record_all=False)
print(list(res["tank"].index))                       # [0.0, 50.0, 150.0, 250.0, 300.0]: the first step ends when the tank is full
```

## Events in YAML scenarios

Scenario files support the declarative `set` action only, so they are compatible with [safe mode](safe-mode.md). The
simulator below is a vectorized battery fleet, which a YAML scenario can create by itself:

```python
from energysim import world

scenario = """
world: {start_time: 0, stop_time: 7200, t_macro: 600, t_macro_min: 60}
simulators:
  - name: fleet
    type: battery_fleet
    step_size: 60
    options: {size: 10, initial_soc: 0.52}
    outputs: [mean_soc, total_actual_p_kw]
events:
  - {time: 1300, name: discharge, set: {fleet.p_cmd_kw: -11.0}}
triggers:
  - {name: low, condition: "fleet.mean_soc < 0.5", set: {fleet.p_cmd_kw: 0.0}}
"""
w = world.from_yaml(scenario, safe=True)
res = w.simulate(pbar=False, record_all=False)
print(list(res["fleet"].index)[:5])                  # [0.0, 600.0, 1200.0, 1300.0, 1900.0]
for ev in res.meta["events"]:
    print(ev["name"], ev["kind"], ev["time"])        # discharge scheduled 1300.0 / low trigger 1900.0
```

Keys: events take `time`, `name`, `set`; triggers take `name`, `condition`, `direction`, `set` and optional
`variables`. Unknown keys are rejected. `to_dict()` / `to_yaml()` serialise string-condition triggers and `set`
events; callables are skipped with a warning.

```python
from energysim import world

scenario = """
world: {start_time: 0, stop_time: 600, t_macro: 60}
simulators:
  - {name: sig, type: signal, signal: "t", step_size: 60}
events:
  - {time: 130, name: noted, set: {sig.y: 1}}
"""
w = world.from_yaml(scenario, safe=True)
print(w.scheduled_events)
print(w.to_dict()["events"])
```

!!! warning "Limits"
    Events are handled predictively: nothing is re-run. A trigger fires at the first boundary after the
    crossing, so its accuracy is one macro step unless hints or a small `t_macro_min` make steps shorter. Hint
    exceptions are logged and ignored. Validation (`validate()`) reports events and triggers that reference
    unknown simulators.

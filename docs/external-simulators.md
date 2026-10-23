# External simulators

Anything that can be driven by `step`, `get_value` and `set_value` can join a world: a Python model, a wrapper around
a solver or a REST API, a hardware-in-the-loop bridge. You implement a `SimulatorAdapter` subclass and register it with
`add_external_simulator` (or `add_model`, `add_simulator('external', ...)`).

## The minimal adapter

```python
from energysim import SimulatorAdapter, world


class Battery(SimulatorAdapter):
    """Coulomb-counting battery. Input `power` [kW] (+ charges), outputs `soc` and `power`."""

    def __init__(self, inputs=None, outputs=None, capacity_kwh=10.0, max_kw=5.0, efficiency=0.95):
        self.inputs = list(inputs or ["power"])
        self.outputs = list(outputs or ["soc", "power"])
        self.capacity_kwh, self.max_kw, self.efficiency = capacity_kwh, max_kw, efficiency
        self.soc, self.power = 0.5, 0.0

    def init(self):                                   # called at the start of EVERY run
        self.soc, self.power = 0.5, 0.0

    def step(self, time):                             # one micro step of self.step_size seconds
        p = max(-self.max_kw, min(self.max_kw, self.power))
        eta = self.efficiency if p > 0 else 1.0 / self.efficiency
        self.soc = min(0.95, max(0.05, self.soc + p * eta * self.step_size / 3600.0 / self.capacity_kwh))

    def get_value(self, parameters, time):            # a LIST, one value per requested name
        values = {"soc": self.soc, "power": self.power}
        return [values[p] for p in parameters]

    def set_value(self, parameters, values):
        for p, v in zip(parameters, values):
            if p == "power":
                self.power = v

    def get_available_variables(self):                # lets the world check connection names
        return {"inputs": self.inputs, "outputs": self.outputs}


w = world(start_time=0, stop_time=3600, t_macro=300)
w.add_signal("p_set", 4.0, step_size=300, units={"y": "kW"})
w.add_external_simulator("bat", model=Battery(), step_size=60, units={"power": "kW", "soc": "1"})
w.add_connections({"p_set.y": "bat.power"})

res = w.simulate(pbar=False, record_all=False)
print(res["bat.soc"].iloc[[0, -1]].round(3).tolist())          # [0.5, ...]
print(w.simulate(pbar=False, record_all=False)["bat.soc"].iloc[-1] == res["bat.soc"].iloc[-1])   # re-runnable
```

Rules:

* `step_size` is set by the world from the `add_*` call; use `self.step_size` inside `step`.
* `get_value` must return a **list** of the same length as `parameters`.
* `init()` may be called again after `cleanup()`, so `init()` must (re)create all run state. Keep heavy, immutable
  objects in `__init__`.
* Raise `SimulatorVariableError` for unknown names rather than returning a default; the world reports it with the
  simulator name.

## How to hand the adapter to the world

`add_external_simulator(sim_name, model=..., inputs=None, outputs=None, step_size=900, power_unit=None, units=None,
**kwargs)` (also `add_simulator(sim_type='external', sim_name=..., model=...)` and, for instances, `add_model`):

| `model` is | What happens |
|---|---|
| a `SimulatorAdapter` instance (or anything with `step`, `get_value`, `set_value`) | used as is |
| a class or factory callable | called as `factory(**kwargs)`; `inputs`, `outputs`, `step_size`, `sim_name` and your extra keyword arguments are passed **only if its signature accepts them** |
| `'path/to/file.py:ClassName'` | the file is loaded under a unique module name (`sys.path` is not modified) and `ClassName` is used as a factory |
| `'path/to/file.py'` | as above, with the class named `external_simulator` |
| a directory (legacy) | module `<sim_name>.py` in that directory, class `external_simulator`; the directory is on `sys.path` only while importing |

```python
from pathlib import Path

from energysim import world

Path("my_models.py").write_text('''
from energysim import SimulatorAdapter


class Gain(SimulatorAdapter):
    def __init__(self, inputs=None, outputs=None, factor=2.0):
        self.inputs, self.outputs, self.factor, self.x = ["x"], ["y"], factor, 0.0

    def step(self, time):
        pass

    def get_value(self, parameters, time):
        return [self.factor * self.x for _ in parameters]

    def set_value(self, parameters, values):
        self.x = values[0]
''', encoding="utf-8")

w = world(start_time=0, stop_time=60, t_macro=30)
w.add_signal("src", "t", step_size=30, units={"y": "1"})
w.add_external_simulator("gain", model="my_models.py:Gain", factor=3.0, step_size=30,
                         units={"x": "1", "y": "1"})          # extra keyword arguments reach Gain(...)
w.add_connections({"src.y": "gain.x"})
print(w.simulate(pbar=False, record_all=False)["gain.y"].tolist())       # [0.0, 90.0, 90.0]: the last row has no new exchange
```

In a YAML scenario: `{name: gain, type: external, model: my_models.py:Gain, step_size: 30, options: {factor: 3.0}}`.
External simulators execute your code, so they are refused in [safe mode](safe-mode.md).

## Lifecycle

Per run the world calls, in order, on every adapter:

1. `configure(start_time, stop_time, macro_step)` (optional; receive the time frame),
2. initial values from `options({'init': ...})` / `set_parameters` via `set_parameters(dict)` (default:
   `set_value`) **after** `init()`, or **before** `init()` if the class sets `SET_PARAMS_BEFORE_INIT = True`,
3. `init()`,
4. per macro step: `set_value` for connected inputs, then `advance(t0, t1)` (default: `step(t)` every `step_size`;
   a shorter last piece goes to `step_partial(t, dt)`),
5. `cleanup()` (always, also after errors).

```python
from energysim import SimulatorAdapter, world


class Trace(SimulatorAdapter):
    SET_PARAMS_BEFORE_INIT = False                  # the default: parameters are applied after init()

    def __init__(self):
        self.inputs, self.outputs, self.calls, self.k = ["k"], ["y"], [], 1.0

    def configure(self, start_time, stop_time, macro_step):
        self.calls.append(("configure", start_time, stop_time, macro_step))

    def init(self):
        self.calls.append(("init",))

    def step(self, time):
        pass

    def get_value(self, parameters, time):
        return [self.k for _ in parameters]

    def set_value(self, parameters, values):
        self.k = values[0]

    def cleanup(self):
        self.calls.append(("cleanup",))


tr = Trace()
w = world(start_time=0, stop_time=20, t_macro=10)
w.add_external_simulator("tr", model=tr, step_size=10, units={"k": "1", "y": "1"})
w.options({"init": {"tr": (["k"], [5.0])}})
w.simulate(pbar=False, record_all=False)
print(tr.calls)    # [('configure', 0, 20, 10), ('init',), ('cleanup',)]
print(tr.k)        # 5.0
```

## Optional capabilities

| Override | When |
|---|---|
| `advance(start, stop)` | your model has its own solver and can take a whole macro interval (FMU model exchange does) |
| `advance_with_recording(start, stop, outputs)` | with `record_all=True` the world wants `[[t, v1, ...], ...]` rows for the interval; the default steps and samples every micro step |
| `step_partial(time, dt)` | integrate correctly over a final step shorter than `step_size` (event-driven macro steps); default calls `step` |
| `max_step(time)`, `next_event_time(time)` | tell the world about known events, see [Events](events.md) |
| `get_available_variables()` | enable connection-name checks and `validate()` warnings |
| `cleanup()` | close files, sockets, engines |

`save_state()` / `restore_state()` are **gone** in 3.0 (they served the removed iterative coupling): delete them from
existing adapters.

```python
import math

from energysim import SimulatorAdapter, world


class Decay(SimulatorAdapter):
    """x' = -x / tau integrated exactly over any interval, so partial steps are exact too."""

    def __init__(self, tau=100.0):
        self.tau, self.x = tau, 1.0
        self.inputs, self.outputs = [], ["x"]

    def init(self):
        self.x = 1.0

    def step(self, time):
        self.step_partial(time, self.step_size)

    def step_partial(self, time, dt):
        self.x *= math.exp(-dt / self.tau)

    def get_value(self, parameters, time):
        return [self.x for _ in parameters]

    def set_value(self, parameters, values):
        pass


w = world(start_time=0, stop_time=130, t_macro=60, t_macro_min=1)        # 130 is not a multiple of 60
w.add_external_simulator("d", model=Decay(), step_size=7.0, units={"x": "1"})
res = w.simulate(pbar=False, record_all=False)
print(list(res["d"].index), round(res["d.x"].iloc[-1], 6) == round(math.exp(-1.3), 6))   # [0.0, 60.0, 120.0, 130.0] True
```

## Checklist for a well-behaved adapter

* Declare units for every variable you expose (`units=` on the call, or a `units` dict attribute on the adapter), see
  [Units](units.md).
* Re-runnable: `init()` resets everything the run changes.
* No global state: two worlds (or an [ensemble](ensemble.md)) may hold two instances. Worlds are deep-copied by
  `Sweep`, so avoid unpicklable handles in `__init__`; open them in `init()`.
* Exceptions from `step` abort the run and are re-raised by `simulate()` after `cleanup()`; the partial results are
  in `world.last_results`.
* A template with all hooks is in `src/energysim/external_simulator.py`.

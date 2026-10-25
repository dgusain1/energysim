# Authoring and parametrizing models

Every built-in physical model is a `ModelComponent` (`energysim.models.common`), a `SimulatorAdapter` with an
ergonomic contract. You can write your own in a few lines, reuse the lookup helpers (`Curve1D`, `Surface2D`) and the
`OperatingStateMachine`, or just re-parametrize an existing model.

## What `ModelComponent` gives you

| You write | The framework provides |
|---|---|
| `register_variables(inputs=[...], outputs=[...], defaults={...})` in `__init__` | strict variable access: `set_value` accepts only declared inputs and registered parameters, `get_value` only declared outputs / inputs / parameters; anything else raises `SimulatorVariableError` listing the valid names |
| `register_parameters('a', 'b')` | numeric attributes that `set_value`, `options({'init': ...})`, [ensembles](ensemble.md) and sweeps may overwrite |
| `step_model(dt, time)` | `step` / `step_partial` / `advance` (a shorter final step gets the true `dt`) |
| a class attribute `units = {'var': 'unit'}` | unit-aware connections and `Results.units`, see [Units](units.md) |
| nothing | outputs that are valid right after `init()` (evaluated with `step_model(0.0, 0.0)` and the state restored), re-runnable `init()` / `reset()` (the first `init()` snapshots the instance, later ones restore it), `describe()` / `describe_class()` |
| optionally `time_to_event()` | [event hints](events.md) for variable macro steps |

Rules of thumb: create all state in `__init__` (it is snapshotted by the first `init()`), make `step_model(0.0, t)`
change no state (quiescent at `dt = 0`), write every declared output into `self._outputs_val`, read inputs from
`self._inputs_val`, and give every input and output a unit.

## A complete example: a supercapacitor

```python
import math

from energysim import world
from energysim.base import SimulatorVariableError
from energysim.models.common import ModelComponent


class Supercap(ModelComponent):
    """Supercapacitor module: dV/dt = (I - I_leak) / C and P = V * I."""

    units = {"P_cmd": "kW", "P_actual": "kW", "V": "V", "SoC": "1"}

    def __init__(self, name="supercap", capacitance_f=500.0, v_rated=48.0, i_leak_a=0.002, v_init=24.0,
                 step_size=1.0):
        super().__init__(name=name, step_size=step_size)
        self.capacitance_f = float(capacitance_f)
        self.v_rated = float(v_rated)
        self.i_leak_a = float(i_leak_a)
        self.v = float(v_init)
        self.register_variables(inputs=["P_cmd"], outputs=["P_actual", "V", "SoC"], defaults={"P_cmd": 0.0})
        self.register_parameters("capacitance_f", "i_leak_a")

    def step_model(self, dt, time):
        p_w = 1000.0 * self._inputs_val["P_cmd"]
        i = p_w / max(self.v, 1.0)
        if (i > 0 and self.v >= self.v_rated - 0.01) or (i < 0 and self.v <= 1.0):
            i = 0.0                                             # full / empty
        self.v = min(self.v_rated, max(0.0, self.v + (i - self.i_leak_a) / self.capacitance_f * dt))
        self._outputs_val["P_actual"] = self.v * i / 1000.0
        self._outputs_val["V"] = self.v
        self._outputs_val["SoC"] = (self.v / self.v_rated) ** 2

    def time_to_event(self):                                    # seconds until the module is full
        i = 1000.0 * self._inputs_val["P_cmd"] / max(self.v, 1.0)
        if i <= 0 or self.v >= self.v_rated:
            return math.inf
        return (self.v_rated - self.v) * self.capacitance_f / i


cap = Supercap()
cap.init()
print(cap.get_value(["V", "SoC"], 0.0))                       # valid before the first step: [24.0, 0.25]
print(cap.describe()["inputs"], cap.describe()["units"]["V"])  # ['P_cmd'] V
try:
    cap.set_value(["nonsense"], [1.0])
except SimulatorVariableError as exc:
    print(exc)                      # supercap: 'nonsense' is not a declared input or parameter. ...

w = world(start_time=0, stop_time=600, t_macro=60, t_macro_min=1)
w.add_signal("cmd", {"type": "step", "time": 120, "before": 0.0, "after": 5.0}, step_size=60, units={"y": "kW"})
w.add_model("cap", cap, step_size=1)
w.add_connections({"cmd.y": "cap.P_cmd"})

res = w.simulate(pbar=False, record_all=False)
print(res["cap.V"].iloc[-1])                                  # full at 48 V
print([round(t, 1) for t in res["cap"].index[:7]])            # steps shrink towards the full point
print(res.units["cap.V"], w.simulate(pbar=False, record_all=False)["cap.V"].iloc[-1])   # re-runnable

w2 = world(start_time=0, stop_time=60, t_macro=60)
w2.add_model("cap", Supercap, v_init=12.0)                    # a class: constructed with the extra keywords
print(w2.get_adapter("cap").v)                                # 12.0
```

`add_model(name, model, ...)` also accepts a class or factory (`w.add_model("cap", Supercap, v_init=12.0)`; extra
keyword arguments are passed to it). Inputs and outputs default to the registered ones; pass `outputs=[...]` to
record fewer.

## Lookup curves and surfaces

`Curve1D(x, y, clamp=True, min_val=None, max_val=None, name='')` interpolates piecewise linearly and, by default,
**clamps** queries to the data range (no wild extrapolation) and to optional output bounds. `Surface2D(x, y, z)` is
the bilinear counterpart with `z.shape == (len(x), len(y))`. Both are immutable by convention, are shared (not
copied) by the `init()` snapshot, and serialise with `to_dict()` / `from_dict()`. Like `Curve1D`, `Surface2D` accepts
NumPy arrays of query points, so one call evaluates a whole vectorized fleet.

```python
from energysim.models.common import Curve1D, Surface2D

load = [0.2, 0.4, 0.6, 0.8, 1.0]
eta = [0.28, 0.36, 0.42, 0.44, 0.435]            # efficiency from a test bench (your data)
eff = Curve1D(load, eta, min_val=0.1, max_val=0.5, name="efficiency")

print(eff(0.72))                # 0.432 (interpolated)
print(eff(2.0))                 # 0.435 (clamped to the last point)
print(eff([0.3, 0.5]))          # array([0.32, 0.39])
print(Curve1D.from_dict(eff.to_dict())(0.72))

derate = Surface2D(x=[0, 10, 20], y=[0, 1], z=[[1, 2], [3, 4], [5, 6]])
print(derate(5, 0.5))           # 2.5
```

Several built-in models take curves as constructor arguments, for example `BatteryBESS(ocv_curve=..., rin_curve=...,
inverter_curve=...)`, `GasTurbine_OCGT(ambient_derate_curve=..., heat_rate_curve=...)` and
`CombinedCycle_CCGT(ambient_derate_curve=...)`.

## Start-up timers and ramps: `OperatingStateMachine`

Thermal plants cannot jump to full load. `OperatingStateMachine` tracks the state (`OFF`, `STANDBY`, `PURGE`,
`STARTING`, `RUNNING`, `COOLDOWN`, `FAULT`), how long the unit was offline (which picks a hot / warm / cold start
duration), minimum up and down times, and ramp limits. You drive it from `step_model`:

```python
from energysim import world
from energysim.models.common import ModelComponent, OperatingState, OperatingStateMachine


class Peaker(ModelComponent):
    """Gas peaker with a 5 minute start-up and ramp limits (MW)."""

    units = {"P_cmd": "MW", "P_actual": "MW", "state": "-"}

    def __init__(self, p_rated_mw=25.0, step_size=10.0):
        super().__init__(name="peaker", step_size=step_size)
        self.p_rated_mw = p_rated_mw
        self.sm = OperatingStateMachine(ramp_up_rate_min=5.0, ramp_down_rate_min=10.0,
                                        hot_start_duration_sec=300.0, warm_start_duration_sec=300.0,
                                        cold_start_duration_sec=300.0)
        self.p = 0.0
        self.register_variables(inputs=["P_cmd"], outputs=["P_actual", "state"], defaults={"P_cmd": 0.0})

    def step_model(self, dt, time):
        sm = self.sm
        cmd = min(self._inputs_val["P_cmd"], self.p_rated_mw)
        sm.update_time(dt)
        if cmd > 0 and sm.state in (OperatingState.OFF, OperatingState.STANDBY):
            sm.current_startup_category = sm.determine_startup_category()
            sm.transition(OperatingState.STARTING)
        if sm.state == OperatingState.STARTING and sm.time_in_state_sec >= sm.get_startup_duration():
            sm.transition(OperatingState.RUNNING)
        if sm.state == OperatingState.RUNNING:
            self.p = sm.apply_ramp(self.p, cmd, dt)
            if cmd <= 0 and self.p <= 0:
                sm.transition(OperatingState.OFF)
        self._outputs_val["P_actual"] = self.p
        self._outputs_val["state"] = sm.state.value


w = world(start_time=0, stop_time=1200, t_macro=60)
w.add_signal("cmd", 20.0, step_size=60, units={"y": "MW"})
w.add_model("gt", Peaker(), step_size=10)
w.add_connections({"cmd.y": "gt.P_cmd"})
res = w.simulate(pbar=False, record_all=False)
print(res["gt"].loc[[240.0, 360.0, 540.0], ["P_actual", "state"]])      # STARTING -> RUNNING -> ramped to 20 MW
```

The built-in `GasTurbine_OCGT`, `CombinedCycle_CCGT`, `ReciprocatingGenset` and `CHP_SteamTurbine` use the same
machine; their `time_to_event()` reports when a start-up or minimum up/down timer ends.

## Re-parametrizing a built-in model

Constructor arguments are the parameters. `describe()` lists them with their defaults, inputs, outputs and units;
the generated [component pages](physical-models.md) document each model's variables and validity range.

```python
from energysim.models import CombinedCycle_CCGT, PEMElectrolyzer
from energysim.models.common import Curve1D

# Site-specific ambient derating for a combined cycle (multiplier on rated power vs ambient degC)
derate = Curve1D([-10, 0, 15, 30, 45], [1.10, 1.05, 1.00, 0.90, 0.78], name="site derate")
ccgt = CombinedCycle_CCGT(name="site_ccgt", p_rated_mw=250.0, min_stable_gen_pct=30.0,
                          ramp_rate_mw_min=20.0, ambient_derate_curve=derate)

pem = PEMElectrolyzer(name="pem", p_nominal_mw=5.0, min_turndown=0.05, t_stack_nom_c=65.0)

print(sorted(ccgt.describe()["parameters"])[:4])
print(pem.describe()["outputs"][:3])
```

A model whose parameters are inconsistent refuses to construct: for example a `PEMElectrolyzer` whose
`p_nominal_mw` exceeds what its cell count and area can absorb raises a `ValueError` that names the fix.

## Your own archetype catalog

The built-in `energysim.models.catalog` holds generic archetypes (`BESS_2h_2MW`, `BESS_4h_2MW`, `Alkaline_2MW`,
`PEM_5MW`, `SOEC_1MW`, `CCGT_570MW`, `Gas_Turbine_60MW`, `Gas_Engine_10MW`, `Solar_Farm`, `Wind_Turbine_4MW`), each
an **illustrative archetype with a provenance note** (`catalog.PROVENANCE`), not a manufacturer datasheet. The
manufacturer-branded names of 2.x (`Tesla_Megapack_2XL`, `Nel_A1000`, `GE_7HA_02`, ...) are deprecated aliases that
emit a `DeprecationWarning`. Put your own verified equipment into a module of factories in the same style:

```python
from energysim.models import BatteryBESS, catalog


def substation_battery(name="booster", **kwargs):
    """2 MW / 2 MWh battery for our substations (parameters from our own acceptance tests)."""
    params = dict(capacity_mwh=2.0, p_max_mw=2.0, soc_init=0.8)
    params.update(kwargs)
    return BatteryBESS(name=name, **params)


bess = substation_battery(soc_init=0.6)
generic = catalog.BESS_2h_2MW(name="reference")
print(bess.capacity_mwh, generic.capacity_mwh, catalog.PROVENANCE["BESS_2h_2MW"])
```

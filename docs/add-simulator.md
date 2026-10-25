# Simulator adapters

Every simulator enters a world through `add_simulator(sim_type, sim_name, model=..., inputs, outputs, step_size,
mode, power_unit, units, **kwargs)` or through a dedicated method (`add_signal`, `add_csv`, `add_powerflow`, ...).
Both forms build the same adapter. After a run, `world.get_adapter(name)` returns the adapter and `world.get_model(name)`
the native object inside it.

| `sim_type` | Dedicated method | `model` is | Needs |
|---|---|---|---|
| `'signal'` | `add_signal` | number, callable, expression string, structured spec | core |
| `'csv'` | `add_csv` | path of a CSV with a `time` column | core |
| `'script'` | `add_script` | path of a `.py` file (or a directory), plus `function=` | core |
| `'model'` | `add_model` | a `ModelComponent` / adapter instance, class or factory | core |
| `'external'` | `add_external_simulator` | adapter instance, class, `'file.py:Class'` | core |
| `'container'`, `'fleet'`, `'battery_fleet'`, `'building_fleet'`, `'pv_fleet'`, `'function_fleet'` | `add_container` / `add_fleet` | fleet adapter, or built from `kind=`, `size=` | core |
| `'powerflow'` (`'pandapower'`) | `add_powerflow` | pandapower net, path (`.json`, or pickle `.p` / `.pkl` / `.pickle`; pickle is refused in safe mode) or factory | `energysim[powerflow]` |
| `'pypsa'` | `add_pypsa` | PyPSA `Network`, path or factory | `energysim[pypsa]` |
| `'opendss'` | `add_opendss` | path of the master `.dss` file | `energysim[opendss]` |
| `'fmu'` | `add_fmu` | path of an `.fmu` | `energysim[fmu]` |
| `'powerfactory'` | `add_powerfactory` | project name | PowerFactory + licence |
| `'matlab'` | `add_matlab` | path of a `.m` function file | MATLAB Engine or `oct2py` + Octave |

Common arguments:

* `model`: the object, path or name. Aliases (`sim_loc`, `net_loc`, `net`, `network`, `path`, `fmu_path`, `csv_path`,
  `dss_path`, `script_path`, `project_name`, `container`, ...) are accepted; giving two different ones raises.
* `inputs`, `outputs`: variable names; `outputs` are recorded in the results.
* `step_size`: micro step in seconds. The macro step `t_macro` should be a multiple of it (`validate()` warns).
* `mode`: `'ac'` (default) or `'dc'` for power-flow simulators (`'dc'` is not available for OpenDSS).
* `units`, `power_unit`: unit declarations, see [Units](units.md).
* `allow_import=True`: allow a dotted reference such as `'pandapower.networks.case14'` to be imported and called
  (never in [safe mode](safe-mode.md)).

## Signals

A signal is a source with one output, `y`. It is read-only: a connection into it warns and is discarded.

```python
import math

from energysim import world

w = world(start_time=0, stop_time=7200, t_macro=900)
w.add_signal("const", 5.0, step_size=900)                                           # a number
w.add_signal("expr", "3 + 2 * sin(2 * pi * t / 3600)", step_size=900)               # a safe expression of t [s]
w.add_signal("step", {"type": "step", "time": 3600, "before": 0, "after": 10}, step_size=900)   # a structured spec
w.add_signal("fn", lambda t: 10.0 * math.cos(t / 1800.0), step_size=900)            # a Python callable
res = w.simulate(pbar=False, record_all=False)
print(res["expr.y"].round(2).tolist()[:3], res["step.y"].tolist()[3:6])
```

Strings are parsed by the safe-expression grammar and never `eval`-ed (functions `sin cos tan exp log sqrt abs min max
floor ceil`, constants `pi`, `e`, the variable `t`; an optional `lambda t:` prefix is accepted). Structured specs:
`constant`, `sine` (`amplitude`, `frequency` or `period`, `phase`, `offset`), `step`, `ramp`, `piecewise`
(`breakpoints: [[t, v], ...]`, `interpolation: linear | previous`) and `expression`. A callable may return a number or
a one-element list. `breakpoints=[...]` (and the breakpoints of step / ramp / piecewise specs) are
[event hints](events.md).

## CSV profiles

A CSV with a `time` column in seconds at a **fixed interval** and one column per variable. The value at time `t` is the
row with the largest `time <= t` (zero-order hold), clamped to the first and last row. Unknown output columns raise
`SimulatorVariableError` immediately; CSV simulators are read-only.

```python
from pathlib import Path

from energysim import world

Path("profile.csv").write_text("time,load_kw,temp_c\n0,100,5\n900,120,6\n1800,150,7\n2700,130,7\n3600,110,6\n",
                               encoding="utf-8")

w = world(start_time=0, stop_time=3600, t_macro=900)
w.add_csv("profile", "profile.csv", step_size=900, outputs=["load_kw", "temp_c"], units={"load_kw": "kW"})
res = w.simulate(pbar=False, record_all=False)
print(res["profile.load_kw"].tolist())          # [100, 120, 150, 130, 110]
```

`add_csv(..., delimiter=';')` changes the separator; `align_to_rows=True` reports the next row as an event hint.

## Python script functions

`py_adapter` calls `function(inputs: dict) -> list` once per micro step. The module is loaded **from its file** under a
unique name (two scripts with the same file name never collide, `sys.path` is not changed permanently), and a world
can be simulated repeatedly.

```python
from pathlib import Path

from energysim import world

Path("controller.py").write_text('''
def control(inputs):
    soc = inputs.get("soc", 0.5)
    return [1.0 if soc < 0.5 else -1.0]       # charge below 50 %, otherwise discharge
''', encoding="utf-8")

w = world(start_time=0, stop_time=600, t_macro=60)
w.add_simulator("script", "ctrl", model="controller.py", function="control",
                inputs=["soc"], outputs=["p_cmd"], step_size=60, units={"soc": "1", "p_cmd": "MW"})
w.add_signal("soc", 0.4, step_size=60, units={"y": "1"})
w.add_connections({"soc.y": "ctrl.soc"})
print(w.simulate(pbar=False, record_all=False)["ctrl.p_cmd"].tolist()[-3:])     # [1.0, 1.0, 1.0]
```

Scripts execute arbitrary code; they are refused in safe mode.

## Built-in models and your own classes

`add_model(name, model, ...)` takes an adapter instance, or a class / factory plus keyword arguments. See
[Physical components](physical-models.md) for the library and [Custom models](custom-models.md) /
[External simulators](external-simulators.md) to write your own.

```python
from energysim import world
from energysim.models import PhysicsPV, catalog

w = world(start_time=0, stop_time=3600, t_macro=900)
w.add_signal("sun", "800", step_size=900)
w.add_model("pv", PhysicsPV(p_dc_stc_mw=2.0, inverter_p_ac_max_mw=1.6), step_size=900)
w.add_model("bess", catalog.BESS_2h_2MW, step_size=60, soc_init=0.3)       # a factory with keyword arguments
w.add_connections({"sun.y": "pv.G_poa"})
print(w.simulate(pbar=False, record_all=False)["pv.P_ac"].round(2).tolist())
```

## Vectorized fleets

`add_fleet` / `add_container` with `fleet_type='battery' | 'building' | 'pv' | 'function'` (or a fleet `sim_type`)
simulates thousands of identical units in one NumPy step. Outputs are available as `total_<var>`, `mean_<var>`,
`min_<var>`, `max_<var>` and per entity as `<var>[i]`; a scalar input is broadcast to all units. See
[Vectorized fleets](containers/overview.md).

```python
from energysim import world

w = world(start_time=0, stop_time=3600, t_macro=900)
w.add_fleet(fleet_type="battery", name="evs", size=100, capacity_kwh=50.0, max_charge_kw=11.0,
            initial_soc=0.2, step_size=900)
w.add_signal("dispatch", 7.4, step_size=900, units={"y": "kW"})
w.add_connections({"dispatch.y": "evs.p_cmd"})
res = w.simulate(pbar=False, record_all=False)
print(round(res["evs.mean_soc"].iloc[-1], 3))
```

## pandapower

Variables are `"<element name>.<column>"`, where the name is the `name` of a `load`, `sgen`, `gen`, `storage`,
`ext_grid`, `line`, `trafo` or `bus`. Results come from the `res_<table>` frame when the column exists there
(`Bus2.vm_pu`, `Feeder.loading_percent`, `Feeder.p_from_mw`), otherwise from the element table (inputs such as
`Load1.p_mw`, `Gen1.vm_pu`). Prefix `table:` to disambiguate duplicate names (`"bus:Grid.vm_pu"`). The adapter keeps a
pristine copy of the network and restores it on the next `init()`, so a world can be re-run. `mode='ac'` runs
`runpp`, `mode='dc'` `rundcpp`. pandapower works in MW / MVar; connections convert from kW automatically.

```python
import pandapower as pp

from energysim import world

net = pp.create_empty_network(name="feeder")
b1 = pp.create_bus(net, vn_kv=20.0, name="Bus1")
b2 = pp.create_bus(net, vn_kv=20.0, name="Bus2")
pp.create_ext_grid(net, b1, name="Grid")
pp.create_line(net, b1, b2, length_km=2.5, std_type="NAYY 4x150 SE", name="Feeder")
pp.create_load(net, b2, p_mw=1.2, q_mvar=0.3, name="Load1")

w = world(start_time=0, stop_time=3600, t_macro=900)
w.add_powerflow("grid", net=net, inputs=["Load1.p_mw"],
                outputs=["Bus2.vm_pu", "Feeder.loading_percent"], step_size=900)
w.add_signal("demand", "800 + 400 * t / 3600", step_size=900, units={"y": "kW"})
w.add_connections({"demand.y": "grid.Load1.p_mw"})
res = w.simulate(pbar=False, record_all=False)
print(res["grid.Bus2.vm_pu"].round(4).tolist())
print(w.get_model("grid") is w.get_adapter("grid").network)          # the live pandapower net
```

Optimal power flow is not a `mode`; use `energysim.control.OPFController` ([Control](control.md)).

## PyPSA

One snapshot is solved per macro step (`n.pf()` for `mode='ac'`, `n.lpf()` for `'dc'`); multi-period optimisation is
done by the [rolling-horizon controller](control.md). Variables are `"<component>.<variable>"` with an optional type
prefix to disambiguate (`"Bus:b1.V"`; Generator, Load, StorageUnit, Bus, Line, Transformer):

* inputs: `P`, `Q` of generators, loads and storage units (written to `p_set` / `q_set`; a generator with
  `control='Slack'` ignores `p_set` in a power flow);
* outputs: `P`, `Q` of generators / loads / storage units; for buses `V` (p.u.), `Va`, `Va_deg`, `P`, `Q`; for
  lines and transformers `p0`, `q0`, `p1`, `q1`, `loading_percent`. In `'dc'` mode reactive outputs are 0 and `V` is 1.

PyPSA 0.30 up to (excluding) 2.0 is supported. With PyPSA 1.x the active power of a slack generator is only filled in
when that generator has a `p_set`; bus `P` is always available.

```python
import pypsa

from energysim import world

n = pypsa.Network()
n.set_snapshots([0])
n.add("Bus", "b1", v_nom=20.0)
n.add("Bus", "b2", v_nom=20.0)
n.add("Generator", "grid", bus="b1", control="Slack", p_nom=10.0, p_set=0.0)
n.add("Load", "load", bus="b2", p_set=1.0, q_set=0.2)
n.add("Line", "l12", bus0="b1", bus1="b2", x=0.5, r=0.2, s_nom=5.0)

w = world(start_time=0, stop_time=3600, t_macro=900)
w.add_pypsa("net", net=n, inputs=["load.P"], outputs=["b2.V", "l12.loading_percent", "b1.P"], step_size=900)
w.add_signal("demand", "1 + 0.5 * t / 3600", step_size=900, units={"y": "MW"})
w.add_connections({"demand.y": "net.load.P"})
res = w.simulate(pbar=False, record_all=False)
print(res["net.l12.loading_percent"].round(1).tolist())
```

## OpenDSS

OpenDSS runs through OpenDSSDirect.py, which is a **process-wide singleton**: only one circuit can be active per
Python process (a second active adapter raises `EnergysimError`). Only AC snapshot power flow exists: `mode='dc'`
raises `ValueError`. Each step re-solves the circuit; time is not advanced inside OpenDSS, so drive time-varying
behaviour through inputs.

* inputs: `"<Type>.<Name>.<Property>"` such as `Load.ld1.kW`, `Generator.pv.kW`;
* outputs: `Bus.<bus>.vm_pu` (mean per-unit voltage; `.vm_pu.<node>` for one node), `Bus.<bus>.vm_kv`, `Bus.<bus>.vang`,
  `<Type>.<Name>.p_kw` / `q_kvar` / `losses_kw` / `i_a` / `loading_percent`, and `Circuit.losses_kw`,
  `Circuit.losses_kvar`, `Circuit.total_power_kw`.

```python
from pathlib import Path

from energysim import world

Path("feeder.dss").write_text("""
Clear
New Circuit.feeder basekv=11 pu=1.0 phases=3 bus1=SourceBus
New Line.l1 bus1=SourceBus bus2=b1 length=1 units=km
New Load.ld1 bus1=b1 phases=3 kv=11 kw=500 pf=0.95
Set voltagebases=[11]
Calcvoltagebases
Solve
""", encoding="utf-8")

w = world(start_time=0, stop_time=3600, t_macro=900)
w.add_opendss("dss", dss_loc="feeder.dss", inputs=["Load.ld1.kW"],
              outputs=["Bus.b1.vm_pu", "Load.ld1.p_kw", "Circuit.losses_kw"], step_size=900)
w.add_signal("demand", "500 + 200 * t / 3600", step_size=900, units={"y": "kW"})
w.add_connections({"demand.y": "dss.Load.ld1.kW"})
res = w.simulate(pbar=False, record_all=False)
print(res["dss.Load.ld1.p_kw"].tolist())
```

## FMUs (FMI 1.0 and 2.0)

`add_fmu` reads the model description once and creates a co-simulation adapter (`FmuCsAdapter`) if the FMU has a
co-simulation interface, otherwise a model-exchange adapter (`FmuMeAdapter`, CVode through `fmpy.sundials`, or
`solver_name='Euler'`). Start values from `options({'init': ...})` / `set_parameters` are applied **before** the FMU
leaves initialisation mode. FMU adapters are re-runnable (the FMU is re-instantiated), `validate=False` skips FMPy's
model-description validation, and model-exchange FMUs report their time events as step hints. Variable names are the
FMU's `modelVariables` names (including `der(x)` for derivatives).

The FMUs in `examples/Optimal Dispatch of FER/` were exported for a specific platform, so this snippet is not executed
by the documentation tests:

```python notest
from energysim import world

w = world(start_time=0, stop_time=3600, t_macro=300)
w.add_fmu("plant", "models/plant.fmu", inputs=["cmd"], outputs=["temp"], step_size=1,
          units={"cmd": "kW", "temp": "degC"}, validate=False)
w.options({"init": {"plant": (["tank_volume"], [2.0])}})          # start values, applied before init
```

## DIgSILENT PowerFactory

Requires PowerFactory 2020 or later with the Python API and a licence; CI uses a mock, the real adapter is checked
manually (`tools/powerfactory_manual_check.py`). The adapter activates the project, runs the load flow (`ComLdf`;
`mode='dc'` selects the DC option) every micro step, writes inputs with `SetAttribute` (`e:` prefix added if missing)
and reads outputs with `GetAttribute` (`m:` result attributes). Variables are `"<loc_name>.<attribute>"`.

```python notest
from energysim import world

w = world(start_time=0, stop_time=3600, t_macro=900)
w.add_powerfactory("IEEE13", model="IEEE13", inputs=["Load1.plini"],
                   outputs=["Bus1.m:u", "Line1.m:loading"], step_size=900,
                   pf_path=r"C:\Program Files\DIgSILENT\PowerFactory 2024\Python\3.11")   # needs a licence
```

## MATLAB and GNU Octave

`.m` **functions** (not scripts) with the signature `function [out1, ...] = name(in1, ..., time)`: inputs in the order of
`inputs`, `time` (seconds) appended last, outputs in the order of `outputs`. The file name must equal the function name.
`engine='auto'` picks the MATLAB Engine for Python if installed, otherwise Octave through `oct2py`. Use `persistent`
variables for state.

```python notest
from energysim import world

w = world(start_time=0, stop_time=3600, t_macro=900)
w.add_matlab("heatpump", model="models/heatpump.m", inputs=["P_cmd", "T_source", "T_sink"],
             outputs=["Q_thermal", "COP"], step_size=900, engine="auto")      # needs MATLAB or Octave
```

## Initial values, parameters and signal modification

```python
from energysim import world
from energysim.models import BatteryBESS

w = world(start_time=0, stop_time=600, t_macro=60)
w.add_model("bess", BatteryBESS(capacity_mwh=2.0, p_max_mw=1.0), step_size=60)
w.add_signal("cmd", 1.0, step_size=60, units={"y": "kW"})        # a kW signal into a MW input ...
w.add_connections({"cmd.y": "bess.P_cmd"})                       # ... is converted: 1 kW -> 0.001 MW

w.options({
    "init": {"bess": (["P_cmd"], [-0.5])},                       # initial values / start values
    "parameter_set": {"bess": {"capacity_mwh": 4.0}},            # registered parameters; applied at the start of every run
    "modify_signal": {"cmd.y": [2.0, 0.0]},                      # gain [and offset] on the source value, before unit conversion
    "record_all": False,                                         # default for simulate(record_all=...)
})
res = w.simulate(pbar=False)
print(w.get_adapter("bess").capacity_mwh, res["bess.P_actual"].iloc[-1])      # 4.0 and the applied power [MW]
```

`options` accepts exactly the keys `init`, `parameter_set`, `modify_signal` and `record_all`; anything else raises
`ValueError`. Initial values are applied at the start of every run, before or after `init()` as the adapter requires.
Prefer unit declarations or `from_unit` / `to_unit` on the connection over `modify_signal` for unit changes.

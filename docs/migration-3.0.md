# Migrating from 2.x to 3.0

3.0 is a breaking release. The public entry points are unchanged (`from energysim import world, World`,
`add_simulator`, `add_connections`, `options`, `simulate`, `from_yaml`, `Sweep`), but results, coupling, units and
scenario loading behave differently. This page lists every breaking change with the replacement. The release notes
are in `CHANGELOG.txt`.

## At a glance

| Area | 2.x | 3.0 |
|---|---|---|
| Iterative coupling | `world(coupling='iterative', max_iterations=10, convergence_tolerance=1e-6)` | **Removed.** `coupling='iterative'`, `max_iterations`, `convergence_tolerance` raise `ValueError` (also as YAML `world:` keys). Use `'jacobi'` (default) or `'gauss-seidel'`; loops get a one-macro-step delay. Reduce `t_macro` or merge the coupled components to shrink it. |
| Rollback | `adapter.save_state()` / `adapter.restore_state(state)` | **Removed** from `SimulatorAdapter`, `ModelComponent` and FMU adapters. Delete these methods from your adapters. `add_trigger(..., bisect=True)` raises `NotImplementedError`. |
| Results | `world.simulate()` returned the string `'Done'`; `res = world.results()` gave a dict of DataFrames with a `time` column | `res = world.simulate()` returns a `Results`; `world.results()` is gone. `res['sim']` is a DataFrame indexed by time, `res['sim.var']` a Series. Also `world.last_results`. |
| Dashboard | `world.results(dashboard=True, dashboard_path=...)` | `res.dashboard(path, open=True)`; CLI `--dashboard` / `--dashboard-path`. |
| Result files | `world(res_filename='es_res.h5')`, HDF5 written on every run, `convert_hdf_to_dict`, `convert_to_df`, `export_to_csv`, `record_data`, ... | **Nothing is written by default.** `res_filename` (also as YAML key and CLI `--res-filename`) raises. Persist with `world(out_dir='runs')` / `--out DIR` (one `<timestamp>_<hash>` directory per run: Parquet or CSV, `meta.json`, `scenario.yaml`), `res.save(path)`, `res.to_csv(dir)`, `res.to_hdf(path)`; reload with `Results.load`. The HDF5 helpers were removed from `energysim` and `energysim.utils`. |
| Rows | The last row was the last macro step | Every simulator has a final row at `stop_time` (state at the end of the run). |
| Recording | HDF5 appends per step | Typed per-simulator tables in memory; `record_all=True` (the `simulate()` default) records every micro step. |
| Dependencies | `tables` (HDF5) was a core dependency | Core: numpy, pandas, tqdm, networkx, pyyaml. `pip install energysim[results]` (pyarrow, Parquet) and `energysim[hdf5]` (tables). Without pyarrow `res.save()` falls back to CSV. |
| Units | Best-effort scaling of known units; `world(power_unit='kW')` default | Every connected variable is resolved to a unit (declared, component, connection, name suffix, unit-system default) and converted into the receiver's unit. Unknown unit names and incompatible dimensions raise `ValueError`. `power_unit` defaults to `None` (the unit system's active-power unit). Unspecified units pass through unscaled with a warning; `unit_policy='strict'` makes them an error. See [Units](units.md). |
| Signals from strings | `add_signal('x', 'lambda t: np.sin(t)')` and YAML `signal:` strings were `eval`-ed | Strings are parsed by a whitelisted evaluator (`sin cos tan exp log sqrt abs min max floor ceil`, `pi`, `e`, `t`, arithmetic, comparisons, `if/else`; optional `lambda t:` prefix). Anything else (`np.`, `math.`, attributes, imports) raises `UnsafeExpressionError`. Python callables still work in the Python API; structured specs (`{type: sine, ...}`) were added. |
| Signal return values | Callable had to return a one-element list | A scalar, list or tuple is accepted. |
| Dotted imports | `model: pandapower.networks.case14` was imported and called silently | Needs `allow_import=True` (`add_simulator(..., allow_import=True)`, `from_yaml(..., allow_import=True)`), else `ValueError`. Always refused in safe mode. |
| Untrusted scenarios | none | `world.from_yaml(path, safe=True)` / `energysim run --safe` / `energysim validate --safe`. See [Safe mode](safe-mode.md). |
| Exceptions | `ConnectionError` (shadowed the builtin) | `ConnectionSpecError`; `energysim.ConnectionError` and `energysim.base.ConnectionError` remain as aliases. |
| Adapter lifecycle | `init()` then steps | `configure(start, stop, macro_step)`, then `set_parameters`/initial values, then `init()`. FMUs apply start values before `init()` (`SET_PARAMS_BEFORE_INIT = True`). A world can be simulated repeatedly: `cleanup()` is always called and `init()` re-creates state. |
| `set_parameters` / `options({'init': ...})` | Applied at once | Registered and applied at the start of **every** run. Unknown option keys raise `ValueError`. |
| Step size | `stop_time - start_time` had to be a multiple of `t_macro` | Still required for fixed steps (`ValueError` otherwise); a variable-step run (`t_macro_min < t_macro`) lifts it. Final micro step may be shorter (`step_partial`). |
| Events | none | `world.schedule`, `world.add_trigger`, `t_macro_min`, adapter hints, YAML `events:` / `triggers:`. See [Events](events.md). |
| Sweeps | `Sweep` modified/reused the world; `evaluation_function` | `Sweep` is built on `Ensemble`: every run uses a fresh deep copy of the world (or your factory); `sweep()` returns a `SweepResult` (`table`, `baseline`, `ensemble`, `export_to_csv`, `plot`). New `Ensemble` for samplers, noise and Monte-Carlo studies. See [Ensembles](ensemble.md). |
| Model components | `set_value` wrote any attribute of that name, `get_value` returned `0.0` for unknown names | `set_value` / `get_value` raise `SimulatorVariableError` for undeclared names; outputs are valid right after `init()`; `init()` / `reset()` restore the initial state; `describe()`, `units`, `register_parameters`, `time_to_event`. See [Custom models](custom-models.md). |
| Equipment catalog | `catalog.Tesla_Megapack_2XL`, `Nel_A1000`, `GE_7HA_02`, ... described as calibrated to datasheets | Generic archetypes (`BESS_2h_2MW`, `BESS_4h_2MW`, `Alkaline_2MW`, `PEM_5MW`, `SOEC_1MW`, `CCGT_570MW`, `Gas_Turbine_60MW`, `Gas_Engine_10MW`, `Solar_Farm`, `Wind_Turbine_4MW`) with a `PROVENANCE` note. The brand names still work as **deprecated aliases** (`DeprecationWarning`) with their old numbers and will be removed. |
| Script adapter | module imported via `sys.path` | Loaded from its file under a unique module name per instance; `sys.path` is not modified. |
| External simulators | directory + `<sim_name>.py` containing `external_simulator` | Additionally: an instance, a class / factory, `'path/file.py:ClassName'`, `'path/file.py'`. The directory form still works. |
| CSV adapter | an unknown output column failed late with a pandas error | `SimulatorVariableError` at construction and on read. `align_to_rows=True` adds event hints. |
| OpenDSS | `mode='dc'` accepted | `mode='dc'` raises `ValueError` (OpenDSS has no DC power flow). |
| PyPSA | 0.x variable names | PyPSA 1.x is supported (`>=0.30,<2`); outputs per component type are listed in [Simulator adapters](add-simulator.md). |
| OPF | described as a pandapower feature in the 2.x docs | OPF is `energysim.control.OPFController`, not a power-flow `mode`; `mode` is `'ac'` or `'dc'` only. See [Control](control.md). |
| Containers | lenient addressing | Strict variable addressing, `init`/`reset` snapshot, conservation of energy in the battery fleet (`loss_kw`). |
| World internals | single `world.py` | Mixins in `energysim._world`; `from energysim import world, World` is unchanged. Worlds are deep-copyable and picklable. |

## Code, before and after

### Coupling with feedback

```python notest
# 2.x: iterate until the exchanged values converge (needed save_state / restore_state)
w = world(start_time=0, stop_time=3600, t_macro=900,
          coupling='iterative', max_iterations=10, convergence_tolerance=1e-6)
```

```python
from energysim import world

# 3.0: explicit coupling. A feedback loop is detected, warned about and delayed by one macro step.
w = world(start_time=0, stop_time=3600, t_macro=300, coupling="gauss-seidel")
try:
    world(coupling="iterative")
except ValueError as exc:
    print(exc)       # iterative coupling was removed in 3.0; use 'jacobi' or 'gauss-seidel', reduce t_macro, ...
```

### Results

```python notest
# 2.x
my_world.simulate()
res = my_world.results(dashboard=False)           # dict of DataFrames, 'time' column
soc = res['bess']['SoC']
```

```python
from energysim import world
from energysim.models import BatteryBESS

w = world(start_time=0, stop_time=3600, t_macro=300)
w.add_signal("cmd", 0.5, step_size=300, units={"y": "MW"})
w.add_model("bess", BatteryBESS(capacity_mwh=4.0, p_max_mw=1.0), step_size=60)
w.add_connections({"cmd.y": "bess.P_cmd"})

res = w.simulate(pbar=False, record_all=False)       # a Results object (also w.last_results)
bess = res["bess"]                                   # DataFrame indexed by time [s]
soc = res["bess.SoC"]                                # Series
frame = res.to_frame()                               # wide frame, columns 'sim.var'
print(soc.iloc[[0, -1]].round(3).tolist(), res.units["bess.SoC"])
print(res.summary().loc["bess.SoC", ["min", "max"]].tolist())
res.save("out")                                      # Parquet (or CSV) + meta.json
```

### Persisting runs

| 2.x | 3.0 |
|---|---|
| `world(res_filename='run1.h5')` | `world(out_dir='runs')`, or `world.simulate(out_dir='runs')`, or `res.save('runs/run1')` |
| `energysim run s.yaml --res-filename run1.h5` | `energysim run s.yaml --out runs` (add `--to-csv` for CSV files) |
| `world.results(to_csv=True)` | `res.to_csv('csv_dir')` |

### Units

```python notest
# 2.x: scale and offset by hand, or rely on best-effort name matching
w.add_connections({'chp.power_w': {'to': 'grid.Gen1.p_mw', 'scale': 1e-6}})
```

A connection with a bare `scale` / `offset` and no unit keys still switches automatic conversion off for that
connection, so existing `scale`/`offset` connections keep their meaning. If you relied on a unit being guessed,
declare it, and use `unit_policy='strict'` to find the remaining gaps:

```python
from energysim import world
from energysim.models import BatteryBESS

w = world(start_time=0, stop_time=600, t_macro=300, unit_policy="strict")     # unspecified units raise
w.add_signal("chp", 2.0e6, step_size=300, units={"y": "W"})
w.add_model("bess", BatteryBESS(capacity_mwh=4.0, p_max_mw=1.0), step_size=60)
w.add_connections({"chp.y": "bess.P_cmd"})                       # W -> MW, factor 1e-6
print(w.connection_units()[["from_unit", "to_unit", "scale"]].to_dict("records"))
```

### Signals and imports in scenario files

```python notest
# 2.x: any Python expression was eval-ed
w.add_signal('irr', 'lambda t: max(0, np.sin(t/3600)) * 1000')
```

```python
import math

from energysim import world

w = world(start_time=0, stop_time=3600, t_macro=60)
w.add_signal("irr", "max(0, sin(t / 3600)) * 1000", step_size=60)                          # safe expression, no numpy
w.add_signal("irr2", {"type": "sine", "amplitude": 1000, "period": 7200}, step_size=60)    # or a structured spec
w.add_signal("irr3", lambda t: 1000.0 * max(0.0, math.sin(t / 3600)), step_size=60)         # or a callable (Python API)
print(w.simulate(pbar=False, record_all=False)["irr.y"].max())
```

### Custom adapters

```python notest
# 2.x
class external_simulator(SimulatorAdapter):
    def save_state(self): return {'soc': self.soc}
    def restore_state(self, state): self.soc = state['soc']
```

```python
from energysim import SimulatorAdapter, world


# 3.0: delete save_state / restore_state; re-create state in init(); optionally add event hints.
class Counter(SimulatorAdapter):
    def __init__(self, inputs=None, outputs=None, **kwargs):
        self.inputs, self.outputs = inputs or [], outputs or ["n"]
        self.n = 0

    def init(self):                      # called at the start of every run
        self.n = 0

    def step(self, time):
        self.n += 1

    def get_value(self, parameters, time):
        return [self.n for _ in parameters]

    def set_value(self, parameters, values):
        pass


w = world(start_time=0, stop_time=10, t_macro=5)
w.add_external_simulator("c", model=Counter, outputs=["n"], step_size=1)       # class, instance or 'file.py:Counter'
print(w.simulate(pbar=False, record_all=False)["c.n"].tolist())                # [0, 5, 10]
print(w.simulate(pbar=False, record_all=False)["c.n"].tolist())                # re-runnable: same result
```

## Things that did not change

* `add_simulator(sim_type, sim_name, model=..., inputs, outputs, step_size, mode, power_unit, units)`; the legacy
  keyword aliases (`sim_loc`, `net_loc`, `fmu_path`, ...) still work, but at most one may be given.
* Connection syntax: `{'a.y': 'b.u'}`, lists of `{'from', 'to', 'scale', 'offset', 'from_unit', 'to_unit'}`, fan-in
  (tuple of sources) and fan-out (list of targets).
* `options({'init': ..., 'parameter_set': ..., 'modify_signal': ..., 'record_all': ...})`.
* `world.validate()`, `world.plot()`, `get_adapter`, `get_model`, `to_yaml`, `to_dict`.
* `Sweep(world, sensitivity_info, kind=...)`, `export_to_csv`.

## Upgrade checklist

1. Replace `world.results()` with the `Results` returned by `simulate()`; delete HDF5 handling and `res_filename`.
2. Remove `coupling='iterative'`, `max_iterations`, `convergence_tolerance`, and every `save_state` / `restore_state`.
3. Run `energysim validate -v scenario.yaml` (or `world.validate()` and `world.connection_units()`) and fix every
   *unspecified* or *unknown* unit warning; consider `unit_policy='strict'`.
4. Replace Python expressions in signal strings by safe expressions or structured specs, or pass a callable.
5. Add `allow_import=True` where you load models by dotted path, and `--safe` / `safe=True` for scenario files you
   do not control.
6. Replace branded catalog names by the generic archetypes.
7. Install the extras you use: `results` (Parquet), `hdf5`, `control`, `networks`.

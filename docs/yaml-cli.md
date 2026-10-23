# YAML scenarios and the command line

A scenario file describes a world declaratively: time frame, simulators, connections, and optionally events and
triggers. The same file drives `energysim validate`, `energysim run` and `world.from_yaml`, which makes runs
reproducible, reviewable and runnable headless. Files from untrusted sources should be loaded in
[safe mode](safe-mode.md).

## A complete scenario

The snippet writes a CSV and a scenario file, validates and runs it through the command-line entry point, then loads
it from Python.

```python
from pathlib import Path

from energysim import world
from energysim.cli import main            # what the `energysim` command calls

Path("tariff.csv").write_text("time,price\n0,0.30\n1800,0.12\n3600,0.30\n5400,0.12\n7200,0.30\n",
                              encoding="utf-8")
Path("scenario.yaml").write_text("""
world:
  start_time: 0
  stop_time: 7200
  t_macro: 600
  t_macro_min: 60            # events and hints may shorten the step down to 60 s
  coupling: gauss-seidel     # or jacobi (default)
  unit_policy: lenient       # or strict
simulators:
  - name: tariff
    type: csv
    model: tariff.csv        # relative paths are resolved against the scenario file
    step_size: 600
    outputs: [price]
    units: {price: EUR/kWh}
  - name: solar
    type: signal
    signal: {type: piecewise, breakpoints: [[0, 0], [3600, 40], [7200, 0]]}
    step_size: 60
    units: {y: kW}
  - name: home
    type: battery_fleet
    step_size: 60
    options: {size: 10, initial_soc: 0.2}
    outputs: [mean_soc, total_actual_p_kw]
  - name: evs
    type: battery_fleet
    step_size: 60
    options: {size: 20, initial_soc: 0.3}
    outputs: [mean_soc]
connections:
  - {from: solar.y, to: home.p_cmd, from_unit: kW, to_unit: kW}
events:
  - {time: 1500, name: plug_in, set: {evs.p_cmd_kw: 7.4}}
triggers:
  - {name: enough, condition: "evs.mean_soc > 0.4", set: {evs.p_cmd_kw: 0.0}}
""", encoding="utf-8")

print(main(["validate", "scenario.yaml", "-v"]))                                    # 0 = valid
print(main(["run", "scenario.yaml", "--no-pbar", "--out", "runs", "--to-csv"]))     # 0 = success

w = world.from_yaml("scenario.yaml", safe=True)
res = w.simulate(pbar=False, record_all=False)
print([(e["name"], e["time"]) for e in res.meta["events"]])          # [('plug_in', 1500.0), ('enough', 4200.0)]
print(list(res["evs"].index)[:5])                                    # [0.0, 600.0, 1200.0, 1500.0, 2100.0]
print(w.to_yaml()[:60])                                              # the world serialised back to YAML
```

## Scenario schema

All top-level keys are optional except `simulators`; unknown `world:` keys are rejected.

| `world:` key | Meaning |
|---|---|
| `start_time`, `stop_time` | seconds (defaults 0 and 1000) |
| `t_macro`, `t_macro_min` | maximum / minimum macro step, see [Events](events.md); `t_macro: 0` means the LCM of all step sizes |
| `coupling` | `jacobi` (default) or `gauss-seidel`. `iterative` is **removed** and raises, as do `max_iterations` and `convergence_tolerance` |
| `extrapolation` | `zero-order` (default) or `linear` |
| `power_unit` | shorthand unit for power-like variable names, default: the unit system's active-power unit |
| `unit_system` | `default`, `si`, an overrides mapping `{active_power: MW}`, or `{name: si, overrides: {...}}`, see [Units](units.md) |
| `unit_policy` | `lenient` (default) or `strict` |
| `start_datetime` | wall-clock start; results are then indexed by datetime |
| `seed` | recorded in the results metadata |
| `out_dir` | write every run to `<out_dir>/<timestamp>_<hash>/` |
| `record_all` | `true` records every micro step; default `false` (one row per macro step, as `simulate()`) |
| `logging` | print INFO messages |

`res_filename` is **removed** (it raises): `simulate()` returns a `Results` object and nothing is written to disk
unless you set `out_dir` or pass `--out`.

| `simulators:` entry | Meaning |
|---|---|
| `name`, `type` | required; the type is any `sim_type` of [add_simulator](add-simulator.md) |
| `model` (alias `location`) | the model reference: path, project name, or (with `allow_import`) a dotted import |
| `step_size`, `inputs`, `outputs`, `mode`, `power_unit`, `units` | as in `add_simulator`; `mode` is `ac` or `dc` |
| `parameters` | for `type: model` only: constructor parameters of the component (merged with `options`; a name in both is an error) |
| `signal` | for `type: signal`: a number, a safe expression string or a structured spec |
| `options` | extra keyword arguments of the adapter: `delimiter`, `align_to_rows` (csv), `validate`, `solver_name` (fmu), `size`, `capacity_kwh`, ... (fleets), `function` (script), `engine` (matlab), `breakpoints` (signal) |

A `connections:` entry is `{from, to, scale, offset, from_unit, to_unit}`; `from` may be a list (fan-in, summed) and
`to` a list (fan-out). A plain mapping `{a.y: b.u}` is also accepted. `events:` and `triggers:` support the declarative
`set` action only.

Which simulator types can a scenario create on its own? `signal`, `csv`, `fmu`, `powerflow`, `pypsa`, `opendss`,
`powerfactory`, `matlab`, `script`, `external` (a file or `file.py:Class`) and the built-in fleets (`battery_fleet`,
`building_fleet`, `pv_fleet`). Built-in physical models (`ModelComponent` classes) are created with `type: model`, see below.

## Components in a scenario (`type: model`)

`type: model` instantiates a component from its name. `model:` is one of

* the name of a class exported by `energysim.models` or `energysim.networks` (`PEMElectrolyzer`, `BatteryBESS`,
  `HeatNetwork`, ...) - allowed in [safe mode](safe-mode.md);
* the name of a catalog archetype (`BESS_4h_2MW`, `PEM_5MW`, ...) - allowed in safe mode;
* `path/to/file.py:ClassName` (a class in a Python file, relative to the scenario file) - **not** allowed in safe mode.

Constructor parameters go under `parameters:` (or `options:`, same meaning). Unknown parameter names raise
`ValueError` listing the valid ones, because the component base class would otherwise ignore typos silently. The
simulator name is passed as the component's `name`. For network classes (`HeatNetwork`, `GasNetwork`,
`EnergyNetwork`), `topology:` under `parameters` holds the `producers`, `consumers`, `storages`, `junctions` and `pipes`
(the format of `EnergyNetwork.to_dict()`).

```python
from energysim import world

w = world.from_yaml("""
world: {start_time: 0, stop_time: 1800, t_macro: 60}
simulators:
  - {name: cmd, type: signal, signal: 1.0, step_size: 60, units: {y: MW}}
  - name: ely
    type: model
    model: PEMElectrolyzer
    step_size: 60
    parameters: {p_nominal_mw: 2.0}
  - {name: bess, type: model, model: BESS_4h_2MW, step_size: 60, options: {capacity_mwh: 6.0}}
connections:
  - {from: cmd.y, to: ely.P_cmd}
  - {from: cmd.y, to: bess.P_cmd}
""", safe=True)
print(w.get_model("ely").p_nominal_mw, w.get_model("bess").capacity_mwh)      # 2.0 6.0
res = w.simulate(pbar=False)
print(res["ely.m_dot_H2_kg_hr"].max() > 0)                                    # True
```

**Round trip.** `w.to_dict()` / `w.to_yaml()` write a `type: model` entry back only when the component is an instance
of a class exported by `energysim.models` / `energysim.networks` (a catalog archetype serialises as its class) *and*
its constructor parameters can be recovered: the parameters listed in `describe()['parameters']` are read back from
same-named attributes, only values that differ from the default are written, and the component is rebuilt from them
and compared. Anything else (a subclass of your own, a model with non-plain parameters such as curve objects, or
parameters that do not survive a rebuild) is **not serialised**: `to_dict()` logs a warning and writes the entry
without `model:`. Runtime state is never serialised, only constructor parameters.

## Python API

| Call | Meaning |
|---|---|
| `world.from_yaml(path_or_yaml_string, safe=False, allow_import=False, allow_native=False)` | build a world; `base_dir` for relative paths is the file's directory |
| `world.from_dict(data, base_dir=None, safe=False, allow_import=False, allow_native=False)` | same from a dictionary |
| `w.to_dict()`, `w.to_yaml(filepath=None)` | serialise: world settings, simulators, connections, events and triggers. Python callables (signals, event actions, trigger conditions) cannot be serialised and are skipped with a warning |
| `w.validate(raise_on_error=False)` | pre-flight report |

A `Results` object stores the serialised scenario (`res.meta['scenario']`) and a hash of it, and `--out` / `out_dir`
runs write it as `scenario.yaml` next to the data, so a run directory is self-describing.

## Command line

```bash
energysim --version
energysim validate scenario.yaml [--safe [--allow-native]] [-v]
energysim run scenario.yaml [--out DIR] [--to-csv] [--dashboard] [--dashboard-path FILE] [--offline]
                            [--no-pbar] [--record-all] [--safe [--allow-native]] [--validate-only] [-v]
energysim units [--system default|si] [--set QUANTITY=UNIT ...] [--json]
```

| Command / flag | Description |
|---|---|
| `validate` | Build the world and print the `ValidationReport`; exit `0` if valid, `1` on errors (including a file that cannot be loaded or is refused by `--safe`). |
| `run` | Validate, then simulate (one row per macro step unless `--record-all` or `world.record_all`), print the run summary (`min max mean last count unit`); exit `0` on success, `1` on any failure. |
| `--out DIR` | Write the run to `DIR/<YYYYmmdd-HHMMSS>_<hash8>/`: Parquet (CSV without pyarrow) per simulator, `meta.json`, `scenario.yaml`. |
| `--to-csv` | Also export CSV files into `<run dir>/csv` (implies `--out runs` if no directory is given). |
| `--dashboard`, `--dashboard-path FILE`, `--offline` | Write the HTML dashboard (opened in the browser with `--dashboard`); `--offline` embeds plotly.js (needs `plotly`). |
| `--no-pbar` | Disable the progress bar. |
| `--record-all` | Record every micro step instead of one row per macro step (`run` only). |
| `--validate-only` | Stop after pre-flight validation (`run` only). |
| `--safe` | [Safe mode](safe-mode.md): no code-executing types, FMUs, pickles, dotted imports; file paths and `out_dir` confined to the scenario directory tree. |
| `--allow-native` | With `--safe`: still allow FMUs (native code). |
| `-v`, `--verbose` | Also print the unit system (name and hash) and every connection's unit conversion. |
| `units` | Print the default unit per quantity of a unit system, optionally with overrides, or as JSON. |

`energysim units --set active_power=MW` prints the table with a modified default. The same table is available in Python:
`world.unit_table()`, and `world.connection_units()` shows the resolved units of every connection.

## Pre-flight validation

`energysim validate`, `run` (always) and `simulate(validate=True)` (the default) call `world.validate()`:

1. at least one simulator; `stop_time >= start_time`; a valid `coupling`;
2. every `step_size > 0`; a warning when it does not divide the macro step; for fixed steps, an error unless
   `stop_time - start_time` is a multiple of `t_macro`;
3. events and triggers refer to simulators that exist;
4. connection endpoints exist; variable names are checked against the adapters that can list them; writing into a
   read-only simulator (csv, signal) is reported;
5. algebraic feedback loops are reported with the advice to reduce `t_macro` or merge the components;
6. unit conversions are listed, and connections with unspecified or unknown units produce warnings (errors with
   `unit_policy: strict`).

```python
from energysim import world
from energysim.models import BatteryBESS

w = world(start_time=0, stop_time=600, t_macro=60)
w.add_signal("cmd", 1.0, step_size=60)
w.add_model("bess", BatteryBESS(), step_size=60)
w.add_connections({"cmd.y": "bess.P_cmd"})
w.schedule(30, {"set": {"ghost.x": 1.0}})                    # unknown simulator

report = w.validate()
print(report.is_valid)                      # False
print(report.errors)                        # ["Event 'event_1' sets 'ghost.x' of an unknown simulator."]
print(report.warnings[0][:50])              # Connection cmd.y -> bess.P_cmd: unit of cmd.y is unspecified; ...
```

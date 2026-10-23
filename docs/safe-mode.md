# Safe mode and the trust model

A scenario file describes a simulation. Whether loading it can run arbitrary code depends on what it refers
to. This page defines what `energysim` guarantees for scenario files from untrusted sources and what it does not.

## Trust model

| You load ... | What can execute |
|---|---|
| a scenario with `safe=True` / `--safe` | Only the simulator types and model references listed under [What safe mode allows](#what-safe-mode-allows), on paths inside the scenario directory. Strings are parsed, never `eval`-ed. The full list of defences and limits is in the [threat model](#threat-model). |
| a scenario without safe mode | Everything you reference: Python files (`script`, `external`, `type: model` with `file.py:Class`), MATLAB functions, PowerFactory projects, FMUs (native code), pickled networks (`.p`, `.pkl`) and, with `allow_import=True`, dotted imports such as `pandapower.networks.case14`; it can read and write any path your user can. Treat such a file like a Python script from the same source. |
| Python code that builds a `world` | Whatever the code does. Safe mode does not apply to the Python API. |

Even without safe mode, `energysim` never passes a scenario string to `eval`: signal strings and trigger
conditions use a whitelisted expression parser (below), and dotted imports need an explicit opt-in.

## Turning it on

```bash
energysim validate scenario.yaml --safe
energysim run scenario.yaml --safe --no-pbar
```

```python
from pathlib import Path

from energysim import world

Path("scenario.yaml").write_text("""
world: {start_time: 0, stop_time: 60, t_macro: 60}
simulators:
  - {name: s, type: signal, signal: '2 * t', step_size: 60}
""", encoding="utf-8")

w = world.from_yaml("scenario.yaml", safe=True)        # a file path or a YAML string
w2 = world.from_dict({"world": {"stop_time": 60, "t_macro": 60},
                      "simulators": [{"name": "s", "type": "signal", "signal": 1.0, "step_size": 60}]},
                     safe=True)
print(w.simulate(pbar=False)["s.y"].iloc[-1])
```

A refused scenario raises `ValueError("... safe mode")` from `from_yaml` / `from_dict`; the CLI prints the
message and exits with code 1 (for `validate`, `run` and `run --validate-only` alike).

```python
from energysim import world

hostile = """
world: {start_time: 0, stop_time: 10, t_macro: 10}
simulators:
  - {name: x, type: script, model: ctrl.py}
"""
try:
    world.from_yaml(hostile, safe=True)
except ValueError as exc:
    print(exc)          # Simulator 'x': type 'script' runs arbitrary code and is not allowed in safe mode.
```

## What safe mode refuses

* Simulator types that execute code or external programs: `external`, `script` (`pyscript`), `matlab`,
  `powerfactory`.
* **FMUs** (`type: fmu`): an FMU is a shared library that runs inside the process. Refused unless you pass
  `allow_native=True` (`from_yaml` / `from_dict`) or `--allow-native` (CLI), for FMUs you trust.
* **Pickle model files** (`.p`, `.pkl`, `.pickle`) in `model` / `location` or in any `options` value:
  unpickling executes code. Use a JSON export instead (pandapower: `pp.to_json`).
* **Dotted import references** such as `pandapower.networks.case14` in `model` / `location` / path aliases.
  `allow_import` is forced to `False` in safe mode.
* **Component files**: `type: model` with `model: file.py:Class` imports and runs the file. Class names exported by
  `energysim.models` / `energysim.networks` and catalog archetypes (`model: PEMElectrolyzer`, `BESS_4h_2MW`) are fine.
* **Paths outside the scenario directory tree.** Every `model` / `location` value and every path-like `options` value
  (`csv_path`, `net_loc`, `dss_path`, ...) of the file-based types (`csv`, `fmu`, `powerflow`, `pypsa`, `opendss`)
  and the world's `out_dir` must resolve - after following symlinks and `..` - to a location inside the directory of
  the scenario file (the current working directory when the scenario is given as a string or dictionary).
  Absolute paths outside it and `..` escapes raise `ValueError`; absolute paths inside it are accepted. In safe
  mode the accepted path is replaced by its resolved absolute form, so a later `chdir` cannot redirect it.

## What safe mode allows

* `signal` simulators with expression strings or structured specs, `csv` profiles, `type: model` components by
  class or archetype name, and the `powerflow`, `pypsa`, `opendss` and fleet types, all with files inside the
  scenario tree (and `fmu` with `allow_native=True`).
* `events:` and `triggers:` with the declarative `set` action. There is no `call` / `exec` action in scenario
  files; unknown keys are rejected.

### Safe expressions

Signal strings (`add_signal`, YAML `signal:`) and trigger conditions are parsed with `ast` and only this
subset is accepted; anything else raises `UnsafeExpressionError` (a `ValueError`):

* numeric literals, the variable `t` (seconds) and the constants `pi`, `e`;
* `+ - * / ** %` and unary `+ -` (exponents above 1000 are refused), comparisons (also chained), `and` / `or` /
  `not`, conditional expressions `a if c else b`;
* calls to `sin cos tan exp log sqrt abs min max floor ceil` with positional arguments;
* a top-level list or tuple (the expression parser accepts it, but a signal has the single output `y`: one
  returning more than one value is rejected when it is added, "add one signal per output");
* an optional `lambda t:` prefix for 2.x compatibility (stripped, then parsed with the same grammar);
* in conditions only: `sim.var` / `sim.elem.attr` names and quoted names (`"'fmu.der(x)' > 0"`).

Attribute access on expressions, subscripts, comprehensions, lambdas other than the prefix, strings, imports and
names other than the above are refused. Expressions longer than 2000 characters are refused.

```python
from energysim import world
from energysim._safeexpr import UnsafeExpressionError

w = world(start_time=0, stop_time=60, t_macro=60)
w.add_signal("ok", "max(0, 10 * sin(pi * t / 60))", step_size=60)            # accepted

for bad in ("__import__('os').system('echo hi')", "open('/etc/passwd')", "(1).__class__"):
    try:
        w.add_signal("bad", bad, step_size=60)
    except UnsafeExpressionError as exc:
        print("refused:", bad)
```

### Structured signal specs

The same signals without string syntax; `step`, `ramp` and `piecewise` expose their discontinuities as
[event hints](events.md):

```yaml
- {name: a, type: signal, signal: 5.0}                                         # constant
- {name: b, type: signal, signal: {type: sine, amplitude: 2, period: 3600, offset: 1}}
- {name: c, type: signal, signal: {type: step, time: 600, before: 0, after: 3}}
- {name: d, type: signal, signal: {type: ramp, start_time: 0, end_time: 600, start_value: 0, end_value: 10}}
- {name: e, type: signal, signal: {type: piecewise, breakpoints: [[0, 0], [60, 5], [120, 2]], interpolation: linear}}
- {name: f, type: signal, signal: {type: expression, expr: "3 * sin(t / 60)"}}
```

## Threat model

Safe mode defends against a scenario file written by someone else that you load and run on your machine. The
assumption is that **the scenario file is hostile and the Python environment, the installed packages and the files
inside the scenario directory are trusted**.

| Threat | Mitigation |
|---|---|
| Run arbitrary Python from the scenario (`script`, `external`, `file.py:Class`, dotted imports, `eval` in strings) | Refused types and references; expressions go through a whitelisted AST evaluator; events and triggers support only the declarative `set` action |
| Run native code (FMU shared libraries) | `fmu` refused unless `allow_native=True` |
| Run code through deserialisation (pickle network files) | `.p` / `.pkl` / `.pickle` refused |
| Read files outside the project (`../../etc/passwd`, `C:\Users\...`, symlinks pointing out) | Paths are resolved (symlinks, `..`) and must stay inside the scenario directory tree |
| Write files outside the project (`out_dir`) | `out_dir` must resolve inside the scenario directory tree |
| Hide any of the above in an option value | The checks cover `model`, `location`, every path alias and every string in `options` |

What safe mode does **not** do - it is a filter on scenario *definitions*, not a sandbox:

* **Files inside the scenario tree are trusted.** A scenario can reference a pickle, script or FMU that sits next to
  it only if it passes the checks above; the checks look at the *reference*, not at what the file does. Pickle and
  script references are refused by name and extension, but an OpenDSS (`.dss`) file may itself `Redirect` or `Compile`
  other files, and a model file may name further files that safe mode never sees. Do not run untrusted scenarios
  together with untrusted data files.
* **`--out DIR` is yours.** The command-line `--out` is chosen by the person running the command and is not
  restricted; only the scenario's own `out_dir` is.
* **TOCTOU.** The path check happens when the scenario is loaded. A process that changes the directory tree between
  the check and the use (for example by swapping a file for a symlink) is outside the model.
* **Resource exhaustion.** There are no limits on CPU time, memory or output size; a scenario can request a very long
  run or a very large fleet.
* **The Python API** (`add_external_simulator`, callables as events, `allow_import=True`, `allow_native=True`) is
  outside its scope.

Run untrusted scenarios in a container or a restricted account if any of that matters.

If your own scenarios need scripts, run them without `--safe` and keep them in version control.

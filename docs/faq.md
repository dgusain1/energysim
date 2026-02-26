# FAQ

---

## OPFNotConvergedError

OPF and DCOPF convergence depends on the pandapower network
configuration. Make sure OPF converges *within pandapower alone* before
integrating the network with `energysim`. Common fixes:

- Check that cost functions are defined for all controllable elements.
- Ensure `controllable=True` is set on dispatchable loads/generators.
- Verify voltage and power limits are reasonable.

---

## FMU initialisation error

If you get an initialisation error for an FMU, first check that it works
independently:

```python
from fmpy import simulate_fmu
result = simulate_fmu('/path/to/model.fmu')
print(result)
```

If FMPy validation fails but the FMU simulates correctly, use
`validate=False`:

```python
my_world.add_simulator(
    sim_type='fmu', ..., validate=False)
```

Other remedies:

1. Clear cache and temp folders.
2. Restart the application.
3. Run with administrator privileges.

---

## SimulatorVariableError / SimulatorElementNotFoundError

These exceptions replace the old `sys.exit()` calls. They are raised
when a variable name is invalid or a named element is not found in a
pandapower/PyPSA network.

Common causes:

- Typo in the variable name (e.g. `'Gen1.p_MW'` instead of `'Gen1.p_mw'`).
- Element name does not match the name in the pandapower DataFrame (check `net.gen['name']`, `net.load['name']`, etc.).
- Using an index instead of a name.

You can inspect available variables using introspection:

```python
entry = my_world.simulator_dict["grid"]
print(entry.adapter.get_available_variables())
```

---

## ConnectionError

Raised when a connection string is malformed or references a
non-existent simulator. Common causes:

- Missing `.` separator: `'sim1_var1'` instead of `'sim1.var1'`.
- Simulator not added before `add_connections()` is called.
- Typo in the simulator name.

All connections are validated immediately when `add_connections()` is
called, so errors appear early rather than during simulation.

---

## Simulation hangs / runs very slowly

Check the `step_size` parameter. If a simulator's `step_size` is
too small relative to the macro time-step (`t_macro`), the number of
micro-steps per macro step can be very large. For example, with
`t_macro=900` and `step_size=1`, the simulator takes 900 steps per
exchange.

For power-flow networks, a `step_size` of 300–900 is usually
appropriate.

---

## TypeError: 'SimEntry' object is not subscriptable

This error occurs when code uses the old tuple-based access pattern:

```python
# OLD (broken) — do not use
adapter = my_world.simulator_dict["grid"][1]
```

Since the architecture refactoring, `simulator_dict` values are
`SimEntry` dataclass objects. Use attribute access instead:

```python
# NEW (correct)
adapter = my_world.simulator_dict["grid"].adapter
network = my_world.simulator_dict["grid"].adapter.network
```

---

## Read-only simulator warnings

If you see a warning like:

```
UserWarning: Simulator 'csv1' is read-only (type='csv').
Connection to 'temp' will be silently ignored.
```

This means a connection writes to a CSV or signal simulator, which
cannot accept input values. Check your connection dictionary — the
target should be a writable simulator (FMU, powerflow, external, etc.).

If the connection is intentional (e.g. for documentation purposes),
you can suppress the warning:

```python
import warnings
warnings.filterwarnings('ignore', message='.*read-only.*')
```

---

## Algebraic loop warnings

If you see a warning like:

```
UserWarning: Algebraic loop detected involving simulators:
['battery', 'controller', 'heatpump']
```

This means there is a circular dependency between the listed simulators.
`energysim` handles this automatically:

- With `coupling='jacobi'`, the loop is resolved with a one-step delay (each simulator sees the previous step's outputs).
- With `coupling='gauss-seidel'`, Jacobi coupling is used within the loop.
- With `coupling='iterative'`, the loop is iteratively resolved until convergence.

If strict accuracy is required for the looped variables, use
`coupling='iterative'` with appropriate `max_iterations` and
`convergence_tolerance`.

---

## Iterative coupling does not converge

If iterative coupling fails to converge within `max_iterations`:

1. **Increase `max_iterations`** — some systems need more iterations.
2. **Reduce `t_macro`** — smaller exchange intervals make convergence easier.
3. **Check `convergence_tolerance`** — `1e-6` may be too tight for some variables (e.g. large power values). Try `1e-4`.
4. **Ensure adapters support state save/restore** — if `save_state()` returns `None`, the adapter cannot be rolled back and iterative coupling degrades to Gauss-Seidel.

For FMU CS adapters, state save/restore requires FMI 2.0 FMUs with
`canGetAndSetFMUstate=true` in the model description.

---

## Missing optional dependencies

`energysim` core requires numpy, pandas, matplotlib, networkx, tqdm,
and PyTables. Simulator-specific packages are optional:

| Package         | Purpose                     | pip                                                          | uv                                    |
|-----------------|-----------------------------|--------------------------------------------------------------|---------------------------------------|
| FMPy            | FMU simulators              | `pip install energysim[fmu]`                                 | `uv add energysim[fmu]`               |
| pandapower      | Powerflow simulators        | `pip install energysim[powerflow]`                           | `uv add energysim[powerflow]`         |
| PyPSA           | PyPSA network simulators    | `pip install energysim[pypsa]`                               | `uv add energysim[pypsa]`             |
| Plotly          | Interactive HTML dashboards | `pip install plotly`                                         | `uv add plotly`                       |
| PowerFactory    | DIgSILENT models            | Requires PF 2020+ with Python API — no pip package needed    | —                                     |
| MATLAB Engine   | MATLAB models               | Install from your MATLAB installation (see below)            | —                                     |
| oct2py          | Octave models               | `pip install oct2py`                                         | `uv add oct2py`                       |

**Installing MATLAB Engine for Python:**

```bash
cd "C:\Program Files\MATLAB\R2024b\extern\engines\python"
python setup.py install
```

If a required package is missing, `energysim` will raise an
`ImportError` with a clear message when you try to add that type of
simulator.

---

## PowerFactory: sim_name must match the project name

The `sim_name` passed to `add_simulator(sim_type='powerfactory', ...)`
**must** match the PowerFactory project name exactly. The adapter calls
`ActivateProject(sim_name)` during initialisation. If the names do
not match, PowerFactory will raise an error.

---

## Octave: nargin: invalid function name

This error means the `.m` file could not be found on the Octave search
path. Common causes:

1. The `sim_loc` path is incorrect. Use an absolute path or ensure the working directory contains the `.m` file.
2. The file is a **script**, not a **function**. The first non-comment line must be `function [...]` — `nargin` only works on functions.
3. The function name inside the file does not match the file name.

---

## Which coupling mode should I use?

| Mode              | Best for                                                                          |
|-------------------|-----------------------------------------------------------------------------------|
| `'jacobi'`        | Loosely coupled systems, simple setups, backward compatibility                    |
| `'gauss-seidel'`  | Feed-forward chains, systems with a clear upstream → downstream flow              |
| `'iterative'`     | Tightly coupled systems with algebraic loops, when accuracy of coupled variables is critical |

When in doubt, start with `'jacobi'` (the default) and switch to
`'gauss-seidel'` or `'iterative'` if you observe coupling artifacts
(e.g. oscillations, delayed responses).

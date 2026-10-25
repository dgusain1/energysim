# Ensembles, Monte-Carlo studies and sweeps

`energysim.Ensemble` runs many variants of one co-simulation (a parameter grid, random samples, noisy inputs, or
just different seeds) and collects the outcomes. `energysim.Sweep` is the one-at-a-time / full-factorial special case
built on top of it.

Key properties:

* Every run gets a **fresh world** from your factory, so there is no state leakage between runs and the order of
  parameters never changes a result.
* Seeds are derived with `numpy.random.SeedSequence(master).spawn(n)`: the same master seed gives the same run
  seeds, independent of serial or process execution.
* A failing run is recorded (status, traceback) and does not abort the ensemble unless you ask for
  `on_error='raise'`.

## A first ensemble

The factory has the signature `factory(params: dict, seed: int) -> world` and returns a world that is ready to
simulate. `params` holds the sampled values, `seed` the run seed (use it for stochastic inputs).

```python
from energysim import Ensemble, noisy_signal, world
from energysim.ensemble import Uniform
from energysim.models import BatteryBESS


def build(params, seed):
    w = world(start_time=0, stop_time=6 * 3600, t_macro=900, seed=seed)
    # a set-point of 0.4 MW with seeded, autocorrelated additive noise (sigma 0.5 MW)
    cmd = noisy_signal(lambda t: 0.4, sigma=0.5, seed=seed, rel=False, ar1=0.9, dt=900.0)
    w.add_signal("cmd", cmd, step_size=900, units={"y": "MW"})
    w.add_model("bess", BatteryBESS(capacity_mwh=params.get("capacity_mwh", 4.0), p_max_mw=1.0, soc_init=0.5),
                step_size=60)
    w.add_connections({"cmd.y": "bess.P_cmd"})
    return w


ens = Ensemble(
    build,
    sampler=Uniform({"capacity_mwh": (2.0, 8.0)}, n=8, seed=1),     # 8 random capacities
    seeds=42,                                                       # master seed for the run seeds
    kpis={"soc_end": lambda r: float(r["bess.SoC"].iloc[-1])},      # your own scalar per run
    variables=["bess.SoC"], kpi_names=["min", "max"],               # plus registered KPIs per variable
)
res = ens.run()                                       # an EnsembleResults

print(res)                                            # EnsembleResults(runs=8, completed=8, failed=0, keep='all')
print(res.params.head(3))                             # one row per run: sampled parameters and the 'seed' column
print(res.status["status"].value_counts().to_dict())  # status, error, duration_s per run
print(res.kpis().round(3).head(3))                    # one row per run
print(res.summary().round(3))                         # count, mean, std, min, quantiles, max per KPI
```

### Looking at the runs

```python
fan = res.fan("bess.SoC", quantiles=(0.1, 0.5, 0.9))   # quantile bands over time (needs keep='all')
wide = res.to_frame("bess.SoC")                        # one column per run
one = res.run(0)                                       # the Results of run 0
every = res.ensemble                                   # all runs in one Results (is_ensemble == True)
again = ens.rerun(3)                                   # re-execute run 3 (same params and seed) in this process
print(fan.iloc[[0, -1]].round(3))
print(wide.shape, one["bess.SoC"].iloc[-1], every.is_ensemble, again["bess.SoC"].iloc[-1])
```

Run keys are the integers `0..n-1`. With `keep='kpis'` only the KPI table is retained (use it for thousands of runs);
`runs`, `run()`, `fan()` and `to_frame()` then raise `EnsembleError`.

## Samplers

| Sampler | Use |
|---|---|
| `Grid({'a': [1, 2], 'b': [10, 20, 30]})` | full factorial |
| `Scenarios([{...}, {...}])` | explicit list (`Ensemble(..., scenarios=[...])` is a shortcut) |
| `Uniform({'x': (lo, hi)}, n=, seed=)` | independent uniform draws |
| `Normal({'x': (mean, std)}, n=, seed=)` | independent normal draws |
| `LatinHypercube({'x': (lo, hi)}, n=, seed=, method=)` | stratified samples (scipy's QMC, or a NumPy fallback) |
| `Sobol({'x': (lo, hi)}, n=, seed=)` | scrambled Sobol' points (needs scipy; prefer `n` a power of two) |

Random samplers are deterministic given `seed`. With no sampler, `Ensemble(build, n_runs=100)` gives 100 parameter-free
runs that differ only by their seed, which is what a pure noise study needs. The name `seed` is reserved for the run
seed and cannot be used as a parameter.

## Applying parameters

`apply_params(world, {'sim.var': value})` merges values into `world.options({'init': ...})`; they are applied at the
start of every run. A factory can use it, or you can let `Ensemble.from_world` do it for a parameter-free world factory:

```python
from energysim import Ensemble, world
from energysim.ensemble import Grid
from energysim.models import BatteryBESS


def plain_world():
    w = world(start_time=0, stop_time=3600, t_macro=300)
    w.add_signal("cmd", 0.5, step_size=300, units={"y": "MW"})
    w.add_model("bess", BatteryBESS(capacity_mwh=4.0, p_max_mw=1.0, soc_init=0.5), step_size=60)
    w.add_connections({"cmd.y": "bess.P_cmd"})
    return w


ens = Ensemble.from_world(
    plain_world,
    parameter_map={"capacity": "bess.capacity_mwh"},         # optional: sampled name -> 'sim.var'
    sampler=Grid({"capacity": [2.0, 4.0, 8.0]}),
    kpis={"soc_end": lambda r: float(r["bess.SoC"].iloc[-1])},
)
print(ens.run().kpis()["soc_end"].round(3).tolist())            # a smaller pack charges further
```

Only registered parameters and declared inputs of a `ModelComponent` can be set; anything else raises
`SimulatorVariableError` at the start of the run, which the ensemble records as a failed run.

## Uncertain inputs: `NoisyInput`

`noisy_signal(base, sigma, seed, rel=True, ar1=0.9, dt=1.0)` (class `NoisyInput`) wraps a callable `f(t)`, a number, or
a `pandas.Series` indexed by time. The perturbation is piecewise constant on a grid of `dt` seconds, optionally AR(1)
correlated (`ar1`), multiplicative (`rel=True`, `base * (1 + n)`) or additive. It is a **pure function of `t`** for
a given seed, so it does not depend on how often or in which order it is evaluated, and the result is identical
across serial and process runs. Use the run `seed` of the factory to give every run its own realisation.

## Parallel runs

`Ensemble(..., parallel='process', max_workers=4)` uses a `spawn` process pool. The factory (and every KPI callable)
must then be importable from a module: lambdas and closures (`<lambda>` / `<locals>`) and unpicklable objects are rejected with a hint, and functions defined in `__main__` of an interactive session will fail in the workers. Worker results are bit-identical to the serial ones.

```python notest
# runs in a script, not in the docs test (needs an importable module and a __main__ guard)
from energysim import Ensemble
from my_study import build, final_soc          # module-level functions

if __name__ == "__main__":
    res = Ensemble(build, n_runs=200, seeds=7, parallel="process", max_workers=4,
                   keep="kpis", kpis={"soc_end": final_soc}).run()
```

## Persisting

`Ensemble(..., out_dir="study")` (or `res.save("study")`) writes the parameter, status and KPI tables and, with
`keep='all'`, every run; `EnsembleResults.load("study")` restores them.

## Sweeps

`Sweep(world_or_factory, sensitivity_info, kind='single' | 'parametric')` evaluates one output (a metric of one
`'sim.var'`) over a grid of parameters.

* `kind='single'` varies each parameter on its own around the baseline (the others untouched).
* `kind='parametric'` runs the full factorial grid (at least two parameters).
* `bounds` entries are `[lo, hi]` (10 points) or `[lo, hi, n]`; `metric` is one of `last`, `mean`, `max`, `min`,
  `sum`, `integral` (zero-order-hold integral over seconds).
* A world instance is deep-copied for every run and never modified; pass a factory `f(params, seed)` for worlds that
  cannot be copied.

```python
from energysim import Sweep, world
from energysim.models import BatteryBESS

w = world(start_time=0, stop_time=3600, t_macro=300)
w.add_signal("cmd", 0.5, step_size=300, units={"y": "MW"})
w.add_model("bess", BatteryBESS(capacity_mwh=4.0, p_max_mw=1.0, soc_init=0.5), step_size=60)
w.add_connections({"cmd.y": "bess.P_cmd"})

info = {
    "parameters": ["bess.capacity_mwh", "bess.soc_max"],
    "bounds": [[2.0, 8.0, 4], [0.7, 0.95, 3]],
    "output_parameter": "bess.SoC",
    "metric": "last",
}
single = Sweep(w, info, kind="single").sweep()            # SweepResult
print(single.table.round(3))                              # columns: parameter, value, result
print(single.baseline)                                    # dict {'values': {...}, 'result': float}; None if a factory was passed

grid = Sweep(w, info, kind="parametric").sweep()
print(grid.table.shape)                                   # (12, 3): both parameters plus 'result'
grid.export_to_csv("sweep.csv")
fig = grid.plot()                                         # heatmap for two parameters (needs matplotlib)
```

`SweepResult.ensemble` is the underlying `EnsembleResults`; `Sweep(..., keep='all', parallel='process')` forwards extra
keywords to `Ensemble`. `SweepResult.plot(kind='curves')` plots the output time series of single sweeps and needs
`keep='all'`.

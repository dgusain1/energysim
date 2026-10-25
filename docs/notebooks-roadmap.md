# Notebook roadmap

Status: **planning document.** Notebooks 00-20 ship today (wave 1, 08-16, is built from `scripts/notebook_src/`; 17-20 are the Octave/MATLAB, PowerFactory, FMU and ensemble guides). Wave 2 is a list of ideas, not commitments.

## 1. Candid assessment of the current notebooks

The seven notebooks in [Notebooks and examples](notebooks-gallery.md) are useful as a first contact with the API, but
they are thin as teaching or reference material:

- **Short, single-path demos.** Each notebook has 3 to 7 code cells and roughly 1 000 to 3 000 characters of code in
  total (notebook 06, the fleet notebook, is about 1 000). Each shows one scenario, one set of parameters and one path
  through the API.
- **Little analysis.** Most cells run a simulation and plot a few curves. There are no sensitivity sweeps, no energy or
  cost balances, no comparison against an analytic result, and no discussion of what a plot means for a decision.
- **No cross-domain mixes.** Electricity, heat, gas and hydrogen appear mostly in isolation. Notebook 07 is the closest
  to sector coupling, and it is a single-island system with no grid, no network and no fleet.
- **Mismatched titles.** Notebook 02 is named after OpenDSS, but its code only uses pandapower and PyPSA. OpenDSS
  appears in the markdown only.
- **Uneven depth.** Notebook 04 (custom adapters) and notebook 03 (custom models) are good reference material for
  their topics, but they do not show the results of their own models being checked against anything.
- **No validation cell.** No notebook checks a simulated quantity against an independent calculation. The checked
  benchmarks live in [Validation](validation.md) and in `tests/validation`; notebooks should point at them and reuse the
  same style of check.
- **Missing standard sections.** No notebook states its runtime budget, guards optional dependencies consistently, or
  lists the equations it relies on.

The goal of the expansion is to turn each notebook into a short, honest piece of engineering analysis: a question, the
model, a cross-check, the analysis, and the limits of the result.

## 2. Authoring system

Notebooks are no longer edited as JSON. The source of truth is a percent-format `.py` file:

- **Sources:** `scripts/notebook_src/NN_slug.py`. `# %% [markdown]` cells hold narrative with `# ` prefixes, and
  `# %%` starts a code cell. The diff is readable and the file can be syntax-checked without a kernel.
- **Shared code:** `#@include snippets/x.py` inlines a file from `scripts/notebook_src/snippets/`. Shared helpers (plot
  styling, unit printing, a validation-table helper, optional-dependency guards) are written once and included, so each
  notebook stays self-contained when it is opened.
- **Build:** `python scripts/build_notebooks.py` writes unexecuted `examples/notebooks/NN_slug.ipynb` files.
  `python scripts/build_notebooks.py 12 14` builds a subset. `python scripts/build_notebooks.py --check` parses every code
  cell and writes nothing.
- **Rules:** [`scripts/notebook_src/AUTHORING_NOTES.md`](https://github.com/dgusain1/energysim/blob/master/scripts/notebook_src/AUTHORING_NOTES.md)
  holds the rules for every notebook and the verified API facts (signatures, variable names, units). It is the first
  thing an author reads, and it must be updated when the API changes.
- **Replacing the generator:** `scripts/generate_notebooks.py` is the legacy path that produced notebooks 01-07. Once
  notebooks 01-07 are migrated (section 6), that script is removed so there is one authoring path.

Notebooks are committed without outputs. Only the maintainer executes them, as part of the slow test run.

## 3. Quality bar per notebook

Every notebook in waves 1 and 2 must have all of the following. Reviewers check this list before merge.

1. **Narrative.** A question stated at the top, what the reader should learn, and a closing "Where to go next" cell that
   links the neighbouring notebooks.
2. **Equations.** The governing equations of each physical model written in markdown (LaTeX), with the parameter names
   used in the code. Models with lookup curves state the curve source and its clamping range.
3. **Validation or cross-check cell.** At least one quantity checked against an independent value: a closed-form
   result, an energy balance that must close, a second solver, or the matching benchmark in `tests/validation`. The cell
   prints the residual and asserts a tolerance.
4. **Analysis plots.** At least two plots that answer the question, with axes labelled with units. Matplotlib only. No
   plot is included only for decoration.
5. **Runtime budget.** Under about 60 s on a laptop for the whole notebook, and under 180 s for any single cell (the
   test timeout). The budget is stated in the first markdown cell and re-checked by the author with a scratch run
   before merge (not by executing the notebook in the repository).
6. **Optional-dependency guards.** Every optional package is imported inside `try` / `except ImportError` with a flag
   (`HAS_PYPSA`, `HAS_OPENDSS`, and so on). Dependent cells sit under `if HAS_X:` and print a skip message otherwise. The
   notebook must run to the end with only the core install plus `matplotlib`.
7. **Determinism.** All randomness comes from `np.random.default_rng(seed)` with a seed stated in the notebook. No
   wall-clock time, no unsorted set iteration and no dependence on the platform's temporary directory beyond the
   documented OpenDSS script write.
8. **Honest parameters.** Parameters are illustrative and labelled as such. No manufacturer brand numbers or datasheet
   claims unless the source is cited.

## 4. Continuous integration

Today `tests/docs/test_notebooks.py` executes every notebook under the `slow` marker and skips a notebook only when
pandapower or matplotlib is missing, so a notebook that needs PyPSA or OpenDSS fails rather than skipping. The proposal:

- **Per-extra markers.** Add a notebook-level marker that matches the extras already declared in `pyproject.toml`
  (`requires_pypsa`, `requires_pandapower`, `requires_opendss`, `requires_fmu`, `requires_octave`, and the existing
  `requires_matlab` and `requires_powerfactory`). The test collects the extras each notebook needs, from a small
  front-matter comment in its source, and skips a notebook when a needed extra is missing, reporting which one.
- **A `fast` subset.** Mark notebooks whose total runtime is under 20 s with `fast`. Pull requests run the `fast` subset
  in the default test job; the full set runs nightly, and on release branches. Notebooks needing FMU or
  MATLAB/Octave run only in the nightly job, where the matching runner is available.
- **Build check.** Every pull request that touches `scripts/notebook_src/` runs `build_notebooks.py --check` and
  verifies that the committed `.ipynb` files match a fresh build (no hand edits to generated notebooks).
- **Output hygiene.** The test already runs notebooks in a temporary copy; keep that and add a check that committed
  notebooks contain no outputs.

## 5. Learning path

Numbers stay stable so that links do not break. The path reads top to bottom:

| Stage | Notebooks | Focus |
|---|---|---|
| Foundations | 01, 03 | The engine, signals, units, custom models |
| Single assets, fleets | 06, 08, 09, 10, 11 | Physics and aggregation of BESS, EV, heat-pump and PV fleets |
| Grids | 02, 12, 13, 15 | pandapower, PyPSA, OpenDSS, hosting capacity, solver comparisons |
| Multi-energy and adequacy | 05, 07, 13, 14 | District heat, gas, hydrogen, cold-snap adequacy |
| Integration | 04, 16 | Custom simulator adapters, FMU co-simulation |

Notebooks 17-20 already exist; new notebooks take numbers 21 onward and are added to this table when their sources exist.

## 6. Migration of notebooks 01-07

Notebooks 01-07 are migrated to the percent-format sources and then deepened to the quality bar above. Specifically:

- **01 quickstart:** keep it short; add an energy balance cell and a units demonstration.
- **02 grid:** either add real OpenDSS code to match the title or retitle it. Add a voltage-profile validation.
- **03 custom models:** add a numerical check of the state machine and a sensitivity sweep over the supercapacitor.
- **04 adapters:** add a unit test-style check of the adapter against a direct hydraulic solve.
- **05 industrial BTM:** add cost and CO2 accounting cells and a sensitivity to electrolyzer size.
- **06 fleet:** add a comparison of vectorized and scalar runs with a timing budget, and a convergence check in unit count.
- **07 hydrogen valley:** add the network from wave 1 once it exists, and a cross-check of the hydrogen mass balance.

Migration is done one notebook per pull request, so each diff is reviewable.

## 7. Wave plan

### Wave 1 (specified now, to be written)

| # | Working title | Level | What it adds | Extra dependencies |
|---|---|---|---|---|
| 08 | BESS fleet physics | advanced | A battery fleet with the physics of notebook 06 extended: OCV, internal resistance, thermal and SOH states, with a sweep over C-rate and temperature. Validated against the single-unit `BatteryBESS` model. | none |
| 09 | EV fleet mobility | advanced | Stochastic EV fleet with arrival and departure schedules, state of charge per vehicle, smart versus uncontrolled charging, seeded with `default_rng`. | none |
| 10 | Heat-pump and building fleet | advanced | Many building thermal models with heat pumps, a weather year, and comfort-band violation counts, checked against a lumped-capacity analytic step response. | none |
| 11 | PV smart-inverter fleet | advanced | PV fleet with volt-var and curtailment controls, and the active-power and reactive-power envelope against the inverter rating. | none |
| 12 | DSO feeder with fleets | expert | A pandapower feeder hosting the fleets from 08-11, with voltage and loading checks, and a hosting-capacity search using `Ensemble`. | `powerflow`, `networks` optional for the Ensemble sweep |
| 13 | District heat, Octave house and OpenDSS feeder | expert | A district heat network, a building simulated in Octave, and an OpenDSS feeder in one world, with energy balance closure across the three. | `networks`, `opendss`, Octave (guarded) |
| 14 | Power-gas cold-snap adequacy | expert | A winter cold-snap scenario with gas-fired generation, a gas network with linepack, and a loss-of-load probability computed over sampled weather. | `networks`, `ensemble` results |
| 15 | Curves and 2-D maps with grid solvers | expert | `Surface2D` operating maps (for example voltage against load and PV) produced by OpenDSS, PyPSA and Octave solvers on the same grid, with a solver-agreement table. | `opendss`, `pypsa`, Octave (guarded) |
| 16 | FMU with built-in physics | expert | The repository FMUs coupled with built-in models, with an explicit unit-declaration step, and a check that the FMU and built-in heat pump agree within a stated tolerance. | `fmu` (Windows-64 binaries only, guarded) |

Notebooks 08-11 use only the core install and `matplotlib`, so they can enter the `fast` subset once their runtimes are
measured. Notebooks 12-16 depend on the optional extras and carry the markers from section 4.

### Wave 2 (ideas, not committed)

- **MPC and OPF controllers with fleets:** a receding-horizon controller dispatching the fleets from wave 1 against a
  forecast, compared with a rule-based baseline.
- **Hydrogen valley on a gas network with pandapower:** electrolyzers, storage and a gas network coupled to an
  electrical feeder, with the hydrogen blending limit checked.
- **Microgrid islanding:** a planned island transition with events and a variable macro step around the switch, using
  the event API from [Events](events.md).
- **Ensemble and Monte-Carlo risk of EV plus PV:** a sampled distribution of peak feeder loading and its tail risk.
- **PowerFactory and MATLAB mixes:** notebooks that run only where the licensed engines are present, clearly marked
  as manual-only (`requires_powerfactory`).
- **YAML and CLI scenario authoring:** build a scenario in YAML, run it with `energysim run`, and read the same results
  in the notebook. See [YAML and CLI](yaml-cli.md).
- **Data-driven forecasts:** neural-network forecasters through the `ml` adapters, with forecast error reported against
  the persistence baseline.
- **Carbon accounting:** marginal and average emission factors applied to the multi-energy results, with a per-unit
  breakdown.

## 8. Open questions

- Whether the fleet notebooks should share one weather year so that cross-notebook numbers agree.
- Where the runtime measurements live, so the budget in section 3 is checked against recorded numbers rather than
  estimates.
- Whether the Octave-dependent cells in 13 and 15 are worth keeping in the public repository or should move to a
  maintainer-only notebook.

# energysim: guide for contributors and AI assistants

`energysim` is a Python package for co-simulation of multi-energy systems. A `world`
object couples independent simulators (FMU, pandapower, PyPSA, CSV, signals, user
Python code, MATLAB/Octave, ...) and steps them together.

## Commands

```bash
python -m pip install -e ".[test]"   # package + pytest, pytest-cov, hypothesis
python -m pytest tests               # full test suite
python -m pytest tests -m "not slow" # skip slow tests
ruff check src tests                 # lint (config in pyproject.toml)
mkdocs build --strict                # docs (requirements in docs/requirements.txt)
```

Use `pip install -e ".[all,test]"` when working on optional-dependency code paths.

## Layout

- `src/energysim/world.py`: `world` coordinator. Registers simulators, builds the
  dependency graph (networkx), runs the co-simulation, and exposes `results()`.
- `src/energysim/base.py`: `SimulatorAdapter` abstract contract and error types.
- `src/energysim/*Adapter.py`, `csv_adapter.py`, `signalAdapter.py`, `pyScriptAdapter.py`:
  simulator adapters. They are flat modules, not a sub-package.
- `src/energysim/container.py`: fleets of many identical components.
- `src/energysim/models/`: physical component models. `catalog.py` holds named presets,
  `common/` holds `ModelComponent`, curves and state machines.
- `src/energysim/_safeexpr.py`: restricted expression evaluator for signals in scenarios.
- `src/energysim/cli.py`: `energysim run|validate` and `--version`.
- `src/energysim/dashboard.py`: dashboard helpers for viewing results.
- `tests/`: pytest suite. Run from the repository root.
- `docs/`: MkDocs Material site. `examples/`: scenario YAML files and notebooks.
- Planned, not yet in the tree: `src/energysim/results.py` (typed results API) and
  `src/energysim/control/`, `src/energysim/networks/`.

## Conventions

- **Simulators implement `SimulatorAdapter`** (`base.py`): `configure`, `init`, `step` /
  `advance`, `get_value`, `set_value`, `get_available_variables`, `cleanup`. Adapters
  register with `world` through the `add_*` methods.
- **Physical models subclass `ModelComponent`** (`models/common/base_component.py`), which
  is itself a `SimulatorAdapter`. Add new models to `models/` and expose presets through
  `catalog.py`.
- **No `eval` or `exec` on scenario content.** Signals use `_safeexpr`. Scenario loading
  must stay compatible with `--safe` (see `SECURITY.md`).
- **No rollback.** Iterative coupling and state rollback were removed in 3.0. Supported
  coupling modes are `jacobi` (default) and `gauss-seidel`. Do not reintroduce
  save/restore of simulator state.
- **Results API:** `world.results()` returns the results as pandas DataFrames. Extend
  that path rather than adding new ad-hoc result formats.
- Keep changes small, add tests under `tests/`, and update `docs/` when behaviour changes.
- Python 3.10+ syntax; `from __future__ import annotations` in new modules.

## Before you finish

- `python -m pytest tests -m "not slow"` passes.
- `ruff check src tests` passes.
- Docs build with `mkdocs build --strict` if you touched `docs/`.

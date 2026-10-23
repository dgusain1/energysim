# Security Policy

## Trust model

energysim is built for trusted scenarios. A scenario is a YAML file plus the code and data it
references, and the package treats it the way a Python script is treated:

- **YAML scenarios can load and run code.** Simulator types such as `script`/`pyscript`,
  `external`, `matlab`, and `powerfactory`, and dotted-import model references, execute code
  from disk. Run them only when you trust their author.
- **`--safe` mode** (`energysim run --safe` and `energysim validate --safe`, or
  `safe=True` in the Python API) refuses those code-executing simulator types, FMUs (native
  code; re-enable with `--allow-native` / `allow_native=True`), `file.py:Class` component
  references, dotted-import model references, and pickle model files, and confines every
  model / csv / script / option file path and the scenario's `out_dir` to the scenario
  directory tree (symlinks and `..` are resolved before the check; absolute paths outside it
  are refused). Use it for scenarios from untrusted sources. Safe mode limits what a scenario
  can ask the package to do; it is not a sandbox against a malicious Python environment.
  Assumptions and limits (files inside the scenario tree are trusted, no resource limits,
  the Python API is out of scope, time-of-check/time-of-use) are listed in
  [docs/safe-mode.md](docs/safe-mode.md#threat-model).
- **Signal expressions are not `eval`-ed.** They are parsed and checked by a restricted
  evaluator (`src/energysim/_safeexpr.py`).
- **FMUs are native code.** A Functional Mock-up Unit is a shared library that runs inside the
  process when loaded with `fmpy`. Safe mode refuses FMUs unless you opt in with
  `allow_native=True`; only opt in for FMUs from sources you trust.
- **Pickle is unsafe.** Unpickling a file can run arbitrary code. Some network inputs
  (for example pandapower network files) may be stored as pickle. Do not load pickle files
  from untrusted sources, and prefer the JSON or other non-pickle formats where available.

## Supported versions

Security fixes are applied to the latest released 3.x line. Older releases do not receive fixes.

## Reporting a vulnerability

Please do not open a public GitHub issue for a security problem.

Send a private report by email to **digvijay.gusain29@gmail.com**. Include:

- the affected version (`energysim --version`) and Python version,
- a minimal scenario or code snippet that reproduces the problem,
- the impact you expect, and
- whether you intend to publish details, and when.

You will get an acknowledgement within a few working days. Please allow reasonable time for a
fix before any public disclosure. Credit is given in the release notes unless you prefer otherwise.

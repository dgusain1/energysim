#!/usr/bin/env python
"""Regenerate the machine-owned regions of README.md.

Three regions, each delimited by HTML comment markers, are rewritten:

* ``<!-- features:start -->`` ... ``<!-- features:end -->``  feature list built from ``features.yaml``
* ``<!-- extras:start -->``   ... ``<!-- extras:end -->``    install extras built from ``pyproject.toml``
* ``<!-- params:start -->``   ... ``<!-- params:end -->``    ``world()`` / ``add_simulator()`` parameter
  tables built from the real signatures (defaults come from the code; the prose descriptions live in
  :data:`WORLD_DOC` / :data:`ADD_SIM_DOC` below and a missing one is an error).

Usage::

    python scripts/generate_readme_features.py          # rewrite README.md
    python scripts/generate_readme_features.py --check  # exit 1 if README.md is out of date
"""
from __future__ import annotations

import inspect
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

README = os.path.join(ROOT, "README.md")
DOCS_BASE = "https://energysim.readthedocs.io/en/latest/"

WORLD_DOC = {
    "start_time": "Simulation start time in seconds.",
    "stop_time": "Simulation end time in seconds; `stop_time - start_time` must be a multiple of `t_macro` (unless `t_macro_min < t_macro`).",
    "logging": "Print INFO progress messages of the `energysim` logger.",
    "t_macro": "Macro time-step (variable exchange interval); `0` means the LCM of all simulator step sizes.",
    "coupling": "Coupling strategy: `'jacobi'` or `'gauss-seidel'`.",
    "extrapolation": "Input extrapolation between macro steps: `'zero-order'` or `'linear'`.",
    "power_unit": "World default power unit used to resolve unit-less power variables.",
    "start_datetime": "Optional wall-clock start; results are then indexed by datetime.",
    "seed": "Optional random seed, recorded in the results metadata.",
    "out_dir": "If set, every run is written to `<out_dir>/<timestamp>_<hash>/`.",
    "unit_system": "`None`, `'default'`, `'si'`, an overrides dict or a `UnitSystem`.",
    "unit_policy": "`'lenient'` (default; unspecified units pass through with a warning) or `'strict'` (unspecified units on connections raise).",
    "t_macro_min": "Smallest macro step when adapters report step hints or events occur (default: `t_macro`, i.e. fixed step).",
}

ADD_SIM_DOC = {
    "sim_type": "`'fmu'`, `'powerflow'`, `'pypsa'`, `'opendss'`, `'csv'`, `'signal'`, `'external'`, "
                "`'powerfactory'`, `'matlab'`, `'script'`, `'model'` or a fleet kind (`'battery_fleet'`, ...).",
    "sim_name": "Unique identifier for this simulator.",
    "sim_loc": "Path / object of the model (alias of `model`; many other aliases such as `net`, `path` exist).",
    "inputs": "List of input variable names.",
    "outputs": "List of output variable names to record.",
    "step_size": "Micro time-step.",
    "mode": "Power-flow mode: `'ac'` or `'dc'`.",
    "power_unit": "Power unit of this simulator's variables (default: the world's `power_unit`).",
    "units": "Optional `{variable: unit}` mapping that overrides inferred units.",
    "model": "Model object, file path or factory.",
    "allow_import": "Allow dotted-import model references (`'pandapower.networks.case14'`); refused in safe mode.",
}

TYPE_KWARGS = """\
Type-specific keyword arguments:

- **FMU**: `validate` (bool), `variable` (bool), `solver_name`
- **Power flow** (`pandapower`, `pypsa`, `opendss`): `mode` - `'ac'` or `'dc'`
- **CSV**: `delimiter` (str)
- **PowerFactory**: `mode` - `'ac'` or `'dc'` (load flow); `pf_path` (str)
- **MATLAB / Octave**: `engine` - `'auto'`, `'matlab'` or `'octave'`"""


# ----------------------------------------------------------------------------------------------
def _load_yaml(path):
    import yaml

    with open(path, encoding="utf8") as fh:
        return yaml.safe_load(fh)


def _docs_url(path: str) -> str:
    p = path.replace("\\", "/")
    if p.startswith("docs/"):
        p = p[len("docs/"):]
    if p.endswith("index.md"):
        p = p[: -len("index.md")]
    elif p.endswith(".md"):
        p = p[:-3] + "/"
    return DOCS_BASE + p


def render_features() -> str:
    data = _load_yaml(os.path.join(ROOT, "features.yaml"))
    groups = data["groups"]
    by_group: dict = {g: [] for g in groups}
    for f in data["features"]:
        by_group[f["group"]].append(f)
    out = ["<!-- Generated from features.yaml by scripts/generate_readme_features.py. Do not edit by hand. -->", ""]
    for g, title in groups.items():
        items = by_group[g]
        if not items:
            continue
        out.append(f"**{title}**")
        out.append("")
        out.append("| Capability | Status | Description |")
        out.append("|---|---|---|")
        for f in items:
            name = f"[{f['title']}]({_docs_url(f['docs'])})"
            out.append(f"| {name} | {f['status']} | {f['description']} |")
        out.append("")
    out.append("Status legend: `stable` and `beta` features are covered by the test suite listed in "
               "`features.yaml`; `planned` features are not implemented yet; `removed` ones raise a clear error.")
    return "\n".join(out)


def render_extras() -> str:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        import tomli as tomllib
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
        proj = tomllib.load(fh)["project"]
    extras = proj["optional-dependencies"]
    skip = {"docs", "test", "dev"}
    deps = proj["dependencies"]
    lines = []
    for name, reqs in extras.items():
        if name in skip:
            continue
        if name == "all":
            lines.append(("all", "everything above"))
        else:
            lines.append((name, "adds " + ", ".join(reqs)))
    width = max(len(n) for n, _ in lines) + len("energysim[]")
    pip = ["```bash", "pip install energysim            # core: " + ", ".join(deps)]
    uv = ["```bash", "uv add energysim"]
    for n, d in lines:
        pip.append(f"{('pip install energysim[' + n + ']').ljust(width + 12)} # {d}")
        uv.append(f"{('uv add energysim[' + n + ']').ljust(width + 7)} # {d}")
    pip.append("```")
    uv.append("```")
    return "\n".join(
        ["<!-- Generated from pyproject.toml by scripts/generate_readme_features.py. Do not edit by hand. -->", "",
         "With pip:", ""] + pip + ["", "With uv:", ""] + uv)


def _param_table(func, docs: dict, skip=("self", "kwargs")) -> str:
    rows = ["| Parameter | Default | Description |", "|---|---|---|"]
    for name, p in inspect.signature(func).parameters.items():
        if name in skip or p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
            continue
        if name not in docs:
            raise KeyError(f"README generator: no description for parameter '{name}' of {func.__qualname__}; "
                           f"add it to scripts/generate_readme_features.py")
        default = "-" if p.default is inspect.Parameter.empty else f"`{p.default!r}`"
        rows.append(f"| `{name}` | {default} | {docs[name]} |")
    unknown = set(docs) - set(inspect.signature(func).parameters)
    if unknown:
        raise KeyError(f"README generator: stale descriptions for {func.__qualname__}: {sorted(unknown)}")
    return "\n".join(rows)


def render_params() -> str:
    from energysim import world

    return "\n".join([
        "<!-- Generated from the signatures by scripts/generate_readme_features.py. Do not edit by hand. -->",
        "",
        "### `world()` parameters",
        "",
        _param_table(world.__init__, WORLD_DOC),
        "",
        "### `add_simulator()` parameters",
        "",
        _param_table(world.add_simulator, ADD_SIM_DOC),
        "",
        TYPE_KWARGS,
    ])


REGIONS = {"features": render_features, "extras": render_extras, "params": render_params}


def sync(text: str) -> str:
    """Return *text* with every generated region regenerated."""
    for name, fn in REGIONS.items():
        pat = re.compile(rf"(<!-- {name}:start -->)(.*?)(<!-- {name}:end -->)", re.S)
        if not pat.search(text):
            raise SystemExit(f"README.md is missing the <!-- {name}:start --> / <!-- {name}:end --> markers")
        body = "\n" + fn() + "\n"
        text = pat.sub(lambda m: m.group(1) + body + m.group(3), text, count=1)
    return text


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    with open(README, encoding="utf8", newline="") as fh:
        current = fh.read()
    new = sync(current.replace("\r\n", "\n"))
    if "--check" in argv:
        if new != current.replace("\r\n", "\n"):
            print("README.md is out of date; run: python scripts/generate_readme_features.py", file=sys.stderr)
            return 1
        return 0
    with open(README, "w", encoding="utf8", newline="\n") as fh:
        fh.write(new)
    print("README.md updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())

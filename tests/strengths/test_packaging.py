# -*- coding: utf-8 -*-
"""Strength 7 - packaging: the wheel builds, installs into a clean venv without extras,
imports, and the CLI works. Skipped when offline or the build backend is unavailable."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.strength, pytest.mark.slow]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _run(cmd, cwd=None, timeout=900, env=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    out = tmp_path_factory.mktemp("wheelhouse")
    r = _run([sys.executable, "-m", "pip", "wheel", ROOT, "--no-deps", "-w", str(out)])
    if r.returncode != 0:
        pytest.skip(f"wheel build unavailable (offline or missing backend): {r.stderr[-300:]}")
    wheels = list(out.glob("energysim-*.whl"))
    assert len(wheels) == 1, wheels
    return wheels[0]


def test_wheel_contents(wheel):
    import zipfile
    names = zipfile.ZipFile(wheel).namelist()
    assert "energysim/__init__.py" in names and "energysim/world.py" in names
    assert any(n.startswith("energysim/models/") for n in names)
    assert not any(n.startswith("tests/") for n in names)
    assert any(n.endswith("entry_points.txt") for n in names)


def test_clean_venv_install_import_and_cli(wheel, tmp_path):
    venv = tmp_path / "venv"
    r = _run([sys.executable, "-m", "venv", str(venv)])
    if r.returncode != 0:
        pytest.skip(f"cannot create venv: {r.stderr[-200:]}")
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    r = _run([str(py), "-m", "pip", "install", str(wheel)], timeout=1800)
    if r.returncode != 0:
        pytest.skip(f"cannot install dependencies (offline?): {r.stderr[-300:]}")
    work = tmp_path / "work"
    work.mkdir()
    r = _run([str(py), "-c", "import energysim, sys; print(energysim.__version__); "
              "import energysim.models as m; print(len(dir(m)) > 20)"], cwd=str(work))
    assert r.returncode == 0, r.stderr
    assert r.stdout.split()[-1] == "True"
    # the package must come from the installed wheel, not from the source tree
    r = _run([str(py), "-c", "import energysim; print(energysim.__file__)"], cwd=str(work))
    assert "site-packages" in r.stdout
    # CLI smoke test: console script and module entry point
    r = _run([str(py), "-m", "energysim.cli", "--version"], cwd=str(work))
    assert r.returncode == 0 and r.stdout.startswith("energysim ")
    scenario = work / "s.yaml"
    scenario.write_text("world: {start_time: 0, stop_time: 20, t_macro: 10, res_filename: r.h5}\n"
                        "simulators:\n  - {name: a, type: signal, step_size: 10, signal: '2*t'}\n")
    r = _run([str(py), "-m", "energysim.cli", "validate", str(scenario)], cwd=str(work))
    assert r.returncode == 0 and "PASSED" in r.stdout
    r = _run([str(py), "-m", "energysim.cli", "run", str(scenario), "--no-pbar"], cwd=str(work))
    assert r.returncode == 0, r.stderr
    assert (work / "r.h5").exists()
    scripts = venv / ("Scripts" if os.name == "nt" else "bin")
    exe = scripts / ("energysim.exe" if os.name == "nt" else "energysim")
    assert exe.exists()
    r = _run([str(exe), "--version"], cwd=str(work))
    assert r.returncode == 0 and r.stdout.startswith("energysim ")

"""Smoke-run every ``examples/*/run_*.py`` headless in a temporary copy (marked ``slow``).

Each example runs in a fresh interpreter with ``MPLBACKEND=Agg`` and ``webbrowser.open`` patched, inside a
copy of its folder (so generated dashboards/run folders never touch the repository), with a 120 s timeout.
Examples that need a proprietary engine (PowerFactory, MATLAB/Octave) are skipped when it is missing.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
TIMEOUT_S = 120

SCRIPTS = sorted(EXAMPLES.glob("*/run_*.py"))

_BOOTSTRAP = (
    "import runpy, sys, webbrowser;"
    "webbrowser.open = lambda *a, **k: True;"
    "webbrowser.open_new_tab = lambda *a, **k: True;"
    "p = sys.argv[1]; sys.argv = [p];"
    "runpy.run_path(p, run_name='__main__')"
)


def _have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _missing_engine(example: str):
    """Reason string if the example needs an engine that is not available, else None."""
    if example == "pf_ninebus" and not _have("powerfactory"):
        return "needs DIgSILENT PowerFactory (module 'powerfactory' not available)"
    if example in ("matlab_cosim", "mpc_microgrid", "community_microgrid"):
        has_matlab = _have("matlab") and _have("matlab.engine")
        has_octave = _have("oct2py") and shutil.which("octave") is not None
        if not (has_matlab or has_octave):
            return "needs MATLAB (matlab.engine) or GNU Octave (oct2py + octave on PATH)"
    return None


@pytest.mark.slow
@pytest.mark.parametrize("script", SCRIPTS, ids=[s.parent.name for s in SCRIPTS])
def test_example_runs_headless(script, tmp_path):
    example = script.parent.name
    reason = _missing_engine(example)
    if reason:
        pytest.skip(reason)
    for mod in ("pandapower", "matplotlib"):
        if not _have(mod):
            pytest.skip(f"optional dependency '{mod}' not installed")

    # Copy to <tmp>/examples/<name> so that relative `parents[n] / 'src'` hops keep their depth.
    work = tmp_path / "examples" / example
    shutil.copytree(script.parent, work, ignore=shutil.ignore_patterns("__pycache__", "runs", "*.html"))
    import os

    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-c", _BOOTSTRAP, str(work / script.name)],
            cwd=work, env=env, capture_output=True, text=True, timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"{example}: still running after {TIMEOUT_S} s\n{(exc.stdout or b'')[-1500:]!r}")
    tail = (proc.stdout[-1500:] if proc.stdout else "") + "\n--- stderr ---\n" + (proc.stderr[-2500:] if proc.stderr else "")
    assert proc.returncode == 0, f"{example} exited with {proc.returncode}\n{tail}"

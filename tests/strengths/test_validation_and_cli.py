# -*- coding: utf-8 -*-
"""Strength 4 - validation and CLI: scenarios are checked before anything runs, with
stable messages and exit codes; safe mode refuses code-executing definitions."""
from __future__ import annotations

import pytest

from energysim import ValidationError, world
from energysim.cli import main

pytestmark = pytest.mark.strength


def _scenario(tmp_path, body, world_cfg="start_time: 0, stop_time: 100, t_macro: 10"):
    f = tmp_path / "s.yaml"
    f.write_text(f"world: {{{world_cfg}}}\n{body}")
    return str(f)


SIGNALS = "simulators:\n  - {name: a, type: signal, step_size: 10, signal: '2*t'}\n"


def run_cli(capsys, *argv):
    rc = main(list(argv))
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def test_valid_scenario_passes_with_exit_code_0(tmp_path, capsys):
    f = _scenario(tmp_path, SIGNALS)
    rc, out, _ = run_cli(capsys, "validate", f)
    assert rc == 0 and out.splitlines()[0] == "ValidationReport: PASSED"
    rc, out, _ = run_cli(capsys, "run", f, "--validate-only", "--no-pbar")
    assert rc == 0 and "Scenario pre-flight validation PASSED." in out


@pytest.mark.parametrize("body,fragment", [
    ("simulators:\n  - {name: a, type: signal, step_size: 0, signal: '2*t'}\n", "Simulator 'a' has invalid step_size: 0 <= 0."),
    ("simulators: []\n", "No simulators registered in the world."),
])
def test_golden_validation_errors(body, fragment, tmp_path, capsys):
    f = _scenario(tmp_path, body)
    rc, out, _ = run_cli(capsys, "validate", f)
    assert rc == 1
    assert out.splitlines()[0] == "ValidationReport: FAILED"
    assert "  Errors (1):" in out and f"    - {fragment}" in out
    rc, out, err = run_cli(capsys, "run", f, "--validate-only")
    assert rc == 1 and f"[ERROR] {fragment}" in err


def test_span_not_multiple_of_macro_step(tmp_path, capsys):
    f = _scenario(tmp_path, SIGNALS, world_cfg="start_time: 0, stop_time: 95, t_macro: 10")
    rc, out, _ = run_cli(capsys, "validate", f)
    assert rc == 1
    assert "stop_time - start_time = 95 is not an integer multiple of the macro step 10" in out


def test_unknown_simulator_in_connection(tmp_path, capsys):
    f = _scenario(tmp_path, SIGNALS + "connections:\n  - {from: a.y, to: nope.u}\n")
    rc, _, err = run_cli(capsys, "validate", f)
    assert rc == 1
    assert "Simulator 'nope' referenced in connection 'nope.u' does not exist" in err
    assert "Available simulators: ['a']" in err


def test_incompatible_units_reported_before_running(tmp_path, capsys):
    body = ("simulators:\n  - {name: a, type: signal, step_size: 10, signal: '2*t', units: {y: kW}}\n"
            "  - {name: b, type: signal, step_size: 10, signal: '1', units: {y: degC}}\n"
            "connections:\n  - {from: a.y, to: b.y}\n")
    rc, _, err = run_cli(capsys, "validate", _scenario(tmp_path, body))
    assert rc == 1
    assert "Connection 'a.y' -> 'b.y': Incompatible unit conversion: cannot convert 'kW' (power) to 'degC' (temperature)." in err


def test_unknown_world_key_and_removed_iterative_coupling(tmp_path, capsys):
    rc, _, err = run_cli(capsys, "validate", _scenario(tmp_path, SIGNALS, "start_time: 0, stop_time: 10, t_macro: 10, zzz: 1"))
    assert rc == 1 and "Unknown world key(s) ['zzz']" in err
    rc, _, err = run_cli(capsys, "validate", _scenario(tmp_path, SIGNALS, "start_time: 0, stop_time: 10, t_macro: 10, coupling: iterative"))
    assert rc == 1 and "iterative coupling was removed in 3.0" in err


def test_missing_file_exit_code(tmp_path, capsys):
    rc, _, err = run_cli(capsys, "validate", str(tmp_path / "missing.yaml"))
    assert rc == 1 and "Scenario file not found" in err
    rc, _, err = run_cli(capsys, "run", str(tmp_path / "missing.yaml"))
    assert rc == 1 and "Scenario file not found" in err


def test_run_executes_and_exits_0(tmp_path, capsys):
    f = _scenario(tmp_path, SIGNALS)
    out_dir = tmp_path / "runs"
    rc, out, _ = run_cli(capsys, "run", f, "--no-pbar", "--out", str(out_dir))
    assert rc == 0 and "Simulation finished successfully." in out
    run_dirs = [d for d in out_dir.iterdir() if d.is_dir()]
    assert len(run_dirs) == 1 and (run_dirs[0] / "meta.json").exists()


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("energysim ")


@pytest.mark.parametrize("sim", [
    "{name: x, type: script, model: ctrl.py}",
    "{name: x, type: external, model: some_dir}",
    "{name: x, type: matlab, model: f.m}",
    "{name: x, type: powerfactory, model: proj}",
    "{name: x, type: powerflow, model: net.p}",
    "{name: x, type: powerflow, model: pandapower.networks.case14}",
])
def test_safe_mode_refusals(sim, tmp_path, capsys):
    f = _scenario(tmp_path, f"simulators:\n  - {sim}\n")
    for argv in (["validate", f, "--safe"], ["run", f, "--safe", "--validate-only"], ["run", f, "--safe"]):
        rc, out, err = run_cli(capsys, *argv)
        assert rc == 1
        assert "safe mode" in err, (argv, err)
    with pytest.raises(ValueError, match="safe mode"):
        world.from_yaml(f, safe=True)


def test_safe_mode_allows_plain_scenarios(tmp_path, capsys):
    f = _scenario(tmp_path, SIGNALS)
    assert run_cli(capsys, "validate", f, "--safe")[0] == 0
    assert run_cli(capsys, "run", f, "--safe", "--no-pbar")[0] == 0


def test_python_api_raises_validation_error_with_same_report(tmp_path):
    w = world(start_time=0, stop_time=95, t_macro=10)
    w.add_signal("a", "1", step_size=10)
    with pytest.raises(ValidationError, match="integer multiple of the macro step"):
        w.simulate(pbar=False)
    rep = w.validate()
    assert not rep.is_valid and str(rep).startswith("ValidationReport: FAILED")

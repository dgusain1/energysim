"""Contract tests of the PowerFactory adapter against the mock ``powerfactory`` module.

These run in CI. They prove the adapter honours the PowerFactory API contract
it relies on; they do NOT prove behaviour in real PowerFactory (see
``tools/powerfactory_manual_check.py`` for that).
"""
import sys
from pathlib import Path

import pytest

from energysim.base import EnergysimError, SimulatorElementNotFoundError, SimulatorVariableError
from energysim.pfAdapter import pf_adapter

MOCK_DIR = str(Path(__file__).resolve().parent.parent / "mocks")


@pytest.fixture
def pfmock(monkeypatch):
    monkeypatch.syspath_prepend(MOCK_DIR)
    sys.modules.pop("powerfactory", None)
    import powerfactory  # the mock
    powerfactory.reset()
    yield powerfactory
    sys.modules.pop("powerfactory", None)


def make(**kw):
    kw.setdefault("project_name", "MockProject")
    kw.setdefault("inputs", ["Load1.plini"])
    kw.setdefault("outputs", ["Bus1.m:u", "Load1.m:P:bus1"])
    return pf_adapter(**kw)


def test_init_get_set_step_roundtrip(pfmock):
    a = make()
    a.init()
    assert a.get_value(["Bus1.m:u", "Load1.m:P:bus1", "Gen1.m:P:bus1"]) == pytest.approx([0.98, 10.0, -10.0])
    a.set_value(["Load1.plini"], [25.0])             # 'e:' prefix is added automatically
    assert a._get_element("Load1").GetAttribute("e:plini") == 25.0
    before = pfmock.STATE["execute_calls"]
    a.step(0)
    assert pfmock.STATE["execute_calls"] == before + 1
    assert a.get_value(["Bus1.m:u"]) == pytest.approx([0.95])
    assert isinstance(a.get_value(["Bus1.m:u"])[0], float)


def test_dc_mode_selects_dc_load_flow(pfmock):
    a = make(mode="dc")
    a.init()
    assert pfmock.STATE["last_ldf_options"] == {"iopt_net": 2}
    assert a.get_value(["Bus1.m:u"]) == [1.0]
    b = make(mode="ac")
    b.init()
    assert pfmock.STATE["last_ldf_options"] == {"iopt_net": 0}


def test_cleanup_and_reinit(pfmock):
    a = make()
    a.init()
    a.set_value(["Load1.plini"], [40.0])
    a.cleanup()
    assert pfmock.STATE["deactivated"] == 1
    with pytest.raises(EnergysimError, match="not initialised"):
        a.get_value(["Bus1.m:u"])
    a.init()                                          # project re-activated, fresh model
    assert a.get_value(["Load1.m:P:bus1"]) == [10.0]


def test_duplicate_names_warn_first_wins(pfmock):
    pfmock.reset(duplicate_load=True)
    a = make()
    with pytest.warns(UserWarning, match="duplicate element names.*'Load1' x2"):
        a.init()
    assert a.duplicate_names == {"Load1": 2}
    assert a.get_value(["Load1.m:P:bus1"]) == [10.0]   # the first Load1, not the duplicate (99)


def test_unknown_element_and_attribute_errors(pfmock):
    a = make()
    a.init()
    with pytest.raises(SimulatorElementNotFoundError, match="Ghost"):
        a.get_value(["Ghost.m:u"])
    with pytest.raises(SimulatorVariableError, match="nope"):
        a.get_value(["Bus1.m:nope"])
    with pytest.raises(SimulatorVariableError):
        a.set_value(["Load1.nope"], [1.0])
    with pytest.raises(SimulatorVariableError, match="Malformed"):
        a.get_value(["justone"])
    with pytest.raises(SimulatorVariableError, match="not numeric"):
        a._get_element("Bus1")._attrs["m:txt"] = "abc"
        a.get_value(["Bus1.m:txt"])


def test_set_attribute_error_code_raises(pfmock):
    a = make()
    a.init()
    pfmock.STATE["set_rc"] = 1
    with pytest.raises(SimulatorVariableError, match="rejected"):
        a.set_value(["Load1.plini"], [1.0])


def test_load_flow_failure_raises(pfmock):
    a = make()
    a.init()
    pfmock.STATE["execute_rc"] = 2
    with pytest.raises(EnergysimError, match="error code 2"):
        a.step(0)


def test_connection_failures(pfmock):
    pfmock.reset(no_application=True)
    with pytest.raises(RuntimeError, match="Could not connect"):
        make().init()
    pfmock.reset(fail_activate=True)
    with pytest.raises(RuntimeError, match="Could not activate project"):
        make().init()
    pfmock.reset()
    with pytest.raises(RuntimeError, match="Could not activate project 'Other'"):
        make(project_name="Other").init()
    pfmock.reset(no_comldf=True)
    with pytest.raises(RuntimeError, match="ComLdf"):
        make().init()


def test_import_error_and_pf_path(monkeypatch):
    sys.modules.pop("powerfactory", None)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != MOCK_DIR])
    a = make()
    with pytest.raises(ImportError, match="pf_path"):
        a.init()
    with pytest.raises(FileNotFoundError):
        make(pf_path=str(Path(MOCK_DIR) / "does_not_exist")).init()
    # a valid pf_path is added to sys.path and the module becomes importable
    b = make(pf_path=MOCK_DIR)
    b.init()          # fresh mock state (module-level reset() runs on import)
    assert b.get_value(["Bus1.m:u"]) == pytest.approx([0.98])
    sys.modules.pop("powerfactory", None)


def test_available_variables_and_modes():
    a = make()
    v = a.get_available_variables()
    assert v["inputs"] == ["Load1.plini"] and v["outputs"] == ["Bus1.m:u", "Load1.m:P:bus1"]
    assert make(pf="ldf").mode == "ac"
    with pytest.raises(ValueError, match="only supports 'ac' and 'dc'"):
        make(mode="rms")


def test_manual_check_script_runs_green_against_mock(tmp_path):
    """The user-facing manual-check tool must itself work (here against the mock API)."""
    import json
    import subprocess
    script = Path(__file__).resolve().parents[2] / "tools" / "powerfactory_manual_check.py"
    out = tmp_path / "result.json"
    proc = subprocess.run(
        [sys.executable, str(script), "--project", "MockProject", "--pf-path", MOCK_DIR,
         "--element", "Bus1", "--attr", "m:u", "--input-element", "Load1", "--input-attr", "plini",
         "--test-dc", "--json-out", str(out)],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(out.read_text())
    assert report["overall"] == "PASS"
    assert not [s for s in report["steps"] if s["status"] == "FAIL"]
    assert "overall: PASS" in proc.stdout
    # wrong project must produce a FAIL report and a non-zero exit code
    bad = subprocess.run(
        [sys.executable, str(script), "--project", "Nope", "--pf-path", MOCK_DIR, "--element", "Bus1",
         "--attr", "m:u", "--json-out", str(tmp_path / "bad.json")],
        capture_output=True, text=True, timeout=120)
    assert bad.returncode == 1 and "FAIL" in bad.stdout

"""Contract test: every adapter must honour the same ``SimulatorAdapter`` life-cycle.

Each entry of ``SPECS`` builds an adapter plus enough information to exercise the
contract: reading outputs, a set -> step -> get round trip (where meaningful),
``advance`` / ``advance_with_recording`` semantics, ``cleanup`` followed by a
second ``init`` and the shape of ``get_available_variables``.

Adapters whose optional dependency is missing are skipped.
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from energysim.base import SimulatorAdapter

TESTS = Path(__file__).resolve().parent
MOCKS = str(TESTS / "mocks")


class Spec:
    def __init__(self, adapter, outputs, set_var=None, set_val=None, read_var=None, expect=None,
                 step=1.0, tol=1e-9, deterministic=True, empty_ok=True):
        self.adapter = adapter
        self.outputs = outputs        # variables readable right after init()
        self.set_var = set_var        # variable to write (None: read-only adapter)
        self.set_val = set_val
        self.read_var = read_var      # variable that must reflect the write after step()
        self.expect = expect          # expected value of read_var (default: set_val)
        self.step = step
        self.tol = tol
        self.deterministic = deterministic
        self.empty_ok = empty_ok      # get_value([]) must return []


# ---------------------------------------------------------------------------
# factories
# ---------------------------------------------------------------------------

def _csv(tmp_path):
    from energysim.csv_adapter import csv_simulator
    f = tmp_path / "d.csv"
    f.write_text("time,temp\n0,1.0\n1,2.0\n2,3.0\n3,4.0\n")
    a = csv_simulator("c", str(f), step_size=1, outputs=["temp"])
    a.step_size = 1.0
    return Spec(a, ["temp"])


def _signal(tmp_path):
    from energysim.signalAdapter import signal_adapter
    a = signal_adapter("s", lambda t: [2.0 * t])
    a.step_size = 1.0
    # known defect (signalAdapter.py): get_value ignores `parameters`, so get_value([]) == [value]
    return Spec(a, ["y"], empty_ok=False)


def _script(tmp_path):
    from energysim.pyScriptAdapter import py_adapter
    (tmp_path / "echo_mod.py").write_text("def run(inputs):\n    return [inputs.get('u', 0.0) * 2.0]\n")
    a = py_adapter("echo_mod.run", str(tmp_path), inputs=["u"], outputs=["y"])
    a.step_size = 1.0
    return Spec(a, ["y"], set_var="u", set_val=4.0, read_var="y", expect=8.0)


def _external(tmp_path):
    from energysim.external_simulator import external_simulator
    a = external_simulator(inputs=["u"], outputs=["y"])
    a.step_size = 1.0
    return Spec(a, ["y"], set_var="y", set_val=5.0, read_var="y")


def _pandapower(tmp_path):
    pytest.importorskip("pandapower")
    from adapters.conftest import build_pp_net
    from energysim.ppAdapter import pp_adapter
    a = pp_adapter("n", model=build_pp_net(), inputs=["Load1.p_mw"], outputs=["MV2.vm_pu"])
    a.step_size = 60.0
    return Spec(a, ["MV2.vm_pu", "Line1.loading_percent"], set_var="Load1.p_mw", set_val=14.0,
                read_var="Load1.p_mw", step=60.0)


def _pypsa(tmp_path):
    pytest.importorskip("pypsa")
    from adapters.conftest import build_pypsa_net
    from energysim.pypsaAdapter import pypsa_adapter
    a = pypsa_adapter("p", model=build_pypsa_net(), inputs=["ld.P"], outputs=["b1.V"])
    a.step_size = 60.0
    return Spec(a, ["b1.V", "g0.P"], set_var="ld.P", set_val=44.0, read_var="ld.P", step=60.0)


def _opendss(tmp_path):
    pytest.importorskip("opendssdirect")
    import energysim.opendssAdapter as mod
    mod._ACTIVE = None
    from energysim.opendssAdapter import opendss_adapter
    a = opendss_adapter("d", str(TESTS / "fixtures" / "feeder3.dss"), inputs=["Load.ld1.kW"],
                        outputs=["Bus.n2.vm_pu"], step_size=60.0)
    return Spec(a, ["Bus.n2.vm_pu", "Circuit.losses_kw"], set_var="Load.ld1.kW", set_val=650.0,
                read_var="Load.ld1.p_kw", step=60.0, tol=1e-3)


def _fmu(kind):
    def factory(tmp_path):
        pytest.importorskip("fmpy")
        from fixtures.fetch_reference_fmus import get_reference_fmus
        refs = get_reference_fmus()
        if refs is None:
            pytest.skip("Reference FMUs unavailable")
        path = str(refs["Feedthrough"][kind])
        if kind == "cs":
            from energysim.csAdapter import FmuCsAdapter
            a = FmuCsAdapter(path, step_size=0.5, inputs=["Float64_continuous_input"],
                             outputs=["Float64_continuous_output"])
        else:
            from energysim.meAdapter import FmuMeAdapter
            a = FmuMeAdapter(path, step_size=0.5, inputs=["Float64_continuous_input"],
                             outputs=["Float64_continuous_output"], solver_name="Euler")
        a.step_size = 0.5
        return Spec(a, ["Float64_continuous_output"], set_var="Float64_continuous_input", set_val=6.5,
                    read_var="Float64_continuous_output", step=0.5)
    return factory


def _matlab_mock(tmp_path, monkeypatch=None):
    from energysim.matlabAdapter import matlab_adapter

    class Fake:
        def __init__(self):
            pass

        def addpath(self, p):
            pass

        def feval(self, name, *args, nout=1):
            if name == "nargin":
                return 2
            if name == "nargout":
                return 1
            return float(args[0]) * 2.0

        def exit(self):
            pass

    mod = types.ModuleType("oct2py")
    mod.Oct2Py = Fake
    sys.modules["oct2py"] = mod
    a = matlab_adapter("f", str(tmp_path), inputs=["u"], outputs=["y"], engine="octave")
    a.step_size = 1.0
    return Spec(a, ["y"], set_var="u", set_val=3.0, read_var="y", expect=6.0)


def _powerfactory_mock(tmp_path):
    if MOCKS not in sys.path:
        sys.path.insert(0, MOCKS)
    sys.modules.pop("powerfactory", None)
    import powerfactory
    powerfactory.reset()
    from energysim.pfAdapter import pf_adapter
    a = pf_adapter(project_name="MockProject", inputs=["Load1.plini"], outputs=["Bus1.m:u", "Load1.m:P:bus1"], pf_path=MOCKS)
    a.step_size = 60.0
    return Spec(a, ["Bus1.m:u", "Load1.m:P:bus1"], set_var="Load1.plini", set_val=30.0,
                read_var="Load1.m:P:bus1", step=60.0)


FACTORIES = {
    "csv": _csv,
    "signal": _signal,
    "script": _script,
    "external_template": _external,
    "pandapower": _pandapower,
    "pypsa": _pypsa,
    "opendss": _opendss,
    "fmu_cs": _fmu("cs"),
    "fmu_me": _fmu("me"),
    "matlab_mock": _matlab_mock,
    "powerfactory_mock": _powerfactory_mock,
}

ALL = list(FACTORIES)
# py_adapter.cleanup() drops its module/function and init() does not reload them, and its
# input/output state survives -> not re-runnable (pyScriptAdapter.py is outside this task's scope).
RERUN = [pytest.param(n, marks=pytest.mark.xfail(
    reason="py_adapter is not re-runnable after cleanup() (needs fix in pyScriptAdapter.py)", strict=False))
    if n == "script" else n for n in ALL]


@pytest.fixture
def spec(request, tmp_path):
    """Build the adapter under test; always clean up (OpenDSS/PowerFactory are process-wide)."""
    saved_oct = sys.modules.get("oct2py")
    s = FACTORIES[request.param](tmp_path)
    yield s
    try:
        s.adapter.cleanup()
    except Exception:
        pass
    sys.modules.pop("powerfactory", None)
    if saved_oct is None:
        sys.modules.pop("oct2py", None)
    else:
        sys.modules["oct2py"] = saved_oct
    if "energysim.opendssAdapter" in sys.modules:
        sys.modules["energysim.opendssAdapter"]._ACTIVE = None


def configure_if_any(a):
    if hasattr(a, "configure"):
        a.configure(0.0, 10.0 * a.step_size, a.step_size)


def numeric(x):
    return isinstance(x, (int, float, np.integer, np.floating, bool, np.bool_))


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", ALL, indirect=True)
def test_interface_and_class_attributes(spec):
    a = spec.adapter
    assert isinstance(a, SimulatorAdapter)
    for m in ("init", "cleanup", "step", "get_value", "set_value", "advance", "advance_with_recording",
              "set_parameters", "get_available_variables"):
        assert callable(getattr(a, m)), m
    assert isinstance(getattr(type(a), "SET_PARAMS_BEFORE_INIT", False), bool)
    assert a.step_size > 0


@pytest.mark.parametrize("spec", ALL, indirect=True)
def test_get_value_types_and_lengths(spec):
    a = spec.adapter
    configure_if_any(a)
    a.init()
    vals = a.get_value(list(spec.outputs), 0.0)
    assert isinstance(vals, list) and len(vals) == len(spec.outputs)
    assert all(numeric(v) for v in vals), vals
    if spec.empty_ok:
        assert a.get_value([], 0.0) == []
    assert a.get_value([spec.outputs[0]], 0.0) == pytest.approx([vals[0]])


@pytest.mark.parametrize("spec", ALL, indirect=True)
def test_set_then_get_round_trip(spec):
    a = spec.adapter
    if spec.set_var is None:
        pytest.skip("read-only adapter")
    configure_if_any(a)
    a.init()
    a.set_value([spec.set_var], [spec.set_val])
    a.advance(0.0, spec.step)
    expect = spec.set_val if spec.expect is None else spec.expect
    got = a.get_value([spec.read_var], spec.step)[0]
    assert got == pytest.approx(expect, rel=spec.tol, abs=spec.tol)


@pytest.mark.parametrize("spec", ALL, indirect=True)
def test_advance_and_recording_semantics(spec):
    a = spec.adapter
    configure_if_any(a)
    a.init()
    t1 = 3 * spec.step
    rows = a.advance_with_recording(0.0, t1, list(spec.outputs))
    assert len(rows) >= 3
    assert rows[0][0] == 0.0
    assert all(len(r) == 1 + len(spec.outputs) for r in rows)
    times = [r[0] for r in rows]
    assert times == sorted(times) and times[-1] < t1 + 1e-9      # recording happens *before* each step
    a.advance(t1, 5 * spec.step)                                   # continues from where the last call ended
    assert all(numeric(v) for v in a.get_value(list(spec.outputs), 5 * spec.step))
    a.advance(5 * spec.step, 5 * spec.step)                        # empty interval is a no-op
    # recording without outputs still yields time stamps
    rows0 = a.advance_with_recording(5 * spec.step, 6 * spec.step, [])
    assert rows0 and all(len(r) == 1 for r in rows0)


@pytest.mark.parametrize("spec", RERUN, indirect=True)
def test_cleanup_then_init_again(spec):
    a = spec.adapter
    configure_if_any(a)
    a.init()
    first = a.get_value(list(spec.outputs), 0.0)
    if spec.set_var is not None:
        a.set_value([spec.set_var], [spec.set_val])
    a.advance(0.0, 2 * spec.step)
    a.cleanup()
    a.cleanup()                         # idempotent
    configure_if_any(a)
    a.init()                            # re-runnable
    again = a.get_value(list(spec.outputs), 0.0)
    assert again == pytest.approx(first, rel=1e-9, abs=1e-12)
    a.advance(0.0, spec.step)


@pytest.mark.parametrize("spec", ALL, indirect=True)
def test_available_variables_shape(spec):
    av = spec.adapter.get_available_variables()
    assert isinstance(av, dict) and "inputs" in av and "outputs" in av
    for key, lst in av.items():
        assert isinstance(lst, list) and all(isinstance(x, str) for x in lst), key
    configure_if_any(spec.adapter)
    spec.adapter.init()
    declared = set(av["inputs"]) | set(av["outputs"]) | set(av.get("all", []))
    after = spec.adapter.get_available_variables()
    declared |= set(after["inputs"]) | set(after["outputs"]) | set(after.get("all", []))
    if declared:   # adapters that introspect must know the variables the test uses
        for v in spec.outputs:
            assert v in declared, f"{v} missing from get_available_variables()"


@pytest.mark.parametrize("spec", ALL, indirect=True)
def test_set_parameters_default_path(spec):
    a = spec.adapter
    if spec.set_var is None:
        pytest.skip("read-only adapter")
    configure_if_any(a)
    if getattr(type(a), "SET_PARAMS_BEFORE_INIT", False):
        a.set_parameters({spec.set_var: spec.set_val})
        a.init()
    else:
        a.init()
        a.set_parameters({spec.set_var: spec.set_val})
    a.advance(0.0, spec.step)
    expect = spec.set_val if spec.expect is None else spec.expect
    assert a.get_value([spec.read_var], spec.step)[0] == pytest.approx(expect, rel=spec.tol, abs=spec.tol)

"""MATLAB / Octave adapter tests: a mock engine (always) and real Octave (when installed)."""
import shutil
import sys
import types

import numpy as np
import pytest

from energysim.base import EnergysimError, SimulatorVariableError
from energysim.matlabAdapter import matlab_adapter


class FakeOct2Py:
    """Stand-in for ``oct2py.Oct2Py`` driven by a registry of python callables."""
    registry = {}
    instances = []
    fail_exit = False

    def __init__(self):
        self.paths = []
        self.exited = False
        FakeOct2Py.instances.append(self)

    def addpath(self, p):
        self.paths.append(p)

    def feval(self, name, *args, nout=1):
        if name == "nargin":
            return self.registry[args[0]][0]
        if name == "nargout":
            return self.registry[args[0]][1]
        n_in, n_out, fn = self.registry[name]
        res = fn(*args)
        return res

    def exit(self):
        self.exited = True
        if FakeOct2Py.fail_exit:
            raise RuntimeError("octave already dead")


@pytest.fixture
def fake_octave(monkeypatch, tmp_path):
    FakeOct2Py.registry = {}
    FakeOct2Py.instances = []
    FakeOct2Py.fail_exit = False
    mod = types.ModuleType("oct2py")
    mod.Oct2Py = FakeOct2Py
    monkeypatch.setitem(sys.modules, "oct2py", mod)
    return tmp_path


def make(name="heat", inputs=("p", "t"), outputs=("q", "cop"), loc=None, **kw):
    return matlab_adapter(name, str(loc), inputs=list(inputs), outputs=list(outputs), engine="octave", **kw)


def test_mock_roundtrip_scalar_and_multi_output(fake_octave):
    FakeOct2Py.registry["heat"] = (3, 2, lambda p, t, time: (p * 3.0 + time, np.array([[t / 2.0]])))
    a = make(loc=fake_octave)
    a.init()
    a.set_value(["p", "t"], [2, np.float64(10)])
    a.step(5.0)
    assert a.get_value(["q", "cop"], 5.0) == [11.0, 5.0]
    assert all(isinstance(v, float) for v in a.get_value(["q", "cop"]))
    assert a.get_available_variables() == {"inputs": ["p", "t"], "outputs": ["q", "cop"],
                                           "all": ["p", "t", "q", "cop"]}


def test_inputs_are_floats_or_arrays(fake_octave):
    seen = []
    FakeOct2Py.registry["f"] = (3, 1, lambda a, b, time: seen.append((a, b, time)) or 1.0)
    a = make("f", inputs=("a", "b"), outputs=("y",), loc=fake_octave)
    a.init()
    a.set_value(["a", "b"], [3, [1, 2, 3]])
    a.step(1)
    assert isinstance(seen[0][0], float) and seen[0][0] == 3.0
    assert isinstance(seen[0][1], np.ndarray) and seen[0][1].dtype == float
    assert isinstance(seen[0][2], float)
    with pytest.raises(SimulatorVariableError, match="not a declared input"):
        a.set_value(["zzz"], [1])


def test_to_float_failures_raise_instead_of_returning_zero():
    to_f = matlab_adapter._to_float
    assert to_f(3) == 3.0 and to_f(np.array([[4.5, 1]])) == 4.5 and to_f([7]) == 7.0
    for bad in (None, "abc", [], np.array([]), object()):
        with pytest.raises(SimulatorVariableError):
            to_f(bad, "y")


def test_garbage_result_raises_during_step(fake_octave):
    FakeOct2Py.registry["f"] = (2, 1, lambda a, time: "not a number")
    a = make("f", inputs=("a",), outputs=("y",), loc=fake_octave)
    a.init()
    with pytest.raises(SimulatorVariableError, match="output 'y'"):
        a.step(0)


def test_engine_error_in_function_raises(fake_octave):
    def boom(a, time):
        raise RuntimeError("division by zero in heatpump.m")
    FakeOct2Py.registry["f"] = (2, 1, boom)
    a = make("f", inputs=("a",), outputs=("y",), loc=fake_octave)
    a.init()
    with pytest.raises(EnergysimError, match="failed at t=3.*division by zero"):
        a.step(3.0)


def test_wrong_number_of_returned_values_raises(fake_octave):
    FakeOct2Py.registry["f"] = (2, 2, lambda a, time: (1.0,))
    a = make("f", inputs=("a",), outputs=("y", "z"), loc=fake_octave)
    a.init()
    with pytest.raises(SimulatorVariableError, match="returned 1 value"):
        a.step(0)


def test_signature_validation(fake_octave):
    FakeOct2Py.registry["f"] = (5, 1, lambda *a: 1.0)
    with pytest.raises(ValueError, match="expects 5 input"):
        make("f", inputs=("a",), outputs=("y",), loc=fake_octave).init()
    # a failed init must not leave a live engine behind
    assert FakeOct2Py.instances[-1].exited
    FakeOct2Py.registry["g"] = (2, 3, lambda *a: 1.0)
    with pytest.raises(ValueError, match="returns 3 output"):
        make("g", inputs=("a",), outputs=("y",), loc=fake_octave).init()
    with pytest.raises(RuntimeError, match="does not appear to be a valid"):
        make("missing_function", inputs=("a",), outputs=("y",), loc=fake_octave).init()


def test_cleanup_is_robust_and_idempotent_and_reinit_works(fake_octave):
    FakeOct2Py.registry["f"] = (2, 1, lambda a, time: a + time)
    a = make("f", inputs=("a",), outputs=("y",), loc=fake_octave)
    a.cleanup()                       # before init: no-op
    a.init()
    first_engine = a.engine
    a.set_value(["a"], [1])
    a.step(1.0)
    FakeOct2Py.fail_exit = True       # engine dies: cleanup must swallow it
    a.cleanup()
    a.cleanup()
    assert a.engine is None and first_engine.exited
    FakeOct2Py.fail_exit = False
    a.init()                          # new engine, clean state
    assert a.engine is not first_engine
    assert a.get_value(["y"]) == [0.0]
    a.set_value(["a"], [1])
    a.step(1.0)
    assert a.get_value(["y"]) == [2.0]
    with pytest.raises(EnergysimError, match="not initialised"):
        a.cleanup()
        a.step(0.0)


def test_init_errors(tmp_path, fake_octave):
    with pytest.raises(FileNotFoundError):
        make("f", loc=tmp_path / "nonexistent").init()
    with pytest.raises(ValueError, match="engine must be"):
        matlab_adapter("f", str(tmp_path), engine="python")
    with pytest.raises(SimulatorVariableError, match="not a declared output"):
        FakeOct2Py.registry["f"] = (2, 1, lambda a, time: 1.0)
        a = make("f", inputs=("a",), outputs=("y",), loc=tmp_path)
        a.init()
        a.get_value(["nope"])


def test_no_engine_available_message(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "oct2py", None)   # makes `import oct2py` fail
    with pytest.raises(RuntimeError, match="Neither MATLAB Engine nor oct2py"):
        make("f", loc=tmp_path).init()


# ---------------------------------------------------------------------------
# Real Octave
# ---------------------------------------------------------------------------

def _real_octave():
    pytest.importorskip("oct2py")
    if shutil.which("octave") is None and shutil.which("octave-cli") is None:
        pytest.skip("GNU Octave is not installed")


@pytest.mark.requires_octave
def test_real_octave_heatpump(tmp_path):
    _real_octave()
    (tmp_path / "heatpump.m").write_text(
        "function [Q_thermal, COP] = heatpump(P_cmd, T_source, T_sink, time)\n"
        "  COP_carnot = (T_sink + 273.15) / max(T_sink - T_source, 0.1);\n"
        "  COP = 0.45 * COP_carnot;\n"
        "  Q_thermal = P_cmd * COP;\n"
        "end\n")
    a = matlab_adapter("heatpump", str(tmp_path), inputs=["P", "Ts", "Tk"], outputs=["Q", "COP"],
                       engine="octave")
    a.init()
    try:
        a.set_value(["P", "Ts", "Tk"], [2.0, 5.0, 35.0])
        a.step(0.0)
        q, cop = a.get_value(["Q", "COP"])
        exp_cop = 0.45 * (35 + 273.15) / (35 - 5)
        assert cop == pytest.approx(exp_cop) and q == pytest.approx(2.0 * exp_cop)
    finally:
        a.cleanup()
    a.init()                          # restartable
    a.cleanup()


@pytest.mark.requires_octave
def test_real_octave_array_input_and_error(tmp_path):
    _real_octave()
    (tmp_path / "vsum.m").write_text("function y = vsum(v, time)\n  y = sum(v) + time;\nend\n")
    (tmp_path / "bad.m").write_text("function y = bad(a, time)\n  error('boom from octave');\nend\n")
    a = matlab_adapter("vsum", str(tmp_path), inputs=["v"], outputs=["y"], engine="octave")
    a.init()
    try:
        a.set_value(["v"], [[1.0, 2.0, 3.0]])
        a.step(10.0)
        assert a.get_value(["y"]) == [16.0]
    finally:
        a.cleanup()
    b = matlab_adapter("bad", str(tmp_path), inputs=["a"], outputs=["y"], engine="octave")
    b.init()
    try:
        with pytest.raises(EnergysimError, match="boom from octave"):
            b.step(0.0)
    finally:
        b.cleanup()


@pytest.mark.requires_matlab
def test_real_matlab_engine_available():
    pytest.importorskip("matlab.engine")
    pytest.skip("MATLAB licence tests are manual; engine import works")

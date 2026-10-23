# -*- coding: utf-8 -*-
"""
Framework-level tests applied to EVERY ModelComponent in energysim.models:
valid outputs at init, state untouched by init, re-runnability, strict
variable access, introspection.
"""

import enum
import inspect
import math
import numbers
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import energysim.models as models  # noqa: E402
from energysim.base import SimulatorVariableError
from energysim.models.common import ModelComponent, OperatingState, OperatingStateMachine


def _all_model_classes():
    out = []
    for n in sorted(dir(models)):
        obj = getattr(models, n)
        if inspect.isclass(obj) and issubclass(obj, ModelComponent) and obj is not ModelComponent:
            out.append(obj)
    return out


ALL_MODELS = _all_model_classes()
IDS = [c.__name__ for c in ALL_MODELS]


def _norm(obj, depth=0):
    """Reduce an object graph to comparable primitives (curves/callables ignored)."""
    if depth > 8:
        return "<deep>"
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, (bool, str, type(None))):
        return obj
    if isinstance(obj, numbers.Real):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _norm(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_norm(v, depth + 1) for v in obj]
    if type(obj).__name__ in ("Curve1D", "Surface2D") or inspect.isroutine(obj):
        return "<curve>"
    if hasattr(obj, "__dict__"):
        return {k: _norm(v, depth + 1) for k, v in vars(obj).items()}
    return repr(obj)


def _state(m, include_outputs=False):
    skip = {"_initial_snapshot"}
    if not include_outputs:
        skip.add("_outputs_val")
    return _norm({k: v for k, v in vars(m).items() if k not in skip})


def _random_inputs(m, rng):
    vals = []
    for n in m.inputs:
        if n.startswith("T_") or n.lower().startswith("t_"):
            vals.append(float(rng.uniform(5.0, 40.0)))
        else:
            vals.append(float(rng.uniform(0.0, 1.0)) * 5.0)
    return vals


def _finite(v):
    if isinstance(v, str):
        return True
    return math.isfinite(float(v))


def test_discovers_all_models():
    assert len(ALL_MODELS) >= 25


@pytest.mark.parametrize("cls", ALL_MODELS, ids=IDS)
def test_init_leaves_state_unchanged_and_outputs_valid(cls):
    m = cls()
    before = _state(m)
    m.init()
    assert _state(m) == before, "init() must not advance state"
    # Every declared output is available and finite right after init
    vals = m.get_value(m.outputs, 0.0)
    assert len(vals) == len(m.outputs) > 0
    for n, v in zip(m.outputs, vals):
        assert isinstance(v, (float, int, bool, str, np.floating, np.integer)), (n, v)
        assert _finite(v), (n, v)


@pytest.mark.parametrize("cls", ALL_MODELS, ids=IDS)
def test_rerun_restores_initial_state(cls):
    m = cls()
    m.init()
    initial_all = _state(m, include_outputs=True)
    rng = np.random.default_rng(7)
    for k in range(25):
        m.set_value(m.inputs, _random_inputs(m, rng))
        m.step(float(k))
    m.init()
    assert _state(m, include_outputs=True) == initial_all
    # reset() is equivalent
    m.set_value(m.inputs, _random_inputs(m, rng))
    m.advance(0.0, 10.0)
    m.reset()
    assert _state(m, include_outputs=True) == initial_all


@pytest.mark.parametrize("cls", ALL_MODELS, ids=IDS)
def test_two_runs_are_identical(cls):
    def run():
        m = cls()
        m.init()
        rng = np.random.default_rng(3)
        trace = []
        for k in range(30):
            m.set_value(m.inputs, _random_inputs(m, rng))
            m.step(float(k))
            trace.append(_norm(m.get_value(m.outputs, float(k))))
        return m, trace

    m, t1 = run()
    m.init()
    rng = np.random.default_rng(3)
    t2 = []
    for k in range(30):
        m.set_value(m.inputs, _random_inputs(m, rng))
        m.step(float(k))
        t2.append(_norm(m.get_value(m.outputs, float(k))))
    assert t1 == t2


@pytest.mark.parametrize("cls", ALL_MODELS, ids=IDS)
def test_strict_variable_access(cls):
    m = cls()
    m.init()
    with pytest.raises(SimulatorVariableError):
        m.set_value(["definitely_not_a_variable"], [1.0])
    with pytest.raises(SimulatorVariableError):
        m.get_value(["definitely_not_a_variable"], 0.0)
    # Methods and arbitrary attributes must never be overwritten
    for bad in ("step", "step_model", "init", "name", "_states", "sm"):
        if bad in m.inputs:
            continue
        with pytest.raises(SimulatorVariableError):
            m.set_value([bad], [1.0])
    step_before = type(m).step
    assert m.step.__func__ is step_before


@pytest.mark.parametrize("cls", ALL_MODELS, ids=IDS)
def test_available_variables_and_describe(cls):
    m = cls()
    av = m.get_available_variables()
    assert set(av) == {"inputs", "outputs"}
    assert av["inputs"] == m.inputs and av["outputs"] == m.outputs
    d = m.describe()
    assert d["class"] == cls.__name__
    assert d["inputs"] == av["inputs"] and d["outputs"] == av["outputs"]
    assert isinstance(d["doc"], str) and len(d["doc"]) > 20
    assert isinstance(d["parameters"], dict)
    assert cls.describe_class()["class"] == cls.__name__
    # every declared input is settable and readable back
    for n in m.inputs:
        m.set_value([n], [1.0])
        assert m.get_value([n], 0.0)[0] == 1.0


@pytest.mark.parametrize("cls", ALL_MODELS, ids=IDS)
def test_units_cover_all_variables(cls):
    m = cls()
    missing = [n for n in m.inputs + m.outputs if n not in cls.units]
    assert not missing, f"{cls.__name__} lacks units for {missing}"
    d = m.describe()
    assert set(d["units"]) == set(m.inputs + m.outputs)


def test_no_rollback_api_on_model_component():
    assert "save_state" not in ModelComponent.__dict__
    assert "restore_state" not in ModelComponent.__dict__
    assert "_state_snapshot" not in vars(ModelComponent())


class _Toy(ModelComponent):
    """Toy integrator used to test registered parameters."""
    units = {"u": "1", "y": "1"}

    def __init__(self):
        super().__init__(name="toy")
        self.gain = 2.0
        self.label = "text"
        self.acc = 0.0
        self.register_variables(inputs=["u"], outputs=["y"])
        self.register_parameters("gain", "label")

    def step_model(self, dt, time):
        self.acc += self._inputs_val["u"] * self.gain * dt
        self._outputs_val["y"] = self.acc


def test_registered_parameters_numeric_only():
    t = _Toy()
    t.init()
    t.set_value(["gain"], [3.0])
    assert t.gain == 3.0 and t.get_value(["gain"], 0.0) == [3.0]
    with pytest.raises(SimulatorVariableError):
        t.set_value(["gain"], ["fast"])
    with pytest.raises(SimulatorVariableError):
        t.set_value(["label"], [1.0])  # non-numeric attribute
    with pytest.raises(SimulatorVariableError):
        t.set_value(["acc"], [5.0])  # state is not a parameter
    with pytest.raises(AttributeError):
        t.register_parameters("missing")


def test_toy_init_idempotent_and_reset():
    t = _Toy()
    t.init()
    t.set_value(["u"], [1.0])
    t.advance(0.0, 5.0)
    assert t.get_value(["y"], 5.0)[0] == pytest.approx(10.0)
    t.reset()
    assert t.acc == 0.0 and t.get_value(["y"], 0.0)[0] == 0.0
    assert t.get_value(["u"], 0.0)[0] == 0.0  # input restored to its default too


def test_state_machine_transition_and_timers():
    sm = OperatingStateMachine(min_uptime_sec=100.0, min_downtime_sec=50.0)
    assert not hasattr(sm, "is_starting_up")
    sm.update_time(10.0)
    assert sm.time_in_state_sec == 10.0
    assert sm.transition(OperatingState.RUNNING) is True
    assert sm.time_in_state_sec == 0.0
    assert sm.transition(OperatingState.RUNNING) is False  # no change -> timer kept
    sm.update_time(60.0)
    assert not sm.can_stop()
    sm.update_time(60.0)
    assert sm.can_stop()
    sm.transition(OperatingState.OFF)
    sm.update_time(10.0)
    assert not sm.can_start()
    sm.update_time(50.0)
    assert sm.can_start()

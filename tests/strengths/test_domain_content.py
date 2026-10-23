# -*- coding: utf-8 -*-
"""Strength 6 - domain content: every shipped model is self-describing, unit-annotated,
valid immediately after init() and deterministic across re-runs."""
from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

import energysim.models as models
from energysim.models.common import ModelComponent

pytestmark = pytest.mark.strength


def _classes():
    return [getattr(models, n) for n in sorted(dir(models))
            if inspect.isclass(getattr(models, n)) and issubclass(getattr(models, n), ModelComponent)
            and getattr(models, n) is not ModelComponent]


ALL = _classes()
IDS = [c.__name__ for c in ALL]


def test_model_library_is_not_empty():
    assert len(ALL) >= 20


@pytest.mark.parametrize("cls", ALL, ids=IDS)
def test_describe_is_complete(cls):
    d = cls().describe()
    assert d["class"] == cls.__name__
    assert len(d["doc"]) > 80, "models must carry a substantive docstring"
    assert d["inputs"] and d["outputs"]
    assert set(d) >= {"inputs", "outputs", "parameters", "units", "doc", "class"}


@pytest.mark.parametrize("cls", ALL, ids=IDS)
def test_every_variable_has_a_valid_unit(cls):
    from energysim.utils import is_known_unit
    m = cls()
    units = cls.units
    missing = [v for v in m.inputs + m.outputs if v not in units]
    assert not missing, f"{cls.__name__}: no unit for {missing}"
    # string/state outputs use '-', '1' or similar; anything else must be a recognised unit
    # or a documented model-specific engineering unit (checked below).
    engineering = {"kWh/kg", "A", "A/cm2", "V", "W/m2", "m/s", "kg", "MMBtu/h", "kg/h", "t/h", "Nm3/h"}
    for v in m.inputs + m.outputs:
        u = units[v]
        assert is_known_unit(u) or u in engineering, f"{cls.__name__}.{v}: unit {u!r} not recognised"


@pytest.mark.parametrize("cls", ALL, ids=IDS)
def test_initial_outputs_are_valid_without_stepping(cls):
    m = cls()
    m.init()
    for name, val in zip(m.outputs, m.get_value(m.outputs, 0.0)):
        if isinstance(val, str):
            assert val
        else:
            assert math.isfinite(float(val)), f"{cls.__name__}.{name} = {val!r} after init()"


@pytest.mark.parametrize("cls", ALL, ids=IDS)
def test_rerun_determinism(cls):
    def trace(m, seed):
        rng = np.random.default_rng(seed)
        out = []
        for k in range(40):
            vals = [float(rng.uniform(5, 35)) if n.lower().startswith("t_") else float(rng.uniform(0, 5))
                    for n in m.inputs]
            m.set_value(m.inputs, vals)
            m.step(float(k))
            out.append([v if isinstance(v, str) else round(float(v), 12) for v in m.get_value(m.outputs, float(k))])
        return out

    m = cls()
    m.init()
    first = trace(m, 123)
    m.init()
    assert trace(m, 123) == first
    fresh = cls()
    fresh.init()
    assert trace(fresh, 123) == first

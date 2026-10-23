# -*- coding: utf-8 -*-
"""Features removed in 3.0 must stay removed."""
from __future__ import annotations

import inspect

import pytest

import energysim
from energysim import SimulatorAdapter, world
from energysim.container import VectorizedContainerAdapter
from energysim.models import PEMElectrolyzer
from energysim.models.common import ModelComponent

pytestmark = pytest.mark.strength

ROLLBACK_NAMES = ("save_state", "restore_state")


@pytest.mark.parametrize("obj", [SimulatorAdapter, ModelComponent, VectorizedContainerAdapter, world, PEMElectrolyzer])
def test_no_rollback_api_on_public_classes(obj):
    for n in ROLLBACK_NAMES:
        assert not hasattr(obj, n), f"{obj.__name__}.{n} must stay removed"


def test_no_rollback_names_exported():
    public = dir(energysim)
    assert not [n for n in public if "save_state" in n or "restore_state" in n]
    import energysim.models as models
    for name in dir(models):
        obj = getattr(models, name)
        if inspect.isclass(obj):
            assert not any(hasattr(obj, n) for n in ROLLBACK_NAMES), name


def test_iterative_coupling_raises(tmp_path):
    with pytest.raises(ValueError, match="iterative coupling was removed in 3.0"):
        world(coupling="iterative")
    w = world(start_time=0, stop_time=10, t_macro=10)
    w.add_signal("a", "1", step_size=10)
    w.coupling = "iterative"
    with pytest.raises(ValueError, match="iterative coupling was removed in 3.0"):
        w.simulate(pbar=False)
    with pytest.raises(ValueError):
        world.from_dict({"world": {"coupling": "iterative"}, "simulators": []})


def test_iterative_world_options_removed():
    for key in ("max_iterations", "tolerance", "iterative"):
        with pytest.raises((ValueError, TypeError)):
            world.from_dict({"world": {key: 5}, "simulators": []})


def test_world_results_absent_once_results_object_exists(tmp_path):
    """Conditional: before the Results merge ``world.results`` exists; afterwards it must be gone."""
    w = world(start_time=0, stop_time=20, t_macro=10)
    w.add_signal("a", "t", step_size=10)
    ret = w.simulate(pbar=False, record_all=False)
    returns_results_object = not isinstance(ret, str) and (hasattr(ret, "to_frame") or hasattr(ret, "__getitem__"))
    if returns_results_object:
        assert not hasattr(w, "results"), "world.results() was replaced by the Results object returned by simulate()"
    else:
        assert callable(getattr(w, "results", None))

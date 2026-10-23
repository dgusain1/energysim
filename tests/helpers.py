# -*- coding: utf-8 -*-
"""Test helpers shared by tests/validation and tests/strengths.

``run_and_get`` hides the difference between the legacy results API
(``world.results(dashboard=False)``) and the 3.0 ``Results`` object returned
by ``world.simulate()``.
"""
from __future__ import annotations

import pandas as pd

from energysim import SimulatorAdapter


def _extract(res, world):
    out = {}
    for n in list(world.simulator_dict):
        try:
            df = res[n]
        except Exception:
            continue
        out[n] = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    return out


def run_and_get(world, **kw):
    """Run ``world.simulate(**kw)`` and return ``{sim_name: DataFrame}``.

    Works with the legacy API (``world.results(dashboard=False)``) and with
    the new one (``simulate()`` returns an object supporting ``res[sim]`` /
    ``to_frame``; fallback ``world.last_results``).
    """
    kw.setdefault("pbar", False)
    ret = world.simulate(**kw)
    if not isinstance(ret, str) and (hasattr(ret, "to_frame") or hasattr(ret, "__getitem__")):
        res = _extract(ret, world)
        if res:
            return res
    last = getattr(world, "last_results", None)
    if last is not None:
        res = _extract(last, world)
        if res:
            return res
    if hasattr(world, "results"):
        return world.results(dashboard=False)
    raise RuntimeError("Could not obtain results from world")


class Alg(SimulatorAdapter):
    """Stateless algebraic block: after every step, ``y = gain * u + bias`` (identity by default)."""

    def __init__(self, gain=1.0, bias=0.0):
        self.gain, self.bias = gain, bias
        self.u = 0.0
        self.y = 0.0
        self.step_size = 1.0

    def init(self):
        self.u = self.y = 0.0

    def step(self, time):
        self.y = self.gain * self.u + self.bias

    def get_value(self, parameters, time):
        return [self.y for _ in parameters]

    def set_value(self, parameters, values):
        self.u = values[0]

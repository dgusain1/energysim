# -*- coding: utf-8 -*-
"""Scenarios whose results were stored (tests/fixtures/event_baseline.json) with the
fixed-step orchestrator *before* the event-driven macro step was introduced.

``tests/test_events.py`` re-runs them and requires bit-identical results: without
events and with the default ``t_macro_min`` the variable-step loop must reproduce
the old fixed-step loop exactly.

Regenerate (only when a deliberate numerical change is made)::

    PYTHONPATH=src:tests python tests/event_baseline_scenarios.py
"""
from __future__ import annotations

import json
import os

from energysim import world
from energysim.models.storage import BatteryBESS
from energysim.models.thermal import GasTurbine_OCGT
from energysim import SimulatorAdapter

FIXTURE = os.path.join(os.path.dirname(__file__), 'fixtures', 'event_baseline.json')


class _Lag(SimulatorAdapter):
    """First-order lag ``x += (u - x) * dt / tau`` with micro step ``step_size``."""

    def __init__(self, tau=50.0):
        self.tau = tau
        self.u = 0.0
        self.x = 0.0
        self.step_size = 1.0

    def init(self):
        self.u = 0.0
        self.x = 0.0

    def step(self, time):
        self.x += (self.u - self.x) * self.step_size / self.tau

    def get_value(self, parameters, time):
        return [self.x for _ in parameters]

    def set_value(self, parameters, values):
        self.u = values[0]


def _scenario_a():
    w = world(start_time=0, stop_time=3600, t_macro=60)
    w.add_signal('cmd', signal='0.4*sin(2*pi*t/1800)', step_size=60)
    w.add_model('bess', BatteryBESS(capacity_mwh=0.2, p_max_mw=0.5), step_size=10,
                outputs=['P_actual', 'SoC', 'T_batt'])
    w.add_connections([{'from': 'cmd.y', 'to': 'bess.P_cmd', 'scale': 1.0}])
    return w, dict(record_all=False)


def _scenario_b():
    w = world(start_time=0, stop_time=2400, t_macro=30, coupling='gauss-seidel')
    w.add_signal('cmd', signal={'type': 'step', 'time': 120, 'before': 0, 'after': 80}, step_size=30)
    w.add_model('gt', GasTurbine_OCGT(p_rated_mw=100.0, hot_start_duration_sec=301.0,
                                      warm_start_duration_sec=301.0, cold_start_duration_sec=301.0),
                step_size=1, outputs=['P_actual', 'state'])
    w.add_model('lag', _Lag(tau=40.0), step_size=5, outputs=['x'])
    w.add_connections([{'from': 'cmd.y', 'to': 'gt.P_cmd', 'scale': 1.0},
                       {'from': 'gt.P_actual', 'to': 'lag.u', 'scale': 1.0}])
    return w, dict(record_all=True)


def _scenario_c():
    w = world(start_time=10, stop_time=310, t_macro=0, extrapolation='linear')
    w.add_signal('src', signal='5 + t/100', step_size=2)
    w.add_model('a', _Lag(tau=7.0), step_size=2, outputs=['x'])
    w.add_model('b', _Lag(tau=3.0), step_size=3, outputs=['x'])
    w.add_connections([{'from': ('src.y', 'b.x'), 'to': 'a.u', 'scale': 1.0},
                       {'from': 'a.x', 'to': 'b.u', 'scale': 0.5}])
    return w, dict(record_all=False)


SCENARIOS = {'a': _scenario_a, 'b': _scenario_b, 'c': _scenario_c}


def run(name):
    w, kw = SCENARIOS[name]()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = w.simulate(pbar=False, **kw)
    out = {}
    for sim in res.sims:
        df = res[sim]
        out[sim] = {'index': [repr(float(x)) for x in df.index],
                    'columns': {c: [repr(v) if not isinstance(v, str) else v for v in df[c].tolist()]
                                for c in df.columns}}
    return out


def generate():
    data = {name: run(name) for name in SCENARIOS}
    with open(FIXTURE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=0, sort_keys=True)


if __name__ == '__main__':
    generate()

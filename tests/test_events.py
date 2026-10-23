# -*- coding: utf-8 -*-
"""Event-driven / variable macro step (energysim 3.0): scheduled events, triggers,
step-size hints and partial micro-steps.  There is no rollback."""
from __future__ import annotations

import json
import math
import os
import types
import warnings

import numpy as np
import pytest

from energysim import SimulatorAdapter, world
from energysim._safeexpr import UnsafeExpressionError, build_signal, compile_condition
from energysim.models.common import ModelComponent
from energysim.models.hvac_thermal import ThermalStorageTank
from energysim.models.hydrogen import FuelCell_PEM, H2StorageTank
from energysim.models.storage import BatteryBESS, FlowBattery
from energysim.models.thermal import CombinedCycle_CCGT, GasTurbine_OCGT

import event_baseline_scenarios as baseline


class Source(SimulatorAdapter):
    """``y`` = the (unconnected) input ``level``."""

    def __init__(self, level=0.0):
        self.level0 = level
        self.level = level
        self.step_size = 1.0
        self.inputs, self.outputs = ['level'], ['y']

    def init(self):
        self.level = self.level0

    def step(self, time):
        pass

    def get_value(self, parameters, time):
        return [self.level for _ in parameters]

    def set_value(self, parameters, values):
        for p, v in zip(parameters, values):
            if p == 'level':
                self.level = v


class Follower(SimulatorAdapter):
    """``y = u`` after every step."""

    def __init__(self):
        self.u = self.y = 0.0
        self.step_size = 1.0
        self.inputs, self.outputs = ['u'], ['y']

    def init(self):
        self.u = self.y = 0.0

    def step(self, time):
        self.y = self.u

    def get_value(self, parameters, time):
        return [self.y for _ in parameters]

    def set_value(self, parameters, values):
        self.u = values[0]


class Counter(SimulatorAdapter):
    """Records (kind, t, dt) of every micro-step."""

    def __init__(self, step_size=7.0):
        self.step_size = step_size
        self.calls = []

    def step(self, time):
        self.calls.append(('full', time, self.step_size))

    def step_partial(self, time, dt):
        self.calls.append(('partial', time, dt))

    def get_value(self, parameters, time):
        return [time for _ in parameters]

    def set_value(self, parameters, values):
        pass


def _run(w, **kw):
    kw.setdefault('pbar', False)
    kw.setdefault('record_all', False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return w.simulate(**kw)


def _source_follower(**kw):
    w = world(start_time=0, stop_time=600, t_macro=60, **kw)
    w.add_model('src', Source(), step_size=60)
    w.add_model('snk', Follower(), step_size=60)
    w.add_connections([{'from': 'src.y', 'to': 'snk.u', 'scale': 1.0}])
    return w


# ---------------------------------------------------------------------------
# Baseline: no events -> bit-identical to the pre-3.0 fixed-step loop
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', sorted(baseline.SCENARIOS))
def test_no_events_run_is_bit_identical_to_fixed_step_baseline(name):
    with open(baseline.FIXTURE, encoding='utf-8') as f:
        stored = json.load(f)[name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert baseline.run(name) == stored


# ---------------------------------------------------------------------------
# Scheduled events
# ---------------------------------------------------------------------------

def test_step_lands_exactly_on_scheduled_time_and_realigns_to_grid():
    w = _source_follower()
    w.schedule(130, {'set': {'src.level': 5.0}}, name='bump')
    res = _run(w)
    t = list(res['snk'].index)
    assert t == [0, 60, 120, 130, 180, 240, 300, 360, 420, 480, 540, 600]
    ev = res.meta['events']
    assert ev == [{'time': 130.0, 'scheduled_time': 130.0, 'name': 'bump', 'kind': 'scheduled',
                   'success': True, 'error': None}]


def test_set_event_changes_downstream_value_at_the_right_time():
    w = _source_follower()
    w.schedule(130, {'set': {'src.level': 5.0}})
    y = _run(w)['snk.y']
    assert y.loc[130] == 0.0 and y.loc[180] == 5.0      # applied in the step starting at 130
    w2 = _source_follower()
    y2 = _run(w2)['snk.y']
    assert set(y2) == {0.0}


def test_callable_event_and_failed_event_are_recorded():
    seen = []
    w = _source_follower()
    w.schedule(60, lambda world_: seen.append(world_.get_adapter('src').level))
    w.schedule(240, lambda world_: 1 / 0, name='boom')
    w.schedule(600, {'set': {'src.level': 1.0}}, name='too_late')
    res = _run(w)
    assert seen == [0.0]
    by_name = {e['name']: e for e in res.meta['events']}
    assert by_name['boom']['success'] is False and 'ZeroDivisionError' in by_name['boom']['error']
    assert by_name['too_late']['success'] is None and 'not fired' in by_name['too_late']['note']


def test_event_api_validation():
    w = _source_follower()
    with pytest.raises(ValueError):
        w.schedule(10, {'call': 'os.system'})
    with pytest.raises(ValueError):
        w.schedule(10, {'set': {'nodot': 1}})
    with pytest.raises(TypeError):
        w.schedule(10, 'src.level=1')
    with pytest.raises(ValueError):
        w.schedule(float('nan'), {'set': {'src.level': 1}})
    w.schedule(10, {'set': {'ghost.x': 1}})
    rep = w.validate()
    assert not rep.is_valid and any('ghost.x' in e for e in rep.errors)


def test_yaml_events_round_trip_and_safe_mode(tmp_path):
    path = tmp_path / 'ev.yaml'
    path.write_text(
        "world: {start_time: 0, stop_time: 600, t_macro: 60}\n"
        "simulators:\n"
        "  - {name: sig, type: signal, signal: '1', step_size: 60}\n"
        "events:\n"
        "  - {time: 130, name: bump, set: {'src.level': 2.0}}\n"
        "triggers:\n"
        "  - {name: hi, condition: 'snk.y > 1', set: {'src.level': 0.0}}\n", encoding='utf-8')
    w = world.from_yaml(str(path), safe=True)
    assert w.scheduled_events == [{'time': 130.0, 'name': 'bump', 'action': {'set': {'src.level': 2.0}}}]
    w.add_model('src', Source(), step_size=60)
    w.add_model('snk', Follower(), step_size=60)
    w.add_connections([{'from': 'src.y', 'to': 'snk.u', 'scale': 1.0}])
    res = _run(w)
    y = res['snk.y']
    assert y.loc[180] == 2.0
    kinds = [(e['name'], e['kind'], e['time']) for e in res.meta['events']]
    assert kinds == [('bump', 'scheduled', 130.0), ('hi', 'trigger', 180.0)]
    assert y.loc[240] == 0.0                               # trigger action reset the level at 180
    d = w.to_dict()
    assert d['events'] == [{'time': 130.0, 'name': 'bump', 'set': {'src.level': 2.0}}]
    assert d['triggers'][0]['condition'] == 'snk.y > 1'
    for bad in ("events:\n  - {time: 1, call: os.system}\n",
                "events:\n  - {time: 1, set: {'a.b': 1}, exec: x}\n",
                "triggers:\n  - {name: t, condition: '__import__(1)', set: {'a.b': 1}}\n"):
        with pytest.raises(ValueError):
            world.from_yaml("world: {stop_time: 60}\nsimulators: []\n" + bad, safe=True)


# ---------------------------------------------------------------------------
# Variable macro step and hints
# ---------------------------------------------------------------------------

def test_t_macro_min_shrinks_and_grows_within_bounds():
    w = world(start_time=0, stop_time=600, t_macro=60, t_macro_min=10)
    w.add_signal('sig', signal='t', step_size=1, breakpoints=[130, 135])
    w.add_model('snk', Follower(), step_size=1)
    w.add_connections([{'from': 'sig.y', 'to': 'snk.u', 'scale': 1.0}])
    res = _run(w)
    t = np.asarray(res['snk'].index, dtype=float)
    assert list(t[:6]) == [0, 60, 120, 130, 140, 200]
    steps = np.diff(t)
    assert steps.min() >= 10 - 1e-9 and steps.max() <= 60 + 1e-9
    assert len(set(np.round(steps, 9))) > 1                 # irregular rows
    assert res.meta['t_macro_min'] == 10 and res.meta['n_macro_steps'] == len(t) - 1
    assert res.to_frame().index.is_monotonic_increasing


def test_default_t_macro_min_ignores_hints():
    w = world(start_time=0, stop_time=600, t_macro=60)
    w.add_signal('sig', signal={'type': 'step', 'time': 130, 'after': 1}, step_size=60)
    res = _run(w)
    assert list(res['sig'].index) == list(range(0, 601, 60))


def test_variable_step_allows_span_not_multiple_of_t_macro():
    w = world(start_time=0, stop_time=130, t_macro=60, t_macro_min=1)
    w.add_model('c', Counter(step_size=7.0), step_size=7.0)
    res = _run(w)
    assert list(res['c'].index) == [0, 60, 120, 130]
    calls = w.get_adapter('c').calls
    assert sum(dt for _, _, dt in calls) == pytest.approx(130.0)
    with pytest.raises(ValueError, match='integer multiple'):
        w2 = world(start_time=0, stop_time=130, t_macro=60)
        w2.add_model('c', Counter(step_size=10.0), step_size=10.0)
        w2.simulate(pbar=False, validate=False)
    with pytest.raises(ValueError):
        world(t_macro_min=0)


def test_signal_spec_breakpoints_are_hints():
    w = world(start_time=0, stop_time=300, t_macro=100, t_macro_min=1)
    w.add_signal('s', signal={'type': 'piecewise', 'breakpoints': [[0, 0], [42, 1], [250, 2]],
                              'interpolation': 'previous'}, step_size=1)
    res = _run(w)
    assert list(res['s'].index) == [0, 42, 142, 242, 250, 300]
    assert build_signal({'type': 'ramp', 'start_time': 5, 'end_time': 9, 'end_value': 1}).breakpoints == [5.0, 9.0]


def test_csv_align_to_rows_hint(tmp_path):
    p = tmp_path / 'p.csv'
    p.write_text('time,x\n0,1\n45,2\n90,3\n135,4\n', encoding='utf-8')
    w = world(start_time=0, stop_time=120, t_macro=60, t_macro_min=1)
    w.add_csv('c', str(p), step_size=45, outputs=['x'], align_to_rows=True)
    res = _run(w)
    assert list(res['c'].index) == [0, 45, 90, 120]
    assert w.to_dict()['simulators'][0]['options'] == {'align_to_rows': True}
    w2 = world(start_time=0, stop_time=120, t_macro=60, t_macro_min=1)
    w2.add_csv('c', str(p), step_size=45, outputs=['x'])
    assert list(_run(w2)['c'].index) == [0, 60, 120]


def test_fmu_me_next_event_time_hint():
    pytest.importorskip('fmpy')
    from energysim.meAdapter import FmuMeAdapter
    ns = types.SimpleNamespace(_next_event_defined=True, _next_event_time=12.5)
    assert FmuMeAdapter.next_event_time(ns, 10.0) == 12.5
    assert FmuMeAdapter.next_event_time(ns, 12.5) == math.inf
    ns._next_event_defined = False
    assert FmuMeAdapter.next_event_time(ns, 0.0) == math.inf


def test_tank_full_hint_lands_within_one_micro_step():
    def build(**kw):
        w = world(start_time=0, stop_time=3600, t_macro=600, **kw)
        w.add_signal('flow', signal=100.0, step_size=600)          # kg/h
        tank = H2StorageTank(volume_m3=10.0, p_init_bar=300.0)
        w.add_model('tank', tank, step_size=1, outputs=['mass_stored_kg', 'm_dot_rejected_kg_hr'])
        w.add_connections([{'from': 'flow.y', 'to': 'tank.m_dot_in_kg_hr', 'scale': 1.0}])
        return w, tank
    w, tank = build(t_macro_min=1)
    t_true = (tank.mass_capacity_kg - tank.mass_stored_kg) / 100.0 * 3600.0
    res = _run(w)
    m = res['tank.mass_stored_kg']
    full = m.index[m >= tank.mass_capacity_kg - 1e-9]
    err_hint = abs(full[0] - t_true)
    assert err_hint <= 1.0                                        # one micro-step
    w2, _ = build()
    m2 = _run(w2)['tank.mass_stored_kg']
    full2 = m2.index[m2 >= tank.mass_capacity_kg - 1e-9]
    assert abs(full2[0] - t_true) > err_hint and abs(full2[0] - t_true) > 1.0


def test_ocgt_start_up_completion_lands_exactly():
    def build(**kw):
        w = world(start_time=0, stop_time=900, t_macro=60, **kw)
        w.add_signal('cmd', signal={'type': 'step', 'time': 120, 'after': 80}, step_size=60)
        w.add_model('gt', GasTurbine_OCGT(hot_start_duration_sec=301.0, warm_start_duration_sec=301.0,
                                          cold_start_duration_sec=301.0), step_size=1,
                    outputs=['P_actual', 'state'])
        w.add_connections([{'from': 'cmd.y', 'to': 'gt.P_cmd', 'scale': 1.0}])
        return w
    st = _run(build(t_macro_min=1))['gt.state']
    assert st.loc[420] == 'STARTING' and st.loc[421] == 'RUNNING'
    st2 = _run(build())['gt.state']
    assert 421 not in st2.index and st2.loc[420] == 'STARTING' and st2.loc[480] == 'RUNNING'


# ---------------------------------------------------------------------------
# Triggers
# ---------------------------------------------------------------------------

def test_trigger_fires_after_crossing_and_is_recorded():
    w = world(start_time=0, stop_time=600, t_macro=60)
    w.add_signal('ramp', signal='t', step_size=60)
    w.add_model('src', Source(), step_size=60)
    w.add_model('snk', Follower(), step_size=60)
    w.add_connections([{'from': 'src.y', 'to': 'snk.u', 'scale': 1.0}])
    w.add_trigger('over', 'ramp.y > 250', {'set': {'src.level': 7.0}})
    w.add_trigger('both', lambda v: v['ramp.y'] > 400, lambda ww: None, direction='both',
                  variables=['ramp.y'])
    w.add_trigger('fall', 'ramp.y < 100', {'set': {'src.level': -1.0}}, direction='falling')
    res = _run(w)
    ev = {e['name']: e for e in res.meta['events']}
    assert ev['over']['detected_time'] == 300.0 and ev['over']['time'] == 300.0
    assert ev['over']['interval'] == [240.0, 300.0] and 'accuracy limited to the step' in ev['over']['note']
    assert ev['both']['detected_time'] == 420.0
    assert ev['fall']['detected_time'] == 120.0
    y = res['snk.y']
    assert y.loc[300] == -1.0 and y.loc[360] == 7.0


def test_trigger_api_validation():
    w = _source_follower()
    with pytest.raises(NotImplementedError, match='rollback'):
        w.add_trigger('x', 'src.y > 1', {'set': {'src.level': 0}}, bisect=True)
    with pytest.raises(ValueError):
        w.add_trigger('x', 'src.y > 1', {'set': {'src.level': 0}}, direction='up')
    with pytest.raises(ValueError):
        w.add_trigger('x', lambda v: True, {'set': {'src.level': 0}})
    with pytest.raises(UnsafeExpressionError):
        w.add_trigger('x', '__import__("os").system("x")', {'set': {'src.level': 0}})
    w.add_trigger('x', 'src.y > 1', {'set': {'src.level': 0}})
    with pytest.raises(ValueError):
        w.add_trigger('x', 'src.y > 2', {'set': {'src.level': 0}})


def test_compile_condition_dotted_and_quoted_names():
    c = compile_condition("grid.Bus1.vm_pu < 0.95 and 'fmu.der(x)' >= 0 or t > 100")
    assert c.variables == ['grid.Bus1.vm_pu', 'fmu.der(x)']
    assert c({'grid.Bus1.vm_pu': 0.9, 'fmu.der(x)': 1.0}) is True
    assert c({'grid.Bus1.vm_pu': 0.99, 'fmu.der(x)': 1.0}, t=0) is False
    assert c({'grid.Bus1.vm_pu': 0.99, 'fmu.der(x)': 1.0}, t=101) is True
    with pytest.raises(KeyError):
        c({})
    for bad in ('a.b()', '(1).real', '[a.b, 1]', 'lambda: 1'):
        with pytest.raises(UnsafeExpressionError):
            compile_condition(bad)


# ---------------------------------------------------------------------------
# Partial micro-steps (macro interval not a multiple of step_size)
# ---------------------------------------------------------------------------

def test_base_advance_does_shorter_final_micro_step():
    c = Counter(step_size=7.0)
    c.advance(0.0, 30.0)
    assert c.calls == [('full', 0.0, 7.0), ('full', 7.0, 7.0), ('full', 14.0, 7.0), ('full', 21.0, 7.0),
                       ('partial', 28.0, 2.0)]
    c2 = Counter(step_size=7.0)
    rows = c2.advance_with_recording(0.0, 10.0, ['t'])
    assert [r[0] for r in rows] == [0.0, 7.0] and c2.calls[-1] == ('partial', 7.0, 3.0)
    c3 = Counter(step_size=7.0)
    c3.advance(0.0, 21.0)
    assert [k for k, _, _ in c3.calls] == ['full'] * 3
    c4 = Counter(step_size=7.0)
    c4.advance(0.0, 4.0)
    assert c4.calls == [('partial', 0.0, 4.0)]
    assert SimulatorAdapter.step_partial.__doc__ and 'step' in SimulatorAdapter.step_partial.__doc__


class Integrator(ModelComponent):
    def __init__(self, step_size=7.0):
        super().__init__(name='int', step_size=step_size)
        self.x = 0.0
        self.register_variables(inputs=['u'], outputs=['x'], defaults={'u': 1.0})

    def step_model(self, dt, time):
        self.x += self._inputs_val['u'] * dt
        self._outputs_val['x'] = self.x


def test_model_component_partial_steps_integrate_exactly():
    m = Integrator()
    m.init()
    rows = m.advance_with_recording(0.0, 30.0, ['x'])
    assert m.x == pytest.approx(30.0) and len(rows) == 5
    w = world(start_time=0, stop_time=100, t_macro=20)
    w.add_model('i', Integrator(step_size=7.0), step_size=7.0)
    w.schedule(33, {'set': {'i.u': 2.0}})
    res = _run(w, record_all=True)
    x = res['i.x']
    assert x.iloc[-1] == pytest.approx(33.0 + 2 * 67.0)


def test_container_partial_step_uses_dt():
    from energysim import BatteryFleetAdapter
    a = BatteryFleetAdapter(name='f', size=2, step_size=60.0)
    seen = []
    a.step = lambda time, step_size=None: seen.append((time, step_size))
    a.advance(0.0, 150.0)
    assert seen == [(0.0, None), (60.0, None), (120.0, 30.0)] or seen[-1] == (120.0, 30.0)


# ---------------------------------------------------------------------------
# Model event hints
# ---------------------------------------------------------------------------

def test_model_time_to_event_hints():
    gt = GasTurbine_OCGT(hot_start_duration_sec=300, warm_start_duration_sec=300, cold_start_duration_sec=300,
                         min_uptime_sec=600)
    gt.init()
    assert gt.time_to_event() == math.inf
    gt.set_value(['P_cmd'], [80.0])
    gt.advance(0, 100)
    assert gt.time_to_event() == pytest.approx(200.0) and gt.next_event_time(100) == pytest.approx(300.0)
    gt.advance(100, 300)
    assert gt.get_value(['state'], 300)[0] == 'RUNNING'
    assert gt.time_to_event() == pytest.approx(600.0)

    cc = CombinedCycle_CCGT(min_downtime_sec=900)
    cc.init()
    cc.sm.time_offline_sec = 100.0
    assert cc.time_to_event() == pytest.approx(800.0)

    b = BatteryBESS(capacity_mwh=1.0, p_max_mw=0.5, soc_init=0.5, soc_max=0.9)
    b.init()
    assert b.time_to_event() == math.inf
    b.set_value(['P_cmd'], [0.5])
    tte = b.time_to_event()
    assert 0.4 * 3600 / 0.5 * 0.9 < tte < 0.4 * 3600 / 0.5 * 1.1
    b.set_value(['P_cmd'], [-0.5])
    assert b.time_to_event() > 0

    fb = FlowBattery(stack_p_max_mw=1.0, tank_capacity_mwh=2.0, soc_init=0.5)
    fb.init()
    fb.set_value(['P_cmd'], [-1.0])
    assert fb.time_to_event() == pytest.approx((0.5 - 0.02) * 2.0 * 3600 * math.sqrt(0.85))

    h = H2StorageTank()
    h.init()
    h.set_value(['m_dot_out_kg_hr'], [10.0])
    assert h.time_to_event() == pytest.approx(h.mass_stored_kg / 10.0 * 3600.0)

    th = ThermalStorageTank(capacity_mwh_th=10.0, loss_pct_per_day=0.0, soc_init=0.5)
    th.init()
    th.set_value(['Q_cmd_MW'], [5.0])
    assert th.time_to_event() == pytest.approx(0.5 * 10.0 * 3600 / 5.0)

    fc = FuelCell_PEM(p_nominal_mw=1.0, ramp_rate_mw_min=2.0)
    fc.init()
    fc.set_value(['P_cmd'], [1.0])
    assert fc.time_to_event() == pytest.approx(30.0)
    fc.advance(0, 30)
    assert fc.time_to_event() == math.inf


def test_model_component_default_hint_is_infinite():
    m = Integrator()
    m.init()
    assert m.time_to_event() == math.inf and m.next_event_time(5.0) == math.inf
    assert SimulatorAdapter.max_step(m, 0.0) == math.inf


def test_baseline_fixture_exists():
    assert os.path.isfile(baseline.FIXTURE)

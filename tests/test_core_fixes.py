# -*- coding: utf-8 -*-
"""Tests for the 3.0 core fixes (world / base / utils / cli / dashboard / adapters)."""

import json
import math
import os
import sys
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from energysim import world
from energysim.base import SimulatorAdapter, SimEntry


class _Adapter(SimulatorAdapter):
    """Minimal recording adapter."""

    def __init__(self, inputs=None, outputs=None):
        self.inputs = list(inputs or [])
        self.outputs = list(outputs or ['y'])
        self.values = {}
        self.events = []

    def configure(self, start_time, stop_time, macro_step):
        self.events.append(('configure', start_time, stop_time, macro_step))

    def init(self):
        self.events.append(('init',))

    def cleanup(self):
        self.events.append(('cleanup',))

    def step(self, time):
        self.values['y'] = self.values.get('u', 0.0) + time

    def get_value(self, parameters, time):
        return [self.values.get(p, 0.0) for p in parameters]

    def set_value(self, parameters, values):
        for p, v in zip(parameters, values):
            self.values[p] = v
            self.events.append(('set', p, v))


@pytest.fixture
def csv_file(tmp_path):
    p = tmp_path / 'data.csv'
    p.write_text('time,temp,p_kw\n0,20,1\n10,21,2\n20,22,3\n30,23,4\n')
    return str(p)


# =====================================================================
# Task 1: rollback / iterative coupling removed
# =====================================================================

def test_iterative_coupling_rejected():
    with pytest.raises(ValueError, match="iterative coupling was removed in 3.0"):
        world(coupling='iterative')
    with pytest.raises(ValueError, match="coupling"):
        world(coupling='nonsense')


def test_iterative_params_removed():
    with pytest.raises(TypeError):
        world(max_iterations=5)
    with pytest.raises(TypeError):
        world(convergence_tolerance=1e-3)
    assert not hasattr(SimulatorAdapter, 'save_state')
    assert not hasattr(SimulatorAdapter, 'restore_state')


def test_iterative_yaml_keys_rejected():
    with pytest.raises(ValueError, match="iterative coupling was removed"):
        world.from_dict({'world': {'coupling': 'iterative'}, 'simulators': []})
    with pytest.raises(ValueError, match="no longer supported"):
        world.from_dict({'world': {'max_iterations': 3}, 'simulators': []})


def test_algebraic_loop_warning_advice():
    for coupling in ('jacobi', 'gauss-seidel'):
        w = world(start_time=0, stop_time=20, t_macro=10, coupling=coupling)
        w.simulator_dict['a'] = SimEntry('test', _Adapter(['u']), 10, ['y'], ['u'])
        w.simulator_dict['b'] = SimEntry('test', _Adapter(['u']), 10, ['y'], ['u'])
        with pytest.warns(UserWarning):
            w.add_connections({'a.y': 'b.u', 'b.y': 'a.u'})
        report = w.validate()
        msg = [x for x in report.warnings if 'Algebraic feedback loop' in x]
        assert msg and 'reduce t_macro' in msg[0] and 'merge' in msg[0]
        assert 'iterative' not in msg[0]


# =====================================================================
# Task 2: add_simulator for every sim type
# =====================================================================

def test_add_simulator_battery_fleet_types():
    w = world(start_time=0, stop_time=1800, t_macro=900)
    w.add_simulator('battery_fleet', 'bf', size=5, step_size=900, capacity_kwh=10.0)
    w.add_simulator('fleet', 'bf2', kind='battery', size=3, step_size=900)
    w.add_simulator('container', 'bf3', fleet_type='battery', size=2, step_size=900)
    w.add_simulator('building_fleet', 'bld', size=4, step_size=900)
    w.add_simulator('pv_fleet', 'pv', size=2, step_size=900)
    assert w.simulator_dict['bf'].adapter.size == 5
    assert w.simulator_dict['bf2'].adapter.size == 3
    assert w.simulator_dict['bld'].sim_type == 'container'
    w.simulate(pbar=False)
    assert 'bf' in w.last_results


def test_add_simulator_function_container_and_instance():
    from energysim.container import BatteryFleetAdapter

    def upd(time, states, inputs, step_size):
        return {'x': states['x'] + 1.0}

    w = world(start_time=0, stop_time=20, t_macro=10)
    w.add_simulator('vectorized', 'fn', size=3, step_size=10, update_func=upd,
                    initial_states={'x': 0.0}, kind='function')
    inst = BatteryFleetAdapter(name='x', size=2, step_size=10)
    w.add_simulator('container', 'inst', model=inst, step_size=10)
    assert w.get_adapter('inst') is inst
    w.simulate(pbar=False)
    res = w.last_results
    assert res['fn']['mean_x'].tolist() == [0.0, 1.0, 2.0]   # + final row at stop_time


def test_add_simulator_container_without_kind_raises():
    w = world()
    with pytest.raises(ValueError, match="kind"):
        w.add_simulator('container', 'c', size=3)


def test_add_simulator_model_csv_signal_script_external(tmp_path, csv_file):
    script = tmp_path / 'ctrl.py'
    script.write_text('def run(inputs):\n    return [inputs.get("u", 0) * 2]\n')
    ext = tmp_path / 'ext.py'
    ext.write_text(textwrap.dedent('''
        from energysim.base import SimulatorAdapter
        class Ext(SimulatorAdapter):
            def __init__(self, outputs=None):
                self.outputs = outputs or []
                self.v = 7.0
            def step(self, time): pass
            def get_value(self, parameters, time): return [self.v for _ in parameters]
            def set_value(self, parameters, values): pass
    '''))
    w = world(start_time=0, stop_time=20, t_macro=10)
    w.add_simulator('model', 'm', model=_Adapter(), step_size=10)
    w.add_simulator('csv', 'c', model=csv_file, outputs=['temp'], step_size=10)
    w.add_simulator('signal', 's', signal='2*t', step_size=10)
    w.add_simulator('script', 'sc', model=str(script), function='run', inputs=['u'], outputs=['out'],
                    step_size=10)
    w.add_simulator('external', 'ex', model=f'{ext}:Ext', outputs=['v'], step_size=10)
    w.add_connections({'s.y': 'sc.u'})
    w.simulate(pbar=False, record_all=False)
    res = w.last_results
    assert res['s']['y'].tolist() == [0.0, 20.0, 40.0]
    assert res['ex']['v'].tolist() == [7.0, 7.0, 7.0]


def test_add_simulator_alias_normalisation(csv_file):
    w = world(start_time=0, stop_time=20, t_macro=10)
    w.add_simulator('csv', 'a', csv_path=csv_file, outputs=['temp'])
    w.add_simulator('csv', 'b', sim_loc=csv_file, model=csv_file, outputs=['temp'])
    with pytest.raises(ValueError, match="Conflicting model references"):
        w.add_simulator('csv', 'c', sim_loc=csv_file, csv_path='other.csv', outputs=['temp'])
    with pytest.raises(ValueError, match="not recognized"):
        w.add_simulator('bogus', 'd', model=csv_file)


# =====================================================================
# Task 3: LCM and total steps
# =====================================================================

def test_get_lcm_scientific_notation():
    w = world(t_macro=0)
    w.simulator_dict['a'] = SimEntry('test', _Adapter(), 1e-5, ['y'])
    w.simulator_dict['b'] = SimEntry('test', _Adapter(), 2e-5, ['y'])
    w.simulator_dict['c'] = SimEntry('test', _Adapter(), 2, ['y'])
    assert math.isclose(w.get_lcm(), 2.0)
    w2 = world()
    w2.simulator_dict['a'] = SimEntry('test', _Adapter(), 0.1, ['y'])
    w2.simulator_dict['b'] = SimEntry('test', _Adapter(), 0.25, ['y'])
    assert math.isclose(w2.get_lcm(), 0.5)


def test_total_steps_float_tolerance(tmp_path):
    w = world(start_time=0, stop_time=0.3, t_macro=0.1)
    w.add_signal('s', 't', step_size=0.1)
    w.simulate(pbar=False, record_all=False)
    assert len(w.last_results['s']) == 4   # 3 steps (int(0.3/0.1) would give 2) + final row


def test_span_not_multiple_of_macro_raises(tmp_path):
    w = world(start_time=0, stop_time=100, t_macro=60)
    w.add_signal('s', 't', step_size=60)
    rep = w.validate()
    assert not rep.is_valid and any('integer multiple' in e for e in rep.errors)
    with pytest.raises(ValueError, match="integer multiple"):
        w.simulate(pbar=False, validate=False)


# =====================================================================
# Task 4: parameter application order + configure()
# =====================================================================

class _BeforeAdapter(_Adapter):
    SET_PARAMS_BEFORE_INIT = True


def test_param_order_and_configure(tmp_path):
    assert SimulatorAdapter.SET_PARAMS_BEFORE_INIT is False
    w = world(start_time=0, stop_time=20, t_macro=10)
    after, before = _Adapter(), _BeforeAdapter()
    w.add_model('after', model=after, step_size=10)
    w.add_model('before', model=before, step_size=10)
    w.options({'init': {'after': (['k'], [1.0]), 'before': {'k': 2.0}}})
    w.set_parameters({'after': {'j': 3.0}})
    assert ('set', 'j', 3.0) not in after.events   # deferred to the run
    w.simulate(pbar=False)
    ev_a = [e[0] if e[0] != 'set' else e[1] for e in after.events]
    ev_b = [e[0] if e[0] != 'set' else e[1] for e in before.events]
    assert ev_a[:4] == ['configure', 'init', 'k', 'j']
    assert ev_b[:3] == ['configure', 'k', 'init']
    assert after.events[0] == ('configure', 0, 20, 10)


def test_unknown_param_sim_raises():
    w = world()
    w.add_model('m', model=_Adapter(), step_size=10)
    with pytest.raises(ValueError, match="unknown simulator"):
        w.options({'init': {'nope': (['a'], [1])}})


# =====================================================================
# Task 5: cleanup always, init errors, re-runnable, reset
# =====================================================================

class _Boom(_Adapter):
    def step(self, time):
        raise RuntimeError("boom")


class _BadCleanup(_Adapter):
    def cleanup(self):
        super().cleanup()
        raise RuntimeError("cleanup failed")


class _BadInit(_Adapter):
    def init(self):
        raise OSError("cannot start")


def test_cleanup_runs_on_error(tmp_path):
    w = world(start_time=0, stop_time=20, t_macro=10)
    ok, bad_cleanup, boom = _Adapter(), _BadCleanup(), _Boom()
    w.add_model('ok', model=ok, step_size=10)
    w.add_model('bc', model=bad_cleanup, step_size=10)
    w.add_model('boom', model=boom, step_size=10)
    with pytest.raises(RuntimeError, match="boom"):
        w.simulate(pbar=False, record_all=False)
    for a in (ok, bad_cleanup, boom):
        assert ('cleanup',) in a.events


def test_init_failure_names_simulator(tmp_path):
    w = world(start_time=0, stop_time=20, t_macro=10)
    good = _Adapter()
    w.add_model('good', model=good, step_size=10)
    w.add_model('broken', model=_BadInit(), step_size=10)
    with pytest.raises(RuntimeError, match="broken") as ei:
        w.simulate(pbar=False)
    assert isinstance(ei.value.__cause__, OSError)
    assert ('cleanup',) in good.events


def test_world_rerunnable(tmp_path, csv_file):
    script = tmp_path / 'acc.py'
    script.write_text('STATE = [0]\n'
                      'def run(inputs):\n'
                      '    STATE[0] += inputs.get("u", 0)\n'
                      '    return [STATE[0]]\n')
    w = world(start_time=0, stop_time=30, t_macro=10)
    w.add_csv('c', csv_file, step_size=10, outputs=['temp'])
    w.add_signal('s', {'type': 'ramp', 'start_time': 0, 'end_time': 30, 'start_value': 0, 'end_value': 3},
                 step_size=10)
    w.add_script('sc', script_loc=str(script), function='run', inputs=['u'], outputs=['acc'], step_size=10)
    w.add_connections({'s.y': 'sc.u'})
    w.simulate(pbar=False, record_all=False)
    r1 = {k: v.copy() for k, v in w.last_results.items()}
    w.simulate(pbar=False, record_all=False)
    r2 = w.last_results
    for k in r1:
        assert r1[k].equals(r2[k]), k
    w.reset()
    assert w.last_results is None


# =====================================================================
# Task 6: safe expressions / signals / safe mode
# =====================================================================

def test_safe_expression_signals():
    from energysim._safeexpr import compile_expression, build_signal
    assert compile_expression('lambda t: 2*t + 1')(3) == [7]
    assert compile_expression('max(0, sin(t)) if t > pi else -1')(0) == [-1]
    assert compile_expression('[t, t % 4]')(5) == [5, 1]
    assert build_signal({'type': 'constant', 'value': 5})(10) == [5.0]
    assert build_signal({'type': 'step', 'time': 10, 'before': 1, 'after': 2})(9) == [1.0]
    assert build_signal({'type': 'step', 'time': 10, 'before': 1, 'after': 2})(10) == [2.0]
    assert math.isclose(build_signal({'type': 'sine', 'amplitude': 2, 'period': 40})(10)[0], 2.0)
    assert build_signal({'type': 'ramp', 'start_time': 0, 'end_time': 10,
                         'start_value': 0, 'end_value': 10})(20) == [10.0]
    pw = build_signal({'type': 'piecewise', 'breakpoints': [[0, 0], [10, 10], [20, 0]]})
    assert pw(5) == [5.0] and pw(15) == [5.0] and pw(99) == [0.0]
    pw2 = build_signal({'type': 'piecewise', 'breakpoints': [[0, 1], [10, 2]], 'interpolation': 'previous'})
    assert pw2(9) == [1.0]
    with pytest.raises(ValueError):
        build_signal({'type': 'magic'})


@pytest.mark.parametrize('payload', [
    "__import__('os').system('echo pwned > pwned.txt')",
    "().__class__.__bases__[0].__subclasses__()",
    "open('x', 'w')",
    "lambda t: __import__('os')",
    "t.__class__",
    "globals()",
    "[t for t in (1,)]",
    "(lambda: 1)()",
    "'abc'",
    "[t] * 100000000",
    "exec('1')",
])
def test_hostile_signal_strings_rejected(payload, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    called = []
    monkeypatch.setattr(os, 'system', lambda *a: called.append(a) or 0)
    w = world()
    with pytest.raises(ValueError):
        w.add_signal('s', payload)
    assert not called
    assert not (tmp_path / 'pwned.txt').exists()
    assert not (tmp_path / 'x').exists()


def test_yaml_signal_specs_and_roundtrip(tmp_path):
    yml = """
world: {start_time: 0, stop_time: 20, t_macro: 10}
simulators:
  - {name: a, type: signal, step_size: 10, signal: {type: constant, value: 5}}
  - {name: b, type: signal, step_size: 10, options: {signal: "lambda t: t*2"}}
  - {name: c, type: signal, step_size: 10, model: "t + 1"}
"""
    w = world.from_yaml(yml)
    w.simulate(pbar=False, record_all=False)
    res = w.last_results
    assert res['a']['y'].tolist() == [5.0, 5.0, 5.0]
    assert res['b']['y'].tolist() == [0.0, 20.0, 40.0]
    assert res['c']['y'].tolist() == [1.0, 11.0, 21.0]
    w2 = world.from_yaml(w.to_yaml())
    assert w2.simulator_dict['a'].adapter.get_value(['y'], 3) == [5.0]
    assert w2.simulator_dict['b'].adapter.get_value(['y'], 3) == [6]


def test_safe_mode_rejects_code_execution(tmp_path):
    base = "world: {start_time: 0, stop_time: 10, t_macro: 10}\nsimulators:\n"
    for sim in ("  - {name: x, type: external, model: some_dir}\n",
                "  - {name: x, type: script, model: ctrl.py}\n",
                "  - {name: x, type: matlab, model: f.m}\n",
                "  - {name: x, type: powerfactory, model: proj}\n",
                "  - {name: x, type: powerflow, model: net.p}\n",
                "  - {name: x, type: pypsa, model: grid.pkl}\n",
                "  - {name: x, type: powerflow, model: pandapower.networks.case14}\n"):
        with pytest.raises(ValueError, match="safe mode"):
            world.from_yaml(base + sim, safe=True)


def test_dotted_import_requires_opt_in():
    yml = ("world: {start_time: 0, stop_time: 10, t_macro: 10}\nsimulators:\n"
           "  - {name: x, type: powerflow, model: os.getcwd}\n")
    with pytest.raises(ValueError, match="allow_import"):
        world.from_yaml(yml)
    from energysim.utils import resolve_model_reference
    with pytest.raises(ValueError, match="allow_import"):
        resolve_model_reference('os.getcwd')
    assert resolve_model_reference('os.getcwd', allow_import=True) == os.getcwd()
    obj = object()
    assert resolve_model_reference(obj) is obj
    assert resolve_model_reference('IEEE13') == 'IEEE13'
    assert resolve_model_reference('grid.json') == 'grid.json'


def test_cli_safe_flag(tmp_path):
    from energysim.cli import main as cli_main
    f = tmp_path / 's.yaml'
    f.write_text("world: {start_time: 0, stop_time: 10, t_macro: 10}\nsimulators:\n"
                 "  - {name: x, type: script, model: ctrl.py}\n")
    assert cli_main(['validate', str(f), '--safe']) == 1
    assert cli_main(['run', str(f), '--safe', '--validate-only']) == 1
    ok = tmp_path / 'ok.yaml'
    ok.write_text("world: {start_time: 0, stop_time: 10, t_macro: 10}\nsimulators:\n"
                  "  - {name: s, type: signal, step_size: 10, signal: '3'}\n")
    assert cli_main(['validate', str(ok), '--safe']) == 0
    assert cli_main(['run', str(ok), '--safe', '--no-pbar']) == 0


# =====================================================================
# Task 7: dashboard / filename security
# =====================================================================

HOSTILE = '</script><script>alert(1)</script>'


def test_dashboard_escapes_hostile_names():
    import pandas as pd
    from energysim.dashboard import _generate_html, _build_traces_json
    df = pd.DataFrame({'time': [0, 1], HOSTILE: [1.0, 2.0], 'q"\'<b>': [3.0, float('nan')]})
    html_text = _generate_html(_build_traces_json({HOSTILE: df}), title=HOSTILE + '"\'',
                               time_label=HOSTILE)
    assert HOSTILE not in html_text
    assert '<script>alert(1)' not in html_text
    assert html_text.count('</script>') == 2  # plotly loader + main block only
    # the embedded JSON still decodes to the original names
    start = html_text.index('const SIMS = ') + len('const SIMS = ')
    end = html_text.index(';\n', start)
    sims = json.loads(html_text[start:end])
    assert sims[0]['name'] == HOSTILE
    assert HOSTILE in sims[0]['variables']
    assert ' ' not in json.dumps(sims)  # sanity
    # names are inserted via textContent, not innerHTML concatenation
    assert "'<span class=\"name\">'+sim.name" not in html_text
    assert 'textContent' in html_text


def test_dashboard_json_line_separators():
    from energysim.dashboard import _json_for_script
    s = _json_for_script({'a': 'x y <&>'})
    assert ' ' not in s and ' ' not in s and '<' not in s and '&' not in s
    assert json.loads(s) == {'a': 'x y <&>'}


def test_export_csv_sanitises_names(tmp_path):
    import pandas as pd
    from energysim.results import Results
    from energysim.utils import safe_filename_component
    res = Results({'../../evil': pd.DataFrame({'v': [0.0]}, index=pd.Index([0.0], name='time')),
                   'a/b\\c:d': pd.DataFrame({'v': [0.0]}, index=pd.Index([0.0], name='time'))})
    res.to_csv(tmp_path / 'out')
    files = sorted(f for f in os.listdir(tmp_path / 'out') if f != 'meta.json')
    assert len(files) == 2
    for f in files:
        assert '/' not in f and '\\' not in f and '..' not in f and ':' not in f
    assert not (tmp_path.parent / 'evil.csv').exists()
    assert safe_filename_component('..') == '_'
    back = Results.load(tmp_path / 'out')
    assert sorted(back.sims) == sorted(res.sims)


# =====================================================================
# Task 8: units
# =====================================================================

def test_unit_table_and_strictness():
    from energysim.utils import get_unit_scale, get_unit_conversion
    assert math.isclose(get_unit_scale('GW', 'MW'), 1e3)
    assert math.isclose(get_unit_scale('Mvar', 'kvar'), 1e3)
    assert math.isclose(get_unit_scale('mV', 'V'), 1e-3)
    assert math.isclose(get_unit_scale('MV', 'kV'), 1e3)
    assert math.isclose(get_unit_scale('kWh', 'MJ'), 3.6)
    assert math.isclose(get_unit_scale('GWh', 'MWh'), 1e3)
    assert math.isclose(get_unit_scale('t/h', 'kg/s'), 1000 / 3600)
    assert math.isclose(get_unit_scale('kg/h', 'kg/s'), 1 / 3600)
    assert math.isclose(get_unit_scale('bar', 'kPa'), 100)
    assert math.isclose(get_unit_scale('MPa', 'bar'), 10)
    assert math.isclose(get_unit_scale('h', 'min'), 60)
    assert get_unit_scale('mw', 'MW') == 1.0          # unambiguous case-insensitive
    sc, off = get_unit_conversion('degC', 'K')
    assert sc == 1.0 and math.isclose(off, 273.15)
    sc, off = get_unit_conversion('K', 'degC')
    assert math.isclose(300 * sc + off, 26.85)
    for a, b in (('furlong', 'kW'), ('kW', 'parsec')):
        with pytest.raises(ValueError, match="Unknown unit"):
            get_unit_scale(a, b)
    with pytest.raises(ValueError, match="Ambiguous"):
        get_unit_scale('mv', 'V')
    for a, b in (('kg/s', 'Nm3/h'), ('m3/h', 'Nm3/h'), ('pu', 'kV'), ('kW', 'kWh'), ('degC', 'kW')):
        with pytest.raises(ValueError, match="Incompatible"):
            get_unit_scale(a, b)
    with pytest.raises(ValueError, match="offset"):
        get_unit_scale('degC', 'K')
    assert get_unit_scale(None, 'kW') == 1.0


def test_infer_variable_unit_rules():
    from energysim.utils import infer_variable_unit
    assert infer_variable_unit('Load.L1.p_mw') == 'MW'
    assert infer_variable_unit('battery.e_kwh') == 'kWh'
    assert infer_variable_unit('t_in_degc') == 'degC'
    assert infer_variable_unit('T_room', sim_type='fmu', default_unit='W') is None
    assert infer_variable_unit('y', default_unit='kW') is None
    assert infer_variable_unit('P_cmd', default_unit='kW') == 'kW'
    assert infer_variable_unit('Generator.g1.p_set', sim_type='pypsa') == 'MW'
    assert infer_variable_unit('power', sim_type='fmu') == 'W'


def test_world_unit_inference_and_errors(tmp_path, csv_file):
    w = world(start_time=0, stop_time=10, t_macro=10, power_unit='kW')
    fmu_like = _Adapter(inputs=['P'], outputs=['T_room', 'P_el'])
    w.simulator_dict['fmu'] = SimEntry('fmu', fmu_like, 10, ['T_room'], ['P'], power_unit='W')
    model = _Adapter(inputs=['t_in', 'p_cmd'], outputs=['y'])
    w.add_model('m', model=model, step_size=10, power_unit='kW')
    w.add_csv('csv', csv_file, step_size=10, outputs=['temp', 'p_kw'])
    w.add_connections([
        {'from': 'fmu.T_room', 'to': 'm.t_in'},     # temperature: never scaled
        {'from': 'fmu.P_el', 'to': 'm.p_cmd'},       # power names: W -> kW
        {'from': 'csv.p_kw', 'to': 'fmu.P'},         # kW suffix -> FMU W
        {'from': 'csv.temp', 'to': 'm.y'},           # unknown both sides -> no scaling
    ])
    scales = [c.scale for c in w._parsed_connections]
    assert scales[0] == 1.0 and w._parsed_connections[0].from_unit is None
    assert math.isclose(scales[1], 1e-3)
    assert math.isclose(scales[2], 1e3)
    assert scales[3] == 1.0

    w2 = world(start_time=0, stop_time=10, t_macro=10)
    w2.add_model('a', model=_Adapter(outputs=['T']), step_size=10, units={'T': 'degC'})
    w2.add_model('b', model=_Adapter(inputs=['T']), step_size=10, units={'T': 'K'})
    w2.add_model('c', model=_Adapter(inputs=['P']), step_size=10, units={'P': 'kW'})
    w2.add_connections({'a.T': 'b.T'})
    c = w2._parsed_connections[0]
    assert c.scale == 1.0 and math.isclose(c.offset, 273.15)
    with pytest.raises(ValueError, match="Incompatible"):
        w2.add_connections({'a.T': 'c.P'})
    with pytest.raises(ValueError, match="Unknown unit"):
        w2.add_connections([{'from': 'a.T', 'to': 'b.T', 'from_unit': 'bogus', 'to_unit': 'K'}])
    with pytest.raises(ValueError, match="Unknown unit"):
        w2.add_model('d', model=_Adapter(), units={'x': 'zorkmids'})


def test_to_dict_does_not_double_apply_units():
    w = world(start_time=0, stop_time=10, t_macro=10)
    w.add_signal('s', '1', units={'y': 'kW'}, step_size=10)
    w.add_model('m', model=_Adapter(inputs=['p_mw']), step_size=10)
    w.add_connections([{'from': 's.y', 'to': 'm.p_mw', 'scale': 2.0, 'from_unit': 'kW'}])
    assert math.isclose(w._parsed_connections[0].scale, 2e-3)
    d = w.to_dict()
    assert d['connections'][0]['scale'] == 2.0


# =====================================================================
# Task 9: errors instead of asserts / prints, duplicates
# =====================================================================

def test_duplicate_names_raise(csv_file):
    w = world()
    w.add_csv('dup', csv_file, outputs=['temp'])
    with pytest.raises(ValueError, match="already exists"):
        w.add_csv('dup', csv_file, outputs=['temp'])
    with pytest.raises(ValueError, match="already exists"):
        w.add_signal('dup', '1')
    with pytest.raises(ValueError, match="already exists"):
        w.add_model('dup', model=_Adapter())
    with pytest.raises(ValueError, match="already exists"):
        w.add_simulator('battery_fleet', 'dup', size=2)
    with pytest.raises(ValueError, match="already exists"):
        w.add_external_simulator('dup', model=_Adapter())


def test_validation_errors_are_exceptions():
    from energysim import ConnectionSpecError, ConnectionError as CE
    assert CE is ConnectionSpecError
    w = world()
    with pytest.raises(ValueError):
        w.add_connections({'a.b': 'c.d'})
    with pytest.raises(ValueError):
        w.add_csv(None, 'x.csv')
    with pytest.raises(ValueError):
        w.options({'init': {}})
    w.add_model('m', model=_Adapter(), step_size=1)
    with pytest.raises(ConnectionSpecError):
        w.add_connections({'m.y': 'nope.u'})
    with pytest.raises(ConnectionSpecError):
        w.add_connections({'my': 'm.u'})
    with pytest.raises(ValueError):
        world(extrapolation='cubic')


def test_world_uses_logging_not_print(capsys, caplog, tmp_path):
    import logging
    w = world(start_time=0, stop_time=10, t_macro=10)
    with caplog.at_level(logging.INFO, logger='energysim'):
        w.add_signal('s', '1', step_size=10)
        w.simulate(pbar=False)
    assert capsys.readouterr().out == ''
    assert any("Added signal simulator 's'" in r.getMessage() for r in caplog.records)


# =====================================================================
# Task 10: external simulators
# =====================================================================

def _write_ext(path, cls_name='external_simulator', value=1.0, with_kwargs=False):
    sig = 'self, inputs=None, outputs=None, step_size=None, sim_name=None' if with_kwargs \
        else 'self, outputs=None'
    path.write_text(textwrap.dedent(f'''
        from energysim.base import SimulatorAdapter
        class {cls_name}(SimulatorAdapter):
            def __init__({sig}):
                self.outputs = outputs or []
                self.kw = dict(locals())
            def step(self, time): pass
            def get_value(self, parameters, time): return [{value} for _ in parameters]
            def set_value(self, parameters, values): pass
    '''))


def test_external_instance_factory_and_file(tmp_path):
    inst = _Adapter()
    w = world()
    w.add_external_simulator('inst', model=inst)
    assert w.get_adapter('inst') is inst

    w.add_external_simulator('cls', model=_Adapter, outputs=['y'])
    assert isinstance(w.get_adapter('cls'), _Adapter)

    f = tmp_path / 'my_ext.py'
    _write_ext(f, 'MySim', with_kwargs=True)
    before = list(sys.path)
    w.add_external_simulator('file', model=f'{f}:MySim', outputs=['v'], step_size=5)
    assert sys.path == before
    ad = w.get_adapter('file')
    assert ad.kw['step_size'] == 5 and ad.kw['sim_name'] == 'file'

    g = tmp_path / 'plain.py'
    _write_ext(g, 'Plain', with_kwargs=False)
    w.add_external_simulator('plain', model=f'{g}:Plain', outputs=['v'], step_size=5)
    assert 'step_size' not in w.get_adapter('plain').kw


def test_external_legacy_directory(tmp_path):
    d = tmp_path / 'legacy'
    d.mkdir()
    _write_ext(d / 'legacy_ext_sim_xyz.py', value=3.0, with_kwargs=True)
    before = list(sys.path)
    w = world()
    w.add_external_simulator('legacy_ext_sim_xyz', sim_loc=str(d), outputs=['v'])
    assert sys.path == before
    assert w.get_adapter('legacy_ext_sim_xyz').get_value(['v'], 0) == [3.0]
    sys.modules.pop('legacy_ext_sim_xyz', None)


# =====================================================================
# Task 11: py script adapter isolation
# =====================================================================

def test_pyscript_same_module_name_no_collision(tmp_path):
    from energysim.pyScriptAdapter import py_adapter
    for n, val in (('a', 1), ('b', 2)):
        d = tmp_path / n
        d.mkdir()
        (d / 'ctrl.py').write_text(f'def run(inputs):\n    return [{val}]\n')
    before = list(sys.path)
    pa = py_adapter('ctrl.run', str(tmp_path / 'a'), outputs=['o'])
    pb = py_adapter('ctrl.run', str(tmp_path / 'b'), outputs=['o'])
    assert sys.path == before
    assert 'ctrl' not in sys.modules
    pa.step(0)
    pb.step(0)
    assert pa.get_value(['o']) == [1] and pb.get_value(['o']) == [2]
    pc = py_adapter('run', str(tmp_path / 'a' / 'ctrl.py'), outputs=['o'])
    pc.step(0)
    assert pc.get_value(['o']) == [1]
    pa.cleanup()
    pa.init()   # reloads after cleanup
    pa.step(0)
    assert pa.get_value(['o']) == [1]


def test_pyscript_sibling_imports(tmp_path):
    from energysim.pyScriptAdapter import py_adapter
    (tmp_path / 'helper_mod_qq.py').write_text('K = 4\n')
    (tmp_path / 'main_qq.py').write_text('import helper_mod_qq\ndef run(i):\n    return [helper_mod_qq.K]\n')
    p = py_adapter('main_qq.run', str(tmp_path), outputs=['o'])
    p.step(0)
    assert p.get_value(['o']) == [4]
    sys.modules.pop('helper_mod_qq', None)


# =====================================================================
# Task 12 / 13: options and silent failures
# =====================================================================

def test_options_unknown_key_and_record_all(tmp_path, csv_file):
    w = world(start_time=0, stop_time=30, t_macro=30)
    w.add_csv('c', csv_file, step_size=10, outputs=['temp'])
    with pytest.raises(ValueError, match="Unknown option"):
        w.options({'record_alll': True})
    with pytest.raises(ValueError, match="modify_signal"):
        w.options({'modify_signal': {'c.temp': [1, 2, 3]}})
    w.options({'record_all': False})
    w.simulate(pbar=False)
    assert len(w.last_results['c']) == 2   # t=0 + final row
    w.options({'record_all': True})
    w.simulate(pbar=False)
    assert len(w.last_results['c']) == 4   # micro steps 0,10,20 + final row


def test_csv_unknown_column_raises(csv_file):
    from energysim.base import SimulatorVariableError
    from energysim.csv_adapter import csv_simulator
    with pytest.raises(SimulatorVariableError):
        csv_simulator('c', csv_file, outputs=['nope'])
    c = csv_simulator('c', csv_file, outputs=['temp'])
    with pytest.raises(SimulatorVariableError):
        c.get_value(['nope'], 0)

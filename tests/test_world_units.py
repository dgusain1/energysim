# -*- coding: utf-8 -*-
"""World integration of the default unit system (energysim 3.0).

Resolution order, receiver-unit conversion, the unspecified/strict policy,
serialisation, visibility (logs, tables, validate, CLI, dashboard).
"""
from __future__ import annotations

import json
import logging
import math

import pandas as pd
import pytest

from energysim import SimulatorAdapter, world
from energysim.cli import main as cli_main
from energysim.models.common import ModelComponent
from energysim.signalAdapter import signal_adapter
from energysim.units import UnitSystem


class Sink(SimulatorAdapter):
    """Stores every received input by name; outputs read from ``values``."""

    def __init__(self, inputs=(), outputs=(), units=None, values=None, all_vars=None):
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.received = {}
        self.values = dict(values or {})
        self.step_size = 1.0
        self._all = all_vars
        if units is not None:
            self.units = dict(units)

    def step(self, time):
        pass

    def get_value(self, parameters, time):
        return [self.values.get(p, 0.0) for p in parameters]

    def set_value(self, parameters, values):
        for p, v in zip(parameters, values):
            self.received[p] = v

    def get_available_variables(self):
        d = {'inputs': list(self.inputs), 'outputs': list(self.outputs)}
        if self._all is not None:
            d['all'] = list(self._all)
        return d


def _w(**kw):
    kw.setdefault('start_time', 0)
    kw.setdefault('stop_time', 10)
    kw.setdefault('t_macro', 10)
    return world(**kw)


# ---------------------------------------------------------------------------
# Resolution order
# ---------------------------------------------------------------------------

def test_resolution_order_each_level():
    w = _w()
    src = Sink(outputs=['P_out', 'y', 'p_mw', 'flow'], units={'P_out': 'MW', 'y': 'kW'})
    w.add_model('src', src, step_size=10, units={'y': 'W'})
    # 1. simulator-level units beat the component's units
    assert w.resolve_variable_unit('src', 'y') == ('W', 'declared')
    # 2. component units
    assert w.resolve_variable_unit('src', 'P_out') == ('MW', 'component')
    # 3. connection unit beats suffix / quantity
    assert w.resolve_variable_unit('src', 'flow', 'kg/h') == ('kg/h', 'connection')
    # 4. suffix
    assert w.resolve_variable_unit('src', 'p_mw') == ('MW', 'suffix')
    # 5. default by quantity (unit system)
    assert w.resolve_variable_unit('src', 'P_cmd') == ('kW', 'default-by-quantity')
    assert w.resolve_variable_unit('src', 'T_amb') == ('degC', 'default-by-quantity')
    # 6. unspecified
    assert w.resolve_variable_unit('src', 'flow') == (None, 'unspecified')


def test_power_unit_is_shorthand_for_power_like_names():
    w = _w()
    w.add_model('m', Sink(outputs=['P', 'p_kw', 'temp']), step_size=10, power_unit='MW')
    assert w.resolve_variable_unit('m', 'P') == ('MW', 'declared')
    assert w.resolve_variable_unit('m', 'p_kw') == ('kW', 'suffix')     # suffix wins over power_unit
    assert w.resolve_variable_unit('m', 'temp') == ('degC', 'default-by-quantity')


def test_conversion_to_receiver_unit_and_record():
    w = _w()
    src = Sink(outputs=['y'], values={'y': 2500.0})
    dst = Sink(inputs=['p_mw'])
    w.add_model('a', src, step_size=10, units={'y': 'kW'})
    w.add_model('b', dst, step_size=10)
    w.add_connections({'a.y': 'b.p_mw'})
    c = w._parsed_connections[0]
    assert (c.from_unit, c.to_unit) == ('kW', 'MW')
    assert (c.from_source, c.to_source) == ('declared', 'suffix')
    assert c.scale == pytest.approx(1e-3) and c.offset == 0.0
    assert c.source == 'declared -> suffix'
    w.exchange_variable_values(0.0)
    assert dst.received['p_mw'] == pytest.approx(2.5)


def test_fan_out_converts_into_each_receivers_unit():
    w = _w()
    src = Sink(outputs=['y'], values={'y': 1.5})
    k, m = Sink(inputs=['p_kw']), Sink(inputs=['p_mw'])
    w.add_model('src', src, step_size=10, units={'y': 'MW'})
    w.add_model('k', k, step_size=10)
    w.add_model('m', m, step_size=10)
    w.add_connections({'src.y': ['k.p_kw', 'm.p_mw']})
    w.exchange_variable_values(0.0)
    assert k.received['p_kw'] == pytest.approx(1500.0)
    assert m.received['p_mw'] == pytest.approx(1.5)
    # Gauss-Seidel path as well
    k.received.clear()
    m.received.clear()
    w._exchange_inputs_for('k', 0.0)
    w._exchange_inputs_for('m', 0.0)
    assert k.received['p_kw'] == pytest.approx(1500.0) and m.received['p_mw'] == pytest.approx(1.5)
    df = w.connection_units()
    assert list(df['to']) == ['k.p_kw', 'm.p_mw']
    assert list(df['scale']) == pytest.approx([1000.0, 1.0])


def test_fan_in_sources_with_different_units_are_summed_in_a_common_unit():
    w = _w()
    a = Sink(outputs=['p_kw'], values={'p_kw': 500.0})
    b = Sink(outputs=['p_mw'], values={'p_mw': 1.0})
    dst = Sink(inputs=['p_mw'])
    for n, s in (('a', a), ('b', b), ('d', dst)):
        w.add_model(n, s, step_size=10)
    w.add_connections([{'from': ('a.p_kw', 'b.p_mw'), 'to': 'd.p_mw'}])
    w.exchange_variable_values(0.0)
    assert dst.received['p_mw'] == pytest.approx(1.5)


def test_unspecified_is_never_scaled_and_warns_once(caplog):
    w = _w()
    src = Sink(outputs=['y'], values={'y': 7.0})
    dst = Sink(inputs=['p_mw'])
    w.add_model('a', src, step_size=10)
    w.add_model('b', dst, step_size=10)
    with caplog.at_level(logging.WARNING, logger='energysim'):
        w.add_connections({'a.y': 'b.p_mw'})
        w.add_connections({'a.y': 'b.p_mw'})     # same connection again: no second warning
    msgs = [r.getMessage() for r in caplog.records if 'unspecified' in r.getMessage()]
    assert len(msgs) == 1 and 'a.y' in msgs[0]
    c = w._parsed_connections[0]
    assert (c.scale, c.offset, c.from_unit, c.to_unit) == (1.0, 0.0, None, 'MW')
    w.exchange_variable_values(0.0)
    assert dst.received['p_mw'] == 7.0


def test_guessed_incompatible_units_do_not_raise_but_explicit_do():
    w = _w()
    w.add_model('a', Sink(outputs=['temp']), step_size=10)
    w.add_model('b', Sink(inputs=['P_set', 'p_mw']), step_size=10)
    w.add_connections({'a.temp': 'b.P_set'})            # degC (guess) -> kW (guess): passed through
    assert w._parsed_connections[0].scale == 1.0
    assert any('incompatible' in m for m in w.validate().warnings)
    w2 = _w()
    w2.add_model('a', Sink(outputs=['t_degc']), step_size=10)
    w2.add_model('b', Sink(inputs=['p_mw']), step_size=10)
    with pytest.raises(ValueError, match='(?i)incompatible'):
        w2.add_connections({'a.t_degc': 'b.p_mw'})


def test_strict_policy_raises_for_unspecified_connected_variables():
    w = _w(unit_policy='strict')
    w.add_model('a', Sink(outputs=['y']), step_size=10)
    w.add_model('b', Sink(inputs=['p_mw']), step_size=10)
    with pytest.raises(ValueError, match="unit_policy='strict'"):
        w.add_connections({'a.y': 'b.p_mw'})
    w.add_connections([{'from': 'a.y', 'to': 'b.p_mw', 'from_unit': 'kW'}])   # declared on the connection
    assert w._parsed_connections[0].scale == pytest.approx(1e-3)
    with pytest.raises(ValueError, match='unit_policy'):
        world(unit_policy='paranoid')


def test_strict_policy_validate_reports_errors():
    w = _w()
    w.add_model('a', Sink(outputs=['y']), step_size=10)
    w.add_model('b', Sink(inputs=['u']), step_size=10)
    w.add_connections({'a.y': 'b.u'})
    assert w.validate().is_valid
    w.unit_policy = 'strict'
    rep = w.validate()
    assert not rep.is_valid and any('unspecified' in e for e in rep.errors)


def test_component_units_are_used_and_unknown_component_units_pass_through():
    class Gen(ModelComponent):
        units = {'P_out': 'MW', 'fuel': 'MMBtu/h'}

        def __init__(self):
            super().__init__(name='g')
            self.register_variables(inputs=['u'], outputs=['P_out', 'fuel'])

        def step_model(self, dt, time):
            self._outputs_val['P_out'] = 2.0
            self._outputs_val['fuel'] = 3.0

    w = _w()
    w.add_model('g', Gen(), step_size=10)
    k = Sink(inputs=['p_kw', 'fuel_in'])
    w.add_model('k', k, step_size=10)
    w.add_connections({'g.P_out': 'k.p_kw', 'g.fuel': 'k.fuel_in'})
    c0, c1 = w._parsed_connections
    assert c0.scale == pytest.approx(1000.0) and c0.from_source == 'component'
    assert c1.scale == 1.0 and c1.from_unit == 'MMBtu/h'
    res = w.simulate(pbar=False, record_all=False)
    assert res.units['g.P_out'] == 'MW' and res.units['g.fuel'] == 'MMBtu/h'


def test_connection_unit_is_validated_and_declared_wins(caplog):
    w = _w()
    w.add_model('a', Sink(outputs=['y']), step_size=10, units={'y': 'kW'})
    w.add_model('b', Sink(inputs=['u']), step_size=10, units={'u': 'MW'})
    with pytest.raises(ValueError, match='Unknown unit'):
        w.add_connections([{'from': 'a.y', 'to': 'b.u', 'from_unit': 'bogus'}])
    with caplog.at_level(logging.WARNING, logger='energysim'):
        w.add_connections([{'from': 'a.y', 'to': 'b.u', 'from_unit': 'W'}])
    assert w._parsed_connections[0].scale == pytest.approx(1e-3)       # declared kW wins
    assert any('ignored' in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Unit systems
# ---------------------------------------------------------------------------

def test_si_preset_world_uses_si_defaults():
    w = _w(unit_system='si')
    src = Sink(outputs=['P_gen'], values={'P_gen': 2000.0})     # active power -> W in SI
    dst = Sink(inputs=['p_kw'])
    w.add_model('a', src, step_size=10)
    w.add_model('b', dst, step_size=10)
    w.add_connections({'a.P_gen': 'b.p_kw'})
    assert w._parsed_connections[0].from_unit == 'W'
    w.exchange_variable_values(0.0)
    assert dst.received['p_kw'] == pytest.approx(2.0)
    assert w.resolve_variable_unit('a', 'T_room') == ('K', 'default-by-quantity')
    assert w.unit_system == UnitSystem.si()


def test_temperature_default_by_quantity_converts_to_declared_kelvin():
    w = _w()
    src = Sink(outputs=['T_out'], values={'T_out': 20.0})
    fmu = Sink(inputs=['T_in'], units={'T_in': 'K'})
    w.add_model('a', src, step_size=10)
    w.add_model('f', fmu, step_size=10)
    w.add_connections({'a.T_out': 'f.T_in'})
    w.exchange_variable_values(0.0)
    assert fmu.received['T_in'] == pytest.approx(293.15)


@pytest.mark.parametrize('spec', ['si', {'active_power': 'MW'}, UnitSystem.si().with_overrides({'pressure': 'kPa'}),
                                  {'name': 'si', 'overrides': {'voltage': 'kV'}}])
def test_unit_system_yaml_round_trip(spec):
    w = _w(unit_system=spec, unit_policy='strict')
    w.add_signal('s', '1', step_size=10, units={'y': 'kW'})
    w.add_signal('m', '0', step_size=10, units={'y': 'MW'})
    with pytest.warns(UserWarning, match='read-only'):
        w.add_connections({'s.y': 'm.y'})
    d = w.to_dict()
    assert 'unit_system' in d['world'] and d['world']['unit_policy'] == 'strict'
    with pytest.warns(UserWarning, match='read-only'):
        w2 = world.from_yaml(w.to_yaml())
    assert w2.unit_system == w.unit_system
    assert w2.unit_system.hash == w.unit_system.hash
    assert w2.unit_policy == 'strict'
    assert w2._parsed_connections[0].scale == pytest.approx(1e-3)


def test_default_world_serialises_default_unit_system():
    d = _w().to_dict()['world']
    assert d['unit_system'] == {'name': 'energysim-default'}
    assert 'power_unit' not in d and 'unit_policy' not in d
    assert world.from_dict({'world': {'unit_system': 'si'}}).unit_system == UnitSystem.si()


def test_results_units_and_meta_unit_system():
    w = _w(unit_system={'active_power': 'MW'})
    w.add_model('a', Sink(outputs=['P_gen', 'p_kw', 'x']), step_size=10)
    res = w.simulate(pbar=False, record_all=False)
    assert res.units['a.P_gen'] == 'MW'
    assert res.units['a.p_kw'] == 'kW'
    assert res.units['a.x'] is None
    meta = res.meta['unit_system']
    assert meta['hash'] == w.unit_system.hash and meta['defaults']['active_power'] == 'MW'
    assert meta['name'] == 'custom'


# ---------------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------------

def test_log_lines(caplog):
    with caplog.at_level(logging.DEBUG, logger='energysim'):
        w = _w(logging=True)
        w.add_model('a', Sink(outputs=['y'], values={'y': 1000.0}), step_size=10, units={'y': 'kW'})
        w.add_model('b', Sink(inputs=['p_mw']), step_size=10)
        w.add_connections({'a.y': 'b.p_mw'})
        caplog.clear()
        w.simulate(pbar=False, record_all=False)
    text = [r.getMessage() for r in caplog.records]
    assert any(t.startswith('Default unit system: unit system ') for t in text)
    line = [t for t in text if t.startswith('Unit conversion:')]
    assert line == ['Unit conversion: a.y [kW] -> b.p_mw [MW]  x0.001  (source: declared -> suffix)']


def test_logging_true_logs_unit_system_on_creation(caplog):
    with caplog.at_level(logging.INFO, logger='energysim'):
        world(logging=True, unit_system='si')
    assert any("Default unit system: unit system 'si'" in r.getMessage() for r in caplog.records)


def test_unit_table_and_connection_units(capsys):
    w = _w(unit_system='si')
    table = w.unit_table()
    assert table == UnitSystem.si().format_table()
    w.unit_table(show=True)
    assert "Unit system 'si'" in capsys.readouterr().out
    w.add_model('a', Sink(outputs=['y']), step_size=10, units={'y': 'degC'})
    w.add_model('b', Sink(inputs=['T']), step_size=10, units={'T': 'K'})
    w.add_connections({'a.y': 'b.T'})
    df = w.connection_units()
    assert isinstance(df, pd.DataFrame)
    row = df.iloc[0]
    assert (row['from'], row['to'], row['from_unit'], row['to_unit']) == ('a.y', 'b.T', 'degC', 'K')
    assert row['scale'] == 1.0 and row['offset'] == pytest.approx(273.15)
    assert w.conversion_lines() == ['a.y [degC] -> b.T [K]  x1 +273.15  (source: declared)']


def test_validate_lists_conversions_and_unspecified_warnings():
    w = _w()
    w.add_model('a', Sink(outputs=['y', 'z']), step_size=10, units={'y': 'MW'})
    w.add_model('b', Sink(inputs=['p_kw', 'u']), step_size=10)
    w.add_connections({'a.y': 'b.p_kw', 'a.z': 'b.u'})
    rep = w.validate()
    assert rep.is_valid
    assert rep.conversions[0] == 'a.y [MW] -> b.p_kw [kW]  x1000  (source: declared -> suffix)'
    assert any('unspecified' in m and 'a.z' in m for m in rep.warnings)
    assert rep.unit_system.startswith('energysim-default [')
    assert 'Unit conversions (2)' in rep.format_units()


def test_dashboard_labels_carry_units(tmp_path):
    from energysim.dashboard import _build_traces_json, variable_label
    w = _w()
    w.add_model('a', Sink(outputs=['p_mw', 'x']), step_size=10)
    res = w.simulate(pbar=False, record_all=False)
    sims = _build_traces_json(res)
    labels = sims[0]['labels']
    assert labels == {'p_mw': 'a.p_mw [MW]', 'x': 'a.x'}
    assert variable_label('s', 'v', 'kW') == 's.v [kW]'
    html = open(res.dashboard(path=str(tmp_path / 'd.html')), encoding='utf-8').read()
    assert 'a.p_mw [MW]' in html and 'simData.labels' in html


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_units_default(capsys):
    assert cli_main(['units']) == 0
    out = capsys.readouterr().out
    assert out.startswith("Unit system 'energysim-default'")
    assert 'active_power' in out and 'kW' in out


def test_cli_units_si_json(capsys):
    assert cli_main(['units', '--system', 'si', '--json']) == 0
    data = json.loads(capsys.readouterr().out)
    assert data['name'] == 'si' and data['defaults']['active_power'] == 'W'
    assert data['hash'] == UnitSystem.si().hash and len(data['table']) == len(data['defaults'])


def test_cli_units_set_override_and_wrong_dimension(capsys):
    assert cli_main(['units', '--set', 'active_power=MW', '--set', 'pressure=kPa']) == 0
    out = capsys.readouterr().out
    assert 'active_power               MW' in out
    assert cli_main(['units', '--set', 'active_power=degC']) == 1
    assert 'not valid for quantity' in capsys.readouterr().err
    assert cli_main(['units', '--set', 'nonsense']) == 1


def test_cli_validate_and_run_verbose_print_units(tmp_path, capsys):
    path = tmp_path / 'sc.yaml'
    path.write_text(
        "world: {start_time: 0, stop_time: 20, t_macro: 10, unit_system: si}\n"
        "simulators:\n"
        "  - {name: s, type: signal, signal: '1000', step_size: 10, units: {y: kW}}\n"
        "  - {name: t, type: signal, signal: '2', step_size: 10, units: {y: MW}}\n"
        "connections:\n"
        "  - {from: s.y, to: t.y}\n", encoding='utf-8')
    assert cli_main(['validate', str(path), '-v']) == 0
    out = capsys.readouterr().out
    assert f'Unit system: si [{UnitSystem.si().hash}]' in out
    assert 's.y [kW] -> t.y [MW]  x0.001  (source: declared)' in out
    assert cli_main(['run', str(path), '--no-pbar', '--verbose']) == 0
    assert 's.y [kW] -> t.y [MW]' in capsys.readouterr().out
    assert cli_main(['validate', str(path)]) == 0
    assert 'Unit conversions' not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Small adapter fixes
# ---------------------------------------------------------------------------

def test_signal_get_value_empty_list():
    s = signal_adapter('s', lambda t: [1.0])
    assert s.get_value([], 0.0) == []
    assert s.get_value(['y'], 0.0) == [1.0]


def test_model_component_output_that_is_also_an_input_passes_through():
    class Through(ModelComponent):
        def __init__(self):
            super().__init__(name='t')
            self.register_variables(inputs=['x'], outputs=['x', 'y'])

        def step_model(self, dt, time):
            self._outputs_val['y'] = 2.0 * self._inputs_val['x']

    m = Through()
    m.init()
    m.set_value(['x'], [4.0])
    assert m.get_value(['x'], 0.0) == [4.0]


def test_validation_uses_all_variables_for_fmu_internals():
    import warnings
    w = _w()
    fmu = Sink(inputs=['u'], outputs=['y'], all_vars=['u', 'y', 'local_state'])
    w.add_model('f', fmu, step_size=10)
    w.add_model('d', Sink(inputs=['v']), step_size=10)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        w.add_connections({'f.local_state': 'd.v'})
    assert not any('local_state' in m and 'not found' in m for m in w.validate().warnings)
    with pytest.warns(UserWarning, match='not found'):
        w.add_connections({'f.missing': 'd.v'})


def test_utils_helpers_delegate_to_units_module():
    from energysim import units, utils
    assert utils.parse_unit is units.parse_unit
    assert utils.get_unit_conversion('degC', 'K') == units.get_unit_conversion('degC', 'K')
    assert utils.get_unit_scale('kW', 'MW') == pytest.approx(1e-3)
    assert utils.infer_variable_unit('Gen.p_mw') == 'MW'
    assert utils.infer_variable_unit('x_kg_h') == 'kg/h'           # suffix table of energysim.units
    assert utils.infer_variable_unit('P', sim_type='fmu') == 'W'
    assert math.isclose(utils.get_unit_scale('MMBtu/h', 'MMBtu/h') if units.is_known_unit('MMBtu/h') else 1.0, 1.0)

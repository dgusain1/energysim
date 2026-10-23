# -*- coding: utf-8 -*-
"""YAML ``type: model``, single-output signals, safe-mode hardening, record_all default, event hold."""
from __future__ import annotations

import os
import textwrap

import pytest
import yaml

from energysim import world
from energysim.cli import main as cli_main
from energysim.models import BatteryBESS, PEMElectrolyzer
from energysim.models.catalog import BESS_4h_2MW
from energysim.networks import HeatNetwork

BASE = "world: {start_time: 0, stop_time: 600, t_macro: 60}\n"


# ---------------------------------------------------------------- type: model

def test_yaml_model_class_with_parameters():
    w = world.from_yaml(BASE + textwrap.dedent("""
        simulators:
          - {name: cmd, type: signal, signal: 1.0, step_size: 60, units: {y: MW}}
          - name: ely
            type: model
            model: PEMElectrolyzer
            step_size: 60
            parameters: {p_nominal_mw: 2.0}
        connections:
          - {from: cmd.y, to: ely.P_cmd}
    """), safe=True)
    ely = w.get_model('ely')
    assert isinstance(ely, PEMElectrolyzer) and ely.p_nominal_mw == 2.0 and ely.name == 'ely'
    res = w.simulate(pbar=False)
    assert res['ely.m_dot_H2_kg_hr'].max() > 0.0


def test_yaml_model_options_alias_and_archetype():
    w = world.from_yaml(BASE + textwrap.dedent("""
        simulators:
          - {name: b, type: model, model: BESS_4h_2MW, step_size: 60, options: {capacity_mwh: 6.0}}
    """), safe=True)
    b = w.get_model('b')
    assert isinstance(b, BatteryBESS) and b.capacity_mwh == 6.0 and b.p_max_mw == 2.0


def test_yaml_model_errors_are_clear():
    with pytest.raises(ValueError, match="unknown model.*Did you mean.*PEMElectrolyzer"):
        world.from_yaml(BASE + "simulators:\n  - {name: x, type: model, model: PEMElectrolyser}\n")
    with pytest.raises(ValueError, match=r"no parameter\(s\) \['p_nominal']"):
        world.from_yaml(BASE + "simulators:\n  - {name: x, type: model, model: PEMElectrolyzer, "
                               "parameters: {p_nominal: 1}}\n")
    with pytest.raises(ValueError, match="both 'options' and 'parameters'"):
        world.from_yaml(BASE + "simulators:\n  - {name: x, type: model, model: PEMElectrolyzer, "
                               "options: {p_nominal_mw: 1}, parameters: {p_nominal_mw: 2}}\n")
    with pytest.raises(ValueError, match="only valid for type 'model'"):
        world.from_yaml(BASE + "simulators:\n  - {name: x, type: signal, signal: 1, parameters: {a: 1}}\n")


def test_yaml_model_from_file_not_in_safe_mode(tmp_path):
    (tmp_path / "plant.py").write_text(textwrap.dedent("""
        from energysim.models import BatteryBESS
        class Plant(BatteryBESS):
            pass
    """), encoding="utf-8")
    scen = tmp_path / "s.yaml"
    scen.write_text(BASE + "simulators:\n  - {name: p, type: model, model: 'plant.py:Plant', "
                           "step_size: 60, options: {capacity_mwh: 3.0}}\n", encoding="utf-8")
    w = world.from_yaml(str(scen))
    assert type(w.get_model('p')).__name__ == 'Plant' and w.get_model('p').capacity_mwh == 3.0
    with pytest.raises(ValueError, match="not allowed in safe mode"):
        world.from_yaml(str(scen), safe=True)
    assert cli_main(['validate', str(scen), '--safe']) == 1


def test_model_roundtrip_to_dict_and_yaml():
    w = world(start_time=0, stop_time=600, t_macro=60)
    w.add_signal('cmd', 0.5, step_size=60, units={'y': 'MW'})
    w.add_model('bess', BESS_4h_2MW(capacity_mwh=6.0), step_size=60)
    w.add_model('ely', PEMElectrolyzer(p_nominal_mw=3.0), step_size=60)
    w.add_connections({'cmd.y': 'bess.P_cmd'})
    d = w.to_dict()
    sims = {s['name']: s for s in d['simulators']}
    assert sims['bess']['model'] == 'BatteryBESS' and sims['bess']['parameters']['capacity_mwh'] == 6.0
    assert sims['ely']['model'] == 'PEMElectrolyzer' and sims['ely']['parameters'] == {'p_nominal_mw': 3.0}
    w2 = world.from_yaml(w.to_yaml(), safe=True)
    assert w2.get_model('bess').capacity_mwh == 6.0 and w2.get_model('ely').p_nominal_mw == 3.0
    a = w.simulate(pbar=False)['bess.SoC']
    b = w2.simulate(pbar=False)['bess.SoC']
    assert a.tolist() == pytest.approx(b.tolist())
    assert w2.to_dict() == d


def test_unserialisable_model_is_skipped_with_warning(caplog):
    class Custom(BatteryBESS):
        pass

    w = world(start_time=0, stop_time=60, t_macro=60)
    w.add_model('c', Custom(), step_size=60)
    with caplog.at_level('WARNING', logger='energysim'):
        d = w.to_dict()
    assert 'model' not in d['simulators'][0]
    assert any('not serialised' in r.message for r in caplog.records)


def test_network_roundtrip_via_topology():
    net = HeatNetwork(name='dh')
    net.add_producer('plant', capacity=5.0, cost=10.0)
    net.add_consumer('city', demand=3.0)
    net.add_pipe('plant', 'city', capacity=10.0)
    w = world(start_time=0, stop_time=120, t_macro=60)
    w.add_model('dh', net, step_size=60)
    d = w.to_dict()
    assert d['simulators'][0]['model'] == 'HeatNetwork' and 'topology' in d['simulators'][0]['parameters']
    w2 = world.from_dict(yaml.safe_load(w.to_yaml()), safe=True)
    assert w2.get_model('dh').to_dict() == net.to_dict()


# ---------------------------------------------------------------- multi-valued signals

@pytest.mark.parametrize("spec", ["[t, 2 * t]", lambda t: [1.0, 2.0], lambda t: (1.0, 2.0)])
def test_multi_valued_signal_is_rejected(spec):
    w = world(start_time=0, stop_time=60, t_macro=60)
    with pytest.raises(ValueError, match="add one signal per output"):
        w.add_signal('s', spec, step_size=60)
    assert 's' not in w.simulator_dict


def test_multi_valued_signal_rejected_from_yaml():
    with pytest.raises(ValueError, match="single output 'y'"):
        world.from_yaml(BASE + "simulators:\n  - {name: s, type: signal, signal: '[1, 2]', step_size: 60}\n")


def test_list_of_one_callable_still_works():
    w = world(start_time=0, stop_time=60, t_macro=60)
    w.add_signal('s', lambda t: [4.0], step_size=60)
    w.add_signal('u', "[3]", step_size=60)
    res = w.simulate(pbar=False)
    assert res['s.y'].iloc[-1] == 4.0 and res['u.y'].iloc[-1] == 3.0


# ---------------------------------------------------------------- safe mode hardening

def test_safe_mode_refuses_fmu_unless_allow_native(tmp_path):
    scen = BASE + "simulators:\n  - {name: f, type: fmu, model: m.fmu, step_size: 60}\n"
    with pytest.raises(ValueError, match="native code"):
        world.from_yaml(scen, safe=True)
    scen_file = tmp_path / "s.yaml"
    scen_file.write_text(scen, encoding="utf-8")
    assert cli_main(['validate', str(scen_file), '--safe']) == 1
    # allow_native passes the safe-mode check; the (missing) FMU is then the only complaint
    with pytest.raises(Exception) as exc:
        world.from_yaml(scen, safe=True, allow_native=True)
    assert "safe mode" not in str(exc.value)


def test_safe_mode_confines_paths_to_scenario_tree(tmp_path):
    sub = tmp_path / "scen"
    sub.mkdir()
    (sub / "ok.csv").write_text("time,v\n0,1\n60,2\n", encoding="utf-8")
    outside = tmp_path / "secret.csv"
    outside.write_text("time,v\n0,1\n60,2\n", encoding="utf-8")

    def scenario(model, extra=""):
        p = sub / "s.yaml"
        p.write_text(BASE + extra + "simulators:\n  - {name: c, type: csv, model: '%s', step_size: 60, "
                     "outputs: [v]}\n" % model, encoding="utf-8")
        return str(p)

    w = world.from_yaml(scenario("ok.csv"), safe=True)
    assert os.path.isabs(w.simulator_dict['c'].model)
    assert w.simulate(pbar=False)['c.v'].iloc[-1] == 2.0
    w = world.from_yaml(scenario((sub / "ok.csv").as_posix()), safe=True)       # absolute but inside: fine
    assert w.simulate(pbar=False)['c.v'].iloc[0] == 1.0

    for bad in ("../secret.csv", outside.as_posix(), "sub/../../secret.csv"):
        with pytest.raises(ValueError, match="outside the scenario directory"):
            world.from_yaml(scenario(bad), safe=True)
    # not in safe mode the same paths are fine
    world.from_yaml(scenario("../secret.csv"))
    assert cli_main(['validate', scenario("../secret.csv"), '--safe']) == 1
    assert cli_main(['validate', scenario("ok.csv"), '--safe']) == 0


def test_safe_mode_refuses_symlink_escape(tmp_path):
    sub = tmp_path / "scen"
    sub.mkdir()
    target = tmp_path / "secret.csv"
    target.write_text("time,v\n0,1\n", encoding="utf-8")
    try:
        os.symlink(target, sub / "link.csv")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available")
    p = sub / "s.yaml"
    p.write_text(BASE + "simulators:\n  - {name: c, type: csv, model: link.csv, step_size: 60, outputs: [v]}\n",
                 encoding="utf-8")
    with pytest.raises(ValueError, match="outside the scenario directory"):
        world.from_yaml(str(p), safe=True)


def test_safe_mode_confines_out_dir(tmp_path):
    sub = tmp_path / "scen"
    sub.mkdir()

    def scenario(out):
        p = sub / "s.yaml"
        p.write_text("world: {start_time: 0, stop_time: 60, t_macro: 60, out_dir: '%s'}\n"
                     "simulators:\n  - {name: s, type: signal, signal: 1, step_size: 60}\n" % out,
                     encoding="utf-8")
        return str(p)

    w = world.from_yaml(scenario("runs"), safe=True)
    assert w.out_dir == os.path.realpath(sub / "runs")
    for bad in ("../elsewhere", (tmp_path / "abs").as_posix()):
        with pytest.raises(ValueError, match="out_dir.*outside the scenario directory"):
            world.from_yaml(scenario(bad), safe=True)
    assert world.from_yaml(scenario("../elsewhere")).out_dir == "../elsewhere"     # unrestricted otherwise


def test_safe_mode_string_scenarios_are_confined_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scen = BASE + "simulators:\n  - {name: c, type: csv, model: '../x.csv', step_size: 60, outputs: [v]}\n"
    with pytest.raises(ValueError, match="outside the scenario directory"):
        world.from_yaml(scen, safe=True)


# ---------------------------------------------------------------- recording default

def _ramp_world(**kw):
    w = world(start_time=0, stop_time=120, t_macro=60, **kw)
    w.add_signal('s', 't', step_size=10)
    return w


def test_simulate_defaults_to_macro_step_recording():
    res = _ramp_world().simulate(pbar=False)
    assert list(res['s.y'].index) == [0.0, 60.0, 120.0]
    assert res.meta['record_all'] is False
    micro = _ramp_world().simulate(pbar=False, record_all=True)
    assert len(micro['s.y']) > 3 and micro.meta['record_all'] is True


def test_record_all_option_and_yaml_key():
    w = _ramp_world()
    w.options({'record_all': True})
    assert len(w.simulate(pbar=False)['s.y']) > 3
    w2 = world.from_yaml("world: {start_time: 0, stop_time: 120, t_macro: 60, record_all: true}\n"
                         "simulators:\n  - {name: s, type: signal, signal: t, step_size: 10}\n")
    assert w2.record_all is True and w2.to_dict()['world']['record_all'] is True
    assert len(w2.simulate(pbar=False)['s.y']) > 3
    with pytest.raises(ValueError, match="record_all"):
        world.from_dict({'world': {'record_all': 'yes'}, 'simulators': []})


def test_cli_record_all_flag(tmp_path, capsys):
    f = tmp_path / "s.yaml"
    f.write_text("world: {start_time: 0, stop_time: 120, t_macro: 60}\n"
                 "simulators:\n  - {name: s, type: signal, signal: t, step_size: 10}\n", encoding="utf-8")
    def count():
        line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("s.y")][0]
        return int(line.split()[5])

    assert cli_main(['run', str(f), '--no-pbar']) == 0
    assert count() == 3
    assert cli_main(['run', str(f), '--no-pbar', '--record-all']) == 0
    assert count() > 3


# ---------------------------------------------------------------- set events on connected inputs

@pytest.mark.parametrize("coupling", ["jacobi", "gauss-seidel"])
def test_set_event_on_connected_input_holds_for_its_step(coupling):
    w = world(start_time=0, stop_time=300, t_macro=60, coupling=coupling)
    w.add_signal('src', 1.0, step_size=60, units={'y': 'kW'})
    w.add_model('b', BatteryBESS(capacity_mwh=1.0, p_max_mw=1.0), step_size=60)
    w.add_connections({'src.y': 'b.P_cmd'})
    w.schedule(120, {'set': {'b.P_cmd': 0.0}})
    seen = []
    adapter = w.get_adapter('b')
    original = adapter.set_value

    def spy(params, values):
        seen.append((list(params), list(values)))
        return original(params, values)

    adapter.set_value = spy
    w.simulate(pbar=False)
    assert w.last_results.meta['events'][0]['success'] is True
    values = [v[0] for p, v in seen if p == ['P_cmd']]
    # the event value (0.0) is written once and the exchange of that step is skipped;
    # the exchange resumes with the next step
    assert values.count(0.0) == 1
    i = values.index(0.0)
    assert values[i + 1] != 0.0 and all(v != 0.0 for v in values[:i])


def test_set_event_hold_changes_the_physics():
    def run(hold_event):
        w = world(start_time=0, stop_time=300, t_macro=60)
        w.add_signal('src', 0.5, step_size=60, units={'y': 'MW'})
        w.add_model('b', BatteryBESS(capacity_mwh=1.0, p_max_mw=1.0), step_size=60)
        w.add_connections({'src.y': 'b.P_cmd'})
        if hold_event:
            w.schedule(120, {'set': {'b.P_cmd': 0.0}})
        return w.simulate(pbar=False)['b.P_actual']

    base, held = run(False), run(True)
    assert base.loc[180.0] != 0.0
    assert held.loc[180.0] == pytest.approx(0.0, abs=1e-6)     # the held 0 MW was in force for [120, 180)
    assert held.loc[240.0] == pytest.approx(base.loc[240.0], rel=0.2)   # exchange resumed afterwards

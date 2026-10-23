# -*- coding: utf-8 -*-
"""Tests for energysim.results (Results / Recorder), world integration, dashboard and CLI."""
from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

import energysim
from energysim import world
from energysim.base import SimulatorAdapter
from energysim.results import Recorder, Results, register_kpi, unregister_kpi, available_kpis

try:
    import pyarrow  # noqa: F401
    HAVE_PYARROW = True
except ImportError:  # pragma: no cover
    HAVE_PYARROW = False

try:
    import tables  # noqa: F401
    HAVE_TABLES = True
except ImportError:  # pragma: no cover
    HAVE_TABLES = False


class _Typed(SimulatorAdapter):
    """Outputs of every type: float, int, bool, str; int widens to float later."""

    def __init__(self):
        self.outputs = ['f', 'i', 'b', 's', 'w']
        self.inputs = []
        self.t = 0.0

    def step(self, time):
        self.t = time + self.step_size

    def get_value(self, parameters, time):
        vals = {'f': self.t * 1.5, 'i': int(self.t), 'b': self.t >= 20,
                's': 'on' if self.t >= 20 else 'off', 'w': int(self.t) if self.t < 20 else self.t + 0.5}
        return [vals[p] for p in parameters]

    def set_value(self, parameters, values):
        pass


class _Boom(SimulatorAdapter):
    def __init__(self):
        self.outputs = ['y']
        self.inputs = []

    def step(self, time):
        if time >= 20:
            raise RuntimeError("boom at 20")

    def get_value(self, parameters, time):
        return [time]

    def set_value(self, parameters, values):
        pass


class _Warner(_Boom):
    def step(self, time):
        warnings.warn("adapter says hi", UserWarning, stacklevel=2)


def _typed_world(**kw):
    w = world(start_time=0, stop_time=40, t_macro=10, **kw)
    w.add_model('m', model=_Typed(), step_size=10)
    return w


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def test_simulate_returns_results_with_final_row():
    w = world(start_time=0, stop_time=40, t_macro=10)
    w.add_signal('s', '2*t', step_size=10)
    res = w.simulate(pbar=False)
    assert isinstance(res, Results) and res is w.last_results
    assert res['s.y'].index.tolist() == [0.0, 10.0, 20.0, 30.0, 40.0]
    assert res['s.y'].tolist() == [0.0, 20.0, 40.0, 60.0, 80.0]
    assert res.status == 'completed'
    assert not hasattr(w, 'results')


@pytest.mark.parametrize('chunk', [65536, 2])
def test_typed_columns(chunk, monkeypatch):
    monkeypatch.setattr(world, '_results_chunk_rows', chunk)
    res = _typed_world().simulate(pbar=False, record_all=False)
    df = res['m']
    assert df['f'].dtype == np.float64
    assert df['i'].dtype == np.int64
    assert df['b'].dtype == bool
    assert df['s'].dtype == object
    assert df['w'].dtype == np.float64            # int widened to float
    assert df['s'].tolist() == ['off', 'off', 'on', 'on', 'on']
    assert df['b'].tolist() == [False, False, True, True, True]
    assert df['i'].tolist() == [0, 10, 20, 30, 40]


def test_ocgt_string_state_recorded():
    from energysim.models.thermal import GasTurbine_OCGT
    w = world(start_time=0, stop_time=3600, t_macro=600)
    w.add_model('ocgt', model=GasTurbine_OCGT(name='ocgt'), step_size=600)
    w.add_signal('cmd', 50.0, step_size=600, units={'y': 'MW'})
    w.add_connections({'cmd.y': 'ocgt.P_cmd'})
    res = w.simulate(pbar=False)
    states = res['ocgt.state']
    assert states.dtype == object
    assert any(isinstance(v, str) for v in states)


def test_record_all_rows_and_irregular_times():
    w = world(start_time=0, stop_time=30, t_macro=30)
    w.add_signal('s', 't', step_size=10)
    res = w.simulate(pbar=False, record_all=True)
    assert res['s.y'].index.tolist() == [0.0, 10.0, 20.0, 30.0]

    rec = Recorder({'a': ['x'], 'b': ['y']})
    rec.add_rows('a', [[0.0, 1.0], [0.7, 2.0], [0.7, 2.5], [3.1, 3.0]])   # irregular, duplicate t
    rec.add_rows('b', [[0.0, 'a'], [1.0, 'b']])
    r = Results(rec.finalize())
    assert r['a.x'].index.tolist() == [0.0, 0.7, 0.7, 3.1]
    wide = r.to_frame()
    assert wide.index.tolist() == [0.0, 0.7, 1.0, 3.1]
    assert wide['a.x'].tolist() == [1.0, 2.5, 2.5, 3.0]          # duplicate keeps the last, ffill
    assert wide['b.y'].tolist() == ['a', 'a', 'b', 'b']


def test_recorder_row_width_checked():
    rec = Recorder({'a': ['x', 'y']}, chunk_rows=1)
    with pytest.raises(ValueError, match="2 output"):
        rec.add_row('a', [0.0, 1.0])


def test_million_rows_recording():
    rec = Recorder({'s': ['p', 'state']})
    t0 = time.perf_counter()
    for i in range(1_000_000):
        rec.add_row('s', [float(i), i * 0.5, 'on'])
    df = rec.finalize()['s']
    assert time.perf_counter() - t0 < 20
    assert len(df) == 1_000_000 and df['p'].dtype == np.float64
    assert df.memory_usage(deep=False).sum() < 50e6


def test_performance_80_simulators():
    """40 PEM electrolysers + 40 signals for 360 macro steps (was minutes with HDF5)."""
    from energysim.models.hydrogen import PEMElectrolyzer
    w = world(start_time=0, stop_time=60 * 360, t_macro=60)
    conns = {}
    for i in range(40):
        w.add_model(f'pem{i}', model=PEMElectrolyzer(name=f'pem{i}', p_nominal_mw=3.0), step_size=60)
        w.add_signal(f'sig{i}', 1000.0 + i, step_size=60, units={'y': 'kW'})
        conns[f'sig{i}.y'] = f'pem{i}.P_cmd'
    w.add_connections(conns)
    t0 = time.perf_counter()
    res = w.simulate(pbar=False)
    assert time.perf_counter() - t0 < 10
    assert len(res['pem0']) == 361 and len(res.sims) == 80
    assert res.to_frame().shape == (361, 40 * 10 + 40)


# ---------------------------------------------------------------------------
# Results API
# ---------------------------------------------------------------------------

def test_access_api_and_variables():
    res = _typed_world().simulate(pbar=False, record_all=False)
    assert res.sims == ['m']
    assert res.variables('m') == ['f', 'i', 'b', 's', 'w']
    assert res.variables()[0] == 'm.f'
    assert 'm.f' in res and 'm' in res and 'm.zz' not in res
    s = res['m.f']
    assert s.name == 'm.f' and s.index.name == 'time'
    s.iloc[0] = 999.0                                  # copies: stored data unchanged
    assert res['m.f'].iloc[0] == 0.0
    with pytest.raises(KeyError):
        res['nope.x']
    summ = res.summary()
    assert summ.loc['m.f', 'max'] == 60.0 and np.isnan(summ.loc['m.s', 'last'])   # non-numeric: NaN
    assert res['m.s'].iloc[-1] == 'on'


def test_summary_is_numeric_and_units_are_separate():
    res = _typed_world().simulate(pbar=False, record_all=False)
    summ = res.summary()
    assert list(summ.columns) == ['min', 'max', 'mean', 'last', 'count']
    assert all(pd.api.types.is_numeric_dtype(t) for t in summ.dtypes)
    assert summ.loc['m.f', ['min', 'max']].round().tolist() == [0.0, 60.0]     # no object-dtype row
    assert summ.round(2).loc['m.f', 'max'] == 60.0
    desc = res.describe_variables()
    assert list(desc.columns) == ['unit', 'dtype'] and list(desc.index) == list(summ.index)
    assert desc.loc['m.f', 'dtype'] == 'float64' and desc.loc['m.i', 'dtype'].startswith('int')


def test_to_frame_resample_and_start_datetime():
    w = world(start_time=0, stop_time=3600, t_macro=900, start_datetime='2024-01-01 00:00')
    w.add_signal('a', 't', step_size=900)
    w.add_signal('b', {'type': 'constant', 'value': 2}, step_size=900)
    res = w.simulate(pbar=False)
    s = res['a.y']
    assert isinstance(s.index, pd.DatetimeIndex)
    assert s.index[-1] == pd.Timestamp('2024-01-01 01:00')
    wide = res.to_frame()
    assert list(wide.columns) == ['a.y', 'b.y'] and len(wide) == 5
    rs = res.to_frame(resample='30min')
    assert len(rs) == 3 and rs['a.y'].tolist() == [0.0, 1800.0, 3600.0]
    rs2 = res.to_frame(variables=['a.y'], resample=450, start_datetime='2030-06-01')
    assert rs2.index[0] == pd.Timestamp('2030-06-01') and len(rs2) == 9
    assert rs2['a.y'].iloc[1] == 0.0                     # zero-order hold
    assert res.meta['start_datetime'].startswith('2024-01-01')


def test_units_resolution():
    w = world(start_time=0, stop_time=20, t_macro=10)
    w.add_signal('p', 5.0, step_size=10, units={'y': 'MW'})          # declared
    w.add_model('m', model=_Boom(), step_size=10, outputs=['y'])
    w.add_signal('q', 1.0, step_size=10)
    res = w.simulate(pbar=False)
    u = res.units
    assert u['p.y'] == 'MW'
    assert u['q.y'] is None and u['m.y'] is None
    assert 'unit_system' in res.meta and isinstance(res.meta['unit_system'], dict)


def test_units_suffix_and_connection():
    class A(_Boom):
        def __init__(self):
            self.outputs = ['p_kw', 'T']
            self.inputs = ['u']

        def get_value(self, parameters, time):
            return [1.0 for _ in parameters]

    w = world(start_time=0, stop_time=20, t_macro=10)
    w.add_model('a', model=A(), step_size=10)
    w.add_model('b', model=A(), step_size=10)
    w.add_connections([{'from': 'a.T', 'to': 'b.u', 'from_unit': 'degC', 'to_unit': 'K'}])
    res = w.simulate(pbar=False)
    assert res.units['a.p_kw'] == 'kW'
    assert res.units['a.T'] == 'degC'


def test_kpis_energy_integral():
    w = world(start_time=0, stop_time=3600, t_macro=900)
    w.add_signal('load', 1000.0, step_size=900, units={'y': 'kW'})
    w.add_signal('gen', 2.0, step_size=900, units={'y': 'MW'})
    w.add_signal('x', 't', step_size=900)
    res = w.simulate(pbar=False)
    k = res.kpis()
    assert math.isclose(k.loc['load.y', 'energy_kwh'], 1000.0)
    assert math.isclose(k.loc['load.y', 'energy_kwh_trapz'], 1000.0)
    assert math.isclose(k.loc['gen.y', 'energy_kwh'], 2000.0)
    assert math.isnan(k.loc['x.y', 'energy_kwh'])
    assert k.loc['x.y', 'max'] == 3600.0 and k.loc['x.y', 'mean'] == 1800.0
    assert math.isclose(k.loc['x.y', 'time_weighted_mean'], 1350.0)   # ZOH
    register_kpi('peak_x2', lambda s, unit: float(s.max() * 2))
    try:
        assert 'peak_x2' in available_kpis()
        assert res.kpis(['peak_x2']).loc['x.y', 'peak_x2'] == 7200.0
    finally:
        unregister_kpi('peak_x2')
    with pytest.raises(KeyError):
        res.kpis(['no_such_kpi'])


# ---------------------------------------------------------------------------
# Meta / failure / warnings
# ---------------------------------------------------------------------------

def test_meta_contents_and_hash_stability():
    def build(stop=40):
        w = world(start_time=0, stop_time=stop, t_macro=10, seed=42)
        w.add_signal('s', '2*t', step_size=10)
        return w
    m1 = build().simulate(pbar=False).meta
    m2 = build().simulate(pbar=False).meta
    m3 = build(50).simulate(pbar=False).meta
    for key in ('run_id', 'created', 'status', 'wall_time_s', 'versions', 'scenario', 'scenario_sha256',
                'seed', 'start_datetime', 'coupling', 't_macro', 'units', 'warnings', 'unit_system'):
        assert key in m1, key
    assert m1['versions']['energysim'] == energysim.__version__
    assert {'python', 'numpy', 'pandas', 'pyarrow'} <= set(m1['versions'])
    assert m1['seed'] == 42 and m1['t_macro'] == 10 and m1['coupling'] == 'jacobi'
    assert m1['scenario_sha256'] == m2['scenario_sha256'] != m3['scenario_sha256']
    assert m1['run_id'].endswith(m1['scenario_sha256'][:8])
    json.dumps(m1, default=str)


def test_failed_run_partial_results():
    w = world(start_time=0, stop_time=40, t_macro=10)
    w.add_signal('s', 't', step_size=10)
    w.add_model('boom', model=_Boom(), step_size=10)
    with pytest.raises(RuntimeError, match="boom at 20"):
        w.simulate(pbar=False, record_all=False)
    res = w.last_results
    assert res is not None and res.status == 'failed'
    assert 'boom at 20' in res.meta['error']
    assert res['s.y'].index.tolist() == [0.0, 10.0]   # completed macro steps only


def test_warnings_captured_and_reemitted():
    w = world(start_time=0, stop_time=20, t_macro=10)
    w.add_model('w', model=_Warner(), step_size=10)
    with pytest.warns(UserWarning, match="adapter says hi"):
        res = w.simulate(pbar=False, record_all=False)
    assert any('adapter says hi' in x['message'] for x in res.meta['warnings'])


def test_res_filename_removed():
    with pytest.raises(TypeError):
        world(res_filename='x.h5')
    with pytest.raises(ValueError, match="res_filename"):
        world.from_yaml("world: {start_time: 0, stop_time: 10, t_macro: 10, res_filename: a.h5}\n")
    for name in ('convert_hdf_to_dict', 'convert_to_df', 'convert_to_csv', 'export_to_csv',
                 'record_data', 'create_results_recorder'):
        assert not hasattr(energysim, name)
        assert not hasattr(energysim.utils, name)


def test_yaml_world_keys_roundtrip(tmp_path):
    yml = ("world: {start_time: 0, stop_time: 20, t_macro: 10, seed: 7, start_datetime: '2024-05-01',"
           " out_dir: %s}\nsimulators:\n  - {name: s, type: signal, step_size: 10, signal: '3'}\n"
           % (tmp_path / 'runs').as_posix())
    w = world.from_yaml(yml)
    assert w.seed == 7 and w.out_dir.endswith('runs')
    res = w.simulate(pbar=False)
    assert res.run_dir and os.path.isdir(res.run_dir)
    w2 = world.from_yaml(w.to_yaml())
    assert w2.seed == 7 and str(w2.start_datetime).startswith('2024-05-01')


# ---------------------------------------------------------------------------
# Persistence / run directories / streaming
# ---------------------------------------------------------------------------

def test_no_files_written_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    w = _typed_world()
    w.simulate(pbar=False)
    assert os.listdir(tmp_path) == []


@pytest.mark.parametrize('fmt', ['parquet', 'csv', 'hdf5'])
def test_save_load_roundtrip(fmt, tmp_path):
    if fmt == 'parquet' and not HAVE_PYARROW:
        pytest.skip('pyarrow missing')
    if fmt == 'hdf5' and not HAVE_TABLES:
        pytest.skip('tables missing')
    res = _typed_world(start_datetime='2024-01-01').simulate(pbar=False, record_all=False)
    path = tmp_path / ('r.h5' if fmt == 'hdf5' else 'out')
    res.save(path, format=fmt)
    back = Results.load(path)
    assert back.sims == res.sims
    for sim in res.sims:
        pd.testing.assert_frame_equal(back._raw(sim), res._raw(sim), check_dtype=(fmt != 'csv'))
    assert back['m.s'].tolist() == res['m.s'].tolist()
    assert back['m.b'].dtype == bool
    assert back.units == res.units
    assert back.meta['scenario_sha256'] == res.meta['scenario_sha256']
    assert isinstance(back['m.f'].index, pd.DatetimeIndex)


def test_save_auto_without_pyarrow_falls_back_to_csv(tmp_path, monkeypatch):
    import energysim.results as R
    res = _typed_world().simulate(pbar=False)
    monkeypatch.setattr(R, '_have_pyarrow', lambda: False)
    with pytest.warns(UserWarning, match="pyarrow"):
        res.save(tmp_path / 'auto')
    assert any(f.endswith('.csv') for f in os.listdir(tmp_path / 'auto'))
    with pytest.raises(ImportError, match="pyarrow"):
        res.save(tmp_path / 'pq', format='parquet')


def test_out_dir_run_directory(tmp_path):
    w = _typed_world(out_dir=str(tmp_path / 'runs'))
    res = w.simulate(pbar=False)
    run_dir = res.run_dir
    assert os.path.dirname(run_dir) == str(tmp_path / 'runs')
    name = os.path.basename(run_dir)
    assert name.split('_')[1] == res.meta['scenario_sha256'][:8]
    files = set(os.listdir(run_dir))
    assert {'meta.json', 'scenario.yaml'} <= files
    assert ('m.parquet' if HAVE_PYARROW else 'm.csv') in files
    assert '_parts' not in files
    meta = json.load(open(os.path.join(run_dir, 'meta.json'), encoding='utf-8'))
    assert meta['status'] == 'completed'
    back = Results.load(run_dir)
    assert back['m.s'].tolist() == res['m.s'].tolist()
    # second run with simulate(out_dir=...) -> separate directory
    res2 = w.simulate(pbar=False, out_dir=str(tmp_path / 'other'))
    assert res2.run_dir != run_dir and os.path.isdir(res2.run_dir)


@pytest.mark.skipif(not HAVE_PYARROW, reason='pyarrow missing')
def test_streaming_equals_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(world, '_results_chunk_rows', 3)
    mem = _typed_world().simulate(pbar=False, record_all=True)
    streamed = _typed_world(out_dir=str(tmp_path)).simulate(pbar=False, record_all=True)
    assert len(mem['m']) == 5
    for sim in mem.sims:
        pd.testing.assert_frame_equal(streamed[sim], mem[sim])


def test_failed_run_writes_meta(tmp_path):
    w = world(start_time=0, stop_time=40, t_macro=10, out_dir=str(tmp_path))
    w.add_model('boom', model=_Boom(), step_size=10)
    with pytest.raises(RuntimeError):
        w.simulate(pbar=False)
    meta = json.load(open(os.path.join(w.last_results.run_dir, 'meta.json'), encoding='utf-8'))
    assert meta['status'] == 'failed'


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------

def test_concat_run_dimension(tmp_path):
    runs = []
    for gain in (1.0, 2.0, 3.0):
        w = world(start_time=0, stop_time=20, t_macro=10)
        w.add_signal('s', f'{gain}*t', step_size=10)
        runs.append(w.simulate(pbar=False))
    ens = Results.concat(runs, keys=[{'gain': 1.0}, {'gain': 2.0}, {'gain': 3.0}])
    assert ens.is_ensemble and len(ens.runs) == 3
    assert ens.params['gain'].tolist() == [1.0, 2.0, 3.0]
    assert ens.run(1)['s.y'].tolist() == [0.0, 20.0, 40.0]
    wide = ens['s.y']
    assert list(wide.columns) == [0, 1, 2] and wide[2].iloc[-1] == 60.0
    assert ens.kpis(['max']).loc[(2, 's.y'), 'max'] == 60.0
    assert ens.to_frame().shape == (3, 3)
    ens2 = Results.concat(runs[:2], keys=['lo', 'hi'])
    assert ens2.run('hi') is runs[1] and ens2.run_keys == ['lo', 'hi']
    if HAVE_PYARROW:
        ens.save(tmp_path / 'ens')
        back = Results.load(tmp_path / 'ens')
        assert back.is_ensemble and back.params['gain'].tolist() == [1.0, 2.0, 3.0]
        assert back.run(2)['s.y'].tolist() == [0.0, 30.0, 60.0]
    single = runs[0]
    assert single.runs == [single] and single.run(0) is single


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def test_dashboard_from_results_no_browser(tmp_path, monkeypatch):
    import webbrowser
    opened = []
    monkeypatch.setattr(webbrowser, 'open', lambda *a, **k: opened.append(a))
    w = world(start_time=0, stop_time=4 * 3600, t_macro=60)
    w.add_signal('s', 'sin(t/600)', step_size=60)
    res = w.simulate(pbar=False)
    path = res.dashboard(tmp_path / 'd.html', max_points=100)
    assert not opened
    html_text = open(path, encoding='utf-8').read()
    assert 'Time (h)' in html_text and 'cdn.plot.ly' in html_text
    start = html_text.index('const SIMS = ') + len('const SIMS = ')
    sims = json.loads(html_text[start:html_text.index(';\n', start)])
    xs = sims[0]['xs']['y']
    assert len(xs) <= 102 and len(sims[0]['data']['y']) == len(xs)
    assert max(sims[0]['data']['y']) > 0.99 and min(sims[0]['data']['y']) < -0.99   # peaks kept
    assert xs[-1] == 4.0
    res.dashboard(tmp_path / 'e.html', open=True)
    assert len(opened) == 1


def test_dashboard_dict_input_not_mutated(tmp_path):
    from energysim.dashboard import generate_dashboard
    df = pd.DataFrame({'time': [0.0, 3600.0, 7200.0], 'v': [1.0, 2.0, 3.0], 'state': ['a', 'b', 'c']})
    orig = df.copy()
    path = generate_dashboard({'x': df}, output_path=tmp_path / 'd.html')
    pd.testing.assert_frame_equal(df, orig)
    assert 'Time (h)' in open(path, encoding='utf-8').read()


def test_dashboard_offline_and_escaping(tmp_path):
    from energysim.dashboard import generate_dashboard
    hostile = '</script><script>alert(1)</script>'
    df = pd.DataFrame({hostile: [1.0, 2.0]}, index=pd.Index([0.0, 1.0], name='time'))
    res = Results({hostile: df})
    try:
        import plotly  # noqa: F401
        ctx = warnings.catch_warnings()
    except ImportError:
        ctx = pytest.warns(UserWarning, match="plotly")
    with ctx:
        path = generate_dashboard(res, output_path=tmp_path / 'off.html', offline=True, title=hostile)
    text = open(path, encoding='utf-8').read()
    assert '<script>alert(1)' not in text
    assert text.count('</script>') == 2
    try:
        import plotly  # noqa: F401
        assert 'cdn.plot.ly/plotly-2.35.0.min.js" charset' not in text
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_out_csv_and_dashboard(tmp_path, capsys, monkeypatch):
    import webbrowser
    monkeypatch.setattr(webbrowser, 'open', lambda *a, **k: None)
    from energysim.cli import main as cli_main
    f = tmp_path / 's.yaml'
    f.write_text("world: {start_time: 0, stop_time: 20, t_macro: 10}\nsimulators:\n"
                 "  - {name: s, type: signal, step_size: 10, signal: '3'}\n")
    out = tmp_path / 'runs'
    dash = tmp_path / 'dash.html'
    assert cli_main(['run', str(f), '--out', str(out), '--to-csv', '--no-pbar',
                     '--dashboard-path', str(dash)]) == 0
    printed = capsys.readouterr().out
    run_dirs = os.listdir(out)
    assert len(run_dirs) == 1
    run_dir = out / run_dirs[0]
    assert str(run_dir) in printed and 's.y' in printed
    assert (run_dir / 'meta.json').exists() and (run_dir / 'csv' / 's.csv').exists()
    assert dash.exists()
    with pytest.raises(SystemExit):
        cli_main(['run', str(f), '--res-filename', 'x.h5'])

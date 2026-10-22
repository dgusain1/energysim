# -*- coding: utf-8 -*-
"""
Test suite for the refactored energysim architecture.

Tests cover:
  - SimulatorAdapter ABC contract
  - SimEntry dataclass
  - Adapter inheritance verification
  - CSV, signal, and world orchestration (no FMU hardware needed)
  - Pre-parsed connections
  - Uniform advance() / cleanup() calls
"""

import sys
import os

# Ensure the source tree is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from energysim.base import SimulatorAdapter, SimEntry
import numpy as np
import pytest


# =====================================================================
# 1. ABC contract tests
# =====================================================================


class _MinimalAdapter(SimulatorAdapter):
    """Concrete adapter implementing only the required abstract methods."""

    def __init__(self):
        self.step_calls = []
        self._values = {}

    def step(self, time):
        self.step_calls.append(time)

    def get_value(self, parameters, time):
        return [self._values.get(p, 0.0) for p in parameters]

    def set_value(self, parameters, values):
        for p, v in zip(parameters, values):
            self._values[p] = v


def test_abc_cannot_instantiate():
    """SimulatorAdapter itself should not be instantiable."""
    with pytest.raises(TypeError):
        SimulatorAdapter()


def test_minimal_adapter_satisfies_abc():
    """A concrete adapter with all abstract methods can be created."""
    a = _MinimalAdapter()
    assert isinstance(a, SimulatorAdapter)


def test_default_advance_loops_step():
    """The default advance() should call step() at step_size intervals."""
    a = _MinimalAdapter()
    a.step_size = 10.0
    a.advance(0, 50)
    assert len(a.step_calls) == 5
    np.testing.assert_allclose(a.step_calls, [0, 10, 20, 30, 40], atol=1e-12)


def test_default_set_parameters_delegates_to_set_value():
    """set_parameters() should call set_value() by default."""
    a = _MinimalAdapter()
    a.set_parameters({'x': 1.0, 'y': 2.0})
    assert a.get_value(['x', 'y'], 0) == [1.0, 2.0]


def test_default_init_and_cleanup_are_noop():
    """init() and cleanup() should not raise when not overridden."""
    a = _MinimalAdapter()
    a.init()
    a.cleanup()


# =====================================================================
# 2. SimEntry dataclass tests
# =====================================================================

def test_sim_entry_creation():
    a = _MinimalAdapter()
    entry = SimEntry(sim_type='test', adapter=a, step_size=1.0, outputs=['x'])
    assert entry.sim_type == 'test'
    assert entry.adapter is a
    assert entry.variable_step is False
    assert entry.pf_mode is None


# =====================================================================
# 3. Adapter inheritance tests
# =====================================================================

def test_csv_adapter_is_simulator_adapter():
    from energysim.csv_adapter import csv_simulator
    assert issubclass(csv_simulator, SimulatorAdapter)


def test_signal_adapter_is_simulator_adapter():
    from energysim.signalAdapter import signal_adapter
    assert issubclass(signal_adapter, SimulatorAdapter)


def test_pp_adapter_is_simulator_adapter():
    from energysim.ppAdapter import pp_adapter
    assert issubclass(pp_adapter, SimulatorAdapter)


def test_pypsa_adapter_is_simulator_adapter():
    from energysim.pypsaAdapter import pypsa_adapter
    assert issubclass(pypsa_adapter, SimulatorAdapter)


def test_cs_adapter_is_simulator_adapter():
    from energysim.csAdapter import FmuCsAdapter
    assert issubclass(FmuCsAdapter, SimulatorAdapter)


def test_me_adapter_is_simulator_adapter():
    from energysim.meAdapter import FmuMeAdapter
    assert issubclass(FmuMeAdapter, SimulatorAdapter)


def test_py_adapter_is_simulator_adapter():
    from energysim.pyScriptAdapter import py_adapter
    assert issubclass(py_adapter, SimulatorAdapter)


# =====================================================================
# 4. CSV adapter functional tests
# =====================================================================

@pytest.fixture
def csv_file(tmp_path):
    """Create a simple CSV data file in a temp directory."""
    csv_path = tmp_path / 'data.csv'
    csv_path.write_text('time,temp,pressure\n0,20,100\n900,21,101\n1800,22,102\n')
    return str(csv_path)


def test_csv_adapter_step_and_get_value(csv_file):
    from energysim.csv_adapter import csv_simulator
    c = csv_simulator('test_csv', csv_file, step_size=900, outputs=['temp', 'pressure'])
    c.init()
    c.step(0)
    vals = c.get_value(['temp'], 0)
    assert vals == [20]
    vals2 = c.get_value(['temp', 'pressure'], 900)
    assert vals2 == [21, 101]


# =====================================================================
# 5. Signal adapter functional tests
# =====================================================================

def test_signal_adapter_step_and_get_value():
    from energysim.signalAdapter import signal_adapter
    s = signal_adapter(signal_name='sig1', signal=lambda t: t * 2)
    s.init()
    s.step(5)
    vals = s.get_value(['y'], 5)
    assert vals == [10]


def test_signal_adapter_returns_list_for_tuple():
    from energysim.signalAdapter import signal_adapter
    s = signal_adapter(signal_name='sig2', signal=lambda t: (t, t + 1))
    vals = s.get_value(['y'], 3)
    assert vals == [3, 4]


# =====================================================================
# 6. World orchestration tests (csv + signal, no FMU)
# =====================================================================

@pytest.fixture
def world_with_csv_and_signal(csv_file, tmp_path):
    """Set up a world with a CSV source and a signal source connected."""
    from energysim import world
    res_file = str(tmp_path / 'test_res.h5')
    w = world(start_time=0, stop_time=1800, t_macro=900, logging=False, res_filename=res_file)
    w.add_simulator(sim_type='csv', sim_name='csv1', sim_loc=csv_file,
                    outputs=['temp'], step_size=900)
    w.add_signal('sig1', signal=lambda t: t * 0.1, step_size=900)
    return w


def test_world_uses_sim_entry(world_with_csv_and_signal):
    w = world_with_csv_and_signal
    for name, entry in w.simulator_dict.items():
        assert isinstance(entry, SimEntry), f"{name} should be a SimEntry"


def test_world_simulate_no_connections(world_with_csv_and_signal):
    """simulate() should complete even with no connections."""
    w = world_with_csv_and_signal
    w._parsed_connections = []  # no connections
    result = w.simulate(pbar=False, record_all=False)
    assert result == 'Done'


def test_world_simulate_with_connection(csv_file, tmp_path):
    """Set up a connection and simulate — signal feeds into csv (which ignores set_value)."""
    from energysim import world
    res_file = str(tmp_path / 'test_connected.h5')
    w = world(start_time=0, stop_time=1800, t_macro=900, logging=False, res_filename=res_file)
    w.add_simulator(sim_type='csv', sim_name='csv1', sim_loc=csv_file,
                    outputs=['temp'], step_size=900)
    w.add_signal('sig1', signal=lambda t: t * 0.1, step_size=900)
    # Signal output → csv input (csv ignores set_value, but the pipeline should not crash)
    w.add_connections({'sig1.y': 'csv1.temp'})
    result = w.simulate(pbar=False, record_all=False)
    assert result == 'Done'


def test_world_results(csv_file, tmp_path):
    from energysim import world
    res_file = str(tmp_path / 'test_results.h5')
    w = world(start_time=0, stop_time=1800, t_macro=900, logging=False, res_filename=res_file)
    w.add_simulator(sim_type='csv', sim_name='csv1', sim_loc=csv_file,
                    outputs=['temp'], step_size=900)
    w.add_signal('sig1', signal=lambda t: t * 0.1, step_size=900)
    w._parsed_connections = []
    w.simulate(pbar=False, record_all=False)
    res = w.results(to_csv=False, dashboard=False)
    assert 'csv1' in res
    assert 'sig1' in res
    assert 'temp' in res['csv1'].columns


def test_world_record_all_true(csv_file, tmp_path):
    """record_all=True should capture more data points per step."""
    from energysim import world
    res_file = str(tmp_path / 'test_rec_all.h5')
    w = world(start_time=0, stop_time=1800, t_macro=900, logging=False, res_filename=res_file)
    w.add_simulator(sim_type='csv', sim_name='csv1', sim_loc=csv_file,
                    outputs=['temp'], step_size=900)
    w._parsed_connections = []
    w.simulate(pbar=False, record_all=True)
    res = w.results(to_csv=False, dashboard=False)
    # With record_all and step_size == t_macro, each macro step yields 1 record
    assert len(res['csv1']) >= 2


# =====================================================================
# 7. Pre-parsed connections tests
# =====================================================================

def test_parsed_connections_single(csv_file, tmp_path):
    from energysim import world
    res_file = str(tmp_path / 'test_parsed.h5')
    w = world(start_time=0, stop_time=1800, t_macro=900, res_filename=res_file)
    w.add_simulator(sim_type='csv', sim_name='csv1', sim_loc=csv_file,
                    outputs=['temp'], step_size=900)
    w.add_signal('sig1', signal=lambda t: t, step_size=900)
    w.add_connections({'sig1.y': 'csv1.temp'})
    assert len(w._parsed_connections) == 1
    out, inp = w._parsed_connections[0]
    assert out == ('sig1', 'y', 'sig1.y')
    assert inp == ('csv1', 'temp')


def test_parsed_connections_fan_out(csv_file, tmp_path):
    from energysim import world
    res_file = str(tmp_path / 'test_fanout.h5')
    w = world(start_time=0, stop_time=1800, t_macro=900, res_filename=res_file)
    w.add_simulator(sim_type='csv', sim_name='csv1', sim_loc=csv_file,
                    outputs=['temp'], step_size=900)
    w.add_simulator(sim_type='csv', sim_name='csv2', sim_loc=csv_file,
                    outputs=['temp'], step_size=900)
    w.add_signal('sig1', signal=lambda t: t, step_size=900)
    w.add_connections({'sig1.y': ('csv1.temp', 'csv2.temp')})
    assert len(w._parsed_connections) == 1
    out, inp = w._parsed_connections[0]
    assert isinstance(inp, list)
    assert len(inp) == 2


def test_parsed_connections_fan_in(csv_file, tmp_path):
    from energysim import world
    res_file = str(tmp_path / 'test_fanin.h5')
    w = world(start_time=0, stop_time=1800, t_macro=900, res_filename=res_file)
    w.add_simulator(sim_type='csv', sim_name='csv1', sim_loc=csv_file,
                    outputs=['temp'], step_size=900)
    w.add_signal('sig1', signal=lambda t: t, step_size=900)
    w.add_signal('sig2', signal=lambda t: t * 2, step_size=900)
    w.add_connections({('sig1.y', 'sig2.y'): 'csv1.temp'})
    assert len(w._parsed_connections) == 1
    out, inp = w._parsed_connections[0]
    assert isinstance(out, tuple)
    assert len(out) == 2


# =====================================================================
# 8. PyScript adapter tests
# =====================================================================

def test_pyscript_adapter(tmp_path):
    """py_adapter should import a module and call its function on step()."""
    # Create a simple Python module
    mod_dir = tmp_path / 'scripts'
    mod_dir.mkdir()
    (mod_dir / 'my_module.py').write_text(
        'def my_func(inputs):\n'
        '    x = inputs.get("x", 0)\n'
        '    return [x * 2]\n'
    )
    from energysim.pyScriptAdapter import py_adapter
    pa = py_adapter(
        script_name='my_module.my_func',
        script_loc=str(mod_dir),
        inputs=['x'],
        outputs=['result']
    )
    pa.set_value(['x'], [5])
    pa.step(0)
    assert pa.get_value(['result'], 0) == [10]


# =====================================================================
# 9. Setup.py dependencies check
# =====================================================================

def test_setup_optional_deps():
    """Verify h5py was removed and fmpy/pandapower/pypsa are optional."""
    setup_path = os.path.join(os.path.dirname(__file__), 'src', 'setup.py')
    with open(setup_path) as f:
        content = f.read()
    assert 'h5py' not in content
    assert "extras_require" in content
    assert "'fmu'" in content or '"fmu"' in content


# =====================================================================
# 10. Review improvements and bug fixes tests
# =====================================================================

def test_setup_license_matches_pyproject():
    """Verify setup.py specifies GPL-3.0-or-later matching pyproject.toml."""
    setup_path = os.path.join(os.path.dirname(__file__), 'src', 'setup.py')
    with open(setup_path) as f:
        content = f.read()
    assert "license='GPL-3.0-or-later'" in content


def test_csv_adapter_short_file(tmp_path):
    """Short CSV files (<=3 rows) should validate without autocorr NaN errors."""
    from energysim.csv_adapter import csv_simulator
    csv_path = tmp_path / 'short.csv'
    csv_path.write_text('time,temp\n0,10\n1,15\n')
    c = csv_simulator('short_csv', str(csv_path), step_size=1, outputs=['temp'])
    assert c.get_value(['temp'], 0) == [10]
    assert c.get_value(['temp'], 1) == [15]


def test_pyscript_adapter_parameter_filtering(tmp_path):
    """py_adapter should return only requested parameters and raise on missing vars."""
    from energysim.pyScriptAdapter import py_adapter
    from energysim.base import SimulatorVariableError

    mod_dir = tmp_path / 'multi_scripts'
    mod_dir.mkdir()
    (mod_dir / 'calc.py').write_text(
        'def run(inputs):\n'
        '    return [10.0, 20.0]\n'
    )
    pa = py_adapter(
        script_name='calc.run',
        script_loc=str(mod_dir),
        inputs=[],
        outputs=['out1', 'out2']
    )
    pa.step(0)
    assert pa.get_value(['out2'], 0) == [20.0]
    assert pa.get_value(['out1', 'out2'], 0) == [10.0, 20.0]
    with pytest.raises(SimulatorVariableError):
        pa.get_value(['nonexistent'], 0)


def test_convert_hdf_to_dict_to_csv_returns_dict(tmp_path):
    """convert_hdf_to_dict with to_csv=True must return the dictionary of DataFrames."""
    from energysim.utils import convert_hdf_to_dict, create_results_recorder, record_data
    res_h5 = str(tmp_path / 'test_export.h5')
    sim_dict = {'sim1': ['val']}
    create_results_recorder(res_h5, sim_dict)
    record_data(res_h5, {'sim1': np.array([[0.0, 42.0]])})

    export_dir = str(tmp_path / 'my_res')
    res = convert_hdf_to_dict(res_h5, sim_dict=sim_dict, to_csv=True, res_folder_name=export_dir)
    assert isinstance(res, dict)
    assert 'sim1' in res
    assert 'val' in res['sim1'].columns
    assert os.path.exists(os.path.join(export_dir, 'res_sim1.csv'))


def test_dashboard_escapes_html_title():
    """_generate_html should escape special characters in the title."""
    from energysim.dashboard import _generate_html
    malicious_title = 'energysim</title><script>alert("xss")</script>'
    html_text = _generate_html([], title=malicious_title)
    assert '<script>alert("xss")</script>' not in html_text
    assert '&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;' in html_text


def test_world_pep8_alias():
    """Verify World alias is available and identical to world."""
    from energysim import World, world
    assert World is world


def test_parameter_sweep_sim_entry_compatibility():
    """Verify Sweep works with modern SimEntry dataclasses."""
    from energysim import world, Sweep
    w = world(start_time=0, stop_time=100, t_macro=10)
    w.add_signal('sig1', signal=lambda t: t, step_size=10)
    sweep_info = {
        'parameters': ['sig1.y'],
        'bounds': [[0, 10, 2]],
        'output_parameter': 'sig1.y'
    }
    s = Sweep(w, sweep_info, kind='single')
    assert s.powerflow_exists is False
    assert s.simulators_dict['sig1'].sim_type == 'signal'


def test_package_public_api_and_modular_imports():
    """Verify clean modular exports and backwards compatibility."""
    import energysim
    assert hasattr(energysim, '__version__')
    assert energysim.__version__ == '2.1.9.1'
    assert hasattr(energysim, 'world')
    assert hasattr(energysim, 'World')
    assert hasattr(energysim, 'Sweep')
    assert hasattr(energysim, 'SimulatorAdapter')
    assert hasattr(energysim, 'SimEntry')

    from energysim.world import world as world_mod, World as World_mod
    assert world_mod is energysim.world
    assert World_mod is energysim.World


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

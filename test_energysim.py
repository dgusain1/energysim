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

from energysim.base import SimulatorAdapter, SimEntry
import sys
import os
import numpy as np
import pytest

# Ensure the source tree is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

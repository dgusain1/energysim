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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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

def test_pyproject_optional_deps():
    """Verify h5py was removed and fmpy/pandapower/pypsa/opendss are optional."""
    pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    with open(pyproject_path) as f:
        content = f.read()
    assert 'h5py' not in content
    assert "optional-dependencies" in content
    assert "fmu =" in content
    assert "opendss =" in content


# =====================================================================
# 10. Review improvements and bug fixes tests
# =====================================================================

def test_pyproject_license_specification():
    """Verify pyproject.toml specifies GPL-3.0-or-later."""
    pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    with open(pyproject_path) as f:
        content = f.read()
    assert 'license = "GPL-3.0-or-later"' in content


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


def test_validation_empty_world():
    """Validating a world with no simulators should fail."""
    from energysim import world, ValidationError
    w = world()
    report = w.validate(raise_on_error=False)
    assert not report.is_valid
    assert any("No simulators registered" in err for err in report.errors)
    with pytest.raises(ValidationError):
        w.validate(raise_on_error=True)


def test_validation_step_size_divisibility():
    """Warn when simulator step size does not divide t_macro."""
    from energysim import world
    # 3.0: the span must be an integer multiple of t_macro (else a validation error)
    w = world(start_time=0, stop_time=120, t_macro=60)
    w.add_signal('sig1', signal=lambda t: t, step_size=45.0)  # 60 % 45 != 0
    report = w.validate(raise_on_error=False)
    assert report.is_valid  # warning, not error
    assert any("does not divide" in warn for warn in report.warnings)


def test_validation_algebraic_loop_advice():
    """Warn when an algebraic cycle is present."""
    from energysim import world
    w = world(start_time=0, stop_time=100, t_macro=10, coupling='gauss-seidel')
    a1 = _MinimalAdapter()
    a2 = _MinimalAdapter()
    from energysim.base import SimEntry
    w.simulator_dict['sim1'] = SimEntry('test', a1, step_size=10, outputs=['y'], inputs=['u'])
    w.simulator_dict['sim2'] = SimEntry('test', a2, step_size=10, outputs=['y'], inputs=['u'])
    w.add_connections({'sim1.y': 'sim2.u', 'sim2.y': 'sim1.u'})
    report = w.validate(raise_on_error=False)
    assert any("Algebraic feedback loop" in warn for warn in report.warnings)


def test_connection_scaling_and_offset():
    """Verify that scale and offset are correctly applied during variable exchange."""
    from energysim import world
    w = world(start_time=0, stop_time=10, t_macro=2)
    a1 = _MinimalAdapter()
    a2 = _MinimalAdapter()
    a1.set_value(['y'], [100.0])

    from energysim.base import SimEntry
    w.simulator_dict['src'] = SimEntry('test', a1, step_size=2, outputs=['y'], inputs=[])
    w.simulator_dict['dst'] = SimEntry('test', a2, step_size=2, outputs=[], inputs=['u'])

    # Apply y * 0.01 + 5 = 100 * 0.01 + 5 = 6.0
    w.add_connections({'src.y': ('dst.u', {'scale': 0.01, 'offset': 5.0})})
    w.exchange_variable_values(0)

    assert a2.get_value(['u'], 0) == [6.0]


def test_yaml_export_and_import_roundtrip(tmp_path):
    """Verify world can be exported to YAML and reloaded identically."""
    from energysim import world
    w = world(start_time=0, stop_time=3600, t_macro=300, coupling='jacobi')
    # 3.0: only expression/structured signals are serialisable (no eval of lambdas)
    w.add_signal('solar', signal="100.0", step_size=300)
    w.add_signal('load', signal={'type': 'constant', 'value': 50.0}, step_size=300)
    w.add_connections([
        {'from': 'solar.y', 'to': 'load.y', 'scale': 0.5, 'offset': 10.0}
    ])

    yaml_file = tmp_path / 'scenario.yaml'
    w.to_yaml(str(yaml_file))

    assert yaml_file.exists()
    w_loaded = world.from_yaml(str(yaml_file))

    assert w_loaded.start_time == 0
    assert w_loaded.stop_time == 3600
    assert w_loaded.exchange == 300
    assert 'solar' in w_loaded.simulator_dict
    assert 'load' in w_loaded.simulator_dict
    assert len(w_loaded._parsed_connections) == 1
    # Check that scale and offset were preserved
    conn = w_loaded._parsed_connections[0]
    assert conn.scale == 0.5
    assert conn.offset == 10.0


def test_cli_validate_and_run(tmp_path):
    """Test CLI commands validate and run with temporary scenario."""
    from energysim.cli import main as cli_main
    yaml_file = tmp_path / 'test_cli.yaml'
    yaml_content = """
world:
  start_time: 0
  stop_time: 100
  t_macro: 10
  coupling: jacobi
  res_filename: test_cli.h5
simulators:
  - name: s1
    type: signal
    step_size: 10
    options:
      signal: "lambda t: t"
connections: []
"""
    yaml_file.write_text(yaml_content)

    # Test validate
    ret_val = cli_main(["validate", str(yaml_file)])
    assert ret_val == 0

    # Test run --validate-only
    ret_run_val = cli_main(["run", str(yaml_file), "--validate-only"])
    assert ret_run_val == 0


def test_opendss_adapter_contract():
    """Verify opendss_adapter satisfies SimulatorAdapter."""
    from energysim.opendssAdapter import opendss_adapter
    from energysim.base import SimulatorAdapter
    assert issubclass(opendss_adapter, SimulatorAdapter)
    adapter = opendss_adapter('dss1', 'dummy.dss', inputs=['Load.L1.kW'], outputs=['Bus.B1.vm_pu'])
    assert adapter.dss_name == 'dss1'
    vars_dict = adapter.get_available_variables()
    assert vars_dict['inputs'] == ['Load.L1.kW']
    assert vars_dict['outputs'] == ['Bus.B1.vm_pu']


def test_pypsa_pattern_b_limitation_doc():
    """Verify PyPSA adapter documents Pattern B limitation and supports mode."""
    from energysim.pypsaAdapter import pypsa_adapter
    assert "Pattern B" in pypsa_adapter.__doc__


def test_mode_standardization_and_rejection():
    """Verify that power flow adapters standardize on 'ac' and 'dc' and reject unsupported modes."""
    from energysim.opendssAdapter import opendss_adapter
    from energysim.pfAdapter import pf_adapter

    # OpenDSS mode normalization
    dss_ac = opendss_adapter('dss1', 'dummy.dss', mode='ac')
    assert dss_ac.mode == 'ac'
    with pytest.raises(ValueError, match="does not provide a DC power flow"):
        opendss_adapter('dss2', 'dummy.dss', mode='dc')
    with pytest.raises(ValueError, match="only supports 'ac' and 'dc'"):
        opendss_adapter('dss3', 'dummy.dss', mode='unsupported')

    # PowerFactory mode normalization & rejection of shc / rms
    pf_ac = pf_adapter('proj1', mode='ac')
    assert pf_ac.mode == 'ac'
    pf_dc = pf_adapter('proj2', mode='dc')
    assert pf_dc.mode == 'dc'
    pf_legacy = pf_adapter('proj3', pf='ldf')
    assert pf_legacy.mode == 'ac'
    with pytest.raises(ValueError, match="only supports 'ac' and 'dc'"):
        pf_adapter('proj4', mode='shc')
    with pytest.raises(ValueError, match="only supports 'ac' and 'dc'"):
        pf_adapter('proj5', mode='rms')


def test_glass_box_get_adapter_and_get_model(tmp_path):
    """Verify world.get_adapter() and world.get_model() provide direct glass-box access."""
    from energysim import world
    csv_file = tmp_path / "glassbox.csv"
    csv_file.write_text("time,temp\n0,20\n10,21\n20,22\n")

    w = world(start_time=0, stop_time=20, t_macro=10)
    w.add_signal('sig', lambda t: [42.0])
    w.add_csv('csv_data', str(csv_file), step_size=10, outputs=['temp'], delimiter=',')

    # Test get_adapter
    sig_adapter = w.get_adapter('sig')
    assert hasattr(sig_adapter, 'step')
    csv_adapter = w.get_adapter('csv_data')
    assert hasattr(csv_adapter, 'step')

    with pytest.raises(KeyError, match="not registered"):
        w.get_adapter('non_existent')

    # Test get_model
    sig_model = w.get_model('sig')
    assert callable(sig_model)
    assert sig_model(0) == [42.0]

    csv_df = w.get_model('csv_data')
    import pandas as pd
    assert isinstance(csv_df, pd.DataFrame)
    assert 'temp' in csv_df.columns


def test_unit_conversion_utilities():
    """Verify get_unit_scale and infer_variable_unit."""
    import math
    from energysim.utils import get_unit_scale, infer_variable_unit

    assert math.isclose(get_unit_scale('W', 'kW'), 1e-3)
    assert math.isclose(get_unit_scale('kW', 'MW'), 1e-3)
    assert math.isclose(get_unit_scale('MW', 'kW'), 1000.0)
    assert math.isclose(get_unit_scale('W', 'MW'), 1e-6)
    assert math.isclose(get_unit_scale('kvar', 'mvar'), 1e-3)
    assert math.isclose(get_unit_scale('V', 'kV'), 1e-3)

    # Incompatible units
    with pytest.raises(ValueError, match="Incompatible unit conversion"):
        get_unit_scale('W', 'kV')

    # Inferred units
    assert infer_variable_unit('Gen.p_mw') == 'MW'
    assert infer_variable_unit('Load.kW') == 'kW'
    assert infer_variable_unit('Bus.vm_pu') == 'pu'
    # 3.0: the simulator-type fallback only applies to power-like names
    assert infer_variable_unit('unknown', sim_type='fmu') is None
    assert infer_variable_unit('P', sim_type='fmu') == 'W'


def test_world_connections_unit_scaling(tmp_path):
    """Verify that world auto-scales or explicitly scales variables based on declared/inferred units."""
    import math
    from energysim import world
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("time,p_mw\n0,1.5\n10,2.0\n")

    w = world(start_time=0, stop_time=10, t_macro=10, power_unit='kW')
    # 3.0: 'y' is not a power-like name, so its unit must be declared explicitly
    w.add_signal('sig_kw', lambda t: [500.0], power_unit='kW', units={'y': 'kW'})
    w.add_csv('csv_grid', str(csv_file), step_size=10, outputs=['p_mw'], power_unit='MW')

    # Connect sig_kw (500 kW) to csv_grid.p_mw (MW)
    # Automatic conversion: kW -> MW should give scale = 1e-3
    w.add_connections({'sig_kw.y': 'csv_grid.p_mw'})
    parsed_conn = w._parsed_connections[0]
    assert math.isclose(parsed_conn.scale, 1e-3)
    assert parsed_conn.from_unit.lower() == 'kw'
    assert parsed_conn.to_unit.lower() == 'mw'


def test_model_parameter_and_domain_convenience_kwargs(tmp_path):
    """Verify that add_simulator and convenience methods accept 'model' and domain-specific kwargs."""
    from energysim import world
    from energysim.utils import resolve_model_reference

    csv_file = tmp_path / "test_model.csv"
    csv_file.write_text("time,val\n0,10\n10,20\n")

    w = world(start_time=0, stop_time=10, t_macro=10)

    # 1. Generic add_simulator using model=...
    w.add_simulator('csv', 'csv_generic', model=str(csv_file), outputs=['val'])
    assert w.simulator_dict['csv_generic'].model == str(csv_file)
    assert w.simulator_dict['csv_generic'].sim_loc == str(csv_file)

    # Backward compatibility: sim_loc property setter
    w.simulator_dict['csv_generic'].sim_loc = "updated_loc"
    assert w.simulator_dict['csv_generic'].model == "updated_loc"

    # 2. add_csv with domain-specific csv_path
    w.add_csv('csv_domain', csv_path=str(csv_file), outputs=['val'])
    assert w.simulator_dict['csv_domain'].model == str(csv_file)

    # 3. add_simulator with callable model
    sig_fn = lambda t: [99.0]
    w.add_simulator('signal', 'sig_model', model=sig_fn)
    assert w.simulator_dict['sig_model'].model == sig_fn

    # 4. resolve_model_reference utility
    assert resolve_model_reference(123) == 123
    assert resolve_model_reference(str(csv_file)) == str(csv_file)
    # Dotted import resolution requires explicit opt-in (3.0)
    with pytest.raises(ValueError, match="allow_import"):
        resolve_model_reference('math.cos')
    res_fn = resolve_model_reference('math.cos', allow_import=True)
    assert callable(res_fn)
    assert res_fn(0) == 1.0


def test_yaml_model_field_roundtrip(tmp_path):
    """Verify YAML serialization and loading using 'model'."""
    from energysim import world

    csv_file = tmp_path / "yaml_model.csv"
    csv_file.write_text("time,flow\n0,1\n10,2\n")

    yaml_str = f"""
world:
  start_time: 0
  stop_time: 20
  t_macro: 10
simulators:
  - name: source
    type: csv
    model: "{csv_file.as_posix()}"
    step_size: 10
    outputs: [flow]
connections: []
"""
    w = world.from_yaml(yaml_str)
    assert 'source' in w.simulator_dict
    assert w.simulator_dict['source'].model == csv_file.as_posix()

    # to_dict roundtrip includes model
    d = w.to_dict()
    assert d['simulators'][0]['model'] == csv_file.as_posix()


def test_end_to_end_pandapower_signal_cosim(tmp_path):
    """Verify an end-to-end co-simulation coupling Pandapower with a dynamic signal."""
    pytest.importorskip("pandapower")
    import pandapower as pp
    from energysim import world

    # Create an in-memory network with named elements
    net = pp.create_empty_network(name='test_microgrid')
    b1 = pp.create_bus(net, vn_kv=110, name='Bus1')
    b2 = pp.create_bus(net, vn_kv=110, name='Bus2')
    pp.create_ext_grid(net, b1, name='SlackGrid')
    pp.create_line(net, b1, b2, length_km=10, std_type='149-AL1/24-ST1A 110.0', name='Line1')
    pp.create_load(net, b2, p_mw=20, q_mvar=10, name='Load1')

    res_file = tmp_path / "cosim_res.h5"
    w = world(start_time=0, stop_time=20, t_macro=10, power_unit='kW', res_filename=str(res_file))

    # Add powerflow simulator with in-memory net
    w.add_powerflow('grid', net=net, inputs=['Load1.p_mw'], outputs=['Bus2.vm_pu'], step_size=10, mode='ac')

    # Add dynamic load controller signal (in kW: 15,000 kW = 15 MW)
    w.add_signal('load_signal', lambda t: [15000.0], step_size=10, units={'y': 'kW'})

    # Connect signal to grid load (automatic conversion kW -> MW)
    w.add_connections({'load_signal.y': 'grid.Load1.p_mw'})

    # Simulate and check results
    w.simulate(pbar=False)
    results = w.results(dashboard=False)

    assert 'grid' in results
    assert 'load_signal' in results
    vm_pu_values = results['grid']['Bus2.vm_pu'].dropna().tolist()
    assert len(vm_pu_values) >= 2
    # Verify realistic voltage range for 110 kV bus
    for v in vm_pu_values:
        assert 0.9 <= v <= 1.1


def test_battery_fleet_adapter():
    """Verify BatteryFleetAdapter vectorized operations, indexing, and aggregations."""
    import numpy as np
    from energysim.container import BatteryFleetAdapter

    fleet = BatteryFleetAdapter(
        name='ev_fleet',
        size=50,
        capacity_kwh=50.0,
        max_charge_kw=11.0,
        max_discharge_kw=11.0,
        initial_soc=0.5,
        eta_charge=0.9,
    )
    assert fleet.size == 50

    # 1. Test set_value broadcast
    fleet.set_value(['p_cmd'], 10.0)  # charge at 10 kW
    assert np.all(fleet.input_values['p_cmd_kw'] == 10.0)
    assert np.all(fleet.get_value(['p_cmd'])[0] == 10.0)

    # 2. Step for 1 hour (3600 seconds)
    fleet.step(time=0, step_size=3600)

    # Expected SoC change: 10 kW * 0.9 * 1h / 50 kWh = +0.18 -> SoC = 0.68
    soc_vals = fleet.get_value(['mean_soc', 'total_actual_p_kw', 'soc[0]'])
    mean_soc, total_p, soc_0 = soc_vals
    assert abs(mean_soc - 0.68) < 1e-4
    assert abs(total_p - 500.0) < 1e-4  # 50 * 10 kW
    assert abs(soc_0 - 0.68) < 1e-4

    # 3. Test indexed update
    fleet.set_value(['p_cmd[0]'], -10.0)  # discharge unit 0
    assert fleet.input_values['p_cmd_kw'][0] == -10.0
    assert fleet.input_values['p_cmd_kw'][1] == 10.0

    # 4. Saturation clipping at SoC = 1.0
    fleet.states['soc'][:] = 0.99
    fleet.set_value(['p_cmd'], 11.0)
    fleet.step(time=3600, step_size=3600)
    # SoC should be clamped to soc_max (0.95) and actual_p_kw reduced
    assert np.all(fleet.states['soc'] == fleet.soc_max)


def test_building_fleet_adapter():
    """Verify BuildingFleetAdapter 1R1C thermal dynamics and broadcast inputs."""
    import numpy as np
    from energysim.container import BuildingFleetAdapter

    fleet = BuildingFleetAdapter(
        name='heat_pumps',
        size=10,
        c_th_kwh_per_k=20.0,
        r_th_k_per_kw=2.0,
        initial_t_in=20.0,
    )

    # Broadcast ambient temperature 0 deg C and 5 kW heating per building
    fleet.set_value(['t_amb_degc', 'p_heat_kw'], [0.0, 5.0])
    fleet.step(time=0, step_size=3600)

    # Building should cool down towards equilibrium (t_eq = 0 + 5*2 = 10 deg C)
    mean_t = fleet.get_value(['mean_t_in_degc'])[0]
    assert mean_t < 20.0
    assert mean_t > 10.0

    total_p = fleet.get_value(['total_p_heat_kw'])[0]
    assert total_p == 50.0  # 10 * 5 kW


def test_pv_fleet_adapter():
    """Verify PVFleetAdapter heterogeneous capacities and temperature derating."""
    import numpy as np
    from energysim.container import PVFleetAdapter

    # Heterogeneous capacities: 5, 10, 15 kWp
    p_peaks = np.array([5.0, 10.0, 15.0])
    pv = PVFleetAdapter(
        name='pv_arrays',
        size=3,
        peak_power_kw=p_peaks,
        gamma_p=0.0,
        pr=1.0,
    )

    # Standard test conditions: G=1000 W/m2, T_amb=25 C
    pv.set_value(['g_irr_w_m2', 't_amb_degc'], [1000.0, 25.0])
    pv.step(time=0, step_size=900)

    p_pv = pv.get_value(['p_pv_kw'])[0]
    assert np.allclose(p_pv, p_peaks, atol=1e-2)

    total_pv = pv.get_value(['total_p_pv_kw'])[0]
    assert abs(total_pv - 30.0) < 1e-2


def test_vectorized_function_adapter():
    """Verify custom user function container dynamics."""
    import numpy as np
    from energysim.container import VectorizedFunctionAdapter

    # Linear decay model: x(t+dt) = x(t) - k * x(t) * dt + u
    def decay_dynamics(time, states, inputs, step_size):
        x = states['x']
        u = inputs.get('u', np.zeros_like(x))
        k = 0.1
        return {'x': x - k * x * (step_size / 3600.0) + u}

    container = VectorizedFunctionAdapter(
        name='decay_fleet',
        size=5,
        update_fn=decay_dynamics,
        initial_states={'x': np.ones(5) * 10.0},
        input_names=['u'],
    )

    container.set_value(['u'], 0.0)
    container.step(time=0, step_size=3600)
    mean_x = container.get_value(['mean_x'])[0]
    assert abs(mean_x - 9.0) < 1e-4


def test_world_container_co_simulation(tmp_path):
    """Verify end-to-end co-simulation coupling a fleet to signals and grid in world."""
    import numpy as np
    from energysim import world
    from energysim.container import BatteryFleetAdapter

    res_file = tmp_path / "fleet_res.h5"
    w = world(start_time=0, stop_time=3600, t_macro=900, power_unit='kW', res_filename=str(res_file))

    # Add 100-unit EV Battery Fleet via add_fleet
    w.add_fleet(
        fleet_type='battery',
        name='ev_fleet',
        size=100,
        capacity_kwh=50.0,
        max_charge_kw=11.0,
        initial_soc=0.2,
        step_size=900,
        power_unit='kW',
    )

    # Dynamic charging schedule: 7.4 kW charge command
    w.add_signal('charge_dispatch', lambda t: [7.4], step_size=900, power_unit='kW')

    # Connect dispatch signal to fleet charging command (broadcasts to all 100 EVs)
    w.add_connections({'charge_dispatch.y': 'ev_fleet.p_cmd'})

    w.simulate(pbar=False)
    results = w.results(dashboard=False)

    assert 'ev_fleet' in results
    # Check that SoC increased over the hour
    mean_soc_series = results['ev_fleet']['mean_soc'].dropna().tolist()
    assert len(mean_soc_series) >= 4
    assert mean_soc_series[-1] > mean_soc_series[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


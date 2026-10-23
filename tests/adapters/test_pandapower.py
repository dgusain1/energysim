"""Integration tests for the pandapower adapter (real pandapower, no mocks)."""
import copy
import warnings

import numpy as np
import pytest

pp = pytest.importorskip("pandapower")

from energysim.base import SimulatorElementNotFoundError, SimulatorVariableError  # noqa: E402
from energysim.ppAdapter import pp_adapter  # noqa: E402

pytestmark = pytest.mark.requires_pandapower


def _direct(net, mode="ac"):
    ref = copy.deepcopy(net)
    (pp.runpp if mode == "ac" else pp.rundcpp)(ref)
    return ref


def test_ac_results_match_direct_runpp(pp_net):
    ref = _direct(pp_net)
    a = pp_adapter("n", model=pp_net, mode="ac")
    a.init()
    got = a.get_value(["MV2.vm_pu", "MV2.va_degree", "Line1.loading_percent", "Line1.i_ka",
                       "Line1.p_from_mw", "T1.loading_percent", "Load1.p_mw", "PV1.q_mvar",
                       "Grid.p_mw", "Bat1.p_mw", "HV.vm_pu"])
    exp = [ref.res_bus.at[2, "vm_pu"], ref.res_bus.at[2, "va_degree"], ref.res_line.at[0, "loading_percent"],
           ref.res_line.at[0, "i_ka"], ref.res_line.at[0, "p_from_mw"], ref.res_trafo.at[0, "loading_percent"],
           ref.res_load.at[0, "p_mw"], ref.res_sgen.at[0, "q_mvar"], ref.res_ext_grid.at[0, "p_mw"],
           ref.res_storage.at[0, "p_mw"], ref.res_bus.at[0, "vm_pu"]]
    assert got == pytest.approx(exp)


def test_dc_results_match_direct_rundcpp(pp_net):
    ref = _direct(pp_net, "dc")
    a = pp_adapter("n", model=pp_net, mode="dc")
    a.init()
    assert a.mode == "dc"
    got = a.get_value(["Line1.p_from_mw", "Grid.p_mw", "MV2.va_degree"])
    exp = [ref.res_line.at[0, "p_from_mw"], ref.res_ext_grid.at[0, "p_mw"], ref.res_bus.at[2, "va_degree"]]
    assert got == pytest.approx(exp)


def test_set_value_then_step_matches_direct(pp_net):
    ref = copy.deepcopy(pp_net)
    ref.load.at[0, "p_mw"] = 18.0
    ref.sgen.at[0, "p_mw"] = 7.5
    pp.runpp(ref)
    a = pp_adapter("n", model=pp_net)
    a.init()
    a.set_value(["Load1.p_mw", "PV1.p_mw"], [18.0, 7.5])
    a.step(0)
    assert a.get_value(["Line1.loading_percent", "MV2.vm_pu"]) == pytest.approx(
        [ref.res_line.at[0, "loading_percent"], ref.res_bus.at[2, "vm_pu"]])


def test_example_simple_known_case():
    net = pp.networks.example_simple()
    ref = _direct(net)
    with pytest.warns(UserWarning, match="unnamed"):
        a = pp_adapter("simple", model=net)
    a.init()
    got = a.get_value(["Line 1.loading_percent", "110kV/20kV transformer.loading_percent",
                       "MV Bus 1.vm_pu", "static generator.p_mw", "load.p_mw", "generator.p_mw"])
    exp = [ref.res_line.at[0, "loading_percent"], ref.res_trafo.at[0, "loading_percent"],
           ref.res_bus.loc[net.bus.index[net.bus.name == "MV Bus 1"][0], "vm_pu"],
           ref.res_sgen.at[0, "p_mw"], ref.res_load.at[0, "p_mw"], ref.res_gen.at[0, "p_mw"]]
    assert got == pytest.approx(exp)


def test_cp1_cost_parameter_roundtrip(pp_net):
    pp.create_poly_cost(pp_net, 0, "sgen", cp1_eur_per_mw=12.0)
    a = pp_adapter("n", model=pp_net)
    a.set_value(["PV1.cp1_eur_per_mw"], [30.0])
    assert a.get_value(["PV1.cp1_eur_per_mw"]) == [30.0]
    with pytest.raises(SimulatorVariableError, match="Cost parameter"):
        a.get_value(["Load1.cp1_eur_per_mw"])


def test_rerun_is_identical_and_network_survives_cleanup(pp_net):
    a = pp_adapter("n", model=pp_net)
    a.init()
    first = a.get_value(["MV2.vm_pu", "Line1.loading_percent"])
    a.set_value(["Load1.p_mw"], [25.0])
    a.step(0)
    changed = a.get_value(["MV2.vm_pu"])
    assert changed[0] != pytest.approx(first[0])
    a.cleanup()
    assert a.network is not None and len(a.network.bus) == 3  # not deleted
    a.init()                                                  # pristine again
    assert a.get_value(["Load1.p_mw"]) == [10.0]
    assert a.get_value(["MV2.vm_pu", "Line1.loading_percent"]) == pytest.approx(first)


def test_set_value_after_cleanup_applies_to_fresh_network(pp_net):
    a = pp_adapter("n", model=pp_net)
    a.init()
    a.set_value(["Load1.p_mw"], [25.0])
    a.cleanup()
    a.set_value(["Load1.p_mw"], [12.0])   # world applies parameters before init()
    a.init()
    assert a.get_value(["Load1.p_mw"]) == [12.0]


def test_errors_are_exceptions(pp_net):
    a = pp_adapter("n", model=pp_net)
    a.init()
    with pytest.raises(SimulatorElementNotFoundError):
        a.get_value(["nope.p_mw"])
    with pytest.raises(SimulatorVariableError):
        a.get_value(["Load1.nonexistent"])
    with pytest.raises(SimulatorVariableError):
        a.set_value(["Load1.nonexistent"], [1.0])
    with pytest.raises(SimulatorVariableError, match="Malformed"):
        a.get_value(["Load1"])
    with pytest.raises(ValueError, match="Unsupported calculation mode"):
        pp_adapter("n", model=pp_net, mode="opf")
    with pytest.raises(SimulatorElementNotFoundError):
        pp_adapter("n", model=pp_net, inputs=["ghost.p_mw"])


def test_dotted_names_and_table_prefix():
    net = pp.create_empty_network()
    b = pp.create_bus(net, vn_kv=20.0, name="B.1")
    pp.create_ext_grid(net, b, name="G")
    pp.create_load(net, b, p_mw=1.0, name="B.1")  # same name as the bus
    with pytest.warns(UserWarning, match="several tables"):
        a = pp_adapter("n", model=net)
    a.init()
    assert a.get_value(["load:B.1.p_mw"]) == [1.0]
    assert a.get_value(["bus:B.1.vm_pu"])[0] == pytest.approx(1.0)


def test_pickle_warns_json_does_not(tmp_path, pp_net):
    pkl = str(tmp_path / "net.p")
    js = str(tmp_path / "net.json")
    pp.to_pickle(pp_net, pkl)
    pp.to_json(pp_net, js)
    with pytest.warns(UserWarning, match="trusted"):
        pp_adapter("n", net_loc=pkl)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pp_adapter("n", net_loc=js)
    with pytest.raises(ValueError, match="Unsupported pandapower network file"):
        (tmp_path / "net.xyz").write_text("x")
        pp_adapter("n", net_loc=str(tmp_path / "net.xyz"))
    with pytest.raises(FileNotFoundError):
        pp_adapter("n", net_loc=str(tmp_path / "missing.json"))


def test_get_available_variables_is_cached_and_validates_typos(pp_net):
    a = pp_adapter("n", model=pp_net)
    v1 = a.get_available_variables()
    assert "Line1.loading_percent" in v1["outputs"]
    assert "T1.loading_percent" in v1["outputs"]
    assert "Load1.p_mw" in v1["inputs"] and "Load1.p_mw" in v1["outputs"]
    assert "Grid.p_mw" in v1["outputs"] and "Bat1.p_mw" in v1["inputs"]
    assert "Load1.p_mww" not in v1["inputs"]
    v1["inputs"].clear()                      # returned lists are copies
    assert a.get_available_variables()["inputs"]


def test_introspection_is_fast_on_large_network():
    import time
    net = pp.networks.case118()
    net.bus["name"] = [f"bus{i}" for i in net.bus.index]
    net.load["name"] = [f"load{i}" for i in net.load.index]
    net.gen["name"] = [f"gen{i}" for i in net.gen.index]
    net.ext_grid["name"] = [f"eg{i}" for i in net.ext_grid.index]
    net.line["name"] = [f"line{i}" for i in net.line.index]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = pp_adapter("big", model=net)
    t0 = time.perf_counter()
    avail = a.get_available_variables()
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    a.get_available_variables()
    second = time.perf_counter() - t0
    assert len(avail["outputs"]) > 1000
    assert first < 2.0 and second < 0.2


def test_numpy_scalars_accepted(pp_net):
    a = pp_adapter("n", model=pp_net)
    a.init()
    a.set_value(["Load1.p_mw"], [np.float64(11.0)])
    a.step(0)
    assert float(a.get_value(["Load1.p_mw"])[0]) == pytest.approx(11.0)

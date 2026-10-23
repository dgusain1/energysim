"""Integration tests for the PyPSA adapter (real PyPSA, compared with direct calls)."""
import logging
import math

import pytest

pypsa = pytest.importorskip("pypsa")

from energysim.base import SimulatorElementNotFoundError, SimulatorVariableError  # noqa: E402
from energysim.pypsaAdapter import pypsa_adapter  # noqa: E402

from adapters.conftest import build_pypsa_net  # noqa: E402

pytestmark = pytest.mark.requires_pypsa


def _snap(net):
    return net.snapshots[0]


def test_ac_matches_direct_pf(pypsa_net):
    ref = build_pypsa_net()
    ref.pf()
    s = _snap(ref)
    a = pypsa_adapter("p", model=pypsa_net, mode="ac")
    a.init()
    got = a.get_value(["g0.P", "g0.Q", "g1.P", "ld.P", "ld.Q", "st.P",
                       "b0.V", "b1.V", "b2.V", "b1.Va", "b2.Va_deg", "l01.p0", "l01.q0", "l12.p1", "l12.q1"])
    exp = [ref.generators_t.p.at[s, "g0"], ref.generators_t.q.at[s, "g0"], ref.generators_t.p.at[s, "g1"],
           ref.loads_t.p.at[s, "ld"], ref.loads_t.q.at[s, "ld"], ref.storage_units_t.p.at[s, "st"],
           ref.buses_t.v_mag_pu.at[s, "b0"], ref.buses_t.v_mag_pu.at[s, "b1"], ref.buses_t.v_mag_pu.at[s, "b2"],
           ref.buses_t.v_ang.at[s, "b1"], math.degrees(ref.buses_t.v_ang.at[s, "b2"]),
           ref.lines_t.p0.at[s, "l01"], ref.lines_t.q0.at[s, "l01"],
           ref.lines_t.p1.at[s, "l12"], ref.lines_t.q1.at[s, "l12"]]
    assert got == pytest.approx(exp, abs=1e-9)
    # AC: reactive power and losses are non-trivial
    assert got[1] != 0.0 and got[6] == pytest.approx(1.0)


def test_dc_matches_direct_lpf(pypsa_net):
    ref = build_pypsa_net()
    ref.lpf()
    s = _snap(ref)
    a = pypsa_adapter("p", model=pypsa_net, mode="dc")
    a.init()
    assert a.mode == "dc"
    got = a.get_value(["g0.P", "l01.p0", "l12.p0", "b1.Va", "b0.V", "g0.Q", "l01.q0"])
    exp = [ref.generators_t.p.at[s, "g0"], ref.lines_t.p0.at[s, "l01"], ref.lines_t.p0.at[s, "l12"],
           ref.buses_t.v_ang.at[s, "b1"], 1.0, 0.0, 0.0]
    assert got == pytest.approx(exp, abs=1e-9)
    assert got[0] == pytest.approx(9.0)    # slack = 30 load - 20 gen - 1 storage (lossless DC)


@pytest.mark.parametrize("mode", ["ac", "dc"])
def test_set_value_loads_generators_storage(pypsa_net, mode):
    ref = build_pypsa_net()
    ref.loads.at["ld", "p_set"] = 45.0
    ref.loads.at["ld", "q_set"] = 8.0
    ref.generators.at["g1", "p_set"] = 12.0
    ref.generators.at["g1", "q_set"] = 1.0
    ref.storage_units.at["st", "p_set"] = -3.0
    ref.storage_units.at["st", "q_set"] = 0.5
    (ref.pf if mode == "ac" else ref.lpf)()
    s = _snap(ref)

    a = pypsa_adapter("p", model=pypsa_net, mode=mode,
                      inputs=["ld.P", "ld.Q", "g1.P", "g1.Q", "st.P", "st.Q"])
    a.init()
    a.set_value(["ld.P", "ld.Q", "g1.P", "g1.Q", "st.P", "st.Q"], [45.0, 8.0, 12.0, 1.0, -3.0, 0.5])
    a.step(0)
    got = a.get_value(["g0.P", "ld.P", "g1.P", "st.P", "l01.p0", "b2.Va"])
    exp = [ref.generators_t.p.at[s, "g0"], ref.loads_t.p.at[s, "ld"], ref.generators_t.p.at[s, "g1"],
           ref.storage_units_t.p.at[s, "st"], ref.lines_t.p0.at[s, "l01"], ref.buses_t.v_ang.at[s, "b2"]]
    assert got == pytest.approx(exp, abs=1e-9)
    assert a.network.loads.at["ld", "p_set"] == 45.0


def test_line_loading_percent(pypsa_net):
    a = pypsa_adapter("p", model=pypsa_net)
    a.init()
    p0, q0, p1, q1 = a.get_value(["l01.p0", "l01.q0", "l01.p1", "l01.q1"])
    expected = max(math.hypot(p0, q0), math.hypot(p1, q1)) / 100.0 * 100.0
    assert a.get_value(["l01.loading_percent"])[0] == pytest.approx(expected)


def test_time_varying_series_does_not_shadow_set_value():
    n = build_pypsa_net()
    n.loads_t.p_set["ld"] = [30.0]
    a = pypsa_adapter("p", model=n)
    a.init()
    a.set_value(["ld.P"], [50.0])
    a.step(0)
    assert a.get_value(["ld.P"]) == pytest.approx([50.0])


def test_rerun_is_identical_and_network_survives_cleanup(pypsa_net):
    a = pypsa_adapter("p", model=pypsa_net)
    a.init()
    first = a.get_value(["g0.P", "b2.V", "l12.p0"])
    a.set_value(["ld.P"], [60.0])
    a.step(0)
    assert a.get_value(["g0.P"])[0] != pytest.approx(first[0])
    a.cleanup()
    assert a.network is not None and len(a.network.buses) == 3
    a.init()
    assert a.network.loads.at["ld", "p_set"] == 30.0
    assert a.get_value(["g0.P", "b2.V", "l12.p0"]) == pytest.approx(first)


def test_errors_are_exceptions_not_asserts(pypsa_net):
    with pytest.raises(SimulatorVariableError, match="Use P or Q"):
        pypsa_adapter("p", model=pypsa_net, inputs=["ld.V"])
    with pytest.raises(SimulatorElementNotFoundError):
        pypsa_adapter("p", model=pypsa_net, inputs=["ghost.P"])
    with pytest.raises(SimulatorElementNotFoundError, match="Generator, Load and StorageUnit"):
        pypsa_adapter("p", model=pypsa_net, inputs=["b0.P"])
    with pytest.raises(SimulatorVariableError):
        pypsa_adapter("p", model=pypsa_net, outputs=["g0.V"])
    with pytest.raises(ValueError, match="Unsupported calculation mode"):
        pypsa_adapter("p", model=pypsa_net, mode="opf")
    multi = build_pypsa_net()
    multi.set_snapshots(range(3))
    with pytest.raises(ValueError, match="exactly one snapshot"):
        pypsa_adapter("p", model=multi)
    a = pypsa_adapter("p", model=pypsa_net)
    with pytest.raises(SimulatorVariableError, match="run a power flow"):
        a.get_value(["g0.P"])  # before init() there are no results yet


def test_logger_level_uses_pypsa_package_logger(pypsa_net):
    pypsa_adapter("p", model=pypsa_net, logger_level="ERROR")
    assert logging.getLogger("pypsa").level == logging.ERROR
    pypsa_adapter("p", model=pypsa_net, logger_level="CRITICAL")
    assert logging.getLogger("pypsa").level == logging.CRITICAL


def test_netcdf_roundtrip_loading(tmp_path, pypsa_net):
    path = str(tmp_path / "net.nc")
    pypsa_net.export_to_netcdf(path)
    a = pypsa_adapter("p", net_loc=path)
    a.init()
    assert a.get_value(["b0.V"])[0] == pytest.approx(1.0)


def test_available_variables(pypsa_net):
    v = pypsa_adapter("p", model=pypsa_net).get_available_variables()
    assert {"ld.P", "g1.Q", "st.P"} <= set(v["inputs"])
    assert {"b1.V", "l01.p0", "l12.loading_percent", "g0.P"} <= set(v["outputs"])
    assert "g0.Pp" not in v["outputs"]

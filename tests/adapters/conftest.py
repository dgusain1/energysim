"""Fixtures shared by the adapter integration tests."""
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture
def feeder_dss():
    """Path of the 3-bus OpenDSS feeder used by the OpenDSS tests."""
    return str(FIXTURES / "feeder3.dss")


def build_pp_net():
    """Small named pandapower network: slack, line, trafo, load, sgen, storage."""
    pp = pytest.importorskip("pandapower")
    net = pp.create_empty_network(name="es_test")
    hv = pp.create_bus(net, vn_kv=110.0, name="HV")
    mv1 = pp.create_bus(net, vn_kv=20.0, name="MV1")
    mv2 = pp.create_bus(net, vn_kv=20.0, name="MV2")
    pp.create_ext_grid(net, hv, vm_pu=1.02, name="Grid")
    pp.create_transformer(net, hv, mv1, std_type="40 MVA 110/20 kV", name="T1")
    pp.create_line(net, mv1, mv2, length_km=5.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV", name="Line1")
    pp.create_load(net, mv2, p_mw=10.0, q_mvar=2.0, name="Load1")
    pp.create_sgen(net, mv1, p_mw=3.0, q_mvar=0.0, name="PV1")
    pp.create_storage(net, mv2, p_mw=1.0, max_e_mwh=5.0, name="Bat1")
    return net


@pytest.fixture
def pp_net():
    return build_pp_net()


def build_pypsa_net():
    """3-bus AC network: slack generator, PQ generator, load, storage, 2 lines."""
    pypsa = pytest.importorskip("pypsa")
    n = pypsa.Network()
    for b in ("b0", "b1", "b2"):
        n.add("Bus", b, v_nom=110.0)
    n.add("Line", "l01", bus0="b0", bus1="b1", x=0.1, r=0.01, s_nom=100.0)
    n.add("Line", "l12", bus0="b1", bus1="b2", x=0.1, r=0.01, s_nom=100.0)
    n.add("Generator", "g0", bus="b0", control="Slack", p_set=0.0)
    n.add("Generator", "g1", bus="b2", control="PQ", p_set=20.0, q_set=2.0)
    n.add("Load", "ld", bus="b1", p_set=30.0, q_set=5.0)
    n.add("StorageUnit", "st", bus="b2", p_set=1.0, q_set=0.0)
    return n


@pytest.fixture
def pypsa_net():
    return build_pypsa_net()

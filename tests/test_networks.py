# -*- coding: utf-8 -*-
"""Tests for the network-lite layer (gas / district heat)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from energysim import World, networks
from energysim.base import SimulatorVariableError
from energysim.networks import EnergyNetwork, GasNetwork, HeatNetwork, NetworkError
from energysim.networks import base as nbase

scipy = pytest.importorskip("scipy")


def _get(net, var, t=0.0):
    return net.get_value([var], t)[0]


def _step(net, dt=1.0, t=0.0, **inputs):
    if inputs:
        net.set_value(list(inputs), list(inputs.values()))
    net.step_model(dt, t)


def _check_balance(net, tol=1e-9):
    b = net.balance
    scale = max(1.0, b["supply_MW"], b["demand_MW"])
    assert abs(net.balance_residual()) <= tol * scale, b
    assert abs(b["delivered_MW"] + b["unserved_MW"] - b["demand_MW"]) <= tol * scale, b


@pytest.fixture
def no_scipy(monkeypatch):
    monkeypatch.setattr(nbase, "_get_linprog", lambda: None)


# ---------------------------------------------------------------------------
# Basic balances
# ---------------------------------------------------------------------------

def test_single_producer_consumer_with_loss():
    n = EnergyNetwork("grid")
    n.add_producer("p", 10.0, cost=5.0)
    n.add_consumer("c", 4.0)
    n.add_pipe("p", "c", 8.0, loss_pct=10.0)
    n.init()
    # outputs valid straight after init
    assert _get(n, "c.delivered_MW") == pytest.approx(4.0)
    assert _get(n, "p.dispatch_MW") == pytest.approx(4.0 / 0.9)
    assert _get(n, "p-c.loss_MW") == pytest.approx(4.0 / 0.9 * 0.1)
    assert _get(n, "p-c.utilization") == pytest.approx(4.0 / 0.9 / 8.0)
    assert _get(n, "total_losses_MW") == pytest.approx(4.0 / 9.0)
    assert _get(n, "total_unserved_MW") == pytest.approx(0.0, abs=1e-12)
    _check_balance(n)


def test_pipe_capacity_limit_gives_unserved():
    n = EnergyNetwork("grid")
    n.add_producer("p", 20.0)
    n.add_consumer("c", 5.0)
    n.add_pipe("p", "c", 3.0)
    n.init()
    _step(n)
    assert _get(n, "c.delivered_MW") == pytest.approx(3.0)
    assert _get(n, "c.unserved_MW") == pytest.approx(2.0)
    assert _get(n, "total_unserved_MW") == pytest.approx(2.0)
    assert _get(n, "p-c.utilization") == pytest.approx(1.0)
    _check_balance(n)


def test_producer_availability_input_limits_dispatch():
    n = EnergyNetwork("grid")
    n.add_producer("p", 10.0)
    n.add_consumer("c", 6.0)
    n.add_pipe("p", "c", 10.0)
    n.init()
    _step(n, **{"p.available_MW": 2.5})
    assert _get(n, "p.dispatch_MW") == pytest.approx(2.5)
    assert _get(n, "c.unserved_MW") == pytest.approx(3.5)


def _two_producer_net(cls=EnergyNetwork):
    n = cls("grid")
    n.add_producer("cheap", 3.0, cost=10.0)
    n.add_producer("dear", 10.0, cost=40.0)
    n.add_junction("j")
    n.add_consumer("c", 5.0)
    n.add_pipe("cheap", "j", 10.0)
    n.add_pipe("dear", "j", 10.0)
    n.add_pipe("j", "c", 10.0)
    return n


def test_merit_order_two_producers():
    n = _two_producer_net()
    n.init()
    _step(n)
    assert _get(n, "cheap.dispatch_MW") == pytest.approx(3.0)
    assert _get(n, "dear.dispatch_MW") == pytest.approx(2.0)
    # prices are inputs: flip the order
    _step(n, **{"cheap.cost": 90.0})
    assert _get(n, "cheap.dispatch_MW") == pytest.approx(0.0, abs=1e-9)
    assert _get(n, "dear.dispatch_MW") == pytest.approx(5.0)
    _check_balance(n)


def test_carrier_efficiency_scales_cost_and_equal_costs_are_deterministic():
    n = EnergyNetwork("grid")
    n.add_producer("boiler", 10.0, cost=30.0, carrier_efficiency=0.5)   # 60 per MWh injected
    n.add_producer("chp", 10.0, cost=40.0)
    n.add_junction("j")
    n.add_consumer("c", 4.0)
    n.add_pipe("boiler", "j", 10.0)
    n.add_pipe("chp", "j", 10.0)
    n.add_pipe("j", "c", 10.0)
    n.init()
    _step(n)
    assert _get(n, "chp.dispatch_MW") == pytest.approx(4.0)
    # equal cost: first-declared producer wins, every time
    m = EnergyNetwork("tie")
    m.add_producer("a", 10.0, cost=7.0)
    m.add_producer("b", 10.0, cost=7.0)
    m.add_junction("j")
    m.add_consumer("c", 4.0)
    for src in "ab":
        m.add_pipe(src, "j", 10.0)
    m.add_pipe("j", "c", 10.0)
    m.init()
    for _ in range(3):
        _step(m)
        assert _get(m, "a.dispatch_MW") == pytest.approx(4.0)
        assert _get(m, "b.dispatch_MW") == pytest.approx(0.0, abs=1e-9)


def test_scarcity_serves_consumers_in_insertion_order():
    n = EnergyNetwork("grid")
    n.add_producer("p", 5.0)
    n.add_junction("j")
    n.add_consumer("first", 4.0)
    n.add_consumer("second", 4.0)
    n.add_pipe("p", "j", 10.0)
    n.add_pipe("j", "first", 10.0)
    n.add_pipe("j", "second", 10.0)
    n.init()
    _step(n)
    assert _get(n, "first.delivered_MW") == pytest.approx(4.0)
    assert _get(n, "second.delivered_MW") == pytest.approx(1.0)
    assert _get(n, "second.unserved_MW") == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _storage_net():
    n = EnergyNetwork("grid")
    n.add_producer("p", 10.0, cost=1.0)
    n.add_storage("s", capacity_mwh=1.0, p_max=2.0, soc_init=0.5)
    n.add_consumer("c", 2.0)
    n.add_pipe("p", "s", 10.0)
    n.add_pipe("s", "c", 10.0)
    return n


def test_storage_charge_then_discharge():
    n = _storage_net()
    n.init()
    _step(n, dt=900.0, **{"s.p_cmd_MW": -2.0})          # charge 2 MW for 0.25 h
    assert _get(n, "s.soc") == pytest.approx(1.0)
    assert _get(n, "p.dispatch_MW") == pytest.approx(4.0)
    assert n.balance["storage_charge_MW"] == pytest.approx(2.0)
    _check_balance(n)
    # full: further charging is refused
    _step(n, dt=900.0, **{"s.p_cmd_MW": -2.0})
    assert n.balance["storage_charge_MW"] == pytest.approx(0.0, abs=1e-9)
    # discharge covers demand while the producer is unavailable
    _step(n, dt=1800.0, **{"s.p_cmd_MW": 1.0, "p.available_MW": 0.0, "c.demand_MW": 1.0})
    assert _get(n, "c.delivered_MW") == pytest.approx(1.0)
    assert n.balance["storage_discharge_MW"] == pytest.approx(1.0)
    assert _get(n, "s.soc") == pytest.approx(0.5)
    _check_balance(n)
    # empty: discharge limited by stored energy
    _step(n, dt=3600.0, **{"s.p_cmd_MW": 2.0, "c.demand_MW": 2.0})
    _step(n, dt=3600.0)
    assert _get(n, "s.soc") == pytest.approx(0.0, abs=1e-9)
    assert _get(n, "c.unserved_MW") == pytest.approx(2.0)   # producer still unavailable
    _check_balance(n)


def test_storage_standing_loss():
    n = EnergyNetwork("grid")
    n.add_storage("s", capacity_mwh=10.0, p_max=1.0, soc_init=1.0, loss_pct_per_day=10.0)
    n.init()
    _step(n, dt=86400.0)
    assert _get(n, "s.soc") == pytest.approx(0.9)
    assert n.balance["standing_loss_MW"] == pytest.approx(1.0 / 24.0)


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def _random_forest(rng, cls=EnergyNetwork, n_nodes=None):
    n_nodes = n_nodes or int(rng.integers(3, 9))
    net = cls(f"rf")
    names = []
    kinds = []
    for i in range(n_nodes):
        kind = ["producer", "consumer", "junction", "storage"][int(rng.choice(4, p=[0.35, 0.35, 0.2, 0.1]))]
        if i == 0:
            kind = "producer"
        if i == 1:
            kind = "consumer"
        nm = f"n{i}"
        if kind == "producer":
            net.add_producer(nm, float(rng.uniform(1, 8)), cost=float(rng.uniform(1, 60)),
                             carrier_efficiency=float(rng.uniform(0.6, 1.0)))
        elif kind == "consumer":
            net.add_consumer(nm, float(rng.uniform(0, 6)))
        elif kind == "storage":
            net.add_storage(nm, float(rng.uniform(0.5, 3)), float(rng.uniform(0.5, 3)),
                            soc_init=float(rng.uniform(0, 1)))
        else:
            net.add_junction(nm)
        names.append(nm)
        kinds.append(kind)
    for i in range(1, n_nodes):
        parent = names[int(rng.integers(0, i))]
        a, b = (parent, names[i]) if rng.random() < 0.5 else (names[i], parent)
        net.add_pipe(a, b, float(rng.uniform(0.5, 9)), loss_pct=float(rng.uniform(0, 15)))
    return net


def _outputs(net):
    return {k: v for k, v in zip(net.outputs, net.get_value(net.outputs, 0.0))}


@pytest.mark.parametrize("seed", range(25))
def test_radial_fallback_matches_lp_on_trees(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    net_lp = _random_forest(rng)
    net_py = copy.deepcopy(net_lp)
    cmds = []
    for _ in range(4):
        cmd = {}
        for s in net_lp._storages:
            cmd[f"{s}.p_cmd_MW"] = float(rng.uniform(-2, 2))
        for c in net_lp._consumers:
            cmd[f"{c}.demand_MW"] = float(rng.uniform(0, 6))
        cmds.append(cmd)
    net_lp.init()
    with monkeypatch.context() as m:
        m.setattr(nbase, "_get_linprog", lambda: None)
        net_py.init()
        for t, cmd in enumerate(cmds):
            net_py.set_value(list(cmd), list(cmd.values()))
            net_py.step_model(600.0, t)
            _check_balance(net_py)
            ref = _outputs(net_py)
            # solve the same step with scipy on an independent copy of the pre-step state
            net_lp.set_value(list(cmd), list(cmd.values()))
            net_lp.step_model(600.0, t)
            got = _outputs(net_lp)
            for k in ref:
                assert got[k] == pytest.approx(ref[k], abs=1e-6), (seed, t, k)


def test_meshed_network_via_lp_and_error_without_scipy(monkeypatch):
    n = EnergyNetwork("mesh")
    n.add_producer("p", 20.0, cost=1.0)
    n.add_junction("a")
    n.add_junction("b")
    n.add_consumer("c", 9.0)
    n.add_pipe("p", "a", 5.0)
    n.add_pipe("p", "b", 5.0)
    n.add_pipe("a", "c", 5.0)
    n.add_pipe("b", "c", 5.0)
    assert n._is_meshed()
    n.init()
    _step(n)
    assert _get(n, "c.delivered_MW") == pytest.approx(9.0)
    assert _get(n, "c.unserved_MW") == pytest.approx(0.0, abs=1e-9)
    # both branches carry load, neither exceeds capacity
    assert max(_get(n, "p-a.utilization"), _get(n, "p-b.utilization")) <= 1.0 + 1e-9
    _check_balance(n)
    _step(n, **{"c.demand_MW": 12.0})
    assert _get(n, "c.unserved_MW") == pytest.approx(2.0)
    monkeypatch.setattr(nbase, "_get_linprog", lambda: None)
    with pytest.raises(NetworkError, match="meshed"):
        n.init()


def test_parallel_pipes_count_as_mesh():
    n = EnergyNetwork("par")
    n.add_producer("p", 10.0)
    n.add_consumer("c", 8.0)
    n.add_pipe("p", "c", 5.0)
    n.add_pipe("p", "c", 5.0)
    assert n._is_meshed()
    n.init()
    _step(n)
    assert _get(n, "c.delivered_MW") == pytest.approx(8.0)
    assert n._pipes[1]["name"] == "p-c-2"


# ---------------------------------------------------------------------------
# Heat
# ---------------------------------------------------------------------------

def _heat_net(delay=3.0, loss=0.0):
    n = HeatNetwork("dh", supply_T_C=90.0, return_T_C=50.0)
    n.add_producer("plant", 20.0, cost=2.0)
    n.add_consumer("c", 2.0)
    n.add_pipe("plant", "c", 20.0, loss_pct=loss, delay_s=delay)
    return n


def test_heat_demand_step_arrives_after_delay():
    n = _heat_net(delay=3.0)
    n.init()
    delivered, dispatch, unserved = [], [], []
    for k in range(10):
        _step(n, dt=1.0, t=float(k), **{"c.demand_MW": 2.0 if k < 4 else 5.0})
        delivered.append(_get(n, "c.delivered_MW"))
        dispatch.append(_get(n, "plant.dispatch_MW"))
        unserved.append(_get(n, "c.unserved_MW"))
    # producer reacts immediately, the consumer only sees it 3 s later
    assert dispatch[4] == pytest.approx(5.0)
    assert delivered[:7] == pytest.approx([2, 2, 2, 2, 2, 2, 2])
    assert unserved[4:7] == pytest.approx([3.0, 3.0, 3.0])
    assert delivered[7:] == pytest.approx([5.0, 5.0, 5.0])
    assert unserved[7:] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_heat_demand_drop_reports_overshoot_as_loss_and_conserves():
    n = _heat_net(delay=2.0, loss=5.0)
    n.init()
    for k, d in enumerate([4, 4, 4, 1, 1, 1, 1, 6, 6, 6, 0, 0, 0]):
        _step(n, dt=1.0, t=float(k), **{"c.demand_MW": float(d)})
        _check_balance(n)
        assert _get(n, "c.delivered_MW") <= d + 1e-12
    assert n.balance["delivered_MW"] + n.balance["unserved_MW"] == pytest.approx(n.balance["demand_MW"])


def test_heat_transit_energy_tracks_inflight():
    n = _heat_net(delay=2.0)
    n.init()
    _step(n, dt=1.0, t=0.0, **{"c.demand_MW": 2.0})
    _step(n, dt=1.0, t=1.0, **{"c.demand_MW": 6.0})
    # step 2 emitted 6 MW but only the older 2 MW arrived: 4 MWh in the pipe
    assert _get(n, "transit_MWh") == pytest.approx(4.0 / 3600.0)
    assert n.balance["dtransit_MW"] == pytest.approx(4.0)


def test_heat_temperature_drops_with_losses():
    n = HeatNetwork("dh", supply_T_C=90.0, return_T_C=50.0)
    n.add_producer("plant", 50.0)
    n.add_junction("j")
    n.add_consumer("near", 5.0)
    n.add_consumer("far", 5.0)
    n.add_pipe("plant", "j", 50.0, loss_pct=10.0)
    n.add_pipe("j", "near", 50.0)
    n.add_pipe("j", "far", 50.0, loss_pct=20.0)
    n.init()
    _step(n)
    assert _get(n, "near.supply_T_C") == pytest.approx(90.0 - 40.0 * 0.1)
    assert _get(n, "far.supply_T_C") == pytest.approx(90.0 - 40.0 * (1 - 0.9 * 0.8))
    assert _get(n, "mean_supply_T") == pytest.approx(
        (_get(n, "near.supply_T_C") + _get(n, "far.supply_T_C")) / 2)
    n.set_value(["supply_T_C"], [100.0])      # registered parameter
    _step(n)
    assert _get(n, "near.supply_T_C") == pytest.approx(100.0 - 50.0 * 0.1)


# ---------------------------------------------------------------------------
# Gas
# ---------------------------------------------------------------------------

def test_gas_unit_conversion_nm3h():
    g = GasNetwork("gas", unit="Nm3/h", lhv_kwh_per_nm3=10.0)
    g.add_producer("src", 10.0)
    g.add_consumer("c", 1.0)
    g.add_pipe("src", "c", 10.0)
    g.init()
    _step(g, **{"c.demand_MW": 500.0})            # 500 Nm3/h = 5 MW
    assert _get(g, "c.delivered_MW") == pytest.approx(500.0)
    assert _get(g, "src.dispatch_MW") == pytest.approx(500.0)
    assert g.balance["supply_MW"] == pytest.approx(5.0)
    assert g.describe()["units"]["c.demand_MW"] == "Nm3/h"
    g.set_value(["lhv_kwh_per_nm3"], [11.0])
    _step(g, **{"c.demand_MW": 500.0})
    assert g.balance["demand_MW"] == pytest.approx(5.5)
    with pytest.raises(NetworkError):
        GasNetwork("bad", unit="kg/s")


def test_gas_linepack_buffers_deficit_then_refills():
    g = GasNetwork("gas", p_min_bar=30.0, p_max_bar=70.0)
    g.add_producer("src", 10.0)
    g.add_consumer("c", 6.0)
    g.add_pipe("src", "c", 4.0, linepack_capacity_mwh=1.0, linepack_init=0.5)
    g.init()
    assert _get(g, "src-c.pressure_bar") == pytest.approx(50.0)
    h = 900.0
    _step(g, dt=h)                    # deficit 2 MW for 0.25 h drains the 0.5 MWh linepack
    assert _get(g, "c.unserved_MW") == pytest.approx(0.0, abs=1e-9)
    assert _get(g, "src-c.linepack_MWh") == pytest.approx(0.0, abs=1e-9)
    assert _get(g, "src-c.pressure_bar") == pytest.approx(30.0)
    _check_balance(g)
    _step(g, dt=h)                    # empty: the withdrawal is now unserved
    assert _get(g, "c.unserved_MW") == pytest.approx(2.0)
    _check_balance(g)
    _step(g, dt=h, **{"c.demand_MW": 1.0})   # spare capacity refills up to the initial fill
    assert _get(g, "src-c.linepack_MWh") == pytest.approx(0.5)
    assert _get(g, "src-c.pressure_bar") == pytest.approx(50.0)
    assert _get(g, "total_linepack_MWh") == pytest.approx(0.5)
    _check_balance(g)
    # a producer covers demand first: no linepack use at all
    _step(g, dt=h, **{"c.demand_MW": 3.0})
    assert _get(g, "src-c.linepack_MWh") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Conservation over random networks (incl. meshes, delays, linepack)
# ---------------------------------------------------------------------------

def _random_mesh(rng, cls):
    n_nodes = int(rng.integers(4, 10))
    net = cls("rm")
    names = [f"n{i}" for i in range(n_nodes)]
    for i, nm in enumerate(names):
        kind = "producer" if i < 2 else ("consumer" if i < 4 else
                                         ["producer", "consumer", "junction", "storage"][int(rng.integers(0, 4))])
        if kind == "producer":
            net.add_producer(nm, float(rng.uniform(1, 8)), cost=float(rng.uniform(0, 50)),
                             carrier_efficiency=float(rng.uniform(0.5, 1)))
        elif kind == "consumer":
            net.add_consumer(nm, float(rng.uniform(0, 6)))
        elif kind == "storage":
            net.add_storage(nm, float(rng.uniform(0.2, 2)), float(rng.uniform(0.5, 3)),
                            soc_init=float(rng.uniform(0, 1)), loss_pct_per_day=float(rng.uniform(0, 20)))
        else:
            net.add_junction(nm)
    for _ in range(int(rng.integers(n_nodes, 2 * n_nodes))):
        i, j = sorted(rng.choice(n_nodes, size=2, replace=False))
        kw = dict(capacity=float(rng.uniform(0.5, 8)), loss_pct=float(rng.uniform(0, 20)))
        if cls is HeatNetwork:
            kw["delay_s"] = float(rng.integers(0, 4))
        if cls is GasNetwork:
            kw.update(linepack_capacity_mwh=float(rng.uniform(0, 1)), linepack_init=float(rng.uniform(0, 1)))
        net.add_pipe(names[i], names[j], **kw)
    return net


@pytest.mark.parametrize("cls", [EnergyNetwork, HeatNetwork, GasNetwork])
@pytest.mark.parametrize("seed", range(12))
def test_conservation_random_networks(cls, seed):
    rng = np.random.default_rng(1000 + seed)
    net = _random_mesh(rng, cls)
    net.init()
    for t in range(25):
        inputs = {}
        for c in net._consumers:
            inputs[f"{c}.demand_MW"] = float(rng.uniform(0, 7))
        for s in net._storages:
            inputs[f"{s}.p_cmd_MW"] = float(rng.uniform(-3, 3))
        for p in net._producers:
            inputs[f"{p}.available_MW"] = float(rng.uniform(0, 9))
            inputs[f"{p}.cost"] = float(rng.uniform(0, 50))
        net.set_value(list(inputs), list(inputs.values()))
        net.step_model(float(rng.choice([60.0, 900.0, 3600.0])), float(t))
        _check_balance(net)
        out = _outputs(net)
        assert all(np.isfinite(v) for v in out.values())
        assert out["total_unserved_MW"] >= -1e-12
        assert out["total_losses_MW"] >= -1e-9
        for s in net._storages:
            assert -1e-9 <= out[f"{s}.soc"] <= 1 + 1e-9
        for p in net._pipes:
            assert out[f"{p['name']}.utilization"] <= 1 + 1e-9
        for p, spec in net._producers.items():
            assert out[f"{p}.dispatch_MW"] <= min(inputs[f"{p}.available_MW"], spec["capacity"]) + 1e-9


@pytest.mark.parametrize("cls", [HeatNetwork, GasNetwork])
@pytest.mark.parametrize("seed", range(6))
def test_conservation_random_forests_without_scipy(cls, seed, no_scipy):
    rng = np.random.default_rng(seed)
    net = _random_forest(rng, cls)
    net.init()
    for t in range(10):
        inputs = {f"{c}.demand_MW": float(rng.uniform(0, 7)) for c in net._consumers}
        net.set_value(list(inputs), list(inputs.values()))
        net.step_model(300.0, float(t))
        _check_balance(net)


# ---------------------------------------------------------------------------
# Framework contract
# ---------------------------------------------------------------------------

def _state(net):
    return copy.deepcopy({k: v for k, v in net.__dict__.items()
                          if k in ("_soc", "_linepack", "_dq", "_clock", "_transit_MWh")})


@pytest.mark.parametrize("make", [
    lambda: _storage_net(),
    lambda: _heat_net(delay=3.0, loss=4.0),
    lambda: _gas_with_linepack(),
])
def test_dt_zero_leaves_state_unchanged(make):
    net = make()
    net.init()
    for k in range(4):
        _step(net, dt=60.0, t=float(k))
    before = _state(net)
    net.step_model(0.0, 5.0)
    assert _state(net) == before


def _gas_with_linepack():
    g = GasNetwork("gas")
    g.add_producer("src", 10.0)
    g.add_consumer("c", 6.0)
    g.add_pipe("src", "c", 4.0, linepack_capacity_mwh=1.0)
    return g


@pytest.mark.parametrize("make", [
    lambda: _storage_net(),
    lambda: _heat_net(delay=2.0, loss=3.0),
    lambda: _gas_with_linepack(),
])
def test_rerun_is_deterministic_after_init(make):
    net = make()
    demands = [1.0, 3.0, 6.0, 2.0, 0.5, 4.0]

    def run():
        net.init()
        trace = []
        for k, d in enumerate(demands):
            net.set_value(["c.demand_MW"], [d])
            if "s.p_cmd_MW" in net.inputs:
                net.set_value(["s.p_cmd_MW"], [-1.0 if k % 2 else 1.0])
            net.step_model(600.0, float(k))
            trace.append(_outputs(net))
        return trace

    first = run()
    second = run()
    assert first == second


def test_strict_variable_access_and_describe():
    n = _two_producer_net(HeatNetwork)
    n.init()
    with pytest.raises(SimulatorVariableError):
        n.set_value(["nope.demand_MW"], [1.0])
    with pytest.raises(SimulatorVariableError):
        n.get_value(["nope.x"], 0.0)
    d = n.describe()
    assert d["class"] == "HeatNetwork"
    assert "c.demand_MW" in d["inputs"] and "c.delivered_MW" in d["outputs"]
    assert d["units"]["c.demand_MW"] == "MW"
    assert d["units"]["cheap.cost"] == "currency/MWh"
    assert d["units"]["mean_supply_T"] == "degC"
    assert d["topology"]["pipes"][0]["a"] == "cheap"
    assert set(n.inputs + n.outputs) <= set(n.units)
    for cls in (EnergyNetwork, HeatNetwork, GasNetwork):
        assert cls.describe_class()["class"] == cls.__name__


def test_topology_text_and_dict_roundtrip():
    g = GasNetwork("g", unit="Nm3/h", lhv_kwh_per_nm3=9.5)
    g.add_producer("src", 10.0, cost=3.0)
    g.add_storage("sto", 2.0, 1.0)
    g.add_consumer("c", 2.0)
    g.add_junction("j")
    g.add_pipe("src", "j", 8.0, loss_pct=1.0, length_km=12.0)
    g.add_pipe("j", "sto", 3.0, linepack_capacity_mwh=1.0)
    g.add_pipe("sto", "c", 5.0)
    text = g.describe_topology()
    assert "producer  src" in text and "storage   sto" in text and "radial" in text
    spec = g.to_dict()
    g2 = EnergyNetwork.from_dict(spec)          # dispatches on spec['type']
    assert isinstance(g2, GasNetwork) and g2.unit == "Nm3/h"
    assert g2.to_dict() == spec
    assert sorted(g2.inputs) == sorted(g.inputs) and sorted(g2.outputs) == sorted(g.outputs)
    h = EnergyNetwork.from_dict({"type": "heat", "producers": {"p": {"capacity": 5.0}},
                                 "consumers": {"c": {"demand": 1.0}},
                                 "pipes": [{"a": "p", "b": "c", "capacity": 5.0, "delay_s": 2}]})
    assert isinstance(h, HeatNetwork)
    h.init()
    with pytest.raises(NetworkError):
        EnergyNetwork.from_dict({"type": "steam"})
    with pytest.raises(NetworkError):
        EnergyNetwork.from_dict({"bogus": 1})


@pytest.mark.parametrize("build", [
    lambda n: n.add_producer("a.b", 1.0),
    lambda n: n.add_producer("p", -1.0),
    lambda n: n.add_producer("p", 1.0, carrier_efficiency=0.0),
    lambda n: n.add_consumer("c", -2.0),
    lambda n: n.add_storage("s", 0.0, 1.0),
    lambda n: n.add_storage("s", 1.0, 1.0, soc_init=1.5),
    lambda n: n.add_pipe("p", "ghost", 1.0),
    lambda n: n.add_pipe("p", "p", 1.0),
    lambda n: n.add_pipe("p", "c", 1.0, loss_pct=100.0),
    lambda n: n.add_junction("p"),
])
def test_builder_validation(build):
    n = EnergyNetwork("v")
    n.add_producer("p", 5.0)
    n.add_consumer("c", 1.0)
    with pytest.raises(NetworkError):
        build(n)


# ---------------------------------------------------------------------------
# World integration
# ---------------------------------------------------------------------------

def test_world_integration_with_signals_and_models(tmp_path):
    from energysim.models import BatteryBESS, PEMElectrolyzer
    w = World(start_time=0, stop_time=600, t_macro=60, res_filename=str(tmp_path / "net.h5"))
    net = HeatNetwork("grid", supply_T_C=85.0, return_T_C=45.0)
    net.add_producer("boiler", 6.0, cost=40.0)
    net.add_producer("waste", 3.0, cost=0.0)
    net.add_junction("hub")
    net.add_consumer("district", 2.0)
    net.add_pipe("boiler", "hub", 10.0, loss_pct=2.0)
    net.add_pipe("waste", "hub", 10.0, loss_pct=2.0)
    net.add_pipe("hub", "district", 10.0, loss_pct=3.0, delay_s=120.0)

    w.add_model("grid", model=net)
    w.add_model("electrolyzer", model=PEMElectrolyzer(name="electrolyzer", p_nominal_mw=3.0))
    w.add_model("bess", model=BatteryBESS(name="bess", capacity_mwh=4.0, p_max_mw=2.0, soc_init=0.6))
    w.add_signal("demand", signal=lambda t: 2.0 if t < 300 else 6.0)
    w.add_signal("h2_cmd", signal=lambda t: 2.0)
    w.add_signal("bess_cmd", signal=lambda t: 0.5)
    w.add_connections({
        "demand.y": "grid.district.demand_MW",
        "h2_cmd.y": "electrolyzer.P_cmd",
        "bess_cmd.y": "bess.P_cmd",
        # electrolyzer power draw -> waste-heat availability (toy coupling)
        "electrolyzer.P_actual": "grid.waste.available_MW",
    })
    w.simulate(pbar=False)
    res = w.results(to_csv=False)["grid"]
    assert "district.delivered_MW" in res.columns and "total_losses_MW" in res.columns
    assert res["district.delivered_MW"].iloc[0] == pytest.approx(2.0, abs=1e-6)
    # the demand step at t=300 is served in full only after the 120 s pipe delay
    assert res["district.unserved_MW"].max() > 1.0
    assert res["district.unserved_MW"].iloc[-1] == pytest.approx(0.0, abs=1e-6)
    assert (res["waste.dispatch_MW"] <= res["waste.available_MW"] + 1e-9).all() \
        if "waste.available_MW" in res.columns else True
    assert res["boiler.dispatch_MW"].iloc[-1] > 0.0


def test_world_integration_gas_network_with_signal_demand(tmp_path):
    w = World(start_time=0, stop_time=300, t_macro=60, res_filename=str(tmp_path / "gas.h5"))
    g = GasNetwork("gas", unit="Nm3/h")
    g.add_producer("src", 20.0)
    g.add_consumer("boiler", 0.0)
    g.add_pipe("src", "boiler", 15.0, linepack_capacity_mwh=0.5)
    w.add_model("gas", model=g)
    w.add_signal("load", signal=lambda t: 1000.0)       # Nm3/h
    w.add_connections({"load.y": "gas.boiler.demand_MW"})
    w.simulate(pbar=False)
    res = w.results(to_csv=False)["gas"]
    assert res["boiler.delivered_MW"].iloc[-1] == pytest.approx(1000.0)
    assert res["src.dispatch_MW"].iloc[-1] == pytest.approx(1000.0)


def test_lazy_export():
    import energysim
    assert energysim.networks is networks
    assert {"EnergyNetwork", "HeatNetwork", "GasNetwork"} <= set(networks.__all__)

# -*- coding: utf-8 -*-
"""Tests for energysim.control (controllers, forecasts, LP blocks, MPC, OPF, PyPSA)."""

from __future__ import annotations

import copy
import math
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

from energysim import world, SimulatorVariableError  # noqa: E402
from energysim.base import SimulatorAdapter  # noqa: E402
from energysim.control import (  # noqa: E402
    CallableForecast, Controller, ElectrolyzerBlock, GridBlock, LoadBlock, MPCController, NoisyForecast,
    PerfectForecast, PersistenceForecast, RuleController, StorageBlock, block_from_component,
    STATUS_INFEASIBLE, STATUS_OPTIMAL,
)
from energysim.models import BatteryBESS  # noqa: E402
from energysim.models.common.base_component import ModelComponent  # noqa: E402

DAY_PRICES = np.array([42, 38, 35, 33, 34, 40, 55, 72, 80, 70, 60, 52,
                       48, 45, 47, 55, 68, 90, 105, 95, 78, 62, 50, 45], dtype=float)


def price_at(t):
    """Hourly price (EUR/MWh), periodic over one day."""
    return float(DAY_PRICES[int(t // 3600) % 24])


def price_fc(t, times):
    return np.array([price_at(x) for x in times])


def load_at(t):
    """Piecewise constant (15 min) load in MW."""
    k = int(t // 900)
    return 0.55 + 0.25 * math.sin(2 * math.pi * k / 96.0) + 0.1 * ((k * 7) % 5) / 4.0


def load_fc(t, times):
    return np.array([load_at(x) for x in times])


def make_world(tmp_path, monkeypatch, stop, t_macro):
    monkeypatch.chdir(tmp_path)
    return world(start_time=0, stop_time=stop, t_macro=t_macro, power_unit="MW")


def run_world(w):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w.simulate(pbar=False)


class IdealStorage(ModelComponent):
    """Energy-bucket storage matching StorageBlock exactly; logs the applied power per step."""
    units = {"P_cmd": "MW", "P_actual": "MW", "SoC": "1"}

    def __init__(self, name="store", capacity=4.0, p_max=1.0, eta=0.95, soc_init=0.5, soc_min=0.1,
                 soc_max=0.9, step_size=900.0):
        super().__init__(name=name, step_size=step_size)
        self.capacity, self.p_max, self.eta = capacity, p_max, eta
        self.soc, self.soc_min, self.soc_max = soc_init, soc_min, soc_max
        self.p_actual = 0.0
        self.log = []
        self.register_variables(inputs=["P_cmd"], outputs=["P_actual", "SoC"])

    def step_model(self, dt, time):
        p = max(-self.p_max, min(self.p_max, float(self._inputs_val.get("P_cmd", 0.0))))
        h = dt / 3600.0
        e = self.soc * self.capacity
        e_new = e + (self.eta * p * h if p > 0 else p / self.eta * h)
        e_new = min(max(e_new, self.soc_min * self.capacity), self.soc_max * self.capacity)
        if dt > 0:
            p = ((e_new - e) / (self.eta * h)) if p > 0 else ((e_new - e) * self.eta / h)
            self.log.append((time, dt, p))
        self.soc = e_new / self.capacity
        self.p_actual = p
        self._outputs_val["P_actual"] = p
        self._outputs_val["SoC"] = self.soc


class LoggedBESS(BatteryBESS):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.log = []

    def step_model(self, dt, time):
        super().step_model(dt, time)
        if dt > 0:
            self.log.append((time, dt, self.p_actual, self.soc))


def storage_block(name="store", **kw):
    args = dict(capacity=4.0, p_charge_max=1.0, eta_c=0.95, eta_d=0.95, soc_min=0.1, soc_max=0.9, soc0=0.5)
    args.update(kw)
    return StorageBlock(name, **args)


def realized_cost(plant_log, load_fn=load_at, price_fn=price_at):
    cost = 0.0
    for t, dt, p in plant_log:
        cost += price_fn(t) * (load_fn(t) + p) * dt / 3600.0
    return cost


# ---------------------------------------------------------------------------
# (a) arbitrage LP vs an independent formulation
# ---------------------------------------------------------------------------

def independent_arbitrage_lp(prices, load, C, pmax, eta_c, eta_d, smin, smax, s0, dt_h, export_factor):
    from scipy.optimize import linprog
    N = len(prices)
    # x = [pc(N), pd(N), imp(N), exp(N), e(N)]   e[k] = energy after interval k
    nv = 5 * N
    c = np.zeros(nv)
    c[2 * N:3 * N] = prices * dt_h
    c[3 * N:4 * N] = -export_factor * prices * dt_h
    A_eq, b_eq = [], []
    for k in range(N):
        row = np.zeros(nv)  # balance: imp - exp + pd - pc = load
        row[2 * N + k], row[3 * N + k], row[N + k], row[k] = 1, -1, 1, -1
        A_eq.append(row)
        b_eq.append(load[k])
        row = np.zeros(nv)  # e[k] - e[k-1] - eta_c pc dt + pd/eta_d dt = 0
        row[4 * N + k] = 1
        if k > 0:
            row[4 * N + k - 1] = -1
        row[k] = -eta_c * dt_h
        row[N + k] = dt_h / eta_d
        A_eq.append(row)
        b_eq.append(s0 * C if k == 0 else 0.0)
    bounds = [(0, pmax)] * (2 * N) + [(0, None)] * (2 * N) + [(smin * C, smax * C)] * N
    res = linprog(c, A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds, method="highs")
    assert res.status == 0
    return res.fun, res.x[:N] - res.x[N:2 * N]


def brute_force_lossless(prices, C_steps, s0_steps):
    """Exhaustive DP over integer SoC levels (lossless, 1 step per interval) for a cross-check."""
    N = len(prices)
    best = {s0_steps: 0.0}
    for k in range(N):
        nxt = {}
        for s, cost in best.items():
            for u in (-1, 0, 1):
                s2 = s + u
                if 0 <= s2 <= C_steps:
                    c2 = cost + prices[k] * u
                    if c2 < nxt.get(s2, np.inf):
                        nxt[s2] = c2
        best = nxt
    return min(best.values())


def test_a_battery_arbitrage_matches_independent_lp():
    load = np.full(24, 0.7)
    blocks = [storage_block("bess"), LoadBlock("load", load),
              GridBlock("grid", price_import="price", price_export=CallableForecast(lambda t, ts: 0.8 * price_fc(t, ts)))]
    mpc = MPCController(blocks, forecasts={"price": CallableForecast(price_fc)}, horizon=24, dt=3600)
    plan = mpc.plan_open_loop(0.0)
    assert plan.status == STATUS_OPTIMAL
    ref, p_ref = independent_arbitrage_lp(DAY_PRICES, load, 4.0, 1.0, 0.95, 0.95, 0.1, 0.9, 0.5, 1.0, 0.8)
    assert plan.objective == pytest.approx(ref, abs=1e-6)
    # Lossless, unit-step case against exhaustive dynamic programming.
    mpc2 = MPCController([StorageBlock("b", capacity=3.0, p_charge_max=1.0, soc0=1 / 3),
                          GridBlock("grid", price_import=DAY_PRICES, price_export=DAY_PRICES)],
                         horizon=24, dt=3600)
    plan2 = mpc2.plan_open_loop(0.0)
    assert plan2.objective == pytest.approx(brute_force_lossless(DAY_PRICES, 3, 1), abs=1e-6)


# ---------------------------------------------------------------------------
# (b) closed loop in a world with a real BatteryBESS plant
# ---------------------------------------------------------------------------

def test_b_closed_loop_constraints_with_bess_plant(tmp_path, monkeypatch):
    dt = 900
    stop = 24 * 3600
    plant = LoggedBESS(name="bess", capacity_mwh=2.0, p_max_mw=0.5, soc_init=0.3, soc_min=0.1, soc_max=0.9,
                       step_size=60)
    import_limit = 1.0
    prices_df = pd.DataFrame({"time": np.arange(0, 49) * 3600.0,
                              "price": [price_at(h * 3600) for h in range(49)]})
    mpc = MPCController.from_components(
        {"bess": plant},
        forecasts={"price": PerfectForecast(prices_df, "price"), "load": CallableForecast(load_fc)},
        loads={"load": "load"}, prices="price", grid={"import_limit": import_limit, "price_export": 0.0},
        horizon=48, dt=dt, extra_inputs=["load_now", "price_now"],
    )
    w = make_world(tmp_path, monkeypatch, stop, dt)
    w.add_signal("load", load_at)
    w.add_signal("price", price_at)
    w.add_model("bess", plant, step_size=60)
    w.add_model("mpc", mpc, step_size=dt)
    w.add_connections({
        "mpc.bess.P_cmd": "bess.P_cmd",
        "bess.SoC": "mpc.bess.SoC",
        "load.y": "mpc.load_now",
        "price.y": "mpc.price_now",
    })
    run_world(w)

    assert len(mpc.history) == 1 + stop // dt
    assert all(h["solver_status"] == STATUS_OPTIMAL for h in mpc.history)
    assert all(h["fallback"] == 0 for h in mpc.history)
    # signals reached the controller
    assert mpc.history[5]["measurements"]["price_now"] == pytest.approx(price_at(4 * dt))
    # plan never uses unserved energy, SoC plans inside bounds
    for plan in mpc.plans:
        assert np.all(plan.values["unserved.electricity"] < 1e-7)
        assert np.all(plan.values["bess.SoC_plan"] >= 0.1 - 1e-9)
        assert np.all(plan.values["bess.SoC_plan"] <= 0.9 + 1e-9)
    # plant: power limits, SoC bounds and the import limit hold at every micro step
    socs = np.array([r[3] for r in plant.log])
    assert socs.min() >= 0.1 - 1e-9 and socs.max() <= 0.9 + 1e-9
    for t, _, p, _ in plant.log:
        assert abs(p) <= 0.5 + 1e-9
        assert load_at(t) + p <= import_limit + 1e-6
    # the battery was actually used
    assert max(abs(r[2]) for r in plant.log) > 0.2


# ---------------------------------------------------------------------------
# (c) receding horizon vs full-horizon optimum (perfect forecast, matched model)
# ---------------------------------------------------------------------------

def _rh_world(tmp_path, monkeypatch, mpc, plant, stop, dt):
    w = make_world(tmp_path, monkeypatch, stop, dt)
    w.add_model("store", plant, step_size=dt)
    w.add_model("mpc", mpc, step_size=dt)
    w.add_connections({"mpc.store.P_cmd": "store.P_cmd", "store.SoC": "mpc.store.SoC"})
    run_world(w)
    return realized_cost(plant.log)


@pytest.mark.parametrize("horizon", [96, 192])
def test_c_receding_horizon_close_to_full_horizon_optimum(tmp_path, monkeypatch, horizon):
    dt, stop = 900, 48 * 3600
    n = stop // dt

    def make_mpc():
        return MPCController(
            [storage_block(), LoadBlock("load", "load"), GridBlock("grid", price_import="price")],
            forecasts={"price": CallableForecast(price_fc), "load": CallableForecast(load_fc)},
            horizon=horizon, dt=dt, horizon_end=stop)

    full = make_mpc().plan_open_loop(0.0, n_steps=n)
    assert full.ok
    plant = IdealStorage(step_size=dt)
    rh_cost = _rh_world(tmp_path, monkeypatch, make_mpc(), plant, stop, dt)
    assert len(plant.log) == n
    assert rh_cost >= full.objective - 1e-6
    assert rh_cost <= full.objective * 1.01
    if horizon >= n:  # shrinking horizon covering everything: identical to the optimum
        assert rh_cost == pytest.approx(full.objective, rel=1e-7)


# ---------------------------------------------------------------------------
# (d) MPC beats a naive rule-based controller
# ---------------------------------------------------------------------------

def test_d_mpc_beats_rule_based(tmp_path, monkeypatch):
    dt, stop = 900, 24 * 3600
    med = float(np.median(DAY_PRICES))

    def rule(t, meas, fc):
        p = price_at(t)
        soc = meas["store.SoC"]
        if p < med and soc < 0.9:
            return {"store.P_cmd": 1.0}
        if p > med and soc > 0.1:
            return {"store.P_cmd": -1.0}
        return {"store.P_cmd": 0.0}

    rule_ctrl = RuleController(rule, ["store.SoC"], ["store.P_cmd"], name="rule",
                               initial_measurements={"store.SoC": 0.5}, actuation_delay=True)
    plant_r = IdealStorage(step_size=dt)
    w = make_world(tmp_path, monkeypatch, stop, dt)
    w.add_model("store", plant_r, step_size=dt)
    w.add_model("mpc", rule_ctrl, step_size=dt)
    w.add_connections({"mpc.store.P_cmd": "store.P_cmd", "store.SoC": "mpc.store.SoC"})
    run_world(w)
    rule_cost = realized_cost(plant_r.log)

    mpc = MPCController([storage_block(), LoadBlock("load", "load"), GridBlock("grid", price_import="price")],
                        forecasts={"price": CallableForecast(price_fc), "load": CallableForecast(load_fc)},
                        horizon=96, dt=dt)
    plant_m = IdealStorage(step_size=dt)
    mpc_cost = _rh_world(tmp_path, monkeypatch, mpc, plant_m, stop, dt)
    no_storage = sum(price_at(t) * load_at(t) * dt / 3600 for t in range(0, stop, dt))
    assert mpc_cost < rule_cost
    assert mpc_cost < no_storage


# ---------------------------------------------------------------------------
# (e) electrolyser + storage + price (MILP and LP)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_binary", [True, False])
def test_e_electrolyzer_storage_price(use_binary):
    N = 24
    curve = [(1.0, 20.0), (3.0, 57.0), (5.0, 90.0)]       # concave (kg/h)
    blocks = [
        GridBlock("grid", price_import=DAY_PRICES, import_limit=6.0),
        ElectrolyzerBlock("ely", p_max=5.0, p_min=1.0, curve=curve, use_binary=use_binary),
        StorageBlock("tank", capacity=600.0, p_charge_max=200.0, carrier="hydrogen", soc0=0.2, soc_min=0.0,
                     soc_max=1.0, soc_measurement="SoC"),
        LoadBlock("h2", 40.0, carrier="hydrogen"),
        storage_block("bess"),
    ]
    mpc = MPCController(blocks, horizon=N, dt=3600, unserved_penalty=None)
    plan = mpc.plan_open_loop(0.0)
    assert plan.ok
    p = plan["ely.P_cmd"]
    h2 = plan["ely.H2_plan"]
    tank = plan["tank.P_cmd"]
    assert np.allclose(h2 - tank, 40.0, atol=1e-6)            # hydrogen balance
    assert np.all(h2 <= np.interp(p, *zip(*curve)) + 1e-6)   # on/under the curve
    if use_binary:
        assert np.all((p < 1e-6) | (p >= 1.0 - 1e-6))        # minimum turndown respected
    # electrolyser runs preferentially at cheap hours
    assert np.dot(p, DAY_PRICES) / max(p.sum(), 1e-9) < DAY_PRICES.mean()
    # electrolyser from the component library builds and solves
    from energysim.models import PEMElectrolyzer
    pem_block = block_from_component(PEMElectrolyzer(p_nominal_mw=5.0), name="pem", use_binary=use_binary)
    mpc2 = MPCController([GridBlock("grid", price_import=DAY_PRICES), pem_block,
                          GridBlock("offtake", carrier="hydrogen", import_limit=0.0, price_export=3.5)],
                         horizon=N, dt=3600)
    plan2 = mpc2.plan_open_loop(0.0)
    assert plan2.ok and plan2["pem.P_cmd"].max() > 0


def test_e_milp_objective_not_better_than_relaxation():
    common = dict(horizon=24, dt=3600, unserved_penalty=None)

    def build(binary):
        return MPCController([
            GridBlock("grid", price_import=DAY_PRICES),
            ElectrolyzerBlock("ely", p_max=5.0, p_min=2.0, spec_consumption=50.0, use_binary=binary),
            LoadBlock("h2", 30.0, carrier="hydrogen"),
            StorageBlock("tank", capacity=500.0, p_charge_max=100.0, carrier="hydrogen", soc0=0.1),
        ], **common)
    milp, lp = build(True).plan_open_loop(0.0), build(False).plan_open_loop(0.0)
    assert milp.ok and lp.ok
    assert lp.objective <= milp.objective + 1e-6


# ---------------------------------------------------------------------------
# (f) infeasibility, status and fallback
# ---------------------------------------------------------------------------

def test_f_infeasible_status_and_fallback():
    limit = CallableForecast(lambda t, ts: np.where(ts < 2700, 5.0, 0.5))
    mpc = MPCController([storage_block("bess", capacity=0.2, p_charge_max=0.1), LoadBlock("load", 2.0),
                         GridBlock("grid", price_import=50.0, import_limit="limit")],
                        forecasts={"limit": limit}, horizon=2, dt=900, unserved_penalty=None, fallback="zero")
    mpc.init()
    assert mpc.diagnostics["solver_status"] == STATUS_OPTIMAL
    mpc.advance(0, 900)          # plan for [900, 2700): feasible
    assert mpc.diagnostics["solver_status"] == STATUS_OPTIMAL
    planned_1800 = float(mpc.last_plan["bess.P_cmd"][1])
    mpc.advance(900, 1800)       # horizon [1800, 3600): infeasible -> previous plan
    d = mpc.diagnostics
    assert d["solver_status"] == STATUS_INFEASIBLE and d["fallback"] == 1
    assert mpc.setpoints["bess.P_cmd"] == pytest.approx(planned_1800)
    mpc.advance(1800, 2700)      # no plan covers 2700 -> rule ('zero')
    d = mpc.diagnostics
    assert d["solver_status"] == STATUS_INFEASIBLE and d["fallback"] == 2
    assert all(v == 0.0 for v in mpc.setpoints.values())
    assert math.isnan(d["objective"])
    # callable rule
    mpc.fallback_rule = lambda t, m, f: {"bess.P_cmd": -0.05}
    mpc.advance(2700, 3600)
    assert mpc.setpoints["bess.P_cmd"] == -0.05 and mpc.diagnostics["fallback"] == 2


# ---------------------------------------------------------------------------
# (g) forecast providers
# ---------------------------------------------------------------------------

def test_g_forecast_providers(tmp_path):
    df = pd.DataFrame({"time": [0.0, 3600.0, 7200.0], "load": [1.0, 3.0, 2.0], "other": [0, 0, 0]})
    f = PerfectForecast(df, "load")
    assert np.allclose(f(0, [0, 1800, 3600, 9000]), [1.0, 2.0, 3.0, 2.0])
    csv = tmp_path / "p.csv"
    df.to_csv(csv, index=False)
    assert np.allclose(PerfectForecast(str(csv), "load")(0, [5400.0]), [2.5])
    s = pd.Series([5.0, 7.0], index=[0.0, 10.0])
    assert np.allclose(PerfectForecast(s)(0, [5.0]), [6.0])
    per = PerfectForecast(([0.0, 3600.0], [1.0, 2.0]), periodic=7200.0)
    assert np.allclose(per(0, [7200.0 + 1800.0]), [1.5])
    with pytest.raises(ValueError):
        PerfectForecast(df)  # ambiguous column

    pers = PersistenceForecast("load_now", default=4.0)
    assert np.allclose(pers(0, [0, 1, 2]), 4.0)
    assert np.allclose(pers(0, [0, 1], {"load_now": 2.5}), 2.5)
    assert np.allclose(pers(0, [0, 1], {}), 2.5)   # keeps last value
    pers.reset()
    assert np.allclose(pers(0, [0]), 4.0)

    cf = CallableForecast(lambda t, ts: 3.0)
    assert np.allclose(cf(0, [0, 1, 2]), 3.0)
    with pytest.raises(ValueError):
        CallableForecast(lambda t, ts: [1.0, 2.0])(0, [0, 1, 2])


# ---------------------------------------------------------------------------
# (h) OPFController vs direct pandapower runopp
# ---------------------------------------------------------------------------

def _opf_net():
    pp = pytest.importorskip("pandapower")
    net = pp.create_empty_network()
    b0 = pp.create_bus(net, 110.0, name="B0", min_vm_pu=0.9, max_vm_pu=1.1)
    b1 = pp.create_bus(net, 110.0, name="B1", min_vm_pu=0.9, max_vm_pu=1.1)
    pp.create_line(net, b0, b1, 10.0, "149-AL1/24-ST1A 110.0", max_loading_percent=100.0)
    eg = pp.create_ext_grid(net, b0, name="grid", min_p_mw=-100, max_p_mw=100)
    g = pp.create_sgen(net, b1, p_mw=5.0, name="pv", max_p_mw=10.0, min_p_mw=0.0, controllable=True)
    ld = pp.create_load(net, b1, p_mw=20.0, name="flex", max_p_mw=25.0, min_p_mw=5.0)
    pp.create_load(net, b1, p_mw=8.0, name="base")
    pp.create_poly_cost(net, eg, "ext_grid", cp1_eur_per_mw=40.0)
    pp.create_poly_cost(net, g, "sgen", cp1_eur_per_mw=5.0)
    pp.create_poly_cost(net, ld, "load", cp1_eur_per_mw=-30.0)
    return pp, net


def test_h_opf_controller_matches_direct_runopp():
    pp, net = _opf_net()
    from energysim.control import OPFController
    ctrl = OPFController(net, inputs=["pv.max_p_mw", "base.p_mw", "flex.cp1_eur_per_mw"],
                         outputs=["pv.p_mw", "flex.p_mw", "grid.p_mw", "B1.vm_pu", "cost",
                                  "max_line_loading_percent"], controllable=["flex"])
    ctrl.init()
    assert ctrl.diagnostics["solver_status"] == STATUS_OPTIMAL
    ctrl.set_value(["pv.max_p_mw", "base.p_mw", "flex.cp1_eur_per_mw"], [6.0, 12.0, -45.0])
    ctrl.advance(0, 900)
    assert ctrl.diagnostics["solver_status"] == STATUS_OPTIMAL

    ref = copy.deepcopy(ctrl._network0)   # the network as configured (controllable flex load)
    assert bool(ref.load.at[0, "controllable"]) and "controllable" not in net.load.columns
    ref.sgen.at[0, "max_p_mw"] = 6.0
    ref.load.at[1, "p_mw"] = 12.0
    ref.poly_cost.loc[(ref.poly_cost.et == "load") & (ref.poly_cost.element == 0), "cp1_eur_per_mw"] = -45.0
    pp.runopp(ref, numba=False)
    sp = ctrl.setpoints
    assert sp["pv.p_mw"] == pytest.approx(ref.res_sgen.p_mw[0], abs=1e-4)
    assert sp["flex.p_mw"] == pytest.approx(ref.res_load.p_mw[0], abs=1e-4)
    assert sp["grid.p_mw"] == pytest.approx(ref.res_ext_grid.p_mw[0], abs=1e-4)
    assert sp["cost"] == pytest.approx(ref.res_cost, rel=1e-6)
    assert ctrl.diagnostics["objective"] == pytest.approx(ref.res_cost, rel=1e-6)
    # the user's network was copied, not modified
    assert net.sgen.at[0, "max_p_mw"] == 10.0
    # strict names
    with pytest.raises(SimulatorVariableError):
        OPFController(net, inputs=["nope.p_mw"])
    with pytest.raises(SimulatorVariableError):
        OPFController(net, outputs=["pv.not_a_column"])
    with pytest.raises(SimulatorVariableError):
        ctrl.set_value(["pv.p_mw"], [1.0])


def test_h_opf_controller_in_world_drives_powerflow_plant(tmp_path, monkeypatch):
    pp, net = _opf_net()
    from energysim.control import OPFController
    ctrl = OPFController(net, inputs=["base.p_mw"], outputs=["flex.p_mw", "pv.p_mw"], controllable=["flex"])
    w = make_world(tmp_path, monkeypatch, 3600, 900)
    w.add_signal("base", lambda t: 8.0 + t / 900.0)
    w.add_model("opf", ctrl, step_size=900)
    w.add_powerflow("grid", model=copy.deepcopy(net), inputs=["flex.p_mw", "pv.p_mw"],
                    outputs=["flex.p_mw"], step_size=900)
    w.add_connections({"base.y": "opf.base.p_mw", "opf.flex.p_mw": "grid.flex.p_mw",
                       "opf.pv.p_mw": "grid.pv.p_mw"})
    run_world(w)
    assert len(ctrl.history) == 5
    assert all(h["solver_status"] == STATUS_OPTIMAL for h in ctrl.history)
    assert ctrl.history[-1]["measurements"]["base.p_mw"] == pytest.approx(8.0 + 3.0)


# ---------------------------------------------------------------------------
# (i) determinism, strict variable names, controller API
# ---------------------------------------------------------------------------

def test_i_rerun_determinism_and_strict_names(tmp_path, monkeypatch):
    dt, stop = 900, 6 * 3600
    plant = IdealStorage(step_size=dt)
    mpc = MPCController([storage_block(), LoadBlock("load", "load"), GridBlock("grid", price_import="price")],
                        forecasts={"price": NoisyForecast(CallableForecast(price_fc), 5.0, seed=3),
                                   "load": CallableForecast(load_fc)}, horizon=16, dt=dt)
    w = make_world(tmp_path, monkeypatch, stop, dt)
    w.add_model("store", plant, step_size=dt)
    w.add_model("mpc", mpc, step_size=dt)
    w.add_connections({"mpc.store.P_cmd": "store.P_cmd", "store.SoC": "mpc.store.SoC"})
    run_world(w)
    first = [(h["setpoints"], h["objective"], h["measurements"]) for h in mpc.history]
    log1 = list(plant.log)
    run_world(w)
    second = [(h["setpoints"], h["objective"], h["measurements"]) for h in mpc.history]
    assert first == second
    assert plant.log == log1

    avail = mpc.get_available_variables()
    assert avail["inputs"] == ["store.SoC"]
    assert {"store.P_cmd", "load.P_set", "grid.P_import", "grid.P_export", "solver_status", "solve_time_s",
            "objective", "fallback"} == set(avail["outputs"])
    with pytest.raises(SimulatorVariableError):
        mpc.set_value(["store.soc"], [0.5])
    with pytest.raises(SimulatorVariableError):
        mpc.set_value(["store.P_cmd"], [0.5])     # outputs are not writable
    with pytest.raises(SimulatorVariableError):
        mpc.get_value(["nope"], 0.0)
    assert mpc.get_value(["solver_status"], 0.0) == [0.0]

    class Bad(Controller):
        def decide(self, t, m, f):
            return {"typo": 1.0}

    bad = Bad(measurement_inputs=["x"], setpoint_outputs=["u"])
    bad.init()  # no initial measurements -> no decision at init
    assert bad.get_value(["u", "solver_status"]) == [0.0, -1.0]
    with pytest.raises(SimulatorVariableError):
        bad.step(0.0)
    with pytest.raises(ValueError):
        Bad(measurement_inputs=["x"], setpoint_outputs=["x"])
    with pytest.raises(ValueError):
        Bad(setpoint_outputs=["objective"])
    with pytest.raises(SimulatorVariableError):
        Bad(measurement_inputs=["x"], setpoint_outputs=["u"], initial_setpoints={"v": 1.0})


def test_i_block_from_component_library():
    from energysim import models as m
    comps = [m.BatteryBESS(), m.FlowBattery(), m.Flywheel(), m.PEMElectrolyzer(), m.AlkalineElectrolyzer(),
             m.IndustrialHeatPump(), m.ElectricBoiler(), m.ThermalStorageTank(), m.GasTurbine_OCGT(),
             m.ReciprocatingGenset()]
    blocks = [block_from_component(c) for c in comps]
    assert len({b.name for b in blocks}) == len(blocks)
    bess = blocks[0]
    assert 0.9 < bess.eta_c <= 1.0 and 0.9 < bess.eta_d <= 1.0
    mpc = MPCController(blocks + [
        GridBlock("grid", price_import=DAY_PRICES, price_export=0.0),
        GridBlock("h2market", carrier="hydrogen", import_limit=0.0, price_export=2.0),
        LoadBlock("heat", 15.0, carrier="heat"),
        LoadBlock("power", 20.0),
    ], horizon=12, dt=3600)
    plan = mpc.plan_open_loop(0.0)
    assert plan.ok
    with pytest.raises(TypeError):
        block_from_component(object())


# ---------------------------------------------------------------------------
# (j) NoisyForecast reproducibility
# ---------------------------------------------------------------------------

def test_j_noisy_forecast_seed_reproducible():
    base = CallableForecast(lambda t, ts: 10.0)
    times = np.arange(8) * 900.0
    a = NoisyForecast(base, sigma=1.0, seed=42, growth=0.5)
    b = NoisyForecast(base, sigma=1.0, seed=42, growth=0.5)
    c = NoisyForecast(base, sigma=1.0, seed=43, growth=0.5)
    xa1, xa2 = a(0, times), a(0, times)
    assert np.allclose(xa1, b(0, times)) and np.allclose(xa2, b(0, times))
    assert not np.allclose(xa1, c(0, times))
    assert not np.allclose(xa1, xa2)
    a.reset()
    assert np.allclose(a(0, times), xa1)
    ens = a.ensemble(0, times, 50)
    assert ens.shape == (50, 8)
    assert ens[:, -1].std() > ens[:, 0].std()   # uncertainty grows with lead time
    first = NoisyForecast(base, sigma=1.0, seed=1, exact_first=True)
    assert first(0, times)[0] == 10.0
    rel = NoisyForecast(CallableForecast(lambda t, ts: 0.0), sigma=0.3, seed=1, relative=True)
    assert np.allclose(rel(0, times), 0.0)


# ---------------------------------------------------------------------------
# PyPSA rolling horizon (optional)
# ---------------------------------------------------------------------------

def _pypsa_solver_works(pypsa):
    """linopy + HiGHS must actually solve (version clashes, e.g. with polars, are common)."""
    t = pypsa.Network()
    t.set_snapshots(range(2))
    t.add("Bus", "b")
    t.add("Generator", "g", bus="b", p_nom=1.0, marginal_cost=1.0)
    t.add("Load", "l", bus="b", p_set=0.5)
    try:
        status, _ = t.optimize(solver_name="highs", log_to_console=False)
    except Exception:
        return False
    return status == "ok"


def test_pypsa_rolling_horizon_first_step_matches_direct_optimize():
    pypsa = pytest.importorskip("pypsa")
    pytest.importorskip("linopy")
    pytest.importorskip("highspy")
    from energysim.control import PyPSARollingHorizonController
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if not _pypsa_solver_works(pypsa):
            pytest.skip("pypsa/linopy/highs installed but unable to solve in this environment")

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2024-01-01", periods=24, freq="h"))
    n.add("Bus", "el")
    n.add("Generator", "grid", bus="el", p_nom=10.0, marginal_cost=pd.Series(DAY_PRICES, index=n.snapshots))
    n.add("Load", "load", bus="el", p_set=pd.Series(0.5, index=n.snapshots))
    n.add("StorageUnit", "bess", bus="el", p_nom=1.0, max_hours=4.0, efficiency_store=0.95,
          efficiency_dispatch=0.95, state_of_charge_initial=2.0, cyclic_state_of_charge=False)
    ctrl = PyPSARollingHorizonController(
        n, window=8, setpoints={"bess.P_cmd": ("StorageUnit", "p", "bess", -1.0)},
        measurements={"bess.SoC": ("StorageUnit", "state_of_charge_initial", "bess", 4.0)},
        initial_measurements={"bess.SoC": 0.5}, actuation_delay=False)
    assert ctrl.snapshot_seconds == 3600.0
    ctrl.init()
    assert ctrl.diagnostics["solver_status"] == STATUS_OPTIMAL

    ref = copy.deepcopy(n)
    ref.optimize(snapshots=ref.snapshots[:8], solver_name="highs", log_to_console=False)
    assert ctrl.setpoints["bess.P_cmd"] == pytest.approx(-ref.storage_units_t.p.iloc[0]["bess"], abs=1e-6)
    ctrl.set_value(["bess.SoC"], [0.25])
    ctrl.advance(3600.0, 7200.0)
    assert ctrl.diagnostics["solver_status"] == STATUS_OPTIMAL
    assert ctrl.network.storage_units.at["bess", "state_of_charge_initial"] == pytest.approx(1.0)
    with pytest.raises(SimulatorVariableError):
        PyPSARollingHorizonController(n, window=4, setpoints={"x": ("StorageUnit", "p", "nope")})

# -*- coding: utf-8 -*-
"""Numerical behaviour of the co-simulation orchestrator: convergence order of the
coupling schemes against analytic solutions, extrapolation, and conservation laws
across component chains."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from energysim import SimulatorAdapter, world
from energysim.models import BatteryBESS, H2StorageTank, PEMElectrolyzer, PhysicsPV
from energysim.models.hydrogen import HHV_H2_MJ_KG
from helpers import run_and_get
from refs import assert_ref

pytestmark = pytest.mark.validation


class LinearSystem(SimulatorAdapter):
    """x' = a*x + sum_j b_j*u_j, integrated exactly over each interval with u held constant."""

    def __init__(self, a, b, x0, inputs=("u",)):
        self.a, self.b, self.x0 = a, dict(b), x0
        self.x = x0
        self.u = {k: 0.0 for k in inputs}
        self.step_size = 1.0

    def init(self):
        self.x = self.x0
        self.u = {k: 0.0 for k in self.u}

    def advance(self, t0, t1):
        h = t1 - t0
        e = math.exp(self.a * h)
        forcing = sum(self.b[k] * v for k, v in self.u.items())
        self.x = e * self.x + (e - 1.0) / self.a * forcing

    def step(self, time):
        self.advance(time, time + self.step_size)

    def advance_with_recording(self, t0, t1, outputs):
        rows = [[t0] + [self.x for _ in outputs]]
        self.advance(t0, t1)
        return rows

    def get_value(self, parameters, time):
        return [self.x for _ in parameters]

    def set_value(self, parameters, values):
        for p, v in zip(parameters, values):
            self.u[p] = v


_A, _C, _D = -0.1, -0.3, 0.5


def _cascade_error(h, coupling, tmp_path, extrapolation="zero-order", horizon=40.0):
    """A: x' = a x (x0=1), B: y' = c y + d x ; error of y(T) vs the analytic solution."""
    w = world(start_time=0, stop_time=horizon, t_macro=h, coupling=coupling, extrapolation=extrapolation,
              res_filename=str(tmp_path / f"c_{coupling}_{h}.h5"))
    a_sys = LinearSystem(_A, {"u": 0.0}, 1.0)
    b_sys = LinearSystem(_C, {"u": _D}, 0.0)
    w.add_model("A", a_sys, inputs=["u"], outputs=["x"], step_size=h)
    w.add_model("B", b_sys, inputs=["u"], outputs=["x"], step_size=h)
    w.add_connections({"A.x": "B.u"})
    w.simulate(pbar=False, record_all=False)
    y_exact = _D / (_A - _C) * (math.exp(_A * horizon) - math.exp(_C * horizon))
    return abs(b_sys.x - y_exact)


def _observed_order(errs):
    return math.log2(errs[-2] / errs[-1])


@pytest.mark.parametrize("coupling", ["jacobi", "gauss-seidel"])
def test_cascade_converges_first_order(coupling, tmp_path):
    hs = [1.0, 0.5, 0.25, 0.125]
    errs = [_cascade_error(h, coupling, tmp_path) for h in hs]
    assert all(b < a for a, b in zip(errs, errs[1:]))
    assert_ref(_observed_order(errs), "orchestrator", f"{coupling}_cascade_convergence_order", expected=1.0)


def test_gauss_seidel_order_not_below_jacobi(tmp_path):
    hs = [1.0, 0.5, 0.25, 0.125]
    oj = _observed_order([_cascade_error(h, "jacobi", tmp_path) for h in hs])
    og = _observed_order([_cascade_error(h, "gauss-seidel", tmp_path) for h in hs])
    assert_ref(og - oj, "orchestrator", "gauss_seidel_order_minus_jacobi_order", expected=0.0)


def _cyclic_error(h, coupling, tmp_path, horizon=20.0):
    """Two-way coupled linear system x' = Mx split over two adapters; error vs expm(M t)."""
    scipy_linalg = pytest.importorskip("scipy.linalg")
    m = np.array([[-0.2, 0.3], [-0.4, -0.5]])
    x0 = np.array([1.0, 0.5])
    w = world(start_time=0, stop_time=horizon, t_macro=h, coupling=coupling,
              res_filename=str(tmp_path / f"cy_{coupling}_{h}.h5"))
    s1 = LinearSystem(m[0, 0], {"u": m[0, 1]}, x0[0])
    s2 = LinearSystem(m[1, 1], {"u": m[1, 0]}, x0[1])
    w.add_model("S1", s1, inputs=["u"], outputs=["x"], step_size=h)
    w.add_model("S2", s2, inputs=["u"], outputs=["x"], step_size=h)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # algebraic-loop warning is expected
        w.add_connections({"S1.x": "S2.u", "S2.x": "S1.u"})
        w.simulate(pbar=False, record_all=False)
    exact = scipy_linalg.expm(m * horizon) @ x0
    return float(np.linalg.norm(np.array([s1.x, s2.x]) - exact)), (s1.x, s2.x)


def test_cyclic_system_jacobi_first_order_and_gs_degenerates_to_jacobi(tmp_path):
    hs = [0.5, 0.25, 0.125, 0.0625]
    errs = [_cyclic_error(h, "jacobi", tmp_path)[0] for h in hs]
    assert_ref(_observed_order(errs), "orchestrator", "cyclic_jacobi_convergence_order", expected=1.0)
    # inside a strongly connected component Gauss-Seidel falls back to Jacobi (documented one-step delay)
    _, xj = _cyclic_error(0.25, "jacobi", tmp_path)
    _, xg = _cyclic_error(0.25, "gauss-seidel", tmp_path)
    assert_ref(xg[0], "orchestrator", "cyclic_gs_equals_jacobi", expected=xj[0])
    assert_ref(xg[1], "orchestrator", "cyclic_gs_equals_jacobi", expected=xj[1])


class _Hold(SimulatorAdapter):
    """Receiver that records the input it holds during every macro step."""

    def __init__(self):
        self.u = 0.0
        self.log = []
        self.step_size = 1.0

    def init(self):
        self.u, self.log = 0.0, []

    def advance(self, t0, t1):
        self.log.append((t0, self.u))

    def step(self, time):
        self.advance(time, time + self.step_size)

    def get_value(self, parameters, time):
        return [self.u for _ in parameters]

    def set_value(self, parameters, values):
        self.u = values[0]


def _extrapolation_error(h, mode, tmp_path):
    w = world(start_time=0, stop_time=60.0, t_macro=h, extrapolation=mode,
              res_filename=str(tmp_path / f"ex_{mode}_{h}.h5"))
    w.add_signal("src", signal=lambda t: [math.sin(0.2 * t)], step_size=h)
    rx = _Hold()
    w.add_model("rx", rx, inputs=["u"], outputs=["u"], step_size=h)
    w.add_connections({"src.y": "rx.u"})
    w.simulate(pbar=False, record_all=False)
    # linear extrapolation predicts the signal at the END of the macro step
    errs = [abs(u - math.sin(0.2 * (t + h))) for t, u in rx.log[2:]]
    return float(np.mean(errs))


def test_linear_extrapolation_improves_prediction_of_smooth_signal(tmp_path):
    e_zoh = [_extrapolation_error(h, "zero-order", tmp_path) for h in (1.0, 0.5)]
    e_lin = [_extrapolation_error(h, "linear", tmp_path) for h in (1.0, 0.5)]
    assert e_lin[0] < 0.3 * e_zoh[0] and e_lin[1] < 0.15 * e_zoh[1]
    assert_ref(math.log2(e_zoh[0] / e_zoh[1]), "orchestrator", "zero_order_hold_prediction_order", expected=1.0)
    assert_ref(math.log2(e_lin[0] / e_lin[1]), "orchestrator", "linear_extrapolation_prediction_order", expected=2.0)


# ---------------------------------------------------------------------------
# Conservation across PV -> BESS -> electrolyzer -> tank
# ---------------------------------------------------------------------------

def _build_chain(tmp_path, seed, steps=240, macro=60.0):
    rng = np.random.default_rng(seed)
    g = np.clip(rng.normal(650.0, 350.0, steps + 2), 0.0, 1200.0)
    demand = rng.uniform(0.0, 150.0, steps + 2)
    w = world(start_time=0, stop_time=steps * macro, t_macro=macro, power_unit="MW",
              res_filename=str(tmp_path / f"chain_{seed}.h5"))
    w.add_signal("sun", signal=lambda t: [float(g[min(int(t // macro), steps + 1)])], step_size=macro)
    w.add_signal("draw", signal=lambda t: [float(demand[min(int(t // macro), steps + 1)])], step_size=macro,
                 units={"y": "kg/h"})
    pv = PhysicsPV(p_dc_stc_mw=5.0, inverter_p_ac_max_mw=4.5)
    bess = BatteryBESS(capacity_mwh=0.6, p_max_mw=1.5, soc_init=0.5)
    ely = PEMElectrolyzer(p_nominal_mw=5.0, ramp_rate_pct_per_s=0.05)
    tank = H2StorageTank(volume_m3=4.0, p_init_bar=60.0, p_max_bar=120.0)
    w.add_model("pv", pv, step_size=macro, outputs=["P_ac"])
    w.add_model("bess", bess, step_size=macro,
                outputs=["P_actual", "SoC", "SOH", "Q_heat_kW"])
    w.add_model("ely", ely, step_size=macro,
                outputs=["P_actual", "Q_waste_MW", "m_dot_H2_kg_hr", "eta_faraday"])
    w.add_model("tank", tank, step_size=macro,
                outputs=["mass_stored_kg", "m_dot_rejected_kg_hr", "m_dot_unserved_kg_hr"])
    w.add_connections([
        {"from": "sun.y", "to": "pv.G_poa", "scale": 1.0},
        {"from": "pv.P_ac", "to": "bess.P_cmd", "scale": 0.3},
        {"from": "pv.P_ac", "to": "ely.P_cmd", "scale": 0.7},
        {"from": "ely.m_dot_H2_kg_hr", "to": "tank.m_dot_in_kg_hr"},
        {"from": "draw.y", "to": "tank.m_dot_out_kg_hr"},
    ])
    return w, bess, macro, demand


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_chain_conservation_pv_bess_electrolyzer_tank(seed, tmp_path):
    w, bess, macro, demand = _build_chain(tmp_path, seed)
    res = run_and_get(w, record_all=False)
    pv, b, e, t = res["pv"], res["bess"], res["ely"], res["tank"]
    h_h = macro / 3600.0
    n = len(b)
    assert n >= 200

    # --- PV split is complete and bounded
    assert (pv["P_ac"] >= 0).all() and (pv["P_ac"] <= 4.5 + 1e-12).all()

    # --- BESS: AC energy = stored + joule heat + inverter loss, every step
    worst = 0.0
    for k in range(n - 1):
        p_ac = b["P_actual"].iloc[k + 1]
        eta = float(bess.inverter_curve(abs(p_ac) / 1.5)) if p_ac != 0 else 1.0   # efficiency at the delivered power
        p_dc = p_ac * eta if p_ac > 0 else p_ac / eta
        d_stored = (b["SoC"].iloc[k + 1] - b["SoC"].iloc[k]) * 0.6 * b["SOH"].iloc[k]
        resid = p_ac * h_h - (d_stored + b["Q_heat_kW"].iloc[k + 1] / 1000.0 * h_h + (p_ac - p_dc) * h_h)
        worst = max(worst, abs(resid))
    assert_ref(worst, "orchestrator", "chain_bess_energy_residual_MWh", expected=0.0)
    assert b["SoC"].between(0.05 - 1e-9, 0.95 + 1e-9).all()

    # --- electrolyzer: electricity = waste heat + hydrogen HHV energy (before crossover)
    # below the thermoneutral voltage the heat term is clipped to zero (endothermic regime not modelled)
    run = e[e["Q_waste_MW"] > 0.0]
    assert len(run) > 20
    h2_ideal = run["m_dot_H2_kg_hr"] / 3600.0 / run["eta_faraday"]
    resid = (run["Q_waste_MW"] + h2_ideal * HHV_H2_MJ_KG - run["P_actual"]).abs() / run["P_actual"]
    assert_ref(resid.max(), "orchestrator", "chain_electrolyzer_energy_relative_residual", expected=0.0)
    assert (e["P_actual"].iloc[1:].to_numpy() <= 0.7 * pv["P_ac"].iloc[:-1].to_numpy() + 0.0 + 5.0).all()
    assert (e["P_actual"] <= 5.0 + 1e-12).all()

    # --- tank: mass conservation including rejected / unserved flows
    worst_m = 0.0
    for k in range(n - 1):
        dm = t["mass_stored_kg"].iloc[k + 1] - t["mass_stored_kg"].iloc[k]
        flow = (e["m_dot_H2_kg_hr"].iloc[k] - demand[k]
                - t["m_dot_rejected_kg_hr"].iloc[k + 1] + t["m_dot_unserved_kg_hr"].iloc[k + 1]) * h_h
        worst_m = max(worst_m, abs(dm - flow))
    assert_ref(worst_m, "orchestrator", "chain_tank_mass_residual_kg", expected=0.0)

    # --- end to end: every kg of hydrogen produced is stored, delivered, rejected or accounted unserved
    produced = e["m_dot_H2_kg_hr"].iloc[:-1].sum() * h_h
    stored = t["mass_stored_kg"].iloc[-1] - t["mass_stored_kg"].iloc[0]
    delivered = demand[: n - 1].sum() * h_h
    rejected = t["m_dot_rejected_kg_hr"].iloc[1:].sum() * h_h
    unserved = t["m_dot_unserved_kg_hr"].iloc[1:].sum() * h_h
    assert_ref(produced - delivered - rejected + unserved, "orchestrator", "chain_hydrogen_end_to_end_kg",
               expected=stored)
    assert produced > 0.0

# -*- coding: utf-8 -*-
"""Strength 1 - adapter uniformity: very different simulator kinds satisfy one interface
and behave identically from the world's point of view."""
from __future__ import annotations

import pytest

from energysim import SimulatorAdapter, world
from energysim.models import PhysicsPV
from helpers import run_and_get

pytestmark = pytest.mark.strength


class _Echo(SimulatorAdapter):
    """External-simulator style adapter: o = 2 * u (after each step)."""

    def __init__(self):
        self.u = 0.0
        self.o = 0.0

    def init(self):
        self.u = self.o = 0.0

    def step(self, time):
        self.o = 2.0 * self.u

    def get_value(self, parameters, time):
        return [self.o if p == "o" else self.u for p in parameters]

    def set_value(self, parameters, values):
        for p, v in zip(parameters, values):
            if p == "u":
                self.u = v


def _csv(w, tmp):
    p = tmp / "d.csv"
    p.write_text("time,v\n0,1\n10,2\n20,3\n30,4\n")
    w.add_csv("x", str(p), step_size=10, outputs=["v"])
    return None, "v"


def _signal(w, tmp):
    w.add_signal("x", "2*t", step_size=10)
    return None, "y"


def _script(w, tmp):
    (tmp / "ctrl.py").write_text("def run(inputs):\n    return [2.0 * inputs.get('u', 0.0)]\n")
    w.add_script(script_name="run", script_loc=str(tmp / "ctrl.py"), sim_name="x",
                 inputs=["u"], outputs=["o"], step_size=10)
    return "u", "o"


def _external(w, tmp):
    w.add_external_simulator("x", model=_Echo, inputs=["u"], outputs=["o"], step_size=10)
    return "u", "o"


def _model(w, tmp):
    w.add_model("x", PhysicsPV(), inputs=["G_poa"], outputs=["P_ac"], step_size=10)
    return "G_poa", "P_ac"


def _fleet(w, tmp):
    w.add_container("x", kind="pv", size=4, inputs=["irradiance_w_m2"], outputs=["total_p_kw"], step_size=10)
    return "irradiance_w_m2", "total_p_kw"


def _building_fleet(w, tmp):
    w.add_container("x", kind="building", size=4, inputs=["t_ambient"], outputs=["mean_t_indoor"], step_size=10)
    return "t_ambient", "mean_t_indoor"


def _pandapower(w, tmp):
    pp = pytest.importorskip("pandapower")
    net = pp.create_empty_network(name="n")
    b1 = pp.create_bus(net, vn_kv=20.0, name="B1")
    b2 = pp.create_bus(net, vn_kv=20.0, name="B2")
    pp.create_ext_grid(net, b1, name="G")
    pp.create_line_from_parameters(net, b1, b2, length_km=1.0, r_ohm_per_km=0.1, x_ohm_per_km=0.1,
                                   c_nf_per_km=0.0, max_i_ka=1.0, name="L1")
    pp.create_load(net, b2, p_mw=1.0, name="Ld")
    w.add_powerflow("x", net=net, inputs=["Ld.p_mw"], outputs=["B2.vm_pu"], step_size=10)
    return "Ld.p_mw", "B2.vm_pu"


KINDS = {
    "csv": _csv, "signal": _signal, "script": _script, "external": _external,
    "model": _model, "pv_fleet": _fleet, "building_fleet": _building_fleet, "pandapower": _pandapower,
}


def _build(kind, tmp_path, with_driver=True):
    w = world(start_time=0, stop_time=40, t_macro=10)
    in_var, out_var = KINDS[kind](w, tmp_path)
    if in_var and with_driver:
        w.add_signal("drv", "5.0", step_size=10)
        w.add_connections({"drv.y": f"x.{in_var}"})
    return w, in_var, out_var


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_adapter_contract_smoke(kind, tmp_path):
    w, in_var, out_var = _build(kind, tmp_path, with_driver=False)
    a = w.get_adapter("x")
    for name in ("init", "step", "get_value", "set_value", "advance", "cleanup", "get_available_variables"):
        assert callable(getattr(a, name, None)), f"{kind}: missing {name}"
    a.init()
    if in_var:
        a.set_value([in_var], [3.0])
    a.step(0.0)
    val = a.get_value([out_var], 0.0)
    assert isinstance(val, list) and len(val) == 1
    a.advance(0.0, 10.0)
    av = a.get_available_variables()
    assert set(av) >= {"inputs", "outputs"} and isinstance(av["outputs"], list)
    a.cleanup()


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_every_kind_runs_in_world_and_is_rerunnable(kind, tmp_path):
    w, in_var, out_var = _build(kind, tmp_path)
    first = run_and_get(w, record_all=False)["x"]
    assert out_var in first.columns and len(first) == 5  # 4 macro steps + final row at stop_time
    assert first[out_var].notna().all()
    try:
        second = run_and_get(w, record_all=False)["x"]
    except AttributeError as exc:     # pragma: no cover - known adapter bug, reported
        if kind == "pandapower" and "network" in str(exc):
            pytest.xfail("pp_adapter.cleanup() deletes self.network, so the world cannot be re-run")
        raise
    assert first[out_var].tolist() == pytest.approx(second[out_var].tolist(), nan_ok=False)


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_kinds_are_interchangeable_as_sources(kind, tmp_path):
    """The same downstream consumer accepts the output of any adapter kind, unchanged."""
    w, in_var, out_var = _build(kind, tmp_path)
    w.add_external_simulator("sink", model=_Echo, inputs=["u"], outputs=["o"], step_size=10)
    w.add_connections({"drv.y": f"x.{in_var}", f"x.{out_var}": "sink.u"} if in_var
                      else {f"x.{out_var}": "sink.u"})
    res = run_and_get(w, record_all=False)
    assert res["sink"]["o"].notna().all()
    assert res["sink"]["o"].iloc[-1] == pytest.approx(2.0 * float(res["x"][out_var].iloc[-2]))

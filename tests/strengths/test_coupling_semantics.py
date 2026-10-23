# -*- coding: utf-8 -*-
"""Strength 2 - coupling semantics: topological ordering, loop detection and the
Jacobi / Gauss-Seidel lag behaviour, derived from first principles with stateless
algebraic adapters."""
from __future__ import annotations

import warnings

import pytest

from energysim import world
from helpers import Alg, run_and_get

pytestmark = pytest.mark.strength


def _chain_world(tmp_path, coupling, names=("A", "B", "C"), reverse_registration=False, macro=10.0, steps=8):
    w = world(start_time=0, stop_time=macro * steps, t_macro=macro, coupling=coupling,
              res_filename=str(tmp_path / f"{coupling}.h5"))
    w.add_signal("S", "t", step_size=macro)
    order = list(names)[::-1] if reverse_registration else list(names)
    for n in order:
        w.add_model(n, Alg(), inputs=["u"], outputs=["y"], step_size=macro)
    links = {"S.y": f"{names[0]}.u"}
    for up, down in zip(names, names[1:]):
        links[f"{up}.y"] = f"{down}.u"
    w.add_connections(links)
    return w


def _order(w):
    return list(getattr(w, "execution_order", None) or w._execution_order)


def test_topological_order_independent_of_registration_order(tmp_path):
    w = _chain_world(tmp_path, "jacobi", reverse_registration=True)
    order = _order(w)
    assert order.index("S") < order.index("A") < order.index("B") < order.index("C")


def test_jacobi_lags_one_macro_step_per_stage(tmp_path):
    macro = 10.0
    w = _chain_world(tmp_path, "jacobi", macro=macro)
    res = run_and_get(w, record_all=False)
    # row k holds the output computed in step k-1; the source value at the exchange of step j is j*macro.
    # Jacobi: stage s (A=1, B=2, C=3) output after step j shows the source value of step j-(s-1).
    for stage, name in enumerate("ABC", start=1):
        y = res[name]["y"].tolist()
        for k in range(1, len(y)):
            j = k - 1                                   # step that produced row k
            expected = max(0, j - (stage - 1)) * macro if j - (stage - 1) >= 0 else 0.0
            assert y[k] == pytest.approx(expected), (name, k, y)


def test_gauss_seidel_downstream_sees_fresh_upstream_output(tmp_path):
    macro = 10.0
    w = _chain_world(tmp_path, "gauss-seidel", macro=macro)
    res = run_and_get(w, record_all=False)
    for name in "ABC":
        y = res[name]["y"].tolist()
        for k in range(1, len(y)):
            assert y[k] == pytest.approx((k - 1) * macro), (name, k, y)   # no lag at all


def test_gauss_seidel_equals_jacobi_for_first_stage_and_beats_it_downstream(tmp_path):
    rj = run_and_get(_chain_world(tmp_path, "jacobi"), record_all=False)
    rg = run_and_get(_chain_world(tmp_path, "gauss-seidel"), record_all=False)
    assert rj["A"]["y"].tolist() == rg["A"]["y"].tolist()
    # the downstream lag of Jacobi is exactly (stage - 1) macro steps
    assert rj["C"]["y"].tolist()[3:] == rg["C"]["y"].tolist()[1:-2]


def _loop_world(tmp_path, coupling):
    w = world(start_time=0, stop_time=60, t_macro=10, coupling=coupling,
              res_filename=str(tmp_path / f"loop_{coupling}.h5"))
    w.add_signal("S", "1", step_size=10)
    w.add_model("P", Alg(gain=0.5), inputs=["u"], outputs=["y"], step_size=10)
    w.add_model("Q", Alg(gain=0.5), inputs=["u"], outputs=["y"], step_size=10)
    w.add_model("R", Alg(), inputs=["u"], outputs=["y"], step_size=10)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        w.add_connections([{"from": "S.y", "to": "P.u"}, {"from": "P.y", "to": "Q.u"},
                           {"from": "Q.y", "to": "P.u"}, {"from": "Q.y", "to": "R.u"}])
    return w, caught


@pytest.mark.parametrize("coupling", ["jacobi", "gauss-seidel"])
def test_scc_detection_warning_and_validation_advice(coupling, tmp_path):
    w, caught = _loop_world(tmp_path, coupling)
    assert any("loop" in str(c.message).lower() for c in caught)
    sccs = getattr(w, "_cycle_sccs", None)
    if sccs is not None:
        assert [set(s) for s in sccs] == [{"P", "Q"}]
    rep = w.validate()
    assert rep.is_valid
    assert any("feedback loop" in m and "P" in m and "Q" in m for m in rep.warnings)
    order = _order(w)
    assert order.index("S") < min(order.index("P"), order.index("Q")) and order.index("R") > order.index("Q")


def test_feedback_loop_uses_one_step_delay_and_converges(tmp_path):
    """P.y = 0.5*(S + Q.y) style loop: with a one-step delay the iteration converges to the fixed point."""
    macro = 10.0
    w = world(start_time=0, stop_time=macro * 40, t_macro=macro, res_filename=str(tmp_path / "fp.h5"))
    w.add_signal("S", "1", step_size=macro)
    w.add_model("P", Alg(gain=0.5, bias=0.5), inputs=["u"], outputs=["y"], step_size=macro)   # y = 0.5 u + 0.5
    w.add_model("Q", Alg(gain=0.5), inputs=["u"], outputs=["y"], step_size=macro)             # y = 0.5 u
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w.add_connections({"P.y": "Q.u", "Q.y": "P.u"})
        res = run_and_get(w, record_all=False)
    # fixed point: y_p = 0.5 y_q + 0.5, y_q = 0.5 y_p  ->  y_p = 2/3, y_q = 1/3
    assert res["P"]["y"].iloc[-1] == pytest.approx(2.0 / 3.0, abs=1e-6)
    assert res["Q"]["y"].iloc[-1] == pytest.approx(1.0 / 3.0, abs=1e-6)

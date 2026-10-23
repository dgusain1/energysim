# -*- coding: utf-8 -*-
"""Tests for the rewritten energysim.Sweep (built on Ensemble)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energysim import Sweep
from energysim.ensemble import EnsembleError
from test_ensemble import CAP_KWH, T_STOP, build, factory

DT_ERR = 60 / 3600  # one macro step of tolerance (kWh per kW) in the closed form


def info(**kw):
    d = {"parameters": ["bat.cap"], "bounds": [[50.0, 200.0, 4]], "output_parameter": "bat.soc"}
    d.update(kw)
    return d


def soc_closed_form(p_off, cap, soc0=0.2, p=0.0):
    return soc0 + (p + p_off) * (T_STOP / 3600.0) / cap


def test_single_sweep_model_parameter_via_init():
    w = build(10.0)
    res = Sweep(w, info(), kind="single").sweep()
    t = res.table
    assert list(t.columns) == ["parameter", "value", "result"] and len(t) == 4
    exp = soc_closed_form(10.0, t["value"].to_numpy())
    assert np.allclose(t["result"], exp, atol=10.0 * DT_ERR / 50.0)
    assert res.baseline["values"]["bat.cap"] == CAP_KWH
    assert res.baseline["result"] == pytest.approx(soc_closed_form(10.0, CAP_KWH), abs=10.0 * DT_ERR / CAP_KWH)


def test_world_instance_is_not_modified():
    w = build(10.0)
    Sweep(w, info(), kind="single").sweep()
    assert w.init_dict == {} and w.last_results is None
    again = w.simulate(pbar=False)["bat.soc"].iloc[-1]
    assert again == pytest.approx(soc_closed_form(10.0, CAP_KWH), abs=10.0 * DT_ERR / CAP_KWH)


def test_parameter_order_does_not_change_results():
    w = build(10.0)
    two = {"parameters": ["bat.cap", "bat.p_off"], "bounds": [[50.0, 100.0, 3], [0.0, 10.0, 3]],
           "output_parameter": "bat.soc"}
    rev = {"parameters": ["bat.p_off", "bat.cap"], "bounds": [[0.0, 10.0, 3], [50.0, 100.0, 3]],
           "output_parameter": "bat.soc"}
    a = Sweep(w, two, kind="single").sweep().table.set_index(["parameter", "value"])["result"]
    b = Sweep(w, rev, kind="single").sweep().table.set_index(["parameter", "value"])["result"]
    assert np.array_equal(a.sort_index().to_numpy(), b.sort_index().to_numpy())
    pa = Sweep(w, two, kind="parametric").sweep().table
    pb = Sweep(w, rev, kind="parametric").sweep().table
    pa = pa.set_index(["bat.cap", "bat.p_off"])["result"].sort_index()
    pb = pb.set_index(["bat.cap", "bat.p_off"])["result"].sort_index()
    assert np.array_equal(pa.to_numpy(), pb.to_numpy())


def test_single_no_state_leakage_other_params_stay_baseline():
    w = build(10.0)
    two = {"parameters": ["bat.cap", "bat.p_off"], "bounds": [[50.0, 100.0, 2], [0.0, 10.0, 2]],
           "output_parameter": "bat.soc"}
    t = Sweep(w, two, kind="single").sweep().table
    cap_rows = t[t["parameter"] == "bat.cap"]
    # p_off must be at its baseline (0) in the cap sweep, cap at 100 in the p_off sweep
    assert np.allclose(cap_rows["result"], soc_closed_form(10.0, cap_rows["value"]), atol=0.05)
    off_rows = t[t["parameter"] == "bat.p_off"]
    assert np.allclose(off_rows["result"], soc_closed_form(10.0, CAP_KWH, p=0.0) + off_rows["value"] / CAP_KWH,
                       atol=0.05)


def test_parametric_grid_and_closed_form():
    w = build(10.0)
    pi = {"parameters": ["bat.cap", "bat.p_off"], "bounds": [[50.0, 100.0, 3], [0.0, 20.0, 4]],
          "output_parameter": "bat.soc"}
    res = Sweep(w, pi, kind="parametric").sweep()
    t = res.table
    assert list(t.columns) == ["bat.cap", "bat.p_off", "result"] and len(t) == 12
    exp = soc_closed_form(t["bat.p_off"], t["bat.cap"], p=10.0)
    assert np.allclose(t["result"], exp, atol=30.0 * DT_ERR / 50.0)


def test_parametric_requires_two_parameters_and_bad_kind():
    with pytest.raises(ValueError):
        Sweep(build(1.0), info(), kind="parametric")
    with pytest.raises(ValueError):
        Sweep(build(1.0), info(), kind="bogus")
    with pytest.raises(KeyError):
        Sweep(build(1.0), {"parameters": ["bat.cap"]})
    with pytest.raises(ValueError):
        Sweep(build(1.0), info(metric="median"))


def test_sweep_via_factory_and_metrics():
    def get(metric):
        i = {"parameters": ["p"], "bounds": [[0.0, 20.0, 3]], "output_parameter": "bat.soc", "metric": metric}
        return Sweep(factory, i, kind="single").sweep().table["result"].to_numpy()

    last, mx, mn = get("last"), get("max"), get("min")
    assert np.allclose(last, [0.2, 0.2 + 10 / CAP_KWH, 0.2 + 20 / CAP_KWH], atol=0.01)
    assert np.allclose(mx, last) and np.allclose(mn, 0.2, atol=0.01)
    mean, total = get("mean"), get("sum")
    assert np.all(mean <= mx + 1e-12) and np.all(total >= 0)
    integ = get("integral")
    assert integ[0] == pytest.approx(0.2 * (T_STOP - 60), rel=0.02)


def test_export_to_csv_and_plot(tmp_path):
    res = Sweep(build(10.0), info(), kind="single").sweep()
    out = tmp_path / "s.csv"
    res.export_to_csv(out)
    back = pd.read_csv(out)
    assert list(back.columns) == ["parameter", "value", "result"] and len(back) == 4
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    fig = res.plot()
    assert fig.axes
    fig2 = res.plot(kind="curves")
    assert fig2.axes[0].lines
    import matplotlib.pyplot as plt
    plt.close("all")
    pi = {"parameters": ["bat.cap", "bat.p_off"], "bounds": [[50.0, 100.0, 3], [0.0, 20.0, 4]],
          "output_parameter": "bat.soc"}
    pres = Sweep(build(10.0), pi, kind="parametric").sweep()
    assert pres.plot().axes
    plt.close("all")
    three = {"parameters": ["bat.cap", "bat.p_off", "bat.soc0"],
             "bounds": [[50.0, 100.0, 2], [0.0, 1.0, 2], [0.1, 0.2, 2]], "output_parameter": "bat.soc"}
    with pytest.raises(ValueError, match="exactly two"):
        Sweep(build(10.0), three, kind="parametric").sweep().plot()


def test_non_copyable_world_requires_factory():
    w = build(10.0)
    w._lock = __import__("threading").Lock()
    with pytest.raises(EnsembleError, match="factory"):
        Sweep(w, info())


def test_default_n_points_is_ten():
    res = Sweep(build(10.0), {"parameters": ["bat.cap"], "bounds": [[50.0, 100.0]],
                              "output_parameter": "bat.soc"}).sweep()
    assert len(res.table) == 10
    with pytest.raises(ValueError):
        Sweep(build(10.0), {"parameters": ["bat.cap"], "bounds": [[1.0]], "output_parameter": "bat.soc"})

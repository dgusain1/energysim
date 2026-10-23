# -*- coding: utf-8 -*-
"""Tests for energysim.ensemble (samplers, Ensemble, EnsembleResults, NoisyInput)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energysim import Ensemble, EnsembleResults, noisy_signal, world
from energysim.ensemble import EnsembleError, Grid, LatinHypercube, Normal, Scenarios, Sobol, Uniform, apply_params
from energysim.ensemble.noise import NoisyInput
from energysim.models.common import ModelComponent

CAP_KWH = 100.0
T_STOP = 3600


class Bess(ModelComponent):
    """SoC = soc0 + integral((p + p_off) dt) / (3600 * cap); closed form for tests."""
    units = {"p": "kW", "soc": "1"}

    def __init__(self):
        super().__init__(name="bat")
        self.cap = CAP_KWH
        self.soc0 = 0.2
        self.p_off = 0.0
        self.energy = 0.0
        self.register_variables(inputs=["p"], outputs=["soc"])
        self.register_parameters("cap", "soc0", "p_off")

    def step_model(self, dt, time):
        self.energy += (self._inputs_val["p"] + self.p_off) * dt / 3600.0
        self._outputs_val["soc"] = self.soc0 + self.energy / self.cap


def build(power, sigma=0.0, seed=0):
    w = world(start_time=0, stop_time=T_STOP, t_macro=60)
    w.add_signal("cmd", noisy_signal(power, sigma, seed=seed, rel=True, ar1=0.9, dt=60.0) if sigma else power,
                 step_size=60)
    w.add_model("bat", Bess(), step_size=60)
    w.add_connections({"cmd.y": "bat.p"})
    return w


def factory(params, seed):
    """Module-level (importable) factory."""
    w = build(params.get("p", 10.0), params.get("sigma", 0.0), seed)
    if params.get("fail"):
        raise RuntimeError("boom")
    if "cap" in params:
        apply_params(w, {"bat.cap": params["cap"]})
    return w


def plain_world():
    return build(10.0)


def final_soc(res):
    return float(res["bat.soc"].iloc[-1])


# --------------------------------------------------------------------------- samplers

def test_grid_full_factorial():
    pts = Grid({"a": [1, 2], "b": [10, 20, 30]}).sample()
    assert len(pts) == 6 and pts[0] == {"a": 1, "b": 10} and pts[-1] == {"a": 2, "b": 30}
    with pytest.raises(ValueError):
        Grid({"a": [1]}).sample(5)


@pytest.mark.parametrize("make", [
    lambda: Uniform({"x": (0, 1), "y": (5, 6)}, n=20, seed=3),
    lambda: Normal({"x": (0, 1), "y": (5, 2)}, n=20, seed=3),
    lambda: LatinHypercube({"x": (0, 1), "y": (5, 6)}, n=20, seed=3),
    lambda: LatinHypercube({"x": (0, 1), "y": (5, 6)}, n=20, seed=3, method="numpy"),
    lambda: Sobol({"x": (0, 1), "y": (5, 6)}, n=16, seed=3),
])
def test_random_samplers_deterministic(make):
    pytest.importorskip("scipy") if isinstance(make(), Sobol) else None
    a, b = make().sample(), make().sample()
    assert a == b and len(a) == len(make().sample())
    other = make()
    other.seed = 4
    assert other.sample() != a


def test_uniform_normal_statistics():
    u = pd.DataFrame(Uniform({"x": (2, 4)}, seed=1).sample(5000))["x"]
    assert u.min() >= 2 and u.max() <= 4 and abs(u.mean() - 3) < 0.05
    n = pd.DataFrame(Normal({"x": (10, 2)}, seed=1).sample(5000))["x"]
    assert abs(n.mean() - 10) < 0.15 and abs(n.std() - 2) < 0.15


def test_lhs_stratification_numpy_and_default():
    for method in ("numpy", "auto"):
        n = 25
        pts = pd.DataFrame(LatinHypercube({"x": (0, 1), "y": (0, 10)}, n=n, seed=0, method=method).sample())
        assert sorted((pts["x"] * n).astype(int)) == list(range(n))
        assert sorted((pts["y"] / 10 * n).astype(int)) == list(range(n))


def test_scenarios_sampler():
    s = Scenarios([{"a": 1}, {"a": 2}])
    assert s.sample() == [{"a": 1}, {"a": 2}] and len(s) == 2


# --------------------------------------------------------------------------- ensemble basics

def test_apply_params_and_closed_form():
    w = plain_world()
    apply_params(w, {"bat.cap": 50.0})
    soc = final_soc(w.simulate(pbar=False))
    assert soc == pytest.approx(0.2 + 10.0 * (T_STOP / 3600.0) / 50.0, abs=10.0 * 60 / 3600 / 50 + 1e-9)
    # re-applying overrides and merges
    apply_params(w, {"bat.soc0": 0.5})
    assert w.init_dict["bat"][0] == ["cap", "soc0"] or set(w.init_dict["bat"]) >= {"cap", "soc0"}


def test_ensemble_grid_run_and_tables():
    ens = Ensemble(factory, sampler=Grid({"p": [5.0, 10.0], "cap": [50.0, 100.0]}),
                   kpis={"soc_end": final_soc})
    res = ens.run()
    assert isinstance(res, EnsembleResults) and len(res) == 4
    assert res.failed == [] and list(res.status["status"].unique()) == ["completed"]
    assert {"p", "cap", "seed"} <= set(res.params.columns)
    k = res.kpis()["soc_end"]
    p, cap = res.params["p"], res.params["cap"]
    exp = 0.2 + p * 1.0 / cap
    assert np.allclose(k, exp, atol=1e-2)
    assert len(res.runs) == 4 and res.ensemble.is_ensemble
    assert res.run(2).sims


def test_seeds_derived_and_stored():
    a = Ensemble(factory, n_runs=4, seeds=7).run().params["seed"].tolist()
    b = Ensemble(factory, n_runs=4, seeds=7).run().params["seed"].tolist()
    c = Ensemble(factory, n_runs=4, seeds=8).run().params["seed"].tolist()
    assert a == b and a != c and len(set(a)) == 4
    d = Ensemble(factory, n_runs=3, seeds=[1, 2, 3]).run().params["seed"].tolist()
    assert d == [1, 2, 3]
    with pytest.raises(EnsembleError):
        Ensemble(factory, n_runs=3, seeds=[1, 2]).run()
    with pytest.raises(EnsembleError):
        Ensemble(factory).run()


def test_failure_isolation_and_raise():
    sc = [{"p": 1.0}, {"fail": True}, {"p": 3.0}]
    res = Ensemble(factory, scenarios=sc, kpis={"soc_end": final_soc}).run()
    assert res.failed == [1] and res.completed == [0, 2]
    assert res.status.loc[1, "status"] == "failed" and "boom" in res.status.loc[1, "error"]
    assert np.isnan(res.kpis().loc[1, "soc_end"])
    assert len(res.runs) == 2 and len(res.summary().index) == 1
    with pytest.raises(EnsembleError, match="boom"):
        Ensemble(factory, scenarios=sc, on_error="raise").run()


def test_summary_fan_to_frame_against_numpy():
    res = Ensemble(factory, sampler=Uniform({"p": (0.0, 20.0)}, n=12, seed=2), kpis={"soc_end": final_soc}).run()
    vals = res.kpis()["soc_end"].to_numpy()
    s = res.summary(quantiles=(0.1, 0.5, 0.9))
    assert s.loc["soc_end", "count"] == 12
    assert s.loc["soc_end", "mean"] == pytest.approx(vals.mean())
    assert s.loc["soc_end", "std"] == pytest.approx(vals.std(ddof=1))
    for q in (0.1, 0.5, 0.9):
        assert s.loc["soc_end", f"q{q:g}"] == pytest.approx(np.quantile(vals, q))
    wide = res.to_frame("bat.soc")
    assert wide.shape[1] == 12 and wide.index.is_monotonic_increasing
    fan = res.fan("bat.soc", (0.1, 0.5, 0.9))
    assert list(fan.columns) == ["q0.1", "q0.5", "q0.9"]
    assert np.allclose(fan["q0.5"].to_numpy(), np.quantile(wide.to_numpy(), 0.5, axis=1))
    assert (fan["q0.1"] <= fan["q0.9"] + 1e-12).all()


def test_default_kpis_from_registry():
    res = Ensemble(factory, n_runs=2, variables=["bat.soc"], kpi_names=["min", "max"]).run()
    assert set(res.kpis().columns) == {"bat.soc:min", "bat.soc:max"}
    res = Ensemble(factory, n_runs=2).run()
    assert "bat.soc:mean" in res.kpis().columns


def test_keep_kpis_only():
    res = Ensemble(factory, n_runs=3, keep="kpis", kpis={"soc_end": final_soc}).run()
    assert res.kpis()["soc_end"].notna().all()
    with pytest.raises(EnsembleError, match="keep"):
        res.runs
    with pytest.raises(EnsembleError):
        res.fan("bat.soc")


def test_save_load_roundtrip(tmp_path):
    sc = [{"p": 1.0}, {"fail": True}, {"p": 3.0}]
    res = Ensemble(factory, scenarios=sc, kpis={"soc_end": final_soc}, out_dir=tmp_path / "ens").run()
    loaded = EnsembleResults.load(tmp_path / "ens")
    assert loaded.completed == [0, 2] and loaded.failed == [1]
    pd.testing.assert_series_equal(loaded.kpis()["soc_end"], res.kpis()["soc_end"], check_names=False)
    assert loaded.params["seed"].tolist() == res.params["seed"].tolist()
    assert np.allclose(loaded.run(2)["bat.soc"].to_numpy(), res.run(2)["bat.soc"].to_numpy())
    # kpi-only save/load
    r2 = Ensemble(factory, n_runs=2, keep="kpis", kpis={"soc_end": final_soc}).run()
    r2.save(tmp_path / "k")
    l2 = EnsembleResults.load(tmp_path / "k")
    assert l2.keep == "kpis" and len(l2) == 2
    with pytest.raises(FileNotFoundError):
        EnsembleResults.load(tmp_path / "nope")


def test_rerun_reproduces_single_run():
    ens = Ensemble(factory, sampler=Uniform({"p": (5.0, 15.0), "sigma": (0.1, 0.3)}, n=5, seed=1))
    res = ens.run()
    again = ens.rerun(3)
    assert np.array_equal(again["bat.soc"].to_numpy(), res.run(3)["bat.soc"].to_numpy())
    assert np.array_equal(res.rerun(3)["bat.soc"].to_numpy(), res.run(3)["bat.soc"].to_numpy())
    with pytest.raises(KeyError):
        ens.rerun(99)


def test_from_world_with_parameter_map():
    ens = Ensemble.from_world(plain_world, {"capacity": "bat.cap"}, sampler=Grid({"capacity": [50.0, 100.0]}),
                              kpis={"soc_end": final_soc})
    k = ens.run().kpis()["soc_end"]
    assert k.iloc[0] > k.iloc[1]  # smaller pack charges further
    ens2 = Ensemble.from_world(plain_world, sampler=Grid({"bat.soc0": [0.1, 0.4]}), kpis={"soc_end": final_soc})
    k2 = ens2.run().kpis()["soc_end"]
    assert k2.iloc[1] - k2.iloc[0] == pytest.approx(0.3, abs=1e-9)


def test_invalid_config():
    with pytest.raises(ValueError):
        Ensemble(factory, n_runs=1, parallel="threads")
    with pytest.raises(ValueError):
        Ensemble(factory, n_runs=1, keep="some")
    with pytest.raises(ValueError):
        Ensemble(factory, n_runs=1, sampler=Grid({"a": [1]}), scenarios=[{}])
    with pytest.raises(TypeError):
        Ensemble("nope")


# --------------------------------------------------------------------------- process mode

def _mp_available():
    try:
        import multiprocessing as mp
        mp.get_context("spawn")
        return True
    except Exception:
        return False


needs_mp = pytest.mark.skipif(not _mp_available(), reason="multiprocessing unavailable")


def test_closure_rejected_in_process_mode_with_hint():
    def local_factory(params, seed):
        return plain_world()

    with pytest.raises(EnsembleError, match="module level"):
        Ensemble(local_factory, n_runs=2, parallel="process").run()
    with pytest.raises(EnsembleError, match="module level"):
        Ensemble(lambda p, s: plain_world(), n_runs=2, parallel="process").run()
    with pytest.raises(EnsembleError, match="KPI"):
        Ensemble(factory, n_runs=2, parallel="process", kpis={"x": lambda r: 1.0}).run()


@needs_mp
def test_process_equals_serial_bit_for_bit():
    kw = dict(sampler=Uniform({"p": (5.0, 15.0), "sigma": (0.1, 0.3)}, n=4, seed=11), seeds=5,
              kpis={"soc_end": final_soc})
    ser = Ensemble(factory, **kw).run()
    par = Ensemble(factory, parallel="process", max_workers=2, **kw).run()
    assert ser.params.equals(par.params)
    assert ser.kpis().equals(par.kpis())
    for k in ser.completed:
        assert np.array_equal(ser.run(k)["bat.soc"].to_numpy(), par.run(k)["bat.soc"].to_numpy())


@needs_mp
def test_process_failure_isolation():
    res = Ensemble(factory, scenarios=[{"p": 1.0}, {"fail": True}], parallel="process", max_workers=2,
                   keep="kpis", kpis={"soc_end": final_soc}).run()
    assert res.failed == [1] and res.completed == [0]


# --------------------------------------------------------------------------- noise

def test_noisy_input_reproducible_and_order_independent():
    a = NoisyInput(lambda t: 10.0, 0.1, seed=4, ar1=0.8)
    b = NoisyInput(lambda t: 10.0, 0.1, seed=4, ar1=0.8)
    ts = [0, 1, 2, 3, 50, 10, 4]
    va = [a(t) for t in ts]
    vb = [b(t) for t in sorted(ts)]
    assert va == [b(t) for t in ts]
    assert sorted(va) == sorted(vb)
    assert a(7.0) == a(7.0)
    c = NoisyInput(lambda t: 10.0, 0.1, seed=5, ar1=0.8)
    assert [c(t) for t in ts] != va
    a.reset()
    assert [a(t) for t in ts] == va


def test_noisy_input_statistics_ar1_and_modes():
    n = NoisyInput(1.0, 0.2, seed=1, rel=True, ar1=0.9)
    x = np.array([n(t) - 1.0 for t in range(20000)])
    assert x.std() == pytest.approx(0.2, rel=0.1)
    assert np.corrcoef(x[:-1], x[1:])[0, 1] == pytest.approx(0.9, abs=0.03)
    w = NoisyInput(1.0, 0.2, seed=1, ar1=0.0)
    y = np.array([w(t) - 1.0 for t in range(20000)])
    assert abs(np.corrcoef(y[:-1], y[1:])[0, 1]) < 0.03
    add = NoisyInput(lambda t: 100.0, 5.0, seed=2, rel=False, ar1=0.0)
    z = np.array([add(t) for t in range(5000)])
    assert z.mean() == pytest.approx(100.0, abs=0.5) and z.std() == pytest.approx(5.0, rel=0.1)
    ser = pd.Series([0.0, 10.0], index=[0.0, 10.0])
    assert NoisyInput(ser, 0.0)(5.0) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        NoisyInput(1.0, 0.1, ar1=1.0)
    with pytest.raises(TypeError):
        NoisyInput("x", 0.1)


def test_noisy_world_ensemble_spread():
    res = Ensemble(factory, sampler=Uniform({"sigma": (0.3, 0.3)}, n=6, seed=0), seeds=3,
                   kpis={"soc_end": final_soc}).run()
    k = res.kpis()["soc_end"].to_numpy()
    assert k.std() > 0

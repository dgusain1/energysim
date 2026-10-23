# -*- coding: utf-8 -*-
"""Strength 5 - vectorized fleets: N units in one adapter are numerically identical to N
scalar simulations, and fast."""
from __future__ import annotations

import time

import numpy as np
import pytest

from energysim.container import BatteryFleetAdapter, BuildingFleetAdapter, PVFleetAdapter

pytestmark = pytest.mark.strength


def _scalar_battery(soc, cap, p_max_c, p_max_d, p_cmd, eff, soc_min, soc_max, dt_s):
    """Independent single-unit implementation of the documented battery rules."""
    h = dt_s / 3600.0
    p = min(max(p_cmd, -p_max_d), p_max_c)
    if p > 0:
        p = min(p, max(0.0, (soc_max - soc) * cap) / h / eff)
        soc += p * eff * h / cap
    elif p < 0:
        p = max(p, -max(0.0, (soc - soc_min) * cap) / h * eff)
        soc += p / eff * h / cap
    return min(max(soc, soc_min), soc_max), p


def test_battery_fleet_equals_scalar_loop_heterogeneous():
    rng = np.random.default_rng(1)
    n, steps, dt = 40, 96, 900.0
    cap = rng.uniform(20, 80, n)
    pc = rng.uniform(5, 15, n)
    pd_ = rng.uniform(5, 15, n)
    soc0 = rng.uniform(0.1, 0.9, n)
    fleet = BatteryFleetAdapter(size=n, capacity_kwh=cap, max_charge_kw=pc, max_discharge_kw=pd_,
                                initial_soc=soc0, efficiency=0.93, step_size=dt)
    fleet.init()
    socs = soc0.copy()
    for k in range(steps):
        cmd = rng.uniform(-20, 20, n)
        fleet.set_value(["p_cmd_kw"], [cmd])
        fleet.step(k * dt)
        for i in range(n):
            socs[i], p = _scalar_battery(socs[i], cap[i], pc[i], pd_[i], cmd[i], 0.93, 0.05, 0.95, dt)
            assert fleet.states["actual_p_kw"][i] == pytest.approx(p, abs=1e-9)
    np.testing.assert_allclose(fleet.states["soc"], socs, atol=1e-12)


@pytest.mark.parametrize("size", [1, 7, 500])
def test_identical_units_behave_identically_and_match_single_unit(size):
    kw = dict(capacity_kwh=60.0, max_charge_kw=11.0, max_discharge_kw=11.0, initial_soc=0.4, step_size=900.0)
    many, one = BatteryFleetAdapter(size=size, **kw), BatteryFleetAdapter(size=1, **kw)
    for f in (many, one):
        f.init()
    rng = np.random.default_rng(3)
    for k in range(60):
        cmd = float(rng.uniform(-15, 15))
        for f in (many, one):
            f.set_value(["p_cmd_kw"], [cmd])
            f.step(k * 900.0)
    soc_many = many.get_value(["soc"], 0.0)[0]
    assert np.ptp(soc_many) == 0.0
    assert soc_many[0] == one.get_value(["soc"], 0.0)[0][0]
    assert many.get_value(["total_actual_p_kw"], 0.0)[0] == pytest.approx(size * one.get_value(["actual_p_kw"], 0.0)[0][0])


def test_building_fleet_equals_scalar_loop():
    rng = np.random.default_rng(2)
    n, steps, dt = 25, 200, 900.0
    c = rng.uniform(8, 25, n)
    r = rng.uniform(1.5, 4.0, n)
    t0 = rng.uniform(17, 23, n)
    fleet = BuildingFleetAdapter(size=n, c_th_kwh_per_k=c, r_th_k_per_kw=r, initial_t_indoor=t0, step_size=dt)
    fleet.init()
    ref = t0.copy()
    for k in range(steps):
        t_amb, q, qs = rng.uniform(-5, 15), rng.uniform(0, 5, n), rng.uniform(0, 1, n)
        fleet.set_value(["t_ambient"], [t_amb])
        fleet.set_value(["q_heat_kw", "q_solar_kw"], [q, qs])
        fleet.step(k * dt)
        for i in range(n):
            ref[i] += ((q[i] + qs[i] - (ref[i] - t_amb) / r[i]) / c[i]) * dt / 3600.0
    np.testing.assert_allclose(fleet.states["t_indoor"], ref, rtol=1e-13)


def test_pv_fleet_equals_scalar_formula():
    rng = np.random.default_rng(4)
    n = 30
    peak = rng.uniform(2, 10, n)
    fleet = PVFleetAdapter(size=n, peak_kw=peak, gamma_temp_coeff=-0.004, performance_ratio=0.85)
    fleet.init()
    for _ in range(20):
        g = rng.uniform(-50, 1100, n)
        ta = rng.uniform(-5, 35, n)
        fleet.set_value(["irradiance_w_m2", "t_ambient"], [g, ta])
        fleet.step(0.0)
        for i in range(n):
            gi = max(0.0, g[i])
            tc = ta[i] + gi / 800.0 * 25.0
            p = max(0.0, peak[i] * gi / 1000.0 * (1 - 0.004 * (tc - 25.0)) * 0.85)
            assert fleet.states["p_kw"][i] == pytest.approx(p, rel=1e-12, abs=1e-12)


def test_large_fleet_timing_budget_and_speedup():
    n, steps = 50_000, 96
    fleet = BatteryFleetAdapter(size=n, step_size=900.0)
    fleet.init()
    cmd = np.random.default_rng(0).uniform(-11, 11, n)
    fleet.set_value(["p_cmd_kw"], [cmd])
    t0 = time.perf_counter()
    for k in range(steps):
        fleet.step(k * 900.0)
    vec = time.perf_counter() - t0
    assert vec < 5.0, f"50k batteries x 96 steps took {vec:.2f}s"      # generous: typically ~0.1 s

    m = 500
    t0 = time.perf_counter()
    socs = [0.5] * m
    for k in range(steps):
        for i in range(m):
            socs[i], _ = _scalar_battery(socs[i], 50.0, 11.0, 11.0, float(cmd[i]), 0.95, 0.05, 0.95, 900.0)
    scalar_per_unit = (time.perf_counter() - t0) / m
    assert (vec / n) * 3.0 < scalar_per_unit       # at least 3x cheaper per unit than a Python loop (typically ~6x)

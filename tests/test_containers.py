# -*- coding: utf-8 -*-
"""Equivalence, conservation and API tests for the vectorized containers."""

import os
import sys
import time as _time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from energysim.base import SimulatorVariableError  # noqa: E402
from energysim.container import (  # noqa: E402
    BatteryFleetAdapter, BuildingFleetAdapter, PVFleetAdapter, VectorizedFunctionAdapter,
)

N = 17
STEPS = 96
DT = 900.0


# ---------------------------------------------------------------------------
# Reference scalar implementations
# ---------------------------------------------------------------------------

def scalar_battery(cap, pmax_c, pmax_d, soc, eff, smin, smax, p_cmds, dt=DT):
    h = dt / 3600.0
    trace = []
    for p in p_cmds:
        p = min(max(p, -pmax_d), pmax_c)
        if p > 0:
            p = min(p, max(0.0, (smax - soc) * cap) / h / eff)
            soc += p * eff * h / cap
        elif p < 0:
            p = max(p, -max(0.0, (soc - smin) * cap) / h * eff)
            soc += p / eff * h / cap
        soc = min(max(soc, smin), smax)
        trace.append((soc, p))
    return trace


def scalar_building(c, r, t, t_amb_seq, q_heat_seq, q_sol_seq, dt=DT):
    h = dt / 3600.0
    out = []
    for ta, qh, qs in zip(t_amb_seq, q_heat_seq, q_sol_seq):
        t = t + (qh + qs - (t - ta) / r) / c * h
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# Fleet == N independent units
# ---------------------------------------------------------------------------

def test_identical_battery_fleet_equals_scalar_loop():
    rng = np.random.default_rng(1)
    cmds = rng.uniform(-15, 15, STEPS)
    fleet = BatteryFleetAdapter(size=N, capacity_kwh=40.0, max_charge_kw=11.0, max_discharge_kw=9.0,
                                initial_soc=0.5, efficiency=0.92, soc_min=0.1, soc_max=0.9, step_size=DT)
    ref = scalar_battery(40.0, 11.0, 9.0, 0.5, 0.92, 0.1, 0.9, cmds)
    for k, p in enumerate(cmds):
        fleet.set_value(['p_cmd'], p)
        fleet.step(k * DT)
        soc = fleet.get_value(['soc'])[0]
        p_act = fleet.get_value(['actual_p_kw'])[0]
        assert np.allclose(soc, ref[k][0], atol=1e-9, rtol=0)
        assert np.allclose(p_act, ref[k][1], atol=1e-9, rtol=0)
        assert np.ptp(soc) < 1e-12  # all identical


def test_heterogeneous_battery_fleet_equals_per_unit_fleets():
    rng = np.random.default_rng(2)
    cap = rng.uniform(20, 80, N)
    pc = rng.uniform(5, 15, N)
    soc0 = rng.uniform(0.15, 0.85, N)
    cmds = rng.uniform(-15, 15, (STEPS, N))
    fleet = BatteryFleetAdapter(size=N, capacity_kwh=cap, max_charge_kw=pc, max_discharge_kw=pc,
                                initial_soc=soc0, step_size=DT)
    singles = [BatteryFleetAdapter(size=1, capacity_kwh=cap[i], max_charge_kw=pc[i], max_discharge_kw=pc[i],
                                   initial_soc=soc0[i], step_size=DT) for i in range(N)]
    for k in range(STEPS):
        fleet.set_value(['p_cmd'], cmds[k])
        fleet.step(k * DT)
        for i, s in enumerate(singles):
            s.set_value(['p_cmd'], cmds[k, i])
            s.step(k * DT)
        got = fleet.get_value(['soc'])[0]
        want = np.array([s.get_value(['soc'])[0][0] for s in singles])
        assert np.allclose(got, want, atol=1e-9, rtol=0)


def test_battery_energy_conservation():
    rng = np.random.default_rng(3)
    cap = rng.uniform(20, 80, N)
    fleet = BatteryFleetAdapter(size=N, capacity_kwh=cap, initial_soc=0.5, efficiency=0.9, step_size=DT)
    h = DT / 3600.0
    e_in = np.zeros(N)
    losses = np.zeros(N)
    stored0 = fleet.states['soc'] * cap
    for k in range(STEPS):
        fleet.set_value(['p_cmd'], rng.uniform(-14, 14, N))
        fleet.step(k * DT)
        p = fleet.get_value(['actual_p_kw'])[0]
        e_in += p * h
        losses += fleet.get_value(['loss_kw'])[0] * h
    stored = fleet.states['soc'] * cap - stored0
    assert np.all(losses >= -1e-12)
    assert np.allclose(e_in - stored, losses, atol=1e-9, rtol=0)
    # losses are the efficiency penalty: for pure charge, loss == (1-eff)*E_in
    f = BatteryFleetAdapter(size=1, capacity_kwh=1000.0, initial_soc=0.1, efficiency=0.9, max_charge_kw=10, step_size=3600)
    f.set_value(['p_cmd'], 10.0)
    f.step(0)
    assert f.get_value(['loss_kw'])[0][0] == pytest.approx(1.0)
    assert f.get_value(['soc'])[0][0] == pytest.approx(0.1 + 9.0 / 1000.0)


def test_battery_bounds_and_zero_step():
    f = BatteryFleetAdapter(size=3, initial_soc=[0.05, 0.5, 0.95], step_size=DT)
    f.set_value(['p_cmd'], [[-11.0, 11.0, 11.0]])
    f.step(0)
    soc = f.get_value(['soc'])[0]
    assert np.all((soc >= f.soc_min - 1e-12) & (soc <= f.soc_max + 1e-12))
    assert f.get_value(['actual_p_kw'])[0][0] == 0.0 and f.get_value(['actual_p_kw'])[0][2] == 0.0
    before = f.get_value(['soc'])[0]
    f.step(0, step_size=0.0)
    assert np.array_equal(f.get_value(['soc'])[0], before)


def test_building_fleet_equals_scalar_1r1c():
    rng = np.random.default_rng(4)
    ta = rng.uniform(-5, 15, STEPS)
    qh = rng.uniform(0, 6, STEPS)
    qs = rng.uniform(0, 2, STEPS)
    fleet = BuildingFleetAdapter(size=N, c_th_kwh_per_k=12.0, r_th_k_per_kw=3.0, initial_t_indoor=19.0, step_size=DT)
    ref = scalar_building(12.0, 3.0, 19.0, ta, qh, qs)
    for k in range(STEPS):
        fleet.set_value(['t_amb', 'q_heat', 'q_solar'], [ta[k], qh[k], qs[k]])
        fleet.step(k * DT)
        assert np.allclose(fleet.get_value(['t_indoor'])[0], ref[k], atol=1e-9, rtol=0)


def test_pv_fleet_matches_formula():
    rng = np.random.default_rng(5)
    peak = rng.uniform(2, 10, N)
    g = rng.uniform(-50, 1100, N)
    ta = rng.uniform(-5, 35, N)
    pv = PVFleetAdapter(size=N, peak_kw=peak, gamma_temp_coeff=-0.0038, performance_ratio=0.82)
    pv.set_value(['irradiance', 't_amb'], [g, ta])
    pv.step(0)
    gp = np.maximum(g, 0.0)
    expected = np.maximum(0.0, peak * gp / 1000.0 * (1 - 0.0038 * (ta + gp / 800.0 * 25.0 - 25.0)) * 0.82)
    assert np.allclose(pv.get_value(['p_pv'])[0], expected, atol=1e-9, rtol=0)
    assert pv.get_value(['total_p_kw'])[0] == pytest.approx(expected.sum(), abs=1e-9)


# ---------------------------------------------------------------------------
# Addressing, aliases, errors
# ---------------------------------------------------------------------------

def _fleet():
    return BatteryFleetAdapter(size=5, initial_soc=[0.1, 0.2, 0.3, 0.4, 0.5])


def test_index_and_aggregation_syntax():
    f = _fleet()
    soc_vec, s2, mean, mn, mx, tot, summ = f.get_value(
        ['soc', 'soc[2]', 'mean_soc', 'min_soc', 'max_soc', 'total_soc', 'sum_soc'])
    assert np.allclose(soc_vec, [0.1, 0.2, 0.3, 0.4, 0.5])
    assert s2 == pytest.approx(0.3)
    assert (mean, mn, mx) == (pytest.approx(0.3), pytest.approx(0.1), pytest.approx(0.5))
    assert tot == pytest.approx(1.5) and summ == pytest.approx(1.5)
    assert f.get_value([' soc[ 4 ] '])[0] == pytest.approx(0.5)
    # returned vector is a copy
    soc_vec[0] = 99.0
    assert f.get_value(['soc[0]'])[0] == pytest.approx(0.1)


def test_aliases_resolve_to_canonical_variables():
    f = _fleet()
    f.set_value(['p_req', 'p_cmd[1]'], [3.0, 7.0])
    got = f.get_value(['p_cmd_kw', 'p_cmd', 'p_req'])
    assert got[0][0] == 3.0 and got[0][1] == 7.0
    assert np.array_equal(got[0], got[1]) and np.array_equal(got[1], got[2])
    b = BuildingFleetAdapter(size=2)
    b.set_value(['t_amb_degc'], 3.0)
    assert b.get_value(['t_ambient'])[0].tolist() == [3.0, 3.0]


def test_exact_name_beats_aggregation_prefix():
    f = VectorizedFunctionAdapter(name='x', size=3, update_func=lambda *a: {},
                                  initial_states={'min_temp': [1.0, 2.0, 3.0], 'temp': 9.0})
    assert f.get_value(['min_temp'])[0].tolist() == [1.0, 2.0, 3.0]
    assert f.get_value(['min_min_temp'])[0] == 1.0
    assert f.get_value(['min_temp[1]'])[0] == 2.0


@pytest.mark.parametrize("name", ['soc[5]', 'soc[-1]', 'soc[100]'])
def test_bad_index_raises(name):
    f = _fleet()
    with pytest.raises(SimulatorVariableError):
        f.get_value([name])
    with pytest.raises(SimulatorVariableError):
        f.set_value([name.replace('soc', 'p_cmd')], 1.0)


def test_unknown_variable_and_length_errors():
    f = _fleet()
    with pytest.raises(SimulatorVariableError):
        f.get_value(['nope'])
    with pytest.raises(SimulatorVariableError):
        f.get_value(['mean_nope'])
    with pytest.raises(SimulatorVariableError):
        f.set_value(['nope'], 1.0)
    with pytest.raises(SimulatorVariableError):
        f.set_value(['p_cmd[abc]'], 1.0)
    with pytest.raises(ValueError):
        f.set_value(['p_cmd'], [[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        f.set_value(['p_cmd', 'soc'], [1.0])
    with pytest.raises(ValueError):
        BatteryFleetAdapter(size=5, capacity_kwh=[1.0, 2.0])
    with pytest.raises(ValueError):
        BatteryFleetAdapter(size=5, initial_soc=[0.5] * 4)
    with pytest.raises(ValueError):
        BatteryFleetAdapter(size=0)


def test_get_available_variables_lists_aggregations():
    avail = BatteryFleetAdapter(size=2).get_available_variables()
    for v in ('soc', 'actual_p_kw', 'loss_kw', 'p_cmd_kw', 'p_cmd'):
        assert v in avail['inputs']
        for pre in ('', 'total_', 'mean_', 'min_', 'max_'):
            assert f'{pre}{v}' in avail['outputs']
    assert len(avail['outputs']) == 5 * len(avail['inputs'])


# ---------------------------------------------------------------------------
# init / reset
# ---------------------------------------------------------------------------

def test_init_snapshot_and_reset_restore_initial_states():
    f = BatteryFleetAdapter(size=4, initial_soc=0.5, step_size=DT)
    f.set_value(['soc'], [[0.2, 0.3, 0.4, 0.5]])   # user-specified initial condition before init
    f.init()
    f.set_value(['p_cmd'], 11.0)
    for k in range(10):
        f.step(k * DT)
    assert not np.allclose(f.get_value(['soc'])[0], [0.2, 0.3, 0.4, 0.5])
    f.init()   # re-run
    assert np.allclose(f.get_value(['soc'])[0], [0.2, 0.3, 0.4, 0.5])
    assert np.all(f.get_value(['p_cmd'])[0] == 0.0)
    assert np.all(f.get_value(['actual_p_kw'])[0] == 0.0)
    f.set_value(['soc'], 0.9)
    f.reset()
    assert np.allclose(f.get_value(['soc'])[0], [0.2, 0.3, 0.4, 0.5])


def test_reset_without_init_uses_registered_values():
    b = BuildingFleetAdapter(size=3, initial_t_indoor=21.0, step_size=DT)
    b.set_value(['q_heat'], 5.0)
    for k in range(5):
        b.step(k * DT)
    assert b.get_value(['mean_t_indoor'])[0] != 21.0
    b.reset()
    assert b.get_value(['t_indoor'])[0].tolist() == [21.0] * 3
    assert b.get_value(['q_heat'])[0].tolist() == [0.0] * 3


def test_no_rollback_api_on_containers():
    for cls in (BatteryFleetAdapter, BuildingFleetAdapter, PVFleetAdapter):
        assert 'save_state' not in cls.__dict__ and 'restore_state' not in cls.__dict__


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

def test_large_fleet_timing():
    n = 100_000
    fleet = BatteryFleetAdapter(size=n, initial_soc=0.5, step_size=DT)
    rng = np.random.default_rng(0)
    cmd = rng.uniform(-11, 11, n)
    t0 = _time.perf_counter()
    for k in range(96):
        fleet.set_value(['p_cmd'], cmd)
        fleet.step(k * DT)
    fleet.get_value(['total_actual_p_kw', 'mean_soc'])
    elapsed = _time.perf_counter() - t0
    assert elapsed < 10.0, f"100k units x 96 steps took {elapsed:.2f}s"
    assert np.all(np.isfinite(fleet.states['soc']))

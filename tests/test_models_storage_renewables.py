# -*- coding: utf-8 -*-
"""Property and physics tests for storage and renewable generation models."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from energysim.models.storage import BatteryBESS, FlowBattery, Flywheel  # noqa: E402
from energysim.models.renewables import PhysicsPV, WindTurbine, SolarThermalCollector  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]


def _get(m, name):
    return m.get_value([name], 0.0)[0]


# =============================================================================
# BatteryBESS
# =============================================================================

def test_bess_initial_outputs_reflect_initial_state():
    b = BatteryBESS(capacity_mwh=2.0, p_max_mw=1.0, soc_init=0.37, t_init_c=31.0)
    b.init()
    assert _get(b, "SoC") == pytest.approx(0.37)
    assert _get(b, "T_batt") == pytest.approx(31.0)
    assert _get(b, "SOH") == pytest.approx(1.0)
    assert _get(b, "P_actual") == 0.0
    assert _get(b, "I_current") == 0.0
    assert _get(b, "V_terminal") == pytest.approx(float(b.ocv_curve(0.37)))
    assert b.soc == pytest.approx(0.37)


def _energy_residual(b, profile, dt):
    """Return (sum |energy in - heat - stored| , total throughput) over a profile."""
    resid = 0.0
    thru = 0.0
    for k, p in enumerate(profile):
        b.set_value(["P_cmd"], [p])
        soc0, soh0 = b.soc, b.soh
        b.step_model(dt, k * dt)
        p_dc = _get(b, "V_terminal") * _get(b, "I_current") / 1e6      # MW, DC terminal
        heat_mwh = _get(b, "Q_heat_kW") / 1000.0 * dt / 3600.0
        e_dc = p_dc * dt / 3600.0
        stored = (b.soc - soc0) * b.capacity_mwh * soh0
        resid += abs(e_dc - heat_mwh - stored)
        thru += abs(e_dc)
    return resid, thru


def test_bess_energy_conservation_charge_heat_stored():
    b = BatteryBESS(capacity_mwh=1.0, p_max_mw=0.5, soc_init=0.5)
    b.init()
    profile = [0.4] * 100 + [-0.5] * 100 + [0.2] * 50 + [-0.1] * 50
    resid, thru = _energy_residual(b, profile, dt=10.0)
    assert thru > 0
    assert resid < 1e-9 * max(thru, 1.0)


def test_bess_energy_conservation_at_soc_limits():
    # Large steps force the SoC-window scaling branch; balance must still close.
    b = BatteryBESS(capacity_mwh=1.0, p_max_mw=0.5, soc_init=0.9)
    b.init()
    profile = [0.5] * 6 + [-0.5] * 12 + [0.5] * 12
    resid, thru = _energy_residual(b, profile, dt=900.0)
    assert resid < 1e-9 * max(thru, 1.0)
    assert b.soc <= b.soc_max + 1e-12 and b.soc >= b.soc_min - 1e-12


def test_bess_joule_heat_is_positive_and_roundtrip_is_lossy():
    b = BatteryBESS(capacity_mwh=1.0, p_max_mw=0.5, soc_init=0.5)
    b.init()
    e_ac = 0.0
    for k in range(200):
        b.set_value(["P_cmd"], [0.4])
        b.step_model(5.0, k)
        e_ac += _get(b, "P_actual") * 5.0 / 3600.0
        assert _get(b, "Q_heat_kW") >= 0.0
    soc_after_charge = b.soc
    for k in range(2000):
        b.set_value(["P_cmd"], [-0.4])
        b.step_model(5.0, k)
        e_ac += _get(b, "P_actual") * 5.0 / 3600.0
        if b.soc <= 0.5:
            break
    assert soc_after_charge > 0.5
    # Grid net energy to return to the same SoC is positive (losses)
    assert e_ac > 0.0


def _cycle(b, target_fec):
    """Cycle ``b`` between SoC 5 % and 95 % until ``target_fec`` full equivalent cycles."""
    thru = 0.0
    charging = True
    k = 0
    last_soh = 1.0
    while thru < 2.0 * target_fec * b.capacity_mwh and k < 100000:
        if b.soc >= 0.95:
            charging = False
        elif b.soc <= 0.05:
            charging = True
        b.set_value(["P_cmd"], [b.p_max_mw if charging else -b.p_max_mw])
        b.step_model(1800.0, k * 1800.0)
        thru += abs(_get(b, "V_terminal") * _get(b, "I_current")) / 1e6 * 0.5
        soh = _get(b, "SOH")
        assert soh <= last_soh + 1e-15
        last_soh = soh
        k += 1
    return thru / (2.0 * b.capacity_mwh)


def test_bess_cycle_life_reaches_eol_fade_at_cycle_life():
    # Scaled-down cycle life so the test is fast; the formula is linear in FEC.
    b = BatteryBESS(capacity_mwh=1.0, p_max_mw=1.0, soc_init=0.5, soc_min=0.0, soc_max=1.0,
                    calendar_fade_pct_per_year_25c=0.0, cycle_life_to_eol=100.0, eol_fade=0.20)
    b.init()
    fec = _cycle(b, 100.0)
    assert fec >= 100.0
    assert _get(b, "SOH") == pytest.approx(1.0 - 0.20 * fec / 100.0, abs=1e-9)
    assert _get(b, "SOH") == pytest.approx(0.80, abs=3e-3)


def test_bess_default_cycle_life_is_4000_fec_to_80_percent():
    b = BatteryBESS(capacity_mwh=1.0, p_max_mw=1.0, soc_min=0.0, soc_max=1.0, calendar_fade_pct_per_year_25c=0.0)
    assert b.cycle_life_to_eol == 4000.0 and b.eol_fade == 0.20
    b.init()
    fec = _cycle(b, 20.0)
    # SOH at 4000 FEC would be 0.80: check the per-cycle slope 0.2/4000
    assert 1.0 - _get(b, "SOH") == pytest.approx(0.20 * fec / 4000.0, rel=1e-6)


def test_bess_calendar_fade_default_and_arrhenius():
    def idle_year(temp):
        b = BatteryBESS(t_init_c=temp, calendar_fade_pct_per_year_25c=2.0)
        b.init()
        b.set_value(["T_amb"], [temp])
        for k in range(365):
            b.step_model(86400.0, k * 86400.0)
        return _get(b, "SOH")

    soh25 = idle_year(25.0)
    assert 1.0 - soh25 == pytest.approx(0.02 * 365.0 / 365.25, rel=1e-3)
    assert 1.0 - idle_year(40.0) > 1.0 - soh25 > 1.0 - idle_year(10.0) > 0.0


def test_bess_soh_floor_and_params_exposed():
    b = BatteryBESS(cycle_life_to_eol=1.0, eol_fade=0.9, soc_min=0.0, soc_max=1.0)
    b.init()
    for k in range(200):
        b.set_value(["P_cmd"], [0.5 if k % 2 == 0 else -0.5])
        b.step_model(900.0, k)
    assert _get(b, "SOH") == pytest.approx(0.5)
    d = b.describe()["parameters"]
    assert d["calendar_fade_pct_per_year_25c"] == 2.0
    assert d["cycle_life_to_eol"] == 4000.0 and d["eol_fade"] == 0.20
    assert "Thevenin" in BatteryBESS.__doc__ and "Rint" in BatteryBESS.__doc__


def test_bess_idle_temperature_relaxes_to_ambient():
    b = BatteryBESS(t_init_c=40.0, c_th_kj_k=100.0, r_th_k_w=0.01)
    b.init()
    b.set_value(["T_amb"], [20.0])
    prev = 40.0
    for k in range(200):
        b.step_model(10.0, k)
        t = _get(b, "T_batt")
        assert 20.0 <= t <= prev + 1e-12
        prev = t
    assert prev < 25.0


@pytest.mark.parametrize("seed", SEEDS)
def test_bess_random_profile_invariants(seed):
    rng = np.random.default_rng(seed)
    b = BatteryBESS(capacity_mwh=float(rng.uniform(0.5, 5)), p_max_mw=float(rng.uniform(0.2, 2)),
                    soc_init=float(rng.uniform(0.1, 0.9)))
    b.init()
    last_soh = 1.0
    for k in range(300):
        b.set_value(["P_cmd", "T_amb"], [float(rng.uniform(-3, 3)), float(rng.uniform(-10, 40))])
        b.step_model(float(rng.uniform(1, 60)), k)
        vals = b.get_value(b.outputs, k)
        assert all(math.isfinite(v) for v in vals)
        assert b.soc_min - 1e-9 <= b.soc <= b.soc_max + 1e-9
        assert abs(_get(b, "P_actual")) <= b.p_max_mw + 1e-12
        assert _get(b, "Q_heat_kW") >= 0.0
        assert 0.5 <= _get(b, "SOH") <= last_soh + 1e-15
        last_soh = _get(b, "SOH")


# =============================================================================
# FlowBattery / Flywheel
# =============================================================================

def test_flow_battery_initial_and_bounds():
    f = FlowBattery(stack_p_max_mw=1.0, tank_capacity_mwh=2.0, soc_init=0.3)
    f.init()
    assert _get(f, "SoC") == pytest.approx(0.3)
    assert _get(f, "P_actual") == 0.0 and _get(f, "P_parasitic_kW") == 0.0


def test_flow_battery_energy_balance_and_idle_has_no_parasitics():
    f = FlowBattery(stack_p_max_mw=1.0, tank_capacity_mwh=4.0, soc_init=0.5, stack_efficiency=0.81)
    f.init()
    # charge: stored = sqrt(eta) * stack energy
    f.set_value(["P_cmd"], [0.5])
    f.step_model(3600.0, 0.0)
    assert _get(f, "SoC") == pytest.approx(0.5 + 0.9 * 0.5 / 4.0)
    assert _get(f, "P_actual") > 0.5
    # idle -> neither SoC change nor parasitics
    f.set_value(["P_cmd"], [0.0])
    soc = _get(f, "SoC")
    f.step_model(60.0, 1.0)
    assert _get(f, "SoC") == soc and _get(f, "P_actual") == 0.0


@pytest.mark.parametrize("seed", SEEDS)
def test_flow_battery_and_flywheel_random_invariants(seed):
    rng = np.random.default_rng(seed)
    f = FlowBattery(soc_init=float(rng.uniform(0, 1)))
    w = Flywheel(soc_init=float(rng.uniform(0, 1)))
    f.init()
    w.init()
    for k in range(300):
        p = float(rng.uniform(-2, 2))
        dt = float(rng.uniform(0.5, 600))
        f.set_value(["P_cmd"], [p])
        w.set_value(["P_cmd"], [p])
        f.step_model(dt, k)
        w.step_model(dt, k)
        for m in (f, w):
            assert 0.0 <= m.soc <= 1.0
            assert all(math.isfinite(v) for v in m.get_value(m.outputs, k))
        assert abs(_get(w, "P_actual")) <= w.p_max + 1e-12
        # parasitics are non-negative and P_actual consistent with them
        assert _get(f, "P_parasitic_kW") >= 0.0


def test_flywheel_energy_balance_without_self_discharge():
    w = Flywheel(power_max_mw=0.5, energy_capacity_kwh=100.0, soc_init=0.5, roundtrip_eff=0.81,
                 self_discharge_pct_hr=0.0)
    w.init()
    w.set_value(["P_cmd"], [0.1])
    w.step_model(360.0, 0.0)  # 0.1 MW * 0.1 h = 10 kWh in
    assert _get(w, "SoC") == pytest.approx(0.5 + 0.9 * 10.0 / 100.0)
    w.set_value(["P_cmd"], [-0.1])
    w.step_model(360.0, 1.0)  # 10 kWh out costs 10/0.9 kWh
    assert _get(w, "SoC") == pytest.approx(0.59 - 10.0 / 0.9 / 100.0)


def test_flywheel_self_discharge_and_limits():
    w = Flywheel(soc_init=0.5, self_discharge_pct_hr=10.0)
    w.init()
    soc = _get(w, "SoC")
    w.step_model(3600.0, 0.0)
    assert _get(w, "SoC") < soc
    # full flywheel cannot charge, P_actual reports what was really exchanged
    w2 = Flywheel(soc_init=0.99, self_discharge_pct_hr=0.0)
    w2.init()
    w2.set_value(["P_cmd"], [0.5])
    w2.step_model(10.0, 0.0)
    assert _get(w2, "P_actual") == pytest.approx(0.0, abs=1e-12)


# =============================================================================
# PhysicsPV
# =============================================================================

def _pv_p(pv, g, t=25.0, v=1.0, curtail=None):
    names = ["G_poa", "T_amb", "v_wind"]
    vals = [g, t, v]
    if curtail is not None:
        names.append("curtail_limit")
        vals.append(curtail)
    pv.set_value(names, vals)
    pv.step_model(1.0, 0.0)
    return _get(pv, "P_ac")


def test_pv_single_curtail_attribute_and_initial_outputs():
    pv = PhysicsPV()
    assert hasattr(pv, "curtail_limit_mw") and not hasattr(pv, "curtail_limit")
    pv.init()
    assert _get(pv, "P_ac") == 0.0 and _get(pv, "P_dc") == 0.0
    assert _get(pv, "curtail_limit") == pv.curtail_limit_mw


def test_pv_monotone_in_irradiance_and_clipped():
    pv = PhysicsPV(p_dc_stc_mw=2.0, inverter_p_ac_max_mw=1.5)
    pv.init()
    prev = -1.0
    for g in np.linspace(0, 1300, 66):
        p = _pv_p(pv, float(g))
        assert 0.0 <= p <= 1.5 + 1e-12
        assert p >= prev - 1e-9
        prev = p
    assert _pv_p(pv, 1300.0) == pytest.approx(1.5)


def test_pv_hotter_cell_produces_less_and_curtailment_respected():
    pv = PhysicsPV(p_dc_stc_mw=1.0, inverter_p_ac_max_mw=2.0)
    pv.init()
    assert _pv_p(pv, 800.0, t=0.0) > _pv_p(pv, 800.0, t=40.0)
    assert _pv_p(pv, 800.0, t=20.0, v=10.0) > _pv_p(pv, 800.0, t=20.0, v=0.5)  # wind cools the cell
    for lim in (0.0, 0.1, 0.5):
        assert _pv_p(pv, 900.0, curtail=lim) <= lim + 1e-12


@pytest.mark.parametrize("seed", SEEDS)
def test_pv_random_invariants(seed):
    rng = np.random.default_rng(seed)
    pv = PhysicsPV(p_dc_stc_mw=float(rng.uniform(0.5, 5)), inverter_p_ac_max_mw=float(rng.uniform(0.5, 4)))
    pv.init()
    for _ in range(100):
        pv.set_value(["G_poa", "T_amb", "v_wind"],
                     [float(rng.uniform(-50, 1400)), float(rng.uniform(-20, 50)), float(rng.uniform(0, 20))])
        pv.step_model(1.0, 0.0)
        vals = pv.get_value(pv.outputs, 0.0)
        assert all(math.isfinite(v) for v in vals)
        assert 0.0 <= _get(pv, "P_ac") <= pv.inverter_p_ac_max_mw + 1e-12
        assert _get(pv, "P_dc") >= _get(pv, "P_ac") - 1e-12


# =============================================================================
# WindTurbine
# =============================================================================

def _wind_ss(wt, v, t=15.0, n=300):
    wt.set_value(["v_wind", "T_amb"], [v, t])
    for k in range(n):
        wt.step_model(1.0, float(k))
    return _get(wt, "P_actual")


def test_wind_density_correction_does_not_derate_rated_plateau():
    hot = WindTurbine(p_rated_mw=3.0)
    cold = WindTurbine(p_rated_mw=3.0)
    hot.init()
    cold.init()
    assert _wind_ss(hot, 15.0, 35.0) == pytest.approx(3.0, rel=1e-3)
    assert _wind_ss(cold, 15.0, 15.0) == pytest.approx(3.0, rel=1e-3)


def test_wind_density_still_derates_partial_load_and_altitude():
    def p(v, t, alt):
        wt = WindTurbine(p_rated_mw=3.0, altitude_m=alt)
        wt.init()
        return _wind_ss(wt, v, t)

    assert p(8.0, 35.0, 0.0) < p(8.0, 15.0, 0.0) < p(8.0, -10.0, 0.0)
    assert p(8.0, 15.0, 2500.0) < p(8.0, 15.0, 0.0)
    assert p(15.0, 15.0, 2000.0) == pytest.approx(3.0, rel=1e-3)  # still rated far above rated speed


def test_wind_initial_cutin_cutout_and_monotone():
    wt = WindTurbine(p_rated_mw=3.0)
    wt.init()
    assert _get(wt, "P_actual") == 0.0
    assert _wind_ss(wt, 1.0) == pytest.approx(0.0, abs=1e-9)
    assert _wind_ss(wt, 26.0) == pytest.approx(0.0, abs=1e-9)
    prev = -1.0
    for v in np.linspace(0.0, 24.5, 50):
        wt2 = WindTurbine(p_rated_mw=3.0)
        wt2.init()
        p = _wind_ss(wt2, float(v))
        assert 0.0 <= p <= 3.0 + 1e-9
        assert p >= prev - 1e-6
        prev = p


@pytest.mark.parametrize("seed", SEEDS)
def test_wind_random_invariants(seed):
    rng = np.random.default_rng(seed)
    wt = WindTurbine(p_rated_mw=float(rng.uniform(1, 8)), altitude_m=float(rng.uniform(0, 3000)))
    wt.init()
    for k in range(300):
        wt.set_value(["v_wind", "T_amb"], [float(rng.uniform(0, 35)), float(rng.uniform(-30, 45))])
        wt.step_model(float(rng.uniform(0.1, 5)), k)
        assert 0.0 <= _get(wt, "P_actual") <= wt.p_rated_mw + 1e-9


# =============================================================================
# SolarThermalCollector
# =============================================================================

def test_solar_thermal_properties():
    s = SolarThermalCollector(area_m2=1000.0, eta0=0.8)
    s.init()
    assert _get(s, "Q_thermal_MW") == 0.0
    prev = None
    for t_in in (20.0, 40.0, 60.0, 80.0, 100.0):
        s.set_value(["G_solar", "T_in", "T_amb"], [800.0, t_in, 20.0])
        s.step_model(1.0, 0.0)
        q = _get(s, "Q_thermal_MW")
        assert 0.0 <= q <= 1000.0 * 800.0 * 0.8 / 1e6 + 1e-12
        if prev is not None:
            assert q <= prev + 1e-12
        prev = q
    s.set_value(["G_solar"], [5.0])
    s.step_model(1.0, 0.0)
    assert _get(s, "Q_thermal_MW") == 0.0

# -*- coding: utf-8 -*-
"""Property and physics tests for thermal, HVAC, load and carbon models."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from energysim.models.thermal import GasTurbine_OCGT, CombinedCycle_CCGT, ReciprocatingGenset, CHP_SteamTurbine  # noqa: E402
from energysim.models.hvac_thermal import (  # noqa: E402
    IndustrialHeatPump, ElectricBoiler, ElectricChiller, AbsorptionChiller, ThermalStorageTank,
)
from energysim.models.loads import DataCenterLoad, IndustrialProcessSteam  # noqa: E402
from energysim.models.carbon import CarbonCapture_Amine  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]


def _get(m, name):
    return m.get_value([name], 0.0)[0]


def _run(m, n, dt=1.0, t0=0.0):
    for k in range(n):
        m.step_model(dt, t0 + k * dt)


# =============================================================================
# OCGT
# =============================================================================

def _ocgt(**kw):
    kw.setdefault("p_rated_mw", 100.0)
    kw.setdefault("min_stable_gen_pct", 30.0)
    kw.setdefault("cold_start_duration_sec", 60.0)
    kw.setdefault("ramp_rate_mw_min", 60.0)
    m = GasTurbine_OCGT(**kw)
    m.init()
    return m


def test_ocgt_initial_outputs():
    m = _ocgt()
    assert _get(m, "state") == "OFF"
    assert _get(m, "P_actual") == 0.0 and _get(m, "fuel_consumption_MMBtu_hr") == 0.0


def test_ocgt_start_completes_even_if_command_drops():
    m = _ocgt()
    m.set_value(["P_cmd"], [80.0])
    m.step_model(1.0, 0.0)
    assert _get(m, "state") == "STARTING"
    m.set_value(["P_cmd"], [0.0])  # command dips during start
    seen_running = False
    for k in range(120):
        m.step_model(1.0, k)
        if _get(m, "state") == "RUNNING":
            seen_running = True
            break
    assert seen_running
    assert _get(m, "P_actual") == pytest.approx(m.msg_mw)  # synchronises at MSG, no jump to setpoint


def test_ocgt_ramps_from_msg_after_start():
    m = _ocgt(ramp_rate_mw_min=60.0)  # 1 MW/s
    m.set_value(["P_cmd"], [90.0])
    for k in range(200):
        m.step_model(1.0, k)
        if _get(m, "state") == "RUNNING":
            break
    assert _get(m, "P_actual") == pytest.approx(30.0)
    m.step_model(1.0, 0.0)
    assert _get(m, "P_actual") == pytest.approx(31.0)
    p_prev = _get(m, "P_actual")
    for k in range(100):
        m.step_model(1.0, k)
        p = _get(m, "P_actual")
        assert p - p_prev <= 1.0 + 1e-9
        p_prev = p
    assert p_prev == pytest.approx(90.0)


def test_ocgt_min_uptime_enforced():
    m = _ocgt(min_uptime_sec=300.0, min_downtime_sec=0.0, hot_start_duration_sec=10.0, cold_start_duration_sec=10.0)
    m.set_value(["P_cmd"], [80.0])
    for k in range(60):
        m.step_model(1.0, k)
    assert _get(m, "state") == "RUNNING"
    t_run0 = m.sm.time_in_state_sec
    m.set_value(["P_cmd"], [0.0])
    # Cannot stop before min uptime: held at MSG
    while m.sm.time_in_state_sec < 290.0:
        m.step_model(1.0, 0.0)
        assert _get(m, "state") == "RUNNING"
        assert _get(m, "P_actual") >= m.msg_mw - 1e-9
    # After min uptime it ramps down and shuts off
    for k in range(200):
        m.step_model(1.0, k)
    assert _get(m, "state") == "OFF"
    assert _get(m, "P_actual") == 0.0 and _get(m, "fuel_consumption_MMBtu_hr") == 0.0
    assert t_run0 > 0


def test_ocgt_min_downtime_enforced():
    m = _ocgt(min_uptime_sec=0.0, min_downtime_sec=600.0, hot_start_duration_sec=10.0, cold_start_duration_sec=10.0)
    m.set_value(["P_cmd"], [80.0])
    for k in range(60):
        m.step_model(1.0, k)
    m.set_value(["P_cmd"], [0.0])
    for k in range(200):
        m.step_model(1.0, k)
    assert _get(m, "state") == "OFF"
    m.set_value(["P_cmd"], [80.0])
    for k in range(300):  # < 600 s offline in total
        m.step_model(1.0, k)
    assert _get(m, "state") == "OFF"
    for k in range(400):
        m.step_model(1.0, k)
    assert _get(m, "state") in ("STARTING", "RUNNING")


def test_ocgt_time_in_state_resets_on_change():
    m = _ocgt(cold_start_duration_sec=30.0)
    m.set_value(["P_cmd"], [80.0])
    m.step_model(1.0, 0.0)
    assert m.sm.state.value == "STARTING"
    assert m.sm.time_in_state_sec <= 1.0 + 1e-9


def test_ocgt_ambient_derate_and_monotone_fuel():
    def steady(t, cmd=100.0):
        m = _ocgt(cold_start_duration_sec=5.0)
        m.set_value(["P_cmd", "T_amb"], [cmd, t])
        _run(m, 400)
        return m

    cold, ref, hot = steady(-10.0), steady(15.0), steady(40.0)
    assert _get(hot, "P_actual") < _get(ref, "P_actual") <= 100.0 + 1e-9
    assert _get(hot, "P_actual") == pytest.approx(100.0 * float(hot.amb_curve(40.0)), rel=1e-6)
    fuels = []
    for cmd in (40.0, 60.0, 80.0, 100.0):
        fuels.append(_get(steady(15.0, cmd), "fuel_consumption_MMBtu_hr"))
    assert all(a < b for a, b in zip(fuels, fuels[1:]))


@pytest.mark.parametrize("seed", SEEDS)
def test_ocgt_random_invariants(seed):
    rng = np.random.default_rng(seed)
    m = _ocgt(min_uptime_sec=60.0, min_downtime_sec=60.0, cold_start_duration_sec=20.0, hot_start_duration_sec=10.0)
    last_p = 0.0
    for k in range(600):
        if k % 40 == 0:
            m.set_value(["P_cmd", "T_amb"], [float(rng.uniform(0, 110)), float(rng.uniform(-15, 42))])
        m.step_model(1.0, k)
        p = _get(m, "P_actual")
        assert 0.0 <= p <= 100.0 * 1.12 + 1e-9
        assert abs(p - last_p) <= 60.0 / 60.0 + 1e-9 or last_p == 0.0 or p == 0.0 or _get(m, "state") == "RUNNING"
        assert _get(m, "fuel_consumption_MMBtu_hr") >= 0.0 and _get(m, "Q_exhaust_MW") >= 0.0
        assert _get(m, "state") in ("OFF", "STARTING", "RUNNING")
        last_p = p


# =============================================================================
# CCGT
# =============================================================================

def test_ccgt_uses_ambient_input():
    def steady(t):
        m = CombinedCycle_CCGT(p_rated_mw=400.0, ramp_rate_mw_min=600.0)
        m.init()
        m.set_value(["P_cmd", "T_amb"], [400.0, t])
        _run(m, 120)
        return _get(m, "P_actual")

    assert steady(45.0) < steady(25.0) < steady(15.0) + 1e-9
    assert steady(15.0) == pytest.approx(400.0)
    assert steady(45.0) == pytest.approx(400.0 * 0.90, rel=1e-6)


def test_ccgt_balance_and_split():
    m = CombinedCycle_CCGT(p_rated_mw=400.0, gt_ratio=0.65, ramp_rate_mw_min=600.0)
    m.init()
    assert _get(m, "state") == "OFF" and _get(m, "P_actual") == 0.0
    m.set_value(["P_cmd"], [300.0])
    _run(m, 60)
    p, gt, st = _get(m, "P_actual"), _get(m, "P_gt"), _get(m, "P_st")
    assert gt + st == pytest.approx(p)
    fuel_mw = _get(m, "fuel_consumption_MMBtu_hr") * 1.055e6 / 3600.0 / 1000.0
    assert fuel_mw > p  # energy in exceeds electric out
    assert _get(m, "Q_condenser_MW") >= 0.0
    assert _get(m, "Q_condenser_MW") + p <= fuel_mw + 1e-9
    assert _get(m, "CO2_emissions_t_hr") == pytest.approx(_get(m, "fuel_consumption_MMBtu_hr") * 0.0592)


def test_ccgt_below_msg_stays_off_and_ramp_limit():
    m = CombinedCycle_CCGT(p_rated_mw=400.0, min_stable_gen_pct=40.0, ramp_rate_mw_min=60.0)
    m.init()
    m.set_value(["P_cmd"], [100.0])  # below MSG (160 MW)
    _run(m, 30)
    assert _get(m, "P_actual") == 0.0 and _get(m, "state") == "OFF"
    m.set_value(["P_cmd"], [300.0])
    last = 0.0
    for k in range(30):
        m.step_model(1.0, k)
        p = _get(m, "P_actual")
        assert p - last <= 1.0 + 1e-9
        last = p


def test_ccgt_min_uptime_and_downtime_optional():
    m = CombinedCycle_CCGT(p_rated_mw=100.0, ramp_rate_mw_min=6000.0, min_uptime_sec=100.0, min_downtime_sec=100.0)
    m.init()
    m.set_value(["P_cmd"], [80.0])
    _run(m, 10)
    assert _get(m, "state") == "RUNNING"
    m.set_value(["P_cmd"], [0.0])
    _run(m, 50)
    assert _get(m, "state") == "RUNNING" and _get(m, "P_actual") >= m.msg_mw - 1e-9
    _run(m, 80)
    assert _get(m, "state") == "OFF"
    m.set_value(["P_cmd"], [80.0])
    _run(m, 50)
    assert _get(m, "state") == "OFF"
    _run(m, 100)
    assert _get(m, "state") == "RUNNING"


@pytest.mark.parametrize("seed", SEEDS)
def test_ccgt_random_invariants(seed):
    rng = np.random.default_rng(seed)
    m = CombinedCycle_CCGT(p_rated_mw=200.0, ramp_rate_mw_min=30.0)
    m.init()
    for k in range(300):
        m.set_value(["P_cmd", "T_amb"], [float(rng.uniform(0, 250)), float(rng.uniform(-20, 45))])
        m.step_model(1.0, k)
        assert all(math.isfinite(v) for v in m.get_value(m.outputs[:-1], k))
        assert 0.0 <= _get(m, "P_actual") <= 200.0 * 1.06 + 1e-9
        assert _get(m, "Q_condenser_MW") >= 0.0 and _get(m, "CO2_emissions_t_hr") >= 0.0


def test_ccgt_docstring_is_honest():
    doc = CombinedCycle_CCGT.__doc__
    assert "no HRSG" in doc


# =============================================================================
# Genset / CHP
# =============================================================================

def test_genset_energy_balance_and_limits():
    g = ReciprocatingGenset(p_rated_mw=10.0, electrical_efficiency=0.44, heat_recovery_efficiency=0.42)
    g.init()
    assert _get(g, "P_actual") == 0.0
    for cmd in (-5.0, 0.0, 3.0, 10.0, 25.0):
        g.set_value(["P_cmd"], [cmd])
        g.step_model(1.0, 0.0)
        p = _get(g, "P_actual")
        assert 0.0 <= p <= 10.0
        fuel_mw = _get(g, "fuel_consumption_MMBtu_hr") * 1055.06 / 3600.0
        assert p + _get(g, "Q_thermal_MW") <= fuel_mw + 1e-9


def test_chp_initial_and_extraction_penalty():
    c = CHP_SteamTurbine(p_max_mw=50.0, power_loss_coefficient_beta=0.18)
    c.init()
    assert _get(c, "P_actual") == pytest.approx(50.0)  # default steam flow input 50 kg/s
    c.set_value(["steam_flow_kg_s", "heat_demand_MW"], [40.0, 0.0])
    c.step_model(1.0, 0.0)
    p0 = _get(c, "P_actual")
    prev = p0
    for q in (5.0, 20.0, 60.0, 200.0):
        c.set_value(["heat_demand_MW"], [q])
        c.step_model(1.0, 0.0)
        assert _get(c, "P_actual") <= prev + 1e-12
        assert _get(c, "Q_heat_delivered_MW") <= min(q, 80.0) + 1e-12
        assert _get(c, "P_actual") >= 0.0
        prev = _get(c, "P_actual")
    assert "correlation" in CHP_SteamTurbine.__doc__


# =============================================================================
# HVAC
# =============================================================================

def test_heat_pump_initial_cop_and_energy_balance():
    hp = IndustrialHeatPump(q_max_thermal_mw=5.0, carnot_efficiency=0.5, t_source_nom_c=15.0, t_sink_nom_c=75.0)
    hp.init()
    expected = 0.5 * (75.0 + 273.15) / (75.0 - 15.0)
    assert _get(hp, "COP") == pytest.approx(expected)
    assert _get(hp, "P_elec_MW") == 0.0
    hp.set_value(["Q_heat_cmd_MW"], [4.0])
    hp.step_model(1.0, 0.0)
    q, p, qs = _get(hp, "Q_heat_MW"), _get(hp, "P_elec_MW"), _get(hp, "Q_absorbed_source_MW")
    assert q == pytest.approx(p + qs)
    assert p == pytest.approx(q / _get(hp, "COP"))


def test_heat_pump_cop_monotone_in_lift_and_bounded():
    hp = IndustrialHeatPump()
    hp.init()
    prev = None
    for t_sink in range(35, 131, 5):
        hp.set_value(["T_source", "T_sink"], [20.0, float(t_sink)])
        hp.step_model(1.0, 0.0)
        cop = _get(hp, "COP")
        assert 1.2 <= cop <= 7.0
        if prev is not None:
            assert cop <= prev + 1e-12
        prev = cop


def test_boiler_and_chiller_properties():
    b = ElectricBoiler(p_max_mw=10.0, efficiency=0.98, ramp_rate_mw_min=60.0)
    b.init()
    b.set_value(["P_cmd"], [25.0])
    last = 0.0
    for k in range(30):
        b.step_model(1.0, k)
        p = _get(b, "P_actual")
        assert 0.0 <= p <= 10.0 and p - last <= 1.0 + 1e-9
        assert _get(b, "Q_thermal_MW") == pytest.approx(0.98 * p)
        last = p

    c = ElectricChiller(q_cooling_max_mw=5.0, cop_nominal=5.5)
    c.init()
    assert _get(c, "COP") == pytest.approx(5.5)
    prev = None
    for t in range(0, 46, 5):
        c.set_value(["Q_cool_cmd_MW", "T_amb"], [3.0, float(t)])
        c.step_model(1.0, 0.0)
        assert _get(c, "Q_cool_actual_MW") * 1.0 == pytest.approx(3.0)
        assert _get(c, "P_elec_MW") == pytest.approx(3.0 / _get(c, "COP"))
        if prev is not None:
            assert _get(c, "COP") <= prev + 1e-12
        prev = _get(c, "COP")


def test_absorption_chiller_heat_balance():
    a = AbsorptionChiller(q_cooling_max_mw=5.0, cop_thermal=0.8)
    a.init()
    a.set_value(["Q_cool_cmd_MW", "Q_waste_heat_available_MW"], [4.0, 2.0])
    a.step_model(1.0, 0.0)
    assert _get(a, "Q_heat_consumed_MW") <= 2.0 + 1e-12
    assert _get(a, "Q_cool_actual_MW") == pytest.approx(0.8 * _get(a, "Q_heat_consumed_MW"))
    assert _get(a, "Q_cool_actual_MW") == pytest.approx(1.6)
    a.set_value(["Q_waste_heat_available_MW"], [100.0])
    a.step_model(1.0, 0.0)
    assert _get(a, "Q_cool_actual_MW") == pytest.approx(4.0)
    a.set_value(["Q_cool_cmd_MW"], [50.0])
    a.step_model(1.0, 0.0)
    assert _get(a, "Q_cool_actual_MW") <= 5.0 + 1e-12


@pytest.mark.parametrize("seed", SEEDS)
def test_thermal_storage_energy_balance_random(seed):
    rng = np.random.default_rng(seed)
    t = ThermalStorageTank(capacity_mwh_th=20.0, max_flow_rate_mw_th=8.0, loss_pct_per_day=5.0,
                           soc_init=float(rng.uniform(0.0, 1.0)))
    t.init()
    e0 = _get(t, "SoC") * t.capacity_mwh
    e_net = 0.0
    for k in range(400):
        dt = float(rng.uniform(10, 1800))
        t.set_value(["Q_cmd_MW"], [float(rng.uniform(-12, 12))])
        t.step_model(dt, k)
        soc = _get(t, "SoC")
        assert 0.0 <= soc <= 1.0
        q, loss = _get(t, "Q_actual_MW"), _get(t, "Q_loss_MW")
        assert abs(q) <= t.max_flow + 1e-12 and loss >= 0.0
        e_net += (q - loss) * dt / 3600.0
    assert _get(t, "SoC") * t.capacity_mwh == pytest.approx(e0 + e_net, abs=1e-9)


def test_thermal_storage_full_and_empty_limits_and_loss():
    t = ThermalStorageTank(capacity_mwh_th=10.0, max_flow_rate_mw_th=5.0, loss_pct_per_day=0.0, soc_init=0.5)
    t.init()
    assert _get(t, "SoC") == 0.5 and _get(t, "Q_actual_MW") == 0.0
    t.set_value(["Q_cmd_MW"], [5.0])
    for k in range(10):
        t.step_model(3600.0, k)
    assert _get(t, "SoC") == pytest.approx(1.0)
    assert _get(t, "Q_actual_MW") == pytest.approx(0.0, abs=1e-9)  # already full: no flow reported
    t.set_value(["Q_cmd_MW"], [-5.0])
    for k in range(10):
        t.step_model(3600.0, k)
    assert _get(t, "SoC") == pytest.approx(0.0, abs=1e-12)
    # standing loss decays an idle tank
    t2 = ThermalStorageTank(capacity_mwh_th=10.0, loss_pct_per_day=10.0, soc_init=0.8)
    t2.init()
    for k in range(24):
        t2.step_model(3600.0, k)
    assert 0.8 * 0.85 < _get(t2, "SoC") < 0.8
    assert "energy-bucket" in ThermalStorageTank.__doc__


# =============================================================================
# Loads and carbon
# =============================================================================

def test_data_center_pue_and_balance():
    d = DataCenterLoad(it_capacity_mw=100.0, base_pue=1.15)
    d.init()
    assert _get(d, "P_it_MW") == pytest.approx(100.0)  # defaults to full IT capacity
    prev = None
    for t in range(-10, 41, 5):
        d.set_value(["P_it_cmd_MW", "T_amb"], [60.0, float(t)])
        d.step_model(1.0, 0.0)
        pue = _get(d, "PUE")
        assert pue >= 1.0
        assert _get(d, "P_total_grid_MW") == pytest.approx(60.0 * pue)
        assert _get(d, "P_cooling_MW") == pytest.approx(_get(d, "P_total_grid_MW") - 60.0)
        assert _get(d, "Q_waste_heat_MW") <= _get(d, "P_total_grid_MW")
        if prev is not None:
            assert pue >= prev - 1e-12
        prev = pue
    d.set_value(["P_it_cmd_MW"], [1000.0])
    d.step_model(1.0, 0.0)
    assert _get(d, "P_it_MW") == pytest.approx(100.0)


def test_steam_header_balance():
    s = IndustrialProcessSteam(nominal_demand_t_hr=100.0, enthalpy_kj_kg=2800.0)
    s.init()
    assert _get(s, "steam_demand_t_hr") == pytest.approx(100.0)
    assert _get(s, "Q_demand_MW_th") == pytest.approx(100.0 * 1000.0 * 2800.0 / 3.6e6)
    for supply in (0.0, 60.0, 100.0, 150.0):
        s.set_value(["steam_supply_t_hr"], [supply])
        s.step_model(1.0, 0.0)
        assert _get(s, "steam_deficit_t_hr") == pytest.approx(max(0.0, 100.0 - supply))


def test_carbon_capture_mass_and_energy_balance():
    c = CarbonCapture_Amine(capture_rate_nom=0.9, spec_steam_reboiler_gj_tco2=3.0, spec_elec_kwh_tco2=120.0)
    c.init()
    assert _get(c, "m_dot_CO2_captured_t_hr") == 0.0
    prev = 0.0
    for co2 in (10.0, 50.0, 100.0):
        c.set_value(["m_dot_CO2_in_t_hr"], [co2])
        c.step_model(1.0, 0.0)
        cap = _get(c, "m_dot_CO2_captured_t_hr")
        assert cap == pytest.approx(0.9 * co2) and cap <= co2
        assert _get(c, "Q_steam_reboiler_MW_th") == pytest.approx(cap * 3.0 * 1000.0 / 3600.0)
        assert _get(c, "P_elec_parasitic_MW") == pytest.approx(cap * 120.0 / 1000.0)
        assert _get(c, "P_elec_parasitic_MW") > prev
        prev = _get(c, "P_elec_parasitic_MW")

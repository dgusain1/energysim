# -*- coding: utf-8 -*-
"""Analytic / textbook benchmarks of the energysim physical models.

Every test records its reference and tolerance in
``validation/data/validation_cases.csv`` (see ``refs.py``).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from energysim import world
from energysim.container import BuildingFleetAdapter, BatteryFleetAdapter
from energysim.models import (
    AbsorptionChiller, BatteryBESS, CarbonCapture_Amine, CombinedCycle_CCGT, Curve1D,
    H2Compressor, H2StorageTank, IndustrialHeatPump, MethanationReactor, PEMElectrolyzer,
    ThermalStorageTank, WindTurbine,
)
from energysim.models.hydrogen import (
    FARADAY_CONSTANT, MOLAR_MASS_CO2, MOLAR_MASS_H2, MOLAR_MASS_H2O, R_GAS,
)
from helpers import run_and_get
from refs import assert_ref, ref

pytestmark = pytest.mark.validation


# ---------------------------------------------------------------------------
# RC thermal envelope
# ---------------------------------------------------------------------------

def _rc_closed_form(t_s, t0, t_amb, q_kw, r, c):
    tau_h = r * c
    t_inf = t_amb + r * q_kw
    return t_inf + (t0 - t_inf) * math.exp(-(t_s / 3600.0) / tau_h)


def _scalar_rc_euler(dt_s, n, t0, t_amb, q_kw, r, c):
    """Independent scalar explicit-Euler reference implementation."""
    t = t0
    for _ in range(n):
        t += ((q_kw - (t - t_amb) / r) / c) * (dt_s / 3600.0)
    return t


def test_rc_step_response_fleet_vs_closed_form():
    r, c, t0, t_amb, q = 2.5, 15.0, 20.0, 5.0, 2.0
    horizon = 24 * 3600.0
    errs = {}
    for dt in (900.0, 450.0):
        fleet = BuildingFleetAdapter(size=3, c_th_kwh_per_k=c, r_th_k_per_kw=r, initial_t_indoor=t0, step_size=dt)
        fleet.init()
        fleet.set_value(["t_ambient", "q_heat_kw"], [t_amb, q])
        n = int(round(horizon / dt))
        for k in range(n):
            fleet.step(k * dt)
        t_num = fleet.get_value(["t_indoor"], horizon)[0]
        t_ref = _rc_closed_form(horizon, t0, t_amb, q, r, c)
        errs[dt] = abs(t_num - t_ref).max()
        assert np.allclose(t_num, t_num[0])           # identical units stay identical
    assert_ref(errs[900.0], "rc_thermal", "max_abs_error_K_dt900", expected=0.0)
    # explicit Euler is first order in dt
    order = math.log2(errs[900.0] / errs[450.0])
    assert_ref(order, "rc_thermal", "euler_convergence_order", expected=1.0)


def test_rc_scalar_equivalent_matches_fleet_and_closed_form():
    r, c, t0, t_amb, q, dt, n = 2.5, 15.0, 20.0, 5.0, 2.0, 900.0, 96
    fleet = BuildingFleetAdapter(size=1, c_th_kwh_per_k=c, r_th_k_per_kw=r, initial_t_indoor=t0, step_size=dt)
    fleet.init()
    fleet.set_value(["t_ambient", "q_heat_kw"], [t_amb, q])
    for k in range(n):
        fleet.step(k * dt)
    t_fleet = fleet.get_value(["t_indoor"], 0.0)[0][0]
    t_scalar = _scalar_rc_euler(dt, n, t0, t_amb, q, r, c)
    assert_ref(t_fleet, "rc_thermal", "fleet_vs_scalar_loop", expected=t_scalar)
    t_exact = _rc_closed_form(n * dt, t0, t_amb, q, r, c)
    assert abs(t_scalar - t_exact) < 0.1


def test_rc_step_response_through_world(tmp_path):
    """Same step response when the fleet is driven through the co-simulation orchestrator."""
    r, c, t0, t_amb, q, dt = 2.5, 15.0, 20.0, 5.0, 2.0, 900.0
    w = world(start_time=0, stop_time=6 * 3600, t_macro=dt, res_filename=str(tmp_path / "rc.h5"))
    w.add_signal("amb", signal=t_amb, step_size=dt, units={"y": "degC"})
    w.add_signal("heat", signal=q, step_size=dt, units={"y": "kW"})
    w.add_container("fleet", kind="building", size=1, c_th_kwh_per_k=c, r_th_k_per_kw=r,
                    initial_t_indoor=t0, step_size=dt, inputs=["t_ambient", "q_heat_kw"],
                    outputs=["mean_t_indoor"])
    w.add_connections({"amb.y": "fleet.t_ambient", "heat.y": "fleet.q_heat_kw"})
    res = run_and_get(w, record_all=False)["fleet"]
    t_num = res["mean_t_indoor"].to_numpy()
    times = np.arange(len(t_num)) * dt          # row k is the state at k * t_macro (time column unit is auto-scaled)
    t_ref = np.array([_rc_closed_form(tt, t0, t_amb, q, r, c) for tt in times])
    assert np.max(np.abs(t_num - t_ref)) < 0.05
    assert_ref(np.max(np.abs(t_num - t_ref)), "rc_thermal", "world_run_max_abs_error_K", expected=0.0)


# ---------------------------------------------------------------------------
# PEM electrolyzer: Faraday's law, stoichiometry, thermoneutral voltage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("frac", [0.1, 0.25, 0.5, 0.75, 1.0])
def test_pem_faraday_law_exact(frac):
    m = PEMElectrolyzer(p_nominal_mw=5.0)
    m.init()
    m.set_value(["P_cmd"], [5.0 * frac])
    m.step(0.0)
    i, eta = m.get_value(["I_stack_A", "eta_faraday"], 0.0)
    h2_kg_h = m.get_value(["m_dot_H2_kg_hr"], 0.0)[0]
    # n_dot_H2 = eta_F * N_cells * I / (2F)  [mol/s]
    n_dot = eta * m.n_cells * i / (2.0 * FARADAY_CONSTANT)
    assert_ref(h2_kg_h, "pem_electrolyzer", "faraday_h2_mass_flow", expected=n_dot * 2.01588e-3 * 3600.0)
    # power balance: electric power = N_cells * I * V_cell
    p_mw = m.get_value(["P_actual"], 0.0)[0]
    v = m.get_value(["V_cell"], 0.0)[0]
    assert_ref(m.n_cells * i * v / 1e6, "pem_electrolyzer", "stack_power_balance", expected=p_mw)


@pytest.mark.parametrize("frac", [0.1, 0.5, 1.0])
def test_pem_water_stoichiometry(frac):
    m = PEMElectrolyzer(p_nominal_mw=5.0)
    m.init()
    m.set_value(["P_cmd"], [5.0 * frac])
    m.step(0.0)
    h2, h2o = m.get_value(["m_dot_H2_kg_hr", "m_dot_H2O_kg_hr"], 0.0)
    # H2O -> H2 + 1/2 O2 : one mole of water per mole of hydrogen
    assert_ref(h2o / h2, "pem_electrolyzer", "water_to_h2_mass_ratio", expected=18.01528e-3 / 2.01588e-3)
    assert 8.9 < h2o / h2 < 9.0


def test_pem_thermoneutral_voltage_consistent_with_hhv():
    from energysim.models.hydrogen import HHV_H2_MJ_KG, V_THERMONEUTRAL_HHV
    v_tn = HHV_H2_MJ_KG * 1e6 * MOLAR_MASS_H2 / (2.0 * FARADAY_CONSTANT)
    assert_ref(V_THERMONEUTRAL_HHV, "pem_electrolyzer", "thermoneutral_voltage_V", expected=v_tn)
    # electricity = waste heat + hydrogen HHV energy (gas crossover ignored in the heat term)
    m = PEMElectrolyzer(p_nominal_mw=5.0)
    m.init()
    m.set_value(["P_cmd"], [5.0])
    m.step(0.0)
    p, q, h2, eta = m.get_value(["P_actual", "Q_waste_MW", "m_dot_H2_kg_hr", "eta_faraday"], 0.0)
    h2_ideal_kg_s = h2 / 3600.0 / eta        # before crossover loss
    assert_ref(q + h2_ideal_kg_s * HHV_H2_MJ_KG, "pem_electrolyzer", "heat_plus_hhv_equals_power", expected=p)
    assert 0.6 < h2_ideal_kg_s * HHV_H2_MJ_KG / p < 0.85   # sane HHV efficiency range


def test_faraday_constant_codata():
    assert_ref(FARADAY_CONSTANT, "constants", "faraday_constant_C_per_mol")
    assert_ref(R_GAS, "constants", "gas_constant_J_per_mol_K")


# ---------------------------------------------------------------------------
# Sabatier methanation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("h2_feed", [20.0, 100.0, 200.0])
def test_sabatier_stoichiometry(h2_feed):
    eff = 0.95
    m = MethanationReactor(max_h2_feed_kg_hr=200.0, conversion_efficiency=eff)
    m.init()
    m.set_value(["m_dot_H2_kg_hr"], [h2_feed])
    m.step(0.0)
    ch4, co2, h2o, unreacted, fed = m.get_value(
        ["m_dot_CH4_kg_hr", "m_dot_CO2_consumed_kg_hr", "m_dot_H2O_kg_hr",
         "m_dot_H2_unreacted_kg_hr", "m_dot_H2_fed_kg_hr"], 0.0)
    n_h2 = fed * eff / MOLAR_MASS_H2
    m_ch4 = MOLAR_MASS_CO2 + 4 * MOLAR_MASS_H2 - 2 * MOLAR_MASS_H2O
    assert_ref(co2 / MOLAR_MASS_CO2, "sabatier", "mol_co2_per_mol_h2_reacted", expected=n_h2 / 4.0)
    assert_ref(ch4 / m_ch4, "sabatier", "mol_ch4_per_mol_h2_reacted", expected=n_h2 / 4.0)
    assert_ref(h2o / MOLAR_MASS_H2O, "sabatier", "mol_h2o_per_mol_h2_reacted", expected=n_h2 / 2.0)
    assert_ref(fed + co2, "sabatier", "mass_balance", expected=ch4 + h2o + unreacted)


def test_sabatier_heat_of_reaction_vs_formation_enthalpies():
    # dH = dHf(CH4) + 2 dHf(H2O,g) - dHf(CO2)  [kJ/mol], standard formation enthalpies at 298 K
    d_h = -74.87 + 2.0 * (-241.83) - (-393.51)
    m = MethanationReactor(max_h2_feed_kg_hr=200.0, conversion_efficiency=1.0)
    m.init()
    m.set_value(["m_dot_H2_kg_hr"], [100.0])
    m.step(0.0)
    q_mw = m.get_value(["Q_reaction_MW"], 0.0)[0]
    n_ch4_per_s = (100.0 / MOLAR_MASS_H2 / 3600.0) / 4.0
    assert_ref(q_mw * 1e3 / n_ch4_per_s, "sabatier", "reaction_heat_kJ_per_mol_CH4", expected=-d_h)


# ---------------------------------------------------------------------------
# Heat pump: Carnot bound and monotonic COP
# ---------------------------------------------------------------------------

def _hp_cop(eta, t_src, t_snk):
    m = IndustrialHeatPump(carnot_efficiency=eta)
    m.init()
    m.set_value(["Q_heat_cmd_MW", "T_source", "T_sink"], [5.0, t_src, t_snk])
    m.step(0.0)
    return m.get_value(["COP"], 0.0)[0], m


def test_heatpump_cop_respects_carnot_bound():
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(300):
        t_src = rng.uniform(-10, 60)
        lift = rng.uniform(10, 90)
        t_snk = min(t_src + lift, 120.0)
        for eta in (0.3, 0.5, 0.65):
            cop, m = _hp_cop(eta, t_src, t_snk)
            carnot = (t_snk + 273.15) / (t_snk - t_src)
            worst = max(worst, cop / carnot)
            assert cop <= carnot + 1e-12
            p, q = m.get_value(["P_elec_MW", "Q_heat_MW"], 0.0)
            assert_ref(p * cop, "heat_pump", "energy_balance_q_equals_p_times_cop", expected=q)
    assert_ref(worst, "heat_pump", "max_cop_over_carnot", expected=0.65)


def test_heatpump_cop_equals_eta_times_carnot_inside_clamps():
    cop, _ = _hp_cop(0.5, 15.0, 75.0)
    assert_ref(cop, "heat_pump", "cop_eta_times_carnot", expected=0.5 * (75 + 273.15) / 60.0)


def test_heatpump_cop_monotonic_in_lift():
    lifts = np.arange(10.0, 91.0, 5.0)
    cops = [_hp_cop(0.5, 10.0, 10.0 + lift)[0] for lift in lifts]
    # non-increasing everywhere, strictly decreasing wherever the COP is not clamped to its limits
    assert all(b <= a for a, b in zip(cops, cops[1:])), cops
    free = [(a, b) for a, b in zip(cops, cops[1:]) if 1.2 < b and a < 7.0]
    assert len(free) >= 8 and all(b < a for a, b in free), cops


# ---------------------------------------------------------------------------
# Wind turbine
# ---------------------------------------------------------------------------

# Manufacturer-style tabulated power curve (wind speed m/s -> kW) of a generic 2 MW class turbine.
_PUBLISHED_V = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 25, 25.01, 35]
_PUBLISHED_KW = [0, 0, 66, 154, 282, 460, 696, 996, 1341, 1641, 1812, 1850, 1850, 0, 0]


def _steady_wind(m, v, t_amb=15.0):
    m.init()
    m.set_value(["v_wind", "T_amb"], [v, t_amb])
    for k in range(5):
        m.step(float(k))
    return m.get_value(["P_actual"], 0.0)[0]


@pytest.mark.parametrize("v", [3.5, 4.0, 5.5, 7.0, 8.5, 10.0, 12.0, 20.0])
def test_wind_power_curve_points(v):
    curve = Curve1D(_PUBLISHED_V, [p / 1000.0 for p in _PUBLISHED_KW], min_val=0.0, max_val=1.85)
    m = WindTurbine(p_rated_mw=1.85, cut_in_speed_ms=3.0, rated_speed_ms=13.0, cut_out_speed_ms=25.0,
                    rotor_time_constant_sec=1e-9, power_curve=curve)
    p = _steady_wind(m, v)
    expected = float(np.interp(v, _PUBLISHED_V, [x / 1000.0 for x in _PUBLISHED_KW]))
    # density at 15 C / sea level equals the 1.225 kg/m3 reference to ~1e-5
    assert_ref(p, "wind_turbine", "power_curve_interpolation_MW", expected=expected)


def test_wind_cut_in_cut_out_and_rated_plateau():
    curve = Curve1D(_PUBLISHED_V, [p / 1000.0 for p in _PUBLISHED_KW], min_val=0.0, max_val=1.85)
    m = WindTurbine(p_rated_mw=1.85, cut_in_speed_ms=3.0, rated_speed_ms=13.0, cut_out_speed_ms=25.0,
                    rotor_time_constant_sec=1e-9, power_curve=curve)
    assert _steady_wind(m, 2.0) == 0.0
    assert _steady_wind(m, 26.0) == 0.0
    assert _steady_wind(m, 18.0) == pytest.approx(1.85)


def test_wind_density_correction_power_proportional_to_rho():
    v_grid = np.linspace(0.0, 30.0, 3001)
    cubic = Curve1D(v_grid, 1e-3 * v_grid ** 3, min_val=0.0, max_val=1e9)
    def power(alt, t_amb):
        m = WindTurbine(p_rated_mw=1e9, cut_out_speed_ms=1e9, rotor_time_constant_sec=1e-9,
                        power_curve=cubic, altitude_m=alt)
        return _steady_wind(m, 7.0, t_amb)
    rho = lambda alt, t: 101325.0 * (1 - 2.25577e-5 * alt) ** 5.25588 / (287.05 * (t + 273.15))
    p0 = power(0.0, 15.0)
    for alt, t in ((1500.0, 5.0), (2500.0, -5.0), (0.0, 35.0)):
        assert_ref(power(alt, t) / p0, "wind_turbine", "power_ratio_vs_air_density", expected=rho(alt, t) / rho(0.0, 15.0))


def test_wind_rotor_lag_first_order():
    tau, dt = 4.0, 0.01
    curve = Curve1D([0, 3, 12, 40], [0, 0, 3.0, 3.0], min_val=0.0, max_val=3.0)
    m = WindTurbine(p_rated_mw=3.0, rotor_time_constant_sec=tau, power_curve=curve, cut_out_speed_ms=60.0)
    m.init()
    m.set_value(["v_wind", "T_amb"], [12.0, 15.0])
    n = int(tau / dt)
    for k in range(n):
        m.step_model(dt, k * dt)
    p = m.get_value(["P_actual"], 0.0)[0]
    assert_ref(p, "wind_turbine", "rotor_lag_at_one_tau_MW", expected=3.0 * (1 - math.exp(-1.0)))


# ---------------------------------------------------------------------------
# Batteries
# ---------------------------------------------------------------------------

def test_battery_fleet_round_trip_efficiency():
    eff = 0.92
    f = BatteryFleetAdapter(size=1, capacity_kwh=100.0, max_charge_kw=50.0, max_discharge_kw=50.0,
                            initial_soc=0.3, soc_min=0.0, soc_max=1.0, efficiency=eff, step_size=60.0)
    f.init()
    n, p_chg = 60, 40.0
    e_in = e_out = 0.0
    f.set_value(["p_cmd_kw"], [p_chg])
    for k in range(n):
        f.step(k * 60.0)
        e_in += f.get_value(["actual_p_kw"], 0)[0][0] * 60.0 / 3600.0
    # discharge power that removes exactly the stored energy again in n steps
    p_dis = p_chg * eff * eff
    f.set_value(["p_cmd_kw"], [-p_dis])
    for k in range(n):
        f.step(k * 60.0)
        e_out -= f.get_value(["actual_p_kw"], 0)[0][0] * 60.0 / 3600.0
    assert f.get_value(["soc"], 0)[0][0] == pytest.approx(0.3, abs=1e-12)
    assert_ref(e_out / e_in, "battery", "fleet_round_trip_efficiency", expected=eff * eff)


def _bess_step_energies(m, p_ac_mw, dt):
    """Run one step; return (E_ac, dE_stored, E_joule, E_inverter_loss) in MWh."""
    soc0, soh0 = m.get_value(["SoC", "SOH"], 0.0)
    m.set_value(["P_cmd"], [p_ac_mw])
    m.step_model(dt, 0.0)
    soc1, p_ac, q_kw = m.get_value(["SoC", "P_actual", "Q_heat_kW"], 0.0)
    eta = float(m.inverter_curve(abs(p_ac_mw) / m.p_max_mw))
    p_dc = p_ac * eta if p_ac > 0 else p_ac / eta
    h = dt / 3600.0
    return p_ac * h, (soc1 - soc0) * m.capacity_mwh * soh0, q_kw / 1000.0 * h, (p_ac - p_dc) * h


def test_bess_energy_conservation_including_heat():
    rng = np.random.default_rng(5)
    m = BatteryBESS(capacity_mwh=2.0, p_max_mw=1.0, soc_init=0.5)
    m.init()
    total = {"ac": 0.0, "stored": 0.0, "joule": 0.0, "inv": 0.0}
    for _ in range(400):
        # small steps keep SoC away from the window limits
        p = rng.uniform(-1.0, 1.0)
        e_ac, d_st, e_j, e_inv = _bess_step_energies(m, p, 10.0)
        assert e_j >= 0.0 and e_inv >= -1e-15      # losses are never negative
        assert_ref(e_ac, "battery", "bess_step_energy_balance", expected=d_st + e_j + e_inv)
        total["ac"] += e_ac
        total["stored"] += d_st
        total["joule"] += e_j
        total["inv"] += e_inv
    assert_ref(total["ac"], "battery", "bess_cumulative_energy_balance",
               expected=total["stored"] + total["joule"] + total["inv"])
    assert total["joule"] > 0.0


def test_bess_round_trip_efficiency_vs_inverter_and_internal_resistance():
    p_mw, dt = 0.25, 5.0
    m = BatteryBESS(capacity_mwh=2.0, p_max_mw=1.0, soc_init=0.5)
    m.init()
    soc_start = 0.5
    e_in = e_out = 0.0
    for _ in range(120):
        e_ac, *_ = _bess_step_energies(m, +p_mw, dt)
        e_in += e_ac
    for _ in range(2000):
        if m.get_value(["SoC"], 0.0)[0] <= soc_start:
            break
        e_ac, *_ = _bess_step_energies(m, -p_mw, dt)
        e_out -= e_ac
    rt = e_out / e_in
    # expected: product of inverter efficiency (both ways) and the internal-resistance efficiency
    eta_inv = float(m.inverter_curve(p_mw / m.p_max_mw))
    voc = float(m.ocv_curve(0.5))
    rin = float(m.rin_curve(0.5))
    def eta_int(p_dc_w):   # fraction of DC power stored in the cells when charging
        i = (-voc + math.sqrt(voc ** 2 + 4 * rin * p_dc_w)) / (2 * rin)
        return voc * i / p_dc_w
    chg = eta_inv * eta_int(p_mw * eta_inv * 1e6)
    # discharging at the same AC power: |P_dc| = P/eta, stored power drawn = |P_dc| + I^2 R
    p_dc_out = p_mw / eta_inv * 1e6
    # stored power for discharge: P_dc_out = Voc*|I| - |I|^2 R  -> |I| from the smaller root
    i_d = (voc - math.sqrt(voc ** 2 - 4 * rin * p_dc_out)) / (2 * rin)
    dis = (p_dc_out / (voc * i_d)) * (1.0 / (1.0 / eta_inv))   # = eta_int_dis * eta_inv
    expected = chg * dis
    assert rt < eta_inv ** 2                                   # internal losses strictly reduce it
    assert_ref(rt, "battery", "bess_round_trip_efficiency", expected=expected)


# ---------------------------------------------------------------------------
# Hydrogen tank: ideal-gas limit and real-gas trend
# ---------------------------------------------------------------------------

def _tank_z(p_bar, t_c=20.0, vol=50.0):
    m = H2StorageTank(volume_m3=vol, p_init_bar=p_bar, p_max_bar=max(p_bar, 10.0) * 2, t_gas_c=t_c)
    m.init()
    mass = m.get_value(["mass_stored_kg"], 0.0)[0]
    n = mass / MOLAR_MASS_H2
    return p_bar * 1e5 * vol / (n * R_GAS * (t_c + 273.15)), mass


def test_tank_ideal_gas_limit_at_low_pressure():
    p_bar, vol, t_c = 0.5, 50.0, 20.0
    _, mass = _tank_z(p_bar, t_c, vol)
    ideal = p_bar * 1e5 * vol / (R_GAS * (t_c + 273.15)) * MOLAR_MASS_H2
    assert_ref(mass, "h2_tank", "mass_at_low_pressure_vs_ideal_gas", expected=ideal)


def test_tank_compressibility_trend_real_gas():
    ps = [1.0, 50.0, 100.0, 200.0, 350.0]
    zs = [_tank_z(p)[0] for p in ps]
    assert all(b > a for a, b in zip(zs, zs[1:])), zs     # Z rises with pressure for H2 (repulsive regime)
    assert zs[0] == pytest.approx(1.0, abs=1e-3)
    assert_ref(zs[-1], "h2_tank", "Z_at_350bar_293K", expected=1.2)


def test_tank_pressure_mass_roundtrip():
    m = H2StorageTank(volume_m3=30.0, p_init_bar=200.0)
    m.init()
    m.set_value(["m_dot_in_kg_hr", "m_dot_out_kg_hr"], [0.0, 0.0])
    p0 = m.get_value(["pressure_bar"], 0.0)[0]
    assert_ref(p0, "h2_tank", "pressure_roundtrip_bar", expected=200.0)


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------

def _textbook_multistage_work(gamma, r_spec, t1_k, p_ratio, n_stages, eta):
    """Specific work of n equal-ratio stages with perfect intercooling back to T1 [J/kg]."""
    cp = gamma / (gamma - 1.0) * r_spec
    return n_stages * cp * t1_k * (p_ratio ** ((gamma - 1.0) / (n_stages * gamma)) - 1.0) / eta


@pytest.mark.parametrize("n_stages", [1, 2, 3, 4])
def test_compressor_work_vs_textbook_isentropic_multistage(n_stages):
    m = H2Compressor(p_suction_bar=30.0, p_discharge_bar=350.0, isentropic_efficiency=0.75,
                     n_stages=n_stages, t_in_c=20.0, t_intercool_c=20.0)
    w_kj_kg, _ = m.specific_work_kj_kg()
    expected = _textbook_multistage_work(1.41, 4124.0, 293.15, 350.0 / 30.0, n_stages, 0.75) / 1000.0
    assert_ref(w_kj_kg, "h2_compressor", "specific_work_kJ_per_kg", expected=expected)


def test_compressor_power_scales_with_flow_and_stage_trend():
    works = []
    for n in (1, 2, 3, 5, 8):
        m = H2Compressor(p_suction_bar=30.0, p_discharge_bar=350.0, n_stages=n, t_intercool_c=20.0, t_in_c=20.0)
        works.append(m.specific_work_kj_kg()[0])
    assert all(b < a for a, b in zip(works, works[1:]))        # intercooling reduces work
    # many stages approach the isothermal limit R T ln(p2/p1) / eta
    m = H2Compressor(p_suction_bar=30.0, p_discharge_bar=350.0, n_stages=400, t_intercool_c=20.0,
                     t_in_c=20.0, isentropic_efficiency=1.0)
    w_iso = 4124.0 * 293.15 * math.log(350.0 / 30.0) / 1000.0
    assert_ref(m.specific_work_kj_kg()[0], "h2_compressor", "isothermal_limit_kJ_per_kg", expected=w_iso)
    # power = specific work x mass flow
    m = H2Compressor(max_flow_kg_hr=500.0)
    m.init()
    m.set_value(["m_dot_in_kg_hr"], [360.0])
    m.step(0.0)
    p_kw = m.get_value(["P_elec_kW"], 0.0)[0]
    assert_ref(p_kw, "h2_compressor", "power_kW_equals_flow_times_work", expected=360.0 / 3600.0 * m.specific_work_kj_kg()[0])


# ---------------------------------------------------------------------------
# Thermal storage
# ---------------------------------------------------------------------------

def test_thermal_storage_energy_balance():
    rng = np.random.default_rng(21)
    m = ThermalStorageTank(capacity_mwh_th=20.0, max_flow_rate_mw_th=10.0, loss_pct_per_day=5.0, soc_init=0.5)
    m.init()
    dt = 60.0
    e0 = m.get_value(["SoC"], 0.0)[0] * 20.0
    net = 0.0
    for k in range(1500):                       # long enough to hit both the full and the empty limit
        m.set_value(["Q_cmd_MW"], [rng.uniform(-10.0, 10.0) * (1.5 if k % 400 < 200 else 0.2)])
        m.step_model(dt, k * dt)
        q, loss = m.get_value(["Q_actual_MW", "Q_loss_MW"], 0.0)
        net += (q - loss) * dt / 3600.0
    e1 = m.get_value(["SoC"], 0.0)[0] * 20.0
    assert_ref(e1 - e0, "thermal_storage", "stored_energy_equals_integral_flow_minus_loss", expected=net)


def test_thermal_storage_standing_loss_decay():
    m = ThermalStorageTank(capacity_mwh_th=20.0, loss_pct_per_day=2.0, soc_init=0.8)
    m.init()
    for k in range(86400 // 60):
        m.step_model(60.0, k * 60.0)
    soc = m.get_value(["SoC"], 0.0)[0]
    assert_ref(soc, "thermal_storage", "standing_loss_after_one_day", expected=0.8 * math.exp(-0.02))


# ---------------------------------------------------------------------------
# Emissions and carbon capture
# ---------------------------------------------------------------------------

def test_ccgt_natural_gas_emission_factor():
    m = CombinedCycle_CCGT(p_rated_mw=400.0, ramp_rate_mw_min=1e6)
    m.init()
    m.set_value(["P_cmd", "T_amb"], [400.0, 15.0])
    for k in range(5):
        m.step_model(60.0, k * 60.0)
    p, fuel, co2 = m.get_value(["P_actual", "fuel_consumption_MMBtu_hr", "CO2_emissions_t_hr"], 0.0)
    assert p == pytest.approx(400.0)
    # IPCC 2006 default: 56.1 kg CO2 / GJ (NCV) ; 1 MMBtu = 1.05506 GJ
    assert_ref(co2 / fuel, "emissions", "natural_gas_t_co2_per_MMBtu")
    assert_ref(co2 / fuel, "emissions", "natural_gas_ipcc_56p1_kg_per_GJ", expected=56.1e-3 * 1.05506)
    # fuel from the heat rate: kJ/h / (kJ per MMBtu)
    assert_ref(fuel, "emissions", "ccgt_fuel_from_heat_rate", expected=400e3 * 6400.0 / 1.05506e6)
    assert 0.33 < co2 / p < 0.39            # t CO2 / MWh of a modern CCGT


def test_carbon_capture_energy_per_tonne():
    m = CarbonCapture_Amine(capture_rate_nom=0.9, spec_steam_reboiler_gj_tco2=3.0, spec_elec_kwh_tco2=120.0)
    m.init()
    m.set_value(["m_dot_CO2_in_t_hr"], [100.0])
    m.step(0.0)
    cap, p_el, q_th = m.get_value(["m_dot_CO2_captured_t_hr", "P_elec_parasitic_MW", "Q_steam_reboiler_MW_th"], 0.0)
    assert_ref(cap, "carbon_capture", "captured_mass_flow_t_h", expected=90.0)
    # MW_th * 3600 s/h / 1000 = GJ/h  ->  per tonne captured
    assert_ref(q_th * 3.6 / cap, "carbon_capture", "reboiler_GJ_per_tonne", expected=3.0)
    assert_ref(p_el * 1000.0 / cap, "carbon_capture", "parasitic_kWh_per_tonne", expected=120.0)
    assert 2.4 <= q_th * 3.6 / cap <= 4.2        # typical amine range (GJ/t), sanity band


def test_absorption_chiller_energy_balance():
    m = AbsorptionChiller(q_cooling_max_mw=5.0, cop_thermal=0.7)
    m.init()
    m.set_value(["Q_cool_cmd_MW", "Q_waste_heat_available_MW"], [3.0, 10.0])
    m.step(0.0)
    cool, heat = m.get_value(["Q_cool_actual_MW", "Q_heat_consumed_MW"], 0.0)
    assert_ref(cool / heat, "absorption_chiller", "thermal_cop", expected=0.7)

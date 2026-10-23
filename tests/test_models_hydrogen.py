# -*- coding: utf-8 -*-
"""Physics, conservation and consistency tests for energysim.models.hydrogen."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from energysim.models.hydrogen import (  # noqa: E402
    PEMElectrolyzer, AlkalineElectrolyzer, SOECElectrolyzer, H2Compressor,
    H2StorageTank, FuelCell_PEM, FuelCell_SOFC, MethanationReactor,
    FARADAY_CONSTANT, MOLAR_MASS_H2, LHV_H2_MJ_KG, MOLAR_MASS_H2O,
)


def _out(model, names):
    return dict(zip(names, model.get_value(list(names), 0.0)))


# ---------------------------------------------------------------------------
# 1. PEM
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p_cmd", [0.3, 1.0, 2.5, 5.0])
def test_pem_faraday_exactness(p_cmd):
    pem = PEMElectrolyzer(p_nominal_mw=5.0)
    pem.set_value(["P_cmd"], [p_cmd])
    pem.step(0.0)
    o = _out(pem, ["m_dot_H2_kg_hr", "I_stack_A", "eta_faraday", "m_dot_H2O_kg_hr"])
    expected = o["eta_faraday"] * pem.n_cells * o["I_stack_A"] / (2 * FARADAY_CONSTANT) * MOLAR_MASS_H2 * 3600.0
    assert o["m_dot_H2_kg_hr"] == pytest.approx(expected, rel=1e-9)
    assert o["m_dot_H2O_kg_hr"] == pytest.approx(o["m_dot_H2_kg_hr"] * MOLAR_MASS_H2O / MOLAR_MASS_H2, rel=1e-9)


def test_pem_current_density_uses_area_and_power_is_consistent():
    pem = PEMElectrolyzer(p_nominal_mw=5.0, cell_area_cm2=1000.0, n_cells=1500)
    pem.set_value(["P_cmd"], [3.0])
    pem.step(0.0)
    o = _out(pem, ["P_actual", "I_stack_A", "j_A_cm2", "V_cell"])
    assert o["I_stack_A"] == pytest.approx(o["j_A_cm2"] * 1000.0, rel=1e-12)
    assert pem.n_cells * o["I_stack_A"] * o["V_cell"] / 1e6 == pytest.approx(o["P_actual"], rel=1e-4)
    # V is the lookup at the actual j
    assert o["V_cell"] == pytest.approx(float(pem.polarization_curve(o["j_A_cm2"])))


def test_pem_nominal_power_matches_geometry():
    pem = PEMElectrolyzer(p_nominal_mw=5.0)
    pem.set_value(["P_cmd"], [5.0])
    pem.step(0.0)
    assert pem.get_value(["j_A_cm2"], 0.0)[0] == pytest.approx(pem.j_nominal, rel=1e-9)
    # bigger area -> lower current density for the same power
    big = PEMElectrolyzer(p_nominal_mw=5.0, cell_area_cm2=2000.0)
    assert big.j_nominal < pem.j_nominal
    # impossible geometry is rejected
    with pytest.raises(ValueError):
        PEMElectrolyzer(p_nominal_mw=50.0, cell_area_cm2=100.0, n_cells=100)


def test_pem_waste_heat_relative_to_thermoneutral_voltage():
    pem = PEMElectrolyzer(p_nominal_mw=5.0)
    pem.set_value(["P_cmd"], [4.0])
    pem.step(0.0)
    o = _out(pem, ["Q_waste_MW", "I_stack_A", "V_cell", "P_actual"])
    assert o["Q_waste_MW"] == pytest.approx(pem.n_cells * o["I_stack_A"] * (o["V_cell"] - 1.48) / 1e6, rel=1e-12)
    assert 0.0 < o["Q_waste_MW"] < o["P_actual"]
    # sub-thermoneutral operation (V < 1.48 V) is clipped to zero heat
    low = PEMElectrolyzer(p_nominal_mw=5.0, min_turndown=0.0005)
    low.set_value(["P_cmd"], [0.005])
    low.step(0.0)
    assert low.get_value(["V_cell"], 0.0)[0] < 1.48
    assert low.get_value(["Q_waste_MW"], 0.0)[0] == 0.0


def test_pem_ramp_limit_and_turndown():
    pem = PEMElectrolyzer(p_nominal_mw=5.0, ramp_rate_pct_per_s=10.0)
    pem.set_value(["P_cmd"], [5.0])
    pem.step(0.0)
    assert pem.get_value(["P_actual"], 0.0)[0] == pytest.approx(0.5)
    pem.step(1.0)
    assert pem.get_value(["P_actual"], 0.0)[0] == pytest.approx(1.0)
    pem.set_value(["P_cmd"], [0.0])
    pem.step(2.0)
    assert pem.get_value(["P_actual"], 0.0)[0] == pytest.approx(0.5)
    # Below min turndown with unlimited ramp -> idle
    idle = PEMElectrolyzer(p_nominal_mw=5.0)
    idle.set_value(["P_cmd"], [0.1])
    idle.step(0.0)
    assert idle.get_value(["m_dot_H2_kg_hr", "Q_waste_MW"], 0.0) == [0.0, 0.0]


# ---------------------------------------------------------------------------
# 2. Alkaline
# ---------------------------------------------------------------------------

def test_alkaline_part_load_penalty_parameter():
    default = AlkalineElectrolyzer(p_nominal_mw=10.0, spec_consumption_kwh_kg=50.0)
    flat = AlkalineElectrolyzer(p_nominal_mw=10.0, spec_consumption_kwh_kg=50.0, part_load_penalty=0.0)
    for m in (default, flat):
        m.set_value(["P_cmd"], [5.0])
        m.step(0.0)
    assert flat.get_value(["m_dot_H2_kg_hr"], 0.0)[0] == pytest.approx(5000.0 / 50.0)
    assert default.get_value(["m_dot_H2_kg_hr"], 0.0)[0] == pytest.approx(5000.0 / (50.0 * (1 + 0.15 * 0.5)))
    # at full load the penalty vanishes
    default.set_value(["P_cmd"], [10.0])
    default.step(1.0)
    assert default.get_value(["m_dot_H2_kg_hr"], 0.0)[0] == pytest.approx(10000.0 / 50.0)


def test_alkaline_ramp_and_state_string():
    ael = AlkalineElectrolyzer(p_nominal_mw=10.0, ramp_rate_pct_per_s=5.0)
    assert ael.get_value(["state"], 0.0) == ["OFF"]
    ael.set_value(["P_cmd"], [10.0])
    ael.step(0.0)
    assert ael.get_value(["P_actual", "state"], 0.0) == [pytest.approx(0.5), "RUNNING"]
    ael.step(1.0)
    assert ael.get_value(["P_actual"], 0.0)[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. SOEC
# ---------------------------------------------------------------------------

def test_soec_guard_and_steam_heat():
    with pytest.raises(ValueError):
        SOECElectrolyzer(operating_temp_c=300.0)
    with pytest.raises(ValueError):
        SOECElectrolyzer(operating_temp_c=1200.0)
    soec = SOECElectrolyzer(p_nominal_mw=2.0, steam_enthalpy_kj_kg=3600.0)
    soec.set_value(["P_cmd", "steam_available_kg_hr"], [1.5, 1e5])
    soec.step(0.0)
    o = _out(soec, ["m_dot_H2_kg_hr", "steam_consumed_kg_hr", "Q_steam_MW_th", "P_actual"])
    assert o["steam_consumed_kg_hr"] == pytest.approx(9.0 * o["m_dot_H2_kg_hr"])
    assert o["Q_steam_MW_th"] == pytest.approx(o["steam_consumed_kg_hr"] / 3600.0 * 3600.0 / 1000.0)
    assert o["P_actual"] == pytest.approx(o["m_dot_H2_kg_hr"] * 38.0 / 1000.0)


def test_soec_steam_limited():
    soec = SOECElectrolyzer(p_nominal_mw=2.0)
    soec.set_value(["P_cmd", "steam_available_kg_hr"], [2.0, 90.0])
    soec.step(0.0)
    o = _out(soec, ["m_dot_H2_kg_hr", "steam_consumed_kg_hr", "P_actual"])
    assert o["m_dot_H2_kg_hr"] == pytest.approx(10.0)
    assert o["steam_consumed_kg_hr"] == pytest.approx(90.0)
    assert o["P_actual"] < 2.0


# ---------------------------------------------------------------------------
# 4. Compressor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_compressor_matches_textbook_multistage(n):
    comp = H2Compressor(max_flow_kg_hr=1000.0, p_suction_bar=30.0, p_discharge_bar=350.0,
                        isentropic_efficiency=0.7, n_stages=n, t_in_c=20.0, t_intercool_c=20.0)
    comp.set_value(["m_dot_in_kg_hr"], [360.0])
    comp.step(0.0)
    g, R = 1.41, 4124.0
    r = (350.0 / 30.0) ** (1.0 / n)
    w = n * g / (g - 1) * R * 293.15 * (r ** ((g - 1) / g) - 1) / 0.7  # J/kg
    assert comp.get_value(["P_elec_kW"], 0.0)[0] == pytest.approx(0.1 * w / 1000.0, rel=1e-9)
    q = comp.get_value(["Q_intercool_kW"], 0.0)[0]
    if n == 1:
        assert q == 0.0
    else:
        assert q > 0.0


def test_compressor_more_stages_less_work_and_flow_cap():
    works = []
    for n in (1, 2, 3):
        c = H2Compressor(n_stages=n, max_flow_kg_hr=100.0)
        c.set_value(["m_dot_in_kg_hr"], [1000.0])
        c.step(0.0)
        works.append(c.get_value(["P_elec_kW"], 0.0)[0])
        assert c.get_value(["m_dot_out_kg_hr"], 0.0)[0] == 100.0
    assert works[0] > works[1] > works[2]
    with pytest.raises(ValueError):
        H2Compressor(n_stages=0)


def test_compressor_intercool_heat_energy_balance():
    # With full intercooling to the inlet temperature, per-stage heat removed = per-stage
    # shaft work x eta (i.e. the gas enthalpy rise) for the first n-1 stages.
    c = H2Compressor(n_stages=3, t_in_c=20.0, t_intercool_c=20.0, isentropic_efficiency=0.75, max_flow_kg_hr=1000)
    c.set_value(["m_dot_in_kg_hr"], [500.0])
    c.step(0.0)
    p, q = c.get_value(["P_elec_kW", "Q_intercool_kW"], 0.0)
    assert q == pytest.approx(p * (2.0 / 3.0), rel=1e-9)


# ---------------------------------------------------------------------------
# 5. Storage tank
# ---------------------------------------------------------------------------

def _tank_step(tank, m_in, m_out, dt):
    m0 = tank.mass_stored_kg
    tank.set_value(["m_dot_in_kg_hr", "m_dot_out_kg_hr"], [m_in, m_out])
    tank.step_model(dt, 0.0)
    rej, uns = tank.get_value(["m_dot_rejected_kg_hr", "m_dot_unserved_kg_hr"], 0.0)
    return (tank.mass_stored_kg - m0), rej, uns


def test_tank_pressure_follows_mass_and_initial_pressure():
    tank = H2StorageTank(volume_m3=20.0, p_init_bar=100.0)
    assert tank.pressure_bar == pytest.approx(100.0, rel=1e-9)
    assert tank.get_value(["pressure_bar"], 0.0)[0] == pytest.approx(100.0, rel=1e-9)
    tank2 = H2StorageTank(volume_m3=20.0, p_init_bar=100.0)
    _tank_step(tank2, 200.0, 0.0, 600.0)
    assert tank2.pressure_bar > 100.0
    assert tank2.pressure_bar == pytest.approx(tank2._pressure_from_mass(tank2.mass_stored_kg))
    assert tank2._mass_from_pressure(tank2.pressure_bar) == pytest.approx(tank2.mass_stored_kg, rel=1e-9)


def test_tank_rejects_inflow_beyond_pmax():
    tank = H2StorageTank(volume_m3=1.0, p_max_bar=350.0, p_init_bar=340.0)
    dm, rej, uns = _tank_step(tank, 10000.0, 0.0, 3600.0)
    assert tank.pressure_bar <= 350.0 + 1e-9
    assert rej > 0.0 and uns == 0.0
    assert tank.mass_stored_kg == pytest.approx(tank.mass_capacity_kg)
    assert 10000.0 - rej == pytest.approx(dm / 1.0)  # dt = 1 h
    # full tank -> everything rejected
    dm, rej, uns = _tank_step(tank, 50.0, 0.0, 3600.0)
    assert rej == pytest.approx(50.0) and dm == pytest.approx(0.0, abs=1e-9)


def test_tank_unserved_and_mass_never_negative():
    tank = H2StorageTank(volume_m3=0.5, p_init_bar=10.0)
    m0 = tank.mass_stored_kg
    dm, rej, uns = _tank_step(tank, 0.0, 1000.0, 3600.0)
    assert tank.mass_stored_kg == 0.0
    assert uns == pytest.approx(1000.0 - m0)
    assert tank.pressure_bar == 0.0
    assert tank.get_value(["SoC"], 0.0)[0] == 0.0


@pytest.mark.parametrize("seed", range(5))
def test_tank_conservation_random(seed):
    rng = np.random.default_rng(seed)
    tank = H2StorageTank(volume_m3=2.0, p_init_bar=float(rng.uniform(5, 340)))
    for _ in range(200):
        m_in, m_out = rng.uniform(0, 400, 2)
        dt = float(rng.choice([30.0, 300.0, 900.0]))
        dm, rej, uns = _tank_step(tank, m_in, m_out, dt)
        assert (m_in - m_out - rej + uns) * dt / 3600.0 == pytest.approx(dm, abs=1e-9)
        assert tank.mass_stored_kg >= 0.0
        assert tank.pressure_bar <= tank.p_max + 1e-9
        assert rej >= 0.0 and uns >= 0.0
        assert 0.0 <= tank.get_value(["SoC"], 0.0)[0] <= 1.0


# ---------------------------------------------------------------------------
# 6. Methanation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("h2_in,eff", [(20.0, 0.95), (100.0, 0.8), (500.0, 0.99), (0.0, 0.9)])
def test_methanation_mass_balance(h2_in, eff):
    m = MethanationReactor(max_h2_feed_kg_hr=200.0, conversion_efficiency=eff)
    m.set_value(["m_dot_H2_kg_hr"], [h2_in])
    m.step(0.0)
    o = _out(m, ["m_dot_H2_fed_kg_hr", "m_dot_CO2_consumed_kg_hr", "m_dot_CH4_kg_hr",
                 "m_dot_H2O_kg_hr", "m_dot_H2_unreacted_kg_hr"])
    lhs = o["m_dot_H2_fed_kg_hr"] + o["m_dot_CO2_consumed_kg_hr"]
    rhs = o["m_dot_CH4_kg_hr"] + o["m_dot_H2O_kg_hr"] + o["m_dot_H2_unreacted_kg_hr"]
    assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-12)
    assert o["m_dot_H2_fed_kg_hr"] == pytest.approx(min(200.0, h2_in))
    assert o["m_dot_H2_unreacted_kg_hr"] == pytest.approx(o["m_dot_H2_fed_kg_hr"] * (1 - eff))


def test_methanation_stoichiometry_and_heat():
    m = MethanationReactor(conversion_efficiency=1.0)
    m.set_value(["m_dot_H2_kg_hr"], [100.0])
    m.step(0.0)
    o = _out(m, ["m_dot_CH4_kg_hr", "m_dot_H2O_kg_hr", "Q_reaction_MW"])
    mol_ch4_hr = 100.0 / 2.01588e-3 / 4.0
    assert o["m_dot_CH4_kg_hr"] == pytest.approx(mol_ch4_hr * 16.04246e-3, rel=1e-9)
    assert o["m_dot_H2O_kg_hr"] / o["m_dot_CH4_kg_hr"] == pytest.approx(2 * 18.01528 / 16.04246, rel=1e-9)
    mol_ch4_s = (o["m_dot_CH4_kg_hr"] / 16.04246e-3) / 3600.0
    assert o["Q_reaction_MW"] == pytest.approx(mol_ch4_s * 165.0 / 1000.0, rel=1e-9)


# ---------------------------------------------------------------------------
# 7. Fuel cells
# ---------------------------------------------------------------------------

def test_pem_fuel_cell_energy_balance_and_idle():
    fc = FuelCell_PEM(p_nominal_mw=1.0, electrical_efficiency=0.5, ramp_rate_mw_min=60.0)
    fc.set_value(["P_cmd"], [0.0])
    fc.step(0.0)
    assert fc.get_value(["P_actual", "m_dot_H2_kg_hr", "Q_waste_MW"], 0.0) == [0.0, 0.0, 0.0]
    fc.set_value(["P_cmd"], [0.8])
    for t in range(5):
        fc.step(float(t))
    p, m, q = fc.get_value(["P_actual", "m_dot_H2_kg_hr", "Q_waste_MW"], 0.0)
    assert p == pytest.approx(0.8)
    fuel_mw = m / 3600.0 * LHV_H2_MJ_KG
    assert fuel_mw == pytest.approx(p + q, rel=1e-12)
    assert p / fuel_mw == pytest.approx(0.5)


def test_pem_fuel_cell_ramp():
    fc = FuelCell_PEM(p_nominal_mw=1.0, ramp_rate_mw_min=6.0)  # 0.1 MW/s
    fc.set_value(["P_cmd"], [1.0])
    fc.step(0.0)
    assert fc.get_value(["P_actual"], 0.0)[0] == pytest.approx(0.1)


def test_sofc_energy_balance_and_idle():
    fc = FuelCell_SOFC(p_nominal_mw=1.0, electrical_efficiency=0.6, thermal_efficiency=0.25)
    fc.set_value(["P_cmd"], [0.0])
    fc.step(0.0)
    assert fc.get_value(["P_actual", "m_dot_H2_kg_hr", "Q_chp_MW", "Q_loss_MW"], 0.0) == [0.0] * 4
    fc.set_value(["P_cmd"], [0.7])
    fc.step(1.0)
    p, m, qc, ql = fc.get_value(["P_actual", "m_dot_H2_kg_hr", "Q_chp_MW", "Q_loss_MW"], 0.0)
    fuel_mw = m / 3600.0 * LHV_H2_MJ_KG
    assert fuel_mw == pytest.approx(p + qc + ql, rel=1e-12)
    assert ql == pytest.approx(fuel_mw * 0.15, rel=1e-9)
    with pytest.raises(ValueError):
        FuelCell_SOFC(electrical_efficiency=0.7, thermal_efficiency=0.4)


# ---------------------------------------------------------------------------
# 10. Property / consistency tests across all models
# ---------------------------------------------------------------------------

def _all_models():
    return [
        (PEMElectrolyzer(), {"P_cmd": [0.0, 0.1, 2.5, 5.0, 9.0], "T_cooling_in": [10.0, 40.0]}),
        (AlkalineElectrolyzer(), {"P_cmd": [0.0, 1.0, 2.5, 7.0, 30.0]}),
        (SOECElectrolyzer(), {"P_cmd": [0.0, 0.5, 2.0, 5.0], "steam_available_kg_hr": [0.0, 50.0, 1e5]}),
        (H2Compressor(), {"m_dot_in_kg_hr": [0.0, 100.0, 5000.0]}),
        (H2StorageTank(), {"m_dot_in_kg_hr": [0.0, 300.0, 9000.0], "m_dot_out_kg_hr": [0.0, 300.0, 9000.0]}),
        (FuelCell_PEM(), {"P_cmd": [0.0, 0.5, 3.0]}),
        (FuelCell_SOFC(), {"P_cmd": [0.0, 0.5, 3.0]}),
        (MethanationReactor(), {"m_dot_H2_kg_hr": [0.0, 10.0, 1000.0]}),
    ]


def _combos(grid):
    keys = list(grid)
    out = [{}]
    for k in keys:
        out = [{**o, k: v} for o in out for v in grid[k]]
    return out


def test_all_models_finite_nonnegative_flows():
    for model, grid in _all_models():
        for combo in _combos(grid):
            model.set_value(list(combo), list(combo.values()))
            for t in range(3):
                model.step_model(60.0, 60.0 * t)
                for name in model.outputs:
                    v = model.get_value([name], 0.0)[0]
                    if isinstance(v, str):
                        continue
                    assert math.isfinite(v), (type(model).__name__, name, combo)
                    if "T_stack" != name:  # temperatures may be anything finite
                        assert v >= -1e-12, (type(model).__name__, name, v, combo)


STATE_ATTRS = {
    "PEMElectrolyzer": ["p_actual_mw", "t_stack_c"],
    "AlkalineElectrolyzer": ["p_actual", "state"],
    "H2StorageTank": ["mass_stored_kg"],
    "FuelCell_PEM": ["p_actual"],
}


def test_dt_zero_leaves_state_unchanged_and_fills_outputs():
    for model, grid in _all_models():
        cls = type(model).__name__
        for combo in _combos(grid):
            model.set_value(list(combo), list(combo.values()))
            model.step_model(30.0, 0.0)
            before = {a: getattr(model, a) for a in STATE_ATTRS.get(cls, [])}
            # change the inputs and step with dt == 0
            for k in combo:
                model.set_value([k], [combo[k] * 0.5 + 1.0])
            model.step_model(0.0, 30.0)
            after = {a: getattr(model, a) for a in STATE_ATTRS.get(cls, [])}
            assert before == after, (cls, combo)
            for name in model.outputs:
                assert name in model._outputs_val


def test_dt_zero_first_step_fills_valid_outputs():
    for model, _ in _all_models():
        model.step_model(0.0, 0.0)
        for name in model.outputs:
            v = model.get_value([name], 0.0)[0]
            assert v is not None
            if not isinstance(v, str):
                assert math.isfinite(v)


def test_units_declared_for_every_variable():
    for model, _ in _all_models():
        cls = type(model)
        assert isinstance(getattr(cls, "units", None), dict), cls.__name__
        for name in list(model.inputs) + list(model.outputs):
            assert name in cls.units, (cls.__name__, name)
            assert isinstance(cls.units[name], str) and cls.units[name]


def test_no_rollback_api_added():
    for model, _ in _all_models():
        assert "save_state" not in type(model).__dict__
        assert "restore_state" not in type(model).__dict__

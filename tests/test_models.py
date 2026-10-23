# -*- coding: utf-8 -*-
"""
Comprehensive Unit and Integration Test Suite for energysim Physical Component Models.
"""

import sys
import os
import pytest
import numpy as np

# Ensure src is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from energysim import world, SimulatorAdapter
from energysim.models.common import Curve1D, Surface2D, OperatingStateMachine, OperatingState, ModelComponent
from energysim.models.storage import BatteryBESS, FlowBattery, Flywheel
from energysim.models.renewables import PhysicsPV, WindTurbine, SolarThermalCollector
from energysim.models.hydrogen import (
    PEMElectrolyzer, AlkalineElectrolyzer, SOECElectrolyzer,
    H2Compressor, H2StorageTank, FuelCell_PEM, FuelCell_SOFC, MethanationReactor,
)
from energysim.models.thermal import GasTurbine_OCGT, CombinedCycle_CCGT, ReciprocatingGenset, CHP_SteamTurbine
from energysim.models.hvac_thermal import (
    IndustrialHeatPump, ElectricBoiler, ElectricChiller, AbsorptionChiller, ThermalStorageTank,
)
from energysim.models.carbon import CarbonCapture_Amine
from energysim.models.loads import DataCenterLoad, IndustrialProcessSteam
from energysim.models import presets


# =============================================================================
# 1. Curve and Surface Utility Tests
# =============================================================================

def test_curve1d_interpolation_and_clamping():
    c = Curve1D(x=[0.0, 0.5, 1.0], y=[10.0, 20.0, 30.0], clamp=True, min_val=5.0, max_val=35.0)
    assert c(0.0) == pytest.approx(10.0)
    assert c(0.25) == pytest.approx(15.0)
    assert c(0.5) == pytest.approx(20.0)
    assert c(1.0) == pytest.approx(30.0)
    # Clamping below and above
    assert c(-0.5) == pytest.approx(10.0)
    assert c(2.0) == pytest.approx(30.0)

    # Serialization
    d = c.to_dict()
    c2 = Curve1D.from_dict(d)
    assert c2(0.25) == pytest.approx(15.0)


def test_surface2d_bilinear_interpolation():
    x = [0.0, 10.0]
    y = [0.0, 10.0]
    z = [[0.0, 10.0],
         [10.0, 20.0]]
    s = Surface2D(x, y, z, clamp=True)
    assert s(0.0, 0.0) == pytest.approx(0.0)
    assert s(10.0, 10.0) == pytest.approx(20.0)
    assert s(5.0, 5.0) == pytest.approx(10.0)
    assert s(-5.0, -5.0) == pytest.approx(0.0)


# =============================================================================
# 2. Operating State Machine Tests
# =============================================================================

def test_state_machine_ramping_and_cold_warm():
    sm = OperatingStateMachine(
        ramp_up_rate_min=60.0,  # 1 unit/sec
        ramp_down_rate_min=60.0,
        cold_start_time_hr=48.0,
        warm_start_time_hr=8.0,
    )
    # Offline for 50 hours -> Cold
    sm.time_offline_sec = 50.0 * 3600.0
    assert sm.determine_startup_category() == "COLD"

    # Offline for 10 hours -> Warm
    sm.time_offline_sec = 10.0 * 3600.0
    assert sm.determine_startup_category() == "WARM"

    # Offline for 2 hours -> Hot
    sm.time_offline_sec = 2.0 * 3600.0
    assert sm.determine_startup_category() == "HOT"

    # Test ramp limit
    val = sm.apply_ramp(current=0.0, target=10.0, dt_sec=2.0)
    assert val == pytest.approx(2.0)


# =============================================================================
# 3. Storage Models Tests
# =============================================================================

def test_battery_bess_charge_discharge_thermal_aging():
    bess = BatteryBESS(capacity_mwh=1.0, p_max_mw=0.5, soc_init=0.5, step_size=10.0)
    assert isinstance(bess, SimulatorAdapter)

    # 1. Charge command of 0.2 MW
    bess.set_value(["P_cmd", "T_amb"], [0.2, 25.0])
    bess.step(0.0)
    vals = bess.get_value(["P_actual", "SoC", "T_batt", "SOH", "V_terminal"], 0.0)
    p_act, soc, t_batt, soh, v_term = vals

    assert p_act == pytest.approx(0.2)
    assert soc > 0.5  # SoC increased
    assert soh <= 1.0  # SOH began micro-degrading
    assert v_term > 800.0  # Terminal voltage under charge exceeds OCV

    # 2. Discharge command of -0.4 MW
    bess.set_value(["P_cmd"], [-0.4])
    bess.step(10.0)
    vals2 = bess.get_value(["P_actual", "SoC"], 10.0)
    assert vals2[0] == pytest.approx(-0.4)
    assert vals2[1] < soc  # Discharged


def test_flow_battery_and_flywheel():
    vrfb = FlowBattery(stack_p_max_mw=1.0, tank_capacity_mwh=4.0, step_size=5.0)
    vrfb.set_value(["P_cmd"], [0.5])
    vrfb.step(0.0)
    p_act = vrfb.get_value(["P_actual"], 0.0)[0]
    assert p_act > 0.5  # Includes pump parasitics

    fly = Flywheel(power_max_mw=0.2, energy_capacity_kwh=50.0, step_size=1.0)
    fly.set_value(["P_cmd"], [0.1])
    fly.step(0.0)
    assert fly.get_value(["P_actual"], 0.0)[0] == pytest.approx(0.1)


# =============================================================================
# 4. Renewable Models Tests
# =============================================================================

def test_physics_pv_irradiance_clipping_curtailment():
    pv = PhysicsPV(p_dc_stc_mw=2.0, inverter_p_ac_max_mw=1.5, step_size=1.0)

    # Night time
    pv.set_value(["G_poa", "T_amb"], [0.0, 20.0])
    pv.step(0.0)
    assert pv.get_value(["P_ac"], 0.0)[0] == 0.0

    # Sunny day STC 1000 W/m2 at 25 C
    pv.set_value(["G_poa", "T_amb"], [1000.0, 25.0])
    pv.step(1.0)
    p_ac = pv.get_value(["P_ac"], 1.0)[0]
    # Inverter clips at 1.5 MW
    assert p_ac == pytest.approx(1.5, rel=1e-2)

    # Curtailment command
    pv.set_value(["curtail_limit"], [0.8])
    pv.step(2.0)
    assert pv.get_value(["P_ac"], 2.0)[0] == pytest.approx(0.8)


def test_wind_turbine_power_curve():
    wt = WindTurbine(p_rated_mw=3.0, cut_in_speed_ms=3.0, rated_speed_ms=12.0, cut_out_speed_ms=25.0, step_size=1.0)

    # Below cut-in
    wt.set_value(["v_wind"], [2.0])
    wt.step(0.0)
    assert wt.get_value(["P_actual"], 0.0)[0] == pytest.approx(0.0, abs=1e-3)

    # Rated wind speed (advance a few seconds to let rotor inertia lag catch up)
    wt.set_value(["v_wind"], [12.0])
    wt.advance(0.0, 30.0)
    p_act = wt.get_value(["P_actual"], 30.0)[0]
    assert p_act > 2.7  # Close to 3.0 MW rated


# =============================================================================
# 5. Hydrogen and PtX Models Tests
# =============================================================================

def test_pem_electrolyzer_production_and_waste_heat():
    pem = PEMElectrolyzer(p_nominal_mw=5.0, step_size=1.0)
    pem.set_value(["P_cmd", "T_cooling_in"], [4.0, 25.0])
    pem.step(0.0)

    res = pem.get_value(["P_actual", "m_dot_H2_kg_hr", "m_dot_H2O_kg_hr", "Q_waste_MW"], 0.0)
    p_act, h2_rate, h2o_rate, q_waste = res

    assert p_act == pytest.approx(4.0)
    assert h2_rate > 50.0  # ~60-80 kg/hr for 4 MW
    assert h2o_rate == pytest.approx(h2_rate * 8.936, rel=0.05)  # Stoichiometric water
    assert q_waste > 0.5  # Significant recoverable waste heat


def test_alkaline_turndown_safety_cutoff():
    ael = AlkalineElectrolyzer(p_nominal_mw=10.0, min_turndown_pct=25.0, step_size=1.0)

    # Below 25% (e.g. 2.0 MW) -> Must trip to STANDBY
    ael.set_value(["P_cmd"], [2.0])
    ael.step(0.0)
    assert ael.get_value(["state", "P_actual", "m_dot_H2_kg_hr"], 0.0) == ["STANDBY", 0.0, 0.0]

    # At or above 25% (e.g. 5.0 MW) -> RUNNING
    ael.set_value(["P_cmd"], [5.0])
    ael.step(1.0)
    state, p_act, h2_rate = ael.get_value(["state", "P_actual", "m_dot_H2_kg_hr"], 1.0)
    assert state == "RUNNING"
    assert p_act == pytest.approx(5.0)
    assert h2_rate > 70.0


def test_soec_steam_electrolysis():
    soec = SOECElectrolyzer(p_nominal_mw=2.0, step_size=1.0)
    # Provide steam and power
    soec.set_value(["P_cmd", "steam_available_kg_hr"], [1.5, 500.0])
    soec.step(0.0)
    p_act, h2_rate, steam_used = soec.get_value(["P_actual", "m_dot_H2_kg_hr", "steam_consumed_kg_hr"], 0.0)
    assert p_act > 0.0
    assert h2_rate > 0.0
    assert steam_used > 0.0


def test_h2_compressor_and_storage():
    comp = H2Compressor(max_flow_kg_hr=200.0, p_suction_bar=30.0, p_discharge_bar=350.0, step_size=1.0)
    comp.set_value(["m_dot_in_kg_hr"], [100.0])
    comp.step(0.0)
    p_kw = comp.get_value(["P_elec_kW"], 0.0)[0]
    assert p_kw > 10.0  # Realistic compressor electrical draw

    tank = H2StorageTank(volume_m3=20.0, p_max_bar=350.0, p_init_bar=100.0, step_size=60.0)
    tank.set_value(["m_dot_in_kg_hr", "m_dot_out_kg_hr"], [100.0, 0.0])
    tank.step(0.0)
    p_bar = tank.get_value(["pressure_bar"], 0.0)[0]
    assert p_bar > 100.0  # Pressure increased due to mass injection


def test_fuel_cells_and_methanation():
    fc = FuelCell_PEM(p_nominal_mw=1.0, step_size=1.0)
    fc.set_value(["P_cmd"], [0.5])
    fc.advance(0.0, 30.0)
    p_act, h2_cons = fc.get_value(["P_actual", "m_dot_H2_kg_hr"], 30.0)
    assert p_act == pytest.approx(0.5)
    assert h2_cons > 10.0

    sabatier = MethanationReactor(max_h2_feed_kg_hr=50.0, step_size=1.0)
    sabatier.set_value(["m_dot_H2_kg_hr"], [20.0])
    sabatier.step(0.0)
    ch4_out, co2_used = sabatier.get_value(["m_dot_CH4_kg_hr", "m_dot_CO2_consumed_kg_hr"], 0.0)
    assert ch4_out > 0.0
    assert co2_used > 0.0


# =============================================================================
# 6. Thermal Turbomachinery & Cogeneration Tests
# =============================================================================

def test_ocgt_ambient_derate_and_startup():
    ocgt = GasTurbine_OCGT(p_rated_mw=100.0, min_stable_gen_pct=30.0, cold_start_duration_sec=60.0, step_size=10.0)
    # Start command of 80 MW at mild ambient 15 C
    ocgt.set_value(["P_cmd", "T_amb"], [80.0, 15.0])
    ocgt.step(0.0)
    state, p_act = ocgt.get_value(["state", "P_actual"], 0.0)
    assert state == "STARTING"
    assert p_act == 0.0

    # Advance past startup timer
    ocgt.advance(0.0, 120.0)
    state, p_act, fuel = ocgt.get_value(["state", "P_actual", "fuel_consumption_MMBtu_hr"], 120.0)
    assert state == "RUNNING"
    assert p_act >= 30.0  # Operating above MSG
    assert fuel > 100.0


def test_ccgt_gt_st_split_and_emissions():
    ccgt = CombinedCycle_CCGT(p_rated_mw=400.0, gt_ratio=0.67, ramp_rate_mw_min=300.0, step_size=1.0)
    ccgt.set_value(["P_cmd"], [300.0])
    ccgt.advance(0.0, 60.0)
    p_act, p_gt, p_st, co2 = ccgt.get_value(["P_actual", "P_gt", "P_st", "CO2_emissions_t_hr"], 60.0)
    assert p_act == pytest.approx(300.0, rel=1e-2)
    assert p_gt == pytest.approx(p_act * 0.67, rel=1e-2)
    assert p_st == pytest.approx(p_act * 0.33, rel=1e-2)
    assert co2 > 50.0  # Realistic CO2 tonnage


# =============================================================================
# 7. HVAC & Thermal Models Tests
# =============================================================================

def test_heat_pump_cop_lift():
    hp = IndustrialHeatPump(q_max_thermal_mw=5.0, carnot_efficiency=0.50, step_size=1.0)
    # Low lift: source 20 C, sink 60 C (deltaT = 40)
    hp.set_value(["Q_heat_cmd_MW", "T_source", "T_sink"], [4.0, 20.0, 60.0])
    hp.step(0.0)
    cop1 = hp.get_value(["COP"], 0.0)[0]

    # High lift: source 0 C, sink 80 C (deltaT = 80)
    hp.set_value(["T_source", "T_sink"], [0.0, 80.0])
    hp.step(1.0)
    cop2 = hp.get_value(["COP"], 1.0)[0]

    # Higher temperature lift must yield lower COP
    assert cop1 > cop2
    assert cop1 > 3.0


def test_electric_boiler_and_chillers():
    boiler = ElectricBoiler(p_max_mw=10.0, efficiency=0.99, step_size=1.0)
    boiler.set_value(["P_cmd"], [5.0])
    boiler.advance(0.0, 10.0)
    p_act, q_th = boiler.get_value(["P_actual", "Q_thermal_MW"], 10.0)
    assert p_act == pytest.approx(5.0)
    assert q_th == pytest.approx(5.0 * 0.99)

    chiller = ElectricChiller(q_cooling_max_mw=2.0, cop_nominal=5.0, step_size=1.0)
    chiller.set_value(["Q_cool_cmd_MW"], [1.5])
    chiller.step(0.0)
    assert chiller.get_value(["P_elec_MW"], 0.0)[0] > 0.2


# =============================================================================
# 8. Carbon Capture and Loads Tests
# =============================================================================

def test_carbon_capture_amine():
    cc = CarbonCapture_Amine(capture_rate_nom=0.90, step_size=1.0)
    cc.set_value(["m_dot_CO2_in_t_hr"], [100.0])
    cc.step(0.0)
    co2_cap, p_elec, q_steam = cc.get_value(
        ["m_dot_CO2_captured_t_hr", "P_elec_parasitic_MW", "Q_steam_reboiler_MW_th"], 0.0
    )
    assert co2_cap == pytest.approx(90.0)
    assert p_elec > 5.0
    assert q_steam > 50.0


def test_data_center_pue_thermal_surge():
    dc = DataCenterLoad(it_capacity_mw=50.0, base_pue=1.15, step_size=1.0)
    # Cool day 10 C
    dc.set_value(["P_it_cmd_MW", "T_amb"], [40.0, 10.0])
    dc.step(0.0)
    p_grid_cool = dc.get_value(["P_total_grid_MW"], 0.0)[0]

    # Hot summer day 40 C
    dc.set_value(["T_amb"], [40.0])
    dc.step(1.0)
    p_grid_hot = dc.get_value(["P_total_grid_MW"], 1.0)[0]

    assert p_grid_hot > p_grid_cool  # Higher cooling demand on hot days


# =============================================================================
# 9. Manufacturer Presets Tests
# =============================================================================

def test_manufacturer_presets_and_catalog():
    """Deprecated branded aliases still work (with a DeprecationWarning) and keep their numbers."""
    from energysim.models import catalog, archetypes
    with pytest.warns(DeprecationWarning):
        tesla = catalog.Tesla_Megapack_2XL(name="bess")
    assert tesla.capacity_mwh == pytest.approx(3.916)
    assert tesla.p_max_mw == pytest.approx(1.929)

    with pytest.warns(DeprecationWarning):
        nel = archetypes.Nel_A1000(name="ael")
    assert nel.p_nominal_mw == pytest.approx(2.2)

    with pytest.warns(DeprecationWarning):
        ge = presets.GE_7HA_02(name="ccgt")
    assert ge.p_rated_mw == pytest.approx(573.0)


# =============================================================================
# 10. End-to-End Multi-Energy Co-Simulation Integration Test
# =============================================================================

def test_full_btm_multi_energy_cosimulation(tmp_path):
    """
    Integrates Solar PV + BESS + PEM Electrolyzer + CCGT into an energysim World.
    Verifies time stepping, variable routing, and results collection.
    """
    res_file = str(tmp_path / "btm_cosim_res.h5")
    my_world = world(start_time=0, stop_time=300, t_macro=60, res_filename=res_file)

    # 1. Instantiate models
    pv = PhysicsPV(name="pv", p_dc_stc_mw=5.0, inverter_p_ac_max_mw=4.0)
    bess = BatteryBESS(name="bess", capacity_mwh=4.0, p_max_mw=2.0, soc_init=0.6)
    pem = PEMElectrolyzer(name="electrolyzer", p_nominal_mw=3.0)
    ccgt = CombinedCycle_CCGT(name="ccgt", p_rated_mw=10.0, min_stable_gen_pct=30.0)

    # 2. Add to world
    my_world.add_model("pv", model=pv, outputs=["P_ac"])
    my_world.add_model("bess", model=bess, outputs=["P_actual", "SoC", "T_batt"])
    my_world.add_model("electrolyzer", model=pem, outputs=["P_actual", "m_dot_H2_kg_hr"])
    my_world.add_model("ccgt", model=ccgt, outputs=["P_actual", "fuel_consumption_MMBtu_hr"])

    # 3. Add constant input signals
    my_world.add_signal("irr_signal", signal=lambda t: 900.0)
    my_world.add_signal("bess_cmd", signal=lambda t: 0.5)
    my_world.add_signal("h2_cmd", signal=lambda t: 2000.0)  # kW; PEM declares P_cmd in MW
    my_world.add_signal("ccgt_cmd", signal=lambda t: 4.0)

    # 4. Connections
    my_world.add_connections({
        "irr_signal.y": "pv.G_poa",
        "bess_cmd.y": "bess.P_cmd",
        "h2_cmd.y": "electrolyzer.P_cmd",
        "ccgt_cmd.y": "ccgt.P_cmd",
    })

    # 5. Simulate
    my_world.simulate(pbar=False)

    # 6. Verify Results
    results = my_world.results(to_csv=False)
    assert "pv" in results
    assert "bess" in results
    assert "electrolyzer" in results
    assert "ccgt" in results

    df_pv = results["pv"]
    df_bess = results["bess"]
    df_pem = results["electrolyzer"]

    # Check that PV produced power
    assert df_pv["P_ac"].iloc[-1] > 2.0
    # Check that BESS SoC advanced
    assert df_bess["SoC"].iloc[-1] > 0.6
    # Check that Hydrogen was produced
    assert df_pem["m_dot_H2_kg_hr"].iloc[-1] > 20.0

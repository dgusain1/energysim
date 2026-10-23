# -*- coding: utf-8 -*-
"""
Commercial Equipment Catalog and Archetypes for energysim.

Provides ready-to-run, calibrated configurations of real-world energy equipment
from major manufacturers (Tesla, Fluence, Nel, ITM Power, Bloom Energy, GE, Siemens, Wartsila, etc.).
"""

from __future__ import annotations
from typing import Any

from .storage import BatteryBESS, FlowBattery
from .renewables import PhysicsPV, WindTurbine
from .hydrogen import PEMElectrolyzer, AlkalineElectrolyzer, SOECElectrolyzer
from .thermal import GasTurbine_OCGT, CombinedCycle_CCGT, ReciprocatingGenset


def Tesla_Megapack_2XL(name: str = "Tesla_Megapack", **kwargs: Any) -> BatteryBESS:
    """Tesla Megapack 2XL (approx 1.93 MW inverter, 3.9 MWh LFP storage)."""
    return BatteryBESS(
        name=name,
        capacity_mwh=3.916,
        p_max_mw=1.929,
        soc_init=kwargs.pop("soc_init", 0.5),
        soc_min=0.05,
        soc_max=0.98,
        nominal_voltage_v=880.0,
        c_th_kj_k=150000.0,
        r_th_k_w=0.02,
        **kwargs,
    )


def Fluence_Gridstack(name: str = "Fluence_BESS", **kwargs: Any) -> BatteryBESS:
    """Fluence Gridstack 4-hour BESS (2.0 MW / 8.0 MWh)."""
    return BatteryBESS(
        name=name,
        capacity_mwh=8.0,
        p_max_mw=2.0,
        soc_init=kwargs.pop("soc_init", 0.5),
        soc_min=0.03,
        soc_max=0.97,
        nominal_voltage_v=1000.0,
        **kwargs,
    )


def Nel_A1000(name: str = "Nel_A1000", **kwargs: Any) -> AlkalineElectrolyzer:
    """Nel Hydrogen A-Series Atmospheric Alkaline Electrolyzer (approx 2.2 MW)."""
    return AlkalineElectrolyzer(
        name=name,
        p_nominal_mw=2.2,
        min_turndown_pct=20.0,
        spec_consumption_kwh_kg=52.0,
        **kwargs,
    )


def ITM_Power_Trident(name: str = "ITM_PEM", **kwargs: Any) -> PEMElectrolyzer:
    """ITM Power 5 MW PEM Electrolyzer stack."""
    return PEMElectrolyzer(
        name=name,
        p_nominal_mw=5.0,
        min_turndown=0.05,
        t_stack_nom_c=70.0,
        **kwargs,
    )


def Bloom_Energy_SOEC(name: str = "Bloom_SOEC", **kwargs: Any) -> SOECElectrolyzer:
    """Bloom Energy Solid Oxide Electrolyzer (1.0 MW)."""
    return SOECElectrolyzer(
        name=name,
        p_nominal_mw=1.0,
        operating_temp_c=750.0,
        spec_elec_kwh_kg=37.5,
        **kwargs,
    )


def GE_7HA_02(name: str = "GE_7HA_CCGT", **kwargs: Any) -> CombinedCycle_CCGT:
    """GE 7HA.02 Combined Cycle Power Plant (approx 573 MW net, ~64% net efficiency)."""
    return CombinedCycle_CCGT(
        name=name,
        p_rated_mw=573.0,
        gt_ratio=0.67,
        heat_rate_full_load_kj_kwh=5625.0,  # ~64% LHV efficiency
        min_stable_gen_pct=35.0,
        ramp_rate_mw_min=30.0,
        **kwargs,
    )


def Siemens_SGT_800(name: str = "Siemens_SGT800", **kwargs: Any) -> GasTurbine_OCGT:
    """Siemens Energy SGT-800 Industrial Gas Turbine (approx 62 MW ISO)."""
    return GasTurbine_OCGT(
        name=name,
        p_rated_mw=62.0,
        heat_rate_full_load_kj_kwh=9200.0,
        ramp_rate_mw_min=15.0,
        min_stable_gen_pct=30.0,
        cold_start_duration_sec=900.0,
        **kwargs,
    )


def Wartsila_34SG(name: str = "Wartsila_Genset", **kwargs: Any) -> ReciprocatingGenset:
    """Wartsila 20V34SG Natural Gas Reciprocating Engine (approx 9.7 MW)."""
    return ReciprocatingGenset(
        name=name,
        p_rated_mw=9.7,
        electrical_efficiency=0.47,
        heat_recovery_efficiency=0.41,
        **kwargs,
    )


def CanadianSolar_HiKu7(name: str = "SolarFarm_10MW", p_mw: float = 10.0, **kwargs: Any) -> PhysicsPV:
    """Utility solar farm based on Canadian Solar HiKu7 modules."""
    return PhysicsPV(
        name=name,
        p_dc_stc_mw=p_mw * 1.25,  # 1.25 DC/AC oversizing ratio
        inverter_p_ac_max_mw=p_mw,
        temp_coeff_p=-0.0034,
        noct_c=42.0,
        **kwargs,
    )


def Vestas_V150_4MW(name: str = "Vestas_V150", **kwargs: Any) -> WindTurbine:
    """Vestas V150-4.2 MW Wind Turbine."""
    return WindTurbine(
        name=name,
        p_rated_mw=4.2,
        cut_in_speed_ms=3.0,
        rated_speed_ms=11.5,
        cut_out_speed_ms=25.0,
        **kwargs,
    )

# -*- coding: utf-8 -*-
"""
Generic equipment archetypes for energysim.

Every factory in this module is an **illustrative archetype, not calibrated to an OEM
datasheet**: the parameters are round, plausible figures for the equipment class and carry
no manufacturer endorsement or data-sheet provenance. See ``PROVENANCE`` for the source
status of each archetype.

The manufacturer-branded names of earlier versions (``Tesla_Megapack_2XL``,
``Fluence_Gridstack``, ``Nel_A1000``, ``ITM_Power_Trident``, ``Bloom_Energy_SOEC``,
``GE_7HA_02``, ``Siemens_SGT_800``, ``Wartsila_34SG``, ``CanadianSolar_HiKu7``,
``Vestas_V150_4MW``) are kept as **deprecated aliases**: they emit a
``DeprecationWarning`` and return the generic archetype configured with the numbers the old
name used. They will be removed in a future release.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, Dict

from .storage import BatteryBESS
from .renewables import PhysicsPV, WindTurbine
from .hydrogen import PEMElectrolyzer, AlkalineElectrolyzer, SOECElectrolyzer
from .thermal import GasTurbine_OCGT, CombinedCycle_CCGT, ReciprocatingGenset

_NOTE = "Illustrative archetype, not calibrated to an OEM datasheet."

PROVENANCE: Dict[str, str] = {
    "BESS_2h_2MW": "Generic 2-hour Li-ion BESS (2 MW / 4 MWh). Round figures; no datasheet source.",
    "BESS_4h_2MW": "Generic 4-hour Li-ion BESS (2 MW / 8 MWh). Round figures; no datasheet source.",
    "Alkaline_2MW": "Generic atmospheric alkaline electrolyzer, 2 MW, ~52 kWh/kg system consumption "
                    "(typical literature range 50-55). Not an OEM datasheet.",
    "PEM_5MW": "Generic 5 MW PEM electrolyzer with an illustrative lookup polarization curve. "
               "Not fitted to any stack.",
    "SOEC_1MW": "Generic 1 MW solid-oxide electrolyzer, ~37.5 kWh/kg electricity (heat supplied "
                "externally). Not an OEM datasheet.",
    "CCGT_570MW": "Generic large-frame combined cycle, ~64 % LHV net at full load. "
                  "Round figures; no datasheet source.",
    "Gas_Turbine_60MW": "Generic industrial open-cycle gas turbine, ~39 % LHV. No datasheet source.",
    "Gas_Engine_10MW": "Generic lean-burn gas genset, 47 % electrical / 41 % heat recovery. "
                       "No datasheet source.",
    "Solar_Farm": "Generic utility PV plant, 1.25 DC/AC ratio, -0.34 %/K. No datasheet source.",
    "Wind_Turbine_4MW": "Generic 4 MW onshore turbine with a simple cubic power curve between "
                        "cut-in and rated wind speed. No datasheet source.",
    # deprecated brand aliases
    "Tesla_Megapack_2XL": "DEPRECATED alias of BESS_2h_2MW; keeps the old 1.929 MW / 3.916 MWh numbers "
                          "which are not backed by a cited datasheet.",
    "Fluence_Gridstack": "DEPRECATED alias of BESS_4h_2MW; old numbers not backed by a cited datasheet.",
    "Nel_A1000": "DEPRECATED alias of Alkaline_2MW; keeps the old 2.2 MW figure, which is suspect "
                 "(a 1000 Nm3/h machine would be ~4.5 MW) and unsourced.",
    "ITM_Power_Trident": "DEPRECATED alias of PEM_5MW; numbers unsourced.",
    "Bloom_Energy_SOEC": "DEPRECATED alias of SOEC_1MW; numbers unsourced.",
    "GE_7HA_02": "DEPRECATED alias of CCGT_570MW; keeps the old 573 MW figure, unsourced.",
    "Siemens_SGT_800": "DEPRECATED alias of Gas_Turbine_60MW; keeps the old 62 MW figure, unsourced.",
    "Wartsila_34SG": "DEPRECATED alias of Gas_Engine_10MW; keeps the old 9.7 MW figure, unsourced.",
    "CanadianSolar_HiKu7": "DEPRECATED alias of Solar_Farm; numbers unsourced.",
    "Vestas_V150_4MW": "DEPRECATED alias of Wind_Turbine_4MW; keeps the old 4.2 MW figure, unsourced.",
}


def _defaults(kwargs: Dict[str, Any], **defaults: Any) -> Dict[str, Any]:
    """Return ``defaults`` overridden by the user's ``kwargs``."""
    merged = dict(defaults)
    merged.update(kwargs)
    return merged


# ---------------------------------------------------------------------------
# Generic archetypes
# ---------------------------------------------------------------------------

def BESS_2h_2MW(name: str = "BESS_2h", **kwargs: Any) -> BatteryBESS:
    """2 MW / 4 MWh (2-hour) Li-ion battery archetype. Illustrative archetype, not calibrated to an OEM datasheet."""
    return BatteryBESS(name=name, **_defaults(
        kwargs, capacity_mwh=4.0, p_max_mw=2.0, soc_init=0.5, soc_min=0.05, soc_max=0.95,
        nominal_voltage_v=800.0))


def BESS_4h_2MW(name: str = "BESS_4h", **kwargs: Any) -> BatteryBESS:
    """2 MW / 8 MWh (4-hour) Li-ion battery archetype. Illustrative archetype, not calibrated to an OEM datasheet."""
    return BatteryBESS(name=name, **_defaults(
        kwargs, capacity_mwh=8.0, p_max_mw=2.0, soc_init=0.5, soc_min=0.03, soc_max=0.97,
        nominal_voltage_v=1000.0))


def Alkaline_2MW(name: str = "Alkaline_2MW", **kwargs: Any) -> AlkalineElectrolyzer:
    """2 MW alkaline electrolyzer archetype. Illustrative archetype, not calibrated to an OEM datasheet."""
    return AlkalineElectrolyzer(name=name, **_defaults(
        kwargs, p_nominal_mw=2.0, min_turndown_pct=20.0, spec_consumption_kwh_kg=52.0))


def PEM_5MW(name: str = "PEM_5MW", **kwargs: Any) -> PEMElectrolyzer:
    """5 MW PEM electrolyzer archetype. Illustrative archetype, not calibrated to an OEM datasheet."""
    return PEMElectrolyzer(name=name, **_defaults(
        kwargs, p_nominal_mw=5.0, min_turndown=0.05, t_stack_nom_c=70.0))


def SOEC_1MW(name: str = "SOEC_1MW", **kwargs: Any) -> SOECElectrolyzer:
    """1 MW solid-oxide electrolyzer archetype. Illustrative archetype, not calibrated to an OEM datasheet."""
    return SOECElectrolyzer(name=name, **_defaults(
        kwargs, p_nominal_mw=1.0, operating_temp_c=750.0, spec_elec_kwh_kg=37.5))


def CCGT_570MW(name: str = "CCGT_570MW", **kwargs: Any) -> CombinedCycle_CCGT:
    """~570 MW combined-cycle plant archetype (~64 % LHV). Illustrative archetype, not calibrated to an OEM datasheet."""
    return CombinedCycle_CCGT(name=name, **_defaults(
        kwargs, p_rated_mw=570.0, gt_ratio=0.67, heat_rate_full_load_kj_kwh=5625.0,
        min_stable_gen_pct=35.0, ramp_rate_mw_min=30.0))


def Gas_Turbine_60MW(name: str = "GasTurbine_60MW", **kwargs: Any) -> GasTurbine_OCGT:
    """~60 MW industrial open-cycle gas turbine archetype. Illustrative archetype, not calibrated to an OEM datasheet."""
    return GasTurbine_OCGT(name=name, **_defaults(
        kwargs, p_rated_mw=60.0, heat_rate_full_load_kj_kwh=9200.0, ramp_rate_mw_min=15.0,
        min_stable_gen_pct=30.0, cold_start_duration_sec=900.0))


def Gas_Engine_10MW(name: str = "GasEngine_10MW", **kwargs: Any) -> ReciprocatingGenset:
    """~10 MW natural-gas reciprocating engine archetype. Illustrative archetype, not calibrated to an OEM datasheet."""
    return ReciprocatingGenset(name=name, **_defaults(
        kwargs, p_rated_mw=10.0, electrical_efficiency=0.47, heat_recovery_efficiency=0.41))


def Solar_Farm(name: str = "SolarFarm_10MW", p_mw: float = 10.0, **kwargs: Any) -> PhysicsPV:
    """Utility PV plant archetype with ``p_mw`` AC export. Illustrative archetype, not calibrated to an OEM datasheet."""
    return PhysicsPV(name=name, **_defaults(
        kwargs, p_dc_stc_mw=p_mw * 1.25, inverter_p_ac_max_mw=p_mw, temp_coeff_p=-0.0034, noct_c=42.0))


def Wind_Turbine_4MW(name: str = "WindTurbine_4MW", **kwargs: Any) -> WindTurbine:
    """~4 MW onshore wind turbine archetype. Illustrative archetype, not calibrated to an OEM datasheet."""
    return WindTurbine(name=name, **_defaults(
        kwargs, p_rated_mw=4.0, cut_in_speed_ms=3.0, rated_speed_ms=11.5, cut_out_speed_ms=25.0))


# ---------------------------------------------------------------------------
# Deprecated manufacturer-branded aliases
# ---------------------------------------------------------------------------

def _deprecated_alias(old: str, new: Callable[..., Any], doc: str, **old_numbers: Any):
    """Build a deprecated factory ``old`` that calls archetype ``new`` with the old numbers."""

    def factory(name: str = None, **kwargs: Any):
        warnings.warn(
            f"{old} is deprecated: manufacturer-branded presets are not backed by datasheets. "
            f"Use the generic archetype {new.__name__} instead.",
            DeprecationWarning, stacklevel=2,
        )
        merged = _defaults(kwargs, **old_numbers)
        if name is not None:
            merged["name"] = name
        return new(**merged)

    factory.__name__ = old
    factory.__qualname__ = old
    factory.__doc__ = f"Deprecated alias of :func:`{new.__name__}` ({doc}). {_NOTE}"
    return factory


Tesla_Megapack_2XL = _deprecated_alias(
    "Tesla_Megapack_2XL", BESS_2h_2MW, "keeps old 1.929 MW / 3.916 MWh",
    name="Tesla_Megapack", capacity_mwh=3.916, p_max_mw=1.929, soc_max=0.98,
    nominal_voltage_v=880.0, c_th_kj_k=150000.0, r_th_k_w=0.02)
Fluence_Gridstack = _deprecated_alias(
    "Fluence_Gridstack", BESS_4h_2MW, "keeps old 2 MW / 8 MWh", name="Fluence_BESS")
Nel_A1000 = _deprecated_alias(
    "Nel_A1000", Alkaline_2MW, "keeps the old, suspect 2.2 MW figure",
    name="Nel_A1000", p_nominal_mw=2.2)
ITM_Power_Trident = _deprecated_alias(
    "ITM_Power_Trident", PEM_5MW, "keeps old numbers", name="ITM_PEM")
Bloom_Energy_SOEC = _deprecated_alias(
    "Bloom_Energy_SOEC", SOEC_1MW, "keeps old numbers", name="Bloom_SOEC")
GE_7HA_02 = _deprecated_alias(
    "GE_7HA_02", CCGT_570MW, "keeps old 573 MW", name="GE_7HA_CCGT", p_rated_mw=573.0)
Siemens_SGT_800 = _deprecated_alias(
    "Siemens_SGT_800", Gas_Turbine_60MW, "keeps old 62 MW", name="Siemens_SGT800", p_rated_mw=62.0)
Wartsila_34SG = _deprecated_alias(
    "Wartsila_34SG", Gas_Engine_10MW, "keeps old 9.7 MW", name="Wartsila_Genset", p_rated_mw=9.7)
CanadianSolar_HiKu7 = _deprecated_alias(
    "CanadianSolar_HiKu7", Solar_Farm, "keeps old numbers", name="SolarFarm_10MW")
Vestas_V150_4MW = _deprecated_alias(
    "Vestas_V150_4MW", Wind_Turbine_4MW, "keeps old 4.2 MW", name="Vestas_V150", p_rated_mw=4.2)

# Make the aliases resolve to the right module in introspection tools.
for _f in (Tesla_Megapack_2XL, Fluence_Gridstack, Nel_A1000, ITM_Power_Trident, Bloom_Energy_SOEC,
           GE_7HA_02, Siemens_SGT_800, Wartsila_34SG, CanadianSolar_HiKu7, Vestas_V150_4MW):
    _f.__module__ = __name__

__all__ = [
    "PROVENANCE",
    "BESS_2h_2MW", "BESS_4h_2MW", "Alkaline_2MW", "PEM_5MW", "SOEC_1MW", "CCGT_570MW",
    "Gas_Turbine_60MW", "Gas_Engine_10MW", "Solar_Farm", "Wind_Turbine_4MW",
    # deprecated aliases
    "Tesla_Megapack_2XL", "Fluence_Gridstack", "Nel_A1000", "ITM_Power_Trident",
    "Bloom_Energy_SOEC", "GE_7HA_02", "Siemens_SGT_800", "Wartsila_34SG",
    "CanadianSolar_HiKu7", "Vestas_V150_4MW",
]

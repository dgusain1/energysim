# -*- coding: utf-8 -*-
"""
Explicit, visible unit system for energysim.

This module is the single source of truth for physical units:

* a registry of **quantities** (``active_power``, ``temperature``, ...), each with
  a dimension, a canonical (default) unit, a carrier tag and a description;
* a registry of **units** with exact scale factors (and offsets for degC/degF);
* :class:`UnitSystem`, a mapping ``quantity -> default unit``.  The energysim
  standard is :meth:`UnitSystem.default`; :meth:`UnitSystem.si` and
  :meth:`UnitSystem.from_overrides` give alternatives;
* name based helpers (:func:`infer_quantity`, :func:`resolve_unit`) used where a
  unit is not declared explicitly.

Policy: a variable whose unit is neither declared nor derivable from a
conventional name suffix is ``unspecified`` and is never silently scaled.

Semantics are a superset of the earlier ``energysim.utils`` unit helpers:
lookup is case-sensitive first (``MV`` != ``mV``), then case-insensitive only
if unambiguous; unknown units raise ``ValueError``; units of different
dimensions raise ``ValueError`` ("Incompatible unit conversion"); affine units
are handled through ``(scale, offset)`` with ``to = from * scale + offset``.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

__all__ = [
    'UnitDef', 'Quantity', 'UnitSystem', 'QUANTITIES', 'KNOWN_UNITS', 'CARRIERS',
    'parse_unit', 'is_known_unit', 'get_unit_conversion', 'get_unit_scale',
    'quantity_of', 'accepted_units', 'conversion_rows',
    'infer_quantity', 'infer_unit_from_name', 'resolve_unit', 'default_system',
]

CARRIERS = ('electric', 'thermal', 'gas', 'hydrogen', 'co2', 'generic')


class UnitDef(NamedTuple):
    """``value_in_base = value * factor + offset`` within one dimension."""
    dimension: str
    factor: float
    offset: float = 0.0


@dataclass(frozen=True)
class Quantity:
    """A physical quantity with its energysim default (canonical) unit."""
    name: str
    dimension: str
    canonical: str
    carrier: str = 'generic'
    description: str = ''
    #: unit spellings resolved *only* in the context of this quantity
    #: (e.g. ``K`` as a temperature *difference*): spelling -> UnitDef
    local_units: Dict[str, UnitDef] = field(default_factory=dict, compare=False, hash=False)


# ---------------------------------------------------------------------------
# Unit registry
# ---------------------------------------------------------------------------

class _Rec(NamedTuple):
    dim: str
    factor: float
    offset: float
    primary: str
    hint: Optional[str]      # quantity this unit spelling belongs to by preference


_UNITS: Dict[str, _Rec] = {}          # every accepted spelling (raw, case-sensitive)
_PRIMARY: Dict[str, List[str]] = {}   # dimension -> primary spellings (registration order)


def _u(dim: str, factor: float, primary: str, *aliases: str, offset: float = 0.0,
       quantity: Optional[str] = None) -> None:
    rec = _Rec(dim, float(factor), float(offset), primary, quantity)
    for n in (primary,) + aliases:
        _UNITS[n] = rec
    _PRIMARY.setdefault(dim, []).append(primary)


# ---- electrical ----
_u('power', 1.0, 'W', 'watt', 'watts', 'W_el')
_u('power', 1e3, 'kW', 'kilowatt', 'kilowatts', 'kW_el', 'kWe', 'kWel')
_u('power', 1e6, 'MW', 'megawatt', 'megawatts', 'MW_el', 'MWe', 'MWel')
_u('power', 1e9, 'GW', 'gigawatt', 'gigawatts', 'GW_el', 'GWe')
_u('power', 1.0, 'W_th', 'Wth', quantity='thermal_power')
_u('power', 1e3, 'kW_th', 'kWth', 'kWt', quantity='thermal_power')
_u('power', 1e6, 'MW_th', 'MWth', 'MWt', quantity='thermal_power')
_u('power', 1e9, 'GW_th', 'GWth', quantity='thermal_power')
_u('reactive_power', 1.0, 'var', 'vars')
_u('reactive_power', 1e3, 'kvar', 'kvars')
_u('reactive_power', 1e6, 'Mvar', 'mvars')
_u('apparent_power', 1.0, 'VA')
_u('apparent_power', 1e3, 'kVA')
_u('apparent_power', 1e6, 'MVA')
_u('voltage', 1e-3, 'mV', 'millivolt', 'millivolts')
_u('voltage', 1.0, 'V', 'volt', 'volts')
_u('voltage', 1e3, 'kV', 'kilovolt', 'kilovolts')
_u('voltage', 1e6, 'MV', 'megavolt', 'megavolts')
_u('current', 1e-3, 'mA')
_u('current', 1.0, 'A', 'amp', 'amps')
_u('current', 1e3, 'kA')
_u('energy', 1.0, 'J', 'joule', 'joules')
_u('energy', 1e3, 'kJ')
_u('energy', 1e6, 'MJ')
_u('energy', 1e9, 'GJ')
_u('energy', 3.6e3, 'Wh')
_u('energy', 3.6e6, 'kWh')
_u('energy', 3.6e9, 'MWh')
_u('energy', 3.6e12, 'GWh')
_u('energy', 3.6e3, 'Wh_th', 'Whth', quantity='thermal_energy')
_u('energy', 3.6e6, 'kWh_th', 'kWhth', quantity='thermal_energy')
_u('energy', 3.6e9, 'MWh_th', 'MWhth', quantity='thermal_energy')
_u('energy', 3.6e12, 'GWh_th', 'GWhth', quantity='thermal_energy')
_u('energy', 3.6e3, 'Wh_LHV', quantity='hydrogen_energy')
_u('energy', 3.6e6, 'kWh_LHV', quantity='hydrogen_energy')
_u('energy', 3.6e9, 'MWh_LHV', quantity='hydrogen_energy')
_u('energy', 3.6e12, 'GWh_LHV', quantity='hydrogen_energy')
_u('reactive_energy', 1.0, 'varh')
_u('reactive_energy', 1e3, 'kvarh')
_u('reactive_energy', 1e6, 'Mvarh')
_u('frequency', 1.0, 'Hz')
_u('frequency', 1e3, 'kHz')
_u('impedance', 1e-3, 'mohm', 'mΩ')
_u('impedance', 1.0, 'ohm', 'Ohm', 'ohms', 'Ω')
_u('impedance', 1e3, 'kohm', 'kΩ')
_u('impedance', 1e6, 'Mohm', 'MΩ')
_u('angle', 1.0, 'rad', 'radian', 'radians')
_u('angle', 0.017453292519943295, 'deg', '°', 'degree', 'degrees')
# dimensionless fractions (per-unit is *not* convertible to volts)
_u('fraction', 1.0, '1', '-', 'fraction')
_u('fraction', 1.0, 'pu', 'p.u.')
_u('fraction', 0.01, '%', 'percent')

# ---- thermal ----
_u('temperature', 1.0, 'K', 'kelvin')
_u('temperature', 1.0, 'degC', '°C', 'C', 'celsius', offset=273.15)
_u('temperature', 5.0 / 9.0, 'degF', '°F', 'F', 'fahrenheit', offset=459.67 * 5.0 / 9.0)
_u('temperature_difference', 1.0, 'delta_K', 'dK', 'K_diff', 'delta_degC', 'dgC')
_u('temperature_difference', 5.0 / 9.0, 'delta_degF', 'dF')
_u('specific_heat', 1.0, 'J/(kg*K)', 'J/kg/K', 'J/kgK')
_u('specific_heat', 1e3, 'kJ/(kg*K)', 'kJ/kg/K', 'kJ/kgK', 'kJ/(kg K)', 'kJ/(kg.K)')
_u('heat_transfer_coefficient', 1.0, 'W/(m2*K)', 'W/m2/K', 'W/m2K', 'W/(m^2*K)', 'W/(m²*K)', 'W/(m2 K)')
_u('heat_transfer_coefficient', 1e3, 'kW/(m2*K)', 'kW/m2/K', 'kW/m2K', 'kW/(m^2*K)')
_u('thermal_resistance', 1.0, 'K/W')
_u('thermal_resistance', 1e-3, 'K/kW')
_u('thermal_resistance', 1e-6, 'K/MW')
_u('thermal_capacitance', 1.0, 'J/K')
_u('thermal_capacitance', 1e3, 'kJ/K')
_u('thermal_capacitance', 1e6, 'MJ/K')
_u('thermal_capacitance', 3.6e3, 'Wh/K')
_u('thermal_capacitance', 3.6e6, 'kWh/K')
_u('thermal_capacitance', 3.6e9, 'MWh/K')

# ---- mass / gas / hydrogen ----
_u('mass', 1e-3, 'g')
_u('mass', 1.0, 'kg')
_u('mass', 1e3, 't', 'tonne', 'tonnes')
_u('mass', 1e6, 'kt')
_u('mass', 1e-3, 'g_CO2', 'gCO2', quantity='co2_mass')
_u('mass', 1.0, 'kg_CO2', 'kgCO2', quantity='co2_mass')
_u('mass', 1e3, 't_CO2', 'tCO2', quantity='co2_mass')
_u('mass', 1e6, 'kt_CO2', 'ktCO2', quantity='co2_mass')
_u('mass_flow', 1e-3, 'g/s')
_u('mass_flow', 1.0, 'kg/s')
_u('mass_flow', 1.0 / 60.0, 'kg/min')
_u('mass_flow', 1.0 / 3600.0, 'kg/h', 'kg/hr')
_u('mass_flow', 1000.0 / 3600.0, 't/h', 't/hr')
_u('molar_flow', 1e-3, 'mmol/s')
_u('molar_flow', 1.0, 'mol/s')
_u('molar_flow', 1.0 / 3600.0, 'mol/h')
_u('molar_flow', 1e3, 'kmol/s')
_u('molar_flow', 1e3 / 3600.0, 'kmol/h')
_u('volume', 1e-6, 'mL', 'cm3', 'cm³', 'cm^3')
_u('volume', 1e-3, 'L', 'l', 'liter', 'litre', 'dm3')
_u('volume', 1.0, 'm3', 'm³', 'm^3')
_u('normal_volume', 1.0, 'Nm3', 'Nm³', 'Nm^3')
# actual and normal volume flow are separate dimensions: no implicit conversion
_u('volume_flow', 1.0, 'm3/s', 'm³/s', 'm^3/s')
_u('volume_flow', 1.0 / 60.0, 'm3/min', 'm³/min')
_u('volume_flow', 1.0 / 3600.0, 'm3/h', 'm³/h', 'm^3/h', 'm3/hr')
_u('volume_flow', 1e-3, 'L/s')
_u('volume_flow', 1e-3 / 60.0, 'L/min')
_u('volume_flow', 1e-3 / 3600.0, 'L/h')
_u('normal_volume_flow', 1.0, 'Nm3/s', 'Nm³/s', 'Nm^3/s')
_u('normal_volume_flow', 1.0 / 60.0, 'Nm3/min', 'Nm³/min')
_u('normal_volume_flow', 1.0 / 3600.0, 'Nm3/h', 'Nm³/h', 'Nm^3/h', 'Nm3/hr')
_u('pressure', 1.0, 'Pa')
_u('pressure', 1e2, 'mbar', 'hPa')
_u('pressure', 1e3, 'kPa')
_u('pressure', 1e5, 'bar')
_u('pressure', 1e6, 'MPa')
_u('pressure', 6894.757293168, 'psi')
_u('density', 1.0, 'kg/m3', 'kg/m³', 'kg/m^3', 'g/L')
_u('density', 1e3, 'g/cm3', 'g/cm³')
_u('specific_energy', 1.0, 'J/kg')
_u('specific_energy', 1e3, 'kJ/kg')
_u('specific_energy', 1e6, 'MJ/kg')
_u('specific_energy', 3.6e3, 'Wh/kg')
_u('specific_energy', 3.6e6, 'kWh/kg', 'MWh/t')

# ---- geometry ----
_u('length', 1e-3, 'mm')
_u('length', 1e-2, 'cm')
_u('length', 1.0, 'm', 'meter', 'metre')
_u('length', 1e3, 'km')
_u('area', 1e-6, 'mm2', 'mm²', 'mm^2')
_u('area', 1e-4, 'cm2', 'cm²', 'cm^2', quantity='cell_area')
_u('area', 1.0, 'm2', 'm²', 'm^2')
_u('current_density', 1e1, 'mA/cm2', 'mA/cm²', 'mA/cm^2')
_u('current_density', 1e4, 'A/cm2', 'A/cm²', 'A/cm^2')
_u('current_density', 1.0, 'A/m2', 'A/m²', 'A/m^2')
_u('current_density', 1e3, 'kA/m2', 'kA/m²')

# ---- time / rates ----
_u('time', 1e-3, 'ms')
_u('time', 1.0, 's', 'sec')
_u('time', 60.0, 'min')
_u('time', 3600.0, 'h', 'hr', 'hour', 'hours')
_u('time', 86400.0, 'd', 'day', 'days')
_u('power_rate', 1.0, 'W/s')
_u('power_rate', 1e3, 'kW/s')
_u('power_rate', 1e6, 'MW/s')
_u('power_rate', 1e3 / 60.0, 'kW/min')
_u('power_rate', 1e6 / 60.0, 'MW/min')
_u('power_rate', 1e3 / 3600.0, 'kW/h')
_u('power_rate', 1e6 / 3600.0, 'MW/h')
_u('fraction_rate', 1.0, '1/s', 'pu/s')
_u('fraction_rate', 1.0 / 60.0, '1/min', 'pu/min')
_u('fraction_rate', 1.0 / 3600.0, '1/h', 'pu/h')
_u('fraction_rate', 0.01, '%/s')
_u('fraction_rate', 0.01 / 60.0, '%/min')
_u('fraction_rate', 0.01 / 3600.0, '%/h')

# ---- economics: currencies are different dimensions, never converted ----
_u('price_eur', 1.0, 'EUR/kWh', '€/kWh')
_u('price_eur', 1e-3, 'EUR/MWh', '€/MWh')
_u('price_eur', 1e-2, 'ct/kWh', 'cEUR/kWh')
_u('price_eur', 3.6e-3, 'EUR/GJ')
_u('price_eur', 3.6e6, 'EUR/J')
_u('price_usd', 1.0, 'USD/kWh', '$/kWh')
_u('price_usd', 1e-3, 'USD/MWh', '$/MWh')
_u('price_usd', 1e-2, 'USDct/kWh', 'cUSD/kWh')
_u('cost_eur', 1.0, 'EUR', '€')
_u('cost_eur', 1e3, 'kEUR')
_u('cost_eur', 1e6, 'MEUR')
_u('cost_usd', 1.0, 'USD', '$')
_u('cost_usd', 1e3, 'kUSD')
_u('cost_usd', 1e6, 'MUSD')
_u('emission_factor', 1.0, 'kg/kWh', 'kgCO2/kWh', 't/MWh', 'tCO2/MWh')
_u('emission_factor', 1e-3, 'g/kWh', 'gCO2/kWh', 'kg/MWh')
_u('emission_factor', 3.6e6, 'kg/J')

KNOWN_UNITS: Tuple[str, ...] = tuple(_UNITS)


def _norm(s: str) -> str:
    """Spelling-insensitive key: drops spaces, '*', '.', '^', '_' and maps superscripts."""
    for a, b in (('³', '3'), ('²', '2'), ('·', ''), ('⋅', ''), ('^', ''), ('*', ''),
                 ('.', ''), ('_', ''), (' ', '')):
        s = s.replace(a, b)
    return s


_NORM: Dict[str, List[str]] = {}
_LOWER: Dict[str, List[str]] = {}
for _n in _UNITS:
    _NORM.setdefault(_norm(_n), []).append(_n)
    _LOWER.setdefault(_norm(_n).lower(), []).append(_n)


def _rec_def(rec: _Rec) -> UnitDef:
    return UnitDef(rec.dim, rec.factor, rec.offset)


def _close(unit: str) -> List[str]:
    pool = list(_UNITS) + [k for k in _NORM]
    m = difflib.get_close_matches(unit, pool, n=5, cutoff=0.6)
    m += difflib.get_close_matches(unit.lower(), [p.lower() for p in pool], n=3, cutoff=0.7)
    out: List[str] = []
    for x in m:
        for s in _NORM.get(x, [x]) if x in _NORM and x not in _UNITS else [x]:
            if s in _UNITS and s not in out:
                out.append(s)
        for s in _LOWER.get(x.lower(), []):
            if s not in out and len(out) < 6:
                out.append(s)
    return out[:6]


def _lookup(unit: str, quantity: Optional['Quantity'] = None) -> _Rec:
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError(f"Invalid unit {unit!r}.")
    u = unit.strip()
    if quantity is not None:
        loc = quantity.local_units.get(u)
        if loc is not None:
            return _Rec(loc.dimension, loc.factor, loc.offset, u, quantity.name)
    if u in _UNITS:
        return _UNITS[u]
    n = _norm(u)
    for table, key in ((_NORM, n), (_LOWER, n.lower())):
        cands = table.get(key, [])
        defs = {_rec_def(_UNITS[c]) for c in cands}
        if len(defs) == 1:
            return _UNITS[cands[0]]
        if len(defs) > 1:
            raise ValueError(
                f"Ambiguous unit '{unit}': could be any of {sorted(cands)}. Use the exact spelling.")
    close = _close(u)
    hint = f" Did you mean: {', '.join(repr(c) for c in close)}?" if close else ''
    raise ValueError(
        f"Unknown unit '{unit}'.{hint} Known units: {', '.join(KNOWN_UNITS)}")


def parse_unit(unit: str) -> UnitDef:
    """Return the :class:`UnitDef` for *unit*; raise ``ValueError`` if unknown or ambiguous."""
    return _rec_def(_lookup(unit))


def is_known_unit(unit: str) -> bool:
    """True if *unit* is a recognised (unambiguous) unit string."""
    try:
        _lookup(unit)
        return True
    except ValueError:
        return False


def get_unit_conversion(from_unit: Optional[str], to_unit: Optional[str],
                        quantity: Optional[str] = None) -> Tuple[float, float]:
    """``(scale, offset)`` with ``value_to = value_from * scale + offset``.

    ``(1.0, 0.0)`` if either unit is ``None``/empty.  Unknown units and
    incompatible dimensions raise ``ValueError``.  *quantity* (a quantity name)
    resolves quantity-local spellings such as ``K`` for temperature differences.
    """
    if not from_unit or not to_unit:
        return 1.0, 0.0
    q = QUANTITIES[quantity] if quantity else None
    a, b = _lookup(from_unit, q), _lookup(to_unit, q)
    if a.dim != b.dim:
        extra = ''
        if {a.dim, b.dim} == {'volume_flow', 'normal_volume_flow'}:
            extra = (" Actual volume flow (m3/h) and normal volume flow (Nm3/h) differ by "
                     "pressure/temperature state; no implicit conversion.")
        raise ValueError(
            f"Incompatible unit conversion: cannot convert '{from_unit}' ({a.dim}) "
            f"to '{to_unit}' ({b.dim}).{extra}")
    return a.factor / b.factor, (a.offset - b.offset) / b.factor


def get_unit_scale(from_unit: Optional[str], to_unit: Optional[str],
                   quantity: Optional[str] = None) -> float:
    """Linear factor from *from_unit* to *to_unit* (1.0 if unspecified); raises for affine pairs."""
    scale, offset = get_unit_conversion(from_unit, to_unit, quantity)
    if offset != 0.0:
        raise ValueError(
            f"Conversion '{from_unit}' -> '{to_unit}' needs an offset; use get_unit_conversion().")
    return scale


# ---------------------------------------------------------------------------
# Quantity registry (declaration order = table order)
# ---------------------------------------------------------------------------

QUANTITIES: Dict[str, Quantity] = {}


def _q(name: str, dim: str, canonical: str, carrier: str, desc: str,
       local: Optional[Dict[str, UnitDef]] = None) -> None:
    assert carrier in CARRIERS, carrier
    QUANTITIES[name] = Quantity(name, dim, canonical, carrier, desc, dict(local or {}))


_DK = UnitDef('temperature_difference', 1.0)
_q('fraction', 'fraction', '1', 'generic', 'Dimensionless fraction / per-unit value')
_q('active_power', 'power', 'kW', 'electric', 'Active (real) electric power')
_q('reactive_power', 'reactive_power', 'kvar', 'electric', 'Reactive power')
_q('apparent_power', 'apparent_power', 'kVA', 'electric', 'Apparent power')
_q('voltage', 'voltage', 'kV', 'electric', 'Voltage magnitude (physical)')
_q('current', 'current', 'A', 'electric', 'Electric current')
_q('energy', 'energy', 'kWh', 'electric', 'Electric (active) energy')
_q('reactive_energy', 'reactive_energy', 'kvarh', 'electric', 'Reactive energy')
_q('frequency', 'frequency', 'Hz', 'electric', 'Electrical frequency')
_q('impedance', 'impedance', 'ohm', 'electric', 'Impedance')
_q('resistance', 'impedance', 'ohm', 'electric', 'Resistance')
_q('voltage_angle', 'angle', 'deg', 'electric', 'Voltage angle')
_q('power_factor', 'fraction', '1', 'electric', 'Power factor')
_q('voltage_pu', 'fraction', 'pu', 'electric', 'Voltage magnitude in per-unit (never converted to volts)')
_q('efficiency', 'fraction', '1', 'generic', 'Efficiency (0..1)')
_q('state_of_charge', 'fraction', '1', 'generic', 'State of charge (0..1)')
_q('thermal_power', 'power', 'kW', 'thermal', 'Thermal (heat/cooling) power')
_q('thermal_energy', 'energy', 'kWh_th', 'thermal', 'Thermal energy')
_q('temperature', 'temperature', 'degC', 'thermal', 'Absolute temperature (K accepted via affine conversion)')
_q('temperature_difference', 'temperature_difference', 'K', 'thermal', 'Temperature difference (no offset)',
   {'K': _DK, 'degC': _DK, 'kelvin': _DK, 'degF': UnitDef('temperature_difference', 5.0 / 9.0)})
_q('specific_heat', 'specific_heat', 'kJ/(kg*K)', 'thermal', 'Specific heat capacity')
_q('heat_transfer_coefficient', 'heat_transfer_coefficient', 'W/(m2*K)', 'thermal', 'Heat transfer coefficient (U/h)')
_q('thermal_resistance', 'thermal_resistance', 'K/kW', 'thermal', 'Thermal resistance')
_q('thermal_capacitance', 'thermal_capacitance', 'kWh/K', 'thermal', 'Thermal capacitance')
_q('mass', 'mass', 'kg', 'generic', 'Mass')
_q('mass_flow', 'mass_flow', 'kg/h', 'gas', 'Mass flow rate')
_q('hydrogen_energy', 'energy', 'kWh', 'hydrogen', 'Hydrogen energy (lower heating value basis)')
_q('volume', 'volume', 'm3', 'gas', 'Actual volume')
_q('normal_volume', 'normal_volume', 'Nm3', 'gas', 'Normal volume')
_q('normal_volume_flow', 'normal_volume_flow', 'Nm3/h', 'gas', 'Normal volume flow')
_q('actual_volume_flow', 'volume_flow', 'm3/h', 'gas', 'Actual volume flow (no implicit conversion to normal volume flow)')
_q('pressure', 'pressure', 'bar', 'gas', 'Pressure')
_q('density', 'density', 'kg/m3', 'gas', 'Density')
_q('specific_energy', 'specific_energy', 'kWh/kg', 'hydrogen', 'Specific energy')
_q('molar_flow', 'molar_flow', 'mol/s', 'hydrogen', 'Molar flow rate')
_q('length', 'length', 'm', 'generic', 'Length')
_q('area', 'area', 'm2', 'generic', 'Area')
_q('cell_area', 'area', 'cm2', 'hydrogen', 'Cell / electrode active area')
_q('current_density', 'current_density', 'A/cm2', 'hydrogen', 'Electrode current density')
_q('time', 'time', 's', 'generic', 'Time / duration')
_q('ramp_rate', 'power_rate', 'kW/min', 'electric', 'Power ramp rate')
_q('fraction_rate', 'fraction_rate', '1/min', 'generic', 'Fractional rate of change (e.g. %/min)')
_q('energy_price', 'price_eur', 'EUR/kWh', 'generic', 'Energy price in EUR (currencies are never converted)')
_q('energy_price_usd', 'price_usd', 'USD/kWh', 'generic', 'Energy price in USD (separate from EUR)')
_q('cost', 'cost_eur', 'EUR', 'generic', 'Cost in EUR')
_q('cost_usd', 'cost_usd', 'USD', 'generic', 'Cost in USD')
_q('co2_mass', 'mass', 'kg', 'co2', 'Emitted CO2 mass')
_q('emission_factor', 'emission_factor', 'kg/kWh', 'co2', 'CO2 emission factor')

_DIM_FIRST: Dict[str, str] = {}
for _qn, _q_ in QUANTITIES.items():
    _DIM_FIRST.setdefault(_q_.dimension, _qn)


def _validate_registry() -> None:
    for q in QUANTITIES.values():
        r = _lookup(q.canonical, q)
        assert r.dim == q.dimension, (q.name, q.canonical, r.dim)


_validate_registry()


def quantity_of(unit: str) -> str:
    """Quantity a unit spelling belongs to (e.g. ``kW_th`` -> ``thermal_power``). Raises if unknown."""
    rec = _lookup(unit)
    return rec.hint or _DIM_FIRST[rec.dim]


def accepted_units(quantity: str) -> List[str]:
    """Primary unit spellings accepted for *quantity* (own-tagged first)."""
    q = QUANTITIES[quantity]
    if q.local_units:
        return [u for u in q.local_units] + [u for u in _PRIMARY.get(q.dimension, []) if u not in q.local_units]
    own = [u for u in _PRIMARY.get(q.dimension, []) if _UNITS[u].hint == quantity]
    shared = [u for u in _PRIMARY.get(q.dimension, []) if _UNITS[u].hint is None]
    return own + shared


def conversion_rows(system: Optional['UnitSystem'] = None) -> List[Dict[str, Any]]:
    """Rows (quantity, unit, factor, offset) converting each accepted unit to the system default."""
    system = system or default_system()
    rows = []
    for qn in QUANTITIES:
        d = system.default_unit(qn)
        for u in accepted_units(qn):
            sc, off = get_unit_conversion(u, d, qn)
            rows.append({'quantity': qn, 'unit': u, 'default_unit': d, 'factor': sc, 'offset': off})
    return rows


# ---------------------------------------------------------------------------
# UnitSystem
# ---------------------------------------------------------------------------

_SI_DEFAULTS = {
    'active_power': 'W', 'reactive_power': 'var', 'apparent_power': 'VA', 'voltage': 'V',
    'current': 'A', 'energy': 'J', 'reactive_energy': 'varh', 'frequency': 'Hz',
    'impedance': 'ohm', 'resistance': 'ohm', 'voltage_angle': 'rad',
    'thermal_power': 'W', 'thermal_energy': 'J', 'temperature': 'K', 'temperature_difference': 'K',
    'specific_heat': 'J/(kg*K)', 'thermal_resistance': 'K/W', 'thermal_capacitance': 'J/K',
    'mass_flow': 'kg/s', 'hydrogen_energy': 'J', 'normal_volume_flow': 'Nm3/s',
    'actual_volume_flow': 'm3/s', 'pressure': 'Pa', 'specific_energy': 'J/kg',
    'cell_area': 'm2', 'current_density': 'A/m2', 'ramp_rate': 'W/s', 'fraction_rate': '1/s',
    'energy_price': 'EUR/J', 'energy_price_usd': 'USD/kWh', 'emission_factor': 'kg/J',
    'normal_volume': 'Nm3',
}


class UnitSystem:
    """Mapping ``quantity -> default unit`` plus conversion helpers."""

    def __init__(self, defaults: Optional[Dict[str, str]] = None, name: str = 'custom'):
        merged = {qn: q.canonical for qn, q in QUANTITIES.items()}
        for qn, u in (defaults or {}).items():
            merged[qn] = self._validated(qn, u)
        self._defaults: Dict[str, str] = merged
        self.name = name

    # -- construction -----------------------------------------------------
    @staticmethod
    def _validated(quantity: str, unit: str) -> str:
        if quantity not in QUANTITIES:
            close = difflib.get_close_matches(quantity, list(QUANTITIES), n=3)
            hint = f" Did you mean: {', '.join(close)}?" if close else ''
            raise ValueError(f"Unknown quantity '{quantity}'.{hint}")
        q = QUANTITIES[quantity]
        rec = _lookup(unit, q)
        if rec.dim != q.dimension:
            raise ValueError(
                f"Unit '{unit}' ({rec.dim}) is not valid for quantity '{quantity}' ({q.dimension}). "
                f"Accepted: {', '.join(accepted_units(quantity))}")
        return unit

    @classmethod
    def default(cls) -> 'UnitSystem':
        """The energysim standard unit system."""
        return cls({}, name='energysim-default')

    @classmethod
    def si(cls) -> 'UnitSystem':
        """SI-flavoured preset (W, V, A, J, Pa, K, kg/s, ...)."""
        return cls(_SI_DEFAULTS, name='si')

    @classmethod
    def from_overrides(cls, overrides: Dict[str, str], base: Optional['UnitSystem'] = None,
                       name: Optional[str] = None) -> 'UnitSystem':
        """*base* (default preset) with per-quantity overrides; each unit is validated."""
        base = base or cls.default()
        merged = dict(base._defaults)
        merged.update(overrides or {})
        return cls(merged, name=name or ('custom' if overrides else base.name))

    def with_overrides(self, overrides: Dict[str, str], name: Optional[str] = None) -> 'UnitSystem':
        return UnitSystem.from_overrides(overrides, base=self, name=name)

    @classmethod
    def coerce(cls, spec: Any) -> 'UnitSystem':
        """Accept ``None`` (default), a UnitSystem, 'default'/'si', or an overrides dict."""
        if spec is None:
            return cls.default()
        if isinstance(spec, UnitSystem):
            return spec
        if isinstance(spec, str):
            if spec.lower() in ('default', 'energysim', 'energysim-default'):
                return cls.default()
            if spec.lower() == 'si':
                return cls.si()
            raise ValueError(f"Unknown unit system preset '{spec}' (use 'default' or 'si').")
        if isinstance(spec, dict):
            return cls.from_overrides(spec)
        raise TypeError(f"Cannot build a UnitSystem from {type(spec).__name__}.")

    # -- queries ----------------------------------------------------------
    @property
    def defaults(self) -> Dict[str, str]:
        return dict(self._defaults)

    def quantities(self) -> List[str]:
        return list(self._defaults)

    def default_unit(self, quantity: str) -> str:
        try:
            return self._defaults[quantity]
        except KeyError:
            self._validated(quantity, '1')  # raises with close matches
            raise

    @staticmethod
    def quantity_of(unit: str) -> str:
        return quantity_of(unit)

    def _q(self, unit: str, quantity: Optional[str]) -> str:
        return quantity if quantity else quantity_of(unit)

    def to_default(self, value: Any, unit: Optional[str], quantity: Optional[str] = None) -> Any:
        """Convert *value* from *unit* to this system's default for the unit's quantity."""
        if not unit:
            return value
        q = self._q(unit, quantity)
        return self.convert(value, unit, self._defaults[q], quantity=q)

    def from_default(self, value: Any, unit: Optional[str], quantity: Optional[str] = None) -> Any:
        """Convert *value* from the default unit of the unit's quantity into *unit*."""
        if not unit:
            return value
        q = self._q(unit, quantity)
        return self.convert(value, self._defaults[q], unit, quantity=q)

    def conversion(self, from_unit: Optional[str], to_unit: Optional[str],
                   quantity: Optional[str] = None) -> Tuple[float, float]:
        return get_unit_conversion(from_unit, to_unit, quantity)

    def scale(self, from_unit: Optional[str], to_unit: Optional[str],
              quantity: Optional[str] = None) -> float:
        return get_unit_scale(from_unit, to_unit, quantity)

    def convert(self, value: Any, from_unit: Optional[str], to_unit: Optional[str],
                quantity: Optional[str] = None) -> Any:
        """Convert (scalars or numpy arrays), including affine degC/K/degF."""
        sc, off = get_unit_conversion(from_unit, to_unit, quantity)
        if sc == 1.0 and off == 0.0:
            return value
        return value * sc + off

    def is_compatible(self, u1: str, u2: str) -> bool:
        try:
            return _lookup(u1).dim == _lookup(u2).dim
        except ValueError:
            return False

    # -- presentation -----------------------------------------------------
    def table(self) -> List[Dict[str, Any]]:
        rows = []
        for qn, q in QUANTITIES.items():
            rows.append({
                'quantity': qn,
                'default_unit': self._defaults[qn],
                'accepted_units': accepted_units(qn),
                'carrier': q.carrier,
                'description': q.description,
            })
        return rows

    def format_table(self, max_units: int = 12) -> str:
        rows = self.table()
        heads = ('quantity', 'default', 'carrier', 'accepted units')
        cells = []
        for r in rows:
            acc = r['accepted_units']
            txt = ', '.join(acc[:max_units]) + (f', ... (+{len(acc) - max_units})' if len(acc) > max_units else '')
            cells.append((r['quantity'], r['default_unit'], r['carrier'], txt))
        w = [max(len(h), *(len(c[i]) for c in cells)) for i, h in enumerate(heads)]
        line = lambda c: '  '.join(c[i].ljust(w[i]) for i in range(3)) + '  ' + c[3]
        out = [f"Unit system '{self.name}' [{self.hash}]", line(heads), '  '.join('-' * x for x in w[:3]) + '  ' + '-' * 14]
        out += [line(c) for c in cells]
        return '\n'.join(out)

    def describe(self) -> str:
        """One-line summary suitable for log lines."""
        key = ('active_power', 'reactive_power', 'voltage', 'energy', 'temperature', 'pressure',
               'mass_flow', 'normal_volume_flow', 'time')
        body = ', '.join(f"{k}={self._defaults[k]}" for k in key)
        return f"unit system '{self.name}' [{self.hash}]: {body} ({len(self._defaults)} quantities)"

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {'name': self.name, 'hash': self.hash, 'defaults': dict(self._defaults)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'UnitSystem':
        return cls(d.get('defaults', {}), name=d.get('name', 'custom'))

    @property
    def hash(self) -> str:
        """Stable short hash of the quantity->unit mapping (name excluded)."""
        blob = json.dumps(self._defaults, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:12]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, UnitSystem) and self._defaults == other._defaults

    def __hash__(self) -> int:
        return int(self.hash, 16)

    def __repr__(self) -> str:
        return f"UnitSystem(name={self.name!r}, hash={self.hash!r})"


_DEFAULT: Optional[UnitSystem] = None


def default_system() -> UnitSystem:
    """Shared standard :meth:`UnitSystem.default` instance."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = UnitSystem.default()
    return _DEFAULT


# ---------------------------------------------------------------------------
# Variable name -> quantity / unit inference
# ---------------------------------------------------------------------------

# suffix (lower-case, last dotted segment) -> unit.  Longest suffixes first.
_SUFFIXES: Tuple[Tuple[str, str], ...] = tuple(sorted((
    ('_nm3_hr', 'Nm3/h'), ('_nm3_h', 'Nm3/h'), ('_nm3_s', 'Nm3/s'),
    ('_m3_hr', 'm3/h'), ('_m3_h', 'm3/h'), ('_m3_s', 'm3/s'),
    ('_kg_hr', 'kg/h'), ('_kg_h', 'kg/h'), ('_kg_s', 'kg/s'), ('_t_hr', 't/h'), ('_t_h', 't/h'),
    ('_eur_kwh', 'EUR/kWh'), ('_eur_mwh', 'EUR/MWh'),
    ('_kwh_th', 'kWh_th'), ('_mwh_th', 'MWh_th'), ('_kw_th', 'kW_th'), ('_mw_th', 'MW_th'),
    ('_kwh_lhv', 'kWh_LHV'), ('_mwh_lhv', 'MWh_LHV'),
    ('_gwh', 'GWh'), ('_mwh', 'MWh'), ('_kwh', 'kWh'), ('_wh', 'Wh'),
    ('_mvarh', 'Mvarh'), ('_kvarh', 'kvarh'),
    ('_mvar', 'Mvar'), ('_kvar', 'kvar'),
    ('_gw', 'GW'), ('_mw', 'MW'), ('_kw', 'kW'),
    ('_mva', 'MVA'), ('_kva', 'kVA'),
    ('_kv', 'kV'), ('_ka', 'kA'), ('_pu', 'pu'), ('_hz', 'Hz'), ('_ohm', 'ohm'),
    ('_degc', 'degC'), ('_degf', 'degF'), ('_deg', 'deg'), ('_rad', 'rad'),
    ('_mbar', 'mbar'), ('_kpa', 'kPa'), ('_mpa', 'MPa'), ('_bar', 'bar'), ('_psi', 'psi'),
    ('_pct', '%'), ('_percent', '%'),
    ('_keur', 'kEUR'), ('_meur', 'MEUR'), ('_eur', 'EUR'),
    ('_nm3', 'Nm3'), ('_kg', 'kg'),
), key=lambda t: -len(t[0])))

# dotted tool attribute names (OpenDSS / PowerFactory)
_DOTTED: Tuple[Tuple[str, str], ...] = (
    ('.pgini', 'MW'), ('.plini', 'MW'), ('.qgini', 'Mvar'), ('.qlini', 'Mvar'),
    ('.m:u', 'pu'), ('.kw', 'kW'), ('.kvar', 'kvar'), ('.kv', 'kV'),
)

_NAME_RULES: Tuple[Tuple[re.Pattern, str], ...] = tuple((re.compile(p), q) for p, q in (
    (r'^(power_?factor|pf|cos_?phi)(_\w+)?$', 'power_factor'),
    (r'^(q|q_(cmd|set|setpoint|ref|req|gen|load|out|in|dispatch|demand)|q(cmd|set|gen|load|ref)|'
     r'reactive_?power\w*)$', 'reactive_power'),
    (r'^(p_th|q_th|p_heat|p_thermal|heat_?(power|flow|rate)\w*|thermal_?power\w*)$', 'thermal_power'),
    (r'^(p|p_(cmd|set|setpoint|ref|req|gen|load|el|elec|out|in|dispatch|demand)|p(cmd|set|gen|load|ref)|'
     r'.*power.*)$', 'active_power'),
    (r'^t_(amb|ambient|room|in|out|set|setpoint|supply|return|cell|bat|water|air|zone|ext|int|\w*temp\w*)$'
     r'|^temp(erature)?(_\w+)?$', 'temperature'),
    (r'^soc(_\w+)?$', 'state_of_charge'),
    (r'^(eta|efficiency)(_\w+)?$', 'efficiency'),
    (r'^(freq|frequency)(_\w+)?$', 'frequency'),
    (r'^pressure(_\w+)?$', 'pressure'),
))


def _last_segment(var_name: str) -> str:
    return var_name.strip().lower().split('.')[-1]


def infer_unit_from_name(var_name: str) -> Optional[str]:
    """Unit implied by a conventional suffix / tool attribute (``p_mw`` -> ``MW``) or ``None``."""
    v = var_name.strip().lower()
    for suf, unit in _DOTTED:
        if v.endswith(suf):
            return unit
    last = v.split('.')[-1]
    for suf, unit in _SUFFIXES:
        if last.endswith(suf):
            return unit
    return None


def infer_quantity(var_name: str, unit: Optional[str] = None) -> Optional[str]:
    """Best-effort quantity of a variable, or ``None`` (never guesses generic names).

    An explicit *unit* wins; then conventional suffixes (``p_mw``, ``t_in_degc``,
    ``vm_pu``); then a few unambiguous full names (``P_cmd``, ``T_amb``, ``soc``).
    """
    if unit:
        try:
            return quantity_of(unit)
        except ValueError:
            return None
    u = infer_unit_from_name(var_name)
    if u is not None:
        last = _last_segment(var_name)
        if u == 'pu':
            return 'voltage_pu' if re.match(r'^(v|vm|vmag|u|um|voltage)(_|$)', last) else 'fraction'
        return quantity_of(u)
    last = _last_segment(var_name)
    for pat, q in _NAME_RULES:
        if pat.match(last):
            return q
    return None


def resolve_unit(var_name: str, declared: Optional[str] = None,
                 default_system: Optional[UnitSystem] = None) -> Tuple[Optional[str], str]:
    """Resolve the unit of a variable -> ``(unit, source)``.

    *source* is ``'declared'`` (validated), ``'suffix'`` (conventional name
    suffix), ``'default-by-quantity'`` (name denotes a quantity; the system's
    default unit is used) or ``'unspecified'`` (``unit`` is ``None``; the
    variable must not be silently scaled).
    """
    system = default_system if default_system is not None else globals()['default_system']()
    if declared:
        parse_unit(declared)
        return declared, 'declared'
    u = infer_unit_from_name(var_name)
    if u is not None:
        return u, 'suffix'
    q = infer_quantity(var_name)
    if q is not None:
        return system.default_unit(q), 'default-by-quantity'
    return None, 'unspecified'

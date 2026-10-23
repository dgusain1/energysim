# -*- coding: utf-8 -*-
import math
import random

import pytest

from energysim import units as U
from energysim.units import UnitSystem


def _all_quantity_unit_pairs():
    for qn in U.QUANTITIES:
        for u in U.accepted_units(qn):
            yield qn, u


# ---------------------------------------------------------------- registry

@pytest.mark.parametrize('qn,unit', list(_all_quantity_unit_pairs()))
def test_every_unit_roundtrips_to_canonical(qn, unit):
    s = UnitSystem.default()
    canon = s.default_unit(qn)
    x = 123.456
    there = s.convert(x, unit, canon, quantity=qn)
    back = s.convert(there, canon, unit, quantity=qn)
    assert math.isclose(back, x, rel_tol=1e-12, abs_tol=1e-12)


def test_every_quantity_canonical_is_valid_and_listed():
    for qn, q in U.QUANTITIES.items():
        assert q.carrier in U.CARRIERS
        assert q.description
        assert U.parse_unit(q.canonical).dimension == q.dimension or q.local_units
        assert q.canonical in U.accepted_units(qn) or q.local_units


EXPECTED_DEFAULTS = {
    'active_power': 'kW', 'reactive_power': 'kvar', 'apparent_power': 'kVA', 'voltage': 'kV',
    'current': 'A', 'energy': 'kWh', 'reactive_energy': 'kvarh', 'frequency': 'Hz',
    'impedance': 'ohm', 'resistance': 'ohm', 'voltage_angle': 'deg', 'power_factor': '1',
    'thermal_power': 'kW', 'thermal_energy': 'kWh_th', 'temperature': 'degC',
    'temperature_difference': 'K', 'specific_heat': 'kJ/(kg*K)',
    'heat_transfer_coefficient': 'W/(m2*K)', 'thermal_resistance': 'K/kW',
    'thermal_capacitance': 'kWh/K', 'mass': 'kg', 'mass_flow': 'kg/h', 'hydrogen_energy': 'kWh',
    'volume': 'm3', 'normal_volume_flow': 'Nm3/h', 'pressure': 'bar', 'density': 'kg/m3',
    'specific_energy': 'kWh/kg', 'molar_flow': 'mol/s', 'length': 'm', 'area': 'm2',
    'cell_area': 'cm2', 'current_density': 'A/cm2', 'time': 's', 'ramp_rate': 'kW/min',
    'fraction_rate': '1/min', 'energy_price': 'EUR/kWh', 'cost': 'EUR', 'co2_mass': 'kg',
    'emission_factor': 'kg/kWh', 'fraction': '1', 'state_of_charge': '1',
}


def test_declared_defaults():
    s = UnitSystem.default()
    for qn, u in EXPECTED_DEFAULTS.items():
        assert s.default_unit(qn) == u, qn


@pytest.mark.parametrize('a,b,factor', [
    ('GW', 'MW', 1e3), ('Mvar', 'kvar', 1e3), ('mV', 'V', 1e-3), ('MV', 'kV', 1e3),
    ('kWh', 'MJ', 3.6), ('GWh', 'MWh', 1e3), ('t/h', 'kg/s', 1000 / 3600), ('kg/h', 'kg/s', 1 / 3600),
    ('bar', 'kPa', 100), ('MPa', 'bar', 10), ('h', 'min', 60), ('d', 'h', 24), ('%', '1', 0.01),
    ('pu', '-', 1.0), ('MW/min', 'kW/min', 1000), ('kW/s', 'kW/min', 60), ('%/min', '1/min', 0.01),
    ('EUR/MWh', 'EUR/kWh', 1e-3), ('ct/kWh', 'EUR/kWh', 0.01), ('kEUR', 'EUR', 1e3), ('MEUR', 'kEUR', 1e3),
    ('t', 'kg', 1e3), ('kt', 't', 1e3), ('t/MWh', 'kg/kWh', 1.0), ('g/kWh', 'kg/kWh', 1e-3),
    ('A/m2', 'A/cm2', 1e-4), ('mA/cm2', 'A/cm2', 1e-3), ('m2', 'cm2', 1e4), ('L', 'm3', 1e-3),
    ('km', 'm', 1e3), ('psi', 'Pa', 6894.757293168), ('mbar', 'bar', 1e-3),
    ('MWh_LHV', 'kWh', 1e3), ('GJ', 'kWh', 1e9 / 3.6e6), ('MW_th', 'kW', 1e3),
    ('kWh_th', 'kWh', 1.0), ('kW', 'W', 1e3), ('Nm3/s', 'Nm3/h', 3600), ('m3/h', 'm3/s', 1 / 3600),
])
def test_table_driven_scales(a, b, factor):
    assert math.isclose(U.get_unit_scale(a, b), factor, rel_tol=1e-9)


@pytest.mark.parametrize('alias,unit', [
    ('kw', 'kW'), ('KW', 'kW'), ('kW_el', 'kW'), ('kWe', 'kW'), ('mw', 'MW'), ('Mw', 'MW'),
    ('m^3/h', 'm3/h'), ('m3/hr', 'm3/h'), ('m³/h', 'm3/h'), ('Nm^3/h', 'Nm3/h'),
    ('kWth', 'kW_th'), ('p.u.', 'pu'), ('PU', 'pu'), ('kvar', 'kvar'), ('kVAr', 'kvar'),
    ('kJ/(kg K)', 'kJ/(kg*K)'), ('kJ/kg/K', 'kJ/(kg*K)'), ('A/cm²', 'A/cm2'),
])
def test_aliases_resolve_to_same_unit(alias, unit):
    assert U.parse_unit(alias) == U.parse_unit(unit)


def test_celsius_aliases_and_carrier_units():
    for a in ('degC', '°C', 'C', 'celsius'):
        assert U.get_unit_conversion(a, 'K') == (1.0, 273.15)
    assert U.quantity_of('kW_th') == 'thermal_power'
    assert U.quantity_of('kW') == 'active_power'
    assert U.quantity_of('kWh_LHV') == 'hydrogen_energy'
    assert U.quantity_of('cm2') == 'cell_area'
    assert U.quantity_of('m2') == 'area'
    assert UnitSystem.default().is_compatible('kW_th', 'kW')


# ---------------------------------------------------------------- errors

def test_unknown_unit_raises_with_close_matches():
    with pytest.raises(ValueError, match="Unknown unit") as ei:
        U.parse_unit('kWhh')
    assert 'kWh' in str(ei.value)
    for a, b in (('furlong', 'kW'), ('kW', 'parsec')):
        with pytest.raises(ValueError, match="Unknown unit"):
            U.get_unit_scale(a, b)
    with pytest.raises(ValueError):
        U.parse_unit('')


def test_ambiguous_and_case_sensitive_first():
    with pytest.raises(ValueError, match="Ambiguous"):
        U.get_unit_scale('mv', 'V')
    assert math.isclose(U.get_unit_scale('MV', 'mV'), 1e9)


def test_incompatible_dimensions_raise():
    for a, b in (('kg/s', 'Nm3/h'), ('m3/h', 'Nm3/h'), ('pu', 'kV'), ('kW', 'kWh'), ('degC', 'kW'),
                 ('EUR/kWh', 'USD/kWh'), ('EUR', 'USD'), ('kvarh', 'kWh'), ('m2', 'm3')):
        with pytest.raises(ValueError, match="Incompatible"):
            U.get_unit_scale(a, b)


def test_actual_vs_normal_volume_flow_not_converted():
    s = UnitSystem.default()
    assert not s.is_compatible('m3/h', 'Nm3/h')
    with pytest.raises(ValueError, match="no implicit conversion"):
        s.convert(1.0, 'm3/h', 'Nm3/h')
    assert U.QUANTITIES['actual_volume_flow'].dimension != U.QUANTITIES['normal_volume_flow'].dimension


def test_currency_not_converted():
    s = UnitSystem.default()
    assert not s.is_compatible('EUR', 'USD')
    assert not s.is_compatible('EUR/MWh', 'USD/MWh')
    assert math.isclose(s.convert(5, 'USD/MWh', 'USD/kWh'), 0.005)
    assert U.QUANTITIES['energy_price'].dimension != U.QUANTITIES['energy_price_usd'].dimension


# ---------------------------------------------------------------- affine

def test_affine_temperature():
    s = UnitSystem.default()
    assert math.isclose(s.convert(300, 'K', 'degC'), 26.85)
    assert math.isclose(s.convert(26.85, 'degC', 'K'), 300)
    assert math.isclose(s.convert(212, 'degF', 'degC'), 100)
    sc, off = s.conversion('degC', 'K')
    assert sc == 1.0 and math.isclose(off, 273.15)
    assert math.isclose(s.to_default(300.0, 'K'), 26.85)
    assert math.isclose(s.from_default(26.85, 'K'), 300.0)
    with pytest.raises(ValueError, match="offset"):
        s.scale('degC', 'K')
    assert s.scale(None, 'kW') == 1.0
    # differences have no offset
    assert s.convert(5.0, 'K', 'degC', quantity='temperature_difference') == 5.0
    assert s.to_default(5.0, 'delta_K') == 5.0


def test_numpy_arrays():
    np = pytest.importorskip('numpy')
    out = UnitSystem.default().to_default(np.array([1.0, 2.0]), 'MW')
    assert list(out) == [1000.0, 2000.0]


# ---------------------------------------------------------------- systems

def test_overrides_validated():
    s = UnitSystem.from_overrides({'active_power': 'MW', 'temperature': 'K'})
    assert s.default_unit('active_power') == 'MW'
    assert s.default_unit('reactive_power') == 'kvar'
    assert math.isclose(s.to_default(2000.0, 'kW'), 2.0)
    assert math.isclose(s.to_default(20.0, 'degC'), 293.15)
    with pytest.raises(ValueError, match="not valid for quantity"):
        UnitSystem.from_overrides({'active_power': 'kWh'})
    with pytest.raises(ValueError, match="Unknown quantity"):
        UnitSystem.from_overrides({'active_pwr': 'kW'})
    with pytest.raises(ValueError, match="Unknown unit"):
        UnitSystem.from_overrides({'active_power': 'zorkmids'})
    with pytest.raises(ValueError):
        UnitSystem.from_overrides({'mass_flow': 'Nm3/h'})


def test_si_preset():
    s = UnitSystem.si()
    assert s.name == 'si'
    for q, u in (('active_power', 'W'), ('voltage', 'V'), ('current', 'A'), ('energy', 'J'),
                 ('pressure', 'Pa'), ('temperature', 'K'), ('mass_flow', 'kg/s')):
        assert s.default_unit(q) == u
    assert math.isclose(s.to_default(1.0, 'kW'), 1000.0)
    assert math.isclose(s.to_default(1.0, 'bar'), 1e5)
    assert s.hash != UnitSystem.default().hash


def test_coerce():
    assert UnitSystem.coerce(None) == UnitSystem.default()
    assert UnitSystem.coerce('si').name == 'si'
    assert UnitSystem.coerce({'active_power': 'MW'}).default_unit('active_power') == 'MW'
    with pytest.raises(ValueError):
        UnitSystem.coerce('bogus')


def test_table_and_format():
    s = UnitSystem.default()
    rows = s.table()
    assert [r['quantity'] for r in rows] == list(U.QUANTITIES)
    txt = s.format_table()
    for r in rows:
        assert r['quantity'] in txt
        assert r['default_unit'] in txt
        assert r['accepted_units'] and r['carrier'] and r['description']
    assert s.hash in txt and s.name in txt
    assert 'kW' in s.describe() and s.hash in s.describe()


def test_hash_stability_and_roundtrip():
    a, b = UnitSystem.default(), UnitSystem.default()
    assert a.hash == b.hash and a == b and hash(a) == hash(b)
    assert len(a.hash) == 12
    # hash pinned to the declared defaults: any change to defaults must be deliberate
    assert a.hash == UnitSystem.from_dict(a.to_dict()).hash
    c = UnitSystem.from_overrides({'active_power': 'MW'}, name='plant')
    d = UnitSystem.from_dict(c.to_dict())
    assert d == c and d.name == 'plant' and d.defaults == c.defaults
    assert c.hash != a.hash
    assert UnitSystem.from_dict({'defaults': {'pressure': 'Pa'}}).default_unit('pressure') == 'Pa'


# ---------------------------------------------------------------- inference

@pytest.mark.parametrize('name,q', [
    ('Load.L1.p_mw', 'active_power'), ('p_kw', 'active_power'), ('q_mvar', 'reactive_power'),
    ('vm_pu', 'voltage_pu'), ('bus.v_kv', 'voltage'), ('battery.e_kwh', 'energy'),
    ('t_in_degc', 'temperature'), ('T_amb', 'temperature'), ('T_room', 'temperature'),
    ('P_cmd', 'active_power'), ('P_set', 'active_power'), ('power', 'active_power'),
    ('q_kw_th', 'thermal_power'), ('e_kwh_th', 'thermal_energy'), ('m_kg_hr', 'mass_flow'),
    ('flow_nm3_hr', 'normal_volume_flow'), ('flow_m3_h', 'actual_volume_flow'), ('p_bar', 'pressure'),
    ('soc', 'state_of_charge'), ('SOC_bat', 'state_of_charge'), ('eta', 'efficiency'),
    ('f_hz', 'frequency'), ('price_eur_kwh', 'energy_price'), ('cost_eur', 'cost'),
    ('Gen.g1.q_set', 'reactive_power'), ('pf', 'power_factor'), ('loading_pct', 'fraction'),
])
def test_infer_quantity(name, q):
    assert U.infer_quantity(name) == q


@pytest.mark.parametrize('name', ['y', 'x', 'value', 'val', 'v', 't', 'i', 'out', 'signal', 'u', 'z1', ''])
def test_infer_quantity_never_guesses(name):
    assert U.infer_quantity(name) is None


def test_infer_quantity_with_unit():
    assert U.infer_quantity('y', 'MW') == 'active_power'
    assert U.infer_quantity('y', 'bogus') is None


def test_resolve_unit_sources():
    s = UnitSystem.default()
    assert U.resolve_unit('y', 'MW', s) == ('MW', 'declared')
    assert U.resolve_unit('p_mw', None, s) == ('MW', 'suffix')
    assert U.resolve_unit('P_cmd', None, s) == ('kW', 'default-by-quantity')
    assert U.resolve_unit('P_cmd', None, UnitSystem.from_overrides({'active_power': 'MW'})) == ('MW', 'default-by-quantity')
    assert U.resolve_unit('y') == (None, 'unspecified')
    with pytest.raises(ValueError, match="Unknown unit"):
        U.resolve_unit('y', 'bogus')


def test_dotted_attributes():
    assert U.infer_unit_from_name('Load.x.kW') == 'kW'
    assert U.infer_unit_from_name('Generator.g.pgini') == 'MW'


# ---------------------------------------------------------------- property

def test_roundtrip_property_seeded():
    rnd = random.Random(1234)
    pairs = []
    for qn in U.QUANTITIES:
        accepted = U.accepted_units(qn)
        for _ in range(3):
            pairs.append((qn, rnd.choice(accepted), rnd.choice(accepted)))
    s = UnitSystem.default()
    for qn, a, b in pairs:
        for _ in range(20):
            x = rnd.uniform(-1e6, 1e6)
            y = s.convert(s.convert(x, a, b, quantity=qn), b, a, quantity=qn)
            assert math.isclose(y, x, rel_tol=1e-9, abs_tol=1e-6), (qn, a, b, x, y)

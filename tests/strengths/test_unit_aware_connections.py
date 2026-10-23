# -*- coding: utf-8 -*-
"""Strength 3 - unit-aware connections: units are converted when (and only when) both
ends are known; anything unknown or incompatible fails loudly."""
from __future__ import annotations

import itertools

import pytest

from energysim import world
from energysim.units import QUANTITIES, UnitSystem, accepted_units, get_unit_conversion
from helpers import Alg

pytestmark = pytest.mark.strength


def _connected(tmp_path, src_unit=None, dst_unit=None, value=1.0, src_name="y", dst_name="u", **conn):
    w = world(start_time=0, stop_time=10, t_macro=10)
    w.add_signal("src", signal=value, step_size=10, units={"y": src_unit} if src_unit else None)
    dst = Alg()
    w.add_model("dst", dst, inputs=[dst_name], outputs=["y"], step_size=10,
                units={dst_name: dst_unit} if dst_unit else None)
    w.add_connections([{"from": f"src.{src_name}", "to": f"dst.{dst_name}", **conn}])
    return w, dst


# (from, to, value, expected)
TABLE = [
    ("kW", "MW", 1500.0, 1.5), ("MW", "kW", 1.5, 1500.0), ("W", "kW", 2500.0, 2.5), ("GW", "MW", 0.002, 2.0),
    ("kWh", "MWh", 4000.0, 4.0), ("MWh", "kWh", 0.5, 500.0), ("MJ", "kWh", 3.6, 1.0), ("Wh", "kWh", 1000.0, 1.0),
    ("kg/h", "kg/s", 3600.0, 1.0), ("t/h", "kg/h", 1.0, 1000.0), ("kg/s", "kg/h", 1.0, 3600.0),
    ("degC", "K", 20.0, 293.15), ("K", "degC", 300.0, 26.85),
    ("bar", "Pa", 2.0, 2e5), ("kPa", "bar", 200.0, 2.0), ("MPa", "bar", 0.1, 1.0),
    ("V", "kV", 20000.0, 20.0), ("kV", "V", 0.4, 400.0), ("pu", "pu", 1.02, 1.02),
    ("min", "s", 2.0, 120.0), ("h", "s", 1.0, 3600.0), ("%", "-", 50.0, 0.5),
    ("kvar", "Mvar", 3000.0, 3.0),
]


@pytest.mark.parametrize("from_unit,to_unit,value,expected", TABLE)
def test_declared_units_are_converted(from_unit, to_unit, value, expected, tmp_path):
    w, dst = _connected(tmp_path, from_unit, to_unit, value)
    w.simulate(pbar=False, record_all=False)
    assert dst.u == pytest.approx(expected, rel=1e-12)


def test_every_supported_unit_pair_is_consistent():
    """For all unit pairs of a quantity: a->b followed by b->a is the identity, and a->a is trivial."""
    n = 0
    for q in QUANTITIES:
        units = accepted_units(q)
        for a, b in itertools.product(units, repeat=2):
            s1, o1 = get_unit_conversion(a, b, q)
            s2, o2 = get_unit_conversion(b, a, q)
            for x in (0.0, 1.0, -3.7, 123.456):
                assert (x * s1 + o1) * s2 + o2 == pytest.approx(x, rel=1e-9, abs=1e-9), (q, a, b)
            if a == b:
                assert (s1, o1) == (1.0, 0.0)
            n += 1
    assert n > 200


def test_unit_system_default_and_si_agree_on_physics():
    default, si = UnitSystem.default(), UnitSystem.si()
    assert default.convert(1.0, "MW", "kW") == pytest.approx(1000.0)
    assert si.convert(1.0, "kWh", "J") == pytest.approx(3.6e6)
    assert not default.is_compatible("kW", "degC") and default.is_compatible("kW", "MW")


def test_unknown_unit_raises(tmp_path):
    with pytest.raises(ValueError, match="(?i)unknown unit"):
        _connected(tmp_path, "furlongs", "kW")
    w = world()
    with pytest.raises(ValueError, match="(?i)unknown unit"):
        w.add_signal("s", 1.0, units={"y": "parsec"})


@pytest.mark.parametrize("from_unit,to_unit", [("kW", "degC"), ("bar", "kWh"), ("kg/h", "MW"), ("pu", "kV")])
def test_explicit_incompatible_units_raise(from_unit, to_unit, tmp_path):
    with pytest.raises(ValueError, match="(?i)incompatible"):
        _connected(tmp_path, from_unit, to_unit)


def test_incompatible_units_via_connection_keys_raise(tmp_path):
    with pytest.raises(ValueError, match="(?i)incompatible"):
        _connected(tmp_path, None, None, from_unit="kW", to_unit="K")


def test_suffix_inference_converts_when_both_ends_known(tmp_path):
    w, dst = _connected(tmp_path, None, None, value=2500.0, src_name="y", dst_name="p_mw", from_unit="kW")
    w.simulate(pbar=False, record_all=False)
    assert dst.u == pytest.approx(2.5)


def test_unknown_units_are_never_guessed(tmp_path):
    """Names that carry no unit information are passed through unchanged."""
    w, dst = _connected(tmp_path, None, None, value=7.0, dst_name="flow_setpoint")
    w.simulate(pbar=False, record_all=False)
    assert dst.u == 7.0


def test_fmu_like_temperature_is_never_rescaled(tmp_path):
    """An FMU-style simulator has a native power unit of W, but a temperature must not be touched."""
    w = world(start_time=0, stop_time=10, t_macro=10, power_unit="kW")
    fmu_like = Alg(gain=1.0)
    w.add_model("fmu", fmu_like, inputs=["u"], outputs=["T_zone", "P"], step_size=10, power_unit="W")
    fmu_like.y = 21.5
    sink = Alg()
    w.add_model("sink", sink, inputs=["T_in", "p_set"], outputs=["y"], step_size=10, power_unit="kW")
    w.add_connections([{"from": "fmu.T_zone", "to": "sink.T_in"}])
    fmu_like.y = 21.5
    w.exchange_variable_values(0.0)
    assert sink.u == 21.5                       # unchanged: no kelvin/celsius/power scaling
    conn = w._parsed_connections[0]
    assert (conn.scale, conn.offset) == (1.0, 0.0)


def test_fmu_like_power_in_watts_is_converted_to_world_unit(tmp_path):
    w = world(start_time=0, stop_time=10, t_macro=10, power_unit="kW")
    src = Alg()
    w.add_model("fmu", src, inputs=["u"], outputs=["P"], step_size=10, power_unit="W")
    sink = Alg()
    w.add_model("sink", sink, inputs=["P_set"], outputs=["y"], step_size=10, power_unit="kW")
    w.add_connections([{"from": "fmu.P", "to": "sink.P_set"}])
    src.y = 3000.0
    w.exchange_variable_values(0.0)
    assert sink.u == pytest.approx(3.0)


def test_manual_scale_disables_automatic_conversion(tmp_path):
    w, dst = _connected(tmp_path, "kW", "MW", value=1000.0, scale=0.5)
    w.simulate(pbar=False, record_all=False)
    assert dst.u == pytest.approx(500.0)

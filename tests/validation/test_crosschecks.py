# -*- coding: utf-8 -*-
"""Cross-checks of energysim models/adapters against independent reference implementations
(pvlib, pandapower, PyPSA)."""
from __future__ import annotations

import cmath
import copy

import numpy as np
import pytest

from energysim import world
from energysim.models import PhysicsPV
from helpers import run_and_get
from refs import assert_ref

pytestmark = pytest.mark.validation


# ---------------------------------------------------------------------------
# PV vs pvlib
# ---------------------------------------------------------------------------

def _pv(g, t_amb=20.0, v=1.0, **kw):
    m = PhysicsPV(p_dc_stc_mw=1.0, inverter_p_ac_max_mw=2.0, **kw)
    m.init()
    m.set_value(["G_poa", "T_amb", "v_wind"], [g, t_amb, v])
    m.step(0.0)
    return m.get_value(["P_ac", "P_dc", "T_cell"], 0.0)


@pytest.mark.parametrize("g", [200.0, 400.0, 600.0, 800.0, 1000.0])
def test_pv_vs_pvlib(g):
    pvlib = pytest.importorskip("pvlib")
    p_ac, p_dc, t_cell = _pv(g)
    # cell temperature: NOCT model vs SAM NOCT and SAPM (open rack glass/glass)
    t_noct = float(pvlib.temperature.noct_sam(g, 20.0, 1.0, noct=45.0, module_efficiency=0.18))
    sapm = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]
    t_sapm = float(pvlib.temperature.sapm_cell(g, 20.0, 1.0, **sapm))
    assert_ref(t_cell, "pv_pvlib", "cell_temperature_vs_noct_sam_K", expected=t_noct)
    assert_ref(t_cell, "pv_pvlib", "cell_temperature_vs_sapm_K", expected=t_sapm)
    # DC power: identical formula to pvlib's PVWatts DC model, evaluated at the same cell temperature
    p_pvw = float(pvlib.pvsystem.pvwatts_dc(g, t_cell, 1.0, -0.0038))
    assert_ref(p_dc, "pv_pvlib", "dc_power_vs_pvwatts_dc_MW", expected=p_pvw)
    # AC power: generic inverter curve vs PVWatts inverter model
    p_inv = float(pvlib.inverter.pvwatts(p_dc * 1e6, 2.0e6, eta_inv_nom=0.96)) / 1e6
    assert_ref(p_ac, "pv_pvlib", "ac_power_vs_pvwatts_inverter_MW", expected=p_inv)


# ---------------------------------------------------------------------------
# pandapower
# ---------------------------------------------------------------------------

# MATPOWER case9 solved bus voltage magnitudes (p.u.), buses 1..9 (3-digit values from the MATPOWER manual).
_CASE9_VM = [1.000, 1.000, 1.000, 0.987, 0.975, 1.003, 0.986, 0.996, 0.958]


def _named_case9():
    pp = pytest.importorskip("pandapower")
    import pandapower.networks as nw
    net = nw.case9()
    net.bus["name"] = [f"B{i + 1}" for i in range(len(net.bus))]
    net.load["name"] = [f"L{i + 1}" for i in range(len(net.load))]
    net.gen["name"] = [f"G{i + 1}" for i in range(len(net.gen))]
    net.ext_grid["name"] = ["Slack"]
    return pp, net


def test_pandapower_case9_voltages_vs_textbook():
    pp, net = _named_case9()
    pp.runpp(net, numba=False)
    for i, v in enumerate(_CASE9_VM):
        assert_ref(net.res_bus.vm_pu.iloc[i], "pandapower", "case9_vm_pu_vs_matpower", expected=v)


def test_pandapower_via_energysim_matches_direct_runpp(tmp_path):
    pp, net = _named_case9()
    ref_net = copy.deepcopy(net)
    p_new = 150.0                          # MW, up from 90 MW at bus 5
    ref_net.load.loc[ref_net.load.name == "L1", "p_mw"] = p_new
    pp.runpp(ref_net, numba=False)
    w = world(start_time=0, stop_time=20, t_macro=10, power_unit="MW", res_filename=str(tmp_path / "pp.h5"))
    w.add_powerflow("grid", net=net, inputs=["L1.p_mw"], outputs=["B9.vm_pu", "B5.vm_pu"], step_size=10)
    w.add_signal("ld", signal=p_new * 1000.0, step_size=10, units={"y": "kW"})   # kW -> MW conversion included
    w.add_connections({"ld.y": "grid.L1.p_mw"})
    res = run_and_get(w, record_all=False)["grid"]
    last = res.iloc[-1]
    assert_ref(last["B9.vm_pu"], "pandapower", "energysim_vs_direct_runpp_vm_pu", expected=ref_net.res_bus.vm_pu.iloc[8])
    assert_ref(last["B5.vm_pu"], "pandapower", "energysim_vs_direct_runpp_vm_pu", expected=ref_net.res_bus.vm_pu.iloc[4])


def test_pandapower_two_bus_vs_independent_solver():
    """Two-bus network solved by an independent fixed-point iteration on the complex voltage."""
    pp = pytest.importorskip("pandapower")
    net = pp.create_empty_network(sn_mva=1.0)
    b0 = pp.create_bus(net, vn_kv=20.0)
    b1 = pp.create_bus(net, vn_kv=20.0)
    pp.create_ext_grid(net, b0, vm_pu=1.0)
    r_ohm, x_ohm = 0.5, 1.0
    pp.create_line_from_parameters(net, b0, b1, length_km=1.0, r_ohm_per_km=r_ohm, x_ohm_per_km=x_ohm,
                                   c_nf_per_km=0.0, max_i_ka=1.0)
    p_mw, q_mvar = 5.0, 2.0
    pp.create_load(net, b1, p_mw=p_mw, q_mvar=q_mvar)
    pp.runpp(net, numba=False)
    z_base = 20.0 ** 2 / 1.0
    y = 1.0 / complex(r_ohm / z_base, x_ohm / z_base)
    s = complex(p_mw, q_mvar)
    v = 1.0 + 0j
    for _ in range(200):                       # Gauss-Seidel on V1 = V0 - Z * conj(S / V1)
        v = 1.0 - (np.conj(s / v)) / y
    assert_ref(net.res_bus.vm_pu.iloc[1], "pandapower", "two_bus_vm_pu_vs_fixed_point", expected=abs(v))
    assert_ref(np.degrees(cmath.phase(v)), "pandapower", "two_bus_angle_deg_vs_fixed_point",
               expected=net.res_bus.va_degree.iloc[1])


# ---------------------------------------------------------------------------
# PyPSA
# ---------------------------------------------------------------------------

def _pypsa_net():
    pypsa = pytest.importorskip("pypsa")
    n = pypsa.Network()
    n.set_snapshots([0])
    n.add("Bus", "b0", v_nom=20.0)
    n.add("Bus", "b1", v_nom=20.0)
    n.add("Line", "l01", bus0="b0", bus1="b1", r=0.5, x=1.0, s_nom=100)
    n.add("Generator", "slack", bus="b0", control="Slack", p_set=0.0)
    n.add("Load", "ld", bus="b1", p_set=5.0, q_set=2.0)
    return n


def test_pypsa_direct_two_bus_vs_independent_solver():
    n = _pypsa_net()
    n.pf()
    # independent solver (per-unit on 1 MVA, 20 kV base)
    y = 1.0 / complex(0.5 / 400.0, 1.0 / 400.0)
    s = complex(5.0, 2.0)
    v = 1.0 + 0j
    for _ in range(200):
        v = 1.0 - np.conj(s / v) / y
    assert_ref(float(n.buses_t.v_mag_pu.at[0, "b1"]), "pypsa", "two_bus_vm_pu_vs_fixed_point", expected=abs(v))


def test_pypsa_via_energysim_matches_direct_pypsa(tmp_path):
    n_es, n_ref = _pypsa_net(), _pypsa_net()
    n_ref.loads.at["ld", "p_set"] = 8.0
    n_ref.pf()
    w = world(start_time=0, stop_time=20, t_macro=10, power_unit="MW", res_filename=str(tmp_path / "ps.h5"))
    try:
        w.add_pypsa("grid", network=n_es, inputs=["ld.P"], outputs=["b1.V", "slack.P"], step_size=10)
    except AttributeError as exc:     # pragma: no cover - environment dependent
        pytest.xfail(f"pypsa adapter incompatible with the installed PyPSA: {exc}")
    w.add_signal("s", signal=8.0, step_size=10, units={"y": "MW"})
    w.add_connections({"s.y": "grid.ld.P"})
    res = run_and_get(w, record_all=False)["grid"].iloc[-1]
    assert_ref(res["b1.V"], "pypsa", "energysim_vs_direct_pypsa_vm_pu", expected=float(n_ref.buses_t.v_mag_pu.at[0, "b1"]))
    assert_ref(res["slack.P"], "pypsa", "energysim_vs_direct_pypsa_slack_P_MW", expected=float(n_ref.generators_t.p.at[0, "slack"]))

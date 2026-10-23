"""Integration tests for the OpenDSS adapter (real OpenDSSDirect.py)."""
import pytest

dss = pytest.importorskip("opendssdirect")

from energysim.base import (  # noqa: E402
    EnergysimError, SimulatorElementNotFoundError, SimulatorVariableError,
)
from energysim.opendssAdapter import opendss_adapter  # noqa: E402

pytestmark = pytest.mark.requires_opendss


@pytest.fixture(autouse=True)
def _clean_dss():
    """The OpenDSS engine is a process-wide singleton: always leave it clean."""
    import energysim.opendssAdapter as mod
    mod._ACTIVE = None
    yield
    mod._ACTIVE = None
    dss.Command("Clear")


def _direct(feeder, edits=()):
    dss.Command("Clear")
    dss.Command(f'Redirect "{feeder}"')
    for e in edits:
        dss.Command(e)
    if edits:
        dss.Solution.Solve()
    out = {}
    dss.Circuit.SetActiveBus("n2")
    out["n2_pu_mean"] = sum(dss.Bus.puVmagAngle()[0::2]) / 3
    out["n2_pu_1"] = dss.Bus.puVmagAngle()[0]
    out["n2_ang_1"] = dss.Bus.puVmagAngle()[1]
    out["n2_kv_base"] = dss.Bus.kVBase()
    dss.Circuit.SetActiveElement("Line.l1")
    p = dss.CktElement.Powers()
    out["l1_p"] = sum(p[0:6:2])
    out["l1_q"] = sum(p[1:6:2])
    cur = dss.CktElement.CurrentsMagAng()
    out["l1_i"] = max(cur[0:6:2])
    out["l1_norm"] = dss.CktElement.NormalAmps()
    out["losses_kw"] = dss.Circuit.Losses()[0] / 1000.0
    out["total_kw"] = dss.Circuit.TotalPower()[0]
    return out


def test_ac_results_match_direct(feeder_dss):
    ref = _direct(feeder_dss)
    a = opendss_adapter("d", feeder_dss)
    a.init()
    got = a.get_value(["Bus.n2.vm_pu", "Bus.n2.vm_pu.1", "Bus.n2.vang", "Bus.n2.kv", "Line.l1.p_kw",
                       "Line.l1.q_kvar", "Line.l1.i_a", "Circuit.losses_kw", "Circuit.total_power_kw"])
    exp = [ref["n2_pu_mean"], ref["n2_pu_1"], ref["n2_ang_1"], ref["n2_kv_base"], ref["l1_p"], ref["l1_q"],
           ref["l1_i"], ref["losses_kw"], ref["total_kw"]]
    assert got == pytest.approx(exp, rel=1e-9)
    # vm_pu is a MAGNITUDE (the old code returned the real part of the first node's phasor)
    assert 0.9 < got[0] < 1.0
    # all three phases carried: line power is the three-phase sum (~ both loads + losses)
    assert got[4] == pytest.approx(800.0 + ref["losses_kw"], rel=1e-3)


def test_bus_voltage_is_magnitude_not_real_part(feeder_dss):
    a = opendss_adapter("d", feeder_dss)
    a.init()
    dss.Circuit.SetActiveBus("n2")
    pu = dss.Bus.PuVoltage()   # complex pairs [re1, im1, ...]
    mags = [abs(complex(pu[i], pu[i + 1])) for i in (0, 2, 4)]
    assert a.get_value(["Bus.n2.vm_pu"])[0] == pytest.approx(sum(mags) / 3)
    assert a.get_value(["Bus.n2.vm_pu.2"])[0] == pytest.approx(mags[1])
    with pytest.raises(SimulatorVariableError, match="Node 9"):
        a.get_value(["Bus.n2.vm_pu.9"])


def test_set_value_matches_direct_edit(feeder_dss):
    ref = _direct(feeder_dss, edits=("edit Load.ld2 kW=900 kvar=120",))
    a = opendss_adapter("d", feeder_dss, inputs=["Load.ld2.kW", "Load.ld2.kvar"])
    a.init()
    base = a.get_value(["Bus.n2.vm_pu"])[0]
    a.set_value(["Load.ld2.kW", "Load.ld2.kvar"], [900.0, 120.0])
    a.step(0)
    assert a.get_value(["Bus.n2.vm_pu"])[0] == pytest.approx(ref["n2_pu_mean"], rel=1e-9)
    assert a.get_value(["Bus.n2.vm_pu"])[0] < base
    assert a.get_value(["Load.ld2.p_kw"])[0] == pytest.approx(900.0, rel=1e-3)
    assert a.get_value(["Line.l1.loading_percent"]) == pytest.approx([ref["l1_i"] / ref["l1_norm"] * 100.0])


def test_dc_mode_rejected():
    with pytest.raises(ValueError, match="does not provide a DC power flow"):
        opendss_adapter("d", "x.dss", mode="dc")
    with pytest.raises(ValueError, match="does not provide a DC power flow"):
        opendss_adapter("d", "x.dss", mode="direct")
    assert opendss_adapter("d", "x.dss", mode="snapshot").mode == "ac"


def test_singleton_guard(feeder_dss):
    a = opendss_adapter("a", feeder_dss)
    b = opendss_adapter("b", feeder_dss)
    a.init()
    with pytest.raises(EnergysimError, match="singleton"):
        b.init()
    a.cleanup()
    b.init()                      # free again after cleanup
    assert b.get_value(["Circuit.losses_kw"])[0] > 0
    with pytest.raises(EnergysimError, match="singleton"):
        a.init()
    b.cleanup()


def test_cleanup_only_clears_own_circuit(feeder_dss):
    a = opendss_adapter("a", feeder_dss)
    b = opendss_adapter("b", feeder_dss)
    a.init()
    b.cleanup()                   # b never initialised: must not wipe a's circuit
    assert a.get_value(["Circuit.losses_kw"])[0] > 0
    a.cleanup()
    with pytest.raises(EnergysimError, match="not initialised"):
        a.get_value(["Circuit.losses_kw"])


def test_rerun_after_cleanup_is_identical(feeder_dss):
    a = opendss_adapter("a", feeder_dss, inputs=["Load.ld1.kW"])
    a.init()
    first = a.get_value(["Bus.n2.vm_pu", "Circuit.losses_kw"])
    a.set_value(["Load.ld1.kW"], [1500.0])
    a.step(0)
    assert a.get_value(["Bus.n2.vm_pu"])[0] != pytest.approx(first[0])
    a.cleanup()
    a.init()
    assert a.get_value(["Bus.n2.vm_pu", "Circuit.losses_kw"]) == pytest.approx(first)


@pytest.mark.parametrize("bad", [
    "Load.ld1.kW; clear",          # command injection via property
    "Load.ld1 x.kW",               # space in name
    "Load.ld1.kW=5 kvar",          # '=' in property
    "Load..kW",                    # empty name
    "Load.ld1",                    # missing property
    "Load.a.b.c",                  # too many parts
    "1Load.ld1.kW",                # type must start with a letter
])
def test_malformed_input_identifiers_rejected(bad):
    with pytest.raises(SimulatorVariableError):
        opendss_adapter("d", "x.dss", inputs=[bad])


def test_value_formatting_is_safe(feeder_dss):
    a = opendss_adapter("d", feeder_dss, inputs=["Load.ld1.kW"])
    a.init()
    for bad in ("5; clear", 'x" y', "a\nb", float("nan"), float("inf"), [1.0, 2.0], object()):
        with pytest.raises(SimulatorVariableError):
            a.set_value(["Load.ld1.kW"], [bad])
    assert a.get_value(["Circuit.losses_kw"])[0] > 0   # circuit untouched
    a.set_value(["Load.ld1.kW"], ["650"])               # numeric strings are numbers
    import numpy as np
    a.set_value(["Load.ld1.kW"], [np.float64(700.0)])
    a.step(0)
    assert a.get_value(["Load.ld1.p_kw"])[0] == pytest.approx(700.0, rel=1e-3)


def test_missing_elements_and_properties_raise(feeder_dss):
    a = opendss_adapter("d", feeder_dss, inputs=["Load.ld1.kW"])
    a.init()
    with pytest.raises(SimulatorElementNotFoundError):
        a.set_value(["Load.ghost.kW"], [1.0])
    with pytest.raises(EnergysimError):
        a.set_value(["Load.ld1.notaproperty"], [1.0])
    with pytest.raises(SimulatorElementNotFoundError):
        a.get_value(["Bus.nobus.vm_pu"])
    with pytest.raises(SimulatorElementNotFoundError):
        a.get_value(["Line.nope.p_kw"])
    with pytest.raises(SimulatorVariableError):
        a.get_value(["Bus.n1.bogus"])
    with pytest.raises(SimulatorVariableError):
        a.get_value(["Circuit.bogus"])
    with pytest.raises(SimulatorVariableError):
        a.get_value(["Line.l1.bogus"])


def test_missing_file_and_script_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        opendss_adapter("d", str(tmp_path / "nope.dss")).init()
    bad = tmp_path / "bad.dss"
    bad.write_text("New Circuit.c basekv=11\nNew Line.l1 bus1=a bus2=b linecode=doesnotexist\nSolve\n")
    a = opendss_adapter("d", str(bad))
    with pytest.raises(EnergysimError, match="LineCode"):
        a.init()
    # failed init must release the singleton so another adapter can start
    import energysim.opendssAdapter as mod
    assert mod._active_owner() is None
    good = tmp_path / "good.dss"
    good.write_text("Clear
New Circuit.c basekv=11 bus1=a
New Line.l1 bus1=a bus2=b r1=0.1 x1=0.1 length=1
Solve
")
    ok = opendss_adapter("e", str(good))
    ok.init()
    ok.cleanup()


def test_missing_voltage_bases_gives_clear_error(tmp_path):
    f = tmp_path / "nobase.dss"
    f.write_text("Clear\nNew Circuit.c basekv=11 bus1=a\nNew Line.l1 bus1=a bus2=b r1=0.1 x1=0.1 length=1\n"
                 "New Load.ld bus1=b kv=11 kw=100 kvar=10\nSolve\n")
    a = opendss_adapter("d", str(f))
    a.init()
    assert a.get_value(["Bus.b.vm_kv"])[0] > 0       # kV magnitude always available
    assert a.get_value(["Line.l1.p_kw"])[0] > 0


def test_get_available_variables_static_and_live(feeder_dss):
    a = opendss_adapter("d", feeder_dss, inputs=["Load.ld1.kW"])
    pre = a.get_available_variables()               # static scan, no circuit needed
    assert {"Load.ld1.kW", "Load.ld2.kvar"} <= set(pre["inputs"])
    assert {"Bus.n2.vm_pu", "Line.l1.p_kw", "Circuit.losses_kw"} <= set(pre["outputs"])
    assert "Bus.n2.vm_pux" not in pre["all"]
    a.init()
    live = a.get_available_variables()
    assert {"Bus.src.vm_pu", "Load.ld2.kW", "Line.l2.p_kw"} <= set(live["all"])

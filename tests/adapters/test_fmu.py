"""FMU adapter tests against the Modelica Reference-FMUs (FMI 2.0).

The reference FMUs are downloaded once into ``tests/.cache`` (sha256 pinned);
the tests skip when offline.  ``<Name>_cs.fmu`` / ``<Name>_me.fmu`` are the
Co-Simulation / Model-Exchange-only derivatives of the same model.
"""
import math
import os
import shutil
import sys
import zipfile

import numpy as np
import pytest

fmpy = pytest.importorskip("fmpy")

from energysim.base import SimulatorVariableError  # noqa: E402
from energysim.csAdapter import FmuCsAdapter  # noqa: E402
from energysim.meAdapter import FmuMeAdapter  # noqa: E402
from fixtures.fetch_reference_fmus import get_reference_fmus  # noqa: E402

pytestmark = pytest.mark.requires_fmu


@pytest.fixture(scope="session")
def fmus():
    refs = get_reference_fmus()
    if refs is None:
        pytest.skip("Reference FMUs unavailable (offline or unsupported platform)")
    return {k: {t: str(p) for t, p in v.items()} for k, v in refs.items()}


@pytest.fixture(scope="session")
def cvode_ok():
    from energysim.meAdapter import _import_cvode_solver
    try:
        _import_cvode_solver()
    except Exception as exc:  # pragma: no cover - platform specific
        return False
    return True


def run_me(path, solver, step, stop, outputs, params=None, macro=1.0, start=0.0, record=None):
    """Drive an ME adapter in macro steps like the world does; return rows [t, *outputs]."""
    a = FmuMeAdapter(path, step_size=step, outputs=outputs, solver_name=solver)
    a.configure(start, stop, macro)
    if params:
        a.set_parameters(params)
    a.init()
    rows = []
    t = start
    try:
        while t < stop - 1e-9:
            rows.extend(a.advance_with_recording(t, t + macro, outputs))
            t += macro
        rows.append([t] + a.get_value(outputs, t))
    finally:
        a.cleanup()
    return np.array(rows)


# ---------------------------------------------------------------------------
# Model Exchange: analytic and reference solutions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("solver", ["Cvode", "Euler"])
def test_me_dahlquist_runs_beyond_100s(fmus, cvode_ok, solver):
    """Regression: ME adapter used to cap at stop_time=100 (Euler) / maxStep=2 s (CVode)."""
    if solver == "Cvode" and not cvode_ok:
        pytest.skip("SUNDIALS/CVode library not loadable on this platform")
    k, h = 0.01, 0.01
    rows = run_me(fmus["Dahlquist"]["me"], solver, h, 300.0, ["x"], params={"k": k}, macro=50.0)
    t, x = rows[:, 0], rows[:, 1]
    assert t[-1] == pytest.approx(300.0)
    assert x[-1] > 0.0, "solution stopped at t=100 or was never integrated"
    if solver == "Cvode":
        assert x[-1] == pytest.approx(math.exp(-k * 300.0), rel=1e-3)
        mid = x[np.argmin(abs(t - 150.0))]
        assert mid == pytest.approx(math.exp(-k * 150.0), rel=1e-3)
    else:  # explicit Euler has the closed-form (1 - k h)^n
        n = np.rint(t / h)
        assert x == pytest.approx((1 - k * h) ** n, rel=1e-6)
        assert x[-1] == pytest.approx(math.exp(-k * 300.0), rel=0.01)


def test_me_cvode_max_step_derived_from_macro_step(fmus, cvode_ok, monkeypatch):
    if not cvode_ok:
        pytest.skip("CVode unavailable")
    import energysim.meAdapter as me
    real = me._import_cvode_solver()
    captured = {}

    def spy(**kw):
        captured.update(kw)
        return real(**kw)

    monkeypatch.setattr(me, "_import_cvode_solver", lambda: spy)
    a = FmuMeAdapter(fmus["Dahlquist"]["me"], step_size=0.1, solver_name="Cvode", stop_time=100)
    a.configure(0.0, 5000.0, 25.0)
    a.init()
    a.cleanup()
    assert captured["maxStep"] == 25.0              # not (stop-start)/50 == 100 nor stop_time(100)/50 == 2
    assert captured["startTime"] == 0.0
    # stand-alone (no configure): falls back to the micro step
    captured.clear()
    b = FmuMeAdapter(fmus["Dahlquist"]["me"], step_size=0.1, solver_name="Cvode")
    b.init()
    b.cleanup()
    assert captured["maxStep"] == 0.1


def test_me_init_applies_inputs_at_start_time(fmus, monkeypatch):
    a = FmuMeAdapter(fmus["Dahlquist"]["me"], step_size=0.1, solver_name="Euler", start_time=7.0)
    calls = []
    a.configure(7.0, 20.0, 1.0)
    a.set_parameters({})
    a._ensure_instance()
    orig = a.input.apply
    a.input.apply = lambda t, *args, **kw: (calls.append(t), orig(t, *args, **kw))[1]
    a.init()                                       # init() builds on the existing instance
    a.cleanup()
    assert calls and calls[0] == 7.0


def test_me_invalid_solver_name(fmus):
    with pytest.raises(ValueError, match="Cvode.*Euler"):
        FmuMeAdapter(fmus["Dahlquist"]["me"], solver_name="RK4")
    assert FmuMeAdapter(fmus["Dahlquist"]["me"], solver_name="euler").solver_name == "Euler"


def test_me_vanderpol_matches_fmpy_reference(fmus, cvode_ok):
    if not cvode_ok:
        pytest.skip("CVode unavailable")
    ref = fmpy.simulate_fmu(fmus["VanDerPol"]["me"], stop_time=60.0, output_interval=1.0,
                            solver="CVode", relative_tolerance=1e-9, record_events=False)
    rows = run_me(fmus["VanDerPol"]["me"], "Cvode", 0.05, 60.0, ["x0", "x1"], macro=10.0)
    for t in (10.0, 30.0, 60.0):
        r = ref[np.isclose(ref["time"], t)][0]
        got = rows[np.isclose(rows[:, 0], t)][0]
        assert got[1] == pytest.approx(r["x0"], abs=2e-3)
        assert got[2] == pytest.approx(r["x1"], abs=2e-3)


def test_me_bouncing_ball_events(fmus, cvode_ok):
    if not cvode_ok:
        pytest.skip("CVode unavailable")
    rows = run_me(fmus["BouncingBall"]["me"], "Cvode", 0.005, 3.0, ["h", "v"], macro=0.5)
    t, h, v = rows[:, 0], rows[:, 1], rows[:, 2]
    assert h.min() > -1e-3                                  # never falls through the floor
    t_impact = math.sqrt(2.0 / 9.81)                          # drop from h=1 m
    # velocity direction flips with e=0.7 right after the first impact
    i = np.argmin(abs(t - (t_impact + 0.02)))
    v_impact = math.sqrt(2.0 * 9.81)
    assert v[i] == pytest.approx(0.7 * v_impact - 9.81 * (t[i] - t_impact), rel=0.05)
    assert v[np.argmin(abs(t - (t_impact - 0.02)))] < 0.0
    # tunable parameter via start value (before init): e=0.3 -> much weaker rebound
    rows2 = run_me(fmus["BouncingBall"]["me"], "Cvode", 0.005, 1.0, ["h", "v"],
                   params={"e": 0.3}, macro=0.5)
    j = np.argmin(abs(rows2[:, 0] - (t_impact + 0.02)))
    assert rows2[j, 2] == pytest.approx(0.3 * v_impact - 9.81 * (rows2[j, 0] - t_impact), rel=0.05)


def test_me_stair_time_events_match_reference(fmus):
    ref = fmpy.simulate_fmu(fmus["Stair"]["me"], stop_time=10.0, output_interval=1.0,
                            solver="Euler", step_size=0.01, record_events=False)
    rows = run_me(fmus["Stair"]["me"], "Euler", 0.25, 10.0, ["counter"], macro=1.0)
    assert len(ref) >= 8
    for t, r in zip(ref["time"], ref["counter"]):
        got = rows[np.isclose(rows[:, 0], t)][-1][1]
        assert got == r, f"counter at t={t}"
    assert rows[-1][1] == 10      # model counts up to 10 and then terminates


# ---------------------------------------------------------------------------
# Co-Simulation
# ---------------------------------------------------------------------------

def test_cs_dahlquist_fixed_step_solution(fmus):
    k, h = 1.0, 0.1
    a = FmuCsAdapter(fmus["Dahlquist"]["cs"], step_size=h, outputs=["x"])
    a.configure(0.0, 5.0, 1.0)
    a.set_parameters({"k": k})
    a.init()
    xs = [a.get_value(["x"], 0)[0]]
    for i in range(5):
        a.advance(i * 1.0, (i + 1) * 1.0)
        xs.append(a.get_value(["x"], 0)[0])
    a.cleanup()
    # the reference FMU integrates with explicit Euler at the communication step
    assert xs == pytest.approx([(1 - k * h) ** (10 * i) for i in range(6)], rel=1e-9)
    assert xs[-1] == pytest.approx(math.exp(-5.0), rel=0.30)


def test_cs_does_not_overshoot_final_partial_step(fmus):
    a = FmuCsAdapter(fmus["Dahlquist"]["cs"], step_size=0.3, outputs=["x"])
    a.configure(0.0, 1.0, 1.0)
    a.init()
    a.advance(0.0, 1.0)      # 0.3 + 0.3 + 0.3 + 0.1
    assert a.get_value(["x"], 1.0)[0] > 0
    a.cleanup()


@pytest.mark.parametrize("kind", ["cs", "me"])
def test_feedthrough_passes_inputs_to_outputs(fmus, kind):
    cls = FmuCsAdapter if kind == "cs" else FmuMeAdapter
    kwargs = {"solver_name": "Euler"} if kind == "me" else {}
    a = cls(fmus["Feedthrough"][kind], step_size=0.5, **kwargs)
    a.configure(0.0, 4.0, 1.0)
    a.init()
    a.set_value(["Float64_continuous_input", "Int32_input", "Boolean_input", "String_input"],
                [3.25, 7, True, "hello"])
    a.advance(0.0, 1.0)
    got = a.get_value(["Float64_continuous_output", "Int32_output", "Boolean_output", "String_output"], 1.0)
    assert got[0] == 3.25 and got[1] == 7 and got[2] is True and got[3] == "hello"
    a.set_value(["Boolean_input"], ["false"])
    a.advance(1.0, 2.0)
    assert a.get_value(["Boolean_output"], 2.0) == [False]
    with pytest.raises(SimulatorVariableError, match="could not be converted"):
        a.set_value(["Boolean_input"], ["maybe"])
    a.cleanup()


# ---------------------------------------------------------------------------
# life-cycle / introspection / errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["cs", "me"])
def test_rerunnable_after_cleanup_and_tempdir_removed(fmus, kind):
    cls = FmuCsAdapter if kind == "cs" else FmuMeAdapter
    kwargs = {"solver_name": "Euler"} if kind == "me" else {}
    a = cls(fmus["Dahlquist"][kind], step_size=0.1, outputs=["x"], **kwargs)
    assert a.SET_PARAMS_BEFORE_INIT is True
    results = []
    dirs = []
    for _ in range(3):
        a.configure(0.0, 2.0, 1.0)
        a.set_parameters({"k": 0.7})
        a.init()
        dirs.append(a.unzipDir)
        assert os.path.isdir(a.unzipDir)
        a.advance(0.0, 2.0)
        results.append(a.get_value(["x"], 2.0)[0])
        a.cleanup()
        assert a.fmu is None and not os.path.exists(dirs[-1])   # temp dir removed, instance freed
    assert results[0] == results[1] == results[2] != 1.0


def test_start_values_do_not_leak_into_next_run(fmus):
    a = FmuCsAdapter(fmus["Dahlquist"]["cs"], step_size=0.1, outputs=["x"])
    a.set_parameters({"k": 3.0})
    a.init()
    assert a.get_value(["k"], 0) == [3.0]
    a.cleanup()
    a.init()
    assert a.get_value(["k"], 0) == [1.0]          # model default again
    a.cleanup()


def test_failed_init_removes_temp_dir(fmus, tmp_path):
    """An FMU without a binary for this platform fails to instantiate: no temp dir may leak."""
    src = fmus["Dahlquist"]["cs"]
    broken = str(tmp_path / "broken.fmu")
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(broken, "w") as zout:
        for item in zin.infolist():
            if not item.filename.startswith("binaries/"):
                zout.writestr(item, zin.read(item.filename))
    a = FmuCsAdapter(broken, validate=False)
    with pytest.raises(Exception):
        a.init()
    assert a.unzipDir is None and a.fmu is None


@pytest.mark.parametrize("kind", ["cs", "me"])
def test_variable_introspection_catches_typos(fmus, kind):
    cls = FmuCsAdapter if kind == "cs" else FmuMeAdapter
    a = cls(fmus["BouncingBall"][kind])
    av = a.get_available_variables()
    assert set(av["outputs"]) == {"h", "v"}
    assert {"g", "e"} <= set(av["inputs"])               # parameters count as inputs
    assert {"h", "v", "der(h)", "der(v)", "g", "e", "v_min", "time"} <= set(av["all"])
    assert "hh" not in av["all"]
    with pytest.raises(SimulatorVariableError, match="'hh' not found"):
        a.get_value(["hh"], 0)
    with pytest.raises(SimulatorVariableError, match="'ee' not found"):
        a.set_value(["ee"], [1.0])
    with pytest.raises(SimulatorVariableError, match="no settable variable"):
        a.set_parameters({"gg": 1.0})
    a.cleanup()
    fv = cls(fmus["Feedthrough"][kind]).get_available_variables()
    assert "Float64_continuous_input" in fv["inputs"] and "Float64_continuous_output" in fv["outputs"]


def test_model_description_read_once(fmus, monkeypatch):
    import energysim.csAdapter as cs
    calls = []
    real = cs.read_model_description
    monkeypatch.setattr(cs, "read_model_description", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    a = FmuCsAdapter(fmus["Dahlquist"]["cs"])
    a.init()
    a.cleanup()
    a.init()
    a.cleanup()
    assert len(calls) == 1


def test_wrong_interface_is_rejected(fmus):
    with pytest.raises(ValueError, match="Model Exchange"):
        FmuMeAdapter(fmus["Dahlquist"]["cs"])
    with pytest.raises(ValueError, match="Co-Simulation"):
        FmuCsAdapter(fmus["Dahlquist"]["me"])


# ---------------------------------------------------------------------------
# through the world
# ---------------------------------------------------------------------------

def _world(tmp_path, stop, t_macro, name="res.h5", **kw):
    from energysim import world
    kw.setdefault("power_unit", "W")        # FMUs default to W: avoid kW<->W scaling of plain signals
    return world(start_time=0, stop_time=stop, t_macro=t_macro, logging=False, **kw)


def _frame(w, sim):
    """Results of *sim* as a frame with a ``time`` column (seconds); includes the final row at stop_time."""
    return w.last_results[sim].reset_index().sort_values("time", kind="stable").reset_index(drop=True)


@pytest.mark.parametrize("kind,solver", [("cs", None), ("me", "Euler"), ("me", "Cvode")])
def test_world_dahlquist_with_options_init(fmus, cvode_ok, tmp_path, kind, solver):
    if solver == "Cvode" and not cvode_ok:
        pytest.skip("CVode unavailable")
    # the reference CS FMU integrates with a fixed internal Euler step of 0.1 s
    k, h, stop = 0.5, (0.1 if kind == "cs" else 0.01), 12.0
    w = _world(tmp_path, stop, 2.0)
    extra = {"solver_name": solver} if solver else {}
    w.add_simulator(sim_type="fmu", sim_name="dq", model=fmus["Dahlquist"][kind], outputs=["x"],
                    step_size=h, **extra)
    w.options({"init": {"dq": (["k"], [k])}})            # start value applied before init()
    w.simulate(pbar=False, record_all=True)
    df = _frame(w, "dq")
    x, t = df["x"].to_numpy(), df["time"].to_numpy()
    n = np.rint(t / h)
    if kind == "cs" or solver == "Euler":
        assert x == pytest.approx((1 - k * h) ** n, rel=1e-6)
    else:
        assert x == pytest.approx(np.exp(-k * t), rel=1e-3, abs=1e-9)
    assert t.max() >= stop - 2 * h


def test_world_me_runs_longer_than_100s(fmus, cvode_ok, tmp_path):
    solvers = ["Euler"] + (["Cvode"] if cvode_ok else [])
    for solver in solvers:
        w = _world(tmp_path, 400.0, 100.0, name=f"long_{solver}.h5")
        w.add_simulator(sim_type="fmu", sim_name="dq", model=fmus["Dahlquist"]["me"], outputs=["x"],
                        step_size=0.5, solver_name=solver)
        w.options({"init": {"dq": (["k"], [0.02])}})
        w.simulate(pbar=False, record_all=False)
        df = _frame(w, "dq")
        assert df["time"].max() >= 300.0
        late = df[df["time"] >= 200.0]["x"].to_numpy()
        assert late[-1] == pytest.approx(math.exp(-0.02 * df["time"].iloc[-1]), rel=0.08)
        assert late[-1] > 0.0


@pytest.mark.parametrize("kind", ["cs", "me"])
def test_world_signal_drives_feedthrough(fmus, tmp_path, kind):
    extra = {"solver_name": "Euler"} if kind == "me" else {}
    w = _world(tmp_path, 6.0, 1.0)
    w.add_signal("src", signal=lambda t: 2.0 * t + 1.0, step_size=1.0)
    w.add_simulator(sim_type="fmu", sim_name="ft", model=fmus["Feedthrough"][kind],
                    inputs=["Float64_continuous_input"], outputs=["Float64_continuous_output"],
                    step_size=0.5, **extra)
    w.add_connections({"src.y": "ft.Float64_continuous_input"})
    w.simulate(pbar=False, record_all=True)
    df = _frame(w, "ft")
    # macro-step rows (the final row at stop_time holds the state after the last step,
    # i.e. the output for the input exchanged at t=5, without a new exchange)
    macro = df[np.isclose(df["time"] % 1.0, 0.0) & (df["time"] < 6.0)]
    assert len(macro) >= 5
    assert df["time"].iloc[-1] == 6.0
    assert macro["Float64_continuous_output"].to_numpy() == pytest.approx(2.0 * macro["time"].to_numpy() + 1.0)


@pytest.mark.parametrize("kind,solver", [("cs", None), ("me", "Euler")])
def test_world_second_simulate_is_identical(fmus, tmp_path, kind, solver):
    extra = {"solver_name": solver} if solver else {}
    w = _world(tmp_path, 4.0, 1.0)
    w.add_simulator(sim_type="fmu", sim_name="dq", model=fmus["Dahlquist"][kind], outputs=["x"],
                    step_size=0.1, **extra)
    w.options({"init": {"dq": (["k"], [0.8])}})
    w.simulate(pbar=False, record_all=True)
    first = _frame(w, "dq")
    w.simulate(pbar=False, record_all=True)
    second = _frame(w, "dq")
    assert first.shape == second.shape
    assert first["x"].to_numpy() == pytest.approx(second["x"].to_numpy(), rel=0, abs=0)
    assert first["x"].iloc[-1] < 1.0


def test_world_validation_flags_variable_typos(fmus, tmp_path):
    w = _world(tmp_path, 4.0, 1.0)
    w.add_signal("src", signal=lambda t: t, step_size=1.0)
    w.add_simulator(sim_type="fmu", sim_name="ft", model=fmus["Feedthrough"]["cs"],
                    inputs=["Float64_continuous_input"], outputs=["Float64_continuous_output"], step_size=1.0)
    with pytest.warns(UserWarning, match="Float64_continuous_inptu"):
        w.add_connections({"src.y": "ft.Float64_continuous_inptu"})


# ---------------------------------------------------------------------------
# The repo's own example FMUs (win64 CS binaries only)
# ---------------------------------------------------------------------------

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "Optimal Dispatch of FER")


@pytest.mark.skipif(not (sys.platform == "win32" and fmpy.platform == "win64"),
                    reason="example FMUs ship win64 binaries only")
@pytest.mark.parametrize("name", ["Hydrogen.ptg_modelB_case2.fmu", "P2H.pth_modelB_case2.fmu"])
def test_example_fmus_cs_win64(name):
    path = os.path.abspath(os.path.join(EXAMPLE_DIR, name))
    if not os.path.exists(path):
        pytest.skip("example FMU missing")
    a = FmuCsAdapter(path, step_size=60.0, validate=False)
    av = a.get_available_variables()
    assert "P_order" in av["inputs"] and av["all"]
    a.configure(0.0, 600.0, 60.0)
    a.init()
    a.advance(0.0, 600.0)
    a.cleanup()
    a.init()
    a.cleanup()

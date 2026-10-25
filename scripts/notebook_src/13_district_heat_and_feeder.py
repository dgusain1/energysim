# %% [markdown]
# # 13 - District heat, house archetypes and an electric feeder (sector coupling)
#
# A cold winter day in a small town where **one heating plant serves both the electricity and the heat
# sectors**. The question is simple: *what does it do to the feeder when the district-heat (DH) plant
# switches from a gas boiler to a heat pump?*
#
# The case has four parts:
#
# 1. **Houses.** Three archetypes (old, medium, new) are simulated as 2R2C lumped-capacitance models
#    with a thermostat and a maximum heating output. Each archetype is scaled by a house count. The
#    model is an Octave function `house_archetype.m` when Octave is available, and an equivalent
#    pure-Python `ModelComponent` otherwise. Both implement the same equations.
# 2. **District heat.** A `HeatNetwork` with producers (an industrial heat pump, a gas boiler and an
#    electric boiler as peaker), a thermal storage, and pipes with delays and losses to the three house
#    groups.
# 3. **Electric feeder.** An OpenDSS feeder (written by this notebook) with a PV cluster, an
#    aggregated individual-heat-pump cluster and the central DH heat pump and electric boiler as large
#    loads.
# 4. **Sector coupling.** The same day is run twice: once with the gas boiler carrying the base load,
#    and once with the heat pump carrying it. We compare feeder voltage and losses.
#
# **Optional dependencies.** OpenDSS needs `opendssdirect`. If it is not installed, the feeder parts
# are skipped and the heat network still runs. Octave needs `oct2py` and a working GNU Octave. Without
# Octave the Python house model is used. No notebook cell runs for more than a few seconds.
#
# All parameters are illustrative. Replace them with measured or design data before using the results.

# %%
import math
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import pandas as pd

from energysim import world
from energysim.models import IndustrialHeatPump, ElectricBoiler
from energysim.models.common.base_component import ModelComponent
from energysim.networks import HeatNetwork

try:
    import opendssdirect  # noqa: F401
    HAS_DSS = True
except ImportError:
    HAS_DSS = False

def _octave_available():
    """oct2py must import AND a GNU Octave session must start."""
    try:
        from oct2py import Oct2Py
    except ImportError:
        return False
    try:
        oc = Oct2Py()
        oc.exit()
        return True
    except Exception:
        return False


HAS_OCT = _octave_available()

WORK_DIR = Path(tempfile.mkdtemp(prefix="energysim_nb13_"))
print(f"OpenDSS available: {HAS_DSS}   oct2py importable: {HAS_OCT}")
print(f"Scratch folder for the .m and .dss files: {WORK_DIR}")

# %% [markdown]
# ## 1. Weather and the house archetypes
#
# The day is a cold winter day: the outdoor temperature swings between about -7 and -1 degC, with the
# minimum around 03:00. Heat demand therefore peaks at night, and PV peaks at midday when demand is low.
# The weather is a function of time, so it is deterministic.

# %%
def t_amb_c(t):
    """Outdoor temperature [degC]: -7 at 03:00, -1 at 15:00."""
    return -4.0 + 3.0 * math.cos(2.0 * math.pi * (t - 15.0 * 3600.0) / 86400.0)


def pv_kw(t):
    """PV cluster output [kW]: a short winter day, sine between 08:00 and 16:00."""
    hr = (t / 3600.0) % 24.0
    if 8.0 <= hr <= 16.0:
        return 1500.0 * math.sin(math.pi * (hr - 8.0) / 8.0)
    return 0.0


def base_load_kw(t):
    """Uncontrolled household and commercial load [kW] with a morning and an evening peak."""
    hr = (t / 3600.0) % 24.0
    return 600.0 + 250.0 * math.exp(-((hr - 8.0) / 1.5) ** 2) + 350.0 * math.exp(-((hr - 19.0) / 2.0) ** 2)


hours = np.arange(0, 86400, 900.0)
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(hours / 3600, [t_amb_c(t) for t in hours], color="tab:blue", label="outdoor temperature [degC]")
ax2 = ax.twinx()
ax2.plot(hours / 3600, [pv_kw(t) for t in hours], color="tab:orange", label="PV cluster [kW]")
ax2.plot(hours / 3600, [base_load_kw(t) for t in hours], color="tab:green", label="base load [kW]")
ax.set_xlabel("Time of day [h]")
ax.set_ylabel("Temperature [degC]")
ax2.set_ylabel("Power [kW]")
ax.set_title("Cold winter day: weather and electric boundary conditions")
lines = ax.get_lines() + ax2.get_lines()
ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### The 2R2C house equations
#
# Each house has an indoor air node (temperature `Ta`, capacity `C_a`) and a building-mass node (`Tm`,
# capacity `C_m`). The thermostat applies a proportional gain `K_th` and saturates at `Q_max`:
#
# $$C_a \frac{dT_a}{dt} = Q_h + H_{am}(T_m - T_a) + H_{ve}(T_e - T_a)$$
#
# $$C_m \frac{dT_m}{dt} = H_{am}(T_a - T_m) + H_{me}(T_e - T_m)$$
#
# $$Q_h = \min\big(Q_{max},\ \max(0,\ K_{th}(T_{set} - T_a))\big)$$
#
# with $T_e$ the outdoor temperature. The `H` coefficients are in kW/K, the capacities in kJ/K and the
# heat in kW. The integrator is explicit Euler with sub-steps of at most 5 s, so the stiff thermostat
# stays stable inside a 15 min macro step.
#
# The three archetypes differ in envelope (`H_ve`, `H_me`), thermal mass (`C_m`) and heating capacity
# (`Q_max`). Each one is simulated as a group: the model reports the indoor temperature of one house and
# the heat demand of the whole group, `n_houses * Q_h`.

# %%
HOUSE_PARAMS = {
    # id: (name, H_ve, H_me, H_am, C_a, C_m, Q_max)  H in kW/K, C in kJ/K, Q_max in kW per house
    1: ("old",    0.22, 0.05, 0.40, 1500.0, 20000.0, 12.0),
    2: ("medium", 0.12, 0.03, 0.30, 1500.0, 15000.0, 8.0),
    3: ("new",    0.06, 0.02, 0.25, 1500.0, 12000.0, 5.0),
}
T_SET_C = 21.0
K_TH_KW_PER_K = 50.0
SUB_STEP_S = 5.0
HOUSE_COUNTS = {1: 400, 2: 600, 3: 300}

HOUSE_M_FILE = r"""function [T_indoor_C, Q_heat_kW] = house_archetype(archetype_id, T_ambient_C, n_houses, time)
% HOUSE_ARCHETYPE  Group of identical 2R2C houses with a thermostat (energysim co-simulation).
%   archetype_id : 1 = old, 2 = medium, 3 = new
%   T_ambient_C  : outdoor temperature [degC]
%   n_houses     : number of houses in the group (scales the heat demand)
%   time         : simulation time [s] (appended by energysim)
%   T_indoor_C   : indoor air temperature of one house [degC]
%   Q_heat_kW    : heat demand of the whole group [kW]
%
%   C_a dTa/dt = Q_h + H_am (Tm - Ta) + H_ve (Te - Ta)
%   C_m dTm/dt = H_am (Ta - Tm) + H_me (Te - Tm)
%   Q_h = min(Q_max, max(0, K_th (T_set - Ta)))
%
%   Explicit Euler with sub-steps of at most 5 s. dt is taken from consecutive time arguments,
%   so the first call only initialises the state.
persistent Ta Tm t_prev
if isempty(Ta)
    Ta = [20 20 20];
    Tm = [20 20 20];
    t_prev = nan(1, 3);
end
k = round(archetype_id);
% columns: H_ve H_me H_am C_a C_m Q_max
P = [0.22 0.05 0.40 1500 20000 12;
     0.12 0.03 0.30 1500 15000  8;
     0.06 0.02 0.25 1500 12000  5];
H_ve = P(k,1); H_me = P(k,2); H_am = P(k,3);
C_a = P(k,4); C_m = P(k,5); Q_max = P(k,6);
T_set = 21.0;
K_th = 50.0;
if isnan(t_prev(k))
    dt = 0;
else
    dt = time - t_prev(k);
end
t_prev(k) = time;
if dt > 0
    n_sub = ceil(dt / 5);
    h = dt / n_sub;
    for i = 1:n_sub
        Q_h = min(Q_max, max(0, K_th * (T_set - Ta(k))));
        dTa = (Q_h + H_am * (Tm(k) - Ta(k)) + H_ve * (T_ambient_C - Ta(k))) / C_a;
        dTm = (H_am * (Ta(k) - Tm(k)) + H_me * (T_ambient_C - Tm(k))) / C_m;
        Ta(k) = Ta(k) + dTa * h;
        Tm(k) = Tm(k) + dTm * h;
    end
end
Q_h = min(Q_max, max(0, K_th * (T_set - Ta(k))));
T_indoor_C = Ta(k);
Q_heat_kW = n_houses * Q_h;
end
"""


def house_py_step(state, archetype_id, t_amb, dt, n_houses):
    """Pure-Python twin of house_archetype.m: same equations, same sub-stepping.

    ``state`` is a dict with ``Ta``, ``Tm`` and ``t_prev`` (NaN on the first call).
    Returns ``(T_indoor_C, Q_heat_kW)``.
    """
    _, h_ve, h_me, h_am, c_a, c_m, q_max = HOUSE_PARAMS[int(round(archetype_id))]
    if dt > 0:
        n_sub = math.ceil(dt / SUB_STEP_S)
        h = dt / n_sub
        for _ in range(n_sub):
            q_h = min(q_max, max(0.0, K_TH_KW_PER_K * (T_SET_C - state["Ta"])))
            d_ta = (q_h + h_am * (state["Tm"] - state["Ta"]) + h_ve * (t_amb - state["Ta"])) / c_a
            d_tm = (h_am * (state["Ta"] - state["Tm"]) + h_me * (t_amb - state["Tm"])) / c_m
            state["Ta"] += d_ta * h
            state["Tm"] += d_tm * h
    q_h = min(q_max, max(0.0, K_TH_KW_PER_K * (T_SET_C - state["Ta"])))
    return state["Ta"], n_houses * q_h


class HouseArchetypeModel(ModelComponent):
    """Python fallback for ``house_archetype.m`` (one archetype group per instance)."""

    units = {"archetype_id": "1", "T_ambient_C": "degC", "n_houses": "1",
             "T_indoor_C": "degC", "Q_heat_kW": "kW"}

    def __init__(self, name="house", step_size=900.0, **kwargs):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self._state = {"Ta": 20.0, "Tm": 20.0, "t_prev": float("nan")}
        self.register_variables(inputs=["archetype_id", "T_ambient_C", "n_houses"],
                                outputs=["T_indoor_C", "Q_heat_kW"],
                                defaults={"archetype_id": 1.0, "T_ambient_C": -4.0, "n_houses": 1.0})

    def step_model(self, dt: float, time: float) -> None:
        arch = float(self._inputs_val.get("archetype_id", 1.0))
        t_amb = float(self._inputs_val.get("T_ambient_C", -4.0))
        n = float(self._inputs_val.get("n_houses", 1.0))
        # same rule as the .m file: dt from consecutive time values, zero on the first call
        prev = self._state["t_prev"]
        dt_eff = 0.0 if math.isnan(prev) else time - prev
        self._state["t_prev"] = time
        t_in, q_kw = house_py_step(self._state, arch, t_amb, dt_eff, n)
        self._outputs_val["T_indoor_C"] = t_in
        self._outputs_val["Q_heat_kW"] = q_kw


def make_house_adapter(name, archetype_id, backend):
    """Return a house-group simulator for the chosen backend ('octave' or 'python')."""
    if backend == "octave":
        from energysim.matlabAdapter import matlab_adapter
        return matlab_adapter(func_name="house_archetype", script_loc=str(WORK_DIR),
                              inputs=["archetype_id", "T_ambient_C", "n_houses"],
                              outputs=["T_indoor_C", "Q_heat_kW"], engine="octave")
    return HouseArchetypeModel(name=name, step_size=900.0)


BACKEND = "octave" if HAS_OCT else "python"
(WORK_DIR / "house_archetype.m").write_text(HOUSE_M_FILE, encoding="utf-8")
print(f"House backend in this run: {BACKEND}")
if not HAS_OCT:
    print("SKIP Octave: oct2py is not installed, so the Python twin of house_archetype.m is used.")

# %%
# Backend check: the same inputs, the same outputs. Only a few short calls, not a simulation.
def compare_house_backends(n_steps=8, dt=900.0):
    if not HAS_OCT:
        print("SKIP backend comparison (oct2py not available).")
        return None
    try:
        oc_adapter = make_house_adapter("house_old", 1, "octave")
        oc_adapter.init()
    except Exception as exc:          # Octave installed but not working
        print(f"SKIP backend comparison (Octave did not start: {exc}).")
        return None
    py_model = make_house_adapter("house_old", 1, "python")
    py_model.init()
    rows = []
    for k in range(n_steps):
        t = k * dt
        t_out = t_amb_c(t)
        for model in (oc_adapter, py_model):
            model.set_value(["archetype_id", "T_ambient_C", "n_houses"], [1.0, t_out, 400.0])
            model.step(t)
        rows.append({"t_s": t, "T_oct": oc_adapter.get_value(["T_indoor_C"], t)[0],
                     "T_py": py_model.get_value(["T_indoor_C"], t)[0],
                     "Q_oct": oc_adapter.get_value(["Q_heat_kW"], t)[0],
                     "Q_py": py_model.get_value(["Q_heat_kW"], t)[0]})
    oc_adapter.cleanup()
    out = pd.DataFrame(rows)
    out["dT"] = out["T_oct"] - out["T_py"]
    out["dQ"] = out["Q_oct"] - out["Q_py"]
    display(out.round(5))
    print(f"max |dT| = {out['dT'].abs().max():.2e} K, max |dQ| = {out['dQ'].abs().max():.2e} kW")
    return out


_ = compare_house_backends()

# %% [markdown]
# ## 2. The district-heat network
#
# `HeatNetwork` is energysim's district-heat model. Producers inject heat, consumers take it, and pipes
# delay and attenuate it. Dispatch minimises cost, where a producer's cost is its cost divided by its carrier
# efficiency, so a cheap heat pump runs first and the peaker only in the coldest hours.
#
# * **Industrial heat pump** (`hp`): a cheap base-load producer, with its electric input supplied by the
#   `IndustrialHeatPump` model. A *gas-boiler scenario* sets its capacity to zero.
# * **Gas boiler** (`gas`): the fallback base load (`carrier_efficiency = 0.92`, so fuel = heat / 0.92).
# * **Electric boiler** (`eboiler`): the expensive peaker. It is an `ElectricBoiler` with efficiency 1.0,
#   so the electric load equals the heat supplied (an electrode boiler is 99 to 100 % efficient).
# * **Storage** (`tank`): a thermal store that charges at night and discharges at peaks.
#
# The consumers are the three house groups. Each pipe has a delay and a loss fraction. Longer routes
# have larger delays and losses.

# %%
def build_dh_network(hp_capacity_mw):
    dh = HeatNetwork(name="dh", step_size=900.0, supply_T_C=80.0, return_T_C=45.0)
    dh.add_producer("hp", capacity=hp_capacity_mw, cost=20.0, carrier_efficiency=1.0)
    dh.add_producer("gas", capacity=6.0, cost=45.0, carrier_efficiency=0.92)
    dh.add_producer("eboiler", capacity=2.0, cost=90.0, carrier_efficiency=1.0)
    dh.add_storage("tank", capacity_mwh=6.0, p_max=2.0, soc_init=0.5, loss_pct_per_day=1.0)
    dh.add_junction("main")
    for src, cap in (("hp", 6.0), ("gas", 8.0), ("eboiler", 3.0), ("tank", 4.0)):
        dh.add_pipe(src, "main", capacity=cap, loss_pct=0.5, delay_s=120.0)
    dh.add_consumer("old", demand=0.0)
    dh.add_consumer("medium", demand=0.0)
    dh.add_consumer("new", demand=0.0)
    dh.add_pipe("main", "old", capacity=6.0, loss_pct=2.0, delay_s=900.0)
    dh.add_pipe("main", "medium", capacity=6.0, loss_pct=1.5, delay_s=600.0)
    dh.add_pipe("main", "new", capacity=4.0, loss_pct=1.0, delay_s=300.0)
    return dh


dh_probe = build_dh_network(4.0)
print(dh_probe.describe_topology())

# %% [markdown]
# ## 3. The electric feeder (OpenDSS)
#
# The feeder is an 11 kV radial line of four segments written to a `.dss` file. The buses are `src`,
# `n1`, `n2`, `n3` and `n4`. The loads and generators are:
#
# * `base` at `n1`: households and commercial load.
# * `ihp` at `n2`: the individual heat-pump cluster, modelled as an aggregate (see section 4).
# * `pv` at `n3`: the PV cluster at the far end of the feeder, so it can raise the voltage at midday.
# * `dh_hp` and `dh_eb` at `n4`: the central DH heat pump and electric boiler, the large loads.
#
# All loads are set in kW through the OpenDSS adapter, which takes `Type.Name.Property` inputs. The loads
# have zero reactive power to keep the example short. Add `kvar` through the same mechanism if reactive
# power matters for your study.

# %%
FEEDER_DSS = """Clear
New Circuit.heat_feeder basekv=11 pu=1.0 phases=3 bus1=src
New Line.s_n1 bus1=src bus2=n1 phases=3 r1=0.12 x1=0.10 length=1.0 units=km
New Line.n1_n2 bus1=n1 bus2=n2 phases=3 r1=0.12 x1=0.10 length=1.0 units=km
New Line.n2_n3 bus1=n2 bus2=n3 phases=3 r1=0.12 x1=0.10 length=1.0 units=km
New Line.n3_n4 bus1=n3 bus2=n4 phases=3 r1=0.12 x1=0.10 length=0.5 units=km
New Load.base bus1=n1 phases=3 kv=11 kw=600 kvar=0 model=1
New Load.ihp bus1=n2 phases=3 kv=11 kw=0 kvar=0 model=1
New Generator.pv bus1=n3 phases=3 kv=11 kw=0 kvar=0 model=1
New Load.dh_hp bus1=n4 phases=3 kv=11 kw=0 kvar=0 model=1
New Load.dh_eb bus1=n4 phases=3 kv=11 kw=0 kvar=0 model=1
Set voltagebases=[11]
Calcvoltagebases
Solve
"""
DSS_FILE = WORK_DIR / "heat_feeder.dss"
DSS_FILE.write_text(FEEDER_DSS, encoding="utf-8")
print(f"Feeder script written to {DSS_FILE}")

# %% [markdown]
# ## 4. The individual heat-pump cluster
#
# Households outside the district heat are heated by air-to-water heat pumps. Simulating each one is not
# needed for a feeder study, so this notebook uses an **aggregated** cluster. It takes the outdoor
# temperature and returns the electric demand of `N` identical homes. Each home needs
# `UA * (21 - T_out)` kW, and its COP follows a Carnot-type curve for a 35 degC floor-heating supply.
#
# The COP is `0.45 * T_sink / (T_sink - T_source)` in kelvin, limited to 1.5 to 4.5.

# %%
class IndividualHeatPumpCluster(ModelComponent):
    """Aggregated fleet of identical air-to-water heat pumps (electric side only)."""

    units = {"T_ambient_C": "degC", "Q_heat_kW": "kW", "P_elec_kW": "kW", "COP": "1"}

    def __init__(self, name="ihp", n_homes=600, ua_kw_per_k=0.12, t_sink_c=35.0, step_size=900.0, **kwargs):
        super().__init__(name=name, step_size=step_size, **kwargs)
        self.n_homes = int(n_homes)
        self.ua = float(ua_kw_per_k)
        self.t_sink = float(t_sink_c)
        self.register_variables(inputs=["T_ambient_C"],
                                outputs=["Q_heat_kW", "P_elec_kW", "COP"],
                                defaults={"T_ambient_C": -4.0})

    def step_model(self, dt: float, time: float) -> None:
        t_out = float(self._inputs_val.get("T_ambient_C", -4.0))
        q_home = max(0.0, self.ua * (T_SET_C - t_out))
        t_src_k, t_snk_k = t_out + 273.15, self.t_sink + 273.15
        cop = min(4.5, max(1.5, 0.45 * t_snk_k / max(t_snk_k - t_src_k, 5.0)))
        self._outputs_val["Q_heat_kW"] = self.n_homes * q_home
        self._outputs_val["COP"] = cop
        self._outputs_val["P_elec_kW"] = self.n_homes * q_home / cop


# %% [markdown]
# ## 5. Assemble the coupled world
#
# `build_world(central)` assembles the whole case. `central="gas"` keeps the heat pump capacity at zero,
# so the gas boiler carries the base load. `central="hp"` gives the heat pump 4 MW of thermal capacity,
# so it takes the base load and the gas boiler becomes the back-up.
#
# A note on the feeder: the OpenDSS adapter is a process-wide singleton, so the two scenarios run one
# after the other. Each `simulate()` cleans up its own circuit at the end.

# %%
PLANT_INPUTS_FEEDER = ["Load.base.kW", "Load.ihp.kW", "Generator.pv.kW", "Load.dh_hp.kW", "Load.dh_eb.kW"]
FEEDER_OUTPUTS = ["Bus.n4.vm_pu", "Bus.n2.vm_pu", "Circuit.losses_kw"]


def build_world(central, t_end=86400.0, t_macro=900.0):
    """Assemble the coupled DH, house, heat-pump and feeder case."""
    hp_cap = 4.0 if central == "hp" else 0.0
    sim = world(start_time=0, stop_time=t_end, t_macro=t_macro, power_unit="MW")

    # boundary conditions
    sim.add_signal("weather", t_amb_c, step_size=900, units={"y": "degC"})
    sim.add_signal("pv", pv_kw, step_size=900, units={"y": "kW"})
    sim.add_signal("base", base_load_kw, step_size=900, units={"y": "kW"})
    sim.add_signal("t_sink", 80.0, step_size=900, units={"y": "degC"})
    for key, arch in (("old", 1), ("medium", 2), ("new", 3)):
        sim.add_signal(f"arch_{key}", float(arch), step_size=900, units={"y": "1"})
        sim.add_signal(f"n_{key}", float(HOUSE_COUNTS[arch]), step_size=900, units={"y": "1"})

    # house groups (district-heat consumers)
    for key, arch in (("old", 1), ("medium", 2), ("new", 3)):
        sim.add_model(f"house_{key}", make_house_adapter(f"house_{key}", arch, BACKEND), step_size=900,
                      outputs=["T_indoor_C", "Q_heat_kW"],
                      units={"T_ambient_C": "degC", "T_indoor_C": "degC", "Q_heat_kW": "kW",
                             "archetype_id": "1", "n_houses": "1"})

    # district-heat plant
    sim.add_model("dh", build_dh_network(hp_cap), step_size=900,
                  outputs=["mean_supply_T", "total_supply_MW", "total_unserved_MW", "total_losses_MW",
                           "hp.dispatch_MW", "gas.dispatch_MW", "eboiler.dispatch_MW",
                           "tank.soc", "old.supply_T_C", "medium.supply_T_C", "new.supply_T_C"])
    sim.add_model("dh_hp", IndustrialHeatPump(name="dh_hp", q_max_thermal_mw=4.0, carnot_efficiency=0.5,
                                              step_size=900), step_size=900,
                  outputs=["P_elec_MW", "Q_heat_MW", "COP"])
    sim.add_model("dh_eb", ElectricBoiler(name="dh_eb", p_max_mw=2.0, efficiency=1.0), step_size=900,
                  outputs=["P_actual", "Q_thermal_MW"])

    # individual heat pumps (electric side)
    sim.add_model("ihp", IndividualHeatPumpCluster(name="ihp", n_homes=600), step_size=900,
                  outputs=["P_elec_kW", "COP", "Q_heat_kW"])

    conn = {
        "weather.y": ["dh_hp.T_source", "ihp.T_ambient_C", "house_old.T_ambient_C",
                      "house_medium.T_ambient_C", "house_new.T_ambient_C"],
        "t_sink.y": "dh_hp.T_sink",
        "dh.hp.dispatch_MW": "dh_hp.Q_heat_cmd_MW",
        "dh.eboiler.dispatch_MW": "dh_eb.P_cmd",
        "house_old.Q_heat_kW": "dh.old.demand_MW",
        "house_medium.Q_heat_kW": "dh.medium.demand_MW",
        "house_new.Q_heat_kW": "dh.new.demand_MW",
        "arch_old.y": "house_old.archetype_id",
        "arch_medium.y": "house_medium.archetype_id",
        "arch_new.y": "house_new.archetype_id",
        "n_old.y": "house_old.n_houses",
        "n_medium.y": "house_medium.n_houses",
        "n_new.y": "house_new.n_houses",
    }

    if HAS_DSS:
        sim.add_opendss("feeder", model=str(DSS_FILE), inputs=PLANT_INPUTS_FEEDER,
                        outputs=FEEDER_OUTPUTS, step_size=900,
                        power_unit="kW",
                        units={name: "kW" for name in PLANT_INPUTS_FEEDER})
        conn.update({
            "pv.y": "feeder.Generator.pv.kW",
            "base.y": "feeder.Load.base.kW",
            "ihp.P_elec_kW": "feeder.Load.ihp.kW",
            "dh_hp.P_elec_MW": "feeder.Load.dh_hp.kW",
            "dh_eb.P_actual": "feeder.Load.dh_eb.kW",
        })
    sim.add_connections(conn)
    return sim


_probe = build_world("hp")
print(_probe.validate())

# %% [markdown]
# ## 6. Run the two scenarios
#
# Each run is one cold day (24 h) with a 15 min macro step, so about 96 steps. The `simulate()` call
# returns the recorded variables as a `Results` object. The two runs use the same weather, houses and
# loads. Only the central plant differs.

# %%
def run_scenario(central):
    sim = build_world(central)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = sim.simulate(pbar=False)
    return res.to_frame()


if HAS_DSS:
    df_gas = run_scenario("gas")
    df_hp = run_scenario("hp")
    print(f"Scenario 'gas': {len(df_gas)} rows, scenario 'hp': {len(df_hp)} rows")
    compare = pd.DataFrame({
        "min V n4 [pu]": [df_gas["feeder.Bus.n4.vm_pu"].min(), df_hp["feeder.Bus.n4.vm_pu"].min()],
        "max V n4 [pu]": [df_gas["feeder.Bus.n4.vm_pu"].max(), df_hp["feeder.Bus.n4.vm_pu"].max()],
        "mean losses [kW]": [df_gas["feeder.Circuit.losses_kw"].mean(), df_hp["feeder.Circuit.losses_kw"].mean()],
        "gas-boiler heat [MWh]": [df_gas["dh.gas.dispatch_MW"].sum() * 0.25, df_hp["dh.gas.dispatch_MW"].sum() * 0.25],
        "HP heat [MWh]": [df_gas["dh.hp.dispatch_MW"].sum() * 0.25, df_hp["dh.hp.dispatch_MW"].sum() * 0.25],
    }, index=["gas boiler", "heat pump"]).round(4)
    display(compare)
else:
    print("SKIP sections 6 feeder parts (opendssdirect not installed). Running the heat network only.")
    df_gas = None
    df_hp = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sim_dh_only = world(start_time=0, stop_time=86400, t_macro=900, power_unit="MW")
        sim_dh_only.add_signal("weather", t_amb_c, step_size=900, units={"y": "degC"})
        sim_dh_only.add_model("dh", build_dh_network(4.0), step_size=900,
                              outputs=["mean_supply_T", "tank.soc", "hp.dispatch_MW", "gas.dispatch_MW"])
        df_dh_only = sim_dh_only.simulate(pbar=False).to_frame()
    display(df_dh_only.iloc[::12].round(3).head(8))

# %% [markdown]
# ## 7. Results: voltage, losses, district-heat temperatures and storage
#
# The plots compare the two scenarios. Voltage is shown at the far end of the feeder (`n4`) and at `n2`,
# where the individual heat pumps sit. The solid line is the gas-boiler case and the dashed line is the
# heat-pump case.

# %%
if HAS_DSS:
    t_h = df_gas.index.to_numpy() / 3600.0
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    ax = axes[0, 0]
    ax.plot(t_h, df_gas["feeder.Bus.n4.vm_pu"], label="n4 - gas boiler")
    ax.plot(t_h, df_hp["feeder.Bus.n4.vm_pu"], "--", label="n4 - heat pump")
    ax.plot(t_h, df_gas["feeder.Bus.n2.vm_pu"], alpha=0.6, label="n2 - gas boiler")
    ax.plot(t_h, df_hp["feeder.Bus.n2.vm_pu"], "--", alpha=0.6, label="n2 - heat pump")
    ax.set_ylabel("Voltage [pu]")
    ax.set_title("Feeder voltage (sector coupling)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t_h, df_gas["feeder.Circuit.losses_kw"], label="gas boiler")
    ax.plot(t_h, df_hp["feeder.Circuit.losses_kw"], "--", label="heat pump")
    ax.set_ylabel("Feeder losses [kW]")
    ax.set_title("Feeder losses")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t_h, df_hp["dh.mean_supply_T"], label="supply at consumers (mean)")
    ax.plot(t_h, df_hp["dh.new.supply_T_C"], alpha=0.7, label="supply to new houses")
    ax.plot(t_h, df_hp["dh.old.supply_T_C"], alpha=0.7, label="supply to old houses")
    ax.axhline(45.0, color="k", lw=0.8, ls=":", label="nominal return temperature (45 degC)")
    ax.set_ylabel("Temperature [degC]")
    ax.set_xlabel("Time [h]")
    ax.set_title("District-heat temperatures (heat-pump case)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t_h, df_hp["dh.hp.dispatch_MW"], label="HP heat [MW]")
    ax.plot(t_h, df_hp["dh.gas.dispatch_MW"], label="gas boiler heat [MW]")
    ax.plot(t_h, df_hp["dh.eboiler.dispatch_MW"], label="electric boiler [MW]")
    ax.plot(t_h, df_hp["dh.tank.soc"] * 3.0, "k--", label="storage SoC x 3 [-]")
    ax.set_ylabel("Power [MW] / SoC")
    ax.set_xlabel("Time [h]")
    ax.set_title("Heat dispatch and storage use (heat-pump case)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("The return temperature is nominal in this network model (energysim does not track it through the "
          "consumers). Supply temperatures drop with pipe losses, which the plot shows.")
else:
    print("SKIP section 7 (OpenDSS not available).")

# %% [markdown]
# ## 8. What the sector coupling does
#
# Switching the central plant from the gas boiler to the heat pump changes only the electric side: the gas
# boiler has no electric load, while the heat pump draws its electric input at `n4`, the far end of the
# feeder. The size of the effect depends on how the heat pump's electric demand lines up with the PV output,
# the base load and the coldest hours. Read the voltage and loss table above for the numbers, which depend on
# the illustrative parameters. Two checks are worth doing before you trust the answer:
#
# * Run the same day with the storage removed and see how much of the peak the store shifts.
# * Replace the aggregated heat-pump cluster with individual feeder loads, to see how much diversity
#   reduces the peak.

# %% [markdown]
# ## Where to go next
#
# * **15 - Curves and 2-D maps with grid solvers**: inverter volt-var and volt-watt control, a COP map
#   and dynamic line rating.
# * **16 - FMU with built-in physics**: a detailed electrolyser FMU next to the built-in hydrogen chain.
# * **02 - Grid co-simulation with pandapower, PyPSA and OpenDSS**: the basic feeder and grid setup.
# * **07 - Green hydrogen valley**: a larger multi-energy case.

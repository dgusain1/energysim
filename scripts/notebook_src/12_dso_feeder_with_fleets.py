# %% [markdown]
# # 12 - DSO view: an LV feeder with heat-pump, PV and EV fleets
#
# A distribution system operator (DSO) cares about three questions on a low-voltage (LV) feeder:
# **Are the bus voltages inside the limits?** **Is the transformer or a cable overloaded?**
# **How much more heat pumps and rooftop PV can the feeder host before either happens?**
#
# This notebook answers them with energysim, in three steps:
#
# 1. **A pandapower LV feeder.** A 20/0.4 kV transformer (`0.63 MVA 20/0.4 kV`, a standard type
#    from pandapower) feeds eight load buses `B1 ... B8` along a radial `NAYY 4x150 SE` cable. Every
#    element has a unique name, which is how energysim addresses variables (`"<element>.<column>"`).
# 2. **Fleets of homes on every bus.** Each bus has
#    - a *base load* signal (the non-controllable household load),
#    - a compact **heat-pump + 1R1C building fleet** (thermostat, COP as a lookup of outdoor temperature),
#    - a compact **smart-PV fleet** that curtails its output with a **volt-watt** curve on the bus voltage,
#    - a built-in **`BatteryFleetAdapter`** acting as home EV chargers that start charging when the
#      evening plug-in signal rises.
#
#    The fleet totals (kW) are summed into the pandapower load and sgen `p_mw` inputs of their bus.
# 3. **Ensemble study.** We sweep PV penetration, heat-pump penetration and the volt-watt function on/off,
#    and collect the KPIs that matter for a DSO: maximum and minimum voltage, maximum transformer loading and
#    the share of bus-time outside the statutory band 0.95 to 1.05 pu.
#
# ## Equations and assumptions
#
# * **Building (1R1C, per home i):** `C_i dT_i/dt = Q_heat,i - (T_i - T_amb) / R_i`, explicit Euler at the
#   15-minute macro step (`R C` is about 50 to 160 h, so the step is far below the time constant).
# * **Thermostat:** on below `T_set - 0.5 K`, off above `T_set + 0.5 K`, otherwise unchanged.
# * **Heat pump electric power:** `P_el = Q_heat / COP(T_amb)`, with `COP` an illustrative lookup curve.
# * **PV available power:** `P_av = P_peak * G/1000 * [1 + gamma (T_cell - 25)] * PR`.
# * **Volt-watt:** `P_pv = P_av * f(v)`, where `f` equals 1 up to 1.03 pu, falls to 0.6 at 1.06 pu and to 0 at 1.10 pu.
#   The voltage `v` is the bus voltage that pandapower reports at the previous macro step, so the loop has a
#   macro-step delay. The applied factor follows the target with a first-order response of 30 min
#   (`f += (1 - exp(-h/30 min)) (f_target - f)`). Without that response a hard switch with a delay chatters between 0
#   and full output; the 30-min response is an assumption about the aggregate fleet, not a standards value.
# * **EV charging:** a 3.7 kW wall box per car on about 16 % of homes (4 cars per bus), no vehicle-to-grid
#   (`max_discharge = 0`), charging stops automatically at 95 % state of charge. The load profiles are not diversified
#   beyond this, so a larger EV share quickly overloads the 0.63 MVA transformer at night.
# * **Reactive power:** all loads and PV are modelled at unity power factor (`q_mvar = 0`). This is a simplification:
#   reactive power from heat pumps and inverters would raise voltage drops, so the voltage results are optimistic.
# * **Parameters are illustrative.** Replace the cable type, transformer rating, COP curve and load profiles with
#   network and metering data before using the numbers for planning.
#
# All fleet parameters are drawn from fixed random seeds, so every run is reproducible.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

import pandapower as pp

from energysim import world, Ensemble
from energysim.ensemble import Grid
from energysim.container import BatteryFleetAdapter

# %% [markdown]
# ## The fleet models
#
# The two compact fleets below are shared with notebook 14 and are written as `VectorizedContainerAdapter`
# subclasses, so one step updates all homes with NumPy operations. The source of each one lives in
# `scripts/notebook_src/snippets/`. The heat-pump fleet takes a `cop_curve` argument, which notebook 14 uses to
# model a COP that collapses in deep cold.

# %%
#@include snippets/hp_fleet_compact.py

# %%
#@include snippets/smart_pv_compact.py

# %% [markdown]
# ## The LV feeder (pandapower)
#
# The feeder is radial: `MV` (20 kV slack, `Grid`) -> transformer `T1` -> `B1` -> `L1` -> `B2` -> ... -> `B8`.
# Each load bus carries a load `LD_Bn` and a PV generator `PV_Bn`. Both are fed from the fleet totals.
# The network is built fresh for each world, because the pandapower adapter owns the network it is given.

# %%
N_BUS = 8
HOMES_PER_BUS = 25
EV_PER_BUS = 4                       # about 16 % of homes have a home charger (assumed diversity)
TRAFO_TYPE = "0.63 MVA 20/0.4 kV"
CABLE_TYPE = "NAYY 4x150 SE"
LINE_LEN_KM = [0.05] * 7                                    # 50 m segments L1 ... L7 (0.35 km feeder)
BUS_NAMES = [f"B{i}" for i in range(1, N_BUS + 1)]


def build_feeder():
    """Return a fresh pandapower LV feeder with uniquely named elements."""
    net = pp.create_empty_network(name="lv_feeder_12", f_hz=50.0, sn_mva=1.0)
    mv = pp.create_bus(net, vn_kv=20.0, name="MV")
    pp.create_ext_grid(net, mv, vm_pu=1.0, name="Grid")
    buses = [pp.create_bus(net, vn_kv=0.4, name=b) for b in BUS_NAMES]
    pp.create_transformer(net, mv, buses[0], std_type=TRAFO_TYPE, name="T1")
    for k, length in enumerate(LINE_LEN_KM):
        pp.create_line(net, buses[k], buses[k + 1], length_km=length, std_type=CABLE_TYPE, name=f"L{k + 1}")
    for k, bus in enumerate(buses):
        pp.create_load(net, bus, p_mw=0.0, q_mvar=0.0, name=f"LD_{BUS_NAMES[k]}")
        pp.create_sgen(net, bus, p_mw=0.0, q_mvar=0.0, name=f"PV_{BUS_NAMES[k]}")
    return net


net_preview = build_feeder()
display(pd.DataFrame({"element": ["trafo", "lines", "buses", "loads", "sgens"],
                      "count": [len(net_preview.trafo), len(net_preview.line), len(net_preview.bus),
                                len(net_preview.load), len(net_preview.sgen)]}))

# %% [markdown]
# ## Drivers: weather, irradiance, base load and the EV plug-in profile
#
# The day is a clear winter day (outdoor temperature between about -2 and 4 degC, PV peak about 600 W/m2).
# Base load follows a morning and evening shape, and each bus gets its own average load level so that the feeder
# is not perfectly uniform. The EV plug-in signal ramps up after 18:00 and holds until 07:00. Each bus has a
# small arrival offset, so the evening peak is spread out.

# %%
CHARGER_KW = 3.7
HP_QMAX_KW = 3.5                                                   # heat-pump capacity per home [kW th]
rng_drivers = np.random.default_rng(12)
BASE_BUS_KW = 25.0 * rng_drivers.uniform(0.9, 1.1, N_BUS)          # average base load per bus [kW]
EV_OFFSET_H = rng_drivers.uniform(-0.5, 0.5, N_BUS)                # arrival offset per bus [h]
EV_SOC0 = rng_drivers.uniform(0.2, 0.5, (N_BUS, EV_PER_BUS))       # arrival state of charge
U_PV = rng_drivers.random((N_BUS, HOMES_PER_BUS))                   # per-home draws (monotone penetration)
U_HP = rng_drivers.random((N_BUS, HOMES_PER_BUS))


def hour(t):
    return (float(t) / 3600.0) % 24.0


def t_ambient(t):
    return 1.0 + 3.0 * np.cos(2.0 * np.pi * (hour(t) - 15.0) / 24.0)     # degC


def irradiance(t):
    h = hour(t)
    return 600.0 * np.sin(np.pi * (h - 7.0) / 10.0) if 7.0 < h < 17.0 else 0.0   # W/m2


def base_shape(t):
    h = hour(t)
    return 0.6 + 0.4 * np.exp(-0.5 * ((h - 19.0) / 2.5) ** 2) + 0.2 * np.exp(-0.5 * ((h - 8.0) / 2.0) ** 2)


def plug_in_fraction(h):
    if h >= 18.0:
        return min(1.0, (h - 18.0) / 2.0)
    if h < 7.0:
        return 1.0
    return 0.0


def make_plug_signal(offset_h):
    """Charger power command per car [kW] for a bus whose arrivals are shifted by ``offset_h``."""
    def plug_kw(t):
        return CHARGER_KW * plug_in_fraction((hour(t) - offset_h) % 24.0)
    return plug_kw


def make_base_signal(kw):
    return lambda t: float(kw * base_shape(t))


# %% [markdown]
# ## Building the co-simulation
#
# `make_world(pv_pen, hp_pen, volt_watt)` assembles the whole study. Penetration is the share of homes with
# PV (`pv_pen`) or with a heat pump (`hp_pen`). A home without the technology keeps its place in the fleet but
# has zero peak power or zero heat-pump capacity, so changing the penetration never reorders homes.
#
# Connections:
# * `t_amb.y` and `irr.y` are broadcast to every fleet (fan-out lists).
# * `(base, hp, ev)` are **summed** (fan-in tuple) into `feeder.LD_Bn.p_mw`.
# * `pv.total_p_kw` goes to `feeder.PV_Bn.p_mw` (positive = generation).
# * `feeder.Bn.vm_pu` goes back into `pv_Bn.v_pu`: the volt-watt loop closes through the pandapower output.

# %%
def make_world(pv_pen=0.5, hp_pen=0.5, volt_watt=True, t_end_s=24 * 3600, t_macro=900):
    """Return a configured world for one penetration / volt-watt case (not yet simulated)."""
    sim = world(start_time=0, stop_time=t_end_s, t_macro=t_macro)
    sim.add_signal("t_amb", lambda t: float(t_ambient(t)), step_size=900, units={"y": "degC"})
    sim.add_signal("irr", lambda t: float(irradiance(t)), step_size=900, units={"y": "W/m2"})

    connections = {
        "t_amb.y": [f"hp_{b}.t_ambient_degc" for b in BUS_NAMES]
                   + [f"pv_{b}.t_ambient_degc" for b in BUS_NAMES],
        "irr.y": [f"pv_{b}.irradiance_w_m2" for b in BUS_NAMES],
    }

    for k, b in enumerate(BUS_NAMES):
        pv_on = (U_PV[k] < pv_pen).astype(float)
        hp_on = U_HP[k] < hp_pen

        sim.add_signal(f"base_{b}", make_base_signal(BASE_BUS_KW[k]), step_size=900, units={"y": "kW"})
        sim.add_signal(f"plug_{b}", make_plug_signal(EV_OFFSET_H[k]), step_size=900, units={"y": "kW"})

        sim.add_fleet(
            sim_name=f"hp_{b}",
            container=HeatPumpFleet(name=f"hp_{b}", size=HOMES_PER_BUS, rng_seed=100 + k,
                                    q_heat_max_kw=np.where(hp_on, HP_QMAX_KW, 0.0), step_size=900),
            outputs=["total_p_kw", "mean_t_indoor", "min_t_indoor"],
        )
        sim.add_fleet(
            sim_name=f"pv_{b}",
            container=SmartPVFleet(name=f"pv_{b}", size=HOMES_PER_BUS, peak_kw=5.0 * pv_on,
                                   volt_watt=volt_watt, step_size=900),
            outputs=["total_p_kw", "min_curtail_factor"],
        )
        sim.add_fleet(
            sim_name=f"ev_{b}",
            container=BatteryFleetAdapter(name=f"ev_{b}", size=EV_PER_BUS, capacity_kwh=50.0,
                                          max_charge_kw=CHARGER_KW, max_discharge_kw=0.0,
                                          initial_soc=EV_SOC0[k], soc_min=0.1, soc_max=0.95,
                                          efficiency=0.92, step_size=900),
            outputs=["total_actual_p_kw", "mean_soc"],
        )

        connections[(f"base_{b}.y", f"hp_{b}.total_p_kw", f"ev_{b}.total_actual_p_kw")] = f"feeder.LD_{b}.p_mw"
        connections[f"pv_{b}.total_p_kw"] = f"feeder.PV_{b}.p_mw"
        connections[f"feeder.{b}.vm_pu"] = f"pv_{b}.v_pu"
        connections[f"plug_{b}.y"] = f"ev_{b}.p_cmd_kw"

    feeder_outputs = ([f"{b}.vm_pu" for b in BUS_NAMES]
                      + ["T1.loading_percent"] + [f"L{i}.loading_percent" for i in range(1, N_BUS)]
                      + ["Grid.p_mw"])
    sim.add_powerflow(
        network_name="feeder", model=build_feeder(),
        inputs=[f"LD_{b}.p_mw" for b in BUS_NAMES] + [f"PV_{b}.p_mw" for b in BUS_NAMES],
        outputs=feeder_outputs, step_size=900, mode="ac",
    )
    sim.add_connections(connections)
    return sim


# %% [markdown]
# ## One day, base case: 50 % PV, 50 % heat pumps, volt-watt on
#
# The first run shows the physics before the sweep. We plot the voltage at every bus against time (a bus-by-time
# heat map), and the loading of the transformer and of the cables.

# %%
sim_base = make_world(pv_pen=0.5, hp_pen=0.5, volt_watt=True)
res_base = sim_base.simulate(pbar=False)

hours = res_base["feeder.B1.vm_pu"].index / 3600.0
vm_map = np.vstack([res_base[f"feeder.{b}.vm_pu"].values for b in BUS_NAMES])

fig, ax = plt.subplots(figsize=(9, 3.8))
im = ax.imshow(vm_map, aspect="auto", origin="lower", cmap="RdYlBu_r", vmin=0.93, vmax=1.07,
               extent=[hours[0], hours[-1], 0.5, N_BUS + 0.5])
ax.set_yticks(range(1, N_BUS + 1), BUS_NAMES)
ax.set_xlabel("Time of day [h]")
ax.set_ylabel("Feeder bus")
ax.set_title("Voltage profile (bus x time), 50 % PV, 50 % heat pumps, volt-watt on")
cb = fig.colorbar(im, ax=ax)
cb.set_label("Voltage [pu]  (statutory band 0.95 to 1.05)")
plt.tight_layout()
plt.show()

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
ax1.plot(hours, res_base["feeder.T1.loading_percent"], label="T1 transformer", color="C3")
ax1.axhline(100.0, color="k", ls="--", lw=0.8)
ax1.set_xlabel("Time of day [h]")
ax1.set_ylabel("Loading [% of rating]")
ax1.set_title("Transformer loading")
ax1.legend()

line_cols = [f"feeder.L{i}.loading_percent" for i in range(1, N_BUS)]
line_max = pd.concat([res_base[c] for c in line_cols], axis=1).max(axis=1)
ax2.plot(hours, line_max, color="C0", label="worst cable (max over L1..L7)")
ax2.axhline(100.0, color="k", ls="--", lw=0.8)
ax2.set_xlabel("Time of day [h]")
ax2.set_ylabel("Loading [% of rating]")
ax2.set_title("Cable loading")
ax2.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Ensemble study: hosting capacity
#
# The sweep covers PV penetration (0, 50 and 100 % of homes), heat-pump penetration (0, 50 and 100 %) and the
# volt-watt function (off or on): 18 runs. Each run is a fresh world built by `make_world`. To keep the sweep
# quick, the runs use a 30-minute macro step (the power flow dominates the run time, so the 15-minute plots above
# are the reference). `keep='kpis'` keeps only the KPI table, so memory stays small. The KPIs are:
#
# * `v_max_pu`, `v_min_pu`: extreme voltages over all buses and times,
# * `trafo_max_pct`: peak transformer loading,
# * `share_outside`: fraction of bus-time samples outside 0.95 to 1.05 pu (the DSO's time-outside-band KPI).

# %%
V_COLS = [f"feeder.{b}.vm_pu" for b in BUS_NAMES]


def kpi_v_max(r):
    return max(float(r[c].max()) for c in V_COLS)


def kpi_v_min(r):
    return min(float(r[c].min()) for c in V_COLS)


def kpi_trafo_max(r):
    return float(r["feeder.T1.loading_percent"].max())


def kpi_share_outside(r):
    vals = np.concatenate([r[c].values for c in V_COLS])
    return float(np.mean((vals < 0.95) | (vals > 1.05)))


KPIS = {"v_max_pu": kpi_v_max, "v_min_pu": kpi_v_min,
        "trafo_max_pct": kpi_trafo_max, "share_outside": kpi_share_outside}

study = Ensemble(
    world_factory=lambda params, seed: make_world(t_macro=1800, **params),   # 30 min step: keeps the sweep quick
    sampler=Grid({"pv_pen": [0.0, 0.5, 1.0], "hp_pen": [0.0, 0.5, 1.0], "volt_watt": [False, True]}),
    kpis=KPIS,
    keep="kpis",
).run()

hosting = pd.concat([study.params, study.kpis()], axis=1)
display(hosting.round(3))

# %% [markdown]
# ## Hosting-capacity heat maps
#
# Each panel shows one KPI as a function of PV penetration (rows) and heat-pump penetration (columns), with
# volt-watt off on the left and on at the right. Cells above the statutory limit are the ones to read.
# For `share_outside`, 0 is ideal. For `trafo_max_pct`, values above 100 % mean the transformer is overloaded.

# %%
def pivot_kpi(table, kpi, volt_watt):
    sub = table[table["volt_watt"] == volt_watt]
    return sub.pivot(index="pv_pen", columns="hp_pen", values=kpi).sort_index(ascending=True)


fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for col, vw in enumerate([False, True]):
    for row, (kpi, cmap, label) in enumerate([("share_outside", "Reds", "Share of time outside band [-]"),
                                              ("trafo_max_pct", "Oranges", "Peak transformer loading [%]")]):
        ax = axes[row, col]
        piv = pivot_kpi(hosting, kpi, vw)
        im = ax.imshow(piv.values, origin="lower", cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(piv.columns)), [f"{c:.0%}" for c in piv.columns])
        ax.set_yticks(range(len(piv.index)), [f"{r:.0%}" for r in piv.index])
        ax.set_xlabel("Heat-pump penetration")
        ax.set_ylabel("PV penetration")
        ax.set_title(f"{'Volt-watt on' if vw else 'Volt-watt off'}")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.2f}" if kpi == "share_outside" else f"{piv.values[i, j]:.0f}",
                        ha="center", va="center", color="k")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(label)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## What to read from the results
#
# * With **volt-watt off**, high PV penetration is expected to push the far-end buses above 1.05 pu around
#   midday, because reverse power flow lifts the voltage along the feeder. This is the usual constraint on rooftop PV.
#   The `share_outside` column shows where it happens.
# * With **volt-watt on**, the inverters should curtail as the voltage rises, which should lower `share_outside` at
#   high PV. The price is curtailed energy. That is not in the KPIs here, but an `available - delivered` energy KPI
#   would show it.
# * **Heat pumps and EV charging coincide in the evening.** The transformer peak is expected to move most with
#   heat-pump penetration, because the night-time load is the largest.
#
# Caveats: unity power factor, a single 24 h winter day, one heat-pump model, a 30-minute step in the sweep, and a
# one-step delay in the volt-watt loop (damped by a 30-minute fleet response). Use this study to compare options and
# to find the order of magnitude of the limit, not as a final planning number.

# %% [markdown]
# ## Where to go next
#
# * **Notebook 14** couples the power system with a gas network during a multi-day cold snap, where the heat-pump
#   fleet and its COP collapse become a system question.
# * **Notebook 03** shows how to write your own `ModelComponent`, and **notebook 06** how the vectorized fleets scale
#   to tens of thousands of units.
# * For a feeder with real measurements, replace the synthetic profiles with smart-meter data and add reactive power
#   (`q_mvar`) from heat pumps and inverters. The files in `snippets/` can be copied into your project as they are.

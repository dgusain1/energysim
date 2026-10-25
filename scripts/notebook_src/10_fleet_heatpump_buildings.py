# %% [markdown]
# # 10 - A Fleet of 2,000 Heat-Pump Houses with Real Thermal Physics
#
# The built-in `BuildingFleetAdapter` is a lumped 1R1C envelope with no heating equipment. Here we
# subclass `VectorizedContainerAdapter` and build a fleet where **every house carries its own physics**:
#
# 1. **2R2C building**: an air node (fast, a few hours) and an envelope mass node (slow, a day or more)
#    coupled by $H_{am}$, with solar gains through a per-house glazing aperture.
# 2. **Air-source heat pump** with COP and heating-capacity maps stored as `Surface2D(T_source, T_sink)`,
#    a weather-compensated supply temperature (heating curve) and a defrost derate `Curve1D(T_amb)`.
# 3. **Thermostat hysteresis with minimum on/off timers** (anti short-cycling), all as vectorized masks.
# 4. **Electric resistance backup** that only runs while the heat pump is on and the house is still cold.
# 5. **Night setback** with a random set point per house, plus a **demand-response (DR) lock** that
#    switches off a fixed random subset of houses.
#
# Everything is NumPy over the N houses, so 2,000 homes for a week run in seconds.
#
# **Caveat up front:** the COP and capacity maps below are *Carnot-fraction based* (illustrative
# physics, not a product). They are generated from $\eta_c \, T_{sink}/(T_{sink}-T_{source})$ and a simple
# capacity slope. Replace the two grids with EN 14511 test points for a real heat pump.

# %% [markdown]
# ## Step 0: Imports and constants

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from energysim import world
from energysim.container import VectorizedContainerAdapter, BuildingFleetAdapter
from energysim.models.common import Curve1D, Surface2D

N_HOMES = 2000          # houses in the fleet
SEED = 2024             # every random draw is seeded
HOUR = 3600.0
DAY = 86400.0
STEP_S = 300.0          # fleet micro-step (s): resolves the 10 min minimum on/off timers
MACRO_S = 900           # world macro step (s): signals and recorded outputs
print(f"Fleet size {N_HOMES:,} homes, micro-step {STEP_S:.0f} s, macro-step {MACRO_S} s")

# %% [markdown]
# ## Step 1: Heat-pump maps as `Surface2D`
#
# The COP follows a Carnot fraction, $\mathrm{COP} = \eta_c \, T_{sink}/(T_{sink}-T_{source})$ with
# $\eta_c = 0.40$ (temperatures in kelvin). The heating capacity of an 8 kW reference unit falls as the air
# gets colder (about 2.5 % per K below 7 degC) and as the supply temperature rises (0.6 % per K above 35 degC).
# Both maps are bilinear lookups, and `Surface2D` accepts array arguments, so one call evaluates the whole fleet
# at its own supply temperature.

# %%
T_SRC_GRID = np.arange(-20.0, 20.01, 2.5)      # air (source) temperature, degC
T_SINK_GRID = np.arange(25.0, 60.01, 5.0)      # condenser (supply) temperature, degC
ETA_CARNOT = 0.40                              # Carnot fraction (placeholder for test data)
CAP_REF_KW = 8.0                               # reference unit capacity at A7/W35 (kW)


def build_hp_maps():
    """Return (COP surface, capacity surface in kW for the reference unit)."""
    ts, tk = np.meshgrid(T_SRC_GRID, T_SINK_GRID, indexing="ij")
    cop = np.clip(ETA_CARNOT * (tk + 273.15) / np.maximum(tk - ts, 1.0), 1.0, 6.0)
    cap = np.clip(CAP_REF_KW * (1.0 + 0.025 * (ts - 7.0)) * (1.0 - 0.006 * (tk - 35.0)), 2.0, 10.0)
    return (Surface2D(T_SRC_GRID, T_SINK_GRID, cop, name="COP (Carnot fraction)"),
            Surface2D(T_SRC_GRID, T_SINK_GRID, cap, name="capacity kW, 8 kW reference"))


COP_MAP, CAP_MAP = build_hp_maps()
DEFROST = Curve1D(x=[-15.0, -7.0, 0.0, 2.0, 7.0, 20.0],
                  y=[0.85, 0.88, 0.93, 0.98, 1.00, 1.00], name="defrost derate vs T_amb")

# %% [markdown]
# Plot the two surfaces. The COP rises as the air warms and the supply temperature falls. The capacity
# shrinks in the cold, which is what forces the resistance backup in the cold snap later on.

# %%
ts_mesh, tk_mesh = np.meshgrid(T_SRC_GRID, T_SINK_GRID, indexing="ij")
cop_vals = COP_MAP(ts_mesh.ravel(), tk_mesh.ravel()).reshape(ts_mesh.shape)
cap_vals = CAP_MAP(ts_mesh.ravel(), tk_mesh.ravel()).reshape(ts_mesh.shape)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, z, title, label in [
    (axes[0], cop_vals, "COP map (Surface2D)", "COP (-)"),
    (axes[1], cap_vals, "Heating capacity, 8 kW reference unit", "capacity (kW)"),
]:
    mesh = ax.pcolormesh(T_SINK_GRID, T_SRC_GRID, z, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label=label)
    ax.set_xlabel("sink (supply) temperature (degC)")
    ax.set_ylabel("source (air) temperature (degC)")
    ax.set_title(title)

# COP versus air temperature at two supply temperatures (interpolated from the surface)
t_air_line = np.linspace(-15.0, 15.0, 61)
for t_sink, style in [(35.0, "-"), (50.0, "--")]:
    axes[2].plot(t_air_line, COP_MAP(t_air_line, t_sink), style, label=f"supply {t_sink:.0f} degC")
axes[2].set_xlabel("air temperature (degC)")
axes[2].set_ylabel("COP (-)")
axes[2].set_title("COP vs air temperature")
axes[2].grid(alpha=0.3)
axes[2].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 2: The fleet class `HPHouseFleet`
#
# Per-house parameters are drawn once from a seeded RNG and stored as plain arrays. The state vector holds
# the two house temperatures, the heat-pump and backup status, the anti-short-cycling timer, and running
# totals. The outputs are the per-house states, so the world records `total_`, `mean_`, `min_` aggregations.
#
# **Thermal equations** (kW, kWh/K, K):
#
# $$C_a \frac{dT_a}{dt} = H_{am}(T_m - T_a) + U_a (T_{amb} - T_a) + Q_{hp} + Q_{bk} + (1-f)\,Q_{sol}$$
#
# $$C_m \frac{dT_m}{dt} = -H_{am}(T_m - T_a) + U_m (T_{amb} - T_m) + f\,Q_{sol}$$
#
# with $U_a = 0.3\,UA$ (windows and infiltration) and $U_m = 0.7\,UA$ (opaque envelope), and $Q_{sol} = A\,g\,G/1000$.
#
# **Control, in order each step**: (a) night setback plus pre-heat shift gives the set point; (b) hysteresis
# proposes on or off; (c) minimum on and off timers may veto a switch; (d) the DR lock forces the unit off
# (the lock overrides the timers, as a grid block does); (e) the backup runs only while the heat pump is on
# and the house is still below the comfort floor.

# %%
class HPHouseFleet(VectorizedContainerAdapter):
    """N houses: 2R2C envelope, air-source heat pump, resistance backup, thermostat with timers."""

    units = {
        "t_amb_degc": "degC", "solar_w_m2": "W/m2", "setpoint_shift_k": "K", "dr_block_frac": "pu",
        "t_air_degc": "degC", "t_mass_degc": "degC", "p_el_kw": "kW", "q_heat_kw": "kW",
        "cop": "pu", "comfort_violation": "pu", "hp_on": "pu", "backup_on": "pu",
    }

    def __init__(self, name="homes", size=N_HOMES, step_size=STEP_S, seed=SEED):
        super().__init__(name=name, size=size, step_size=step_size)
        rng = np.random.default_rng(seed)
        n = self.size

        # Building (per house)
        self.ua = rng.uniform(0.20, 0.35, n)          # envelope conductance to ambient (kW/K)
        self.air_share = 0.3                          # share of UA hitting the air node directly
        self.c_air = rng.uniform(1.0, 2.0, n)         # air + light furniture (kWh/K), fast node
        self.c_mass = rng.uniform(8.0, 20.0, n)       # structure (kWh/K), slow node
        self.h_am = rng.uniform(0.8, 1.4, n)          # air <-> structure coupling (kW/K)
        self.aperture = rng.uniform(2.0, 5.0, n)      # effective glazing aperture (m2)
        self.g_value = 0.5                            # solar heat gain coefficient
        self.f_solar_mass = 0.5                       # share of solar gain absorbed by the structure

        # Heat pump and backup (per house)
        self.q_rated = rng.uniform(7.0, 10.0, n)      # rated capacity scale (kW, A7/W35)
        self.flow_offset = rng.uniform(-3.0, 3.0, n)  # radiator sizing spread (K)
        self.backup_kw = rng.uniform(4.0, 6.0, n)     # resistance element, COP 1

        # Thermostat, schedule and DR priority (per house)
        self.sp_day = rng.uniform(20.0, 21.5, n)      # comfort set point (degC)
        self.setback_k = rng.uniform(2.0, 4.0, n)     # night setback (K)
        self.deadband = rng.uniform(0.4, 0.8, n)      # hysteresis band (K)
        self.min_on_s = 600.0                         # minimum run time (s)
        self.min_off_s = 600.0                        # minimum off time (s)
        self.prio = rng.random(n)                     # fixed DR priority: locked when prio < block fraction

        # States
        self.register_state("t_air_degc", rng.uniform(19.5, 21.0, n))
        self.register_state("t_mass_degc", rng.uniform(19.0, 20.5, n))
        self.register_state("hp_on", 0.0)
        self.register_state("backup_on", 0.0)
        self.register_state("t_since_switch", 1.0e4)     # large: the first switch is allowed at once
        self.register_state("starts", 0.0)               # cumulative compressor starts
        self.register_state("p_el_kw", 0.0)
        self.register_state("q_heat_kw", 0.0)
        self.register_state("cop", 0.0)                  # map COP x defrost (reported even when off)
        self.register_state("comfort_violation", 0.0)
        self.register_state("comfort_deficit_kh", 0.0)   # cumulative degree-hours below the comfort floor

        # Inputs (scalars broadcast to all houses)
        self.register_input("t_amb_degc", 0.0)
        self.register_input("solar_w_m2", 0.0)
        self.register_input("setpoint_shift_k", 0.0)     # pre-heat, K
        self.register_input("dr_block_frac", 0.0)        # fraction of houses locked off

    def step(self, time, step_size=None):
        dt = float(step_size) if step_size is not None else self.step_size
        if dt <= 0.0:
            return
        dt_h = dt / 3600.0
        t_amb = self.input_values["t_amb_degc"]
        solar = np.maximum(0.0, self.input_values["solar_w_m2"])
        shift = self.input_values["setpoint_shift_k"]
        locked = self.prio < self.input_values["dr_block_frac"]

        # (a) schedule: night setback, then pre-heat on top
        hour = (time / 3600.0) % 24.0
        night = (hour >= 22.0) | (hour < 6.0)
        sp_sched = self.sp_day - np.where(night, self.setback_k, 0.0)
        sp = sp_sched + shift
        comfort_lo = sp_sched - 1.5                      # comfort floor ignores pre-heat

        t_air = self.states["t_air_degc"]
        t_mass = self.states["t_mass_degc"]
        hp_prev = self.states["hp_on"] > 0.5

        # (b, c, d) hysteresis, minimum on/off timers, DR lock
        below = t_air < sp - 0.5 * self.deadband
        above = t_air > sp + 0.5 * self.deadband
        want = np.where(below, True, np.where(above, False, hp_prev))
        elapsed = self.states["t_since_switch"] + dt
        allowed = np.where(hp_prev, elapsed >= self.min_on_s, elapsed >= self.min_off_s)
        hp_new = np.where(allowed, want, hp_prev) & ~locked
        switched = hp_new != hp_prev
        self.states["starts"] += (switched & hp_new).astype(float)
        self.states["t_since_switch"][:] = np.where(switched, 0.0, np.minimum(elapsed, 1.0e4))
        self.states["hp_on"][:] = hp_new.astype(float)

        # Heat pump: weather-compensated supply temperature, maps, defrost derate
        t_flow = np.clip(20.5 + 1.1 * (20.5 - t_amb) + 2.0 + self.flow_offset, 30.0, 55.0)
        derate = DEFROST(t_amb)
        cop = np.maximum(COP_MAP(t_amb, t_flow) * derate, 1.0)
        q_cap = self.q_rated / CAP_REF_KW * CAP_MAP(t_amb, t_flow) * derate
        q_hp = np.where(hp_new, q_cap, 0.0)
        p_hp = q_hp / cop

        # (e) resistance backup: heat pump running and still below the comfort floor
        bk_prev = self.states["backup_on"] > 0.5
        bk_want = np.where(t_air < comfort_lo, True, np.where(t_air > comfort_lo + 0.5, False, bk_prev))
        bk_new = bk_want & hp_new & ~locked
        q_bk = np.where(bk_new, self.backup_kw, 0.0)
        self.states["backup_on"][:] = bk_new.astype(float)

        # Two-node thermal update, explicit Euler (both derivatives use the old state)
        q_sol = self.aperture * self.g_value * solar / 1000.0
        ua_air = self.air_share * self.ua
        ua_mass = (1.0 - self.air_share) * self.ua
        q_mass_to_air = self.h_am * (t_mass - t_air)
        d_air = (q_mass_to_air + ua_air * (t_amb - t_air) + q_hp + q_bk
                 + (1.0 - self.f_solar_mass) * q_sol) / self.c_air
        d_mass = (-q_mass_to_air + ua_mass * (t_amb - t_mass) + self.f_solar_mass * q_sol) / self.c_mass
        t_air_new = t_air + d_air * dt_h
        t_mass_new = t_mass + d_mass * dt_h

        self.states["t_air_degc"][:] = t_air_new
        self.states["t_mass_degc"][:] = t_mass_new
        self.states["q_heat_kw"][:] = q_hp + q_bk
        self.states["p_el_kw"][:] = p_hp + q_bk              # resistance heat counts 1:1 as electricity
        self.states["cop"][:] = cop
        self.states["comfort_violation"][:] = (t_air_new < comfort_lo).astype(float)
        self.states["comfort_deficit_kh"][:] += np.maximum(0.0, comfort_lo - t_air_new) * dt_h


OUTPUTS = ["total_p_el_kw", "total_q_heat_kw", "mean_t_air_degc", "min_t_air_degc", "mean_cop",
           "total_hp_on", "total_backup_on", "total_starts", "total_comfort_violation",
           "total_comfort_deficit_kh"]


def make_homes(n=N_HOMES, seed=SEED):
    return HPHouseFleet(name="homes", size=n, step_size=STEP_S, seed=seed)

# %% [markdown]
# ## Step 3: Weather, solar and the control signals
#
# The synthetic week is a cold period with a diurnal cycle (coldest near 03 h, warmest near 15 h) and a
# cold snap centred on day 4. Solar gain follows a clear-sky sine shape scaled by a daily clearness index.
# The pre-heat (+1.5 K, 15-17 h) and the DR block (30 % of houses, 17-20 h) are signals that only the DR
# run uses.

# %%
DAILY_CLEARNESS = np.array([0.45, 0.60, 0.25, 0.10, 0.50, 0.70, 0.55])


def ambient_degc(t):
    """Synthetic ambient: diurnal cycle around -3 degC plus a cold snap centred on day 4."""
    h = (t / HOUR) % 24.0
    day = t / DAY
    diurnal = 3.0 * np.sin(2.0 * np.pi * (h - 9.0) / 24.0)
    snap = -7.0 * np.exp(-0.5 * ((day - 3.5) / 0.6) ** 2)
    return -3.0 + diurnal + snap


def solar_w_m2(t):
    """Horizontal irradiance: sunrise 07 h, sunset 17 h, scaled by the day's clearness."""
    h = (t / HOUR) % 24.0
    day_idx = min(int(t // DAY), len(DAILY_CLEARNESS) - 1)
    sun = np.maximum(0.0, np.sin(np.pi * (h - 7.0) / 10.0))
    return 650.0 * DAILY_CLEARNESS[day_idx] * sun


def preheat_k(t):
    """Set point pre-heat: +1.5 K between 15 h and 17 h."""
    h = (t / HOUR) % 24.0
    return 1.5 if 15.0 <= h < 17.0 else 0.0


def dr_block(t):
    """Demand response: 30 % of heat pumps locked off between 17 h and 20 h."""
    h = (t / HOUR) % 24.0
    return 0.30 if 17.0 <= h < 20.0 else 0.0


t_grid = np.arange(0.0, 7.0 * DAY, 900.0)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
ax1.plot(t_grid / DAY, [ambient_degc(t) for t in t_grid], color="tab:blue")
ax1.set_ylabel("ambient (degC)")
ax1.set_title("Synthetic cold week: diurnal cycle and a cold snap on day 4")
ax1.grid(alpha=0.3)
ax2.plot(t_grid / DAY, [solar_w_m2(t) for t in t_grid], color="goldenrod")
ax2.set_ylabel("horizontal irradiance (W/m2)")
ax2.set_xlabel("time (days)")
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 4: Wire the fleet into a world
#
# `add_fleet` overwrites the adapter step size with its own `step_size` argument (default 900 s), so we pass
# `STEP_S` explicitly. The signals use the macro step, and the four fleet inputs are connected to them.

# %%
def run_fleet(n=N_HOMES, dr=False, days=7, seed=SEED):
    """Simulate n houses for `days` days; dr=True adds the pre-heat and DR block."""
    sim = world(start_time=0, stop_time=days * DAY, t_macro=MACRO_S)
    sim.add_signal("t_amb", lambda t: ambient_degc(t), step_size=MACRO_S, units={"y": "degC"})
    sim.add_signal("solar", lambda t: solar_w_m2(t), step_size=MACRO_S, units={"y": "W/m2"})
    sim.add_signal("preheat", lambda t: preheat_k(t) if dr else 0.0, step_size=MACRO_S, units={"y": "K"})
    sim.add_signal("dr", lambda t: dr_block(t) if dr else 0.0, step_size=MACRO_S, units={"y": "pu"})
    sim.add_fleet(sim_name="homes", container=make_homes(n, seed), step_size=STEP_S, outputs=OUTPUTS)
    sim.add_connections({
        "t_amb.y": "homes.t_amb_degc",
        "solar.y": "homes.solar_w_m2",
        "preheat.y": "homes.setpoint_shift_k",
        "dr.y": "homes.dr_block_frac",
    })
    return sim.simulate(pbar=False)


def energy_kwh(series):
    """Integrate a kW series over its time index (hours) with the trapezoid rule."""
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(series.to_numpy(float), series.index.to_numpy(float) / HOUR))


def hourly_profile(series):
    """Average a series over the 7 days by hour of day (index 0.0 ... 23.0)."""
    hours = np.floor((series.index.to_numpy(float) / HOUR) % 24.0)
    return pd.Series(series.to_numpy(float)).groupby(hours).mean()

# %% [markdown]
# ## Step 5: Base week (no demand response)
#
# The top panel shows the aggregate electric load of 2,000 houses. It has the daily evening peak, and it
# jumps during the cold snap as the backup elements join in. The bottom panel shows the indoor temperature
# envelope, which tracks the set points, and the fraction of houses below their comfort floor.

# %%
res_base = run_fleet(N_HOMES, dr=False)
t_h = res_base["homes.total_p_el_kw"].index.to_numpy(float) / HOUR / 24.0    # days

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
ax1.plot(t_h, res_base["homes.total_p_el_kw"] / 1000.0, color="tab:red", label="fleet electric load")
ax1.set_ylabel("electric load (MW)")
ax1.set_title(f"Base week, {N_HOMES:,} heat-pump houses")
ax1.grid(alpha=0.3)
ax1b = ax1.twinx()
ax1b.plot(t_h, [ambient_degc(t) for t in res_base["homes.total_p_el_kw"].index.to_numpy(float)],
          color="grey", alpha=0.5, label="ambient")
ax1b.set_ylabel("ambient (degC)")

ax2.plot(t_h, res_base["homes.mean_t_air_degc"], color="tab:blue", label="mean indoor air")
ax2.plot(t_h, res_base["homes.min_t_air_degc"], color="tab:cyan", alpha=0.7, label="coldest house")
ax2.set_ylabel("indoor air (degC)")
ax2.set_xlabel("time (days)")
ax2.grid(alpha=0.3)
ax2b = ax2.twinx()
ax2b.plot(t_h, res_base["homes.total_backup_on"] / N_HOMES * 100.0, color="tab:purple", alpha=0.7,
          label="houses on backup")
ax2b.set_ylabel("houses on resistance backup (%)")
ax1.legend(loc="upper left")
ax2.legend(loc="lower left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 6: Diversity: peak per home versus fleet size
#
# Each house has its own set point, envelope, and sun exposure, so their peaks do not coincide. The coincident
# peak of the fleet grows more slowly than N, so the peak per home falls. Each size draws its own houses from the
# same seed, so the curve mixes fleet-size and sampling effects; the trend is the point.

# %%
diversity_rows = []
for n_sz in [50, 200, 800, N_HOMES]:
    res_n = res_base if n_sz == N_HOMES else run_fleet(n_sz, dr=False)
    p_n = res_n["homes.total_p_el_kw"]
    diversity_rows.append({
        "homes N": n_sz,
        "coincident peak (kW)": round(float(p_n.max()), 1),
        "peak per home (kW)": round(float(p_n.max()) / n_sz, 3),
        "mean per home (kW)": round(float(p_n.mean()) / n_sz, 3),
    })
diversity = pd.DataFrame(diversity_rows)
display(diversity)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(diversity["homes N"], diversity["peak per home (kW)"], "o-", label="peak per home")
ax.plot(diversity["homes N"], diversity["mean per home (kW)"], "s--", label="mean per home")
ax.set_xscale("log")
ax.set_xlabel("homes in fleet N (log scale)")
ax.set_ylabel("electric power per home (kW)")
ax.set_title("Diversity: the coincident peak per home falls as N grows")
ax.grid(alpha=0.3, which="both")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 7: Demand response: pre-heat, block, and rebound
#
# The DR run adds two actions. Between 15 h and 17 h the set point rises by 1.5 K, which stores heat in
# the structure before the peak. Between 17 h and 20 h the heat pumps of 30 % of the houses are locked off.
# The priority is fixed per house, so the locked set is nested as the fraction grows. The locked houses
# drift cold and then recover after the block, which creates a **rebound** load.
#
# The table compares the average day-profile in three windows. The comfort cost counts home-hours below
# the comfort floor and the cumulative degree-hours below it.

# %%
res_dr = run_fleet(N_HOMES, dr=True)
prof_base = hourly_profile(res_base["homes.total_p_el_kw"]) / 1000.0    # MW, average day
prof_dr = hourly_profile(res_dr["homes.total_p_el_kw"]) / 1000.0

summary = pd.DataFrame({
    "window": ["pre-heat 15-17 h", "block 17-20 h", "rebound 20-24 h"],
    "base energy (MWh/day)": [prof_base.loc[15:16].sum(), prof_base.loc[17:19].sum(), prof_base.loc[20:23].sum()],
    "DR energy (MWh/day)": [prof_dr.loc[15:16].sum(), prof_dr.loc[17:19].sum(), prof_dr.loc[20:23].sum()],
})
summary["change (MWh/day)"] = summary["DR energy (MWh/day)"] - summary["base energy (MWh/day)"]
display(summary.round(2))

peak_base_mw = float(prof_base.loc[17:19].max())
peak_dr_mw = float(prof_dr.loc[17:19].max())
viol_base = energy_kwh(res_base["homes.total_comfort_violation"]) / N_HOMES
viol_dr = energy_kwh(res_dr["homes.total_comfort_violation"]) / N_HOMES
deficit_base = float(res_base["homes.total_comfort_deficit_kh"].iloc[-1]) / N_HOMES
deficit_dr = float(res_dr["homes.total_comfort_deficit_kh"].iloc[-1]) / N_HOMES
print(f"Block-window peak: base {peak_base_mw:.2f} MW, DR {peak_dr_mw:.2f} MW "
      f"({100 * (1 - peak_dr_mw / peak_base_mw):.1f} % lower)")
print(f"Comfort cost: home-hours below comfort per home, base {viol_base:.2f} h, DR {viol_dr:.2f} h")
print(f"Comfort cost: degree-hours below the floor per home, base {deficit_base:.2f} K h, DR {deficit_dr:.2f} K h")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
ax1.plot(prof_base.index + 0.5, prof_base.values, "o-", label="base")
ax1.plot(prof_dr.index + 0.5, prof_dr.values, "s-", label="DR (pre-heat + block)")
ax1.axvspan(15, 17, color="orange", alpha=0.15, label="pre-heat")
ax1.axvspan(17, 20, color="grey", alpha=0.2, label="30 % block")
ax1.set_xlabel("hour of day")
ax1.set_ylabel("average electric load (MW)")
ax1.set_title("Average day: shifted peak and rebound")
ax1.grid(alpha=0.3)
ax1.legend()

ax2.bar(["base", "DR"], [deficit_base, deficit_dr], color=["tab:blue", "tab:orange"])
ax2.set_ylabel("degree-hours below comfort floor per home (K h)")
ax2.set_title("Comfort cost of the DR action")
ax2.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 8: Compact comparison with the built-in `BuildingFleetAdapter`
#
# The built-in fleet is a 1R1C envelope that accepts a heat input directly. It has no heat pump, no
# cycling and no comfort logic of its own. To compare, we give it the same houses (combined capacitance,
# $R = 1/UA$, same solar aperture) and an **ideal** proportional heater: $\dot{Q} = \mathrm{clip}(5\,(T_{sp}-T_{in}), 0, 10)$ kW.
# That heater is a resistive, COP 1 abstraction, so its electricity equals its heat. The table shows what
# the heat-pump fleet adds on top of the envelope: electricity that depends on COP, compressor starts,
# resistance backup, and the comfort cost of a real controller. The built-in model is driven directly
# through `step()`, so it does not need a world.

# %%
IDEAL_GAIN_KW_K = 5.0
IDEAL_MAX_KW = 10.0


def run_builtin_ideal(fleet, n_steps, dt=STEP_S):
    """Drive a BuildingFleetAdapter with the houses' parameters and an ideal proportional heater."""
    bld = BuildingFleetAdapter(name="builtin", size=fleet.size, step_size=dt,
                               c_th_kwh_per_k=fleet.c_air + fleet.c_mass,
                               r_th_k_per_kw=1.0 / fleet.ua,
                               initial_t_indoor=fleet.sp_day)
    q_hist = np.zeros(n_steps)
    t_mean = np.zeros(n_steps)
    deficit_kh = 0.0
    for k in range(n_steps):
        t = k * dt
        hour = (t / HOUR) % 24.0
        night = (hour >= 22.0) | (hour < 6.0)
        sp = fleet.sp_day - np.where(night, fleet.setback_k, 0.0)
        t_in = bld.states["t_indoor"]
        q = np.clip(IDEAL_GAIN_KW_K * (sp - t_in), 0.0, IDEAL_MAX_KW)
        bld.set_value(["t_ambient", "q_heat_kw", "q_solar_kw"],
                      [ambient_degc(t), q, fleet.aperture * fleet.g_value * solar_w_m2(t) / 1000.0])
        bld.step(t, dt)
        deficit_kh += float(np.sum(np.maximum(0.0, sp - 1.5 - bld.states["t_indoor"]))) / fleet.size * dt / HOUR
        q_hist[k] = q.sum()
        t_mean[k] = bld.states["t_indoor"].mean()
    return q_hist, t_mean, deficit_kh


n_steps = int(7 * DAY / STEP_S)
q_builtin, t_builtin, deficit_builtin = run_builtin_ideal(make_homes(N_HOMES), n_steps)

energy_builtin = float(q_builtin.sum()) * STEP_S / HOUR / N_HOMES
energy_hp_el = energy_kwh(res_base["homes.total_p_el_kw"]) / N_HOMES
energy_hp_th = energy_kwh(res_base["homes.total_q_heat_kw"]) / N_HOMES
comparison = pd.DataFrame({
    "built-in 1R1C + ideal heater": [energy_builtin, energy_builtin,
                                     q_builtin.max() / N_HOMES, 0.0, deficit_builtin],
    "heat-pump houses (this notebook)": [energy_hp_th, energy_hp_el,
                                         float(res_base["homes.total_p_el_kw"].max()) / N_HOMES,
                                         float(res_base["homes.total_starts"].iloc[-1]) / N_HOMES,
                                         deficit_base],
}, index=["heat delivered (kWh/home/week)", "electricity (kWh/home/week)",
          "peak per home (kW)", "compressor starts per home", "comfort deficit (K h/home)"])
display(comparison.round(2))

fig, ax = plt.subplots(figsize=(11, 4))
t_days = np.arange(n_steps) * STEP_S / DAY
ax.plot(res_base["homes.total_p_el_kw"].index.to_numpy(float) / DAY,
        res_base["homes.total_p_el_kw"] / N_HOMES, color="tab:red", label="heat-pump houses: electricity per home")
ax.plot(t_days, q_builtin / N_HOMES, color="tab:green", alpha=0.8, label="built-in 1R1C + ideal heater")
ax.set_xlabel("time (days)")
ax.set_ylabel("power per home (kW)")
ax.set_title("Same houses, two models: the heat-pump fleet cycles and falls back to resistance")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Where to go next
#
# - **Notebook 06** shows the plain vectorized battery and PV containers that this fleet extends.
# - **Notebook 04** explains how to build a custom adapter from scratch, when the container pattern above is not enough.
# - **Notebook 11** applies the same subclassing pattern to a PV fleet with smart inverters and cloud transients.
# - **Notebook 12** closes the loop: the PV fleet is placed on a real pandapower feeder to see voltage and
#   curtailment interact with the grid.
#
# To make this model realistic, replace `COP_MAP` and `CAP_MAP` with EN 14511 test points, fit the
# envelope parameters to metered data, and check the timers against your controller's specification.

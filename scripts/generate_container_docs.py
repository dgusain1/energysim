# -*- coding: utf-8 -*-
"""
Generate Vectorized Fleet Container documentation pages, including full tutorial
on how users can build their own custom vectorized fleet containers.
"""

import os

docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))
containers_dir = os.path.join(docs_dir, "containers")
os.makedirs(containers_dir, exist_ok=True)

# 1. overview.md
overview_content = """# Vectorized Fleet Containers

Large-scale co-simulations involving tens of thousands of Distributed Energy Resources (DERs) — such as rooftop solar PV, residential batteries, electric vehicles, and smart thermostats — typically suffer severe performance bottlenecks when implemented as individual interpreted Python objects. 

`energysim` solves this via **Vectorized Container Simulators** (`VectorizedContainerAdapter`).

---

## Core Architecture

Instead of instantiating 10,000 separate model objects and dispatching 10,000 Python method calls every micro time-step, a **Vectorized Container** stores the states and parameters of all $N$ assets in **contiguous 1-D C-memory NumPy arrays**:

```mermaid
flowchart LR
    subgraph Traditional["Traditional Multi-Agent / Object Co-Sim"]
        Obj1["Battery 1 (Python Obj)"] --> Step1["step()"]
        Obj2["Battery 2 (Python Obj)"] --> Step2["step()"]
        ObjN["Battery 10,000 (Python Obj)"] --> StepN["step()"]
        Note1["10,000 interpreted function dispatches per time-step (Seconds/step)"]
    end

    subgraph Vectorized["energysim Vectorized Container"]
        VecArr["States Array: SoC[N], P_cmd[N], Cap[N] (C-contiguous NumPy)"]
        SIMD["Single Vectorized BLAS/SIMD Step (Microseconds/step)"]
        VecArr --> SIMD
    end
```

### Key Architectural Capabilities

1. **Scalar Broadcast & Vector Control**:
   - Setting a scalar input (e.g. `ambient_temperature = 22.0`) automatically broadcasts to all $N$ entities.
   - Passing an $N$-element array updates each entity individually.
   - Setting an indexed variable (e.g. `p_cmd[42] = 5.0`) modifies only the specified unit.
2. **Instant Aggregation Outputs**:
   - Any state vector automatically exposes aggregated metrics: `total_<var>`, `mean_<var>`, `min_<var>`, and `max_<var>`.
3. **Sub-millisecond Step Speeds**:
   - Stepping 10,000 batteries or PV arrays takes fewer than **300 microseconds**, enabling annual (8,760 hours) co-simulations to complete in seconds.

---

## Container Catalog

* [**Storage Fleet**](storage_fleet.md): Heterogeneous residential battery storage and EV charging aggregation.
* [**Solar PV Fleet**](pv_fleet.md): Distributed rooftop solar arrays with spatial irradiance and temperature derate.
* [**Building Thermal Fleet**](building_fleet.md): 1R1C lumped thermal envelope fleets for demand response.
* [**Build Your Own Vectorized Fleet**](custom_fleet.md): Step-by-step tutorial on subclassing `VectorizedContainerAdapter` or using `VectorizedFunctionAdapter`.
"""

# 2. storage_fleet.md
storage_fleet_content = """# Vectorized Storage Fleet (`BatteryFleetAdapter`)

> **Class**: `energysim.container.BatteryFleetAdapter`  
> **Container Alias**: `'battery_fleet'` or `'storage_fleet'`

Simulates a fleet of $N$ heterogeneous battery energy storage systems or electric vehicles in vectorized SIMD memory.

---

## Governing Equations

For every entity $i \in [0, N-1]$:

* **Power Clamping to Headroom**:
  $$P_{\text{req}, i} = \text{clamp}(P_{\text{cmd}, i}, -P_{\text{max\_dis}, i}, P_{\text{max\_chg}, i})$$
  $$P_{\text{actual}, i} = \begin{cases} \min(P_{\text{req}, i}, (\text{SoC}_{\text{max}, i} - \text{SoC}_i) \cdot E_i / (\Delta t \cdot \eta)), & P_{\text{req}, i} \ge 0 \\ \max(P_{\text{req}, i}, -(\text{SoC}_i - \text{SoC}_{\text{min}, i}) \cdot E_i \cdot \eta / \Delta t), & P_{\text{req}, i} < 0 \end{cases}$$

* **Vectorized SoC Integration**:
  $$\text{SoC}_i(t + \Delta t) = \text{SoC}_i(t) + \frac{P_{\text{actual}, i} \cdot \eta^{\pm} \cdot \Delta t}{E_i}$$

---

## Parameters

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'battery_fleet'` | - | Container simulator name identifier |
| `size` | `int` | `100` | - | Number of battery units in the fleet ($N$) |
| `capacity_kwh` | `float` or `np.ndarray` | `50.0` | kWh | Nominal storage capacity (scalar or length-$N$ array) |
| `max_charge_kw` | `float` or `np.ndarray` | `11.0` | kW | Maximum charge power rating |
| `max_discharge_kw` | `float` or `np.ndarray` | `11.0` | kW | Maximum discharge power rating |
| `initial_soc` | `float` or `np.ndarray` | `0.5` | 0-1 | Initial state of charge distribution |
| `efficiency` | `float` | `0.95` | 0-1 | One-way coulombic charging/discharging efficiency |
| `step_size` | `float` | `900.0` | s | Micro integration time-step |

---

## Inputs & Outputs

### Inputs
* `p_cmd_kw`: Fleet power setpoint (scalar broadcast to all units, or length-$N$ dispatch vector).
* Individual indexed input: `p_cmd_kw[i]` sets active power for entity $i$.

### Outputs
* `total_actual_p_kw`: Total aggregate active power exchanged by the entire fleet (kW).
* `mean_soc`: Average fleet state of charge.
* `min_soc`: Minimum state of charge across all units.
* `max_soc`: Maximum state of charge across all units.
* `actual_p_kw[i]`, `soc[i]`: Individual state metrics for unit $i$.

---

## Example

```python
import numpy as np
from energysim import world

# 1. Fleet of 5,000 residential batteries with randomized capacities and initial SoCs
np.random.seed(42)
N = 5000
capacities = np.random.uniform(10.0, 20.0, N)  # 10 to 20 kWh
initial_socs = np.random.uniform(0.2, 0.8, N)

# 2. Add container to world
sim = world(start_time=0, stop_time=86400, t_macro=900)
sim.add_container(
    sim_name="home_batteries",
    container_type="battery_fleet",
    size=N,
    capacity_kwh=capacities,
    max_charge_kw=5.0,
    max_discharge_kw=5.0,
    initial_soc=initial_socs,
    outputs=["total_actual_p_kw", "mean_soc", "min_soc", "max_soc"],
)

# 3. Broadcast midday solar charge command
sim.add_signal("grid_signal", lambda t: 2.5 if 36000 <= t <= 54000 else -2.0)
sim.add_connections({"grid_signal.y": "home_batteries.p_cmd_kw"})

# 4. Simulate
sim.simulate()
results = sim.last_results
print(results["home_batteries"].head())
```
"""

# 3. pv_fleet.md
pv_fleet_content = """# Vectorized Solar PV Fleet (`PVFleetAdapter`)

> **Class**: `energysim.container.PVFleetAdapter`  
> **Container Alias**: `'pv_fleet'`

Simulates $N$ distributed rooftop solar photovoltaic installations with heterogeneous panel capacities, ambient irradiance, and temperature coefficient derating.

---

## Governing Equations

For every installation $i \in [0, N-1]$:

* **Cell Temperature Estimation**:
  $$T_{\text{cell}, i} = T_{\text{amb}} + \frac{G_i}{800} \cdot 25.0$$

* **Active Power Output**:
  $$P_{\text{out}, i} = P_{\text{peak}, i} \cdot \left(\frac{G_i}{1000}\right) \cdot [1 + \gamma (T_{\text{cell}, i} - 25)] \cdot \text{PR}$$

---

## Parameters

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'pv_fleet'` | - | Container name identifier |
| `size` | `int` | `100` | - | Number of rooftop solar installations |
| `peak_kw` | `float` or `np.ndarray` | `5.0` | kWp | Peak DC installed capacity per system |
| `gamma_temp_coeff` | `float` | `-0.004` | 1/°C | Temperature derating coefficient |
| `performance_ratio` | `float` | `0.85` | 0-1 | System derate (inverter + wiring + soiling) |
| `step_size` | `float` | `900.0` | s | Micro integration time-step |

---

## Inputs & Outputs

### Inputs
* `irradiance_w_m2`: Solar irradiance (scalar broadcast or length-$N$ vector for cloud cover).
* `t_ambient`: Ambient temperature (°C).

### Outputs
* `total_p_kw`: Total aggregate solar generation from the entire fleet (kW).
* `mean_p_kw`: Average solar generation per rooftop.
* `p_kw[i]`: Generation from individual rooftop $i$.

---

## Example

```python
import numpy as np
from energysim import world

N = 2500
capacities = np.random.uniform(3.0, 10.0, N)  # 3 kWp to 10 kWp

sim = world(start_time=0, stop_time=86400, t_macro=900)
sim.add_container(
    sim_name="rooftop_pv",
    container_type="pv_fleet",
    size=N,
    peak_kw=capacities,
    outputs=["total_p_kw", "mean_p_kw"],
)

sim.add_signal("solar_irr", lambda t: 900.0 if 21600 <= t <= 64800 else 0.0)
sim.add_signal("temp", lambda t: 25.0)
sim.add_connections({
    "solar_irr.y": "rooftop_pv.irradiance_w_m2",
    "temp.y": "rooftop_pv.t_ambient",
})

sim.simulate()
```
"""

# 4. building_fleet.md
building_fleet_content = """# Vectorized Building Fleet (`BuildingFleetAdapter`)

> **Class**: `energysim.container.BuildingFleetAdapter`  
> **Container Alias**: `'building_fleet'`

Vectorized lumped thermal envelope simulator modeling $N$ residential or commercial buildings using a 1R1C equivalent thermodynamic network for smart thermostat and heat pump demand response studies.

---

## Governing Equations

For every building $i \in [0, N-1]$:

$$\frac{dT_{\text{in}, i}}{dt} = \frac{T_{\text{amb}} - T_{\text{in}, i}}{R_{\text{th}, i} \cdot C_{\text{th}, i}} + \frac{Q_{\text{heat}, i} + Q_{\text{solar}, i}}{C_{\text{th}, i}}$$

---

## Parameters

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'building_fleet'` | - | Container name identifier |
| `size` | `int` | `100` | - | Number of buildings in fleet |
| `c_th_kwh_per_k` | `float` or `np.ndarray` | `15.0` | kWh/K | Thermal capacitance of building thermal mass |
| `r_th_k_per_kw` | `float` or `np.ndarray` | `2.5` | K/kW | Thermal resistance of exterior envelope |
| `initial_t_indoor` | `float` or `np.ndarray` | `20.0` | °C | Initial indoor room temperature |
| `step_size` | `float` | `900.0` | s | Micro integration time-step |

---

## Inputs & Outputs

### Inputs
* `t_ambient`: Outdoor ambient temperature (°C, broadcast to all buildings).
* `q_heat_kw`: Heating power input (kW, broadcast or length-$N$ vector).
* `q_solar_kw`: Passive solar heat gain through windows (kW).

### Outputs
* `mean_t_indoor`: Average indoor temperature across all buildings.
* `min_t_indoor`: Coldest indoor room temperature in fleet.
* `max_t_indoor`: Warmest indoor room temperature in fleet.
* `t_indoor[i]`: Indoor temperature for building $i$.
"""

# 5. custom_fleet.md
custom_fleet_content = """# Building Your Own Vectorized Fleet Container

`energysim` makes it straightforward to author custom high-performance vectorized fleet containers for any large-scale asset class — such as EV fleets with stochastic arrival/departure schedules, water heaters, industrial electrolysis cells, or cold storage facilities.

There are two primary approaches:
1. **Subclassing `VectorizedContainerAdapter`** (Recommended for reusable, robust domain models).
2. **Using `VectorizedFunctionAdapter`** (Quickest for prototyping with a pure NumPy function).

---

## Approach 1: Subclassing `VectorizedContainerAdapter`

By subclassing `VectorizedContainerAdapter`, you define vector states, inputs, aliases, and a fully vectorized `step()` method executed via NumPy C-arrays.

### Step-by-Step Guide: EV Fleet with Commute Logic

Let's build a `VectorizedEVFleet` where $N$ electric vehicles plug in during evening hours, charge at home, and drive to work in the morning:

```python
from typing import Optional, Union
import numpy as np
from energysim.container import VectorizedContainerAdapter


class VectorizedEVFleet(VectorizedContainerAdapter):
    \"\"\"
    Vectorized Electric Vehicle (EV) Fleet Container.

    Models N electric vehicles with randomized battery capacities,
    plug-in status flags, commute consumption, and charging setpoints.
    \"\"\"

    def __init__(
        self,
        name: str = "ev_fleet",
        size: int = 1000,
        battery_kwh: Union[float, np.ndarray] = 60.0,
        max_charger_kw: Union[float, np.ndarray] = 7.4,
        initial_soc: Union[float, np.ndarray] = 0.5,
        step_size: float = 900.0,
        **kwargs,
    ):
        super().__init__(name=name, size=size, step_size=step_size)

        # 1. Initialize parameter vectors (size N)
        if isinstance(battery_kwh, (int, float)):
            self.battery_capacity = np.full(self.size, float(battery_kwh), dtype=np.float64)
        else:
            self.battery_capacity = np.asarray(battery_kwh, dtype=np.float64)

        if isinstance(max_charger_kw, (int, float)):
            self.charger_limit = np.full(self.size, float(max_charger_kw), dtype=np.float64)
        else:
            self.charger_limit = np.asarray(max_charger_kw, dtype=np.float64)

        # 2. Register states (mapped to length-N numpy arrays)
        self.register_state("soc", initial_soc)
        self.register_state("actual_p_kw", 0.0)
        self.register_state("plugged_in", 1.0)  # 1.0 = connected, 0.0 = driving

        # 3. Register inputs
        self.register_input("charge_price_signal", 0.15)  # $/kWh
        self.register_input("grid_power_limit_kw", 1e6)   # Substation constraint

        # 4. Register convenient aliases
        self.register_alias("p_kw", "actual_p_kw")
        self.register_alias("power", "actual_p_kw")

    def step(self, time: float, step_size: Optional[float] = None) -> None:
        \"\"\"Advance all N EVs simultaneously in C-speed vectorized operations.\"\"\"
        dt_hr = (step_size or self.step_size) / 3600.0
        hr_of_day = (time / 3600.0) % 24.0

        soc = self.states["soc"]
        plugged = self.states["plugged_in"]

        # Vectorized commute schedule:
        # EVs unplug and drive between 7:30 and 8:30 AM
        driving_morning = (hr_of_day >= 7.5) & (hr_of_day < 8.5)
        # EVs return and plug in between 17:30 and 18:30 PM
        home_evening = (hr_of_day >= 17.5) | (hr_of_day < 7.5)

        plugged[driving_morning] = 0.0
        plugged[home_evening] = 1.0

        # Energy consumed during commute: 3 kWh drive loss
        commute_mask = driving_morning & (soc > 0.1)
        soc[commute_mask] -= (3.0 / self.battery_capacity[commute_mask]) * dt_hr

        # Charge logic for plugged-in vehicles wanting to reach 90% SoC
        needs_charge = (plugged == 1.0) & (soc < 0.90)
        headroom_kwh = np.maximum(0.0, (0.90 - soc) * self.battery_capacity)
        max_p_headroom_kw = headroom_kwh / dt_hr

        charge_p = np.zeros(self.size, dtype=np.float64)
        charge_p[needs_charge] = np.minimum(self.charger_limit[needs_charge], max_p_headroom_kw[needs_charge])

        # Update SoC
        soc[needs_charge] += (charge_p[needs_charge] * 0.95 * dt_hr) / self.battery_capacity[needs_charge]
        self.states["soc"][:] = np.clip(soc, 0.05, 1.0)
        self.states["actual_p_kw"][:] = charge_p
```

### Simulating the Custom EV Fleet

You can integrate your custom container directly into `energysim`:

```python
from energysim import world

# Instantiate fleet of 10,000 EVs
N = 10000
ev_fleet = VectorizedEVFleet(name="ev_fleet", size=N, battery_kwh=65.0)

sim = world(start_time=0, stop_time=86400, t_macro=900)
sim.add_model("evs", model=ev_fleet, outputs=[
    "total_actual_p_kw", "mean_soc", "min_soc", "max_soc"
])

sim.simulate()
results = sim.last_results

print("Peak aggregate EV charging demand:", results["evs"]["total_actual_p_kw"].max() / 1000.0, "MW")
```

---

## Approach 2: Using `VectorizedFunctionAdapter`

If you prefer not to write a class, you can pass a pure vectorized NumPy function directly to `VectorizedFunctionAdapter`:

```python
import numpy as np
from energysim import world
from energysim.container import VectorizedFunctionAdapter


def my_vectorized_update(time, states, inputs, dt):
    \"\"\"
    Vectorized state update function.
    states: dict mapping var_name -> 1D numpy array
    inputs: dict mapping var_name -> 1D numpy array
    Returns: dict of updated state arrays
    \"\"\"
    dt_hr = dt / 3600.0
    soc = states["soc"]
    p_cmd = inputs["p_setpoint"]  # Length-N array or scalar broadcast

    # Update states in single vectorized operation
    new_soc = np.clip(soc + (p_cmd * 0.92 * dt_hr) / 50.0, 0.0, 1.0)
    return {"soc": new_soc, "actual_p": p_cmd}


sim = world(start_time=0, stop_time=3600, t_macro=900)
custom_container = VectorizedFunctionAdapter(
    name="quick_fleet",
    size=1000,
    update_func=my_vectorized_update,
    initial_states={"soc": 0.5, "actual_p": 0.0},
    inputs=["p_setpoint"],
    step_size=900.0,
)

sim.add_model("quick_fleet", model=custom_container, outputs=["total_actual_p", "mean_soc"])
sim.simulate()
```

---

## Performance Best Practices

1. **Avoid Python Loops**: Never write `for i in range(self.size)` inside `step()`. Use boolean masks (`mask = soc < 0.2`) and NumPy array slices (`soc[mask] += ...`).
2. **Pre-allocate Arrays**: In `__init__`, pre-allocate temporary working buffers if needed, or use in-place assignment (`self.states['soc'][:] = ...`) to prevent memory allocations.
3. **Exploit In-Place Slicing**: Assigning `arr[:] = new_val` preserves existing C-contiguous array buffers, avoiding garbage collector overhead.
"""

files = {
    "overview.md": overview_content,
    "storage_fleet.md": storage_fleet_content,
    "pv_fleet.md": pv_fleet_content,
    "building_fleet.md": building_fleet_content,
    "custom_fleet.md": custom_fleet_content,
}

for name, content in files.items():
    file_path = os.path.join(containers_dir, name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated: {file_path}")

print("Container documentation generated successfully!")

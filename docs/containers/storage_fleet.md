# Vectorized Storage Fleet (`BatteryFleetAdapter`)

> **Class**: `energysim.container.BatteryFleetAdapter`  
> **Container Alias**: `'battery_fleet'` or `'storage_fleet'`

Simulates a fleet of $N$ heterogeneous battery energy storage systems or electric vehicles in vectorized SIMD memory.

---

## Governing Equations

For every entity $i \in [0, N-1]$:

* **Power Clamping to Headroom**:
  $$P_{	ext{req}, i} = 	ext{clamp}(P_{	ext{cmd}, i}, -P_{	ext{max\_dis}, i}, P_{	ext{max\_chg}, i})$$
  $$P_{	ext{actual}, i} = egin{cases} \min(P_{	ext{req}, i}, (	ext{SoC}_{	ext{max}, i} - 	ext{SoC}_i) \cdot E_i / (\Delta t \cdot \eta)), & P_{	ext{req}, i} \ge 0 \ \max(P_{	ext{req}, i}, -(	ext{SoC}_i - 	ext{SoC}_{	ext{min}, i}) \cdot E_i \cdot \eta / \Delta t), & P_{	ext{req}, i} < 0 \end{cases}$$

* **Vectorized SoC Integration**:
  $$	ext{SoC}_i(t + \Delta t) = 	ext{SoC}_i(t) + rac{P_{	ext{actual}, i} \cdot \eta^{\pm} \cdot \Delta t}{E_i}$$

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
sim.simulate(dashboard=False)
results = sim.results(dashboard=False)
print(results["home_batteries"].head())
```

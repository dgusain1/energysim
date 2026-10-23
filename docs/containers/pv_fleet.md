# Vectorized Solar PV Fleet (`PVFleetAdapter`)

> **Class**: `energysim.container.PVFleetAdapter`  
> **Container Alias**: `'pv_fleet'`

Simulates $N$ distributed rooftop solar photovoltaic installations with heterogeneous panel capacities, ambient irradiance, and temperature coefficient derating.

---

## Governing Equations

For every installation $i \in [0, N-1]$:

* **Cell Temperature Estimation**:
  $$T_{	ext{cell}, i} = T_{	ext{amb}} + rac{G_i}{800} \cdot 25.0$$

* **Active Power Output**:
  $$P_{	ext{out}, i} = P_{	ext{peak}, i} \cdot \left(rac{G_i}{1000}ight) \cdot [1 + \gamma (T_{	ext{cell}, i} - 25)] \cdot 	ext{PR}$$

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

sim.simulate(dashboard=False)
```

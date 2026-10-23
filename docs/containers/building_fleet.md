# Vectorized Building Fleet (`BuildingFleetAdapter`)

> **Class**: `energysim.container.BuildingFleetAdapter`  
> **Container Alias**: `'building_fleet'`

Vectorized lumped thermal envelope simulator modeling $N$ residential or commercial buildings using a 1R1C equivalent thermodynamic network for smart thermostat and heat pump demand response studies.

---

## Governing Equations

For every building $i \in [0, N-1]$:

$$rac{dT_{	ext{in}, i}}{dt} = rac{T_{	ext{amb}} - T_{	ext{in}, i}}{R_{	ext{th}, i} \cdot C_{	ext{th}, i}} + rac{Q_{	ext{heat}, i} + Q_{	ext{solar}, i}}{C_{	ext{th}, i}}$$

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

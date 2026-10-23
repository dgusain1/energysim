# High-Pressure Hydrogen Storage Tank

> **Class**: `H2StorageTank`  
> **Module**: `energysim.models.hydrogen`

## Overview

Compressed gaseous hydrogen buffer storage vessel (Type I steel or Type IV composite, 100–700 bar). Features Redlich-Kwong / NIST real gas compressibility factor calculation $Z(p, T)$ and pressure mass balance.

---

### Governing Equations

* **Real Gas Compressibility Equation of State**:
  $$p \cdot V = Z(p, T) \cdot \frac{m}{M_{\text{H}_2}} \cdot R \cdot T$$
  At 350 bar, hydrogen compressibility factor $Z \approx 1.25$ (ideal gas law would overestimate capacity by 25%).

* **Dynamic Pressure Evolution**:
  $$\frac{dm}{dt} = \dot{m}_{\text{in}} - \dot{m}_{\text{out}}$$
  $$p(t) = f_{\text{real-gas}}(m(t), V, T)$$


---

## Parameters & Attributes

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'H2_Tank'` | - | Unique component name identifier |
| `volume_m3` | `float` | `100.0` | m³ | Internal geometrical tank volume |
| `p_max_bar` | `float` | `350.0` | bar | Maximum allowable working pressure (MAWP) |
| `p_min_bar` | `float` | `20.0` | bar | Minimum residual suction pressure |
| `p_init_bar` | `float` | `120.0` | bar | Initial storage tank pressure |
| `temp_c` | `float` | `25.0` | °C | Internal gas temperature |

---

## Co-Simulation Inputs

| Input Variable | Unit | Description |
|---|---|---|
| `m_dot_in_kg_hr` | kg/hr | Inflow mass flow rate entering the tank (from electrolyzer/compressor) |
| `m_dot_out_kg_hr` | kg/hr | Outflow mass flow rate dispatched to industrial loads or fuel cells |

---

## Simulation Outputs

| Output Variable | Unit | Description |
|---|---|---|
| `pressure_bar` | bar | Current internal tank pressure |
| `mass_stored_kg` | kg | Total mass of hydrogen currently stored in the vessel |
| `SoC` | 0-1 | Storage capacity fraction between $p_{	ext{min}}$ and $p_{	ext{max}}$ |

---

## Python Usage Example

```python
from energysim import world
from energysim.models.hydrogen import H2StorageTank

tank = H2StorageTank(name="tank", volume_m3=100.0, p_max_bar=350.0, p_init_bar=120.0)
sim = world(start_time=0, stop_time=3600, t_macro=900)
sim.add_model("tank", model=tank, outputs=["pressure_bar", "mass_stored_kg"])
sim.add_signal("h2_in", lambda t: 150.0)  # 150 kg/hr in
sim.add_connections({"h2_in.y": "tank.m_dot_in_kg_hr"})
sim.simulate(dashboard=False)
```

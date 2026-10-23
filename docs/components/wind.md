# Wind Turbine Generator

> **Class**: `WindTurbine`  
> **Module**: `energysim.models.renewables`

## Overview

Utility-scale wind turbine model featuring piecewise cubic aerodynamic power curves with cut-in, rated, and cut-out wind speed thresholds, alongside air density correction.

---

### Governing Equations

* **Air Density Correction**:
  $$v_{\text{eff}} = v_{\text{wind}} \cdot \left(\frac{\rho}{\rho_0}\right)^{1/3}$$

* **Aerodynamic Power Curve**:
  $$P_{\text{mech}}(v) = \begin{cases} 0, & v < v_{\text{cut-in}} \text{ or } v > v_{\text{cut-out}} \\ P_{\text{rated}} \cdot \left(\frac{v - v_{\text{cut-in}}}{v_{\text{rated}} - v_{\text{cut-in}}}\right)^3, & v_{\text{cut-in}} \le v < v_{\text{rated}} \\ P_{\text{rated}}, & v_{\text{rated}} \le v \le v_{\text{cut-out}} \end{cases}$$


---

## Parameters & Attributes

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'WindTurbine'` | - | Unique component name identifier |
| `p_rated_mw` | `float` | `4.0` | MW | Rated nameplate electrical generator capacity |
| `v_cut_in_m_s` | `float` | `3.0` | m/s | Cut-in wind speed threshold |
| `v_rated_m_s` | `float` | `12.0` | m/s | Rated wind speed threshold |
| `v_cut_out_m_s` | `float` | `25.0` | m/s | Cut-out storm shutdown wind speed threshold |
| `hub_height_m` | `float` | `120.0` | m | Rotor hub height above ground |

---

## Co-Simulation Inputs

| Input Variable | Unit | Description |
|---|---|---|
| `wind_speed_m_s` | m/s | Wind speed at hub height |
| `air_density_kg_m3` | kg/m³ | Ambient air density (default 1.225 kg/m³) |

---

## Simulation Outputs

| Output Variable | Unit | Description |
|---|---|---|
| `P_actual` | MW | Net active power delivered to the collector substation |
| `status` | str | Operating status ('RUNNING', 'PARKED', 'STORM_CUTOUT') |

---

## Python Usage Example

```python
from energysim.models.renewables import WindTurbine

wt = WindTurbine(name="turbine", p_rated_mw=4.0, v_cut_in_m_s=3.0, v_rated_m_s=11.5)
```

# Solar Thermal Collector

> **Class**: `SolarThermalCollector`  
> **Module**: `energysim.models.renewables`

## Overview

Parabolic trough or flat plate solar thermal collector modeled via the Hottel-Whillier-Bliss efficiency equations for district heating networks or industrial process heat.

---

### Governing Equations

* **Hottel-Whillier-Bliss Efficiency**:
  $$\eta_{\text{th}} = \eta_0 - a_1 \frac{T_m - T_{\text{amb}}}{G} - a_2 \frac{(T_m - T_{\text{amb}})^2}{G}$$

* **Thermal Power Output**:
  $$Q_{\text{th}} = A_{\text{collector}} \cdot G \cdot \eta_{\text{th}} = \dot{m} c_p (T_{\text{out}} - T_{\text{in}})$$


---

## Parameters & Attributes

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'SolarThermal'` | - | Unique component name identifier |
| `area_m2` | `float` | `5000.0` | m² | Total collector aperture area |
| `eta0` | `float` | `0.78` | 0-1 | Optical peak efficiency factor |
| `a1` | `float` | `3.5` | W/(m²·K) | Linear heat loss coefficient |
| `a2` | `float` | `0.015` | W/(m²·K²) | Quadratic heat loss coefficient |

---

## Co-Simulation Inputs

| Input Variable | Unit | Description |
|---|---|---|
| `G_dni` | W/m² | Direct normal or global irradiance |
| `T_amb` | °C | Ambient temperature |
| `T_in` | °C | Inlet heat transfer fluid temperature |
| `m_dot_kg_s` | kg/s | Fluid mass flow rate |

---

## Simulation Outputs

| Output Variable | Unit | Description |
|---|---|---|
| `Q_th_MW` | MWth | Thermal power transferred to heat transfer fluid |
| `T_out` | °C | Collector fluid outlet temperature |

---

## Python Usage Example

```python
from energysim.models.renewables import SolarThermalCollector

st = SolarThermalCollector(name="solar_field", area_m2=10000.0)
```

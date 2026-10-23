# Solid Oxide Electrolyzer (SOEC)

> **Class**: `SOECElectrolyzer`  
> **Module**: `energysim.models.hydrogen`

## Overview

High-temperature Solid Oxide Electrolyzer Cell (SOEC) operating at 700–850°C. Operates via steam electrolysis, utilizing external thermal process steam to dramatically lower electrical consumption.

---

### Governing Equations

* **Thermodynamic Energy Balance**:
  $$\Delta H = \Delta G + T \Delta S$$
  As temperature increases, electrical energy demand ($\Delta G$) decreases while thermal demand ($T \Delta S$) increases.

* **Steam Consumption Rate**:
  $$\dot{m}_{\text{steam}} = \dot{m}_{\text{H}_2} \cdot \frac{M_{\text{H}_2\text{O}}}{M_{\text{H}_2}} = 8.936 \cdot \dot{m}_{\text{H}_2} \quad [\text{kg/hr}]$$


---

## Parameters & Attributes

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'SOEC'` | - | Unique component name identifier |
| `p_nominal_mw` | `float` | `5.0` | MW | Nominal electrical stack power rating |
| `specific_electrical_kwh_kg` | `float` | `38.0` | kWh/kg | High-efficiency electrical consumption |
| `steam_demand_kg_per_kg_h2` | `float` | `9.0` | kg/kg | Steam mass demand per kg hydrogen produced |

---

## Co-Simulation Inputs

| Input Variable | Unit | Description |
|---|---|---|
| `P_cmd` | MW | Active power setpoint |
| `T_steam_in` | °C | Superheated steam inlet temperature (700-850°C) |

---

## Simulation Outputs

| Output Variable | Unit | Description |
|---|---|---|
| `P_actual` | MW | Actual electrical power consumed |
| `m_dot_H2_kg_hr` | kg/hr | Hydrogen production rate |
| `steam_consumed_t_hr` | t/hr | Steam mass flow rate consumed from steam network |

---

## Python Usage Example

```python
from energysim.models.hydrogen import SOECElectrolyzer

soec = SOECElectrolyzer(name="soec", p_nominal_mw=5.0, specific_electrical_kwh_kg=38.0)
```

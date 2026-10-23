# Fuel Cell (PEMFC / SOFC)

> **Class**: `FuelCell_PEM, FuelCell_SOFC`  
> **Module**: `energysim.models.hydrogen`

## Overview

Electrochemical fuel cell power systems. Models hydrogen consumption, electrical power generation, electrical efficiency curve derating at part load, and cogeneration waste heat recovery.

---

### Governing Equations

* **Electrical Power from Hydrogen**:
  $$P_{\text{elec}} = \dot{m}_{\text{H}_2} \cdot \text{LHV}_{\text{H}_2} \cdot \eta_{\text{fc}}(P)$$

* **Combined Heat and Power (CHP) Cogeneration**:
  $$Q_{\text{cogeneration}} = \dot{m}_{\text{H}_2} \cdot \text{HHV}_{\text{H}_2} - P_{\text{elec}}$$


---

## Parameters & Attributes

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'FuelCell'` | - | Unique component name identifier |
| `p_rated_mw` | `float` | `2.0` | MW | Rated electrical power output |
| `efficiency_nominal` | `float` | `0.55` | 0-1 | Rated electrical efficiency based on LHV |
| `min_load_pct` | `float` | `10.0` | % | Minimum operational load fraction |

---

## Co-Simulation Inputs

| Input Variable | Unit | Description |
|---|---|---|
| `P_cmd` | MW | Active power dispatch setpoint |

---

## Simulation Outputs

| Output Variable | Unit | Description |
|---|---|---|
| `P_actual` | MW | Net active power delivered to the electrical bus |
| `m_dot_H2_consumed_kg_hr` | kg/hr | Hydrogen fuel consumption rate |
| `Q_heat_MW` | MWth | Cogeneration thermal heat output |

---

## Python Usage Example

```python
from energysim.models.hydrogen import FuelCell_PEM

fc = FuelCell_PEM(name="fc", p_rated_mw=2.0)
```

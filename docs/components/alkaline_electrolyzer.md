# Alkaline Electrolyzer (AEL)

> **Class**: `AlkalineElectrolyzer`  
> **Module**: `energysim.models.hydrogen`

## Overview

Alkaline Water Electrolyzer (AEL) characterized by liquid KOH electrolyte. Features an enforced 20% minimum safety turndown cutoff to avoid hazardous hydrogen-in-oxygen gas crossover and thermal warmup inertia.

---

### Governing Equations

* **Minimum Safety Turndown Cutoff**:
  $$P_{\text{actual}} = \begin{cases} P_{\text{cmd}}, & P_{\text{cmd}} \ge 0.20 \cdot P_{\text{nom}} \\ 0.0 \text{ (Safety Trip / Standby)}, & P_{\text{cmd}} < 0.20 \cdot P_{\text{nom}} \end{cases}$$

* **Gas Crossover Penalty**:
  $$\text{Purity}_{\text{H}_2} = 99.9\% - 0.5\% \cdot \left(\frac{0.20 \cdot P_{\text{nom}}}{P_{\text{actual}}}\right)^2 \quad (\text{near cutoff})$$


---

## Parameters & Attributes

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'AEL'` | - | Unique component name identifier |
| `p_nominal_mw` | `float` | `20.0` | MW | Nominal rated plant power |
| `specific_consumption_kwh_kg` | `float` | `55.0` | kWh/kg | Specific energy consumption at rated load |
| `min_turndown_pct` | `float` | `20.0` | % | Strict safety turndown limit (typically 20-25%) |
| `warmup_time_min` | `float` | `30.0` | min | Cold-start stack warmup duration |

---

## Co-Simulation Inputs

| Input Variable | Unit | Description |
|---|---|---|
| `P_cmd` | MW | Power dispatch command |

---

## Simulation Outputs

| Output Variable | Unit | Description |
|---|---|---|
| `P_actual` | MW | Actual active power absorbed |
| `m_dot_H2_kg_hr` | kg/hr | Hydrogen mass production rate |
| `state` | str | Operating status ('RUNNING', 'CUTOFF_TRIP', 'OFF') |

---

## Python Usage Example

```python
from energysim.models.hydrogen import AlkalineElectrolyzer

ael = AlkalineElectrolyzer(name="ael", p_nominal_mw=20.0, min_turndown_pct=20.0)
```

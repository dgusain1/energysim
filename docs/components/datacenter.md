# Data Center IT & Cooling Load

> **Class**: `DataCenterLoad`  
> **Module**: `energysim.models.loads`

## Overview

Hyperscale data center load modeling baseline IT compute server electrical demand combined with temperature-dependent Power Usage Effectiveness (PUE) cooling curves and waste heat rejection.

---

### Governing Equations

* **Total Facility Power Demand**:
  $$P_{\text{facility}} = P_{\text{IT}} \cdot \text{PUE}(T_{\text{amb}})$$

* **PUE Temperature Characteristic**:
  $$\text{PUE}(T_{\text{amb}}) = \text{PUE}_{\text{base}} + k_{\text{cool}} \cdot \max(0, T_{\text{amb}} - 15.0)$$

* **Low-Grade Waste Heat for Heat Pumps**:
  $$Q_{\text{waste}} = P_{\text{facility}} \cdot \eta_{\text{thermal\_dissipation}} \quad (\sim 95\% \text{ of electrical power rejected at 30-35°C})$$


---

## Parameters & Attributes

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'DataCenter'` | - | Unique component name identifier |
| `p_it_nominal_mw` | `float` | `20.0` | MW | Baseline continuous IT compute electrical server power |
| `pue_base` | `float` | `1.15` | - | Base design PUE during free economizer cooling |
| `pue_temp_coeff` | `float` | `0.01` | 1/°C | PUE increase per °C above free-cooling threshold |

---

## Co-Simulation Inputs

| Input Variable | Unit | Description |
|---|---|---|
| `it_load_fraction` | 0-1 | Compute server utilization fraction |
| `T_amb` | °C | Ambient wet/dry-bulb temperature |

---

## Simulation Outputs

| Output Variable | Unit | Description |
|---|---|---|
| `P_facility_total_MW` | MW | Total facility grid electrical demand (IT + cooling + UPS) |
| `P_cooling_MW` | MW | Active power consumed solely by chillers and HVAC |
| `Q_waste_heat_MWth` | MWth | Low-grade waste heat available for district heating |
| `PUE` | - | Operating Power Usage Effectiveness |

---

## Python Usage Example

```python
from energysim.models.loads import DataCenterLoad

dc = DataCenterLoad(name="dc", p_it_nominal_mw=20.0, pue_base=1.15)
```

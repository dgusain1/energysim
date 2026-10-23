# Industrial Electric Boiler

> **Class**: `ElectricBoiler`  
> **Module**: `energysim.models.hvac_thermal`

## Overview

High-voltage electrode or immersion electric steam/hot water boiler. Converts surplus renewable electricity into process heat at near-unity efficiency (>99%).

---

### Governing Equations

* **Heat Generation**:
  $$Q_{\text{thermal}} = P_{\text{elec}} \cdot \eta_{\text{boiler}}$$


---

## Parameters & Attributes

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'ElectricBoiler'` | - | Unique component name identifier |
| `p_rated_mw` | `float` | `10.0` | MW | Maximum electrical power consumption |
| `efficiency` | `float` | `0.99` | 0-1 | Conversion thermal efficiency |

---

## Co-Simulation Inputs

| Input Variable | Unit | Description |
|---|---|---|
| `P_cmd` | MW | Electrical consumption power request |

---

## Simulation Outputs

| Output Variable | Unit | Description |
|---|---|---|
| `P_actual` | MW | Actual active power absorbed |
| `Q_thermal_MW` | MWth | Useful steam or hot water thermal power delivered |

---

## Python Usage Example

```python
from energysim.models.hvac_thermal import ElectricBoiler

boiler = ElectricBoiler(name="boiler", p_rated_mw=10.0)
```

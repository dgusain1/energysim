# Industrial Process Steam Header

> **Class**: `IndustrialProcessSteam`  
> **Module**: `energysim.models.loads`

## Overview

Industrial refinery or chemical plant steam header mass and enthalpy balance. Coordinates steam generation from CCGT HRSG boilers, electric boilers, and solar thermal collectors against industrial consumer demand.

---

### Governing Equations

* **Steam Mass Balance**:
  $$\dot{m}_{\text{balance}} = \sum \dot{m}_{\text{supply}} - \sum \dot{m}_{\text{demand}}$$
  $$\dot{m}_{\text{deficit}} = \max(0, -\dot{m}_{\text{balance}})$$


---

## Parameters & Attributes

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'SteamHeader'` | - | Unique component name identifier |
| `nominal_demand_t_hr` | `float` | `50.0` | t/hr | Nominal process steam demand |
| `pressure_bar` | `float` | `40.0` | bar | Header steam pressure |

---

## Co-Simulation Inputs

| Input Variable | Unit | Description |
|---|---|---|
| `steam_supplied_t_hr` | t/hr | Total steam supplied from HRSG and electric boilers |

---

## Simulation Outputs

| Output Variable | Unit | Description |
|---|---|---|
| `steam_demand_t_hr` | t/hr | Process steam demand |
| `steam_deficit_t_hr` | t/hr | Unmet steam demand |

---

## Python Usage Example

```python
from energysim.models.loads import IndustrialProcessSteam

steam = IndustrialProcessSteam(name="steam", nominal_demand_t_hr=50.0)
```

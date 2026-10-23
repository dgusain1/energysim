# Combined Heat and Power (CHP) Steam Turbine

> **Class**: `CHP_SteamTurbine`  
> **Module**: `energysim.models.thermal`

## Overview

Backpressure or extraction steam turbine delivering simultaneous active electrical power and low-pressure district heating or process steam.

---

### Governing Equations

* **Power-to-Heat Ratio**:
  $$\sigma = \frac{P_{\text{elec}}}{Q_{\text{heat}}}$$

* **Total Fuel Utilization**:
  $$\eta_{\text{total}} = \frac{P_{\text{elec}} + Q_{\text{heat}}}{\dot{m}_{\text{fuel}} \cdot \text{LHV}} \ge 85\%$$


---

## Parameters & Attributes

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'CHP'` | - | Unique component name identifier |
| `p_rated_mw` | `float` | `10.0` | MW | Rated electrical generator capacity |
| `power_to_heat_ratio` | `float` | `0.8` | - | Design electrical-to-thermal power ratio $\sigma$ |
| `thermal_efficiency_pct` | `float` | `88.0` | % | Total plant fuel utilization efficiency |

---

## Co-Simulation Inputs

| Input Variable | Unit | Description |
|---|---|---|
| `P_cmd` | MW | Electrical dispatch order |
| `heat_demand_mw` | MWth | Process heat dispatch order |

---

## Simulation Outputs

| Output Variable | Unit | Description |
|---|---|---|
| `P_actual` | MW | Actual electricity generated |
| `Q_heat_actual_MW` | MWth | Actual thermal heat delivered |
| `fuel_input_MW` | MW | Total primary fuel consumption rate |

---

## Python Usage Example

```python
from energysim.models.thermal import CHP_SteamTurbine

chp = CHP_SteamTurbine(name="chp_plant", p_rated_mw=10.0, power_to_heat_ratio=0.8)
```

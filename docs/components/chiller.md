# Chillers (Electric & Absorption)

> **Class**: `ElectricChiller, AbsorptionChiller`  
> **Module**: `energysim.models.hvac_thermal`

## Overview

District cooling plant models featuring vapor-compression Electric Chillers (driven by active electrical power) and thermally driven Absorption Chillers (utilizing waste heat from gas turbines or industrial exhausts).

---

### Governing Equations

* **Electric Chiller**:
  $$Q_{\text{cool}} = P_{\text{elec}} \cdot \text{COP}_{\text{chiller}}$$

* **Absorption Chiller (Thermal COP)**:
  $$Q_{\text{cool}} = Q_{\text{waste\_heat\_in}} \cdot \text{COP}_{\text{thermal}} \quad (\text{COP}_{\text{thermal}} \approx 0.7 - 1.2)$$


---

## Parameters & Attributes

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'Chiller'` | - | Unique component name identifier |
| `q_cool_max_mw` | `float` | `5.0` | MWth | Maximum chilled water cooling capacity |
| `cop_nominal` | `float` | `5.5` | - | Design Coefficient of Performance |

---

## Co-Simulation Inputs

| Input Variable | Unit | Description |
|---|---|---|
| `cooling_demand_mw` | MWth | Chilled water thermal load request |

---

## Simulation Outputs

| Output Variable | Unit | Description |
|---|---|---|
| `Q_cooling_delivered_MW` | MWth | Delivered chilled water cooling power |
| `P_electric_consumed_MW` | MW | Compressor electricity consumed (Electric Chiller) |
| `Q_heat_consumed_MW` | MWth | Waste heat consumed (Absorption Chiller) |

---

## Python Usage Example

```python
from energysim.models.hvac_thermal import ElectricChiller

chiller = ElectricChiller(name="chiller", q_cool_max_mw=5.0, cop_nominal=5.5)
```

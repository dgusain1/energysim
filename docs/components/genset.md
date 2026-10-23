# Reciprocating Engine Genset

> **Class**: `ReciprocatingGenset`  
> **Module**: `energysim.models.thermal`

## Overview

Medium-speed or high-speed reciprocating internal combustion engine (diesel, lean-burn natural gas, or dual-fuel). Features ultra-fast synchronization (<2 minutes), high part-load efficiency, and jacket water / exhaust heat recovery.

---

### Governing Equations

* **Fuel Mass Flow Rate**:
  $$\dot{m}_{\text{fuel}} = \frac{P_{\text{actual}}}{\eta_{\text{brake}}(P) \cdot \text{LHV}_{\text{fuel}}} \quad [\text{kg/hr}]$$


---

## Parameters & Attributes

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'Genset'` | - | Unique component name identifier |
| `p_rated_mw` | `float` | `5.0` | MW | Rated electrical power output |
| `efficiency_full_load` | `float` | `0.44` | 0-1 | Brake electrical efficiency at rated load (~44%) |
| `min_load_pct` | `float` | `15.0` | % | Minimum continuous load limit |

---

## Co-Simulation Inputs

| Input Variable | Unit | Description |
|---|---|---|
| `P_cmd` | MW | Power dispatch request |

---

## Simulation Outputs

| Output Variable | Unit | Description |
|---|---|---|
| `P_actual` | MW | Delivered electrical power |
| `fuel_l_hr` | L/hr | Fuel volumetric consumption rate |
| `Q_jacket_water_kW` | kWth | Jacket water thermal heat recovery |
| `Q_exhaust_kW` | kWth | Exhaust gas thermal heat recovery |

---

## Python Usage Example

```python
from energysim.models.thermal import ReciprocatingGenset

genset = ReciprocatingGenset(name="genset", p_rated_mw=5.0)
```

# Open Cycle Gas Turbine (OCGT)

> **Class**: `GasTurbine_OCGT`  
> **Module**: `energysim.models.thermal`

## Overview

Simple-cycle aeroderivative or heavy-duty peaking gas turbine. Features fast start-up operating state machines (Cold/Warm/Hot timers), ramp rate constraints, high-temperature exhaust heat output, and part-load efficiency derate.

---

### Governing Equations

* **Operating State Machine**:
  $$\text{State} \in \{\text{OFF}, \text{STARTING}, \text{RUNNING}, \text{SHUTTING\_DOWN}\}$$
  Startup duration dynamically switches between Hot ($<2\text{ hr}$ downtime), Warm ($2-8\text{ hr}$), and Cold ($>8\text{ hr}$).

* **Startup Fuel Loss**:
  $$\text{Fuel}_{\text{start}} = P_{\text{rated}} \cdot 1000 \cdot \text{HR}_{\text{full}} \cdot 0.15 \quad [\text{kJ during roll/purge}]$$


---

## Parameters & Attributes

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'OCGT'` | - | Unique component name identifier |
| `p_rated_mw` | `float` | `50.0` | MW | Rated generator active power output |
| `heat_rate_full_load_kj_kwh` | `float` | `9500.0` | kJ/kWh | Full load heat rate (~37.9% efficiency) |
| `min_stable_gen_pct` | `float` | `20.0` | % | Minimum stable generation limit |
| `ramp_rate_mw_min` | `float` | `25.0` | MW/min | Aeroderivative fast ramp rate limit |

---

## Co-Simulation Inputs

| Input Variable | Unit | Description |
|---|---|---|
| `P_cmd` | MW | Dispatch setpoint |
| `T_amb` | °C | Compressor inlet air temperature |

---

## Simulation Outputs

| Output Variable | Unit | Description |
|---|---|---|
| `P_actual` | MW | Delivered active power |
| `fuel_consumption_MMBtu_hr` | MMBtu/hr | Fuel gas consumption rate |
| `Q_exhaust_MW` | MWth | Thermal energy in exhaust flue gas |
| `state` | str | Operating state string |

---

## Python Usage Example

```python
from energysim.models.thermal import GasTurbine_OCGT

ocgt = GasTurbine_OCGT(name="peaker", p_rated_mw=50.0, ramp_rate_mw_min=25.0)
```

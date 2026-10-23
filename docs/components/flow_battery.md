# Vanadium Redox Flow Battery (VRFB)

> **Class**: `FlowBattery`  
> **Module**: `energysim.models.storage`

## Overview

Vanadium Redox Flow Battery (VRFB) featuring physically decoupled energy capacity (electrolyte tank volume) and power rating (cell stack membrane area). Models dynamic variable-speed electrolyte pump parasitic losses and stack electrochemical efficiency.

---

### Governing Equations

* **Decoupled Stack Power and Tank Energy**:
  $$E_{\text{tank}} = V_{\text{tank}} \cdot C_{\text{vanadium}} \cdot F \cdot V_{\text{cell,nom}}$$

* **Variable Pumping Parasitic Loss**:
  $$P_{\text{pump}} = P_{\text{base}} + P_{\text{var}} \cdot \left(\frac{|P_{\text{stack}}|}{P_{\text{rated}}}\right)^2$$

* **Net AC Grid Power**:
  $$P_{\text{grid}} = P_{\text{stack}} - P_{\text{pump}}$$


---

## Parameters & Attributes

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'VRFB'` | - | Unique component name identifier |
| `p_rated_mw` | `float` | `1.0` | MW | Rated power of the cell stack |
| `capacity_mwh` | `float` | `8.0` | MWh | Total electrolyte tank energy capacity (e.g. 8-hour duration) |
| `soc_init` | `float` | `0.5` | 0-1 | Initial state of charge fraction of the electrolyte |
| `pump_parasitic_pct` | `float` | `3.0` | % | Parasitic pumping loss percentage at rated flow |
| `eta_stack` | `float` | `0.85` | 0-1 | Electrochemical stack round-trip efficiency |

---

## Co-Simulation Inputs

| Input Variable | Unit | Description |
|---|---|---|
| `P_cmd` | MW | Active power setpoint (positive = charge, negative = discharge) |

---

## Simulation Outputs

| Output Variable | Unit | Description |
|---|---|---|
| `P_actual` | MW | Net active power delivered to or absorbed from the grid |
| `SoC` | 0-1 | Current electrolyte state of charge fraction |
| `P_pump_kW` | kW | Active power consumed by electrolyte circulation pumps |

---

## Python Usage Example

```python
from energysim import world
from energysim.models.storage import FlowBattery

vrfb = FlowBattery(name="vrfb", p_rated_mw=1.0, capacity_mwh=8.0, soc_init=0.4)
sim = world(start_time=0, stop_time=7200, t_macro=900)
sim.add_model("vrfb", model=vrfb, outputs=["P_actual", "SoC", "P_pump_kW"])
sim.simulate(dashboard=False)
```

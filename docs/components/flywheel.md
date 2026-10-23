# Flywheel Kinetic Energy Storage

> **Class**: `Flywheel`  
> **Module**: `energysim.models.storage`

## Overview

High-speed mechanical kinetic energy storage system designed for primary frequency regulation and fast synthetic inertia. Models rotational kinetic energy, magnetic bearing losses, and aerodynamic vacuum drag.

---

### Governing Equations

* **Kinetic Energy Storage**:
  $$E_k = \frac{1}{2} J \omega^2$$

* **Parasitic Self-Discharge (Aerodynamic & Bearing Friction)**:
  $$P_{\text{loss}} = P_{\text{bearing}} + c_{\text{drag}} \cdot \omega^2$$


---

## Parameters & Attributes

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'Flywheel'` | - | Unique component name identifier |
| `p_rated_mw` | `float` | `2.0` | MW | Rated bi-directional power capacity |
| `capacity_kwh` | `float` | `500.0` | kWh | Total kinetic storage capacity |
| `standby_loss_kw` | `float` | `5.0` | kW | Vacuum bearing and friction loss at rated speed |

---

## Co-Simulation Inputs

| Input Variable | Unit | Description |
|---|---|---|
| `P_cmd` | MW | Active power setpoint (positive = charge, negative = discharge) |

---

## Simulation Outputs

| Output Variable | Unit | Description |
|---|---|---|
| `P_actual` | MW | Net active power exchanged with the grid |
| `SoC` | 0-1 | Rotational kinetic energy state of charge ($(\omega / \omega_{	ext{max}})^2$) |
| `speed_rpm` | RPM | Current rotor rotational speed |

---

## Python Usage Example

```python
from energysim.models.storage import Flywheel

fw = Flywheel(name="fw", p_rated_mw=2.0, capacity_kwh=500.0)
```

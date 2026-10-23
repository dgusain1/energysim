# Stratified Thermal Storage Tank

> **Class**: `ThermalStorageTank`  
> **Module**: `energysim.models.hvac_thermal`

## Overview

Sensible heat water storage tank featuring a 1-D vertical temperature thermocline separating hot and cold layers for district heating or cooling peak shaving.

---

### Governing Equations

* **Sensible Heat Energy Balance**:
  $$E_{\text{stored}} = \rho \cdot V \cdot c_p \cdot (T_{\text{hot}} - T_{\text{cold}})$$
  $$\frac{dE}{dt} = Q_{\text{in}} - Q_{\text{out}} - U A (T_{\text{tank}} - T_{\text{amb}})$$


---

## Parameters & Attributes

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'ThermalTank'` | - | Unique component name identifier |
| `volume_m3` | `float` | `2000.0` | m³ | Water vessel volume |
| `t_hot_c` | `float` | `95.0` | °C | Hot layer temperature |
| `t_cold_c` | `float` | `45.0` | °C | Cold return layer temperature |
| `loss_coeff_w_m2_k` | `float` | `0.5` | W/(m²·K) | Insulation heat loss coefficient |

---

## Co-Simulation Inputs

| Input Variable | Unit | Description |
|---|---|---|
| `Q_in_MWth` | MWth | Heat injection rate (charging) |
| `Q_out_MWth` | MWth | Heat extraction rate (discharging) |

---

## Simulation Outputs

| Output Variable | Unit | Description |
|---|---|---|
| `SoC` | 0-1 | Thermal state of charge |
| `energy_stored_mwh` | MWh | Total stored thermal energy |

---

## Python Usage Example

```python
from energysim.models.hvac_thermal import ThermalStorageTank

tank = ThermalStorageTank(name="tes", volume_m3=2000.0)
```

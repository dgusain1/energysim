# Amine Post-Combustion Carbon Capture

> **Class**: `CarbonCapture_Amine`  
> **Module**: `energysim.models.carbon`

## Overview

Post-combustion chemical absorption carbon capture system using monoethanolamine (MEA) or advanced amines. Models flue gas $\text{CO}_2$ capture efficiency and thermal reboiler steam stripping penalty.

---

### Governing Equations

* **Captured Mass Rate**:
  $$\dot{m}_{\text{CO}_2\text{,captured}} = \dot{m}_{\text{CO}_2\text{,in}} \cdot \eta_{\text{capture}} \quad (\eta_{\text{capture}} \approx 90\%)$$

* **Thermal Reboiler Duty**:
  $$Q_{\text{reboiler}} = \dot{m}_{\text{CO}_2\text{,captured}} \cdot q_{\text{specific}} \quad (q_{\text{specific}} \approx 2.5 - 3.5\text{ GJ/tCO}_2)$$

* **Electrical Auxiliary Power**:
  $$P_{\text{aux}} = \dot{m}_{\text{CO}_2\text{,captured}} \cdot w_{\text{specific}} \quad (w_{\text{specific}} \approx 100 - 150\text{ kWh/tCO}_2)$$


---

## Parameters & Attributes

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'CarbonCapture'` | - | Unique component name identifier |
| `capture_efficiency_pct` | `float` | `90.0` | % | Flue gas $\text{CO}_2$ capture recovery rate |
| `specific_reboiler_duty_gj_t` | `float` | `2.8` | GJ/tCO₂ | Thermal steam heat required per ton $\text{CO}_2$ stripped |
| `specific_power_kwh_t` | `float` | `110.0` | kWh/tCO₂ | Electrical pump and compressor power per ton $\text{CO}_2$ |

---

## Co-Simulation Inputs

| Input Variable | Unit | Description |
|---|---|---|
| `CO2_flue_gas_t_hr` | t/hr | Carbon dioxide mass flow entering from flue stack |

---

## Simulation Outputs

| Output Variable | Unit | Description |
|---|---|---|
| `CO2_captured_t_hr` | t/hr | Mass flow rate of captured high-purity $\text{CO}_2$ |
| `steam_demand_MWth` | MWth | Thermal reboiler heat demand extracted from steam network |
| `aux_power_MW` | MW | Parasitic electricity consumed by pumps and blowers |

---

## Python Usage Example

```python
from energysim.models.carbon import CarbonCapture_Amine

ccs = CarbonCapture_Amine(name="ccs", capture_efficiency_pct=90.0)
```

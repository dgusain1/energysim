# PEM Electrolyzer

> **Class**: `PEMElectrolyzer`  
> **Module**: `energysim.models.hydrogen`

## Overview

Proton Exchange Membrane (PEM) Electrolyzer for green hydrogen generation. Features temperature-dependent cell polarization curves (Ulleberg formulation), dynamic Faraday efficiency, cell area current density scaling, and waste heat recovery calculation.

---

### Governing Equations

* **Cell Polarization Characteristic (Ulleberg Model)**:
  $$V_{\text{cell}}(j, T) = E_{\text{rev}}(T, p) + \left(r_1 + r_2 T\right) j + s \cdot \ln\left[\left(t_1 + \frac{t_2}{T} + \frac{t_3}{T^2}\right) j + 1\right]$$

* **Electrochemical Faraday Efficiency**:
  $$\eta_F = \frac{j^2}{f_1 + j^2} \cdot f_2$$

* **Hydrogen Mass Production Rate (Faraday's Law)**:
  $$\dot{m}_{\text{H}_2} = \eta_F \cdot \frac{N_{\text{cells}} \cdot I}{z F} \cdot M_{\text{H}_2} \cdot 3600 \quad [\text{kg/hr}]$$

* **Waste Heat Generation**:
  $$Q_{\text{waste}} = P_{\text{actual}} - \dot{m}_{\text{H}_2} \cdot \text{HHV}_{\text{H}_2} \quad [\text{MWth}]$$


---

## Parameters & Attributes

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'PEM_Electrolyzer'` | - | Unique component name identifier |
| `p_nominal_mw` | `float` | `10.0` | MW | Nominal rated electrical power consumption |
| `specific_consumption_kwh_kg` | `float` | `52.0` | kWh/kg | Specific electrical consumption at rated point |
| `min_turndown_pct` | `float` | `10.0` | % | Minimum operational turndown limit (% of nominal) |
| `max_overload_pct` | `float` | `120.0` | % | Maximum allowable transient overload capability |
| `operating_temp_degc` | `float` | `65.0` | °C | Nominal membrane stack operating temperature |
| `operating_pressure_bar` | `float` | `30.0` | bar | Hydrogen delivery cathode pressure |

---

## Co-Simulation Inputs

| Input Variable | Unit | Description |
|---|---|---|
| `P_cmd` | MW | Electrical power command allocated to the electrolyzer |
| `T_coolant_in` | °C | Inlet cooling water temperature (default 30°C) |

---

## Simulation Outputs

| Output Variable | Unit | Description |
|---|---|---|
| `P_actual` | MW | Actual active power absorbed from the power network |
| `m_dot_H2_kg_hr` | kg/hr | Mass flow rate of high-purity hydrogen produced |
| `m_dot_O2_kg_hr` | kg/hr | Co-produced oxygen mass flow rate ($8 	imes \dot{m}_{	ext{H}_2}$) |
| `Q_waste_MW` | MWth | Thermal power dissipated into cooling water system |
| `efficiency_pct_lhv` | % | Operating system efficiency based on Lower Heating Value (LHV) |

---

## Python Usage Example

```python
from energysim import world
from energysim.models.hydrogen import PEMElectrolyzer

h2_stack = PEMElectrolyzer(name="pem", p_nominal_mw=10.0, specific_consumption_kwh_kg=52.0)
sim = world(start_time=0, stop_time=3600, t_macro=900)
sim.add_model("pem", model=h2_stack, outputs=["P_actual", "m_dot_H2_kg_hr", "Q_waste_MW"])
sim.add_signal("power_setpoint", lambda t: 8.5)
sim.add_connections({"power_setpoint.y": "pem.P_cmd"})
sim.simulate(dashboard=False)
```

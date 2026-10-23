# Battery Energy Storage System (BESS)

> **Class**: `BatteryBESS`  
> **Module**: `energysim.models.storage`

## Overview

Lithium-ion Battery Energy Storage System (LFP or NMC chemistry) modeled with an Equivalent Circuit Model (ECM), state-of-charge dependent open circuit voltage, internal resistance, lumped single-state thermal dynamics, and combined calendar + cyclic SOH degradation.

---

### Governing Equations

* **Open-Circuit Voltage**:
  $$V_{\text{oc}} = V_{\text{base}} + \Delta V_{\text{soc}} \cdot \text{SoC} - \Delta V_{\text{exp}} \cdot e^{-\beta_{\text{soc}} \cdot \text{SoC}}$$

* **DC Equivalent Circuit**:
  $$V_t = V_{\text{oc}} + I \cdot R_{\text{in}}(\text{SoC})$$
  $$P_{\text{dc}} = V_t \cdot I$$

* **Power Conversion System (Inverter) Efficiency**:
  $$P_{\text{grid}} = \begin{cases} P_{\text{dc}} / \eta_{\text{inv}}(P), & \text{discharging } (P < 0) \\ P_{\text{dc}} \cdot \eta_{\text{inv}}(P), & \text{charging } (P > 0) \end{cases}$$

* **Lumped Thermal ODE**:
  $$C_{\text{th}} \frac{dT_{\text{batt}}}{dt} = I^2 R_{\text{in}} - \frac{T_{\text{batt}} - T_{\text{amb}}}{R_{\text{th}}}$$

* **State of Health (SOH) Degradation**:
  $$\Delta \text{SOH} = k_{\text{cal}} \cdot e^{-\frac{E_a}{R T_{\text{batt}}}} \cdot \Delta t + \frac{|P_{\text{dc}}| \Delta t}{2 \cdot E_{\text{cap}} \cdot N_{\text{cycle}}}$$


---

## Parameters & Attributes

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'BESS'` | - | Unique component name identifier |
| `capacity_mwh` | `float` | `4.0` | MWh | Total rated electrical energy storage capacity |
| `p_max_mw` | `float` | `2.0` | MW | Maximum continuous active power rating (charge/discharge) |
| `soc_init` | `float` | `0.5` | 0-1 | Initial battery state of charge fraction |
| `soc_min` | `float` | `0.05` | 0-1 | Lower state of charge operational cutoff threshold |
| `soc_max` | `float` | `0.95` | 0-1 | Upper state of charge operational cutoff threshold |
| `eta_charge` | `float` | `0.96` | 0-1 | One-way coulombic charge efficiency |
| `eta_discharge` | `float` | `0.96` | 0-1 | One-way coulombic discharge efficiency |
| `t_cell_init_degc` | `float` | `25.0` | °C | Initial battery pack cell temperature |
| `c_th_kj_k` | `float` | `15000.0` | kJ/K | Lumped thermal heat capacity of the battery pack |
| `r_th_k_kw` | `float` | `0.05` | K/kW | Thermal resistance between pack and ambient environment |

---

## Co-Simulation Inputs

| Input Variable | Unit | Description |
|---|---|---|
| `P_cmd` | MW | Active power setpoint (positive = charge, negative = discharge) |
| `T_amb` | °C | Ambient surrounding temperature for convective cooling |

---

## Simulation Outputs

| Output Variable | Unit | Description |
|---|---|---|
| `P_actual` | MW | Actual active power delivered to or absorbed from the grid |
| `SoC` | 0-1 | Current battery state of charge fraction |
| `T_batt` | °C | Internal lumped battery pack temperature |
| `SOH` | 0-1 | Remaining battery state of health capacity fraction |
| `V_terminal` | V | DC bus terminal voltage across the cell string |
| `I_current` | A | DC battery current (positive = charging) |
| `Q_heat_kW` | kW | Ohmic joule heat dissipation generated inside the pack |

---

## Python Usage Example

```python
from energysim import world
from energysim.models.storage import BatteryBESS

# 1. Instantiate 4 MWh / 2 MW BESS
bess = BatteryBESS(name="bess", capacity_mwh=4.0, p_max_mw=2.0, soc_init=0.5)

# 2. Add to co-simulation canvas
sim = world(start_time=0, stop_time=3600, t_macro=900)
sim.add_model("bess", model=bess, outputs=["P_actual", "SoC", "T_batt"])
sim.add_signal("charge_cmd", lambda t: 1.5)  # Charge at 1.5 MW
sim.add_connections({"charge_cmd.y": "bess.P_cmd"})

# 3. Simulate
sim.simulate(dashboard=False)
results = sim.results(dashboard=False)
print(results["bess"].head())
```

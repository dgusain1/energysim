# Industrial Heat Pump

> **Class**: `IndustrialHeatPump`  
> **Module**: `energysim.models.hvac_thermal`

## Overview

High-temperature industrial heat pump extracting low-grade waste heat (from data centers, electrolyzers, or industrial effluents) and lifting it to district heating or steam temperatures.

---

### Governing Equations

* **Carnot Coefficient of Performance (COP)**:
  $$\text{COP}_{\text{Carnot}} = \frac{T_{\text{sink}} + 273.15}{(T_{\text{sink}} + 273.15) - (T_{\text{source}} + 273.15)}$$
  $$\text{COP}_{\text{actual}} = \eta_{\text{Carnot}} \cdot \text{COP}_{\text{Carnot}}$$

* **Thermal Heat Delivered**:
  $$Q_{\text{heat}} = P_{\text{elec}} \cdot \text{COP}_{\text{actual}}$$
  $$Q_{\text{source,absorbed}} = Q_{\text{heat}} - P_{\text{elec}}$$


---

## Parameters & Attributes

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'HeatPump'` | - | Unique component name identifier |
| `p_electric_max_mw` | `float` | `2.0` | MW | Maximum electrical compressor power rating |
| `t_source_nom_c` | `float` | `30.0` | °C | Nominal low-grade waste heat source temperature |
| `t_sink_nom_c` | `float` | `85.0` | °C | Nominal high-grade delivery temperature |
| `eta_carnot` | `float` | `0.55` | 0-1 | Carnot efficiency fraction factor (typically 0.45-0.60) |

---

## Co-Simulation Inputs

| Input Variable | Unit | Description |
|---|---|---|
| `P_cmd` | MW | Compressor electrical power setpoint |
| `T_source` | °C | Dynamic evaporator heat source inlet temperature |
| `T_sink` | °C | Dynamic condenser delivery temperature |

---

## Simulation Outputs

| Output Variable | Unit | Description |
|---|---|---|
| `P_actual` | MW | Electrical compressor power absorbed |
| `Q_heat_MW` | MWth | Useful thermal heat injected into heat network |
| `Q_absorbed_MW` | MWth | Waste heat extracted from evaporator source |
| `COP` | - | Actual operating Coefficient of Performance |

---

## Python Usage Example

```python
from energysim import world
from energysim.models.hvac_thermal import IndustrialHeatPump

hp = IndustrialHeatPump(name="hp", p_electric_max_mw=2.0, t_source_nom_c=30.0, t_sink_nom_c=85.0)
sim = world(start_time=0, stop_time=3600, t_macro=900)
sim.add_model("hp", model=hp, outputs=["P_actual", "Q_heat_MW", "COP"])
sim.add_signal("power_cmd", lambda t: 1.5)
sim.add_connections({"power_cmd.y": "hp.P_cmd"})
sim.simulate(dashboard=False)
```

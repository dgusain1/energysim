# Combined Cycle Gas Turbine (CCGT)

> **Class**: `CombinedCycle_CCGT`  
> **Module**: `energysim.models.thermal`

## Overview

Combined Cycle Gas Turbine (CCGT) combining a Brayton gas turbine and a Rankine heat recovery steam turbine (HRSG). Features non-linear part-load heat rate curves, ambient temperature capacity derating, Cold/Warm/Hot start sequence state machines, and continuous $\text{CO}_2$ emissions accounting.

---

### Governing Equations

* **Power Split**:
  $$P_{\text{actual}} = P_{\text{GT}} + P_{\text{ST}} \quad (P_{\text{GT}} \approx 0.67 P_{\text{actual}}, \quad P_{\text{ST}} \approx 0.33 P_{\text{actual}})$$

* **Ambient Temperature Capacity Derating**:
  $$P_{\text{max,avail}} = P_{\text{rated}} \cdot f_{\text{amb}}(T_{\text{amb}})$$

* **Part-Load Heat Rate & Fuel Consumption**:
  $$\text{HR}(\text{Load}) = \text{HR}_{\text{full}} \cdot \left[1.0 + 0.32 \cdot (1.0 - \text{Load})^{1.8}\right]$$
  $$\text{Fuel}_{\text{MMBtu/hr}} = \frac{P_{\text{actual}} \cdot 1000 \cdot \text{HR}(\text{Load})}{1.055 \times 10^6}$$

* **Carbon Emissions Accounting**:
  $$\text{CO}_{2,\text{rate}} = \text{Fuel}_{\text{MMBtu/hr}} \cdot 0.0592 \quad [\text{tCO}_2/\text{hr}]$$


---

## Parameters & Attributes

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'CCGT'` | - | Unique component name identifier |
| `p_rated_mw` | `float` | `400.0` | MW | Total combined cycle plant rated capacity (GT + ST) |
| `gt_ratio` | `float` | `0.67` | 0-1 | Fraction of capacity delivered by gas turbine (typically 2/3) |
| `heat_rate_full_load_kj_kwh` | `float` | `6400.0` | kJ/kWh | Full load LHV heat rate (~56.2% thermal efficiency) |
| `min_stable_gen_pct` | `float` | `40.0` | % | Minimum stable generation (MSG) environmental cutoff |
| `ramp_rate_mw_min` | `float` | `15.0` | MW/min | Maximum mechanical ramp rate constraint |

---

## Co-Simulation Inputs

| Input Variable | Unit | Description |
|---|---|---|
| `P_cmd` | MW | Active power generation dispatch order |
| `T_amb` | °C | Ambient compressor inlet air temperature |

---

## Simulation Outputs

| Output Variable | Unit | Description |
|---|---|---|
| `P_actual` | MW | Total combined active power delivered to the transmission bus |
| `P_gt` | MW | Gas turbine electrical power output |
| `P_st` | MW | Steam turbine electrical power output |
| `fuel_consumption_MMBtu_hr` | MMBtu/hr | Natural gas fuel firing rate |
| `CO2_emissions_t_hr` | t/hr | Instantaneous carbon dioxide emission rate |
| `Q_condenser_MW` | MWth | Cooling tower / condenser heat rejection load |

---

## Python Usage Example

```python
from energysim import world
from energysim.models.thermal import CombinedCycle_CCGT

ccgt = CombinedCycle_CCGT(name="ccgt", p_rated_mw=40.0, min_stable_gen_pct=35.0)
sim = world(start_time=0, stop_time=86400, t_macro=900)
sim.add_model("ccgt", model=ccgt, outputs=["P_actual", "fuel_consumption_MMBtu_hr", "CO2_emissions_t_hr"])
sim.add_signal("dispatch", lambda t: 30.0)
sim.add_connections({"dispatch.y": "ccgt.P_cmd"})
sim.simulate(dashboard=False)
```

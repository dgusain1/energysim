# Equipment Catalog & Archetypes

> **Class**: `energysim.models.catalog`  
> **Module**: `energysim.models.catalog`

## Overview

Pre-calibrated equipment archetypes configured against manufacturer datasheets and test-bench measurements.

---

### Equipment Archetypes List

* **`catalog.Tesla_Megapack_2XL`**: Utility-scale LFP BESS ($3.9\text{ MWh}$, $1.9\text{ MW}$, $92\%$ round-trip efficiency, liquid cooled).
* **`catalog.Nel_A1000`**: Commercial atmospheric alkaline water electrolyzer ($2.2\text{ MW}$, $388\text{ Nm}^3/\text{hr}$, $20\%$ turndown cutoff).
* **`catalog.ITM_Power_Trident`**: Multi-megawatt PEM electrolyzer stack ($5.0\text{ MW}$, $30\text{ bar}$, fast ramping).
* **`catalog.Bloom_Energy_SOEC`**: Solid Oxide Electrolyzer ($1.0\text{ MW}$, steam integrated, $37.5\text{ kWh/kg}$).
* **`catalog.GE_7HA_02`**: Heavy-duty Combined Cycle Gas Turbine ($570\text{ MW}$, $64.0\%$ net efficiency, fast startup).
* **`catalog.Siemens_SGT_800`**: Industrial gas turbine ($54\text{ MW}$, simple cycle or CHP).
* **`catalog.CanadianSolar_HiKu7`**: Monocrystalline PERC PV array archetype ($665\text{ W}$ module rating, $-0.34\%/^\circ\text{C}$).
* **`catalog.Vestas_V150_4MW`**: Low-to-medium wind regime turbine ($4.2\text{ MW}$, $150\text{ m}$ rotor diameter).


---

## Parameters & Attributes

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'catalog_item'` | - | Instance name identifier |
| `custom_kwargs` | `Any` | `Overridable` | - | Any manufacturer parameter can be overridden during instantiation |

---

## Co-Simulation Inputs

| Input Variable | Unit | Description |
|---|---|---|
| `See underlying base model inputs` | - | - |

---

## Simulation Outputs

| Output Variable | Unit | Description |
|---|---|---|
| `See underlying base model outputs` | - | - |

---

## Python Usage Example

```python
from energysim.models import catalog

bess = catalog.Tesla_Megapack_2XL(name="bess", soc_init=0.6)
pem = catalog.ITM_Power_Trident(name="pem")
turbine = catalog.Vestas_V150_4MW(name="vestas")
```

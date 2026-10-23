# Methanation Reactor (Power-to-Gas)

> **Class**: `MethanationReactor`  
> **Module**: `energysim.models.hydrogen`

## Overview

Catalytic Sabatier methanation reactor converting green hydrogen and captured industrial carbon dioxide into synthetic natural gas (SNG / $	ext{CH}_4$) with high-temperature exothermic heat release.

---

### Governing Equations

* **Sabatier Chemical Reaction**:
  $$\text{CO}_2 + 4\text{H}_2 \to \text{CH}_4 + 2\text{H}_2\text{O} \quad (\Delta H_{298\text{K}} = -165\text{ kJ/mol})$$

* **Stoichiometric Reactant Mass Balance**:
  $$\dot{m}_{\text{CH}_4} = \dot{m}_{\text{H}_2} \cdot \frac{M_{\text{CH}_4}}{4 M_{\text{H}_2}} \cdot \eta_{\text{conversion}}$$


---

## Parameters & Attributes

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'Methanation'` | - | Unique component name identifier |
| `capacity_mw_sng` | `float` | `5.0` | MW | Rated synthetic methane energy production capacity |
| `conversion_efficiency` | `float` | `0.82` | 0-1 | Chemical conversion yield factor |

---

## Co-Simulation Inputs

| Input Variable | Unit | Description |
|---|---|---|
| `m_dot_H2_in_kg_hr` | kg/hr | Green hydrogen feed mass flow rate |
| `m_dot_CO2_in_kg_hr` | kg/hr | Captured carbon dioxide feed flow rate |

---

## Simulation Outputs

| Output Variable | Unit | Description |
|---|---|---|
| `m_dot_CH4_kg_hr` | kg/hr | Synthetic natural gas production mass flow rate |
| `Q_exothermic_MWth` | MWth | High-temperature reaction waste heat available for steam generation |

---

## Python Usage Example

```python
from energysim.models.hydrogen import MethanationReactor

p2g = MethanationReactor(name="methanation", capacity_mw_sng=5.0)
```

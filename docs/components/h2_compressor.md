# Hydrogen Compressor

> **Class**: `H2Compressor`  
> **Module**: `energysim.models.hydrogen`

## Overview

Multi-stage reciprocating or centrifugal hydrogen compressor with intercooling, compressing gas from generation pressure (e.g. 30 bar) to pipeline (e.g. 100 bar) or storage (e.g. 350–700 bar).

---

### Governing Equations

* **Multi-Stage Isentropic Compression Power**:
  $$W_{\text{comp}} = \frac{N_{\text{stage}}}{\eta_{\text{is}}} \cdot \dot{m} \cdot Z \cdot R \cdot T_{\text{in}} \cdot \frac{\gamma}{\gamma - 1} \left[\left(\frac{P_{\text{out}}}{P_{\text{in}}}\right)^{\frac{\gamma - 1}{N_{\text{stage}} \gamma}} - 1\right]$$


---

## Parameters & Attributes

| Parameter | Type | Default | Unit | Description |
|---|---|---|---|---|
| `name` | `str` | `'H2_Compressor'` | - | Unique component name identifier |
| `p_in_bar` | `float` | `30.0` | bar | Suction inlet pressure |
| `p_out_bar` | `float` | `350.0` | bar | Discharge delivery pressure |
| `eta_isentropic` | `float` | `0.75` | 0-1 | Stage isentropic efficiency |
| `num_stages` | `int` | `3` | - | Number of intercooled compression stages |

---

## Co-Simulation Inputs

| Input Variable | Unit | Description |
|---|---|---|
| `m_dot_in_kg_hr` | kg/hr | Inlet mass flow rate of hydrogen to compress |

---

## Simulation Outputs

| Output Variable | Unit | Description |
|---|---|---|
| `P_electric_MW` | MW | Electrical motor compression power consumed |
| `m_dot_out_kg_hr` | kg/hr | High-pressure delivery flow rate |
| `Q_intercooling_kW` | kW | Thermal heat rejected through intercoolers |

---

## Python Usage Example

```python
from energysim.models.hydrogen import H2Compressor

comp = H2Compressor(name="comp", p_in_bar=30.0, p_out_bar=350.0)
```

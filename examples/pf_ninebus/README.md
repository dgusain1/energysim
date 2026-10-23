# PowerFactory 9-Bus Co-Simulation Example

## Overview

This example demonstrates co-simulation of a **9-bus power system** modelled
in DIgSILENT PowerFactory with **Power-to-Heat (P2H)** and
**Power-to-Gas (P2G)** flexible loads, driven by renewable generation and
coordinated by a Python dispatch controller.

The setup mirrors the *Optimal Dispatch of Flexible Energy Resources* example
but uses PowerFactory instead of pandapower for the grid model.

## Architecture

```
┌──────────┐       ┌──────────────────────┐       ┌──────────────┐
│  weather  │──────▶│  dispatch_controller │──────▶│  PowerFactory │
│  (CSV)    │       │  (external Python)   │◀──────│  9-bus grid   │
└──────────┘       └──────────────────────┘       └──────────────┘
```

| Component             | sim_type        | Description                              |
|-----------------------|-----------------|------------------------------------------|
| Weather / renewables  | `csv`           | Wind, PV, T_ambient, electricity price   |
| Dispatch controller   | `external`      | Sets P2H / P2G active power commands     |
| 9-bus grid            | `powerfactory`  | Runs load flow via PF Python API         |

## Prerequisites

1. **DIgSILENT PowerFactory 2020+** with a valid licence.
2. A PowerFactory project containing the 9-bus model (see below).
3. The `powerfactory` Python module must be importable, or pass
   `pf_path` pointing to your PF Python directory.

## PowerFactory Model Setup

Create a 9-bus network in PowerFactory with:

- **9 buses** (Bus1 – Bus9)
- **3 generators** (Gen1 = slack, Gen2, Gen3)
- **3 loads**:
  - `Load_Conv` — conventional load (~125 MW)
  - `Load_P2H`  — Power-to-Heat (~50 MW adjustable)
  - `Load_P2G`  — Power-to-Gas (~50 MW adjustable)
- **Renewable sources** (as negative loads or static generators):
  - `WindFarm` — wind generation
  - `PVFarm`   — solar PV generation

## Configuration

Open `run_pf_ninebus.py` and fill in the placeholder values marked with
`# <-- CONFIGURE`:

```python
PF_PROJECT_NAME = "YourProjectName"      # PowerFactory project name
PF_PATH         = r"C:\DIgSILENT\..."    # Path to PF Python directory
```

Also check element names match your PowerFactory model in the
`inputs` / `outputs` lists.

## Run

```bash
python run_pf_ninebus.py
```

## Output

- Console progress bar (96 macro time steps, 15-min intervals, 24 h)
- A `Results` object (`world.last_results`); nothing is written to disk unless you set `out_dir`
- Plots of bus voltages, P2H/P2G dispatch, and renewable generation

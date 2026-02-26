# MPC Microgrid Co-simulation Example

This example demonstrates **Model Predictive Control** (MPC) for a
residential microgrid, using all five energysim simulator types in a
single co-simulation.

## Simulator types used

| Component | `sim_type` | File | Description |
|-----------|-----------|------|-------------|
| Weather / load / price | `csv` | `weather.csv` | 24 h synthetic profiles (15-min) |
| Comfort setpoint | `signal` | *(inline lambda)* | Night setback 17 °C / day 21 °C |
| Battery (13.5 kWh) | `matlab` | `models/battery_mpc.m` | Octave/MATLAB Li-ion model with persistent SoC |
| Building thermal | `external` | `simulators/building.py` | Python lumped-capacitance RC model |
| MPC controller | `external` | `simulators/mpc_controller.py` | Python receding-horizon optimiser (scipy SLSQP) |
| LV grid | `powerflow` | `microgrid.p` | 5-bus pandapower network |

## Coupling topology

```
  weather.csv ─── T_ambient ──►  building  ◄── P_hp_cmd ─┐
       │          solar     ──►            │              │
       │                        T_inside ──► MPC          │
       ├── P_pv ──────────────────────────► controller ──┤
       ├── P_load ────────────────────────►    │          │
       ├── elec_price ────────────────────►    │          │
       │                                       │          │
  setpoint ── T_setpoint ─────────────────►    │          │
       │                                  P_batt_cmd      │
       │                                       │       P_hp_cmd
       │                                       ▼          │
       │                              battery_mpc.m       │
       │                              SoC ──► controller  │
       │                              P_actual            │
       │                                  │               │
       └── P_pv ──► grid ◄── P_actual ───┘               │
            P_load ─►    ◄── P_hp_cmd ───────────────────┘
```

## MPC formulation

At each 15-minute macro step the controller solves:

$$\min \sum_{k=0}^{N-1} \text{price}_k \cdot \max(0,\; P_\text{grid,k}) \cdot \Delta t$$

subject to:
- Battery SoC dynamics and bounds (10 % – 95 %)
- Building temperature dynamics and comfort bounds (T_min – T_max)
- Actuator power limits

The horizon is **N = 16 steps (4 hours)**. Only the first action is
applied (receding horizon). If scipy is not available, the controller
falls back to a rule-based heuristic.

## Requirements

- Python >= 3.9
- energysim (from `src/`)
- pandapower
- scipy
- GNU Octave on PATH + `oct2py`  — or MATLAB Engine

## How to run

```bash
cd examples/mpc_microgrid
python run_mpc.py
```

The script will:
1. Generate `microgrid.p` (pandapower) and `weather.csv` if missing
2. Run a 24-hour co-simulation (96 steps x 900 s)
3. Print results and open an interactive dashboard

## Files

```
mpc_microgrid/
├── generate_data.py          # creates network + CSV
├── run_mpc.py                # main co-simulation script
├── models/
│   └── battery_mpc.m         # Octave/MATLAB battery model
├── simulators/
│   ├── building.py           # Python building thermal model
│   └── mpc_controller.py     # Python MPC controller
└── README.md
```

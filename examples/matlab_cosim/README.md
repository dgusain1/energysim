# MATLAB / Octave Co-simulation Example

This example demonstrates energysim's **MATLAB / GNU Octave adapter** by
coupling three `.m` function models in a residential heating co-simulation.

## Models

| Model | File | Inputs | Outputs | Description |
|-------|------|--------|---------|-------------|
| Heat pump | `models/heatpump.m` | `P_electric`, `T_source`, `T_sink` | `Q_thermal`, `COP` | Carnot-limited air-source heat pump (η_ex = 0.45) |
| Battery | `models/battery.m` | `P_cmd` | `SoC`, `P_actual` | 13.5 kWh Li-ion battery with SoC limits |
| Thermal mass | `models/thermal_mass.m` | `Q_heating`, `T_ambient` | `T_inside` | Single-zone lumped-capacitance building (C = 2500 kJ/K) |

A **CSV** simulator provides a synthetic outdoor temperature and load profile
(24 h, 15-min resolution).

## Coupling topology

```
weather.csv ──T_ambient──► heatpump ──Q_thermal──► thermal_mass
     │                        ▲                        │
     └──T_ambient─────────────┼──────► thermal_mass    │
                              └────────T_inside────────┘
                         (feedback loop)
```

## Requirements

* Python ≥ 3.9
* energysim (install from `src/`)
* **One of:**
  - GNU Octave ≥ 6.0 on PATH + `pip install oct2py`
  - MATLAB R2020b+ with MATLAB Engine for Python installed

The adapter auto-detects the available engine.  
To force one, set `engine='matlab'` or `engine='octave'` in the
`add_simulator()` calls.

## How to run

```bash
cd examples/matlab_cosim
python run_matlab_cosim.py
```

The script will:
1. Generate a synthetic `weather.csv`
2. Launch the MATLAB/Octave engine
3. Run a 24-hour co-simulation (96 steps × 900 s)
4. Print and save results to `results.csv`
5. Open the interactive energysim dashboard (if possible)

## Writing your own `.m` functions

Every `.m` model must be a **function** (not a script) with this signature:

```matlab
function [out1, out2, ...] = my_model(in1, in2, ..., time)
```

Rules:
- **`time`** is always the **last** input argument — energysim appends it
  automatically.
- The number of declared inputs (minus `time`) must match the `inputs` list
  passed to `add_simulator()`.
- The number of declared outputs must match the `outputs` list.
- Use **`persistent`** variables to keep state across time-steps (see
  `battery.m` and `thermal_mass.m` for examples).
- The function file name (without `.m`) must match the function name inside.

## Notes

- The battery model uses `persistent` variables for SoC tracking. These are
  reset each time the Octave/MATLAB engine restarts.
- All models receive the macro time-step boundary time in seconds.
- The heat-pump COP is clamped to [1, 8] for numerical stability.

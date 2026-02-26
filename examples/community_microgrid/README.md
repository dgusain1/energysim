# Community Microgrid — Multi-Physics Co-Simulation Example

A comprehensive community microgrid co-simulation with **electrochemical battery
models**, **thermal building envelopes**, **temperature-derated PV**, and
**multi-zone commercial HVAC** — all orchestrated by an MPC dispatcher.

## Topology

```
                    ┌──────────────────────────┐
                    │     Utility Grid (0.4 kV) │
                    │         (Slack Bus)        │
                    └───────┬──────────┬────────┘
                            │          │
          ┌─────────────────┼──────────┼─────────────────┐
          │                 │          │                  │
     ┌────┴────┐     ┌─────┴────┐  ┌──┴───────┐  ┌──────┴──────┐
     │ House 1 │     │  House 2 │  │   Mall   │  │ Supermarket │
     │ EV + PV │     │ (single) │  │ (EV stn) │  │ (refrig.)   │
     └─────────┘     └──────────┘  └──────────┘  └─────────────┘
          │                                │
     ┌────┴────────────────────────────────┤
     │         Community Battery           │
     │    (50 kWh NMC622 electrochemical)  │
     └─────────────────────────────────────┘
```

## Multi-Physics Models

### Community Battery (`models/community_battery.m`)
- **Cell chemistry**: NMC622 / graphite, 21700 cylindrical, 5 Ah
- **Pack**: 14s × 193p = 2702 cells, 50.1 kWh, 51.8 V nominal
- **Electrochemical**: 2nd-order equivalent circuit (2-RC)
  - OCV from Redlich-Kister thermodynamic expansion
  - R₀ (ohmic, SoC-dependent electrolyte conductivity)
  - R₁-C₁ (charge transfer, Butler-Volmer linearisation)
  - R₂-C₂ (solid-state diffusion, Fickian SPM)
- **Thermal**: Lumped model with Joule + entropic heat, forced-air cooling
- **Kinetics**: Arrhenius temperature dependence on all parameters
- **Aging**: SEI calendar (Arrhenius + SoC stress) + cycle throughput (√Ah)

### Household — 2-person with EV + PV (`simulators/household_2p_ev_pv.py`)
- **Building thermal**: 2R-1C lumped envelope (80 m², concrete, 8 MJ/K)
  with wall/window heat loss and solar gains through glazing
- **PV**: 5 kWp with NOCT cell temperature model + thermal derating
  (γ = −0.4 %/°C)
- **EV**: 40 kWh NMC pack with electrochemical model:
  - SoC/temperature-dependent internal resistance (Arrhenius)
  - CC-CV charging profile (CC → 80 % SoC → CV taper)
  - Battery thermal model (I²R heating + convective cooling)
  - Cold-weather charging penalty
- **Appliances**: Fridge (ambient-dependent duty cycle), lighting, cooking,
  washing machine, dishwasher, entertainment, standby

### Household — single person (`simulators/household_1p.py`)
- **Building thermal**: 2R-1C for 50 m² flat (lighter construction)
- **Appliances**: Smaller subset, microwave-heavy cooking

### Shopping Mall (`simulators/mall.py`)
- **HVAC**: 3-zone thermal model (retail, food court, corridors)
  - Per-zone R-C dynamics with UA products
  - Enthalpy-based ventilation load (outdoor air)
  - Variable COP (Carnot-fraction model)
- **Lighting**: Daylight harvesting (up to 25 % savings)
- **Vertical transport**: 4 escalators + 3 elevators with 25 % regenerative braking
- **Food court**: Electric equipment + kitchen extract ventilation
- **EV chargers**: 10 bays × 7.4 kW, **controllable** via MPC dispatcher

### Supermarket (`simulators/supermarket.py`)
- **Refrigeration**: Multi-temperature-zone thermodynamic model:
  - Freezer (−25 °C): R-C cabinet, compressor COP(ΔT), 20-min defrost cycles
  - Chiller (+2 °C): Open display case, higher UA
  - Condenser reject heat coupled into store HVAC
- **HVAC**: Store zone thermal model with enthalpy ventilation
- **Lighting**: Daylight harvesting in perimeter zones
- **Bakery/Deli**: Electric ovens with morning baking + midday prep schedule

### MPC Dispatcher (`simulators/dispatcher.py`)
- Rolling 4-hour horizon (16 × 15-min steps)
- Optimises battery charge/discharge + mall EV charger power cap
- Objective: energy cost + peak demand penalty + terminal SoC
- Naive forecasters (persistence + diurnal) tracked for analysis

## Files

| File | Description |
|------|-------------|
| `generate_data.py` | Creates pandapower network + weather CSV |
| `run_community.py` | Main co-simulation script |
| `analyze_forecasts.py` | Forecast analysis + battery dashboard (6 plots) |
| `models/community_battery.m` | Electrochemical battery (MATLAB/Octave) |
| `simulators/household_2p_ev_pv.py` | 2-person household agent |
| `simulators/household_1p.py` | Single-person household agent |
| `simulators/mall.py` | Shopping mall agent |
| `simulators/supermarket.py` | Supermarket agent |
| `simulators/dispatcher.py` | MPC controller + forecast logger |

## Requirements

- Python 3.10+
- energysim (this package)
- pandapower, numpy, pandas, matplotlib
- GNU Octave (or MATLAB) for the battery model

## Running

```bash
# Set Octave on PATH (Windows)
$env:PATH = "C:\Program Files\GNU Octave\Octave-11.1.0\mingw64\bin;" + $env:PATH

# Run co-simulation
python run_community.py

# Run forecast analysis (with co-sim results)
python analyze_forecasts.py --live

# Or standalone demo (synthetic data)
python analyze_forecasts.py
```

## Analysis Plots

The `analyze_forecasts.py` script generates:

1. **Prediction Horizon**: MPC forecast windows overlaid on actual load
2. **Stochastic Fan Chart**: Ensemble scenarios with P10–P90 confidence bands
3. **Forecast Error vs Lead Time**: RMSE/MAE degradation with increasing horizon
4. **Load Decomposition**: Stacked area chart of all community agents
5. **Forecast vs Realised**: 15-min-ahead prediction with error shading
6. **Battery Dashboard**: SoC, terminal voltage, cell temperature, internal resistance

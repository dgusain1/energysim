# Industrial Behind-The-Meter (BTM) Multi-Energy Co-Simulation

This example demonstrates how to use `energysim`'s physical component library to model an industrial refinery / petrochemical complex with:

1. **Solar Photovoltaic Plant** (`PhysicsPV`, 12 MWp)
2. **Battery Energy Storage** (`catalog.BESS_2h_2MW`, generic 2 MW / 4 MWh Li-ion archetype)
3. **Combined Cycle Gas Turbine** (`CombinedCycle_CCGT`, 40 MW; a part-load heat-rate curve, fuel and CO2 only, no steam-side model)
4. **Green Hydrogen Electrolyzer** (`PEMElectrolyzer`, 10 MW with Faraday efficiency & waste heat)
5. **High-Pressure H2 Storage Tank** (`H2StorageTank`, 100 m³, 350 bar with real-gas compressibility)
6. **Industrial Process Steam Header** (`IndustrialProcessSteam`, 40 t/hr)

## Running the Example

```bash
python examples/industrial_btm/run_industrial_btm.py
```

## What Happens in the Simulation

- **Diurnal Renewables**: Solar irradiance peaks at noon ($950\text{ W/m}^2$) while ambient temperature varies between $10^\circ\text{C}$ and $30^\circ\text{C}$.
- **Electrolyzer Flexible Dispatch**: The 10 MW PEM electrolyzer increases production during hours of high solar availability ($9\text{h}-16\text{h}$) to produce green hydrogen, feeding into the 350-bar storage tank.
- **BESS Peak Shaving & Arbitrage**: The battery charges on solar surplus at midday and discharges during the evening process load peak.
- **CCGT Baseload**: Delivers firm electric power, tracking part-load heat rates, fuel use and cumulative $\text{CO}_2$ emissions. The process steam header is a separate demand model.

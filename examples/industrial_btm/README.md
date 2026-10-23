# Industrial Behind-The-Meter (BTM) Multi-Energy Co-Simulation

This example demonstrates how to use `energysim`'s physical component library to model an industrial refinery / petrochemical complex with:

1. **Solar Photovoltaic Plant** (`PhysicsPV`, 12 MWp)
2. **Battery Energy Storage** (`Tesla_Megapack_2XL`, 3.9 MWh / 1.9 MW LFP)
3. **Combined Cycle Gas Turbine** (`CombinedCycle_CCGT`, 40 MW with HRSG steam bottoming)
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
- **BESS Peak Shaving & Arbitrage**: The Tesla Megapack charges on solar surplus at midday and discharges during the evening process load peak.
- **CCGT Baseload**: Delivers firm power and process steam, tracking part-load heat rates and cumulative $\text{CO}_2$ emissions.

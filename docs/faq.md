# Frequently Asked Questions & Troubleshooting

A comprehensive troubleshooting guide and reference for common errors, convergence issues, and multi-energy modeling questions in `energysim`.

---

## 1. Multi-Energy Physical Models & Equipment

### Why is my Alkaline Electrolyzer outputting 0 kg/hr and in `STANDBY`?
**Symptom**: You send a power command of 1.5 MW to a 10 MW alkaline electrolyzer (`AlkalineElectrolyzer`), but `P_actual = 0.0 MW`, `m_dot_H2_kg_hr = 0.0`, and `state == 'STANDBY'`.

**Cause**: Liquid KOH alkaline electrolyzers have a **strict physical minimum turndown limit** (typically $20–30\%$). Below this current density, hydrogen gas crossover through the porous separator into the oxygen chamber can reach explosive concentrations ($>4\%\text{ H}_2\text{ in O}_2$). For safety, the stack automatically trips to `STANDBY`.

**Fix**:
* Ensure your dispatch command exceeds the minimum turndown threshold (e.g., $\ge 2.0\text{ MW}$ for a 10 MW unit).
* If deep part-load flexibility (down to 5–10%) is required, use [`PEMElectrolyzer`](physical-models.md#pemelectrolyzer), which operates with solid polymer membranes and permits much lower turndown.

---

### Why is Solar PV output 0.0 MW even with high irradiance?
**Symptom**: `G_poa = 950 W/m²`, but `P_ac = 0.0 MW`.

**Common Causes**:
1. **Accidental curtailment setting**: If `curtail_limit` was connected to a signal that defaults to `0.0`, the inverter clips AC output to zero.
2. **Nighttime irradiance threshold**: Irradiance below $1.0\text{ W/m}^2$ is treated as nighttime.
3. **Severe temperature derating**: Check `T_cell` and `T_amb` — if cell temperature is unphysically high, power can drop to zero.

**Fix**:
Verify the `curtail_limit` input in your connection dictionary:
```python
# Leave unconstrained (or omit curtail_limit entirely)
pv = PhysicsPV(name="pv", p_dc_stc_mw=10.0, inverter_p_ac_max_mw=8.5)
# Default curtail_limit is 1e6 MW (unconstrained)
```

---

### Why is my CCGT or Gas Turbine not ramping immediately?
**Symptom**: You dispatch 50 MW to a gas turbine, but it produces 0 MW for the first few macro steps.

**Cause**: Real gas turbines and combined-cycle plants cannot jump to full load instantaneously. `energysim` uses an [`OperatingStateMachine`](physical-models.md#gasturbine_ocgt) that enforces:
1. **Startup duration**: Units offline $>48\text{h}$ require a `COLD` start (e.g., 20 minutes) for rotor thermal stress relief and boiler drum purge.
2. **Minimum Stable Generation (MSG)**: Units cannot operate steadily below 30–40% rated capacity.
3. **Physical ramp limits**: Default ramp rates are constrained (e.g. 15–30 MW/min).

**Fix**:
For fast-starting peaking studies, adjust the startup timers:
```python
gt = GasTurbine_OCGT(
    name="peaker",
    p_rated_mw=100.0,
    cold_start_duration_sec=300.0,  # 5 min fast start
    ramp_rate_mw_min=50.0,          # 50 MW/min fast ramp
)
```

---

### Why is BESS terminal voltage ($V_t$) different from Open-Circuit Voltage ($V_{\text{oc}}$)?
**Explanation**: `energysim`'s [`BatteryBESS`](physical-models.md#batterybess) uses a state-dependent Equivalent Circuit Model (ECM):
$$V_t = V_{\text{oc}}(\text{SoC}) + I \cdot R_{\text{in}}(\text{SoC})$$

* During **charging** ($I > 0$): terminal voltage rises above $V_{\text{oc}}$ due to ohmic and polarization resistance.
* During **discharging** ($I < 0$): terminal voltage sags below $V_{\text{oc}}$.
* At low SoC ($<10\%$), $R_{\text{in}}$ naturally increases by $2–3\times$, modeling voltage collapse under heavy discharge.

---

### How do I use real manufacturer equipment instead of guessing parameters?
Use the **Equipment Catalog** ([`energysim.models.catalog`](physical-models.md#6-commercial-equipment-catalog-catalog)):
```python
from energysim.models import catalog

bess = catalog.Tesla_Megapack_2XL(name="bess")  # 1.93 MW / 3.92 MWh LFP
ael  = catalog.Nel_A1000(name="ael")            # 2.2 MW Alkaline
pem  = catalog.ITM_Power_Trident(name="pem")    # 5.0 MW PEM
ccgt = catalog.GE_7HA_02(name="ccgt")           # 573 MW Combined Cycle
```

---

## 2. Co-Simulation & Orchestration Errors

### `ConnectionError: Invalid connection or simulator not found`
**Common Causes**:
1. **Missing dot separator**: `'grid_load1'` instead of `'grid.load1'`.
2. **Simulator added after connection**: Simulators must be added via `add_simulator()` or `add_model()` *before* calling `add_connections()`.
3. **Typo in simulator name**: Name in connection dictionary must match `sim_name` exactly.

---

### Power Unit Mismatches (`kW` vs `MW`)
**Symptom**: Power flow network reports voltages near zero or lines overloading by $1000\times$.

**Cause**: Pandapower expects active power in `MW`, while a controller or BESS was outputting in `kW`.

**Fix**:
`energysim` handles unit conversions automatically if you declare the baseline `power_unit`:
```python
# Declare world baseline in kW:
sim = world(start_time=0, stop_time=3600, power_unit='kW')

# Pandapower natively operates in MW. energysim auto-scales:
# 800 kW signal -> 0.8 MW injected into pandapower!
```

---

### `ValidationError: Pre-flight checks failed`
**Common Causes**:
* **Step size mismatch**: A simulator's `step_size` does not divide the macro step `t_macro` evenly (e.g. `step_size=7` with `t_macro=10`).
* **Writing to read-only simulator**: Connecting an output to a CSV reader or signal generator.
* **Unregistered endpoint**: Connection references a simulator or variable that does not exist.

**Remedy**:
Run validation non-destructively to inspect all issues:
```python
report = my_world.validate(raise_on_error=False)
print("Errors:", report.errors)
print("Warnings:", report.warnings)
```

---

### Algebraic Loop Warning
**Symptom**:
```
UserWarning: Algebraic loop detected involving simulators: ['bess', 'grid_controller']
```

**Cause**: Simulator A's output feeds Simulator B, and Simulator B's output feeds Simulator A within the same time step.

**Remedy**:
* `coupling='jacobi'` (default): Resolves loop using a one-step delay (each simulator sees the previous step's value).
* `coupling='iterative'`: Resolves the loop iteratively within each macro step until values converge within `convergence_tolerance`.

---

## 3. Power Grid & External Simulators

### `pandapower.powerflow.LoadflowNotConverged`
**Common Causes**:
1. **Network islanding**: A bus has no galvanic connection to an external grid (`ext_grid`) or generator with slack voltage control.
2. **Voltage collapse**: Reactive power load exceeds feeder transmission capability.
3. **Unrealistic transformer tap position**.

**Fix**:
Run pandapower diagnostic before coupling:
```python
import pandapower as pp
pp.diagnostic(net)
# Or try running DC power flow mode for dispatch studies:
my_world.add_powerflow('grid', net=net, mode='dc')
```

---

### HDF5 / PyTables: `File already open` or Locking Error
**Cause**: A previous simulation crashed or was aborted without releasing the HDF5 file handle on `es_res.h5`.

**Fix**:
```python
# Force-close any dangling PyTables handles in Python:
import tables as tb
tb.file._open_files.close_all()

# Or specify a unique results file name:
sim = world(res_filename="run_case_01.h5")
```

---

### FMU 32-bit vs 64-bit Platform Mismatch
**Symptom**: `FMPyException: The FMU contains no binary for your platform`.

**Cause**: Trying to simulate a 32-bit compiled FMU on a 64-bit Python installation (or vice versa).

**Fix**:
Verify your Python bitness:
```bash
python -c "import struct; print(struct.calcsize('P') * 8)"
```
Re-export the FMU targeting `win64` / `linux64` from OpenModelica or Dymola.

---

### Octave: `nargin: invalid function name`
**Cause**:
1. The `.m` file is a script, not a function. The first line must be `function [out1, ...] = name(in1, ..., time)`.
2. The function name inside the file does not match the `.m` filename on disk.
3. GNU Octave is not found on your system `PATH`.

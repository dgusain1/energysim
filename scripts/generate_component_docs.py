# -*- coding: utf-8 -*-
"""
Generate comprehensive, pandapower-style element reference pages for all physical models
and vectorized fleet containers in energysim.
"""

import os

docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))
components_dir = os.path.join(docs_dir, "components")
containers_dir = os.path.join(docs_dir, "containers")
os.makedirs(components_dir, exist_ok=True)
os.makedirs(containers_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# COMPONENTS DOCUMENTATION DATA
# -----------------------------------------------------------------------------

COMPONENTS = {
    "bess": {
        "title": "Battery Energy Storage System (BESS)",
        "class": "BatteryBESS",
        "module": "energysim.models.storage",
        "desc": "Lithium-ion Battery Energy Storage System (LFP or NMC chemistry) modeled with an Equivalent Circuit Model (ECM), state-of-charge dependent open circuit voltage, internal resistance, lumped single-state thermal dynamics, and combined calendar + cyclic SOH degradation.",
        "equations": r"""### Governing Equations

* **Open-Circuit Voltage**:
  $$V_{\text{oc}} = V_{\text{base}} + \Delta V_{\text{soc}} \cdot \text{SoC} - \Delta V_{\text{exp}} \cdot e^{-\beta_{\text{soc}} \cdot \text{SoC}}$$

* **DC Equivalent Circuit**:
  $$V_t = V_{\text{oc}} + I \cdot R_{\text{in}}(\text{SoC})$$
  $$P_{\text{dc}} = V_t \cdot I$$

* **Power Conversion System (Inverter) Efficiency**:
  $$P_{\text{grid}} = \begin{cases} P_{\text{dc}} / \eta_{\text{inv}}(P), & \text{discharging } (P < 0) \\ P_{\text{dc}} \cdot \eta_{\text{inv}}(P), & \text{charging } (P > 0) \end{cases}$$

* **Lumped Thermal ODE**:
  $$C_{\text{th}} \frac{dT_{\text{batt}}}{dt} = I^2 R_{\text{in}} - \frac{T_{\text{batt}} - T_{\text{amb}}}{R_{\text{th}}}$$

* **State of Health (SOH) Degradation**:
  $$\Delta \text{SOH} = k_{\text{cal}} \cdot e^{-\frac{E_a}{R T_{\text{batt}}}} \cdot \Delta t + \frac{|P_{\text{dc}}| \Delta t}{2 \cdot E_{\text{cap}} \cdot N_{\text{cycle}}}$$
""",
        "params": [
            ("name", "str", "'BESS'", "-", "Unique component name identifier"),
            ("capacity_mwh", "float", "4.0", "MWh", "Total rated electrical energy storage capacity"),
            ("p_max_mw", "float", "2.0", "MW", "Maximum continuous active power rating (charge/discharge)"),
            ("soc_init", "float", "0.5", "0-1", "Initial battery state of charge fraction"),
            ("soc_min", "float", "0.05", "0-1", "Lower state of charge operational cutoff threshold"),
            ("soc_max", "float", "0.95", "0-1", "Upper state of charge operational cutoff threshold"),
            ("eta_charge", "float", "0.96", "0-1", "One-way coulombic charge efficiency"),
            ("eta_discharge", "float", "0.96", "0-1", "One-way coulombic discharge efficiency"),
            ("t_cell_init_degc", "float", "25.0", "°C", "Initial battery pack cell temperature"),
            ("c_th_kj_k", "float", "15000.0", "kJ/K", "Lumped thermal heat capacity of the battery pack"),
            ("r_th_k_kw", "float", "0.05", "K/kW", "Thermal resistance between pack and ambient environment"),
        ],
        "inputs": [
            ("P_cmd", "MW", "Active power setpoint (positive = charge, negative = discharge)"),
            ("T_amb", "°C", "Ambient surrounding temperature for convective cooling"),
        ],
        "outputs": [
            ("P_actual", "MW", "Actual active power delivered to or absorbed from the grid"),
            ("SoC", "0-1", "Current battery state of charge fraction"),
            ("T_batt", "°C", "Internal lumped battery pack temperature"),
            ("SOH", "0-1", "Remaining battery state of health capacity fraction"),
            ("V_terminal", "V", "DC bus terminal voltage across the cell string"),
            ("I_current", "A", "DC battery current (positive = charging)"),
            ("Q_heat_kW", "kW", "Ohmic joule heat dissipation generated inside the pack"),
        ],
        "example": """from energysim import world
from energysim.models.storage import BatteryBESS

# 1. Instantiate 4 MWh / 2 MW BESS
bess = BatteryBESS(name="bess", capacity_mwh=4.0, p_max_mw=2.0, soc_init=0.5)

# 2. Add to co-simulation canvas
sim = world(start_time=0, stop_time=3600, t_macro=900)
sim.add_model("bess", model=bess, outputs=["P_actual", "SoC", "T_batt"])
sim.add_signal("charge_cmd", lambda t: 1.5)  # Charge at 1.5 MW
sim.add_connections({"charge_cmd.y": "bess.P_cmd"})

# 3. Simulate
sim.simulate()
results = sim.last_results
print(results["bess"].head())"""
    },

    "flow_battery": {
        "title": "Vanadium Redox Flow Battery (VRFB)",
        "class": "FlowBattery",
        "module": "energysim.models.storage",
        "desc": "Vanadium Redox Flow Battery (VRFB) featuring physically decoupled energy capacity (electrolyte tank volume) and power rating (cell stack membrane area). Models dynamic variable-speed electrolyte pump parasitic losses and stack electrochemical efficiency.",
        "equations": r"""### Governing Equations

* **Decoupled Stack Power and Tank Energy**:
  $$E_{\text{tank}} = V_{\text{tank}} \cdot C_{\text{vanadium}} \cdot F \cdot V_{\text{cell,nom}}$$

* **Variable Pumping Parasitic Loss**:
  $$P_{\text{pump}} = P_{\text{base}} + P_{\text{var}} \cdot \left(\frac{|P_{\text{stack}}|}{P_{\text{rated}}}\right)^2$$

* **Net AC Grid Power**:
  $$P_{\text{grid}} = P_{\text{stack}} - P_{\text{pump}}$$
""",
        "params": [
            ("name", "str", "'VRFB'", "-", "Unique component name identifier"),
            ("p_rated_mw", "float", "1.0", "MW", "Rated power of the cell stack"),
            ("capacity_mwh", "float", "8.0", "MWh", "Total electrolyte tank energy capacity (e.g. 8-hour duration)"),
            ("soc_init", "float", "0.5", "0-1", "Initial state of charge fraction of the electrolyte"),
            ("pump_parasitic_pct", "float", "3.0", "%", "Parasitic pumping loss percentage at rated flow"),
            ("eta_stack", "float", "0.85", "0-1", "Electrochemical stack round-trip efficiency"),
        ],
        "inputs": [
            ("P_cmd", "MW", "Active power setpoint (positive = charge, negative = discharge)"),
        ],
        "outputs": [
            ("P_actual", "MW", "Net active power delivered to or absorbed from the grid"),
            ("SoC", "0-1", "Current electrolyte state of charge fraction"),
            ("P_pump_kW", "kW", "Active power consumed by electrolyte circulation pumps"),
        ],
        "example": """from energysim import world
from energysim.models.storage import FlowBattery

vrfb = FlowBattery(name="vrfb", p_rated_mw=1.0, capacity_mwh=8.0, soc_init=0.4)
sim = world(start_time=0, stop_time=7200, t_macro=900)
sim.add_model("vrfb", model=vrfb, outputs=["P_actual", "SoC", "P_pump_kW"])
sim.simulate()"""
    },

    "flywheel": {
        "title": "Flywheel Kinetic Energy Storage",
        "class": "Flywheel",
        "module": "energysim.models.storage",
        "desc": "High-speed mechanical kinetic energy storage system designed for primary frequency regulation and fast synthetic inertia. Models rotational kinetic energy, magnetic bearing losses, and aerodynamic vacuum drag.",
        "equations": r"""### Governing Equations

* **Kinetic Energy Storage**:
  $$E_k = \frac{1}{2} J \omega^2$$

* **Parasitic Self-Discharge (Aerodynamic & Bearing Friction)**:
  $$P_{\text{loss}} = P_{\text{bearing}} + c_{\text{drag}} \cdot \omega^2$$
""",
        "params": [
            ("name", "str", "'Flywheel'", "-", "Unique component name identifier"),
            ("p_rated_mw", "float", "2.0", "MW", "Rated bi-directional power capacity"),
            ("capacity_kwh", "float", "500.0", "kWh", "Total kinetic storage capacity"),
            ("standby_loss_kw", "float", "5.0", "kW", "Vacuum bearing and friction loss at rated speed"),
        ],
        "inputs": [
            ("P_cmd", "MW", "Active power setpoint (positive = charge, negative = discharge)"),
        ],
        "outputs": [
            ("P_actual", "MW", "Net active power exchanged with the grid"),
            ("SoC", "0-1", "Rotational kinetic energy state of charge ($(\omega / \omega_{\text{max}})^2$)"),
            ("speed_rpm", "RPM", "Current rotor rotational speed"),
        ],
        "example": """from energysim.models.storage import Flywheel

fw = Flywheel(name="fw", p_rated_mw=2.0, capacity_kwh=500.0)"""
    },

    "pv": {
        "title": "Photovoltaic Solar Plant (PhysicsPV)",
        "class": "PhysicsPV",
        "module": "energysim.models.renewables",
        "desc": "Empirical physics-based utility solar photovoltaic plant model. Features plane-of-array (POA) irradiance absorption, King/NOCT operating cell temperature derating, single-diode thermal coefficient derating, and Sandia inverter clipping efficiency curves.",
        "equations": r"""### Governing Equations

* **Cell Temperature Estimation (King / NOCT Model)**:
  $$T_{\text{cell}} = T_{\text{amb}} + G_{\text{poa}} \cdot \frac{\text{NOCT} - 20}{800}$$

* **Temperature Derated DC Power**:
  $$P_{\text{dc}} = P_{\text{dc,stc}} \cdot \frac{G_{\text{poa}}}{1000} \cdot [1 + \gamma_{\text{temp}} \cdot (T_{\text{cell}} - 25)] \cdot \eta_{\text{soiling}}$$

* **Inverter Conversion & AC Clipping**:
  $$P_{\text{ac}} = \min(P_{\text{inv,max}}, P_{\text{dc}} \cdot \eta_{\text{inv}}(P_{\text{dc}}))$$
""",
        "params": [
            ("name", "str", "'SolarPV'", "-", "Unique component name identifier"),
            ("p_dc_stc_mw", "float", "10.0", "MWp", "Total DC installed nameplate capacity at Standard Test Conditions (STC)"),
            ("inverter_p_ac_max_mw", "float", "8.5", "MW", "Maximum continuous AC inverter capacity (DC/AC clipping ratio)"),
            ("gamma_p_pct_per_c", "float", "-0.35", "%/°C", "Temperature coefficient of maximum power (Pmp)"),
            ("noct_degc", "float", "45.0", "°C", "Nominal Operating Cell Temperature (NOCT)"),
            ("soiling_loss_pct", "float", "2.0", "%", "Dust, snow, and soiling optical loss factor"),
        ],
        "inputs": [
            ("G_poa", "W/m²", "Plane-of-array global solar irradiance"),
            ("T_amb", "°C", "Ambient surrounding air temperature"),
            ("curtail_p_mw", "MW", "Optional grid operator curtailment cap (default unlimited)"),
        ],
        "outputs": [
            ("P_ac", "MW", "Net active AC power injected into the collector substation"),
            ("P_dc", "MW", "Raw DC power generated by the solar string field before clipping"),
            ("T_cell", "°C", "Operating PV cell junction temperature"),
            ("curtailed_mw", "MW", "Lost active power due to inverter clipping or grid curtailment"),
        ],
        "example": """from energysim import world
from energysim.models.renewables import PhysicsPV

# 12 MWp DC field with 10 MW AC inverter
pv = PhysicsPV(name="solar_pv", p_dc_stc_mw=12.0, inverter_p_ac_max_mw=10.0)

sim = world(start_time=0, stop_time=86400, t_macro=900)
sim.add_model("solar_pv", model=pv, outputs=["P_ac", "T_cell"])
sim.add_signal("irr", lambda t: 950.0 if 21600 <= t <= 64800 else 0.0)
sim.add_signal("temp", lambda t: 25.0)
sim.add_connections({"irr.y": "solar_pv.G_poa", "temp.y": "solar_pv.T_amb"})
sim.simulate()"""
    },

    "wind": {
        "title": "Wind Turbine Generator",
        "class": "WindTurbine",
        "module": "energysim.models.renewables",
        "desc": "Utility-scale wind turbine model featuring piecewise cubic aerodynamic power curves with cut-in, rated, and cut-out wind speed thresholds, alongside air density correction.",
        "equations": r"""### Governing Equations

* **Air Density Correction**:
  $$v_{\text{eff}} = v_{\text{wind}} \cdot \left(\frac{\rho}{\rho_0}\right)^{1/3}$$

* **Aerodynamic Power Curve**:
  $$P_{\text{mech}}(v) = \begin{cases} 0, & v < v_{\text{cut-in}} \text{ or } v > v_{\text{cut-out}} \\ P_{\text{rated}} \cdot \left(\frac{v - v_{\text{cut-in}}}{v_{\text{rated}} - v_{\text{cut-in}}}\right)^3, & v_{\text{cut-in}} \le v < v_{\text{rated}} \\ P_{\text{rated}}, & v_{\text{rated}} \le v \le v_{\text{cut-out}} \end{cases}$$
""",
        "params": [
            ("name", "str", "'WindTurbine'", "-", "Unique component name identifier"),
            ("p_rated_mw", "float", "4.0", "MW", "Rated nameplate electrical generator capacity"),
            ("v_cut_in_m_s", "float", "3.0", "m/s", "Cut-in wind speed threshold"),
            ("v_rated_m_s", "float", "12.0", "m/s", "Rated wind speed threshold"),
            ("v_cut_out_m_s", "float", "25.0", "m/s", "Cut-out storm shutdown wind speed threshold"),
            ("hub_height_m", "float", "120.0", "m", "Rotor hub height above ground"),
        ],
        "inputs": [
            ("wind_speed_m_s", "m/s", "Wind speed at hub height"),
            ("air_density_kg_m3", "kg/m³", "Ambient air density (default 1.225 kg/m³)"),
        ],
        "outputs": [
            ("P_actual", "MW", "Net active power delivered to the collector substation"),
            ("status", "str", "Operating status ('RUNNING', 'PARKED', 'STORM_CUTOUT')"),
        ],
        "example": """from energysim.models.renewables import WindTurbine

wt = WindTurbine(name="turbine", p_rated_mw=4.0, v_cut_in_m_s=3.0, v_rated_m_s=11.5)"""
    },

    "solar_thermal": {
        "title": "Solar Thermal Collector",
        "class": "SolarThermalCollector",
        "module": "energysim.models.renewables",
        "desc": "Parabolic trough or flat plate solar thermal collector modeled via the Hottel-Whillier-Bliss efficiency equations for district heating networks or industrial process heat.",
        "equations": r"""### Governing Equations

* **Hottel-Whillier-Bliss Efficiency**:
  $$\eta_{\text{th}} = \eta_0 - a_1 \frac{T_m - T_{\text{amb}}}{G} - a_2 \frac{(T_m - T_{\text{amb}})^2}{G}$$

* **Thermal Power Output**:
  $$Q_{\text{th}} = A_{\text{collector}} \cdot G \cdot \eta_{\text{th}} = \dot{m} c_p (T_{\text{out}} - T_{\text{in}})$$
""",
        "params": [
            ("name", "str", "'SolarThermal'", "-", "Unique component name identifier"),
            ("area_m2", "float", "5000.0", "m²", "Total collector aperture area"),
            ("eta0", "float", "0.78", "0-1", "Optical peak efficiency factor"),
            ("a1", "float", "3.5", "W/(m²·K)", "Linear heat loss coefficient"),
            ("a2", "float", "0.015", "W/(m²·K²)", "Quadratic heat loss coefficient"),
        ],
        "inputs": [
            ("G_dni", "W/m²", "Direct normal or global irradiance"),
            ("T_amb", "°C", "Ambient temperature"),
            ("T_in", "°C", "Inlet heat transfer fluid temperature"),
            ("m_dot_kg_s", "kg/s", "Fluid mass flow rate"),
        ],
        "outputs": [
            ("Q_th_MW", "MWth", "Thermal power transferred to heat transfer fluid"),
            ("T_out", "°C", "Collector fluid outlet temperature"),
        ],
        "example": """from energysim.models.renewables import SolarThermalCollector

st = SolarThermalCollector(name="solar_field", area_m2=10000.0)"""
    },

    "pem_electrolyzer": {
        "title": "PEM Electrolyzer",
        "class": "PEMElectrolyzer",
        "module": "energysim.models.hydrogen",
        "desc": "Proton Exchange Membrane (PEM) Electrolyzer for green hydrogen generation. Features temperature-dependent cell polarization curves (Ulleberg formulation), dynamic Faraday efficiency, cell area current density scaling, and waste heat recovery calculation.",
        "equations": r"""### Governing Equations

* **Cell Polarization Characteristic (Ulleberg Model)**:
  $$V_{\text{cell}}(j, T) = E_{\text{rev}}(T, p) + \left(r_1 + r_2 T\right) j + s \cdot \ln\left[\left(t_1 + \frac{t_2}{T} + \frac{t_3}{T^2}\right) j + 1\right]$$

* **Electrochemical Faraday Efficiency**:
  $$\eta_F = \frac{j^2}{f_1 + j^2} \cdot f_2$$

* **Hydrogen Mass Production Rate (Faraday's Law)**:
  $$\dot{m}_{\text{H}_2} = \eta_F \cdot \frac{N_{\text{cells}} \cdot I}{z F} \cdot M_{\text{H}_2} \cdot 3600 \quad [\text{kg/hr}]$$

* **Waste Heat Generation**:
  $$Q_{\text{waste}} = P_{\text{actual}} - \dot{m}_{\text{H}_2} \cdot \text{HHV}_{\text{H}_2} \quad [\text{MWth}]$$
""",
        "params": [
            ("name", "str", "'PEM_Electrolyzer'", "-", "Unique component name identifier"),
            ("p_nominal_mw", "float", "10.0", "MW", "Nominal rated electrical power consumption"),
            ("specific_consumption_kwh_kg", "float", "52.0", "kWh/kg", "Specific electrical consumption at rated point"),
            ("min_turndown_pct", "float", "10.0", "%", "Minimum operational turndown limit (% of nominal)"),
            ("max_overload_pct", "float", "120.0", "%", "Maximum allowable transient overload capability"),
            ("operating_temp_degc", "float", "65.0", "°C", "Nominal membrane stack operating temperature"),
            ("operating_pressure_bar", "float", "30.0", "bar", "Hydrogen delivery cathode pressure"),
        ],
        "inputs": [
            ("P_cmd", "MW", "Electrical power command allocated to the electrolyzer"),
            ("T_coolant_in", "°C", "Inlet cooling water temperature (default 30°C)"),
        ],
        "outputs": [
            ("P_actual", "MW", "Actual active power absorbed from the power network"),
            ("m_dot_H2_kg_hr", "kg/hr", "Mass flow rate of high-purity hydrogen produced"),
            ("m_dot_O2_kg_hr", "kg/hr", "Co-produced oxygen mass flow rate ($8 \times \dot{m}_{\text{H}_2}$)"),
            ("Q_waste_MW", "MWth", "Thermal power dissipated into cooling water system"),
            ("efficiency_pct_lhv", "%", "Operating system efficiency based on Lower Heating Value (LHV)"),
        ],
        "example": """from energysim import world
from energysim.models.hydrogen import PEMElectrolyzer

h2_stack = PEMElectrolyzer(name="pem", p_nominal_mw=10.0, specific_consumption_kwh_kg=52.0)
sim = world(start_time=0, stop_time=3600, t_macro=900)
sim.add_model("pem", model=h2_stack, outputs=["P_actual", "m_dot_H2_kg_hr", "Q_waste_MW"])
sim.add_signal("power_setpoint", lambda t: 8.5)
sim.add_connections({"power_setpoint.y": "pem.P_cmd"})
sim.simulate()"""
    },

    "alkaline_electrolyzer": {
        "title": "Alkaline Electrolyzer (AEL)",
        "class": "AlkalineElectrolyzer",
        "module": "energysim.models.hydrogen",
        "desc": "Alkaline Water Electrolyzer (AEL) characterized by liquid KOH electrolyte. Features an enforced 20% minimum safety turndown cutoff to avoid hazardous hydrogen-in-oxygen gas crossover and thermal warmup inertia.",
        "equations": r"""### Governing Equations

* **Minimum Safety Turndown Cutoff**:
  $$P_{\text{actual}} = \begin{cases} P_{\text{cmd}}, & P_{\text{cmd}} \ge 0.20 \cdot P_{\text{nom}} \\ 0.0 \text{ (Safety Trip / Standby)}, & P_{\text{cmd}} < 0.20 \cdot P_{\text{nom}} \end{cases}$$

* **Gas Crossover Penalty**:
  $$\text{Purity}_{\text{H}_2} = 99.9\% - 0.5\% \cdot \left(\frac{0.20 \cdot P_{\text{nom}}}{P_{\text{actual}}}\right)^2 \quad (\text{near cutoff})$$
""",
        "params": [
            ("name", "str", "'AEL'", "-", "Unique component name identifier"),
            ("p_nominal_mw", "float", "20.0", "MW", "Nominal rated plant power"),
            ("specific_consumption_kwh_kg", "float", "55.0", "kWh/kg", "Specific energy consumption at rated load"),
            ("min_turndown_pct", "float", "20.0", "%", "Strict safety turndown limit (typically 20-25%)"),
            ("warmup_time_min", "float", "30.0", "min", "Cold-start stack warmup duration"),
        ],
        "inputs": [
            ("P_cmd", "MW", "Power dispatch command"),
        ],
        "outputs": [
            ("P_actual", "MW", "Actual active power absorbed"),
            ("m_dot_H2_kg_hr", "kg/hr", "Hydrogen mass production rate"),
            ("state", "str", "Operating status ('RUNNING', 'CUTOFF_TRIP', 'OFF')"),
        ],
        "example": """from energysim.models.hydrogen import AlkalineElectrolyzer

ael = AlkalineElectrolyzer(name="ael", p_nominal_mw=20.0, min_turndown_pct=20.0)"""
    },

    "soec_electrolyzer": {
        "title": "Solid Oxide Electrolyzer (SOEC)",
        "class": "SOECElectrolyzer",
        "module": "energysim.models.hydrogen",
        "desc": "High-temperature Solid Oxide Electrolyzer Cell (SOEC) operating at 700–850°C. Operates via steam electrolysis, utilizing external thermal process steam to dramatically lower electrical consumption.",
        "equations": r"""### Governing Equations

* **Thermodynamic Energy Balance**:
  $$\Delta H = \Delta G + T \Delta S$$
  As temperature increases, electrical energy demand ($\Delta G$) decreases while thermal demand ($T \Delta S$) increases.

* **Steam Consumption Rate**:
  $$\dot{m}_{\text{steam}} = \dot{m}_{\text{H}_2} \cdot \frac{M_{\text{H}_2\text{O}}}{M_{\text{H}_2}} = 8.936 \cdot \dot{m}_{\text{H}_2} \quad [\text{kg/hr}]$$
""",
        "params": [
            ("name", "str", "'SOEC'", "-", "Unique component name identifier"),
            ("p_nominal_mw", "float", "5.0", "MW", "Nominal electrical stack power rating"),
            ("specific_electrical_kwh_kg", "float", "38.0", "kWh/kg", "High-efficiency electrical consumption"),
            ("steam_demand_kg_per_kg_h2", "float", "9.0", "kg/kg", "Steam mass demand per kg hydrogen produced"),
        ],
        "inputs": [
            ("P_cmd", "MW", "Active power setpoint"),
            ("T_steam_in", "°C", "Superheated steam inlet temperature (700-850°C)"),
        ],
        "outputs": [
            ("P_actual", "MW", "Actual electrical power consumed"),
            ("m_dot_H2_kg_hr", "kg/hr", "Hydrogen production rate"),
            ("steam_consumed_t_hr", "t/hr", "Steam mass flow rate consumed from steam network"),
        ],
        "example": """from energysim.models.hydrogen import SOECElectrolyzer

soec = SOECElectrolyzer(name="soec", p_nominal_mw=5.0, specific_electrical_kwh_kg=38.0)"""
    },

    "h2_compressor": {
        "title": "Hydrogen Compressor",
        "class": "H2Compressor",
        "module": "energysim.models.hydrogen",
        "desc": "Multi-stage reciprocating or centrifugal hydrogen compressor with intercooling, compressing gas from generation pressure (e.g. 30 bar) to pipeline (e.g. 100 bar) or storage (e.g. 350–700 bar).",
        "equations": r"""### Governing Equations

* **Multi-Stage Isentropic Compression Power**:
  $$W_{\text{comp}} = \frac{N_{\text{stage}}}{\eta_{\text{is}}} \cdot \dot{m} \cdot Z \cdot R \cdot T_{\text{in}} \cdot \frac{\gamma}{\gamma - 1} \left[\left(\frac{P_{\text{out}}}{P_{\text{in}}}\right)^{\frac{\gamma - 1}{N_{\text{stage}} \gamma}} - 1\right]$$
""",
        "params": [
            ("name", "str", "'H2_Compressor'", "-", "Unique component name identifier"),
            ("p_in_bar", "float", "30.0", "bar", "Suction inlet pressure"),
            ("p_out_bar", "float", "350.0", "bar", "Discharge delivery pressure"),
            ("eta_isentropic", "float", "0.75", "0-1", "Stage isentropic efficiency"),
            ("num_stages", "int", "3", "-", "Number of intercooled compression stages"),
        ],
        "inputs": [
            ("m_dot_in_kg_hr", "kg/hr", "Inlet mass flow rate of hydrogen to compress"),
        ],
        "outputs": [
            ("P_electric_MW", "MW", "Electrical motor compression power consumed"),
            ("m_dot_out_kg_hr", "kg/hr", "High-pressure delivery flow rate"),
            ("Q_intercooling_kW", "kW", "Thermal heat rejected through intercoolers"),
        ],
        "example": """from energysim.models.hydrogen import H2Compressor

comp = H2Compressor(name="comp", p_in_bar=30.0, p_out_bar=350.0)"""
    },

    "h2_tank": {
        "title": "High-Pressure Hydrogen Storage Tank",
        "class": "H2StorageTank",
        "module": "energysim.models.hydrogen",
        "desc": "Compressed gaseous hydrogen buffer storage vessel (Type I steel or Type IV composite, 100–700 bar). Features Redlich-Kwong / NIST real gas compressibility factor calculation $Z(p, T)$ and pressure mass balance.",
        "equations": r"""### Governing Equations

* **Real Gas Compressibility Equation of State**:
  $$p \cdot V = Z(p, T) \cdot \frac{m}{M_{\text{H}_2}} \cdot R \cdot T$$
  At 350 bar, hydrogen compressibility factor $Z \approx 1.25$ (ideal gas law would overestimate capacity by 25%).

* **Dynamic Pressure Evolution**:
  $$\frac{dm}{dt} = \dot{m}_{\text{in}} - \dot{m}_{\text{out}}$$
  $$p(t) = f_{\text{real-gas}}(m(t), V, T)$$
""",
        "params": [
            ("name", "str", "'H2_Tank'", "-", "Unique component name identifier"),
            ("volume_m3", "float", "100.0", "m³", "Internal geometrical tank volume"),
            ("p_max_bar", "float", "350.0", "bar", "Maximum allowable working pressure (MAWP)"),
            ("p_min_bar", "float", "20.0", "bar", "Minimum residual suction pressure"),
            ("p_init_bar", "float", "120.0", "bar", "Initial storage tank pressure"),
            ("temp_c", "float", "25.0", "°C", "Internal gas temperature"),
        ],
        "inputs": [
            ("m_dot_in_kg_hr", "kg/hr", "Inflow mass flow rate entering the tank (from electrolyzer/compressor)"),
            ("m_dot_out_kg_hr", "kg/hr", "Outflow mass flow rate dispatched to industrial loads or fuel cells"),
        ],
        "outputs": [
            ("pressure_bar", "bar", "Current internal tank pressure"),
            ("mass_stored_kg", "kg", "Total mass of hydrogen currently stored in the vessel"),
            ("SoC", "0-1", "Storage capacity fraction between $p_{\text{min}}$ and $p_{\text{max}}$"),
        ],
        "example": """from energysim import world
from energysim.models.hydrogen import H2StorageTank

tank = H2StorageTank(name="tank", volume_m3=100.0, p_max_bar=350.0, p_init_bar=120.0)
sim = world(start_time=0, stop_time=3600, t_macro=900)
sim.add_model("tank", model=tank, outputs=["pressure_bar", "mass_stored_kg"])
sim.add_signal("h2_in", lambda t: 150.0)  # 150 kg/hr in
sim.add_connections({"h2_in.y": "tank.m_dot_in_kg_hr"})
sim.simulate()"""
    },

    "fuel_cell": {
        "title": "Fuel Cell (PEMFC / SOFC)",
        "class": "FuelCell_PEM, FuelCell_SOFC",
        "module": "energysim.models.hydrogen",
        "desc": "Electrochemical fuel cell power systems. Models hydrogen consumption, electrical power generation, electrical efficiency curve derating at part load, and cogeneration waste heat recovery.",
        "equations": r"""### Governing Equations

* **Electrical Power from Hydrogen**:
  $$P_{\text{elec}} = \dot{m}_{\text{H}_2} \cdot \text{LHV}_{\text{H}_2} \cdot \eta_{\text{fc}}(P)$$

* **Combined Heat and Power (CHP) Cogeneration**:
  $$Q_{\text{cogeneration}} = \dot{m}_{\text{H}_2} \cdot \text{HHV}_{\text{H}_2} - P_{\text{elec}}$$
""",
        "params": [
            ("name", "str", "'FuelCell'", "-", "Unique component name identifier"),
            ("p_rated_mw", "float", "2.0", "MW", "Rated electrical power output"),
            ("efficiency_nominal", "float", "0.55", "0-1", "Rated electrical efficiency based on LHV"),
            ("min_load_pct", "float", "10.0", "%", "Minimum operational load fraction"),
        ],
        "inputs": [
            ("P_cmd", "MW", "Active power dispatch setpoint"),
        ],
        "outputs": [
            ("P_actual", "MW", "Net active power delivered to the electrical bus"),
            ("m_dot_H2_consumed_kg_hr", "kg/hr", "Hydrogen fuel consumption rate"),
            ("Q_heat_MW", "MWth", "Cogeneration thermal heat output"),
        ],
        "example": """from energysim.models.hydrogen import FuelCell_PEM

fc = FuelCell_PEM(name="fc", p_rated_mw=2.0)"""
    },

    "methanation": {
        "title": "Methanation Reactor (Power-to-Gas)",
        "class": "MethanationReactor",
        "module": "energysim.models.hydrogen",
        "desc": "Catalytic Sabatier methanation reactor converting green hydrogen and captured industrial carbon dioxide into synthetic natural gas (SNG / $\text{CH}_4$) with high-temperature exothermic heat release.",
        "equations": r"""### Governing Equations

* **Sabatier Chemical Reaction**:
  $$\text{CO}_2 + 4\text{H}_2 \to \text{CH}_4 + 2\text{H}_2\text{O} \quad (\Delta H_{298\text{K}} = -165\text{ kJ/mol})$$

* **Stoichiometric Reactant Mass Balance**:
  $$\dot{m}_{\text{CH}_4} = \dot{m}_{\text{H}_2} \cdot \frac{M_{\text{CH}_4}}{4 M_{\text{H}_2}} \cdot \eta_{\text{conversion}}$$
""",
        "params": [
            ("name", "str", "'Methanation'", "-", "Unique component name identifier"),
            ("capacity_mw_sng", "float", "5.0", "MW", "Rated synthetic methane energy production capacity"),
            ("conversion_efficiency", "float", "0.82", "0-1", "Chemical conversion yield factor"),
        ],
        "inputs": [
            ("m_dot_H2_in_kg_hr", "kg/hr", "Green hydrogen feed mass flow rate"),
            ("m_dot_CO2_in_kg_hr", "kg/hr", "Captured carbon dioxide feed flow rate"),
        ],
        "outputs": [
            ("m_dot_CH4_kg_hr", "kg/hr", "Synthetic natural gas production mass flow rate"),
            ("Q_exothermic_MWth", "MWth", "High-temperature reaction waste heat available for steam generation"),
        ],
        "example": """from energysim.models.hydrogen import MethanationReactor

p2g = MethanationReactor(name="methanation", capacity_mw_sng=5.0)"""
    },

    "ccgt": {
        "title": "Combined Cycle Gas Turbine (CCGT)",
        "class": "CombinedCycle_CCGT",
        "module": "energysim.models.thermal",
        "desc": "Combined Cycle Gas Turbine (CCGT) combining a Brayton gas turbine and a Rankine heat recovery steam turbine (HRSG). Features non-linear part-load heat rate curves, ambient temperature capacity derating, Cold/Warm/Hot start sequence state machines, and continuous $\\text{CO}_2$ emissions accounting.",
        "equations": r"""### Governing Equations

* **Power Split**:
  $$P_{\text{actual}} = P_{\text{GT}} + P_{\text{ST}} \quad (P_{\text{GT}} \approx 0.67 P_{\text{actual}}, \quad P_{\text{ST}} \approx 0.33 P_{\text{actual}})$$

* **Ambient Temperature Capacity Derating**:
  $$P_{\text{max,avail}} = P_{\text{rated}} \cdot f_{\text{amb}}(T_{\text{amb}})$$

* **Part-Load Heat Rate & Fuel Consumption**:
  $$\text{HR}(\text{Load}) = \text{HR}_{\text{full}} \cdot \left[1.0 + 0.32 \cdot (1.0 - \text{Load})^{1.8}\right]$$
  $$\text{Fuel}_{\text{MMBtu/hr}} = \frac{P_{\text{actual}} \cdot 1000 \cdot \text{HR}(\text{Load})}{1.055 \times 10^6}$$

* **Carbon Emissions Accounting**:
  $$\text{CO}_{2,\text{rate}} = \text{Fuel}_{\text{MMBtu/hr}} \cdot 0.0592 \quad [\text{tCO}_2/\text{hr}]$$
""",
        "params": [
            ("name", "str", "'CCGT'", "-", "Unique component name identifier"),
            ("p_rated_mw", "float", "400.0", "MW", "Total combined cycle plant rated capacity (GT + ST)"),
            ("gt_ratio", "float", "0.67", "0-1", "Fraction of capacity delivered by gas turbine (typically 2/3)"),
            ("heat_rate_full_load_kj_kwh", "float", "6400.0", "kJ/kWh", "Full load LHV heat rate (~56.2% thermal efficiency)"),
            ("min_stable_gen_pct", "float", "40.0", "%", "Minimum stable generation (MSG) environmental cutoff"),
            ("ramp_rate_mw_min", "float", "15.0", "MW/min", "Maximum mechanical ramp rate constraint"),
        ],
        "inputs": [
            ("P_cmd", "MW", "Active power generation dispatch order"),
            ("T_amb", "°C", "Ambient compressor inlet air temperature"),
        ],
        "outputs": [
            ("P_actual", "MW", "Total combined active power delivered to the transmission bus"),
            ("P_gt", "MW", "Gas turbine electrical power output"),
            ("P_st", "MW", "Steam turbine electrical power output"),
            ("fuel_consumption_MMBtu_hr", "MMBtu/hr", "Natural gas fuel firing rate"),
            ("CO2_emissions_t_hr", "t/hr", "Instantaneous carbon dioxide emission rate"),
            ("Q_condenser_MW", "MWth", "Cooling tower / condenser heat rejection load"),
        ],
        "example": """from energysim import world
from energysim.models.thermal import CombinedCycle_CCGT

ccgt = CombinedCycle_CCGT(name="ccgt", p_rated_mw=40.0, min_stable_gen_pct=35.0)
sim = world(start_time=0, stop_time=86400, t_macro=900)
sim.add_model("ccgt", model=ccgt, outputs=["P_actual", "fuel_consumption_MMBtu_hr", "CO2_emissions_t_hr"])
sim.add_signal("dispatch", lambda t: 30.0)
sim.add_connections({"dispatch.y": "ccgt.P_cmd"})
sim.simulate()"""
    },

    "ocgt": {
        "title": "Open Cycle Gas Turbine (OCGT)",
        "class": "GasTurbine_OCGT",
        "module": "energysim.models.thermal",
        "desc": "Simple-cycle aeroderivative or heavy-duty peaking gas turbine. Features fast start-up operating state machines (Cold/Warm/Hot timers), ramp rate constraints, high-temperature exhaust heat output, and part-load efficiency derate.",
        "equations": r"""### Governing Equations

* **Operating State Machine**:
  $$\text{State} \in \{\text{OFF}, \text{STARTING}, \text{RUNNING}, \text{SHUTTING\_DOWN}\}$$
  Startup duration dynamically switches between Hot ($<2\text{ hr}$ downtime), Warm ($2-8\text{ hr}$), and Cold ($>8\text{ hr}$).

* **Startup Fuel Loss**:
  $$\text{Fuel}_{\text{start}} = P_{\text{rated}} \cdot 1000 \cdot \text{HR}_{\text{full}} \cdot 0.15 \quad [\text{kJ during roll/purge}]$$
""",
        "params": [
            ("name", "str", "'OCGT'", "-", "Unique component name identifier"),
            ("p_rated_mw", "float", "50.0", "MW", "Rated generator active power output"),
            ("heat_rate_full_load_kj_kwh", "float", "9500.0", "kJ/kWh", "Full load heat rate (~37.9% efficiency)"),
            ("min_stable_gen_pct", "float", "20.0", "%", "Minimum stable generation limit"),
            ("ramp_rate_mw_min", "float", "25.0", "MW/min", "Aeroderivative fast ramp rate limit"),
        ],
        "inputs": [
            ("P_cmd", "MW", "Dispatch setpoint"),
            ("T_amb", "°C", "Compressor inlet air temperature"),
        ],
        "outputs": [
            ("P_actual", "MW", "Delivered active power"),
            ("fuel_consumption_MMBtu_hr", "MMBtu/hr", "Fuel gas consumption rate"),
            ("Q_exhaust_MW", "MWth", "Thermal energy in exhaust flue gas"),
            ("state", "str", "Operating state string"),
        ],
        "example": """from energysim.models.thermal import GasTurbine_OCGT

ocgt = GasTurbine_OCGT(name="peaker", p_rated_mw=50.0, ramp_rate_mw_min=25.0)"""
    },

    "genset": {
        "title": "Reciprocating Engine Genset",
        "class": "ReciprocatingGenset",
        "module": "energysim.models.thermal",
        "desc": "Medium-speed or high-speed reciprocating internal combustion engine (diesel, lean-burn natural gas, or dual-fuel). Features ultra-fast synchronization (<2 minutes), high part-load efficiency, and jacket water / exhaust heat recovery.",
        "equations": r"""### Governing Equations

* **Fuel Mass Flow Rate**:
  $$\dot{m}_{\text{fuel}} = \frac{P_{\text{actual}}}{\eta_{\text{brake}}(P) \cdot \text{LHV}_{\text{fuel}}} \quad [\text{kg/hr}]$$
""",
        "params": [
            ("name", "str", "'Genset'", "-", "Unique component name identifier"),
            ("p_rated_mw", "float", "5.0", "MW", "Rated electrical power output"),
            ("efficiency_full_load", "float", "0.44", "0-1", "Brake electrical efficiency at rated load (~44%)"),
            ("min_load_pct", "float", "15.0", "%", "Minimum continuous load limit"),
        ],
        "inputs": [
            ("P_cmd", "MW", "Power dispatch request"),
        ],
        "outputs": [
            ("P_actual", "MW", "Delivered electrical power"),
            ("fuel_l_hr", "L/hr", "Fuel volumetric consumption rate"),
            ("Q_jacket_water_kW", "kWth", "Jacket water thermal heat recovery"),
            ("Q_exhaust_kW", "kWth", "Exhaust gas thermal heat recovery"),
        ],
        "example": """from energysim.models.thermal import ReciprocatingGenset

genset = ReciprocatingGenset(name="genset", p_rated_mw=5.0)"""
    },

    "chp": {
        "title": "Combined Heat and Power (CHP) Steam Turbine",
        "class": "CHP_SteamTurbine",
        "module": "energysim.models.thermal",
        "desc": "Backpressure or extraction steam turbine delivering simultaneous active electrical power and low-pressure district heating or process steam.",
        "equations": r"""### Governing Equations

* **Power-to-Heat Ratio**:
  $$\sigma = \frac{P_{\text{elec}}}{Q_{\text{heat}}}$$

* **Total Fuel Utilization**:
  $$\eta_{\text{total}} = \frac{P_{\text{elec}} + Q_{\text{heat}}}{\dot{m}_{\text{fuel}} \cdot \text{LHV}} \ge 85\%$$
""",
        "params": [
            ("name", "str", "'CHP'", "-", "Unique component name identifier"),
            ("p_rated_mw", "float", "10.0", "MW", "Rated electrical generator capacity"),
            ("power_to_heat_ratio", "float", "0.8", "-", "Design electrical-to-thermal power ratio $\\sigma$"),
            ("thermal_efficiency_pct", "float", "88.0", "%", "Total plant fuel utilization efficiency"),
        ],
        "inputs": [
            ("P_cmd", "MW", "Electrical dispatch order"),
            ("heat_demand_mw", "MWth", "Process heat dispatch order"),
        ],
        "outputs": [
            ("P_actual", "MW", "Actual electricity generated"),
            ("Q_heat_actual_MW", "MWth", "Actual thermal heat delivered"),
            ("fuel_input_MW", "MW", "Total primary fuel consumption rate"),
        ],
        "example": """from energysim.models.thermal import CHP_SteamTurbine

chp = CHP_SteamTurbine(name="chp_plant", p_rated_mw=10.0, power_to_heat_ratio=0.8)"""
    },

    "heat_pump": {
        "title": "Industrial Heat Pump",
        "class": "IndustrialHeatPump",
        "module": "energysim.models.hvac_thermal",
        "desc": "High-temperature industrial heat pump extracting low-grade waste heat (from data centers, electrolyzers, or industrial effluents) and lifting it to district heating or steam temperatures.",
        "equations": r"""### Governing Equations

* **Carnot Coefficient of Performance (COP)**:
  $$\text{COP}_{\text{Carnot}} = \frac{T_{\text{sink}} + 273.15}{(T_{\text{sink}} + 273.15) - (T_{\text{source}} + 273.15)}$$
  $$\text{COP}_{\text{actual}} = \eta_{\text{Carnot}} \cdot \text{COP}_{\text{Carnot}}$$

* **Thermal Heat Delivered**:
  $$Q_{\text{heat}} = P_{\text{elec}} \cdot \text{COP}_{\text{actual}}$$
  $$Q_{\text{source,absorbed}} = Q_{\text{heat}} - P_{\text{elec}}$$
""",
        "params": [
            ("name", "str", "'HeatPump'", "-", "Unique component name identifier"),
            ("p_electric_max_mw", "float", "2.0", "MW", "Maximum electrical compressor power rating"),
            ("t_source_nom_c", "float", "30.0", "°C", "Nominal low-grade waste heat source temperature"),
            ("t_sink_nom_c", "float", "85.0", "°C", "Nominal high-grade delivery temperature"),
            ("eta_carnot", "float", "0.55", "0-1", "Carnot efficiency fraction factor (typically 0.45-0.60)"),
        ],
        "inputs": [
            ("P_cmd", "MW", "Compressor electrical power setpoint"),
            ("T_source", "°C", "Dynamic evaporator heat source inlet temperature"),
            ("T_sink", "°C", "Dynamic condenser delivery temperature"),
        ],
        "outputs": [
            ("P_actual", "MW", "Electrical compressor power absorbed"),
            ("Q_heat_MW", "MWth", "Useful thermal heat injected into heat network"),
            ("Q_absorbed_MW", "MWth", "Waste heat extracted from evaporator source"),
            ("COP", "-", "Actual operating Coefficient of Performance"),
        ],
        "example": """from energysim import world
from energysim.models.hvac_thermal import IndustrialHeatPump

hp = IndustrialHeatPump(name="hp", p_electric_max_mw=2.0, t_source_nom_c=30.0, t_sink_nom_c=85.0)
sim = world(start_time=0, stop_time=3600, t_macro=900)
sim.add_model("hp", model=hp, outputs=["P_actual", "Q_heat_MW", "COP"])
sim.add_signal("power_cmd", lambda t: 1.5)
sim.add_connections({"power_cmd.y": "hp.P_cmd"})
sim.simulate()"""
    },

    "electric_boiler": {
        "title": "Industrial Electric Boiler",
        "class": "ElectricBoiler",
        "module": "energysim.models.hvac_thermal",
        "desc": "High-voltage electrode or immersion electric steam/hot water boiler. Converts surplus renewable electricity into process heat at near-unity efficiency (>99%).",
        "equations": r"""### Governing Equations

* **Heat Generation**:
  $$Q_{\text{thermal}} = P_{\text{elec}} \cdot \eta_{\text{boiler}}$$
""",
        "params": [
            ("name", "str", "'ElectricBoiler'", "-", "Unique component name identifier"),
            ("p_rated_mw", "float", "10.0", "MW", "Maximum electrical power consumption"),
            ("efficiency", "float", "0.99", "0-1", "Conversion thermal efficiency"),
        ],
        "inputs": [
            ("P_cmd", "MW", "Electrical consumption power request"),
        ],
        "outputs": [
            ("P_actual", "MW", "Actual active power absorbed"),
            ("Q_thermal_MW", "MWth", "Useful steam or hot water thermal power delivered"),
        ],
        "example": """from energysim.models.hvac_thermal import ElectricBoiler

boiler = ElectricBoiler(name="boiler", p_rated_mw=10.0)"""
    },

    "chiller": {
        "title": "Chillers (Electric & Absorption)",
        "class": "ElectricChiller, AbsorptionChiller",
        "module": "energysim.models.hvac_thermal",
        "desc": "District cooling plant models featuring vapor-compression Electric Chillers (driven by active electrical power) and thermally driven Absorption Chillers (utilizing waste heat from gas turbines or industrial exhausts).",
        "equations": r"""### Governing Equations

* **Electric Chiller**:
  $$Q_{\text{cool}} = P_{\text{elec}} \cdot \text{COP}_{\text{chiller}}$$

* **Absorption Chiller (Thermal COP)**:
  $$Q_{\text{cool}} = Q_{\text{waste\_heat\_in}} \cdot \text{COP}_{\text{thermal}} \quad (\text{COP}_{\text{thermal}} \approx 0.7 - 1.2)$$
""",
        "params": [
            ("name", "str", "'Chiller'", "-", "Unique component name identifier"),
            ("q_cool_max_mw", "float", "5.0", "MWth", "Maximum chilled water cooling capacity"),
            ("cop_nominal", "float", "5.5", "-", "Design Coefficient of Performance"),
        ],
        "inputs": [
            ("cooling_demand_mw", "MWth", "Chilled water thermal load request"),
        ],
        "outputs": [
            ("Q_cooling_delivered_MW", "MWth", "Delivered chilled water cooling power"),
            ("P_electric_consumed_MW", "MW", "Compressor electricity consumed (Electric Chiller)"),
            ("Q_heat_consumed_MW", "MWth", "Waste heat consumed (Absorption Chiller)"),
        ],
        "example": """from energysim.models.hvac_thermal import ElectricChiller

chiller = ElectricChiller(name="chiller", q_cool_max_mw=5.0, cop_nominal=5.5)"""
    },

    "thermal_storage": {
        "title": "Stratified Thermal Storage Tank",
        "class": "ThermalStorageTank",
        "module": "energysim.models.hvac_thermal",
        "desc": "Sensible heat water storage tank featuring a 1-D vertical temperature thermocline separating hot and cold layers for district heating or cooling peak shaving.",
        "equations": r"""### Governing Equations

* **Sensible Heat Energy Balance**:
  $$E_{\text{stored}} = \rho \cdot V \cdot c_p \cdot (T_{\text{hot}} - T_{\text{cold}})$$
  $$\frac{dE}{dt} = Q_{\text{in}} - Q_{\text{out}} - U A (T_{\text{tank}} - T_{\text{amb}})$$
""",
        "params": [
            ("name", "str", "'ThermalTank'", "-", "Unique component name identifier"),
            ("volume_m3", "float", "2000.0", "m³", "Water vessel volume"),
            ("t_hot_c", "float", "95.0", "°C", "Hot layer temperature"),
            ("t_cold_c", "float", "45.0", "°C", "Cold return layer temperature"),
            ("loss_coeff_w_m2_k", "float", "0.5", "W/(m²·K)", "Insulation heat loss coefficient"),
        ],
        "inputs": [
            ("Q_in_MWth", "MWth", "Heat injection rate (charging)"),
            ("Q_out_MWth", "MWth", "Heat extraction rate (discharging)"),
        ],
        "outputs": [
            ("SoC", "0-1", "Thermal state of charge"),
            ("energy_stored_mwh", "MWh", "Total stored thermal energy"),
        ],
        "example": """from energysim.models.hvac_thermal import ThermalStorageTank

tank = ThermalStorageTank(name="tes", volume_m3=2000.0)"""
    },

    "carbon_capture": {
        "title": "Amine Post-Combustion Carbon Capture",
        "class": "CarbonCapture_Amine",
        "module": "energysim.models.carbon",
        "desc": "Post-combustion chemical absorption carbon capture system using monoethanolamine (MEA) or advanced amines. Models flue gas $\\text{CO}_2$ capture efficiency and thermal reboiler steam stripping penalty.",
        "equations": r"""### Governing Equations

* **Captured Mass Rate**:
  $$\dot{m}_{\text{CO}_2\text{,captured}} = \dot{m}_{\text{CO}_2\text{,in}} \cdot \eta_{\text{capture}} \quad (\eta_{\text{capture}} \approx 90\%)$$

* **Thermal Reboiler Duty**:
  $$Q_{\text{reboiler}} = \dot{m}_{\text{CO}_2\text{,captured}} \cdot q_{\text{specific}} \quad (q_{\text{specific}} \approx 2.5 - 3.5\text{ GJ/tCO}_2)$$

* **Electrical Auxiliary Power**:
  $$P_{\text{aux}} = \dot{m}_{\text{CO}_2\text{,captured}} \cdot w_{\text{specific}} \quad (w_{\text{specific}} \approx 100 - 150\text{ kWh/tCO}_2)$$
""",
        "params": [
            ("name", "str", "'CarbonCapture'", "-", "Unique component name identifier"),
            ("capture_efficiency_pct", "float", "90.0", "%", "Flue gas $\\text{CO}_2$ capture recovery rate"),
            ("specific_reboiler_duty_gj_t", "float", "2.8", "GJ/tCO₂", "Thermal steam heat required per ton $\\text{CO}_2$ stripped"),
            ("specific_power_kwh_t", "float", "110.0", "kWh/tCO₂", "Electrical pump and compressor power per ton $\\text{CO}_2$"),
        ],
        "inputs": [
            ("CO2_flue_gas_t_hr", "t/hr", "Carbon dioxide mass flow entering from flue stack"),
        ],
        "outputs": [
            ("CO2_captured_t_hr", "t/hr", "Mass flow rate of captured high-purity $\\text{CO}_2$"),
            ("steam_demand_MWth", "MWth", "Thermal reboiler heat demand extracted from steam network"),
            ("aux_power_MW", "MW", "Parasitic electricity consumed by pumps and blowers"),
        ],
        "example": """from energysim.models.carbon import CarbonCapture_Amine

ccs = CarbonCapture_Amine(name="ccs", capture_efficiency_pct=90.0)"""
    },

    "datacenter": {
        "title": "Data Center IT & Cooling Load",
        "class": "DataCenterLoad",
        "module": "energysim.models.loads",
        "desc": "Hyperscale data center load modeling baseline IT compute server electrical demand combined with temperature-dependent Power Usage Effectiveness (PUE) cooling curves and waste heat rejection.",
        "equations": r"""### Governing Equations

* **Total Facility Power Demand**:
  $$P_{\text{facility}} = P_{\text{IT}} \cdot \text{PUE}(T_{\text{amb}})$$

* **PUE Temperature Characteristic**:
  $$\text{PUE}(T_{\text{amb}}) = \text{PUE}_{\text{base}} + k_{\text{cool}} \cdot \max(0, T_{\text{amb}} - 15.0)$$

* **Low-Grade Waste Heat for Heat Pumps**:
  $$Q_{\text{waste}} = P_{\text{facility}} \cdot \eta_{\text{thermal\_dissipation}} \quad (\sim 95\% \text{ of electrical power rejected at 30-35°C})$$
""",
        "params": [
            ("name", "str", "'DataCenter'", "-", "Unique component name identifier"),
            ("p_it_nominal_mw", "float", "20.0", "MW", "Baseline continuous IT compute electrical server power"),
            ("pue_base", "float", "1.15", "-", "Base design PUE during free economizer cooling"),
            ("pue_temp_coeff", "float", "0.01", "1/°C", "PUE increase per °C above free-cooling threshold"),
        ],
        "inputs": [
            ("it_load_fraction", "0-1", "Compute server utilization fraction"),
            ("T_amb", "°C", "Ambient wet/dry-bulb temperature"),
        ],
        "outputs": [
            ("P_facility_total_MW", "MW", "Total facility grid electrical demand (IT + cooling + UPS)"),
            ("P_cooling_MW", "MW", "Active power consumed solely by chillers and HVAC"),
            ("Q_waste_heat_MWth", "MWth", "Low-grade waste heat available for district heating"),
            ("PUE", "-", "Operating Power Usage Effectiveness"),
        ],
        "example": """from energysim.models.loads import DataCenterLoad

dc = DataCenterLoad(name="dc", p_it_nominal_mw=20.0, pue_base=1.15)"""
    },

    "process_steam": {
        "title": "Industrial Process Steam Header",
        "class": "IndustrialProcessSteam",
        "module": "energysim.models.loads",
        "desc": "Industrial refinery or chemical plant steam header mass and enthalpy balance. Coordinates steam generation from CCGT HRSG boilers, electric boilers, and solar thermal collectors against industrial consumer demand.",
        "equations": r"""### Governing Equations

* **Steam Mass Balance**:
  $$\dot{m}_{\text{balance}} = \sum \dot{m}_{\text{supply}} - \sum \dot{m}_{\text{demand}}$$
  $$\dot{m}_{\text{deficit}} = \max(0, -\dot{m}_{\text{balance}})$$
""",
        "params": [
            ("name", "str", "'SteamHeader'", "-", "Unique component name identifier"),
            ("nominal_demand_t_hr", "float", "50.0", "t/hr", "Nominal process steam demand"),
            ("pressure_bar", "float", "40.0", "bar", "Header steam pressure"),
        ],
        "inputs": [
            ("steam_supplied_t_hr", "t/hr", "Total steam supplied from HRSG and electric boilers"),
        ],
        "outputs": [
            ("steam_demand_t_hr", "t/hr", "Process steam demand"),
            ("steam_deficit_t_hr", "t/hr", "Unmet steam demand"),
        ],
        "example": """from energysim.models.loads import IndustrialProcessSteam

steam = IndustrialProcessSteam(name="steam", nominal_demand_t_hr=50.0)"""
    },

    "catalog": {
        "title": "Equipment Catalog & Archetypes",
        "class": "energysim.models.catalog",
        "module": "energysim.models.catalog",
        "desc": "Pre-calibrated equipment archetypes configured against manufacturer datasheets and test-bench measurements.",
        "equations": r"""### Equipment Archetypes List

* **`catalog.Tesla_Megapack_2XL`**: Utility-scale LFP BESS ($3.9\text{ MWh}$, $1.9\text{ MW}$, $92\%$ round-trip efficiency, liquid cooled).
* **`catalog.Nel_A1000`**: Commercial atmospheric alkaline water electrolyzer ($2.2\text{ MW}$, $388\text{ Nm}^3/\text{hr}$, $20\%$ turndown cutoff).
* **`catalog.ITM_Power_Trident`**: Multi-megawatt PEM electrolyzer stack ($5.0\text{ MW}$, $30\text{ bar}$, fast ramping).
* **`catalog.Bloom_Energy_SOEC`**: Solid Oxide Electrolyzer ($1.0\text{ MW}$, steam integrated, $37.5\text{ kWh/kg}$).
* **`catalog.GE_7HA_02`**: Heavy-duty Combined Cycle Gas Turbine ($570\text{ MW}$, $64.0\%$ net efficiency, fast startup).
* **`catalog.Siemens_SGT_800`**: Industrial gas turbine ($54\text{ MW}$, simple cycle or CHP).
* **`catalog.CanadianSolar_HiKu7`**: Monocrystalline PERC PV array archetype ($665\text{ W}$ module rating, $-0.34\%/^\circ\text{C}$).
* **`catalog.Vestas_V150_4MW`**: Low-to-medium wind regime turbine ($4.2\text{ MW}$, $150\text{ m}$ rotor diameter).
""",
        "params": [
            ("name", "str", "'catalog_item'", "-", "Instance name identifier"),
            ("custom_kwargs", "Any", "Overridable", "-", "Any manufacturer parameter can be overridden during instantiation"),
        ],
        "inputs": [("See underlying base model inputs", "-", "-")],
        "outputs": [("See underlying base model outputs", "-", "-")],
        "example": """from energysim.models import catalog

bess = catalog.Tesla_Megapack_2XL(name="bess", soc_init=0.6)
pem = catalog.ITM_Power_Trident(name="pem")
turbine = catalog.Vestas_V150_4MW(name="vestas")"""
    }
}

# -----------------------------------------------------------------------------
# GENERATE INDIVIDUAL COMPONENT PAGES
# -----------------------------------------------------------------------------

def build_component_md(key, data):
    md = []
    md.append(f"# {data['title']}\n")
    md.append(f"> **Class**: `{data['class']}`  \n> **Module**: `{data['module']}`\n")
    md.append(f"## Overview\n\n{data['desc']}\n")
    md.append(f"---\n\n{data['equations']}\n")
    md.append("---\n\n## Parameters & Attributes\n")
    md.append("| Parameter | Type | Default | Unit | Description |")
    md.append("|---|---|---|---|---|")
    for p in data['params']:
        md.append(f"| `{p[0]}` | `{p[1]}` | `{p[2]}` | {p[3]} | {p[4]} |")
    md.append("\n---\n\n## Co-Simulation Inputs\n")
    md.append("| Input Variable | Unit | Description |")
    md.append("|---|---|---|")
    for inp in data['inputs']:
        md.append(f"| `{inp[0]}` | {inp[1]} | {inp[2]} |")
    md.append("\n---\n\n## Simulation Outputs\n")
    md.append("| Output Variable | Unit | Description |")
    md.append("|---|---|---|")
    for out in data['outputs']:
        md.append(f"| `{out[0]}` | {out[1]} | {out[2]} |")
    md.append("\n---\n\n## Python Usage Example\n")
    md.append("```python\n" + data['example'] + "\n```\n")

    return "\n".join(md)

for k, d in COMPONENTS.items():
    file_path = os.path.join(components_dir, f"{k}.md")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(build_component_md(k, d))
    print(f"Generated: {file_path}")

print("All component pages generated successfully!")

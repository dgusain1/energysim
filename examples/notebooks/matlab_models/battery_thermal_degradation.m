function [SoC, T_cell_degC, P_actual_kW, degradation_pct] = battery_thermal_degradation(P_cmd_kW, T_ambient_degC, time)
    % BATTERY_THERMAL_DEGRADATION
    % Illustrative GNU Octave / MATLAB function model for energysim co-simulation.
    %
    % Function Contract:
    %   Inputs must match the adapter 'inputs' list in exact order:
    %     [P_cmd_kW, T_ambient_degC] + trailing argument 'time' (seconds)
    %   Outputs must match the adapter 'outputs' list in exact order:
    %     [SoC, T_cell_degC, P_actual_kW, degradation_pct]

    % Persistent internal state variables
    persistent internal_soc;
    persistent internal_temp;
    persistent internal_deg;
    persistent last_time;

    % Initialise on first call
    if isempty(internal_soc)
        internal_soc = 0.50;           % 50% initial state of charge
        internal_temp = T_ambient_degC; % initial cell temperature
        internal_deg = 0.00;           % 0% degradation
        last_time = time;
    end

    % Time step calculation (seconds)
    dt = max(time - last_time, 1.0);
    last_time = time;

    % Battery pack specifications
    nominal_capacity_kwh = 100.0;     % 100 kWh pack
    max_power_kw = 50.0;              % 50 kW rating
    efficiency = 0.95;                % 95% one-way efficiency
    thermal_resistance = 0.08;        % degC / kW thermal resistance
    thermal_capacitance = 450.0;      % kJ / degC thermal mass

    % Enforce power limits
    P_actual_kW = min(max(P_cmd_kW, -max_power_kw), max_power_kw);

    % Prevent over-charge and over-discharge
    if (P_actual_kW > 0 && internal_soc >= 0.95)
        P_actual_kW = 0.0;
    elseif (P_actual_kW < 0 && internal_soc <= 0.05)
        P_actual_kW = 0.0;
    end

    % Energy throughput and SoC integration
    energy_delta_kwh = (P_actual_kW * efficiency) * (dt / 3600.0);
    internal_soc = min(max(internal_soc + energy_delta_kwh / nominal_capacity_kwh, 0.05), 0.95);

    % Thermal dynamics (losses generate heat: Q_loss = P * (1 - eta))
    loss_kw = abs(P_actual_kW) * (1.0 - efficiency);
    heat_dissipation_kw = (internal_temp - T_ambient_degC) / thermal_resistance;
    dT_dt = (loss_kw - heat_dissipation_kw) * 1000.0 / (thermal_capacitance * 1000.0); % degC / s
    internal_temp = internal_temp + dT_dt * dt;

    % Semi-empirical Arrhenius capacity degradation rate
    temp_factor = exp(0.04 * (internal_temp - 25.0));
    deg_rate_per_hour = 1e-5 * (1.0 + abs(P_actual_kW)/max_power_kw) * temp_factor;
    internal_deg = internal_deg + deg_rate_per_hour * (dt / 3600.0) * 100.0;

    % Assign outputs
    SoC = internal_soc;
    T_cell_degC = internal_temp;
    degradation_pct = internal_deg;
end

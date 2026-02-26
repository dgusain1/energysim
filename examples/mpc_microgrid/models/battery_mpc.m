function [SoC, P_actual, E_available] = battery_mpc(P_cmd, time)
% BATTERY_MPC  Li-ion battery model for MPC co-simulation.
%
%   [SoC, P_actual, E_available] = battery_mpc(P_cmd, time)
%
%   Inputs
%       P_cmd   - Power command [MW].  Positive = charging,
%                 negative = discharging.
%       time    - Simulation time [s]  (appended by energysim)
%
%   Outputs
%       SoC         - State of charge  [0 .. 1]
%       P_actual    - Realised power   [MW]
%       E_available - Dischargeable energy remaining [MWh]
%
%   Parameters (edit here)
%       E_cap   = 0.0135 MWh  (13.5 kWh, Tesla Powerwall)
%       P_max   = 0.005  MW   (5 kW)
%       eta     = 0.95        round-trip efficiency
%       SoC_min = 0.10
%       SoC_max = 0.95

    % --- Parameters ---
    E_cap   = 0.0135;    % [MWh]
    P_max   = 0.005;     % [MW]
    eta     = 0.95;      % [-]
    SoC_min = 0.10;
    SoC_max = 0.95;

    % --- Persistent state ---
    persistent soc t_prev
    if isempty(soc)
        soc    = 0.50;
        t_prev = time;
    end

    % --- Time step ---
    dt_h = (time - t_prev) / 3600.0;   % hours
    t_prev = time;
    if dt_h <= 0
        SoC         = soc;
        P_actual    = 0.0;
        E_available = (soc - SoC_min) * E_cap;
        return;
    end

    % --- Clamp power ---
    P = max(-P_max, min(P_max, P_cmd));

    % --- Energy balance ---
    sqrt_eta = sqrt(eta);
    if P >= 0
        dE = P * sqrt_eta * dt_h;      % energy stored
    else
        dE = P / sqrt_eta * dt_h;      % energy removed (negative)
    end

    soc_new = soc + dE / E_cap;

    % --- SoC limits ---
    if soc_new > SoC_max
        soc_new = SoC_max;
        if P > 0
            P = (SoC_max - soc) * E_cap / (sqrt_eta * dt_h);
        end
    elseif soc_new < SoC_min
        soc_new = SoC_min;
        if P < 0
            P = (SoC_min - soc) * E_cap * sqrt_eta / dt_h;
        end
    end

    soc = soc_new;

    SoC         = soc;
    P_actual    = P;
    E_available = max(0.0, (soc - SoC_min) * E_cap);
end

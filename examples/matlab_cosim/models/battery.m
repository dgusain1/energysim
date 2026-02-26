function [SoC, P_actual] = battery(P_cmd, time)
% BATTERY  Simple Li-ion battery model with persistent state.
%
%   [SoC, P_actual] = battery(P_cmd, time)
%
%   Inputs
%       P_cmd   - Power command [kW].  Positive = discharge, negative = charge.
%       time    - Current simulation time [s]  (appended by energysim)
%
%   Outputs
%       SoC      - State of charge [0 .. 1]
%       P_actual - Actual power delivered/absorbed [kW]
%                  (may differ from P_cmd when SoC limits are hit)
%
%   The model uses persistent variables to keep track of the SoC and
%   the previous time-step.  energysim calls the function once per
%   macro time-step; the function derives dt from consecutive calls.
%
%   Parameters (edit inside this file):
%       E_cap    = 13.5 kWh   (Tesla Powerwall-sized)
%       eta_ch   = 0.95       charge efficiency
%       eta_dis  = 0.95       discharge efficiency
%       P_max    = 5.0 kW     max charge/discharge power
%       SoC_min  = 0.10
%       SoC_max  = 0.90

    % --- Parameters ---
    E_cap   = 13.5;     % usable capacity      [kWh]
    eta_ch  = 0.95;     % charge efficiency     [-]
    eta_dis = 0.95;     % discharge efficiency  [-]
    P_max   = 5.0;      % max power             [kW]
    SoC_min = 0.10;     % minimum SoC           [-]
    SoC_max = 0.90;     % maximum SoC           [-]

    % --- Persistent state ---
    persistent soc t_prev
    if isempty(soc)
        soc    = 0.50;   % initial SoC
        t_prev = time;
    end

    % --- Time step ---
    dt = (time - t_prev) / 3600.0;   % seconds -> hours
    t_prev = time;
    if dt <= 0
        % First call or zero-length step
        SoC      = soc;
        P_actual = 0.0;
        return;
    end

    % --- Clamp command to rated power ---
    P_actual = max(-P_max, min(P_cmd, P_max));

    % --- Energy balance ---
    if P_actual >= 0
        % Discharging
        dE = P_actual * dt / eta_dis;   % energy removed from battery
    else
        % Charging
        dE = P_actual * dt * eta_ch;    % energy added (negative P -> negative dE if charging adds)
        dE = P_actual * dt * eta_ch;    % dE < 0  (energy stored)
    end

    soc_new = soc - dE / E_cap;

    % --- SoC limits ---
    if soc_new > SoC_max
        % Reduce charge to stay at SoC_max
        dE_avail = (soc - SoC_max) * E_cap;
        if P_actual < 0
            P_actual = -dE_avail / (dt * eta_ch);
        end
        soc_new = SoC_max;
    elseif soc_new < SoC_min
        % Reduce discharge to stay at SoC_min
        dE_avail = (soc - SoC_min) * E_cap;
        if P_actual > 0
            P_actual = dE_avail * eta_dis / dt;
        end
        soc_new = SoC_min;
    end

    soc = soc_new;
    SoC      = soc;
    P_actual = P_actual;
end

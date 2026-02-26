function [Q_thermal, COP] = heatpump(P_electric, T_source, T_sink, time)
% HEATPUMP  Carnot-limited heat-pump model for co-simulation.
%
%   [Q_thermal, COP] = heatpump(P_electric, T_source, T_sink, time)
%
%   Inputs
%       P_electric  - Electrical power command [kW]  (>= 0)
%       T_source    - Source (outdoor / ground) temperature [degC]
%       T_sink      - Sink (indoor / water) temperature  [degC]
%       time        - Current simulation time [s]  (appended by energysim)
%
%   Outputs
%       Q_thermal   - Thermal power delivered [kW]
%       COP         - Instantaneous coefficient of performance [-]
%
%   The COP is calculated from the reversed Carnot cycle with an
%   exergetic efficiency (eta_ex) of 0.45, which is typical for
%   residential air-source heat pumps.
%
%   Example
%       [Q, cop] = heatpump(3.0, 5, 35, 0);

    % --- Parameters ---
    eta_ex     = 0.45;      % exergetic efficiency  [-]
    P_standby  = 0.05;      % standby consumption   [kW]
    COP_min    = 1.0;       % minimum COP clamp     [-]
    COP_max    = 8.0;       % maximum COP clamp     [-]

    % --- Guard: no heating requested ---
    if P_electric <= P_standby
        Q_thermal = 0.0;
        COP       = 0.0;
        return;
    end

    % --- Carnot COP ---
    T_source_K = T_source + 273.15;
    T_sink_K   = T_sink   + 273.15;
    dT         = T_sink_K - T_source_K;

    if dT <= 0
        % Source is warmer than sink – free cooling, COP is very high
        COP_carnot = COP_max;
    else
        COP_carnot = T_sink_K / dT;
    end

    COP = eta_ex * COP_carnot;
    COP = max(COP_min, min(COP, COP_max));   % clamp

    % --- Thermal output ---
    Q_thermal = P_electric * COP;
end

function [T_inside] = thermal_mass(Q_heating, T_ambient, time)
% THERMAL_MASS  Single-zone lumped-capacitance building model.
%
%   [T_inside] = thermal_mass(Q_heating, T_ambient, time)
%
%   Inputs
%       Q_heating  - Thermal power injected by the heating system [kW]
%       T_ambient  - Outdoor ambient temperature [degC]
%       time       - Current simulation time [s]  (appended by energysim)
%
%   Outputs
%       T_inside   - Indoor zone temperature [degC]
%
%   The model uses a simple first-order ODE:
%
%       C * dT/dt = Q_heating - UA * (T_inside - T_ambient)
%
%   integrated with forward Euler at each macro time-step.
%
%   Parameters (edit inside this file):
%       C  = 2500 kJ/K   (thermal capacitance, ~100 m2 well-insulated house)
%       UA =   80 W/K    (overall heat-loss coefficient)

    % --- Parameters ---
    C  = 2500.0;    % thermal capacitance  [kJ/K]
    UA = 0.080;     % heat-loss coeff      [kW/K]  (= 80 W/K)

    % --- Persistent state ---
    persistent T t_prev
    if isempty(T)
        T      = 20.0;   % initial indoor temperature [degC]
        t_prev = time;
    end

    % --- Time step ---
    dt = time - t_prev;   % [s]
    t_prev = time;
    if dt <= 0
        T_inside = T;
        return;
    end

    % --- Forward Euler integration ---
    dTdt = (Q_heating - UA * (T - T_ambient)) / C;   % [K/s]
    T    = T + dTdt * dt;

    T_inside = T;
end

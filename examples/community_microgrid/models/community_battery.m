function [SoC, P_actual, E_available, V_terminal, T_cell, ...
         R_internal, Q_capacity] = community_battery(P_cmd, T_ambient, time)
% ========================================================================
%  COMMUNITY_BATTERY — Electrochemical Multi-Physics Battery Model
% ========================================================================
%
%  50 kWh NMC622 / graphite lithium-ion community battery pack.
%
%  PHYSICS MODELLED
%  ────────────────
%  1. Electrode thermodynamics
%        Open-circuit voltage from Redlich-Kister expansion of
%        the Gibbs mixing energy for NMC622 layered oxide (cathode)
%        and staged graphite (anode).
%
%  2. Electrochemical impedance — 2nd-order equivalent circuit
%        R₀  : ohmic (electrolyte ionic conductivity, current collectors,
%               contact resistance, SEI film)
%        R₁-C₁: charge-transfer kinetics — linearised Butler-Volmer
%               R_ct = R·T / (α·F·i₀·A_electrode)
%               i₀ depends on surface Li concentration and temperature.
%        R₂-C₂: solid-state diffusion — Fickian single-particle model
%               (dominant mode τ_diff = R_p² / 5·D_s)
%               [Bizeray et al., J. Power Sources 296, 2015]
%
%  3. Arrhenius temperature dependence of all kinetic parameters
%        k(T) = k_ref · exp(−E_a/R · (1/T − 1/T_ref))
%
%  4. Lumped thermal model
%        Joule heat (I²R) + reversible entropic heat (I·T·dU/dT)
%        with convective cooling.
%        [Bernardi et al., J. Electrochem. Soc. 132(1), 1985]
%
%  5. Degradation model
%        Calendar: SEI film growth (Arrhenius + SoC stress factor)
%        Cycle:    Ah-throughput capacity fade (√Ah model)
%        Both contribute to capacity fade + resistance growth.
%        [Schmalstieg et al., J. Electrochem. Soc. 165(16), 2018]
%
%  CELL CHEMISTRY
%  ──────────────
%    Cathode  : LiNi₀.₆Mn₀.₂Co₀.₂O₂  (NMC622)
%    Anode    : graphite
%    Form     : 21700 cylindrical cell,  5.0 Ah nominal
%    Voltage  : 2.80 V (cutoff) – 4.20 V (max charge)
%
%  PACK CONFIGURATION
%  ──────────────────
%    14s × 193p  =  2702 cells
%    V_nom = 51.8 V DC     E_nom ≈ 50.1 kWh
%
%  INPUTS
%    P_cmd     [MW]  Power command.  +charge / −discharge.
%    T_ambient [°C]  Ambient temperature.
%    time      [s]   Simulation time (appended by energysim).
%
%  OUTPUTS
%    SoC         [-]    State of charge, 0..1
%    P_actual    [MW]   Realised power at DC terminals
%    E_available [MWh]  Remaining dischargeable energy
%    V_terminal  [V]    Pack terminal voltage
%    T_cell      [°C]   Lumped cell temperature
%    R_internal  [Ω]    Total cell internal resistance (aged)
%    Q_capacity  [Ah]   Effective cell capacity after degradation
% ========================================================================

    % ================================================================
    %  1.  PACK CONFIGURATION
    % ================================================================
    N_s   = 14;            % cells in series
    N_p   = 193;           % parallel strings
    Q_nom = 5.0;           % [Ah]  nominal cell capacity (BOL)
    N_cells = N_s * N_p;   % 2702 cells
    %   V_nom = 14 × 3.7 = 51.8 V,  E = 51.8 × 965 Ah = 49.99 kWh

    % ================================================================
    %  2.  ELECTROCHEMICAL CELL PARAMETERS  (at T_ref = 25 °C)
    % ================================================================
    T_ref_K = 298.15;          % [K]

    % ─── 2a. Ohmic resistance R₀ ───
    %   electrolyte ionic conduction  + current-collector contact
    %   + SEI film ionic resistance
    R0_ref  = 0.012;           % [Ω]  at SoC 0.5, T_ref
    Ea_R0   = 4000;            % [K]  = E_a / R_gas

    % ─── 2b. Charge-transfer resistance R₁ (Butler-Volmer) ───
    %   From linearised BV at small η:
    %     R_ct = R·T / (α·F·i₀·A_elec)
    %   with  α = 0.5  (symmetric transfer coefficient)
    %         i₀ ∝ c_s^α · (c_max − c_s)^(1−α) · exp(−Ea/RT)
    R1_ref   = 0.008;          % [Ω]
    Ea_R1    = 6000;           % [K]
    tau1_ref = 12.0;           % [s]  C₁ = τ₁/R₁ ≈ 1500 F

    % ─── 2c. Diffusion impedance R₂ (Fickian SPM) ───
    %   Dominant mode of Li⁺ solid-state diffusion in spherical
    %   active-material particles (radius R_p ≈ 5 μm):
    %     τ_diff = R_p² / (5·D_s)
    %     R_diff ∝ R_p / (F·A·D_s·c_max)
    R2_ref   = 0.015;          % [Ω]
    Ea_R2    = 3500;           % [K]
    tau2_ref = 280.0;          % [s]

    % ─── 2d. Voltage limits ───
    V_max_cell = 4.20;         % [V]  avoid Li plating at anode
    V_min_cell = 2.80;         % [V]  avoid Cu dissolution

    % ─── 2e. Current limits ───
    I_max_cell =  10.0;        % [A]  2 C charge   (plating / heat)
    I_min_cell = -15.0;        % [A]  3 C discharge

    % ================================================================
    %  3.  THERMAL PARAMETERS  (lumped model)
    % ================================================================
    m_cell  = 0.070;           % [kg]    21700 cell mass
    Cp_cell = 1000;            % [J/(kg·K)]
    m_pack  = N_cells * m_cell * 1.35;   % +35 % packaging / BMS mass
    Cp_pack = Cp_cell;         %  dominated by cell material
    h_conv  = 8.0;             % [W/(m²·K)]  forced-air cooling
    A_surf  = 3.0;             % [m²]   pack external surface area

    % ================================================================
    %  4.  AGING PARAMETERS
    % ================================================================
    %   Calendar:  SEI film growth  (Arrhenius + SoC stress)
    k_cal_ref = 5.0e-10;       % [1/s]  capacity fade rate at T_ref
    Ea_cal    = 5000;          % [K]
    %   Cycle:     Ah-throughput model  q_cyc = k · √(Ah/Q_nom)
    k_cyc     = 2.0e-4;

    % ─── SoC operating window ───
    SoC_min = 0.02;
    SoC_max = 0.98;

    % ================================================================
    %  5.  PERSISTENT STATE VARIABLES
    % ================================================================
    persistent soc T_K Vrc1 Vrc2 q_cal q_cyc Ah_thru t_prev is_init
    if isempty(is_init)
        soc     = 0.50;
        T_K     = T_ambient + 273.15;
        Vrc1    = 0.0;
        Vrc2    = 0.0;
        q_cal   = 0.0;
        q_cyc   = 0.0;
        Ah_thru = 0.0;
        t_prev  = time;
        is_init = true;
    end

    % ================================================================
    %  6.  TIME STEP
    % ================================================================
    dt = time - t_prev;
    t_prev = time;

    if dt <= 0
        U_oc0       = ocv_nmc(soc);
        SoC         = soc;
        P_actual    = 0.0;
        E_available = max(0, (soc - SoC_min) * Q_nom * N_p ...
                          * U_oc0 * N_s / 1.0e6);
        V_terminal  = N_s * U_oc0;
        T_cell      = T_K - 273.15;
        R_internal  = R0_ref;
        Q_capacity  = Q_nom * max(0.5, 1 - q_cal - q_cyc);
        return;
    end
    dt_h = dt / 3600.0;

    % ================================================================
    %  7.  EFFECTIVE CAPACITY  (after degradation)
    % ================================================================
    Q_eff = Q_nom * max(0.5, 1.0 - q_cal - q_cyc);   % floor 50 % EOL

    % ================================================================
    %  8.  TEMPERATURE-DEPENDENT KINETICS  (Arrhenius)
    % ================================================================
    inv_dT = 1.0 / T_K - 1.0 / T_ref_K;

    R0 = R0_ref * exp(Ea_R0 * inv_dT) * soc_factor_R0(soc);
    R1 = R1_ref * exp(Ea_R1 * inv_dT) * bv_factor(soc, T_K, T_ref_K);
    R2 = R2_ref * exp(Ea_R2 * inv_dT);

    tau1 = tau1_ref * exp(Ea_R1 * inv_dT);
    tau2 = tau2_ref * exp(Ea_R2 * inv_dT);

    % Resistance growth from aging  (SEI thickening → R growth)
    R_aging = 1.0 + 0.5 * (q_cal + q_cyc);
    R0 = R0 * R_aging;
    R1 = R1 * R_aging;
    R2 = R2 * R_aging;

    % ================================================================
    %  9.  OPEN-CIRCUIT VOLTAGE  (electrode thermodynamics)
    % ================================================================
    U_oc = ocv_nmc(soc);

    % ================================================================
    %  10. SOLVE FOR CELL CURRENT  (quadratic from 2-RC model)
    % ================================================================
    %
    %   V_cell(t) = U_oc + I·R₀
    %             + Vrc1_old · e^(−dt/τ₁) + R₁·I·(1 − e^(−dt/τ₁))
    %             + Vrc2_old · e^(−dt/τ₂) + R₂·I·(1 − e^(−dt/τ₂))
    %
    %   P = N_s · V_cell · N_p · I  →  quadratic in I:
    %       a·I² + b·I − P = 0
    %
    %   where  U_eff = U_oc + Vrc1·e₁ + Vrc2·e₂
    %          R_eff = R₀ + R₁·(1−e₁) + R₂·(1−e₂)

    e1 = exp(-dt / max(tau1, 0.01));
    e2 = exp(-dt / max(tau2, 0.01));

    U_eff = U_oc + Vrc1 * e1 + Vrc2 * e2;
    R_eff = R0 + R1 * (1.0 - e1) + R2 * (1.0 - e2);

    P_W  = P_cmd * 1.0e6;                       % command in watts
    a_q  = N_s * N_p * R_eff;
    b_q  = N_s * N_p * U_eff;
    disc = b_q * b_q + 4.0 * a_q * P_W;

    if disc < 0
        %  Power exceeds electrochemical capability — clamp to vertex
        I_cell = -b_q / (2.0 * a_q);
    else
        I_cell = (-b_q + sqrt(disc)) / (2.0 * a_q);
    end

    % ================================================================
    %  11. APPLY LIMITS
    % ================================================================
    % ─── Cell current limits ───
    I_cell = max(I_min_cell, min(I_max_cell, I_cell));

    % ─── Voltage limits (via effective resistance) ───
    V_test = U_eff + I_cell * R_eff;
    if V_test > V_max_cell && I_cell > 0
        I_cell = (V_max_cell - U_eff) / R_eff;
    elseif V_test < V_min_cell && I_cell < 0
        I_cell = (V_min_cell - U_eff) / R_eff;
    end
    I_cell = max(I_min_cell, min(I_max_cell, I_cell));

    % ─── SoC limits (Coulombic headroom) ───
    dQ_hi = (SoC_max - soc) * Q_eff;              % Ah charge headroom
    dQ_lo = (SoC_min - soc) * Q_eff;              % Ah discharge headroom (<0)
    I_soc_hi = dQ_hi / (dt_h + 1e-12);
    I_soc_lo = dQ_lo / (dt_h + 1e-12);
    I_cell = max(I_soc_lo, min(I_soc_hi, I_cell));
    I_cell = max(I_min_cell, min(I_max_cell, I_cell));

    % ================================================================
    %  12. UPDATE RC DYNAMICS  (exponential discretisation)
    % ================================================================
    Vrc1 = Vrc1 * e1 + R1 * I_cell * (1.0 - e1);
    Vrc2 = Vrc2 * e2 + R2 * I_cell * (1.0 - e2);

    % ================================================================
    %  13. TERMINAL VOLTAGE
    % ================================================================
    V_cell = U_oc + I_cell * R0 + Vrc1 + Vrc2;
    V_cell = max(V_min_cell, min(V_max_cell, V_cell));

    % ================================================================
    %  14. THERMAL MODEL
    % ================================================================
    %   Q_gen = Q_Joule + Q_reversible
    %
    %   Joule / irreversible:  Q_irr = Σ_cells I² · R_total
    %   Reversible / entropic: Q_rev = Σ_cells I · T · dU/dT
    %       sign: during charge with dU/dT < 0 → endothermic (cooling)
    %             during discharge with dU/dT < 0 → exothermic (heating)
    %
    %   Lumped energy balance:
    %       m·Cp · dT/dt = Q_gen − h·A·(T − T_amb)

    R_tot  = R0 + R1 + R2;
    Q_irr  = N_cells * I_cell * I_cell * R_tot;        % [W]

    dUdT   = entropy_nmc(soc);                          % [V/K]
    Q_rev  = N_cells * I_cell * T_K * dUdT;             % [W]

    Q_gen  = Q_irr + Q_rev;                             % [W]
    T_amb_K = T_ambient + 273.15;
    Q_cool = h_conv * A_surf * (T_K - T_amb_K);         % [W]

    dTdt = (Q_gen - Q_cool) / (m_pack * Cp_pack);
    T_K  = T_K + dTdt * dt;
    T_K  = max(T_amb_K - 5.0, min(333.15, T_K));        % clamp 60 °C

    % ================================================================
    %  15. SOC UPDATE  (Coulomb counting)
    % ================================================================
    dQ_Ah = I_cell * dt_h;                              % [Ah]
    soc   = soc + dQ_Ah / Q_eff;
    soc   = max(SoC_min, min(SoC_max, soc));

    % ================================================================
    %  16. AGING MODEL
    % ================================================================
    % ─── Calendar: SEI film growth  (Arrhenius + SoC stress) ───
    %     dq/dt = k(T) · σ(SoC)
    %     k(T)  = k_ref · exp( −E_a/R · (1/T − 1/T_ref) )
    %     σ(z)  = 1 + 2·(z − 0.5)²   (elevated SoC accelerates SEI)
    inv_dT_ag  = 1.0 / T_K - 1.0 / T_ref_K;
    k_cal_T    = k_cal_ref * exp(-Ea_cal * inv_dT_ag);
    soc_stress = 1.0 + 2.0 * (soc - 0.5)^2;
    q_cal      = q_cal + k_cal_T * soc_stress * dt;

    % ─── Cycle: Ah-throughput  (√Ah model) ───
    %     q_cyc = k_cyc · √(Ah / Q_nom)
    Ah_thru = Ah_thru + abs(dQ_Ah);
    q_cyc   = k_cyc * sqrt(Ah_thru / Q_nom);

    % ================================================================
    %  17. PACK-LEVEL OUTPUTS
    % ================================================================
    SoC         = soc;
    P_actual    = N_s * V_cell * N_p * I_cell / 1.0e6;    % [MW]
    E_available = max(0, (soc - SoC_min) * Q_eff * N_p ...
                      * U_oc * N_s / 1.0e6);               % [MWh]
    V_terminal  = N_s * V_cell;                             % [V]
    T_cell      = T_K - 273.15;                             % [°C]
    R_internal  = R_tot * R_aging;                          % [Ω]
    Q_capacity  = Q_eff;                                    % [Ah]

end  % community_battery


% ====================================================================
%  SUB-FUNCTIONS — electrochemical constitutive relations
% ====================================================================

function U = ocv_nmc(z)
%OCV_NMC  Open-circuit voltage for NMC622 / graphite cell.
%
%   Thermodynamic equilibrium potential as a function of lithiation
%   fraction z ∈ [0, 1].
%
%   Derived from a Redlich-Kister expansion of the Gibbs free energy
%   of mixing in the intercalation electrodes, with exponential
%   corrections for the two-phase regions near z→0 (graphite staging
%   transitions at low lithiation) and z→1 (NMC layered-oxide
%   saturation near full lithiation).
%
%   Polynomial coefficients fitted to coin-cell OCV data
%   (C/25 charge–discharge average) for a commercial NMC622 cell.
%
%      SoC 0.01 → ~2.9 V     SoC 0.50 → ~3.79 V
%      SoC 0.10 → ~3.50 V    SoC 0.90 → ~3.97 V
%      SoC 0.20 → ~3.64 V    SoC 1.00 → ~4.21 V

    z = max(0.005, min(0.995, z));

    U = 3.4323 ...
      + 1.6828  * z ...
      - 4.2105  * z^2 ...
      + 6.7082  * z^3 ...
      - 5.1267  * z^4 ...
      + 1.5533  * z^5 ...
      - 0.5090  * exp(-20.0 * z) ...           % graphite staging (low z)
      + 0.1680  * exp(-15.0 * (1.0 - z));      % NMC saturation  (high z)
end


function dUdT = entropy_nmc(z)
%ENTROPY_NMC  Entropy coefficient dU/dT [V/K] for NMC622 cell.
%
%   Measured by the potentiometric method: cell is equilibrated at
%   fixed SoC, then temperature is cycled and OCV change recorded.
%
%   The oscillatory structure arises from graphite staging transitions
%   on the negative electrode (LiC₁₂ ↔ LiC₆ phases at z ≈ 0.5).
%
%   Typical range:  −6 × 10⁻⁴  to  +1 × 10⁻⁴  V/K

    z = max(0.05, min(0.95, z));

    dUdT = -4.00e-4 ...
         + 2.00e-4 * sin(3.14159265 * z) ...
         - 1.50e-4 * cos(6.28318530 * z) ...
         + 0.80e-4 * sin(9.42477795 * z);        % 3rd harmonic (staging)
end


function f = soc_factor_R0(z)
%SOC_FACTOR_R0  SoC-dependent modulation of ohmic resistance.
%
%   Physics:
%     Low SoC  → electrolyte salt concentration drops in the negative-
%                electrode pores, reducing ionic conductivity → R₀ rises.
%     High SoC → concentration polarisation at the positive electrode
%                surface causes a modest resistance increase.

    f = 1.0 + 0.30 * exp(-8.0 * z) ...
            + 0.10 * exp(-8.0 * (1.0 - z));
end


function f = bv_factor(z, T_K, T_ref_K)
%BV_FACTOR  Butler-Volmer charge-transfer resistance factor.
%
%   From the linearised Butler-Volmer equation (small η):
%
%       R_ct = R·T / (α·F·i₀·A_electrode)
%
%   The exchange current density i₀ depends on surface Li
%   concentration through:
%
%       i₀ = k₀ · c_s^α · (c_max − c_s)^(1−α) · exp(−Ea / (R·T))
%
%   Normalised so that f(0.5, T_ref) = 1.
%
%   At extreme SoC the surface concentration c_s approaches 0 or c_max,
%   which drives i₀ → 0 and R_ct → ∞ — this captures the kinetic
%   limitation observed at end-of-charge / end-of-discharge.

    alpha  = 0.5;
    z      = max(0.01, min(0.99, z));

    i0_z   = z^alpha * (1.0 - z)^(1.0 - alpha);
    i0_ref = 0.5^alpha * 0.5^(1.0 - alpha);       % = 0.5

    T_factor = T_K / T_ref_K;

    f = T_factor / (i0_z / i0_ref + 1.0e-6);
end

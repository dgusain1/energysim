# -*- coding: utf-8 -*-
"""2-person household with EV + rooftop PV — multi-physics agent model.

Load is built from individual appliance profiles (no "base load"):
  fridge, lighting, cooking, washing machine, dishwasher,
  heating (thermal building R-C model), entertainment, EV charging
  (electrochemical CC-CV model), rooftop PV (temperature-derated).

Multi-physics sub-models
────────────────────────
  Building thermal:
    2R-1C lumped model — wall + window thermal resistance,
    internal thermal mass (furniture, air), solar heat gains,
    internal heat gains from occupants + appliances.
    dT_in/dt = (Q_solar + Q_internal + Q_heat − (T_in−T_amb)/R_eq) / C_bldg

  PV array (5 kWp):
    Cell temperature from NOCT model:
      T_cell = T_amb + (NOCT − 20)/800 · G
    Temperature derating:
      P = P_STC · (G/1000) · η_sys · [1 + γ·(T_cell − 25)]
    γ = −0.004 /°C  (crystalline Si, ≈ −0.4 %/°C)

  EV battery (40 kWh NMC):
    Simplified electrochemical model with R_int(SoC, T).
    CC-CV charging: constant current up to 80 % SoC, then
    constant voltage (exponentially tapering current).
    Cold-weather penalty via Arrhenius resistance increase.

Inputs:  T_ambient, solar
Outputs: P_net    [MW]  net load (positive = consuming, negative = export)
         P_pv     [MW]  PV generation (>= 0)
         P_ev     [MW]  EV charging power (>= 0)
         T_indoor [°C]  indoor temperature
"""

import math
from energysim.base import SimulatorAdapter


class external_simulator(SimulatorAdapter):
    """Multi-physics 2-person household with EV (40 kWh) and PV (5 kWp)."""

    # ════════════════════════════════════════════════════════
    #  Building thermal parameters  (2R-1C lumped model)
    # ════════════════════════════════════════════════════════
    R_wall    = 0.0035      # [K/W]  wall+roof conductive resistance
    R_window  = 0.012       # [K/W]  window + infiltration resistance
    C_bldg    = 8.0e6       # [J/K]  thermal capacitance (80 m², heavy)
    A_window  = 8.0         # [m²]   south-facing window area (solar gain)
    SHGC      = 0.60        # [-]    solar heat gain coefficient
    Q_occupant = 0.160      # [kW]   2 persons × 80 W metabolic

    # ════════════════════════════════════════════════════════
    #  PV parameters  (5 kWp crystalline Si)
    # ════════════════════════════════════════════════════════
    PV_KWP    = 5.0         # [kWp]  STC rating
    PV_ETA    = 0.85        # [-]    system efficiency (inverter, wiring, soiling)
    NOCT      = 45.0        # [°C]   nominal operating cell temperature
    GAMMA_T   = -0.004      # [1/°C] power temperature coefficient

    # ════════════════════════════════════════════════════════
    #  EV battery parameters  (electrochemical NMC)
    # ════════════════════════════════════════════════════════
    EV_CAP    = 40.0        # [kWh]  nominal capacity
    EV_V_NOM  = 360.0       # [V]    nominal pack voltage
    EV_R_INT_REF = 0.08     # [Ω]    internal resistance at SoC 0.5, 25 °C
    EV_EA_R   = 3000.0      # [K]    Arrhenius activation (E_a / R_gas)
    EV_P_MAX  = 7.4         # [kW]   L2 EVSE rating
    EV_CV_SOC = 0.80        # [-]    CC→CV transition SoC
    EV_I_MAX  = 20.5        # [A]    max CC current (7.4 kW / 360 V)

    def __init__(self, inputs=None, outputs=None, **kwargs):
        self.inputs = inputs or []
        self.outputs = outputs or []

        # Co-sim inputs
        self.T_ambient = 5.0
        self.solar = 0.0

        # Outputs (MW)
        self.P_net = 0.0
        self.P_pv = 0.0
        self.P_ev = 0.0
        self.T_indoor = 18.0

        # --- Building thermal state ---
        self.T_in = 18.0        # [°C] initial indoor temperature

        # --- EV state ---
        self.ev_soc = 0.40      # start at 40 %
        self.ev_T_batt = 15.0   # [°C] EV battery temperature
        self.ev_home = False

    def init(self):
        pass

    def set_value(self, parameters, values):
        for p, v in zip(parameters, values):
            if p == 'T_ambient':
                self.T_ambient = v
            elif p == 'solar':
                self.solar = v

    def get_value(self, parameters, time):
        result = []
        for p in parameters:
            if p == 'P_net':       result.append(self.P_net)
            elif p == 'P_pv':     result.append(self.P_pv)
            elif p == 'P_ev':     result.append(self.P_ev)
            elif p == 'T_indoor': result.append(self.T_indoor)
            else:                 result.append(0.0)
        return result

    def step(self, time):
        hour = (time % 86400) / 3600.0
        dt_s = self.step_size             # seconds
        dt_h = dt_s / 3600.0

        # ── Individual appliances (kW) ──
        P_fridge   = self._fridge(time)
        P_light    = self._lighting(hour)
        P_cook     = self._cooking(hour)
        P_wash     = self._washing(hour)
        P_dish     = self._dishwasher(hour)
        P_tv       = self._entertainment(hour)
        P_standby  = self._standby()

        Q_internal_kw = (P_fridge + P_light + P_cook + P_wash
                         + P_dish + P_tv + P_standby
                         + self.Q_occupant * self._occupancy(hour))

        # ── Building thermal model (2R-1C) ──
        P_heat = self._building_thermal(hour, dt_s, Q_internal_kw)

        # ── EV charging (electrochemical CC-CV) ──
        P_ev = self._ev_charging_echem(hour, dt_h)

        # ── PV generation (temperature-derated) ──
        P_pv = self._pv_thermal(self.solar)

        # ── Totals (kW → MW) ──
        P_total = (P_fridge + P_light + P_cook + P_wash + P_dish
                   + P_heat + P_tv + P_standby + P_ev)

        self.P_ev  = P_ev / 1000.0
        self.P_pv  = P_pv / 1000.0
        self.P_net = (P_total - P_pv) / 1000.0
        self.T_indoor = self.T_in

    # ════════════════════════════════════════════════════════
    #  Building thermal model
    # ════════════════════════════════════════════════════════

    def _building_thermal(self, hour, dt_s, Q_internal_kw):
        """2R-1C lumped thermal model with solar gain.

        Returns heating power [kW] consumed by the heating system.
        Updates self.T_in (indoor temperature state).
        """
        # Setpoint schedule (°C)
        if 6.0 <= hour < 8.5 or 17.0 <= hour < 23.0:
            T_set = 20.0
        else:
            T_set = 16.0

        # Equivalent parallel thermal resistance
        R_eq = 1.0 / (1.0 / self.R_wall + 1.0 / self.R_window)

        # Solar heat gain through windows [kW]
        Q_solar_kw = self.A_window * self.SHGC * self.solar / 1000.0

        # Total passive heat input [kW]
        Q_passive = Q_internal_kw + Q_solar_kw

        # Heating demand from thermostat [kW]
        err = T_set - self.T_in
        if err > 0:
            P_heat = min(2.0, 0.5 * err)   # P-controller, 2 kW max
        else:
            P_heat = 0.0

        # Energy balance: C · dT/dt = Q_total − (T_in − T_amb) / R_eq
        Q_total_W = (Q_passive + P_heat) * 1000.0      # [W]
        Q_loss_W = (self.T_in - self.T_ambient) / R_eq  # [W]

        dTdt = (Q_total_W - Q_loss_W) / self.C_bldg
        self.T_in += dTdt * dt_s
        self.T_in = max(self.T_ambient - 2.0, min(30.0, self.T_in))

        return P_heat

    @staticmethod
    def _occupancy(hour):
        """Returns fraction of 2 occupants present."""
        if 7.5 <= hour < 17.5:
            return 0.0          # both at work
        elif 6.0 <= hour < 7.5 or 17.5 <= hour < 23.0:
            return 1.0          # both home
        return 0.8              # sleeping

    # ════════════════════════════════════════════════════════
    #  PV model with thermal derating
    # ════════════════════════════════════════════════════════

    def _pv_thermal(self, G):
        """5 kWp PV with NOCT cell temperature and thermal derating.

        Cell temperature model (Ross / NOCT):
            T_cell = T_amb + (NOCT − 20) / 800 · G
        Power derating:
            P = P_STC · (G/1000) · η · [1 + γ·(T_cell − 25)]

        Args:
            G: solar irradiance [W/m²]
        Returns:
            PV power [kW]
        """
        if G <= 0:
            return 0.0

        # Cell temperature
        T_cell = self.T_ambient + (self.NOCT - 20.0) / 800.0 * G

        # Temperature derating factor
        derating = 1.0 + self.GAMMA_T * (T_cell - 25.0)
        derating = max(0.0, derating)

        P_kw = self.PV_KWP * (G / 1000.0) * self.PV_ETA * derating
        return max(0.0, P_kw)

    # ════════════════════════════════════════════════════════
    #  EV battery — electrochemical CC-CV charging model
    # ════════════════════════════════════════════════════════

    def _ev_charging_echem(self, hour, dt_h):
        """Electrochemical EV charging with CC-CV profile.

        CC phase: constant current up to CV_SOC (80 %).
        CV phase: constant voltage, current tapers exponentially.

        Internal resistance varies with SoC and temperature:
            R(SoC, T) = R_ref · f(SoC) · exp(Ea/R · (1/T − 1/T_ref))

        Heat generation: I²R → warms the battery.
        Cold-weather penalty: higher R → lower effective charge rate.
        """
        self.ev_home = (hour >= 18.0 or hour < 7.5)

        if not self.ev_home or self.ev_soc >= 0.98:
            # Battery cools toward ambient when not charging
            tau_cool = 3600.0  # 1 hour thermal time constant
            self.ev_T_batt += (self.T_ambient - self.ev_T_batt) * \
                min(1.0, dt_h * 3600.0 / tau_cool)
            return 0.0

        # SoC-dependent internal resistance factor
        z = max(0.01, min(0.99, self.ev_soc))
        f_soc = 1.0 + 0.3 * math.exp(-6.0 * z) + \
            0.2 * math.exp(-6.0 * (1.0 - z))

        # Temperature-dependent resistance (Arrhenius)
        T_batt_K = self.ev_T_batt + 273.15
        T_ref_K = 298.15
        f_temp = math.exp(self.EV_EA_R * (1.0 / T_batt_K - 1.0 / T_ref_K))
        R_int = self.EV_R_INT_REF * f_soc * f_temp

        # OCV of EV pack (simplified NMC polynomial, pack-level)
        V_ocv = self.EV_V_NOM * (0.88 + 0.24 * z - 0.12 * z * z)

        if self.ev_soc < self.EV_CV_SOC:
            # ── CC phase: constant current ──
            I_cc = min(self.EV_I_MAX,
                       self.EV_P_MAX * 1000.0 / (V_ocv + self.EV_I_MAX * R_int))
            I_charge = I_cc
        else:
            # ── CV phase: voltage held at V_max, current tapers ──
            V_max = self.EV_V_NOM * (0.88 + 0.24 * 0.98 - 0.12 * 0.98**2)
            I_charge = max(0.0, (V_max - V_ocv) / R_int)
            I_charge = min(I_charge, self.EV_I_MAX)

        # Power and energy
        V_term = V_ocv + I_charge * R_int
        P_charge_kw = V_term * I_charge / 1000.0
        P_charge_kw = min(P_charge_kw, self.EV_P_MAX)

        # SoC update (electrochemical Coulomb counting)
        Q_Ah = self.EV_CAP * 1000.0 / self.EV_V_NOM   # pack capacity [Ah]
        dQ = I_charge * dt_h                            # [Ah]
        self.ev_soc += dQ / Q_Ah
        self.ev_soc = max(0.0, min(0.98, self.ev_soc))

        # Thermal: I²R heating + convective cooling
        Q_heat_W = I_charge * I_charge * R_int
        tau_ev_cool = 2400.0    # EV battery thermal time constant [s]
        m_cp_ev = 50.0 * 1000.0  # 50 kg × 1000 J/(kg·K)
        dT = (Q_heat_W - (self.ev_T_batt - self.T_ambient)
              * m_cp_ev / tau_ev_cool) / m_cp_ev * (dt_h * 3600.0)
        self.ev_T_batt += dT
        self.ev_T_batt = max(self.T_ambient - 5.0, min(45.0, self.ev_T_batt))

        return max(0.0, P_charge_kw)

    # ════════════════════════════════════════════════════════
    #  Appliance models (bottom-up)
    # ════════════════════════════════════════════════════════

    def _fridge(self, time):
        """Refrigerator: 0.08 kW base + compressor cycles (20 min on / 40 min off).

        Compressor duty cycle varies slightly with ambient temperature
        (warmer → longer on-time) to model realistic thermostatic control.
        """
        base = 0.08
        duty = min(0.55, 0.33 + 0.01 * max(0, self.T_ambient - 15.0))
        period = 3600.0     # 1 hour cycle
        on_time = duty * period
        t_in_cycle = time % period
        compressor = 0.22 if t_in_cycle < on_time else 0.0
        return base + compressor

    def _lighting(self, hour):
        """Lighting: occupancy + daylight dependent."""
        if 6.0 <= hour < 8.0:
            return 0.15
        elif 8.0 <= hour < 17.0:
            return 0.02
        elif 17.0 <= hour < 23.0:
            return 0.30
        return 0.02

    @staticmethod
    def _cooking(hour):
        """Cooking: breakfast and dinner peaks."""
        if 6.5 <= hour < 7.5:
            return 1.5
        elif 18.0 <= hour < 19.5:
            return 2.0
        return 0.0

    @staticmethod
    def _washing(hour):
        """Washing machine: programmed run mid-morning."""
        if 10.0 <= hour < 11.5:
            return 0.50
        return 0.0

    @staticmethod
    def _dishwasher(hour):
        """Dishwasher: runs after dinner."""
        if 20.0 <= hour < 21.5:
            return 1.2
        return 0.0

    @staticmethod
    def _entertainment(hour):
        """TV + electronics: evening."""
        if 18.0 <= hour < 23.0:
            return 0.20
        return 0.0

    @staticmethod
    def _standby():
        """Always-on standby: router, clocks, chargers."""
        return 0.05

    def cleanup(self):
        pass

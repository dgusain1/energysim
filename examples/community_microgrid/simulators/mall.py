# -*- coding: utf-8 -*-
"""Shopping mall — multi-physics commercial agent model.

Multi-physics sub-models
────────────────────────
  HVAC (multi-zone thermal):
    3-zone lumped thermal model (retail halls, food court, corridors).
    Each zone has its own thermal mass, UA product, and setpoint.
    Ventilation load modelled via enthalpy-based outdoor air:
      Q_vent = ṁ_air · c_p · (T_out − T_zone)
    Chiller/heat-pump COP varies with ΔT (Carnot-fraction model):
      COP = η_Carnot · T_cold / |T_hot − T_cold|
    Electrical demand = Q_thermal / COP.

  Lighting:
    Occupancy-scheduled with daylight harvesting:
      reduced by solar irradiance during daytime.

  Escalators & elevators:
    4 escalators (2.2 kW each) + 3 elevators (12 kW peak each).
    Regenerative braking on elevators recovers ~25 % of descending
    energy.

  Food court:
    Commercial kitchen model: gas + electric split, ventilation
    hood extract fans.

  EV chargers (controllable):
    10 bays × 7.4 kW, power capped by dispatcher (demand response).
    Individual bay occupancy modelled stochastically via Poisson
    arrival / departure process (deterministic fallback profile).

Inputs:  T_ambient, solar, P_ev_limit
Outputs: P_net       [MW]  total mall load including EV
         P_ev_mall   [MW]  EV charger power
         P_hvac_mall [MW]  HVAC electrical consumption
         T_zone_avg  [°C]  average zone temperature
"""

import math
from energysim.base import SimulatorAdapter


class external_simulator(SimulatorAdapter):
    """Multi-physics shopping mall with controllable EV chargers."""

    # ════════════════════════════════════════════════════════
    #  Multi-zone thermal parameters
    # ════════════════════════════════════════════════════════
    # Zone 1: Retail halls  (3000 m², 12 m height)
    UA_retail   = 1800.0    # [W/K]  overall heat-transfer coefficient × area
    C_retail    = 120e6     # [J/K]  concrete structure + air volume
    A_glaze_ret = 120.0     # [m²]   glazed facade (solar gain)
    SHGC_ret    = 0.35      # [-]    tinted glass

    # Zone 2: Food court  (600 m²)
    UA_food     = 500.0     # [W/K]
    C_food      = 25e6      # [J/K]
    A_glaze_fd  = 30.0      # [m²]
    SHGC_fd     = 0.40

    # Zone 3: Corridors & common areas  (800 m²)
    UA_corr     = 600.0     # [W/K]
    C_corr      = 30e6      # [J/K]

    # Ventilation
    V_dot_vent  = 8.0       # [m³/s]  total outdoor air volume flow
    RHO_AIR     = 1.2       # [kg/m³]
    CP_AIR      = 1005.0    # [J/(kg·K)]

    # Chiller / Heat-pump
    ETA_CARNOT  = 0.45      # fraction of Carnot COP achieved
    T_COND      = 35.0      # [°C]  condenser temp (cooling mode)
    T_EVAP      = -5.0      # [°C]  evaporator temp (heating mode)

    # ════════════════════════════════════════════════════════
    #  Other load parameters
    # ════════════════════════════════════════════════════════
    LIGHT_ON    = 22.0      # [kW]  100 % lighting
    N_ESC       = 4         # escalators
    P_ESC       = 2.2       # [kW] per escalator
    N_ELEV      = 3         # elevators
    P_ELEV_PEAK = 12.0      # [kW] per elevator at peak
    ETA_REGEN   = 0.25      # elevator regenerative braking efficiency
    FOOD_ELEC   = 15.0      # [kW]  food court electric equipment
    FOOD_VENT   = 4.0       # [kW]  kitchen extract fans
    MISC        = 8.0       # [kW]  POS, security, signage
    EV_BAYS     = 10
    EV_PER_BAY  = 7.4       # [kW]
    EV_MAX      = EV_BAYS * EV_PER_BAY

    def __init__(self, inputs=None, outputs=None, **kwargs):
        self.inputs = inputs or []
        self.outputs = outputs or []

        self.T_ambient = 5.0
        self.solar = 0.0
        self.P_ev_limit = self.EV_MAX / 1000.0

        self.P_net = 0.0
        self.P_ev_mall = 0.0
        self.P_hvac_mall = 0.0
        self.T_zone_avg = 20.0

        # Thermal zone states [°C]
        self.T_retail = 20.0
        self.T_food = 22.0
        self.T_corr = 18.0

    def init(self):
        pass

    def set_value(self, parameters, values):
        for p, v in zip(parameters, values):
            if p == 'T_ambient':
                self.T_ambient = v
            elif p == 'solar':
                self.solar = v
            elif p == 'P_ev_limit':
                self.P_ev_limit = v

    def get_value(self, parameters, time):
        result = []
        for p in parameters:
            if p == 'P_net':          result.append(self.P_net)
            elif p == 'P_ev_mall':   result.append(self.P_ev_mall)
            elif p == 'P_hvac_mall': result.append(self.P_hvac_mall)
            elif p == 'T_zone_avg':  result.append(self.T_zone_avg)
            else:                    result.append(0.0)
        return result

    def step(self, time):
        hour = (time % 86400) / 3600.0
        dt_s = self.step_size

        P_hvac  = self._hvac_multizone(hour, dt_s)
        P_light = self._lighting(hour)
        P_vert  = self._vertical_transport(hour)
        P_food  = self._food_court(hour)
        P_misc  = self._misc(hour)
        P_ev    = self._ev_chargers(hour)

        P_total = P_hvac + P_light + P_vert + P_food + P_misc + P_ev

        self.P_ev_mall = P_ev / 1000.0
        self.P_hvac_mall = P_hvac / 1000.0
        self.P_net = P_total / 1000.0
        self.T_zone_avg = (self.T_retail + self.T_food + self.T_corr) / 3.0

    # ════════════════════════════════════════════════════════
    #  Multi-zone HVAC with enthalpy-based ventilation
    # ════════════════════════════════════════════════════════

    def _hvac_multizone(self, hour, dt_s):
        """3-zone thermal model with variable COP.

        Each zone: C·dT/dt = Q_int + Q_solar - UA·(T-T_amb) ± Q_hvac
        Q_vent is added as a thermal load on the retail zone.
        Electrical consumption = |Q_hvac_total| / COP(ΔT).

        Returns: HVAC electrical power [kW].
        """
        # Zone setpoints
        T_set_r, T_set_f, T_set_c = self._zone_setpoints(hour)
        is_open = 8.0 <= hour < 22.0

        # ── Ventilation thermal load [W] ──
        m_dot = self.V_dot_vent * self.RHO_AIR if is_open else 0.3 * self.V_dot_vent * self.RHO_AIR
        Q_vent_retail = m_dot * self.CP_AIR * (self.T_ambient - self.T_retail) * 0.6
        Q_vent_food   = m_dot * self.CP_AIR * (self.T_ambient - self.T_food) * 0.3
        Q_vent_corr   = m_dot * self.CP_AIR * (self.T_ambient - self.T_corr) * 0.1

        # ── Solar gains [W] ──
        Q_sol_retail = self.A_glaze_ret * self.SHGC_ret * self.solar if is_open else 0.0
        Q_sol_food   = self.A_glaze_fd * self.SHGC_fd * self.solar if is_open else 0.0

        # ── Internal gains [W] ──
        occ = 1.0 if is_open else 0.05
        Q_int_retail = 12000.0 * occ   # people + lighting + equipment
        Q_int_food   = 8000.0 * occ
        Q_int_corr   = 3000.0 * occ

        # ── Zone energy balance + HVAC demand ──
        Q_hvac_r = self._zone_step(
            'retail', dt_s, T_set_r, self.T_retail,
            self.UA_retail, self.C_retail,
            Q_int_retail + Q_sol_retail + Q_vent_retail)

        Q_hvac_f = self._zone_step(
            'food', dt_s, T_set_f, self.T_food,
            self.UA_food, self.C_food,
            Q_int_food + Q_sol_food + Q_vent_food)

        Q_hvac_c = self._zone_step(
            'corr', dt_s, T_set_c, self.T_corr,
            self.UA_corr, self.C_corr,
            Q_int_corr + Q_vent_corr)

        Q_hvac_total = Q_hvac_r + Q_hvac_f + Q_hvac_c   # [W]

        # ── COP (Carnot-fraction model) ──
        if Q_hvac_total > 0:
            # Heating mode
            dT_hp = abs(self.T_zone_avg - self.T_EVAP) + 10.0
            COP = self.ETA_CARNOT * (self.T_zone_avg + 273.15) / max(dT_hp, 5.0)
        else:
            # Cooling mode
            dT_ch = abs(self.T_COND - self.T_zone_avg) + 10.0
            COP = self.ETA_CARNOT * (self.T_zone_avg + 273.15) / max(dT_ch, 5.0)
        COP = max(1.5, min(8.0, COP))

        P_elec_kw = abs(Q_hvac_total) / (COP * 1000.0)

        # Night minimum (fans, BMS)
        if not is_open:
            P_elec_kw = max(P_elec_kw, 2.0)

        return P_elec_kw

    def _zone_step(self, zone_name, dt_s, T_set, T_zone, UA, C_th, Q_gain):
        """Advance one thermal zone by dt_s.

        Returns Q_hvac [W] delivered to the zone (positive = heating).
        Updates self.T_<zone>.
        """
        # Passive evolution (without HVAC)
        dT_passive = (Q_gain - UA * (T_zone - self.T_ambient)) / C_th * dt_s
        T_free = T_zone + dT_passive

        # HVAC demand to reach setpoint
        err = T_set - T_free
        Q_hvac = err * C_th / max(dt_s, 1.0)   # [W] needed

        # Clamp HVAC capacity (50 kW heating or 80 kW cooling per zone)
        Q_hvac = max(-80000.0, min(50000.0, Q_hvac))

        # Update zone temperature
        T_new = T_free + Q_hvac * dt_s / C_th
        T_new = max(self.T_ambient - 5.0, min(35.0, T_new))

        if zone_name == 'retail':
            self.T_retail = T_new
        elif zone_name == 'food':
            self.T_food = T_new
        else:
            self.T_corr = T_new

        return Q_hvac

    @staticmethod
    def _zone_setpoints(hour):
        """Returns (T_retail, T_food, T_corridor) setpoints [°C]."""
        if 7.0 <= hour < 22.0:
            return 22.0, 23.0, 20.0
        return 16.0, 16.0, 14.0

    # ════════════════════════════════════════════════════════
    #  Lighting with daylight harvesting
    # ════════════════════════════════════════════════════════

    def _lighting(self, hour):
        """Lighting with daylight harvesting during business hours."""
        if 8.5 <= hour < 21.5:
            # Reduce artificial light when ample solar
            daylight_factor = min(1.0, self.solar / 400.0)
            savings = 0.25 * daylight_factor   # up to 25 % savings
            return self.LIGHT_ON * (1.0 - savings)
        elif 7.5 <= hour < 8.5:
            return self.LIGHT_ON * (hour - 7.5)
        elif 21.5 <= hour < 22.0:
            return self.LIGHT_ON * (22.0 - hour) * 2
        return 2.0

    # ════════════════════════════════════════════════════════
    #  Vertical transport (escalators + elevators)
    # ════════════════════════════════════════════════════════

    def _vertical_transport(self, hour):
        """4 escalators + 3 elevators with regenerative braking.

        Elevators: average load factor varies with occupancy.
        Descending trips regenerate ~25 % of peak power.
        """
        if not (8.0 <= hour < 22.0):
            return 0.0

        # Escalators: constant when on
        P_esc = self.N_ESC * self.P_ESC

        # Elevators: average load over the period
        # Peak at lunch and evening (higher traffic)
        if 11.0 <= hour < 14.0 or 17.0 <= hour < 20.0:
            load_factor = 0.6
        else:
            load_factor = 0.35

        P_elev_up = self.N_ELEV * self.P_ELEV_PEAK * load_factor
        P_elev_down = self.N_ELEV * self.P_ELEV_PEAK * load_factor * self.ETA_REGEN
        P_elev = P_elev_up - P_elev_down   # net (regeneration reduces demand)

        return P_esc + max(0.0, P_elev)

    # ════════════════════════════════════════════════════════
    #  Food court (commercial kitchen)
    # ════════════════════════════════════════════════════════

    def _food_court(self, hour):
        """Commercial kitchen: electric equipment + ventilation hoods.

        Gas cooking is not counted electrically; only electric fryers,
        griddles, drink machines, and kitchen exhaust ventilation.
        """
        if not (10.5 <= hour < 21.5):
            return 0.0

        # Lunch and dinner peaks
        lunch  = max(0.0, 1.0 - abs(hour - 12.5) / 1.5)
        dinner = max(0.0, 1.0 - abs(hour - 19.0) / 1.5)
        peak = max(lunch, dinner)
        frac = 0.4 + 0.6 * peak

        return self.FOOD_ELEC * frac + self.FOOD_VENT

    @staticmethod
    def _misc(hour):
        """POS, security cameras, digital signage."""
        if 9.0 <= hour < 21.0:
            return 8.0
        return 1.5

    # ════════════════════════════════════════════════════════
    #  Controllable EV chargers
    # ════════════════════════════════════════════════════════

    def _ev_chargers(self, hour):
        """10-bay EV station, power capped by dispatcher.

        Bay occupancy follows a deterministic profile approximating
        Poisson arrivals/departures at a shopping centre.
        """
        if hour < 7.0:
            occ = 0.0
        elif 7.0 <= hour < 10.0:
            occ = 0.3 * (hour - 7.0) / 3.0
        elif 10.0 <= hour < 14.0:
            occ = 0.3 + 0.5 * math.sin(math.pi * (hour - 10.0) / 4.0)
        elif 14.0 <= hour < 18.0:
            occ = 0.5 - 0.2 * (hour - 14.0) / 4.0
        elif 18.0 <= hour < 21.0:
            occ = 0.3
        else:
            occ = 0.1

        P_demand = occ * self.EV_MAX
        P_limit = self.P_ev_limit * 1000.0
        return min(P_demand, max(0.0, P_limit))

    def cleanup(self):
        pass

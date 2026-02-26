# -*- coding: utf-8 -*-
"""Supermarket — multi-physics commercial agent model.

Multi-physics sub-models
────────────────────────
  Refrigeration (multi-temperature zone):
    Two independent thermodynamic zones:
      Freezer  (−25 °C setpoint):  R-C thermal model of insulated
        cabinet, compressor with COP = f(T_cond, T_evap), defrost
        heater cycling (20 min every 6 hours @ 5 kW).
      Chiller  (+2 °C setpoint):   same structure, higher COP.
    Total thermal mass includes product thermal inertia for
    realistic compressor cycling behaviour.

  HVAC (enthalpy-based):
    Store zone thermal model with ventilation load, internal
    gains (people, lighting, equipment), and refrigeration
    condenser reject heat.
    COP varies with ambient temperature (air-cooled condenser).

  Lighting:
    Daylight harvesting in perimeter zones reduces load by up to
    20 % when solar irradiance > 300 W/m².

  Bakery & deli:
    Electric ovens: 12 kW peak at 04:00–07:00 (baking) and
    10:00–14:00 (deli preparation).

Inputs:  T_ambient, solar
Outputs: P_net       [MW]  total supermarket load
         P_refrig    [MW]  refrigeration compressor power
         P_hvac_sup  [MW]  HVAC electrical consumption
         T_store     [°C]  store zone temperature
"""

import math
from energysim.base import SimulatorAdapter


class external_simulator(SimulatorAdapter):
    """Multi-physics supermarket with multi-zone refrigeration."""

    # ════════════════════════════════════════════════════════
    #  Freezer zone (−25 °C)
    # ════════════════════════════════════════════════════════
    FRZ_T_SET   = -25.0     # [°C] setpoint
    FRZ_UA      = 80.0      # [W/K]  cabinet UA (insulated)
    FRZ_C       = 2.0e6     # [J/K]  product + cabinet thermal mass
    FRZ_Q_PROD  = 500.0     # [W]    product replenishment / door openings
    FRZ_DEFROST = 5000.0    # [W]    defrost heater power
    FRZ_DEF_ON  = 1200.0    # [s]    defrost duration (20 min)
    FRZ_DEF_CYC = 21600.0   # [s]    defrost interval (6 hours)

    # ════════════════════════════════════════════════════════
    #  Chiller zone (+2 °C)
    # ════════════════════════════════════════════════════════
    CHL_T_SET   = 2.0       # [°C]
    CHL_UA      = 200.0     # [W/K]  open display case, higher UA
    CHL_C       = 3.0e6     # [J/K]
    CHL_Q_PROD  = 1500.0    # [W]    product load + customer interaction

    # ════════════════════════════════════════════════════════
    #  Compressor parameters
    # ════════════════════════════════════════════════════════
    ETA_CARNOT  = 0.40      # [-]  fraction of Carnot COP achieved
    T_COND_OFF  = 10.0      # [K]  condenser approach temperature

    # ════════════════════════════════════════════════════════
    #  Store HVAC
    # ════════════════════════════════════════════════════════
    STORE_UA    = 800.0     # [W/K]  store envelope
    STORE_C     = 50e6      # [J/K]  large thermal mass
    STORE_VENT  = 3.0       # [m³/s] outdoor air ventilation rate
    STORE_T_SET_DAY   = 20.0
    STORE_T_SET_NIGHT = 14.0

    # ════════════════════════════════════════════════════════
    #  Other loads
    # ════════════════════════════════════════════════════════
    LIGHT_ON    = 10.0      # [kW]   full store
    BAKERY_PEAK = 12.0      # [kW]   oven peak
    CHECKOUT    = 5.0       # [kW]   POS + misc

    def __init__(self, inputs=None, outputs=None, **kwargs):
        self.inputs = inputs or []
        self.outputs = outputs or []

        self.T_ambient = 5.0
        self.solar = 0.0
        self.P_net = 0.0
        self.P_refrig = 0.0
        self.P_hvac_sup = 0.0
        self.T_store = 18.0

        # State: zone temperatures
        self.T_freezer = self.FRZ_T_SET
        self.T_chiller = self.CHL_T_SET
        self.T_store_zone = 18.0

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
            if p == 'P_net':        result.append(self.P_net)
            elif p == 'P_refrig':  result.append(self.P_refrig)
            elif p == 'P_hvac_sup': result.append(self.P_hvac_sup)
            elif p == 'T_store':   result.append(self.T_store)
            else:                  result.append(0.0)
        return result

    def step(self, time):
        hour = (time % 86400) / 3600.0
        dt_s = self.step_size

        P_frz, Q_cond_frz = self._freezer_step(time, dt_s)
        P_chl, Q_cond_chl = self._chiller_step(dt_s)
        P_refrig = P_frz + P_chl
        Q_cond_total = Q_cond_frz + Q_cond_chl

        P_hvac   = self._store_hvac(hour, dt_s, Q_cond_total)
        P_light  = self._lighting(hour)
        P_bakery = self._bakery(hour)
        P_check  = self._checkout(hour)

        P_total = P_refrig + P_hvac + P_light + P_bakery + P_check

        self.P_refrig = P_refrig / 1000.0
        self.P_hvac_sup = P_hvac / 1000.0
        self.P_net = P_total / 1000.0
        self.T_store = self.T_store_zone

    # ════════════════════════════════════════════════════════
    #  Freezer − thermodynamic model with defrost
    # ════════════════════════════════════════════════════════

    def _freezer_step(self, time, dt_s):
        """Freezer zone: R-C thermal + compressor + defrost cycle.

        Returns (P_compressor [kW], Q_condenser_reject [W]).
        """
        # Defrost heater (20 min every 6 hours)
        t_in_cycle = time % self.FRZ_DEF_CYC
        is_defrost = t_in_cycle < self.FRZ_DEF_ON
        Q_defrost = self.FRZ_DEFROST if is_defrost else 0.0

        # Store zone is the ambient for the freezer cabinet
        T_amb_cab = self.T_store_zone

        # Heat leak into freezer [W]
        Q_leak = self.FRZ_UA * (T_amb_cab - self.T_freezer)

        # Total thermal load on freezer [W]
        Q_load = Q_leak + self.FRZ_Q_PROD + Q_defrost

        # Compressor: remove Q_load (if not defrosting)
        if is_defrost:
            Q_comp = 0.0   # compressor off during defrost
        else:
            Q_comp = Q_load + max(0.0, (self.T_freezer - self.FRZ_T_SET)
                                  * self.FRZ_C / max(dt_s, 1.0))
            Q_comp = max(0.0, Q_comp)

        # COP = η_Carnot · T_cold / (T_hot − T_cold)
        # T_hot = condenser temp = T_ambient + approach offset
        T_cold = self.T_freezer + 273.15
        T_hot = self.T_ambient + self.T_COND_OFF + 273.15
        delta_T = max(T_hot - T_cold, 10.0)
        COP_frz = self.ETA_CARNOT * T_cold / delta_T
        COP_frz = max(0.8, min(4.0, COP_frz))

        P_comp_W = Q_comp / COP_frz if Q_comp > 0 else 0.0
        Q_cond = Q_comp + P_comp_W   # condenser reject = Q_evap + W_comp

        # Update freezer temperature
        # C·dT/dt = Q_load − Q_comp
        net_Q = Q_load - Q_comp
        dT = net_Q / self.FRZ_C * dt_s
        self.T_freezer += dT
        self.T_freezer = max(-35.0, min(0.0, self.T_freezer))

        return P_comp_W / 1000.0, Q_cond   # kW, W

    # ════════════════════════════════════════════════════════
    #  Chiller − thermodynamic model (no defrost)
    # ════════════════════════════════════════════════════════

    def _chiller_step(self, dt_s):
        """Chiller zone: open display cases.

        Higher UA than freezer (open front), moderate product load.
        Returns (P_compressor [kW], Q_condenser_reject [W]).
        """
        T_amb_cab = self.T_store_zone
        Q_leak = self.CHL_UA * (T_amb_cab - self.T_chiller)
        Q_load = Q_leak + self.CHL_Q_PROD

        # Compressor demand
        Q_comp = Q_load + max(0.0, (self.T_chiller - self.CHL_T_SET)
                              * self.CHL_C / max(dt_s, 1.0))
        Q_comp = max(0.0, Q_comp)

        T_cold = self.T_chiller + 273.15
        T_hot = self.T_ambient + self.T_COND_OFF + 273.15
        delta_T = max(T_hot - T_cold, 8.0)
        COP_chl = self.ETA_CARNOT * T_cold / delta_T
        COP_chl = max(1.2, min(6.0, COP_chl))

        P_comp_W = Q_comp / COP_chl if Q_comp > 0 else 0.0
        Q_cond = Q_comp + P_comp_W

        net_Q = Q_load - Q_comp
        dT = net_Q / self.CHL_C * dt_s
        self.T_chiller += dT
        self.T_chiller = max(-5.0, min(10.0, self.T_chiller))

        return P_comp_W / 1000.0, Q_cond   # kW, W

    # ════════════════════════════════════════════════════════
    #  Store HVAC with condenser reject heat
    # ════════════════════════════════════════════════════════

    def _store_hvac(self, hour, dt_s, Q_cond_reject):
        """Store zone HVAC.

        Condenser reject heat from refrigeration is added to the
        store zone as an internal gain in summer (air-cooled racks
        inside the store) or can offset heating in winter.

        Returns: HVAC electrical demand [kW].
        """
        is_open = 6.0 <= hour < 23.0
        T_set = self.STORE_T_SET_DAY if is_open else self.STORE_T_SET_NIGHT

        # Ventilation load [W]
        rate = self.STORE_VENT if is_open else 0.3 * self.STORE_VENT
        Q_vent = rate * 1.2 * 1005.0 * (self.T_ambient - self.T_store_zone)

        # Internal gains [W]
        occ = 1.0 if is_open else 0.05
        Q_people = 4000.0 * occ
        Q_light = self.LIGHT_ON * 1000.0 * (0.8 if is_open else 0.15)
        Q_equip = 2000.0 * occ

        # Condenser heat reject into store (for air-cooled indoor units)
        # In winter this is beneficial; in summer it increases cooling load.
        Q_cond_indoor = Q_cond_reject * 0.6   # 60 % rejects into store

        Q_total = Q_vent + Q_people + Q_light + Q_equip + Q_cond_indoor

        # HVAC demand
        Q_loss = self.STORE_UA * (self.T_store_zone - self.T_ambient)
        dT_free = (Q_total - Q_loss) / self.STORE_C * dt_s
        T_free = self.T_store_zone + dT_free

        err = T_set - T_free
        Q_hvac = err * self.STORE_C / max(dt_s, 1.0)
        Q_hvac = max(-60000.0, min(40000.0, Q_hvac))

        T_new = T_free + Q_hvac * dt_s / self.STORE_C
        self.T_store_zone = max(self.T_ambient - 3.0, min(30.0, T_new))

        # COP for HVAC
        if Q_hvac > 0:
            dT_hp = abs(self.T_store_zone + 5.0 - self.T_ambient) + 8.0
            COP = 0.45 * (self.T_store_zone + 273.15) / max(dT_hp, 5.0)
        else:
            dT_ch = abs(35.0 - self.T_store_zone) + 8.0
            COP = 0.45 * (self.T_store_zone + 273.15) / max(dT_ch, 5.0)
        COP = max(1.5, min(7.0, COP))

        P_hvac_kw = abs(Q_hvac) / (COP * 1000.0)
        if not is_open:
            P_hvac_kw = max(P_hvac_kw, 1.0)

        return P_hvac_kw

    # ════════════════════════════════════════════════════════
    #  Lighting with daylight harvesting
    # ════════════════════════════════════════════════════════

    def _lighting(self, hour):
        """Store lighting with perimeter daylight harvesting."""
        if 6.5 <= hour < 22.5:
            daylight_factor = min(1.0, self.solar / 350.0)
            savings = 0.20 * daylight_factor
            return self.LIGHT_ON * (1.0 - savings)
        elif 6.0 <= hour < 6.5:
            return self.LIGHT_ON * (hour - 6.0) * 2
        elif 22.5 <= hour < 23.0:
            return self.LIGHT_ON * (23.0 - hour) * 2
        return 1.5

    # ════════════════════════════════════════════════════════
    #  Bakery & deli
    # ════════════════════════════════════════════════════════

    @staticmethod
    def _bakery(hour):
        """Electric bakery ovens + deli preparation.

        Early morning baking (04:00–07:00) and deli prep (10:00–14:00).
        """
        if 4.0 <= hour < 7.0:
            # Baking: ramp up, plateau, ramp down
            if hour < 5.0:
                frac = (hour - 4.0)
            elif hour < 6.0:
                frac = 1.0
            else:
                frac = (7.0 - hour)
            return 12.0 * max(0.0, frac)
        elif 10.0 <= hour < 14.0:
            # Deli preparation
            return 6.0
        elif 7.0 <= hour < 22.0:
            return 1.5      # warming displays
        return 0.5

    @staticmethod
    def _checkout(hour):
        """Self-checkout + staffed tills."""
        if 7.0 <= hour < 22.0:
            return 5.0
        return 0.5

    def cleanup(self):
        pass

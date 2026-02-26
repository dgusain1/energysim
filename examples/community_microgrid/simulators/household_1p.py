# -*- coding: utf-8 -*-
"""Single-person household — multi-physics agent model.

No EV, no PV.  Smaller thermal mass than a 2-person home.

Multi-physics sub-models
────────────────────────
  Building thermal (2R-1C):
    Lumped  wall / window thermal resistances, internal thermal
    mass, solar & internal heat gains, thermostat-controlled
    heating.  Smaller building envelope than the 2-person home
    (50 m² flat vs 80 m² house).

  Appliance breakdown:
    fridge (with ambient-dependent duty cycle), lighting,
    cooking (microwave-heavy), heating, entertainment, standby.

Inputs:  T_ambient, solar
Outputs: P_net    [MW]  (always positive — pure consumer)
         T_indoor [°C]  indoor temperature
"""

import math
from energysim.base import SimulatorAdapter


class external_simulator(SimulatorAdapter):
    """Multi-physics single-person household with thermal building model."""

    # ════════════════════════════════════════════════════════
    #  Building thermal parameters (2R-1C, 50 m² flat)
    # ════════════════════════════════════════════════════════
    R_wall    = 0.0050      # [K/W]  wall + roof (less area → higher R)
    R_window  = 0.018       # [K/W]  smaller window area
    C_bldg    = 5.0e6       # [J/K]  lighter construction, 50 m²
    A_window  = 4.0         # [m²]   south-facing window
    SHGC      = 0.55        # [-]    solar heat gain coefficient
    Q_occupant = 0.080      # [kW]   1 person × 80 W metabolic

    def __init__(self, inputs=None, outputs=None, **kwargs):
        self.inputs = inputs or []
        self.outputs = outputs or []

        self.T_ambient = 5.0
        self.solar = 0.0
        self.P_net = 0.0
        self.T_indoor = 17.0

        # Building thermal state
        self.T_in = 17.0

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
            elif p == 'T_indoor': result.append(self.T_indoor)
            else:                 result.append(0.0)
        return result

    def step(self, time):
        hour = (time % 86400) / 3600.0
        dt_s = self.step_size

        P_fridge  = self._fridge(time)
        P_light   = self._lighting(hour)
        P_cook    = self._cooking(hour)
        P_tv      = self._entertainment(hour)
        P_standby = 0.04

        # Internal heat gains for thermal model [kW]
        Q_internal_kw = (P_fridge + P_light + P_cook + P_tv + P_standby
                         + self.Q_occupant * self._occupancy(hour))

        # Building thermal model → heating demand
        P_heat = self._building_thermal(hour, dt_s, Q_internal_kw)

        P_total = P_fridge + P_light + P_cook + P_heat + P_tv + P_standby
        self.P_net = P_total / 1000.0
        self.T_indoor = self.T_in

    # ════════════════════════════════════════════════════════
    #  Building thermal model (2R-1C)
    # ════════════════════════════════════════════════════════

    def _building_thermal(self, hour, dt_s, Q_internal_kw):
        """Lumped thermal model for 50 m² flat.

        Same physics as the 2-person household but with smaller
        envelope, lighter construction, and a single occupant.

        Returns heating power [kW].  Updates self.T_in.
        """
        # Setpoint schedule
        if 6.5 <= hour < 8.5 or 17.5 <= hour < 23.0:
            T_set = 19.0
        else:
            T_set = 15.0

        # Equivalent thermal resistance
        R_eq = 1.0 / (1.0 / self.R_wall + 1.0 / self.R_window)

        # Solar gain through windows [kW]
        Q_solar_kw = self.A_window * self.SHGC * self.solar / 1000.0

        # Passive heat [kW]
        Q_passive = Q_internal_kw + Q_solar_kw

        # Thermostat P-controller, 1.5 kW max
        err = T_set - self.T_in
        P_heat = min(1.5, max(0.0, 0.4 * err))

        # Energy balance
        Q_total_W = (Q_passive + P_heat) * 1000.0
        Q_loss_W = (self.T_in - self.T_ambient) / R_eq

        dTdt = (Q_total_W - Q_loss_W) / self.C_bldg
        self.T_in += dTdt * dt_s
        self.T_in = max(self.T_ambient - 2.0, min(28.0, self.T_in))

        return P_heat

    @staticmethod
    def _occupancy(hour):
        """Single occupant: at home except daytime (WFH some days)."""
        if 8.5 <= hour < 17.0:
            return 0.3          # partial WFH
        return 1.0

    # ════════════════════════════════════════════════════════
    #  Appliance models (bottom-up)
    # ════════════════════════════════════════════════════════

    def _fridge(self, time):
        """Small fridge-freezer with ambient-dependent duty cycle."""
        base = 0.06
        duty = min(0.50, 0.33 + 0.01 * max(0, self.T_ambient - 15.0))
        period = 3600.0
        on_time = duty * period
        t_in_cycle = time % period
        comp = 0.18 if t_in_cycle < on_time else 0.0
        return base + comp

    @staticmethod
    def _lighting(hour):
        if 7.0 <= hour < 8.5:
            return 0.10
        elif 8.5 <= hour < 17.0:
            return 0.01
        elif 17.0 <= hour < 23.0:
            return 0.20
        return 0.01

    @staticmethod
    def _cooking(hour):
        """Quick meals: microwave + kettle."""
        if 7.0 <= hour < 7.5:
            return 1.0
        elif 12.0 <= hour < 12.5:
            return 0.8
        elif 19.0 <= hour < 19.5:
            return 1.2
        return 0.0

    @staticmethod
    def _entertainment(hour):
        if 18.0 <= hour < 23.5:
            return 0.15
        return 0.0

    def cleanup(self):
        pass

# -*- coding: utf-8 -*-
"""Rule-based energy management controller — external simulator for energysim.

Dispatches the battery (price-based charge/discharge) and heat pump
(proportional temperature control on greenhouse setpoint 18–22 °C).
"""

from energysim.base import SimulatorAdapter


class external_simulator(SimulatorAdapter):
    """Rule-based dispatch: battery charge/discharge + heat-pump modulation."""

    def __init__(self, inputs=None, outputs=None, **kwargs):
        self.inputs = inputs or []
        self.outputs = outputs or []

        # Inputs
        self.SoC = 0.5
        self.T_storage = 50.0
        self.T_greenhouse = 18.0
        self.P_pv = 0.0
        self.elec_price = 0.15
        self.heat_demand_signal = 0.0
        self.T_ambient = 5.0

        # Outputs  (MW)
        self.P_battery_cmd = 0.0
        self.P_hp_cmd = 0.0

    def init(self):
        pass

    def set_value(self, parameters, values):
        for p, v in zip(parameters, values):
            if p == 'SoC':
                self.SoC = v
            elif p == 'T_storage':
                self.T_storage = v
            elif p == 'T_greenhouse':
                self.T_greenhouse = v
            elif p == 'P_pv':
                self.P_pv = v
            elif p == 'elec_price':
                self.elec_price = v
            elif p == 'heat_demand_signal':
                self.heat_demand_signal = v
            elif p == 'T_ambient':
                self.T_ambient = v

    def get_value(self, parameters, time):
        result = []
        for p in parameters:
            if p == 'P_battery_cmd':
                result.append(self.P_battery_cmd)
            elif p == 'P_hp_cmd':
                result.append(self.P_hp_cmd)
            else:
                result.append(0.0)
        return result

    def step(self, time):
        # ── Heat-pump control ─────────────────────────────────────
        T_set_low = 18.0
        T_set_high = 22.0

        if self.T_greenhouse < T_set_low:
            P_hp = 0.003            # full power  (3 kW)
        elif self.T_greenhouse > T_set_high:
            P_hp = 0.0
        else:
            # Proportional band
            P_hp = 0.003 * (T_set_high - self.T_greenhouse) / (T_set_high - T_set_low)

        # Safety: stop if thermal tank is too hot
        if self.T_storage > 80.0:
            P_hp = 0.0

        self.P_hp_cmd = P_hp

        # ── Battery control ───────────────────────────────────────
        if self.elec_price < 0.10 and self.SoC < 0.9:
            P_batt = 0.005          # charge at max  (5 kW)
        elif self.elec_price > 0.20 and self.SoC > 0.2:
            P_batt = -0.005         # discharge at max
        elif self.P_pv > 0.003 and self.SoC < 0.95:
            P_batt = min(0.005, self.P_pv - 0.002)   # charge from excess PV
        else:
            P_batt = 0.0

        self.P_battery_cmd = P_batt

    def cleanup(self):
        pass

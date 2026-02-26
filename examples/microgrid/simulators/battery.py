# -*- coding: utf-8 -*-
"""Battery energy storage system — external simulator for energysim.

Models a Li-ion battery (e.g. Tesla Powerwall) with Euler-integrated
state-of-charge dynamics.  Capacity 13.5 kWh, max power 5 kW.
"""

import math
from energysim.base import SimulatorAdapter


class external_simulator(SimulatorAdapter):
    """Li-ion battery with Euler-integrated SoC dynamics."""

    def __init__(self, inputs=None, outputs=None, **kwargs):
        self.inputs = inputs or []
        self.outputs = outputs or []

        # State
        self.SoC = 0.5

        # Parameters
        self.capacity = 0.0135      # MWh (13.5 kWh)
        self.max_power = 0.005      # MW  (5 kW)
        self.efficiency = 0.95

        # I/O variables
        self.P_cmd = 0.0            # MW (positive = charge)
        self.P_actual = 0.0         # MW

    def init(self):
        pass

    def set_value(self, parameters, values):
        for p, v in zip(parameters, values):
            if p == 'P_cmd':
                self.P_cmd = v

    def get_value(self, parameters, time):
        result = []
        for p in parameters:
            if p == 'SoC':
                result.append(self.SoC)
            elif p == 'P_actual':
                result.append(self.P_actual)
            else:
                result.append(0.0)
        return result

    def step(self, time):
        dt = self.step_size
        sqrt_eff = math.sqrt(self.efficiency)

        # Clip command to rated power
        P = max(-self.max_power, min(self.max_power, self.P_cmd))

        # Compute SoC change  (capacity in MWh → MWs = capacity * 3600)
        if P >= 0:
            dSoC = P * sqrt_eff * dt / (self.capacity * 3600)
        else:
            dSoC = P / sqrt_eff * dt / (self.capacity * 3600)

        new_SoC = self.SoC + dSoC

        # Enforce SoC limits
        if new_SoC > 1.0:
            self.SoC = 1.0
            self.P_actual = 0.0
        elif new_SoC < 0.0:
            self.SoC = 0.0
            self.P_actual = 0.0
        else:
            self.SoC = new_SoC
            self.P_actual = P

    def cleanup(self):
        pass

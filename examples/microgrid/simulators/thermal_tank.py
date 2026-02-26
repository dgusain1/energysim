# -*- coding: utf-8 -*-
"""Stratified hot-water tank (single-node) — external simulator for energysim.

500 L water tank with Euler-integrated temperature dynamics.
"""

from energysim.base import SimulatorAdapter


class external_simulator(SimulatorAdapter):
    """Single-node thermal storage tank with Euler integration."""

    def __init__(self, inputs=None, outputs=None, **kwargs):
        self.inputs = inputs or []
        self.outputs = outputs or []

        # State
        self.T_storage = 50.0   # °C

        # Parameters  (internal calculations in kW / kJ)
        self.m_cp = 2093.0      # kJ/K   (500 L water, cp ≈ 4.186 kJ/(kg·K))
        self.UA = 0.005          # kW/K   (5 W/K heat-loss coefficient)

        # I/O variables  (MW at the energysim interface)
        self.Q_in = 0.0         # MW from heat pump
        self.Q_out = 0.0        # MW to greenhouse
        self.T_ambient = 5.0    # °C

        # Output
        self.Q_loss = 0.0       # kW

    def init(self):
        pass

    def set_value(self, parameters, values):
        for p, v in zip(parameters, values):
            if p == 'Q_in':
                self.Q_in = v
            elif p == 'Q_out':
                self.Q_out = v
            elif p == 'T_ambient':
                self.T_ambient = v

    def get_value(self, parameters, time):
        result = []
        for p in parameters:
            if p == 'T_storage':
                result.append(self.T_storage)
            elif p == 'Q_loss':
                result.append(self.Q_loss)
            else:
                result.append(0.0)
        return result

    def step(self, time):
        dt = self.step_size

        # Convert MW → kW for internal energy balance
        Q_in_kw = self.Q_in * 1000.0
        Q_out_kw = self.Q_out * 1000.0

        # Heat loss  (kW)
        self.Q_loss = self.UA * (self.T_storage - self.T_ambient)

        # Temperature change:  (kW) / (kJ/K) * s = K
        #   kW = kJ/s  →  (kJ/s) / (kJ/K) * s = K   ✓
        dT = (Q_in_kw - Q_out_kw - self.Q_loss) * dt / self.m_cp
        self.T_storage += dT
        self.T_storage = max(5.0, min(95.0, self.T_storage))

    def cleanup(self):
        pass

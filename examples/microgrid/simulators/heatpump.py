# -*- coding: utf-8 -*-
"""Air-source heat pump — external simulator for energysim.

Algebraic Carnot-fraction COP model: no internal dynamics.
Rated electrical input 3 kW.
"""

from energysim.base import SimulatorAdapter


class external_simulator(SimulatorAdapter):
    """Algebraic COP model of an air-source heat pump (no dynamics)."""

    def __init__(self, inputs=None, outputs=None, **kwargs):
        self.inputs = inputs or []
        self.outputs = outputs or []

        # I/O variables
        self.P_cmd = 0.0        # MW electrical command
        self.T_source = 5.0     # °C ambient air
        self.T_sink = 50.0      # °C storage temperature

        # Outputs
        self.Q_thermal = 0.0    # MW thermal
        self.COP = 3.0
        self.P_elec = 0.0       # MW

        # Rating
        self.max_elec = 0.003   # MW (3 kW)

    def init(self):
        pass

    def set_value(self, parameters, values):
        for p, v in zip(parameters, values):
            if p == 'P_cmd':
                self.P_cmd = v
            elif p == 'T_source':
                self.T_source = v
            elif p == 'T_sink':
                self.T_sink = v

    def get_value(self, parameters, time):
        result = []
        for p in parameters:
            if p == 'Q_thermal':
                result.append(self.Q_thermal)
            elif p == 'COP':
                result.append(self.COP)
            elif p == 'P_elec':
                result.append(self.P_elec)
            else:
                result.append(0.0)
        return result

    def step(self, time):
        dT = self.T_sink - self.T_source
        dT_safe = max(1.0, dT)

        # Carnot-fraction COP, clipped to [1, 6]
        COP = 0.45 * (self.T_sink + 273.15) / dT_safe
        self.COP = max(1.0, min(6.0, COP))

        # Clip electrical power to rating
        self.P_elec = max(0.0, min(self.max_elec, self.P_cmd))

        # Thermal output
        self.Q_thermal = self.COP * self.P_elec

    def cleanup(self):
        pass

# -*- coding: utf-8 -*-
"""Greenhouse thermal model — external simulator for energysim.

Single-zone greenhouse (~200 m²) with Euler-integrated air temperature.
Solar gain, heat-loss via UA, and supplied heating from thermal storage.
"""

from energysim.base import SimulatorAdapter


class external_simulator(SimulatorAdapter):
    """Single-zone greenhouse with Euler-integrated air temperature."""

    def __init__(self, inputs=None, outputs=None, **kwargs):
        self.inputs = inputs or []
        self.outputs = outputs or []

        # State
        self.T_inside = 18.0        # °C

        # Parameters
        self.m_cp_air = 723.6       # kJ/K   (≈600 m³ moist air)
        self.UA = 0.2               # kW/K
        self.solar_gain_factor = 0.02  # kW per (W/m²) — accounts for area & transmittance

        # I/O variables  (MW at energysim interface, °C, W/m²)
        self.Q_heating = 0.0        # MW from thermal storage
        self.T_ambient = 5.0        # °C
        self.solar_irradiance = 0.0  # W/m²

        # Outputs
        self.Q_demand = 0.0         # MW — estimated heat need to maintain 20 °C

    def init(self):
        pass

    def set_value(self, parameters, values):
        for p, v in zip(parameters, values):
            if p == 'Q_heating':
                self.Q_heating = v
            elif p == 'T_ambient':
                self.T_ambient = v
            elif p == 'solar_irradiance':
                self.solar_irradiance = v

    def get_value(self, parameters, time):
        result = []
        for p in parameters:
            if p == 'T_inside':
                result.append(self.T_inside)
            elif p == 'Q_demand':
                result.append(self.Q_demand)
            else:
                result.append(0.0)
        return result

    def step(self, time):
        dt = self.step_size

        # Solar gain  (kW)
        Q_solar = self.solar_gain_factor * self.solar_irradiance

        # Heat loss  (kW)
        Q_loss = self.UA * (self.T_inside - self.T_ambient)

        # Heating input  MW → kW
        Q_heat_kw = self.Q_heating * 1000.0

        # Temperature change:  (kW) / (kJ/K) * s = K
        dT = (Q_heat_kw + Q_solar - Q_loss) * dt / self.m_cp_air
        self.T_inside += dT
        self.T_inside = max(-10.0, min(50.0, self.T_inside))

        # Simple estimated demand to maintain setpoint (20 °C)
        self.Q_demand = max(0.0, self.UA * (20.0 - self.T_ambient)) / 1000.0  # MW

    def cleanup(self):
        pass

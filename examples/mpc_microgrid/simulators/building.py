# -*- coding: utf-8 -*-
"""Single-zone building thermal model — external simulator for energysim.

Lumped-capacitance (RC) model of a residential building with:
  - Thermal capacitance   C  = 2500 kJ/K  (~100 m² well-insulated)
  - Heat-loss coefficient UA = 80 W/K
  - Solar gain factor     gA = 4.0 m²  (effective glazing area)
  - Internal gains        Q_int = 0.3 kW (appliances + occupants)

Inputs:  P_hp, T_ambient, solar
Outputs: T_inside
"""

from energysim.base import SimulatorAdapter


class external_simulator(SimulatorAdapter):
    """Forward-Euler thermal model of a single zone."""

    # --- Physical parameters ---
    C     = 2500.0     # thermal capacitance  [kJ/K]
    UA    = 0.080      # heat-loss coeff      [kW/K]  (80 W/K)
    gA    = 4.0e-3     # solar gain factor    [kW per W/m²]
    Q_int = 0.3e-3     # internal gains       [MW] → kept in MW for grid compat
    COP_nom = 3.5      # nominal heat-pump COP (simple model)

    def __init__(self, inputs=None, outputs=None, **kwargs):
        self.inputs = inputs or []
        self.outputs = outputs or []

        # State
        self.T_inside = 20.0    # [°C]

        # Inputs (set each macro step via set_value)
        self.P_hp = 0.0         # heat-pump electrical power [MW]
        self.T_ambient = 5.0    # outdoor temperature [°C]
        self.solar = 0.0        # solar irradiance [W/m²]

    def init(self):
        pass

    def set_value(self, parameters, values):
        for p, v in zip(parameters, values):
            if p == 'P_hp':
                self.P_hp = v
            elif p == 'T_ambient':
                self.T_ambient = v
            elif p == 'solar':
                self.solar = v

    def get_value(self, parameters, time):
        result = []
        for p in parameters:
            if p == 'T_inside':
                result.append(self.T_inside)
            else:
                result.append(0.0)
        return result

    def step(self, time):
        dt = self.step_size  # seconds

        # Heat-pump thermal output [kW]
        Q_hp = self.P_hp * 1000.0 * self.COP_nom  # MW→kW * COP

        # Solar gains [kW]
        Q_solar = self.gA * self.solar             # kW

        # Internal gains [kW]
        Q_int = self.Q_int * 1000.0                # MW→kW

        # Heat loss [kW]
        Q_loss = self.UA * (self.T_inside - self.T_ambient)

        # Energy balance: C * dT/dt = Q_hp + Q_solar + Q_int - Q_loss
        dTdt = (Q_hp + Q_solar + Q_int - Q_loss) / self.C  # [K/s]
        self.T_inside += dTdt * dt

    def cleanup(self):
        pass

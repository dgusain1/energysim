# -*- coding: utf-8 -*-
"""Dispatch controller for 9-bus PowerFactory co-simulation.

This external simulator acts as a centralised controller that
dispatches the Power-to-Heat (P2H) and Power-to-Gas (P2G) flexible
loads based on:

  • Renewable surplus / deficit (wind + PV vs. conventional load)
  • Electricity price signal
  • Heat and gas demand requirements
  • Bus voltage feedback (curtail if voltage too low)

The controller prioritises renewable self-consumption:
  1. When generation > load, ramp up P2H / P2G to absorb surplus.
  2. When price is low, increase flexible loads.
  3. When voltage drops below threshold, curtail flexible loads.

Inputs
------
  wind_power   [MW]  Current wind farm output
  pv_power     [MW]  Current PV farm output
  elec_price   [EUR/MWh]  Electricity price
  p2h_demand   [MW]  Heat demand (desired P2H consumption)
  p2g_demand   [MW]  Gas demand (desired P2G consumption)
  bus_v_p2h    [pu]  Voltage at P2H bus
  bus_v_p2g    [pu]  Voltage at P2G bus

Outputs
-------
  P2H_cmd      [MW]  Active power setpoint for P2H load
  P2G_cmd      [MW]  Active power setpoint for P2G load
  ren_surplus  [MW]  Renewable surplus (positive = excess generation)
"""

import math
from energysim.base import SimulatorAdapter


class external_simulator(SimulatorAdapter):
    """Dispatch controller for P2H and P2G flexible loads."""

    # ── Load limits (must match 9-bus model) ──
    P2H_MAX = 50.0   # [MW] max P2H consumption
    P2H_MIN = 5.0    # [MW] min P2H consumption (keep-warm)
    P2G_MAX = 50.0   # [MW] max P2G consumption
    P2G_MIN = 5.0    # [MW] min P2G consumption (standby)

    # ── Conventional load on the 9-bus system ──
    CONV_LOAD = 125.0  # [MW] base conventional load (Load A+B+C in IEEE 9-bus)

    # ── Voltage limits ──
    V_MIN = 0.95     # [pu] curtail flexible load below this
    V_LOW = 0.97     # [pu] start reducing flexible load

    # ── Price thresholds ──
    PRICE_LOW  = 30.0   # [EUR/MWh] cheap → increase flexible load
    PRICE_HIGH = 60.0   # [EUR/MWh] expensive → decrease flexible load

    # ── Ramp rate ──
    RAMP_RATE = 10.0  # [MW/step] maximum change per macro time step

    def __init__(self, inputs=None, outputs=None, **kwargs):
        self.inputs = inputs or []
        self.outputs = outputs or []

        # Input state
        self.wind_power = 50.0
        self.pv_power = 0.0
        self.elec_price = 45.0
        self.p2h_demand = 20.0
        self.p2g_demand = 15.0
        self.bus_v_p2h = 1.0
        self.bus_v_p2g = 1.0

        # Output state
        self.P2H_cmd = 20.0
        self.P2G_cmd = 15.0
        self.ren_surplus = 0.0

        # Internal — previous commands for ramping
        self._prev_p2h = 20.0
        self._prev_p2g = 15.0

    # ── SimulatorAdapter interface ──────────────────────────────

    def init(self):
        pass

    def set_value(self, parameters, values):
        for p, v in zip(parameters, values):
            setattr(self, p, v)

    def get_value(self, parameters, time=None):
        result = []
        for p in parameters:
            result.append(getattr(self, p, 0.0))
        return result

    def step(self, time):
        """Compute P2H and P2G dispatch commands."""
        hour = (time % 86400) / 3600.0

        # ── 1. Renewable surplus calculation ──
        total_ren = self.wind_power + self.pv_power
        self.ren_surplus = total_ren - self.CONV_LOAD

        # ── 2. Base dispatch: follow demand ──
        p2h_target = self.p2h_demand
        p2g_target = self.p2g_demand

        # ── 3. Adjust for renewable surplus ──
        if self.ren_surplus > 0:
            # Surplus renewable → increase flexible load to absorb
            extra = self.ren_surplus
            # Split surplus: 60% to P2H (heat is easier to store), 40% to P2G
            p2h_extra = min(extra * 0.6, self.P2H_MAX - p2h_target)
            p2g_extra = min(extra * 0.4, self.P2G_MAX - p2g_target)
            p2h_target += p2h_extra
            p2g_target += p2g_extra
        else:
            # Deficit → reduce flexible load to shed demand
            deficit = abs(self.ren_surplus)
            reduction = deficit * 0.3  # reduce by 30% of deficit
            p2h_target = max(self.P2H_MIN, p2h_target - reduction * 0.5)
            p2g_target = max(self.P2G_MIN, p2g_target - reduction * 0.5)

        # ── 4. Price-based adjustment ──
        if self.elec_price < self.PRICE_LOW:
            # Cheap electricity → increase flexible loads
            price_factor = 1.0 + 0.3 * (self.PRICE_LOW - self.elec_price) / self.PRICE_LOW
            p2h_target = min(self.P2H_MAX, p2h_target * price_factor)
            p2g_target = min(self.P2G_MAX, p2g_target * price_factor)
        elif self.elec_price > self.PRICE_HIGH:
            # Expensive electricity → reduce flexible loads
            price_factor = max(0.3, 1.0 - 0.4 * (self.elec_price - self.PRICE_HIGH) / self.PRICE_HIGH)
            p2h_target *= price_factor
            p2g_target *= price_factor

        # ── 5. Voltage-based curtailment ──
        v_min_actual = min(self.bus_v_p2h, self.bus_v_p2g)
        if v_min_actual < self.V_MIN:
            # Hard curtailment — drop to minimum
            p2h_target = self.P2H_MIN
            p2g_target = self.P2G_MIN
        elif v_min_actual < self.V_LOW:
            # Proportional reduction
            v_factor = (v_min_actual - self.V_MIN) / (self.V_LOW - self.V_MIN)
            p2h_target = self.P2H_MIN + v_factor * (p2h_target - self.P2H_MIN)
            p2g_target = self.P2G_MIN + v_factor * (p2g_target - self.P2G_MIN)

        # ── 6. Clamp to limits ──
        p2h_target = max(self.P2H_MIN, min(self.P2H_MAX, p2h_target))
        p2g_target = max(self.P2G_MIN, min(self.P2G_MAX, p2g_target))

        # ── 7. Apply ramp rate limit ──
        dp2h = p2h_target - self._prev_p2h
        dp2g = p2g_target - self._prev_p2g
        if abs(dp2h) > self.RAMP_RATE:
            p2h_target = self._prev_p2h + math.copysign(self.RAMP_RATE, dp2h)
        if abs(dp2g) > self.RAMP_RATE:
            p2g_target = self._prev_p2g + math.copysign(self.RAMP_RATE, dp2g)

        # ── 8. Store outputs ──
        self.P2H_cmd = round(p2h_target, 3)
        self.P2G_cmd = round(p2g_target, 3)
        self._prev_p2h = self.P2H_cmd
        self._prev_p2g = self.P2G_cmd

    def cleanup(self):
        pass

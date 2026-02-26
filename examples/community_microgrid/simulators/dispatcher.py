# -*- coding: utf-8 -*-
"""Community MPC dispatcher — battery + EV charger control.

Model Predictive Control (MPC) dispatcher that optimises:
  1. Community battery charge/discharge schedule (P_batt_cmd)
  2. Mall EV charger power limit (P_ev_limit — demand response)

Objective  (over a 4-hour rolling horizon, 16 × 15 min steps):
  min  Σ [ price(k) · P_grid(k) · Δt                   (energy cost)
         + λ_peak · max(P_grid(k), 0)²                  (peak demand penalty)
         + λ_soc  · (SoC(N) − SoC_target)²              (terminal SoC) ]

  where  P_grid = P_h1 + P_h2 + P_mall + P_super + P_batt − P_pv
         subject to battery SoC limits and EV charger bounds.

The dispatcher also logs its forecast vs. actuals for the
post-simulation analysis script (analyze_forecasts.py).

Inputs:
    SoC        [-]     battery state of charge
    P_h1       [MW]    household 1 net load
    P_h2       [MW]    household 2 net load
    P_mall     [MW]    mall net load
    P_super    [MW]    supermarket net load
    P_pv_h1    [MW]    household 1 PV generation
    elec_price [EUR/kWh]  real-time electricity price
    solar      [W/m²]  solar irradiance (for PV forecast)

Outputs:
    P_batt_cmd [MW]    battery power command (+charge / −discharge)
    P_ev_limit [MW]    mall EV charger power cap
"""

import json
import math
from pathlib import Path

from energysim.base import SimulatorAdapter


class external_simulator(SimulatorAdapter):
    """MPC dispatcher for community battery and EV demand response."""

    # ── MPC parameters ──
    HORIZON     = 16        # steps (16 × 15 min = 4 hours)
    DT_H        = 0.25      # horizon step size [hours]

    # ── Battery parameters (must match community_battery.m) ──
    E_BATT      = 0.050     # [MWh]  50 kWh
    P_BATT_MAX  = 0.025     # [MW]   25 kW max charge/discharge
    SOC_MIN     = 0.05
    SOC_MAX     = 0.95
    SOC_TARGET  = 0.50      # target SoC at end of horizon
    ETA_RT      = 0.92      # round-trip efficiency (electrochemical)

    # ── EV charger bounds ──
    EV_P_MAX    = 0.074     # [MW]  74 kW (10 bays × 7.4 kW)
    EV_P_MIN    = 0.010     # [MW]  minimum allowed (keep 1-2 bays)

    # ── Penalty weights ──
    LAM_PEAK    = 500.0     # peak demand penalty
    LAM_SOC     = 20.0      # terminal SoC penalty
    LAM_EV      = 2.0       # EV curtailment penalty

    def __init__(self, inputs=None, outputs=None, **kwargs):
        self.inputs = inputs or []
        self.outputs = outputs or []

        # Inputs (set each step by co-sim)
        self.SoC = 0.50
        self.P_h1 = 0.0
        self.P_h2 = 0.0
        self.P_mall = 0.0
        self.P_super = 0.0
        self.P_pv_h1 = 0.0
        self.elec_price = 0.15
        self.solar = 0.0

        # Outputs
        self.P_batt_cmd = 0.0
        self.P_ev_limit = self.EV_P_MAX

        # Forecast log (for analyze_forecasts.py)
        self.forecast_log = []

    def init(self):
        pass

    def set_value(self, parameters, values):
        for p, v in zip(parameters, values):
            setattr(self, p, v)

    def get_value(self, parameters, time):
        result = []
        for p in parameters:
            if p == 'P_batt_cmd':   result.append(self.P_batt_cmd)
            elif p == 'P_ev_limit': result.append(self.P_ev_limit)
            else:                   result.append(0.0)
        return result

    def step(self, time):
        hour = (time % 86400) / 3600.0

        # ── Build forecasts over horizon ──
        price_fc = self._forecast_price(hour)
        load_fc  = self._forecast_load(hour)
        pv_fc    = self._forecast_pv(hour)

        # ── Current community net load (without battery) ──
        P_community = self.P_h1 + self.P_h2 + self.P_mall + self.P_super

        # ── Log forecast vs actual ──
        self.forecast_log.append({
            'time': time,
            'hour': round(hour, 2),
            'actual_load': round(P_community * 1000, 3),       # kW
            'actual_pv':   round(self.P_pv_h1 * 1000, 3),
            'actual_price': round(self.elec_price, 4),
            'forecast_load': [round(x * 1000, 3) for x in load_fc[:4]],
            'forecast_pv':   [round(x * 1000, 3) for x in pv_fc[:4]],
            'forecast_price': [round(x, 4) for x in price_fc[:4]],
            'SoC': round(self.SoC, 4),
        })

        # ── Solve MPC (sequential quadratic, no scipy needed) ──
        P_batt_opt, P_ev_opt = self._solve_mpc(
            self.SoC, price_fc, load_fc, pv_fc)

        self.P_batt_cmd = P_batt_opt
        self.P_ev_limit = P_ev_opt

    # ════════════════════════════════════════════════════════
    #  MPC solver (analytic / greedy with look-ahead)
    # ════════════════════════════════════════════════════════

    def _solve_mpc(self, soc, price_fc, load_fc, pv_fc):
        """Solve the rolling-horizon optimisation.

        Uses a greedy sweep with cost look-ahead rather than full
        QP to keep it dependency-free, but still captures the
        essential MPC behaviour (price arbitrage, peak shaving,
        SoC management).

        Returns (P_batt_cmd [MW], P_ev_limit [MW]).
        """
        N = min(self.HORIZON, len(price_fc))
        best_cost = float('inf')
        best_P_batt = 0.0
        best_P_ev = self.EV_P_MAX

        sqrt_eta = math.sqrt(self.ETA_RT)

        # Discretise battery power: 11 candidate levels
        P_candidates = []
        for i in range(11):
            frac = -1.0 + 2.0 * i / 10.0  # -1 to +1
            P_candidates.append(frac * self.P_BATT_MAX)

        # Discretise EV limit: 4 levels
        ev_candidates = [self.EV_P_MAX, 0.050, 0.030, self.EV_P_MIN]

        for P_b in P_candidates:
            for P_ev in ev_candidates:
                cost = self._evaluate_trajectory(
                    soc, P_b, P_ev, price_fc, load_fc, pv_fc, N, sqrt_eta)
                if cost < best_cost:
                    best_cost = cost
                    best_P_batt = P_b
                    best_P_ev = P_ev

        return best_P_batt, best_P_ev

    def _evaluate_trajectory(self, soc0, P_b0, P_ev0,
                             price_fc, load_fc, pv_fc, N, sqrt_eta):
        """Evaluate total cost of a candidate (P_batt, P_ev) action.

        Assumes the battery holds P_b0 for the first step, then
        follows a simple rule (charge when cheap, discharge when
        expensive) for the remaining horizon.
        """
        soc = soc0
        total_cost = 0.0
        peak = 0.0

        for k in range(N):
            price = price_fc[k]
            P_load = load_fc[k]
            P_pv = pv_fc[k]

            # Battery power schedule
            if k == 0:
                P_b = P_b0
            else:
                # Simple rule: charge when cheap (bottom 30 %),
                # discharge when expensive (top 30 %)
                prices_sorted = sorted(price_fc[:N])
                p30 = prices_sorted[max(0, int(0.3 * N) - 1)]
                p70 = prices_sorted[min(N - 1, int(0.7 * N))]
                if price <= p30 and soc < self.SOC_MAX - 0.05:
                    P_b = self.P_BATT_MAX * 0.8
                elif price >= p70 and soc > self.SOC_MIN + 0.05:
                    P_b = -self.P_BATT_MAX * 0.8
                else:
                    P_b = 0.0

            # SoC update
            if P_b >= 0:
                dE = P_b * sqrt_eta * self.DT_H
            else:
                dE = P_b / sqrt_eta * self.DT_H
            soc_new = soc + dE / self.E_BATT
            if soc_new > self.SOC_MAX:
                P_b = max(0, (self.SOC_MAX - soc) * self.E_BATT
                          / (sqrt_eta * self.DT_H))
                soc_new = self.SOC_MAX
            elif soc_new < self.SOC_MIN:
                P_b = min(0, (self.SOC_MIN - soc) * self.E_BATT
                          * sqrt_eta / self.DT_H)
                soc_new = self.SOC_MIN
            soc = soc_new

            # Grid power
            P_grid = P_load + P_b - P_pv

            # EVs: if EV limit is reduced, mall load drops by the difference
            if k == 0:
                ev_curtailed = max(0, self.EV_P_MAX - P_ev0)
                P_grid -= ev_curtailed * 0.5   # ~50 % utilisation

            # Cost components
            energy_cost = price * max(0.0, P_grid) * self.DT_H * 1000.0
            peak = max(peak, max(0.0, P_grid))
            total_cost += energy_cost

        # Terminal penalties
        total_cost += self.LAM_PEAK * peak * peak
        total_cost += self.LAM_SOC * (soc - self.SOC_TARGET) ** 2
        total_cost += self.LAM_EV * (self.EV_P_MAX - P_ev0) ** 2

        return total_cost

    # ════════════════════════════════════════════════════════
    #  Naive forecasters (persistence + diurnal)
    # ════════════════════════════════════════════════════════

    def _forecast_price(self, hour):
        """Price forecast: assume current price persists,
        with diurnal pattern overlay."""
        fc = []
        for k in range(self.HORIZON):
            h = (hour + k * self.DT_H) % 24.0
            # Diurnal pattern
            p = 0.15 + 0.12 * math.sin(2 * math.pi * (h - 18.0) / 24.0)
            p += 0.04 if 7.0 <= h <= 9.0 else 0.0
            # Blend with current observation
            if k == 0:
                p = self.elec_price
            else:
                p = 0.3 * self.elec_price + 0.7 * p
            fc.append(max(0.0, p))
        return fc

    def _forecast_load(self, hour):
        """Load forecast: persistence of current community load
        with diurnal modulation."""
        P_now = self.P_h1 + self.P_h2 + self.P_mall + self.P_super
        fc = []
        for k in range(self.HORIZON):
            h = (hour + k * self.DT_H) % 24.0
            # Simple diurnal weight
            if 8.0 <= h < 22.0:
                w = 1.0 + 0.2 * math.sin(math.pi * (h - 8.0) / 14.0)
            else:
                w = 0.6
            if k == 0:
                fc.append(P_now)
            else:
                fc.append(max(0.0, P_now * w))
        return fc

    def _forecast_pv(self, hour):
        """PV forecast: bell-shaped solar curve scaled to current."""
        fc = []
        for k in range(self.HORIZON):
            h = (hour + k * self.DT_H) % 24.0
            if 6.0 <= h <= 18.0:
                solar_est = 850.0 * math.sin(math.pi * (h - 6.0) / 12.0)
            else:
                solar_est = 0.0
            # Scale using current solar as reference
            if self.solar > 10.0 and solar_est > 10.0:
                # Clearness index
                ki = min(2.0, self.solar / max(1.0,
                         850.0 * math.sin(math.pi * max(0.01,
                         (hour - 6.0)) / 12.0)))
                solar_fc = solar_est * ki
            else:
                solar_fc = solar_est
            # Convert to PV power [MW] — 5 kWp, 85 % eff
            pv_mw = 5.0 * (solar_fc / 1000.0) * 0.85 / 1000.0
            fc.append(max(0.0, pv_mw))
        return fc

    def cleanup(self):
        """Save forecast log for analysis script."""
        log_path = Path(__file__).resolve().parent.parent / "forecast_log.json"
        try:
            with open(log_path, 'w') as f:
                json.dump(self.forecast_log, f, indent=2)
        except Exception:
            pass

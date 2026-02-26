# -*- coding: utf-8 -*-
"""Model Predictive Controller — external simulator for energysim.

Implements a receding-horizon MPC that jointly optimises battery and
heat-pump power to **minimise electricity cost** while maintaining
indoor comfort.

Optimisation problem (at each macro step):
    min  Σ_k  price_k · P_grid_k · Δt
    s.t.
        P_grid_k  = P_load + P_hp_k + P_batt_k − P_pv_k
        SoC_{k+1} = SoC_k + η·P_batt_k·Δt / E_cap      (charge)
        T_{k+1}   = T_k + (COP·P_hp_k + Q_solar − UA·(T_k − T_amb)) / C · Δt
        T_min  ≤ T_k ≤ T_max
        SoC_min ≤ SoC_k ≤ SoC_max
        0      ≤ P_hp_k ≤ P_hp_max
        −P_max ≤ P_batt_k ≤ P_max

The controller solves this with ``scipy.optimize.minimize`` (SLSQP) at
each macro time-step and applies only the first action (receding
horizon).

Inputs  (from other simulators):
    SoC, T_inside, T_ambient, solar, P_pv, P_load, elec_price, T_setpoint
Outputs (to actuators):
    P_batt_cmd, P_hp_cmd
"""

import math
import numpy as np
from energysim.base import SimulatorAdapter

try:
    from scipy.optimize import minimize
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class external_simulator(SimulatorAdapter):
    """Receding-horizon MPC for battery + heat-pump dispatch."""

    # --- Horizon & timing ---
    HORIZON_STEPS = 16     # 16 × 900 s = 4 hours ahead
    DT = 900.0             # macro time-step [s]

    # --- Battery parameters (must match battery_mpc.m) ---
    E_CAP   = 0.0135       # MWh
    P_MAX   = 0.005        # MW
    ETA     = 0.95
    SOC_MIN = 0.10
    SOC_MAX = 0.95

    # --- Heat-pump parameters ---
    P_HP_MAX = 0.003       # MW (3 kW electrical)
    COP_NOM  = 3.5

    # --- Building parameters (must match building.py) ---
    C  = 2500.0            # kJ/K
    UA = 0.080             # kW/K
    gA = 4.0e-3            # kW per W/m²

    # --- Comfort bounds ---
    T_MIN = 19.0
    T_MAX = 23.0

    def __init__(self, inputs=None, outputs=None, **kwargs):
        self.inputs = inputs or []
        self.outputs = outputs or []

        # --- Current measurements ---
        self.SoC = 0.5
        self.T_inside = 20.0
        self.T_ambient = 5.0
        self.solar = 0.0
        self.P_pv = 0.0
        self.P_load = 0.002
        self.elec_price = 0.15
        self.T_setpoint = 21.0    # from signal

        # --- Control outputs ---
        self.P_batt_cmd = 0.0
        self.P_hp_cmd = 0.0

    def init(self):
        if not _HAS_SCIPY:
            print("[MPC] WARNING: scipy not found — falling back to "
                  "rule-based dispatch.")

    # ─────────────────────────────────────────────────────────────
    # SimulatorAdapter interface
    # ─────────────────────────────────────────────────────────────

    def set_value(self, parameters, values):
        for p, v in zip(parameters, values):
            if p == 'SoC':          self.SoC = v
            elif p == 'T_inside':   self.T_inside = v
            elif p == 'T_ambient':  self.T_ambient = v
            elif p == 'solar':      self.solar = v
            elif p == 'P_pv':       self.P_pv = v
            elif p == 'P_load':     self.P_load = v
            elif p == 'elec_price': self.elec_price = v
            elif p == 'T_setpoint': self.T_setpoint = v

    def get_value(self, parameters, time):
        result = []
        for p in parameters:
            if p == 'P_batt_cmd':   result.append(self.P_batt_cmd)
            elif p == 'P_hp_cmd':   result.append(self.P_hp_cmd)
            else:                   result.append(0.0)
        return result

    def step(self, time):
        if _HAS_SCIPY:
            self._solve_mpc()
        else:
            self._rule_based()

    def cleanup(self):
        pass

    # ─────────────────────────────────────────────────────────────
    # MPC solver
    # ─────────────────────────────────────────────────────────────

    def _solve_mpc(self):
        """Solve the finite-horizon optimisation using SLSQP."""
        N = self.HORIZON_STEPS
        dt_h = self.DT / 3600.0            # hours
        sqrt_eta = math.sqrt(self.ETA)

        # Assume constant disturbances over the horizon (persistence
        # forecast — the simplest practical approach).
        price    = self.elec_price
        T_amb    = self.T_ambient
        solar    = self.solar
        P_pv     = self.P_pv
        P_load   = self.P_load
        T_set    = self.T_setpoint

        # Decision vector: [P_batt_0 .. P_batt_{N-1},
        #                    P_hp_0   .. P_hp_{N-1}]
        n_vars = 2 * N

        # --- Objective: minimise grid import cost + comfort penalty ---
        def objective(x):
            P_batt = x[:N]
            P_hp   = x[N:]
            cost = 0.0
            for k in range(N):
                P_grid = P_load + P_hp[k] + P_batt[k] - P_pv
                # Grid cost  (only pay for import)
                cost += price * max(0.0, P_grid) * dt_h * 1000.0  # EUR
                # Soft comfort penalty  (penalise deviation from setpoint)
                # — evaluated inside constraints too, but helps gradient
            return cost

        # --- Constraints: SoC & temperature trajectories ---
        def constraint_soc_and_temp(x):
            P_batt = x[:N]
            P_hp   = x[N:]
            ineqs = []
            soc = self.SoC
            T   = self.T_inside
            dt_s = self.DT
            for k in range(N):
                # SoC dynamics
                if P_batt[k] >= 0:
                    dE = P_batt[k] * sqrt_eta * dt_h
                else:
                    dE = P_batt[k] / sqrt_eta * dt_h
                soc = soc + dE / self.E_CAP

                # SoC bounds  (ineq >= 0)
                ineqs.append(soc - self.SOC_MIN)
                ineqs.append(self.SOC_MAX - soc)

                # Temperature dynamics
                Q_hp    = P_hp[k] * 1000.0 * self.COP_NOM      # kW
                Q_solar = self.gA * solar                       # kW
                Q_loss  = self.UA * (T - T_amb)                 # kW
                dTdt    = (Q_hp + Q_solar - Q_loss) / self.C    # K/s
                T       = T + dTdt * dt_s

                # Comfort bounds  (ineq >= 0)
                ineqs.append(T - self.T_MIN)
                ineqs.append(self.T_MAX - T)

            return np.array(ineqs)

        # --- Bounds ---
        bounds = (
            [(-self.P_MAX, self.P_MAX)] * N +     # P_batt
            [(0.0, self.P_HP_MAX)] * N             # P_hp
        )

        # --- Initial guess: idle battery, mid-range HP ---
        x0 = np.zeros(n_vars)
        x0[N:] = self.P_HP_MAX * 0.3   # start with moderate heating

        try:
            res = minimize(
                objective, x0, method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq',
                             'fun': constraint_soc_and_temp},
                options={'maxiter': 200, 'ftol': 1e-8, 'disp': False},
            )
            if res.success:
                self.P_batt_cmd = float(res.x[0])
                self.P_hp_cmd   = float(res.x[N])
            else:
                # Optimiser didn't converge — use first feasible guess
                self._rule_based()
        except Exception:
            self._rule_based()

    # ─────────────────────────────────────────────────────────────
    # Fall-back rule-based dispatch
    # ─────────────────────────────────────────────────────────────

    def _rule_based(self):
        """Simple heuristic used when scipy is unavailable or MPC fails."""
        # Battery: charge when cheap / PV surplus, discharge when expensive
        if self.elec_price < 0.12 and self.SoC < 0.85:
            self.P_batt_cmd = self.P_MAX * 0.8      # charge
        elif self.elec_price > 0.20 and self.SoC > 0.20:
            self.P_batt_cmd = -self.P_MAX * 0.8     # discharge
        elif self.P_pv > self.P_load and self.SoC < 0.90:
            self.P_batt_cmd = min(self.P_MAX, self.P_pv - self.P_load)
        else:
            self.P_batt_cmd = 0.0

        # Heat pump: proportional control towards setpoint
        T_err = self.T_setpoint - self.T_inside
        if T_err > 0:
            self.P_hp_cmd = min(self.P_HP_MAX,
                                self.P_HP_MAX * T_err / 3.0)
        else:
            self.P_hp_cmd = 0.0

# --- Compact heat-pump + 1R1C building fleet (shared by notebooks 12 and 14) -------------------
import numpy as np
from energysim.container import VectorizedContainerAdapter
from energysim.models.common import Curve1D

#: Illustrative COP(T_amb) for an air-source heat pump at 35 C flow (replace with test data).
HP_COP_MILD = Curve1D([-15, -7, 2, 7, 12], [2.2, 2.6, 3.2, 3.7, 4.2], name="COP_mild")


class HeatPumpFleet(VectorizedContainerAdapter):
    """N heat-pump homes: 1R1C envelope, thermostat with hysteresis, COP(T_amb) lookup.

    Per home i (explicit Euler, step h):

        C_i dT_i/dt = Q_heat_i - (T_i - T_amb) / R_i
        on_i  : switches on below  T_set - db/2, off above T_set + db/2 (hysteresis)
        Q_heat_i = on_i * Qmax_i * dr_limit                      [kW heat]
        P_el_i   = Q_heat_i / COP(T_amb)                         [kW electric]

    ``dr_limit`` (0..1) is a demand-response input that caps the compressor capacity; a
    value of 0 switches the fleet off (homes cool down through their envelope).

    Parameters are drawn per home from uniform ranges with a fixed seed, so runs are
    deterministic.  Setting ``q_heat_max_kw`` to 0 for a home removes its heat pump
    (used for the penetration sweeps).

    Inputs : t_ambient_degc [degC], dr_limit [1]
    States : t_indoor [degC], on [0/1], p_kw [kW], q_heat_kw [kW], cop [1]
    Totals : total_p_kw, total_q_heat_kw, mean_t_indoor, min_t_indoor (via the container)
    """

    units = {
        "t_ambient_degc": "degC", "dr_limit": "1",
        "t_indoor": "degC", "on": "1", "p_kw": "kW", "q_heat_kw": "kW", "cop": "1",
    }

    def __init__(self, name="hp_fleet", size=100, cop_curve=HP_COP_MILD, setpoint_degc=20.0,
                 deadband_k=1.0, q_heat_max_kw=None, rng_seed=0, t_init_degc=20.0,
                 step_size=900.0, **kwargs):
        super().__init__(name=name, size=size, step_size=step_size)
        rng = np.random.default_rng(rng_seed)
        self.c_th = rng.uniform(8.0, 16.0, self.size)                  # kWh/K
        self.r_th = rng.uniform(6.0, 10.0, self.size)                  # K/kW
        qmax = rng.uniform(4.0, 7.0, self.size) if q_heat_max_kw is None else \
            self._as_vector(q_heat_max_kw, "q_heat_max_kw")
        self.q_max = np.asarray(qmax, dtype=float)                     # kW heat
        self.cop_curve = cop_curve
        self.setpoint = float(setpoint_degc)
        self.deadband = float(deadband_k)

        t0 = rng.normal(t_init_degc, 0.5, self.size)
        self.register_state("t_indoor", t0)
        self.register_state("on", np.ones(self.size))
        self.register_state("p_kw", 0.0)
        self.register_state("q_heat_kw", 0.0)
        self.register_state("cop", 3.0)
        self.register_input("t_ambient_degc", 0.0)
        self.register_input("dr_limit", 1.0)

    def step(self, time, step_size=None):
        dt_s = float(step_size) if step_size is not None else self.step_size
        if dt_s <= 0.0:
            return
        h = dt_s / 3600.0
        t_in = self.states["t_indoor"]
        t_amb = self.input_values["t_ambient_degc"]
        dr = np.clip(self.input_values["dr_limit"], 0.0, 1.0)

        half = 0.5 * self.deadband
        on = np.where(t_in < self.setpoint - half, 1.0,
                      np.where(t_in > self.setpoint + half, 0.0, self.states["on"]))
        q_heat = on * self.q_max * dr
        cop = np.maximum(np.asarray(self.cop_curve(t_amb), dtype=float), 1.0)
        q_loss = (t_in - t_amb) / self.r_th

        self.states["t_indoor"][:] = t_in + (q_heat - q_loss) / self.c_th * h
        self.states["on"][:] = on
        self.states["q_heat_kw"][:] = q_heat
        self.states["p_kw"][:] = q_heat / cop
        self.states["cop"][:] = cop

# --- Compact smart-PV fleet with volt-watt response (notebook 12) -----------------------------
import numpy as np
from energysim.container import VectorizedContainerAdapter
from energysim.models.common import Curve1D

#: Illustrative volt-watt curve (pu bus voltage -> fraction of available PV power).
#: Full output up to 1.03 pu, linear curtailment to 60 % at 1.06 pu and to 0 at 1.10 pu.
VOLT_WATT_CURVE = Curve1D([0.90, 1.03, 1.06, 1.10], [1.0, 1.0, 0.6, 0.0], name="volt_watt")


class SmartPVFleet(VectorizedContainerAdapter):
    """N rooftop PV systems with an optional volt-watt (Volt-Watt, IEEE 1547-style) function.

    Available power (per system), as in ``PVFleetAdapter``:

        G = max(0, irradiance);  T_cell = T_amb + G/800 * 25
        P_av = P_peak * G/1000 * [1 + gamma (T_cell - 25)] * PR

    Volt-watt: P = P_av * f(v), where v is the *closed-loop* bus voltage (pu) that the
    pandapower adapter reports one macro step earlier, and f is ``VOLT_WATT_CURVE``.
    The applied factor follows the target with a first-order response of time constant
    ``response_s`` (aggregate inverter response). Without it, a hard switch with a one-step
    measurement delay makes the loop chatter between 0 and full output; the response time
    damps that limit cycle.

    Inputs : irradiance_w_m2 [W/m2], t_ambient_degc [degC], v_pu [pu]
    States : p_kw [kW], curtail_factor [1]
    """

    units = {
        "irradiance_w_m2": "W/m2", "t_ambient_degc": "degC", "v_pu": "pu",
        "p_kw": "kW", "curtail_factor": "1",
    }

    def __init__(self, name="pv_fleet", size=100, peak_kw=4.0, volt_watt=True,
                 vw_curve=VOLT_WATT_CURVE, response_s=1800.0, performance_ratio=0.85, gamma=-0.004,
                 step_size=900.0, **kwargs):
        super().__init__(name=name, size=size, step_size=step_size)
        self.peak_kw = self._as_vector(peak_kw, "peak_kw")
        self.volt_watt = bool(volt_watt)
        self.vw_curve = vw_curve
        self.response_s = float(response_s)
        self.pr = float(performance_ratio)
        self.gamma = float(gamma)

        self.register_state("p_kw", 0.0)
        self.register_state("curtail_factor", 1.0)
        self.register_input("irradiance_w_m2", 0.0)
        self.register_input("t_ambient_degc", 25.0)
        self.register_input("v_pu", 1.0)

    def step(self, time, step_size=None):
        dt_s = float(step_size) if step_size is not None else self.step_size
        g = np.maximum(0.0, self.input_values["irradiance_w_m2"])
        t_cell = self.input_values["t_ambient_degc"] + (g / 800.0) * 25.0
        p_av = self.peak_kw * (g / 1000.0) * (1.0 + self.gamma * (t_cell - 25.0)) * self.pr
        p_av = np.maximum(0.0, p_av)
        if self.volt_watt:
            target = np.asarray(self.vw_curve(self.input_values["v_pu"]), dtype=float)
        else:
            target = np.ones(self.size)
        if self.response_s > 0.0 and dt_s > 0.0:
            alpha = 1.0 - np.exp(-dt_s / self.response_s)
            f = self.states["curtail_factor"] + alpha * (target - self.states["curtail_factor"])
        else:
            f = target
        self.states["curtail_factor"][:] = f
        self.states["p_kw"][:] = p_av * f

# -*- coding: utf-8 -*-
"""District-heat network-lite: pipe transport delay and a simple supply temperature."""

from __future__ import annotations

import copy
from typing import Any, Dict, List

from .base import EnergyNetwork, _Result, _num

__all__ = ["HeatNetwork"]


class HeatNetwork(EnergyNetwork):
    """
    District-heat network.

    Adds to :class:`EnergyNetwork`:

    * **Transport delay.**  Energy delivered to a consumer is delayed by the
      flow-weighted mean of the ``delay_s`` of the pipes it travelled through
      (FIFO buffer per consumer, zero-order hold).  Producers dispatch for the
      *current* demand, so a demand step is only fully served after the
      delay (``unserved_MW`` is non-zero in between); energy arriving beyond
      the current demand is reported as a loss.  The first solved state is
      assumed to fill the pipes (no start-up transient).  Energy in transit is
      tracked in ``transit_MWh``.  Storage charging is not delayed.
    * **Temperatures.**  ``<consumer>.supply_T_C = supply_T_C - (supply_T_C -
      return_T_C) * loss_fraction`` where the loss fraction is the flow-weighted
      fraction of energy lost on the way to that consumer; ``mean_supply_T`` is
      the delivered-weighted mean (nominal when nothing is delivered).

    Parameters
    ----------
    supply_T_C, return_T_C : float
        Nominal supply and return temperatures [degC].
    """

    carrier = "heat"
    flow_unit = "MW"

    def __init__(self, name: str = "heat", step_size: float = 1.0,
                 supply_T_C: float = 80.0, return_T_C: float = 45.0, **kwargs):
        super().__init__(name=name, step_size=step_size)
        self.supply_T_C = _num("supply_T_C", supply_T_C)
        self.return_T_C = _num("return_T_C", return_T_C)
        self.register_parameters("supply_T_C", "return_T_C")
        self._clock = 0.0
        self._transit_MWh = 0.0
        self._dq: Dict[str, List[List[float]]] = {}
        self.register_variables(outputs=["mean_supply_T", "transit_MWh"])
        self.units.update({"mean_supply_T": "degC", "transit_MWh": "MWh"})

    def _params_dict(self) -> Dict[str, Any]:
        return {"supply_T_C": self.supply_T_C, "return_T_C": self.return_T_C}

    def add_consumer(self, name: str, demand: float = 0.0) -> "HeatNetwork":
        super().add_consumer(name, demand)
        self._reg(name, outputs=("supply_T_C",), units={"supply_T_C": "degC"})
        self._outputs_val[f"{name}.supply_T_C"] = self.supply_T_C
        return self

    def _finish_step(self, dt, h, res: _Result, ctx) -> Dict[str, Any]:
        live = dt > 0
        dq = self._dq if live else copy.deepcopy(self._dq)
        tau, surv = self._propagate(res, ctx)
        clock = self._clock
        delivered: List[float] = []
        overshoot = 0.0
        arrived_total = 0.0
        lp_total = 0.0
        temps: List[float] = []
        weights: List[float] = []
        for j, n in enumerate(self._consumers):
            y = res.dlv[j]
            lp_total += y
            q = dq.setdefault(n, [])
            q.append([clock + tau[n], y])
            q.sort(key=lambda e: e[0])
            arrived = [e for e in q if e[0] <= clock + 1e-9]
            if arrived:
                a = arrived[-1][1]
                del q[:len(arrived) - 1]
            else:
                a = q[0][1]
            got = min(a, ctx["demands"][j])
            delivered.append(got)
            overshoot += a - got
            arrived_total += a
            frac = 1.0 - surv[n]
            temps.append(self.supply_T_C - (self.supply_T_C - self.return_T_C) * frac)
            self._outputs_val[f"{n}.supply_T_C"] = temps[-1]
            weights.append(got)
        wsum = sum(weights)
        self._outputs_val["mean_supply_T"] = (
            sum(t * w for t, w in zip(temps, weights)) / wsum if wsum > 1e-12 else self.supply_T_C)
        if live:
            self._clock += dt
        return {"delivered": delivered, "overshoot": overshoot,
                "dtransit": lp_total - arrived_total}

    def _apply_extra_state(self, dt, h, res, ctx) -> None:
        if dt > 0:
            self._transit_MWh += self.balance["dtransit_MW"] * h
        self._outputs_val["transit_MWh"] = self._transit_MWh

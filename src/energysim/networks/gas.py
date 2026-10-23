# -*- coding: utf-8 -*-
"""Gas network-lite: unit conversion and pipe linepack as a pressure-banded buffer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import EnergyNetwork, NetworkError, _num

__all__ = ["GasNetwork"]


class GasNetwork(EnergyNetwork):
    """
    Gas network.

    Adds to :class:`EnergyNetwork`:

    * **Units.**  ``unit='MW_LHV'`` (default) or ``'Nm3/h'``.  Run-time
      variables named ``*_MW`` (demands, availabilities, commands, flows,
      dispatch, totals) are exchanged in this unit; ``Nm3/h = MW * 1000 /
      lhv_kwh_per_nm3``.  Builder arguments (capacities, ``p_max``, demands
      at construction) are always in MW (LHV) and MWh.
    * **Linepack.**  ``add_pipe(..., linepack_capacity_mwh=X)`` gives the pipe
      a buffer at its outlet node (state ``<pipe>.linepack_MWh``, initial fill
      ``linepack_init``).  It is used only as a last resort (dearer than any
      producer): when producers or pipes cannot cover demand, linepack is
      withdrawn; when there is spare capacity it is refilled up to its
      initial fill.  Withdrawals beyond the available linepack are unserved.
      ``<pipe>.pressure_bar`` is linear between ``p_min_bar`` (empty) and
      ``p_max_bar`` (full); pipes without linepack report ``p_max_bar``.
      The linepack is a buffer, not hydraulics: it injects at the outlet node
      and does not relax the pipe's own capacity constraint.

    Parameters
    ----------
    unit : {'MW_LHV', 'Nm3/h'}
    lhv_kwh_per_nm3 : float
        Lower heating value used for the Nm3/h conversion.
    p_min_bar, p_max_bar : float
        Default pressure band (pipes may override in ``add_pipe``).
    """

    carrier = "gas"
    flow_unit = "MW_LHV"

    def __init__(self, name: str = "gas", step_size: float = 1.0, unit: str = "MW_LHV",
                 lhv_kwh_per_nm3: float = 10.0, p_min_bar: float = 30.0,
                 p_max_bar: float = 70.0, **kwargs):
        if unit not in ("MW_LHV", "Nm3/h"):
            raise NetworkError(f"GasNetwork unit must be 'MW_LHV' or 'Nm3/h', got {unit!r}.")
        self.unit = unit
        self.lhv_kwh_per_nm3 = _num("lhv_kwh_per_nm3", lhv_kwh_per_nm3, lo=0.0, strict_lo=True)
        super().__init__(name=name, step_size=step_size)
        self.p_min_bar = _num("p_min_bar", p_min_bar, lo=0.0)
        self.p_max_bar = _num("p_max_bar", p_max_bar, lo=0.0)
        if self.p_max_bar < self.p_min_bar:
            raise NetworkError("p_max_bar must be >= p_min_bar.")
        self.register_parameters("lhv_kwh_per_nm3", "p_min_bar", "p_max_bar")
        self._lp_cfg: Dict[str, Dict[str, float]] = {}
        self._linepack: Dict[str, float] = {}      # pipe name -> MWh (state)
        self.register_variables(outputs=["total_linepack_MWh"])
        self.units["total_linepack_MWh"] = "MWh"

    def _fu(self) -> str:
        return self.unit

    def _ext_per_mw(self) -> float:
        if self.unit == "Nm3/h":
            return 1000.0 / self.lhv_kwh_per_nm3
        return 1.0

    def _params_dict(self) -> Dict[str, Any]:
        return {"unit": self.unit, "lhv_kwh_per_nm3": self.lhv_kwh_per_nm3,
                "p_min_bar": self.p_min_bar, "p_max_bar": self.p_max_bar}

    def add_pipe(self, a: str, b: str, capacity: float, loss_pct: float = 0.0,
                 length_km: Optional[float] = None, delay_s: float = 0.0,
                 name: Optional[str] = None, linepack_capacity_mwh: float = 0.0,
                 linepack_init: float = 0.5, p_min_bar: Optional[float] = None,
                 p_max_bar: Optional[float] = None) -> str:
        cap_lp = _num("linepack_capacity_mwh", linepack_capacity_mwh, lo=0.0)
        init = _num("linepack_init", linepack_init, lo=0.0)
        if init > 1.0:
            raise NetworkError("linepack_init must be in [0, 1].")
        pname = super().add_pipe(a, b, capacity, loss_pct, length_km, delay_s, name)
        self._pipes[-1].update(linepack_capacity_mwh=cap_lp, linepack_init=init,
                               p_min_bar=p_min_bar, p_max_bar=p_max_bar)
        self._lp_cfg[pname] = {"cap": cap_lp, "target": init * cap_lp}
        self._linepack[pname] = init * cap_lp
        self._reg(pname, outputs=("linepack_MWh", "pressure_bar"),
                  units={"linepack_MWh": "MWh", "pressure_bar": "bar"})
        return pname

    def _pressure(self, pipe: Dict[str, Any], energy: float) -> float:
        lo = self.p_min_bar if pipe.get("p_min_bar") is None else pipe["p_min_bar"]
        hi = self.p_max_bar if pipe.get("p_max_bar") is None else pipe["p_max_bar"]
        cap = pipe.get("linepack_capacity_mwh", 0.0)
        if cap <= 0:
            return hi
        return lo + (hi - lo) * min(max(energy / cap, 0.0), 1.0)

    def _extra_buffers(self, h: float) -> List[Dict[str, Any]]:
        bufs = []
        for p in self._pipes:
            cfg = self._lp_cfg[p["name"]]
            if cfg["cap"] <= 0:
                continue
            e = self._linepack[p["name"]]
            if h > 0:
                dis = min(p["capacity"], e / h)
                chg = min(p["capacity"], max(cfg["target"] - e, 0.0) / h)
            else:
                dis = p["capacity"] if e > 1e-12 else 0.0
                chg = p["capacity"] if cfg["target"] - e > 1e-12 else 0.0
            bufs.append({"kind": "linepack", "key": p["name"], "node": p["b"],
                         "dis_cap": dis, "chg_cap": chg})
        return bufs

    def _apply_extra_state(self, dt, h, res, ctx) -> None:
        dis = {s["key"]: res.inj[i] for i, s in enumerate(ctx["sources"]) if s["kind"] == "linepack"}
        chg = {t["key"]: res.dlv[j] for j, t in enumerate(ctx["sinks"]) if t["kind"] == "linepack"}
        total = 0.0
        for p in self._pipes:
            n = p["name"]
            e = self._linepack[n]
            if n in dis:
                e1 = min(max(e + (chg[n] - dis[n]) * h, 0.0), self._lp_cfg[n]["cap"])
                if dt > 0:
                    self._linepack[n] = e1
                e = e1
            self._outputs_val[f"{n}.linepack_MWh"] = e
            self._outputs_val[f"{n}.pressure_bar"] = self._pressure(p, e)
            total += e
        self._outputs_val["total_linepack_MWh"] = total

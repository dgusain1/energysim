# -*- coding: utf-8 -*-
"""
Optimal power flow as a controller (pandapower ``runopp`` / ``rundcopp``).

The OPF used to be a power-flow *mode* of the world (``pf='opf'``); in
energysim 3.0 it is a controller: it owns (a copy of) a pandapower network,
receives limits / profiles / costs from other simulators as inputs, solves
the OPF at every macro step and outputs the optimal set-points, which are
connected to the plant (e.g. a ``powerflow`` simulator running ``runpp`` on
the real network, or device models).

Variable names follow the pandapower adapter: ``"<element name>.<column>"``.

* inputs: any column of the element table (``WF.p_mw``, ``P2G.max_p_mw``,
  ``P2G.controllable``...) or a cost coefficient (``P2G.cp1_eur_per_mw``,
  ``cp0_eur``, ``cp2_eur_per_mw2``) of the element's ``poly_cost`` row;
* outputs: result columns (``P2G.p_mw`` from ``res_load``, ``Bus 1.vm_pu``,
  ``grid.p_mw``...) plus ``cost`` (``net.res_cost``) and
  ``max_line_loading_percent`` / ``max_trafo_loading_percent``.
"""

from __future__ import annotations

import copy as _copy
import math
import time as _time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from ..base import SimulatorVariableError
from .base import Controller, FALLBACK_NONE, FALLBACK_PREVIOUS_PLAN
from .lp import STATUS_ERROR, STATUS_INFEASIBLE, STATUS_OPTIMAL

_ELEMENT_KINDS = ("gen", "sgen", "load", "ext_grid", "storage", "bus")
_COST_COLUMNS = ("cp0_eur", "cp1_eur_per_mw", "cp2_eur_per_mw2")
_SPECIAL_OUTPUTS = ("cost", "max_line_loading_percent", "max_trafo_loading_percent")


def _require_pandapower():
    try:
        import pandapower as pp
    except ImportError as exc:
        raise ImportError(
            "OPFController needs pandapower. Install it with: pip install pandapower "
            "(or pip install energysim[powerflow])"
        ) from exc
    return pp


class OPFController(Controller):
    """pandapower OPF wrapped as an energysim controller.

    Parameters
    ----------
    network : pandapowerNet, pp_adapter, or path to a pickle / json network
        The network model (deep-copied unless ``copy=False``).
    inputs : list[str]
        ``"element.column"`` names set on the network before every OPF.
    outputs : list[str]
        ``"element.column"`` result names (and the special outputs ``cost``,
        ``max_line_loading_percent``, ``max_trafo_loading_percent``).
    controllable : iterable of element names, optional
        Elements whose ``controllable`` flag is set to True.
    costs : dict, optional
        ``{element: cp1_eur_per_mw}`` or ``{element: {cp0_eur:..., cp1_eur_per_mw:..., ...}}``;
        missing ``poly_cost`` rows are created with ``pp.create_poly_cost``.
    mode : 'ac' | 'dc'
        ``runopp`` or ``rundcopp``.
    opf_kwargs : dict, optional
        Extra keyword arguments for the pandapower OPF call.

    On non-convergence the previous set-points are kept (``fallback = 1``) and
    ``solver_status`` is 2 (infeasible / not converged) or 4 (other error).
    """

    def __init__(self, network: Any, inputs: Sequence[str] = (), outputs: Sequence[str] = (),
                 controllable: Optional[Iterable[str]] = None,
                 costs: Optional[Mapping[str, Union[float, Mapping[str, float]]]] = None,
                 mode: str = "ac", copy: bool = True, opf_kwargs: Optional[Mapping[str, Any]] = None,
                 name: str = "opf", **kwargs):
        pp = _require_pandapower()
        self._pp = pp
        net = getattr(network, "network", network)
        if isinstance(net, str):
            net = pp.from_json(net) if net.endswith(".json") else pp.from_pickle(net)
        if not hasattr(net, "bus"):
            raise TypeError(f"OPFController needs a pandapower network, got {type(network).__name__}.")
        self.network = _copy.deepcopy(net) if copy else net
        mode = str(mode).lower()
        if mode not in ("ac", "dc"):
            raise ValueError("mode must be 'ac' or 'dc'.")
        self.mode = mode
        self.opf_kwargs = dict(opf_kwargs or {})
        self.opf_kwargs.setdefault("numba", False)

        self._index = self._build_index()
        for el in controllable or ():
            kind, idx = self._locate(el)
            if kind == "bus":
                raise SimulatorVariableError(f"Bus '{el}' cannot be controllable.")
            getattr(self.network, kind).at[idx, "controllable"] = True
        for el, spec in (costs or {}).items():
            spec = {"cp1_eur_per_mw": float(spec)} if not isinstance(spec, Mapping) else dict(spec)
            for col, val in spec.items():
                self._set_cost(el, col, float(val))

        inputs, outputs = list(inputs), list(outputs)
        for p in inputs:
            self._check_input(p)
        for p in outputs:
            self._check_output(p)
        initial = {p: self._read_input(p) for p in inputs}
        initial = {k: v for k, v in initial.items() if isinstance(v, (int, float)) and math.isfinite(v)}
        self._network0 = _copy.deepcopy(self.network)
        kwargs.setdefault("decide_at_init", True)
        super().__init__(name=name, measurement_inputs=inputs, setpoint_outputs=outputs,
                         initial_measurements=initial, actuation_delay=False, **kwargs)

    def reset_controller(self) -> None:
        # Every run starts from the network as configured at construction.
        self.network = _copy.deepcopy(self._network0)
        self.last_error = None

    # ------------------------------------------------------------------
    @classmethod
    def from_world(cls, world, sim_name: str, **kwargs) -> "OPFController":
        """Create the controller from the network of a registered ``powerflow`` simulator."""
        return cls(world.get_model(sim_name), **kwargs)

    # ------------------------------------------------------------------
    def _build_index(self) -> Dict[str, Tuple[str, int]]:
        index: Dict[str, Tuple[str, int]] = {}
        for kind in _ELEMENT_KINDS:
            df = getattr(self.network, kind, None)
            if df is None or "name" not in df:
                continue
            for idx, nm in df["name"].items():
                if nm is None or (isinstance(nm, float) and math.isnan(nm)) or nm == "":
                    continue
                index.setdefault(str(nm), (kind, idx))
        return index

    def _locate(self, element: str) -> Tuple[str, int]:
        if element not in self._index:
            raise SimulatorVariableError(
                f"{getattr(self, 'name', 'opf')}: element '{element}' not found in the network "
                f"(named elements: {sorted(self._index)[:20]})."
            )
        return self._index[element]

    @staticmethod
    def _split(p: str) -> Tuple[str, str]:
        if "." not in p:
            raise SimulatorVariableError(f"OPF variable '{p}' must be 'element.column'.")
        el, col = p.rsplit(".", 1)
        return el, col

    def _cost_row(self, kind: str, idx: int):
        pc = self.network.poly_cost
        m = pc[(pc["et"] == kind) & (pc["element"] == idx)].index
        return m[0] if len(m) else None

    def _set_cost(self, element: str, col: str, value: float) -> None:
        kind, idx = self._locate(element)
        if kind == "bus":
            raise SimulatorVariableError(f"Bus '{element}' cannot have cost parameters.")
        if col not in _COST_COLUMNS:
            raise SimulatorVariableError(f"Unknown cost column '{col}', use one of {_COST_COLUMNS}.")
        row = self._cost_row(kind, idx)
        if row is None:
            self._pp.create_poly_cost(self.network, idx, kind, **{col: value})
        else:
            self.network.poly_cost.at[row, col] = value

    def _check_input(self, p: str) -> None:
        el, col = self._split(p)
        kind, _ = self._locate(el)
        if col in _COST_COLUMNS:
            if kind == "bus":
                raise SimulatorVariableError(f"Bus '{el}' cannot have cost parameters.")
            return
        if col not in getattr(self.network, kind).columns:
            raise SimulatorVariableError(f"OPF input '{p}': '{col}' is not a column of net.{kind}.")

    def _check_output(self, p: str) -> None:
        if p in _SPECIAL_OUTPUTS:
            return
        el, col = self._split(p)
        kind, _ = self._locate(el)
        res = getattr(self.network, f"res_{kind}", None)
        if not ((res is not None and col in res.columns) or col in getattr(self.network, kind).columns):
            raise SimulatorVariableError(f"OPF output '{p}': '{col}' is neither in res_{kind} nor in {kind}.")

    def _read_input(self, p: str) -> Any:
        el, col = self._split(p)
        kind, idx = self._locate(el)
        if col in _COST_COLUMNS:
            row = self._cost_row(kind, idx)
            return float(self.network.poly_cost.at[row, col]) if row is not None else 0.0
        val = getattr(self.network, kind).at[idx, col]
        try:
            return float(val)
        except (TypeError, ValueError):
            return val

    def _write_input(self, p: str, value: float) -> None:
        el, col = self._split(p)
        kind, idx = self._locate(el)
        if col in _COST_COLUMNS:
            self._set_cost(el, col, value)
            return
        table = getattr(self.network, kind)
        if table[col].dtype == bool:
            value = bool(value)
        table.at[idx, col] = value

    def _read_output(self, p: str) -> float:
        net = self.network
        if p == "cost":
            return float(getattr(net, "res_cost", float("nan")))
        if p in ("max_line_loading_percent", "max_trafo_loading_percent"):
            res = net.res_line if p.startswith("max_line") else net.res_trafo
            return float(res["loading_percent"].max()) if len(res) else 0.0
        el, col = self._split(p)
        kind, idx = self._locate(el)
        res = getattr(net, f"res_{kind}", None)
        if res is not None and col in res.columns and idx in res.index:
            return float(res.at[idx, col])
        return float(getattr(net, kind).at[idx, col])

    # ------------------------------------------------------------------
    def run_opf(self) -> None:
        """Run the pandapower OPF on the controller's network (raises on failure)."""
        if self.mode == "ac":
            self._pp.runopp(self.network, **self.opf_kwargs)
        else:
            self._pp.rundcopp(self.network, **self.opf_kwargs)

    def decide(self, t, measurements, forecasts):
        for p, v in measurements.items():
            if v is not None and math.isfinite(v):
                self._write_input(p, v)
        t0 = _time.perf_counter()
        try:
            self.run_opf()
            status = STATUS_OPTIMAL if getattr(self.network, "OPF_converged", True) else STATUS_INFEASIBLE
        except Exception as exc:  # OPFNotConverged and friends
            name = type(exc).__name__
            status = STATUS_INFEASIBLE if "Converge" in name else STATUS_ERROR
            self.last_error = f"{name}: {exc}"
        elapsed = _time.perf_counter() - t0
        if status != STATUS_OPTIMAL:
            return {"solver_status": status, "solve_time_s": elapsed, "objective": float("nan"),
                    "fallback": FALLBACK_PREVIOUS_PLAN}
        out: Dict[str, Any] = {p: self._read_output(p) for p in self._setpoint_names}
        out.update({"solver_status": STATUS_OPTIMAL, "solve_time_s": elapsed,
                    "objective": float(getattr(self.network, "res_cost", float("nan"))),
                    "fallback": FALLBACK_NONE})
        return out

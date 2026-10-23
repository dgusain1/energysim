# -*- coding: utf-8 -*-
"""
Network-lite base: a graph of producers, consumers, storages and junctions
connected by directed, capacity-limited, lossy pipes.

This is deliberately **not** hydraulic modelling.  Every step solves a
quasi-steady energy balance (a small linear programme)::

    minimise   sum(cost * injection) - sum(weight * delivery)
    subject to node balance (pipe loss = flow efficiency), pipe capacity,
               producer availability, storage / buffer limits.

Unserved demand is always reported, never silently dropped.

Solvers
-------
* ``scipy.optimize.linprog`` (HiGHS) when scipy is installed - handles any
  topology including meshed ones.
* A pure-Python simplex on the producer-to-sink *path* formulation when scipy
  is absent.  It is exact for tree-shaped (radial) networks; meshed networks
  raise :class:`NetworkError` without scipy.

All quantities are ``MW`` internally; subclasses may convert at the I/O
boundary (see :class:`~energysim.networks.gas.GasNetwork`).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from energysim.base import EnergysimError
from energysim.models.common import ModelComponent

__all__ = ["EnergyNetwork", "NetworkError"]

_TINY = 1e-12


class NetworkError(EnergysimError):
    """Invalid network topology or an unsolvable balance."""


def _get_linprog():
    """Return ``scipy.optimize.linprog`` or ``None`` when scipy is absent."""
    try:
        from scipy.optimize import linprog  # type: ignore
    except Exception:  # pragma: no cover - depends on environment
        return None
    return linprog


def _clean(x: float) -> float:
    x = float(x)
    return 0.0 if abs(x) < _TINY else x


def _num(label: str, value: Any, lo: Optional[float] = None, hi: Optional[float] = None,
         strict_lo: bool = False) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise NetworkError(f"{label} must be a number, got {value!r}.") from None
    if math.isnan(v):
        raise NetworkError(f"{label} must not be NaN.")
    if lo is not None and (v < lo or (strict_lo and v <= lo)):
        raise NetworkError(f"{label} must be {'>' if strict_lo else '>='} {lo}, got {v}.")
    if hi is not None and v >= hi:
        raise NetworkError(f"{label} must be < {hi}, got {v}.")
    return v


class _Result:
    """Solver output: injections, deliveries and pipe inlet flows (MW)."""

    __slots__ = ("inj", "dlv", "flow")

    def __init__(self, inj, dlv, flow):
        self.inj = [_clean(v) for v in inj]
        self.dlv = [_clean(v) for v in dlv]
        self.flow = [_clean(v) for v in flow]


class EnergyNetwork(ModelComponent):
    """
    Simple energy network (graph of nodes and pipes) with an LP balance.

    Build it with :meth:`add_producer`, :meth:`add_consumer`,
    :meth:`add_storage`, :meth:`add_junction` and :meth:`add_pipe` (or
    :meth:`from_dict`).  Variables are registered as ``<node>.<quantity>``:

    * producer: inputs ``available_MW``, ``cost``; output ``dispatch_MW``
    * consumer: input ``demand_MW``; outputs ``delivered_MW``, ``unserved_MW``
    * storage: input ``p_cmd_MW`` (positive = discharge); output ``soc``
    * pipe (named ``<a>-<b>``): outputs ``flow_MW`` (inlet), ``loss_MW``,
      ``utilization``
    * network: ``total_supply_MW``, ``total_demand_MW``, ``total_unserved_MW``,
      ``total_losses_MW``

    The part before the first dot is the simulator name in world
    connections, e.g. ``grid.consumer1.demand_MW``.

    Notes
    -----
    * ``cost`` is per MWh of primary energy; the injected cost is
      ``cost / carrier_efficiency``.  ``dispatch_MW`` is energy injected into
      the network.
    * Consumers are served in insertion order under scarcity (soft priority).
    * Storage ``p_cmd_MW`` is a command: it is limited by ``p_max`` and the
      stored energy / headroom and by what the network can actually move.
    * ``loss_pct_per_day`` is a standing storage loss reported in
      ``balance['standing_loss_MW']`` (not part of ``total_losses_MW``).
    """

    carrier = "energy"
    flow_unit = "MW"

    def __init__(self, name: str = "network", step_size: float = 1.0, **kwargs):
        super().__init__(name=name, step_size=step_size)
        self.units = {}
        self._nodes: Dict[str, str] = {}        # name -> kind
        self._producers: Dict[str, Dict[str, float]] = {}
        self._consumers: Dict[str, Dict[str, float]] = {}
        self._storages: Dict[str, Dict[str, float]] = {}
        self._pipes: List[Dict[str, Any]] = []
        # dynamic state (restored by the base init snapshot)
        self._soc: Dict[str, float] = {}
        # last-step energy balance, in MW rates
        self.balance: Dict[str, float] = {}
        totals = ["total_supply_MW", "total_demand_MW", "total_unserved_MW", "total_losses_MW"]
        self.register_variables(outputs=totals)
        for t in totals:
            self.units[t] = self._fu()

    def _update_outputs(self) -> None:
        # step_model is quiescent at dt == 0, so no snapshot/restore is needed
        # (and ``balance`` stays valid straight after init()).
        self.step_model(0.0, 0.0)

    # ------------------------------------------------------------------
    # Unit hooks
    # ------------------------------------------------------------------
    def _fu(self) -> str:
        """Unit string of flow quantities (``*_MW`` variables)."""
        return self.flow_unit

    def _ext_per_mw(self) -> float:
        """External flow units per internal MW."""
        return 1.0

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    def _new_node(self, name: str, kind: str) -> str:
        if not isinstance(name, str) or not name:
            raise NetworkError("Node names must be non-empty strings.")
        if "." in name:
            raise NetworkError(f"Node name '{name}' must not contain '.' (it separates variable names).")
        if name in self._nodes or any(p["name"] == name for p in self._pipes):
            raise NetworkError(f"Duplicate name '{name}' in network '{self.name}'.")
        self._nodes[name] = kind
        return name

    def _reg(self, node: str, inputs=(), outputs=(), defaults=None, units=None) -> None:
        self.register_variables(
            inputs=[f"{node}.{q}" for q in inputs],
            outputs=[f"{node}.{q}" for q in outputs],
            defaults={f"{node}.{k}": v for k, v in (defaults or {}).items()},
        )
        for q, u in (units or {}).items():
            self.units[f"{node}.{q}"] = u

    def add_producer(self, name: str, capacity: float, cost: float = 0.0,
                     carrier_efficiency: float = 1.0) -> "EnergyNetwork":
        capacity = _num(f"producer '{name}' capacity", capacity, lo=0.0)
        cost = _num(f"producer '{name}' cost", cost)
        eff = _num(f"producer '{name}' carrier_efficiency", carrier_efficiency, lo=0.0, strict_lo=True)
        self._new_node(name, "producer")
        self._producers[name] = {"capacity": capacity, "cost": cost, "carrier_efficiency": eff}
        fu = self._fu()
        self._reg(name, inputs=("available_MW", "cost"), outputs=("dispatch_MW",),
                  defaults={"available_MW": capacity * self._ext_per_mw(), "cost": cost},
                  units={"available_MW": fu, "cost": "currency/MWh", "dispatch_MW": fu})
        return self

    def add_consumer(self, name: str, demand: float = 0.0) -> "EnergyNetwork":
        demand = _num(f"consumer '{name}' demand", demand, lo=0.0)
        self._new_node(name, "consumer")
        self._consumers[name] = {"demand": demand}
        fu = self._fu()
        self._reg(name, inputs=("demand_MW",), outputs=("delivered_MW", "unserved_MW"),
                  defaults={"demand_MW": demand * self._ext_per_mw()},
                  units={"demand_MW": fu, "delivered_MW": fu, "unserved_MW": fu})
        return self

    def add_storage(self, name: str, capacity_mwh: float, p_max: float, soc_init: float = 0.5,
                    loss_pct_per_day: float = 0.0) -> "EnergyNetwork":
        capacity_mwh = _num(f"storage '{name}' capacity_mwh", capacity_mwh, lo=0.0, strict_lo=True)
        p_max = _num(f"storage '{name}' p_max", p_max, lo=0.0)
        soc_init = _num(f"storage '{name}' soc_init", soc_init, lo=0.0)
        if soc_init > 1.0:
            raise NetworkError(f"storage '{name}' soc_init must be in [0, 1], got {soc_init}.")
        loss = _num(f"storage '{name}' loss_pct_per_day", loss_pct_per_day, lo=0.0, hi=100.0)
        self._new_node(name, "storage")
        self._storages[name] = {"capacity_mwh": capacity_mwh, "p_max": p_max,
                                "soc_init": soc_init, "loss_pct_per_day": loss}
        self._soc[name] = soc_init
        self._reg(name, inputs=("p_cmd_MW",), outputs=("soc",), defaults={"p_cmd_MW": 0.0},
                  units={"p_cmd_MW": self._fu(), "soc": "1"})
        return self

    def add_junction(self, name: str) -> "EnergyNetwork":
        self._new_node(name, "junction")
        return self

    def add_pipe(self, a: str, b: str, capacity: float, loss_pct: float = 0.0,
                 length_km: Optional[float] = None, delay_s: float = 0.0,
                 name: Optional[str] = None) -> str:
        """Add a directed pipe ``a -> b``; returns the pipe name (``"a-b"`` by default)."""
        for n in (a, b):
            if n not in self._nodes:
                raise NetworkError(f"Pipe endpoint '{n}' is not a node of network '{self.name}'.")
        if a == b:
            raise NetworkError(f"Pipe endpoints must differ (got '{a}' twice).")
        label = f"pipe {a}->{b}"
        capacity = _num(f"{label} capacity", capacity, lo=0.0)
        if math.isinf(capacity):
            raise NetworkError(f"{label} capacity must be finite.")
        loss = _num(f"{label} loss_pct", loss_pct, lo=0.0, hi=100.0)
        delay = _num(f"{label} delay_s", delay_s, lo=0.0)
        if length_km is not None:
            length_km = _num(f"{label} length_km", length_km, lo=0.0)
        if name is None:
            name = f"{a}-{b}"
            k = 2
            while name in self._nodes or any(p["name"] == name for p in self._pipes):
                name = f"{a}-{b}-{k}"
                k += 1
        elif name in self._nodes or "." in name or any(p["name"] == name for p in self._pipes):
            raise NetworkError(f"Invalid or duplicate pipe name '{name}'.")
        self._pipes.append({"name": name, "a": a, "b": b, "capacity": capacity,
                            "loss_pct": loss, "length_km": length_km, "delay_s": delay})
        fu = self._fu()
        self._reg(name, outputs=("flow_MW", "loss_MW", "utilization"),
                  units={"flow_MW": fu, "loss_MW": fu, "utilization": "1"})
        return name

    # ------------------------------------------------------------------
    # (De)serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Topology and parameters as plain data (inverse of :meth:`from_dict`)."""
        return {
            "type": self.carrier,
            "name": self.name,
            "parameters": self._params_dict(),
            "producers": {k: dict(v) for k, v in self._producers.items()},
            "consumers": {k: dict(v) for k, v in self._consumers.items()},
            "storages": {k: dict(v) for k, v in self._storages.items()},
            "junctions": [k for k, v in self._nodes.items() if v == "junction"],
            "pipes": [dict(p) for p in self._pipes],
        }

    def _params_dict(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def from_dict(cls, spec: Dict[str, Any]) -> "EnergyNetwork":
        """Build a network from a dict (e.g. loaded from YAML).

        Keys: ``type`` (``'heat'``/``'gas'``/``'energy'``, used when called on
        the base class), ``name``, ``parameters`` (constructor kwargs),
        ``producers``/``consumers``/``storages`` (``{name: {kwargs}}``),
        ``junctions`` (list of names) and ``pipes`` (list of dicts with ``a``,
        ``b`` and the :meth:`add_pipe` keywords).
        """
        if not isinstance(spec, dict):
            raise NetworkError("Network spec must be a dict.")
        target = cls
        if cls is EnergyNetwork:
            from .gas import GasNetwork
            from .heat import HeatNetwork
            kinds = {"energy": EnergyNetwork, "heat": HeatNetwork, "gas": GasNetwork}
            t = str(spec.get("type", "energy")).lower()
            if t not in kinds:
                raise NetworkError(f"Unknown network type '{t}'; expected one of {sorted(kinds)}.")
            target = kinds[t]
        known = {"type", "name", "parameters", "producers", "consumers", "storages", "junctions", "pipes"}
        extra = set(spec) - known
        if extra:
            raise NetworkError(f"Unknown network spec keys: {sorted(extra)}.")
        net = target(name=spec.get("name", "network"), **dict(spec.get("parameters") or {}))
        for n, kw in (spec.get("producers") or {}).items():
            net.add_producer(n, **kw)
        for n, kw in (spec.get("consumers") or {}).items():
            net.add_consumer(n, **kw)
        for n, kw in (spec.get("storages") or {}).items():
            net.add_storage(n, **kw)
        for n in spec.get("junctions") or []:
            net.add_junction(n)
        for kw in spec.get("pipes") or []:
            kw = dict(kw)
            a, b = kw.pop("a"), kw.pop("b")
            net.add_pipe(a, b, **kw)
        return net

    def describe_topology(self) -> str:
        """Human-readable multi-line topology summary."""
        lines = [f"{type(self).__name__} '{self.name}' ({self.carrier}, unit {self._fu()})"]
        for n, p in self._producers.items():
            lines.append(f"  producer  {n}: capacity {p['capacity']} MW, cost {p['cost']}, "
                         f"eff {p['carrier_efficiency']}")
        for n, c in self._consumers.items():
            lines.append(f"  consumer  {n}: demand {c['demand']} MW")
        for n, s in self._storages.items():
            lines.append(f"  storage   {n}: {s['capacity_mwh']} MWh, p_max {s['p_max']} MW, "
                         f"soc0 {s['soc_init']}, loss {s['loss_pct_per_day']} %/day")
        for n, k in self._nodes.items():
            if k == "junction":
                lines.append(f"  junction  {n}")
        for p in self._pipes:
            extra = f", {p['length_km']} km" if p["length_km"] is not None else ""
            lines.append(f"  pipe      {p['a']} -> {p['b']} [{p['name']}]: cap {p['capacity']} MW, "
                         f"loss {p['loss_pct']} %, delay {p['delay_s']} s{extra}")
        lines.append(f"  topology: {'meshed' if self._is_meshed() else 'radial (forest)'}")
        return "\n".join(lines)

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d["units"] = {k: v for k, v in self.units.items() if k in self.inputs or k in self.outputs}
        d["topology"] = self.to_dict()
        return d

    @classmethod
    def describe_class(cls) -> Dict[str, Any]:
        return cls().describe()

    # ------------------------------------------------------------------
    # Topology helpers
    # ------------------------------------------------------------------
    def _is_meshed(self) -> bool:
        """True when the undirected graph contains a cycle (incl. parallel pipes)."""
        parent = {n: n for n in self._nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for p in self._pipes:
            ra, rb = find(p["a"]), find(p["b"])
            if ra == rb:
                return True
            parent[ra] = rb
        return False

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------
    def _extra_buffers(self, h: float) -> List[Dict[str, Any]]:
        """Additional storage-like buffers (gas linepack)."""
        return []

    def _finish_step(self, dt: float, h: float, res: _Result, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process deliveries; return ``delivered`` list, ``overshoot`` and ``dtransit``."""
        return {"delivered": list(res.dlv), "overshoot": 0.0, "dtransit": 0.0}

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------
    def step_model(self, dt: float, time: float) -> None:
        dt = float(dt)
        h = dt / 3600.0
        k = self._ext_per_mw()
        iv = self._inputs_val

        sources: List[Dict[str, Any]] = []   # node, cap, cost (+ bookkeeping)
        sinks: List[Dict[str, Any]] = []
        cmax = 1.0
        for n, p in self._producers.items():
            avail = _num(f"{n}.available_MW", iv.get(f"{n}.available_MW", p["capacity"] * k)) / k
            cost = _num(f"{n}.cost", iv.get(f"{n}.cost", p["cost"]))
            cap = max(0.0, min(avail, p["capacity"]))
            ec = cost / p["carrier_efficiency"]
            cmax = max(cmax, abs(ec))
            sources.append({"kind": "producer", "key": n, "node": n, "cap": cap, "cost": ec})
        ci = []
        for n in self._consumers:
            dem = max(0.0, _num(f"{n}.demand_MW", iv.get(f"{n}.demand_MW", 0.0)) / k)
            ci.append(dem)
            sinks.append({"kind": "consumer", "key": n, "node": n, "cap": dem, "tier": 1.0})
        # storages: command-driven discharge / charge
        for n, s in self._storages.items():
            cmd = _num(f"{n}.p_cmd_MW", iv.get(f"{n}.p_cmd_MW", 0.0)) / k
            e = self._soc[n] * s["capacity_mwh"]
            dis_lim = s["p_max"] if h <= 0 else min(s["p_max"], e / h)
            chg_lim = s["p_max"] if h <= 0 else min(s["p_max"], (s["capacity_mwh"] - e) / h)
            if h <= 0:
                dis_lim = s["p_max"] if e > _TINY else 0.0
                chg_lim = s["p_max"] if s["capacity_mwh"] - e > _TINY else 0.0
            sources.append({"kind": "storage", "key": n, "node": n,
                            "cap": min(max(cmd, 0.0), dis_lim), "cost": 0.0})
            sinks.append({"kind": "storage", "key": n, "node": n,
                          "cap": min(max(-cmd, 0.0), chg_lim), "tier": 0.1})
        for buf in self._extra_buffers(h):
            sources.append({"kind": buf["kind"], "key": buf["key"], "node": buf["node"],
                            "cap": buf["dis_cap"], "cost": None})
            sinks.append({"kind": buf["kind"], "key": buf["key"], "node": buf["node"],
                          "cap": buf["chg_cap"], "tier": 0.02})
        S = cmax
        # Deterministic perturbations: producer index, sink priority, pipe index.
        ns, nt = len(sources), len(sinks)
        for i, s in enumerate(sources):
            if s["cost"] is None:       # linepack discharge: dearer than any producer
                s["cost"] = 25.0 * S
            s["cost"] += 1e-3 * S * (1 + i) / (1 + ns)
        for j, t in enumerate(sinks):
            base = 1e3 * S * t["tier"]
            if t["kind"] not in ("consumer", "storage"):
                base = 20.0 * S         # linepack refill: below discharge cost, above producers
            t["weight"] = base * (1.0 - 1e-3 * j / max(nt, 1))
        pipes = self._pipes
        nidx = self._node_idx
        arcs = [(nidx[p["a"]], nidx[p["b"]], p["capacity"],
                 p["loss_pct"] / 100.0, 1e-4 * S * (1 + e) / (1 + len(pipes)))
                for e, p in enumerate(pipes)]
        res = self._solve(sources, sinks, arcs)

        ctx = {"sources": sources, "sinks": sinks, "demands": ci, "arcs": arcs}
        fin = self._finish_step(dt, h, res, ctx)
        self._store(dt, h, res, ctx, fin)

    # -- node indexing ---------------------------------------------------
    @property
    def _node_idx(self) -> Dict[str, int]:
        return {n: i for i, n in enumerate(self._nodes)}

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------
    def _solve(self, sources, sinks, arcs) -> _Result:
        nn = len(self._nodes)
        idx = self._node_idx
        linprog = _get_linprog()
        if linprog is not None:
            return self._solve_lp(linprog, nn, idx, sources, sinks, arcs)
        if self._is_meshed():
            raise NetworkError(
                f"Network '{self.name}' is meshed (contains a cycle); solving it requires scipy "
                f"(pip install energysim[networks]).")
        return self._solve_tree(idx, sources, sinks, arcs)

    def _solve_lp(self, linprog, nn, idx, sources, sinks, arcs) -> _Result:
        ns, nt, ne = len(sources), len(sinks), len(arcs)
        nv = ns + nt + ne
        if nv == 0:
            return _Result([], [], [])
        c = np.zeros(nv)
        bounds = []
        A = np.zeros((nn, nv))
        for i, s in enumerate(sources):
            c[i] = s["cost"]
            bounds.append((0.0, s["cap"]))
            A[idx[s["node"]], i] += 1.0
        for j, t in enumerate(sinks):
            c[ns + j] = -t["weight"]
            bounds.append((0.0, t["cap"]))
            A[idx[t["node"]], ns + j] -= 1.0
        for e, (a, b, cap, loss, eps) in enumerate(arcs):
            c[ns + nt + e] = eps
            bounds.append((0.0, cap))
            A[a, ns + nt + e] -= 1.0
            A[b, ns + nt + e] += 1.0 - loss
        sol = linprog(c, A_eq=A, b_eq=np.zeros(nn), bounds=bounds, method="highs")
        if sol.status != 0:
            raise NetworkError(f"Network '{self.name}': LP balance failed ({sol.message}).")
        x = sol.x
        return _Result(x[:ns], x[ns:ns + nt], x[ns + nt:])

    def _solve_tree(self, idx, sources, sinks, arcs) -> _Result:
        """Exact solve for forests via simplex on the path formulation (pure Python)."""
        ns, nt, ne = len(sources), len(sinks), len(arcs)
        out: Dict[int, List[int]] = {}
        for e, (a, b, *_rest) in enumerate(arcs):
            out.setdefault(a, []).append(e)
        sinks_at: Dict[int, List[int]] = {}
        for j, t in enumerate(sinks):
            sinks_at.setdefault(idx[t["node"]], []).append(j)

        cols: List[Tuple[int, int, float, List[Tuple[int, float]]]] = []
        coef: List[float] = []
        for i, s in enumerate(sources):
            start = idx[s["node"]]
            stack = [(start, 1.0, [], 0.0)]
            while stack:                      # DFS in deterministic arc order
                node, gain, path, epscost = stack.pop()
                for j in sinks_at.get(node, ()):
                    if gain <= 0.0:
                        continue
                    cc = sinks[j]["weight"] - (s["cost"] + epscost) / gain
                    if cc > 1e-12:
                        cols.append((i, j, gain, list(path)))
                        coef.append(cc)
                for e in reversed(out.get(node, ())):
                    a, b, cap, loss, eps = arcs[e]
                    stack.append((b, gain * (1.0 - loss), path + [(e, gain)], epscost + eps * gain))
        n = len(cols)
        m = ns + nt + ne
        A = [[0.0] * n for _ in range(m)]
        for v, (i, j, g, path) in enumerate(cols):
            A[i][v] = 1.0 / g
            A[ns + j][v] = 1.0
            for e, q in path:
                A[ns + nt + e][v] = q / g
        b = [s["cap"] for s in sources] + [t["cap"] for t in sinks] + [a[2] for a in arcs]
        y = _simplex_max(coef, A, b)
        inj = [0.0] * ns
        dlv = [0.0] * nt
        flow = [0.0] * ne
        for v, (i, j, g, path) in enumerate(cols):
            if y[v] <= 0.0:
                continue
            inj[i] += y[v] / g
            dlv[j] += y[v]
            for e, q in path:
                flow[e] += y[v] * q / g
        return _Result(inj, dlv, flow)

    # ------------------------------------------------------------------
    # Result bookkeeping
    # ------------------------------------------------------------------
    def _store(self, dt, h, res: _Result, ctx, fin) -> None:
        k = self._ext_per_mw()
        sources, sinks = ctx["sources"], ctx["sinks"]
        out = self._outputs_val
        supply = discharge = charge = lp_net = 0.0
        for i, s in enumerate(sources):
            if s["kind"] == "producer":
                supply += res.inj[i]
                out[f"{s['key']}.dispatch_MW"] = res.inj[i] * k
            elif s["kind"] == "storage":
                discharge += res.inj[i]
            else:
                lp_net -= res.inj[i]
        for j, t in enumerate(sinks):
            if t["kind"] == "storage":
                charge += res.dlv[j]
            elif t["kind"] != "consumer":
                lp_net += res.dlv[j]
        delivered = fin["delivered"]
        total_dem = total_unserved = total_deliv = 0.0
        for j, n in enumerate(self._consumers):
            d = ctx["demands"][j]
            dl = min(delivered[j], d)
            un = max(0.0, d - dl)
            total_dem += d
            total_unserved += un
            total_deliv += dl
            out[f"{n}.delivered_MW"] = dl * k
            out[f"{n}.unserved_MW"] = un * k
        pipe_loss = 0.0
        for e, p in enumerate(self._pipes):
            f = res.flow[e]
            loss = f * p["loss_pct"] / 100.0
            pipe_loss += loss
            out[f"{p['name']}.flow_MW"] = f * k
            out[f"{p['name']}.loss_MW"] = loss * k
            out[f"{p['name']}.utilization"] = f / p["capacity"] if p["capacity"] > 0 else 0.0
        losses = pipe_loss + fin["overshoot"]
        # storage state update (no change at dt == 0)
        standing = 0.0
        sto_dis = {s["key"]: res.inj[i] for i, s in enumerate(sources) if s["kind"] == "storage"}
        sto_chg = {t["key"]: res.dlv[j] for j, t in enumerate(sinks) if t["kind"] == "storage"}
        for n, s in self._storages.items():
            cap = s["capacity_mwh"]
            e0 = self._soc[n] * cap
            e1 = e0 + (sto_chg[n] - sto_dis[n]) * h
            loss_mwh = max(e1, 0.0) * s["loss_pct_per_day"] / 100.0 * dt / 86400.0
            e1 = min(max(e1 - loss_mwh, 0.0), cap)
            standing += loss_mwh / h if h > 0 else 0.0
            if dt > 0:
                self._soc[n] = e1 / cap
            out[f"{n}.soc"] = e1 / cap
        out["total_supply_MW"] = supply * k
        out["total_demand_MW"] = total_dem * k
        out["total_unserved_MW"] = total_unserved * k
        out["total_losses_MW"] = losses * k
        self.balance = {
            "supply_MW": supply, "storage_discharge_MW": discharge, "delivered_MW": total_deliv,
            "losses_MW": losses, "storage_charge_MW": charge, "dlinepack_MW": lp_net,
            "dtransit_MW": fin["dtransit"], "standing_loss_MW": standing,
            "demand_MW": total_dem, "unserved_MW": total_unserved,
        }
        self._apply_extra_state(dt, h, res, ctx)

    def _apply_extra_state(self, dt, h, res, ctx) -> None:
        """Hook: update subclass state (linepack, transit) after the solve."""

    def balance_residual(self) -> float:
        """Last-step ``supply + discharge - (delivered + losses + charge + d(buffers))`` in MW."""
        b = self.balance
        return (b["supply_MW"] + b["storage_discharge_MW"]
                - (b["delivered_MW"] + b["losses_MW"] + b["storage_charge_MW"]
                   + b["dlinepack_MW"] + b["dtransit_MW"]))

    # -- flow-weighted propagation (used by heat) ---------------------------
    def _propagate(self, res: _Result, ctx) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Flow-weighted mean transport delay (s) and surviving fraction per node."""
        idx = self._node_idx
        names = list(self._nodes)
        arcs = ctx["arcs"]
        inj = [0.0] * len(names)
        for i, s in enumerate(ctx["sources"]):
            inj[idx[s["node"]]] += res.inj[i]
        indeg = [0] * len(names)
        ins: Dict[int, List[int]] = {}
        outs: Dict[int, List[int]] = {}
        live = [e for e in range(len(arcs)) if res.flow[e] > _TINY]
        for e in live:
            a, b = arcs[e][0], arcs[e][1]
            indeg[b] += 1
            ins.setdefault(b, []).append(e)
            outs.setdefault(a, []).append(e)
        tau = [0.0] * len(names)
        surv = [1.0] * len(names)
        done = [False] * len(names)
        queue = [i for i in range(len(names)) if indeg[i] == 0]
        order: List[int] = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            done[n] = True
            for e in outs.get(n, ()):
                b = arcs[e][1]
                indeg[b] -= 1
                if indeg[b] == 0:
                    queue.append(b)
        for n in order:
            mass = inj[n]
            t_acc = 0.0
            s_acc = inj[n]
            for e in ins.get(n, ()):
                a = arcs[e][0]
                arr = res.flow[e] * (1.0 - arcs[e][3])
                mass += arr
                t_acc += arr * (tau[a] + self._pipes[e]["delay_s"])
                s_acc += arr * surv[a] * (1.0 - arcs[e][3])
            if mass > _TINY:
                tau[n] = t_acc / mass
                surv[n] = s_acc / mass
        return ({names[i]: tau[i] for i in range(len(names))},
                {names[i]: surv[i] for i in range(len(names))})


def _simplex_max(c: Sequence[float], A: List[List[float]], b: Sequence[float]) -> List[float]:
    """Maximise ``c.y`` s.t. ``A y <= b``, ``y >= 0`` (``b >= 0``); Bland's rule, dense tableau."""
    n = len(c)
    m = len(b)
    if n == 0:
        return []
    cs = max(abs(v) for v in c) or 1.0
    T = [A[i][:] + [1.0 if j == i else 0.0 for j in range(m)] + [float(b[i])] for i in range(m)]
    z = [-v / cs for v in c] + [0.0] * (m + 1)
    basis = [n + i for i in range(m)]
    tol = 1e-11
    for _ in range(200000):
        col = -1
        for j in range(n + m):
            if z[j] < -tol:
                col = j
                break
        if col < 0:
            break
        row = -1
        best = math.inf
        for i in range(m):
            a = T[i][col]
            if a > 1e-12:
                r = T[i][-1] / a
                if r < best - 1e-12 or (abs(r - best) <= 1e-12 and basis[i] < basis[row]):
                    best, row = r, i
        if row < 0:
            raise NetworkError("Unbounded balance problem (check pipe capacities).")
        piv = T[row][col]
        T[row] = [v / piv for v in T[row]]
        for i in range(m):
            if i != row:
                f = T[i][col]
                if f != 0.0:
                    Ti, Tr = T[i], T[row]
                    T[i] = [Ti[q] - f * Tr[q] for q in range(n + m + 1)]
        f = z[col]
        z = [z[q] - f * T[row][q] for q in range(n + m + 1)]
        basis[row] = col
    else:  # pragma: no cover
        raise NetworkError("Simplex iteration limit reached.")
    y = [0.0] * n
    for i, bv in enumerate(basis):
        if bv < n:
            y[bv] = max(T[i][-1], 0.0)
    return y

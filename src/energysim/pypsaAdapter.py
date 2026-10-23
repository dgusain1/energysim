"""
PyPSA adapter for energysim.

LIMITATION NOTE (Pattern B - Single-Snapshot Mode):
This adapter operates in Pattern B (Single Snapshot Online Dispatch/Power Flow).
At each co-simulation macro time step, the PyPSA network evaluates a single snapshot.
Multi-period lookahead time series optimization (e.g., continuous multi-day LOPF
with inter-temporal storage/SOC linkages) is evaluated sequentially snapshot-by-snapshot
within the co-simulation loop rather than solving the entire multi-period time-horizon
simultaneously.

Variable naming
---------------
``"<component name>.<variable>"``; a component type prefix disambiguates names
that occur in several component tables (``"Bus:b1.V"``, accepted prefixes:
Generator, Load, StorageUnit, Bus, Line, Transformer).

Inputs (``set_value``): ``P`` / ``Q`` (written to ``p_set`` / ``q_set``) of
generators, loads and storage units.  Note that a generator with
``control='Slack'`` ignores ``p_set`` during a power flow.

Outputs (``get_value``), read from the ``*_t`` result tables:

* Generator / Load / StorageUnit: ``P``, ``Q``
* Bus: ``V`` (p.u.), ``Va`` (radians, PyPSA native), ``Va_deg``, ``P``, ``Q`` (net injection)
* Line / Transformer: ``p0``, ``q0``, ``p1``, ``q1``, ``loading_percent``
  (``max(|S0|, |S1|) / s_nom * 100``)

In ``mode='dc'`` (``network.lpf()``) PyPSA computes active power only: all
``Q`` / reactive flow outputs are 0.0 and ``V`` is 1.0 p.u.

Re-runnable: a pristine copy of the network is kept at construction and
restored on the first ``init()`` after ``cleanup()`` (``cleanup`` never
deletes the network).
"""
from __future__ import annotations

import copy
import logging
import math
import warnings

from .base import SimulatorAdapter, SimulatorVariableError, SimulatorElementNotFoundError

# (static table, time-series table, accepted type prefixes)
_COMPONENTS = (
    ('generators', 'generators_t', 'generator'),
    ('loads', 'loads_t', 'load'),
    ('storage_units', 'storage_units_t', 'storageunit'),
    ('buses', 'buses_t', 'bus'),
    ('lines', 'lines_t', 'line'),
    ('transformers', 'transformers_t', 'transformer'),
)
_SETTABLE = ('generators', 'loads', 'storage_units')
_PQ_OUTPUT = ('generators', 'loads', 'storage_units')
_BRANCH = ('lines', 'transformers')


def _copy_network(net):
    try:
        return net.copy()
    except Exception:
        return copy.deepcopy(net)


class pypsa_adapter(SimulatorAdapter):
    """
    Adapter for PyPSA networks in energysim (AC ``pf`` and DC ``lpf``).

    Design Characteristic / Limitation (Pattern B):
    Operates on a single active snapshot at each macro time step. Multi-snapshot
    lookahead horizons are evaluated sequentially step-by-step.
    """

    SET_PARAMS_BEFORE_INIT = False

    def __init__(self, network_name, net_loc=None, inputs=None, outputs=None, mode='ac', pf=None, logger_level='CRITICAL', model=None):
        import pypsa  # lazy import: pypsa is optional and has heavy dependencies
        # PyPSA >= 1.0 moved the power flow logger; the package-level logger
        # controls every sub-module logger in all versions.
        level = getattr(logging, str(logger_level).upper(), logging.CRITICAL)
        logging.getLogger('pypsa').setLevel(level)
        self.network_name = network_name
        target = model if model is not None else net_loc
        if target is None:
            raise ValueError(f"No network or model specified for PyPSA simulator '{network_name}'.")

        from .utils import resolve_model_reference
        target = resolve_model_reference(target)

        # Standardized powerflow mode: strictly 'ac' or 'dc'
        raw_mode = (mode if (mode != 'ac' or pf is None) else pf) or 'ac'
        raw_mode = raw_mode.lower()
        if raw_mode in ('ac', 'pf'):
            self.mode = 'ac'
        elif raw_mode in ('dc', 'dcpf', 'lpf'):
            self.mode = 'dc'
        else:
            raise ValueError(
                f"Unsupported calculation mode '{raw_mode}'. "
                f"energysim only supports 'ac' and 'dc' power flow modes."
            )
        self.pf = self.mode  # backward compatibility alias

        if hasattr(target, 'snapshots') and hasattr(target, 'buses'):
            self.network = target
            self.net_loc = getattr(target, 'name', None) or network_name
        elif isinstance(target, str):
            self.net_loc = target
            self.network = pypsa.Network(target)
        elif callable(target):
            self.network = target()
            self.net_loc = getattr(self.network, 'name', None) or network_name
        else:
            self.network = target
            self.net_loc = network_name

        if len(self.network.snapshots) != 1:
            raise ValueError(
                "Pattern B limitation: only networks with exactly one snapshot are supported "
                f"(network '{network_name}' has {len(self.network.snapshots)})."
            )
        self.inputs = list(inputs or [])
        self.outputs = list(outputs or [])

        self._pristine = _copy_network(self.network)
        self._needs_restore = False
        self._build_caches()

        for item in self.inputs:
            static, _, attr = self._resolve(item)
            self._input_attr(static, attr, item)
        for item in self.outputs:
            static, _, attr = self._resolve(item)
            self._output_kind(static, attr, item)

    # ------------------------------------------------------------------
    # Name resolution
    # ------------------------------------------------------------------
    def _build_caches(self):
        self._lookup = {}   # name -> (static, tname)
        self._by_type = {}  # (prefix, name) -> (static, tname)
        dups = {}
        for static, tname, prefix in _COMPONENTS:
            df = getattr(self.network, static, None)
            if df is None:
                continue
            for name in df.index:
                self._by_type[(prefix, str(name))] = (static, tname)
                if str(name) in self._lookup:
                    dups.setdefault(str(name), [self._lookup[str(name)][0]]).append(static)
                else:
                    self._lookup[str(name)] = (static, tname)
        if dups:
            detail = "; ".join(f"'{n}' in {k}" for n, k in dups.items())
            warnings.warn(
                f"PyPSA network '{self.network_name}': component names occur in several tables ({detail}). "
                f"The first table wins; use a type prefix (e.g. 'Bus:name.V') to disambiguate.",
                UserWarning, stacklevel=3)

    def _resolve(self, parameter):
        """Return ``(static_table, ele_name, attribute)``."""
        if not isinstance(parameter, str) or '.' not in parameter:
            raise SimulatorVariableError(
                f"Malformed PyPSA variable '{parameter}'. Expected 'component_name.variable'.")
        body, prefix = parameter, None
        if ':' in body:
            head, rest = body.split(':', 1)
            if head.lower() in {c[2] for c in _COMPONENTS}:
                prefix, body = head.lower(), rest
        for pos in [i for i, c in enumerate(body) if c == '.']:
            name, attr = body[:pos], body[pos + 1:]
            if prefix is not None:
                hit = self._by_type.get((prefix, name))
            else:
                hit = self._lookup.get(name)
            if hit is not None:
                return hit[0], name, attr
        raise SimulatorElementNotFoundError(
            f"Could not find '{body.split('.', 1)[0]}' as a component in network '{self.network_name}'."
            f" Make sure element names are correctly specified."
        )

    @staticmethod
    def _input_attr(static, attr, parameter):
        if static not in _SETTABLE:
            raise SimulatorElementNotFoundError(
                f"Only Generator, Load and StorageUnit P, Q inputs are supported (got '{parameter}')."
            )
        if attr not in ('P', 'Q'):
            raise SimulatorVariableError(
                f"Powerflow input variable not valid in '{parameter}'. Use P or Q."
            )
        return attr.lower() + '_set'

    @staticmethod
    def _output_kind(static, attr, parameter):
        """Return the canonical (t-table attribute) name for an output or raise."""
        a = attr.lower()
        if static == 'buses':
            table = {'v': 'v_mag_pu', 'va': 'v_ang', 'va_deg': 'v_ang', 'p': 'p', 'q': 'q'}
        elif static in _BRANCH:
            table = {'p0': 'p0', 'q0': 'q0', 'p1': 'p1', 'q1': 'q1', 'loading_percent': 'loading_percent'}
        else:
            table = {'p': 'p', 'q': 'q'}
        if a not in table:
            raise SimulatorVariableError(
                f"Output variable '{attr}' is not valid for {static[:-1] if static.endswith('s') else static} "
                f"in '{parameter}'. Valid: {sorted(set(table))}."
            )
        return table[a]

    # ------------------------------------------------------------------
    # Life-cycle
    # ------------------------------------------------------------------
    def _ensure_pristine(self):
        if self._needs_restore:
            self.network = _copy_network(self._pristine)
            self._needs_restore = False

    def init(self):
        self._ensure_pristine()
        self.step(0)

    def step(self, time):
        if self.mode == 'ac':
            self.network.lpf()
            self.network.pf(use_seed=True)
        else:
            self.network.lpf()

    def cleanup(self):
        # Never delete the network; restoration happens lazily.
        self._needs_restore = True

    # Backward compatibility alias
    cleanUp = cleanup

    # ------------------------------------------------------------------
    # set / get
    # ------------------------------------------------------------------
    def set_value(self, parameters, values):
        self._ensure_pristine()
        n = self.network
        snap = n.snapshots[0]
        for parameter, value in zip(parameters, values):
            static, name, attr = self._resolve(parameter)
            col = self._input_attr(static, attr, parameter)
            value = float(value)
            getattr(n, static).at[name, col] = value
            # a time-varying series of the same name would override the static value
            series = getattr(getattr(n, static + '_t'), col, None)
            if series is not None and name in series.columns:
                series.at[snap, name] = value

    def get_value(self, parameters, time=None):
        n = self.network
        snap = n.snapshots[0]
        out = []
        for parameter in parameters:
            static, name, attr = self._resolve(parameter)
            key = self._output_kind(static, attr, parameter)
            t = getattr(n, static + '_t')
            if key == 'loading_percent':
                out.append(self._loading(static, name, t, snap))
                continue
            series = getattr(t, key, None)
            if series is not None and name in series.columns:
                val = float(series.at[snap, name])
                if attr.lower() == 'va_deg':
                    val = math.degrees(val)
                out.append(val)
            elif key in ('q', 'q0', 'q1'):
                out.append(0.0)  # DC power flow does not compute reactive power
            else:
                raise SimulatorVariableError(
                    f"No result for '{parameter}': run a power flow first (init()/step())."
                )
        return out

    def _loading(self, static, name, t, snap):
        def val(key):
            s = getattr(t, key, None)
            return float(s.at[snap, name]) if s is not None and name in s.columns else 0.0
        s0 = math.hypot(val('p0'), val('q0'))
        s1 = math.hypot(val('p1'), val('q1'))
        s_nom = float(getattr(self.network, static).at[name, 's_nom'])
        return max(s0, s1) / s_nom * 100.0 if s_nom > 0 else float('nan')

    def getOutput(self):
        return self.get_value(self.outputs)

    def setInput(self, inputValues):
        self.set_value(self.inputs, inputValues)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def get_available_variables(self):
        n = self.network
        inputs, outputs = [], []
        for static in _SETTABLE:
            for name in getattr(n, static).index:
                inputs.extend([f'{name}.P', f'{name}.Q'])
        for static in _PQ_OUTPUT:
            for name in getattr(n, static).index:
                outputs.extend([f'{name}.P', f'{name}.Q'])
        for name in n.buses.index:
            outputs.extend([f'{name}.{v}' for v in ('V', 'Va', 'Va_deg', 'P', 'Q')])
        for static in _BRANCH:
            for name in getattr(n, static).index:
                outputs.extend([f'{name}.{v}' for v in ('p0', 'q0', 'p1', 'q1', 'loading_percent')])
        return {'inputs': inputs, 'outputs': outputs}

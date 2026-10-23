# -*- coding: utf-8 -*-
"""
OpenDSS adapter for energysim using OpenDSSDirect.py.

Provides co-simulation capabilities for distribution systems modeled in
Electric Power Research Institute's (EPRI) OpenDSS.

Notes
-----
* **Only AC (snapshot) power flow is supported.**  OpenDSS has no DC power
  flow; ``mode='dc'`` raises ``ValueError``.
* OpenDSS (through OpenDSSDirect.py) is a **process-wide singleton**: only one
  circuit can be active per Python process.  A second ``opendss_adapter`` that
  calls ``init()`` while another is active raises ``EnergysimError``.
* ``step()`` re-solves the circuit (``Solution.Solve``) in the solution mode
  of the DSS script (snapshot unless the script says otherwise).  Time is
  **not** advanced: if the circuit uses loadshapes/``Set mode=daily``, the
  loadshape position is controlled by the DSS script, not by energysim.  Drive
  time-varying behaviour through inputs (e.g. ``Load.L1.kW``) instead.

Variable naming
---------------
Inputs: ``'<Type>.<Name>.<Property>'`` e.g. ``Load.ld1.kW``,
``Generator.pv.kW``.  Identifiers must match ``[A-Za-z0-9_-]+`` and the
property ``[A-Za-z][A-Za-z0-9_-]*``; values are numbers (or quoted strings).

Outputs:

* ``Bus.<bus>.vm_pu``   mean per-unit voltage magnitude of phase nodes 1-3
  (``Bus.<bus>.vm_pu.<node>`` selects a single node, e.g. ``.1``).  Requires
  voltage bases to be set in the DSS script.
* ``Bus.<bus>.vm_kv``   mean line-to-neutral voltage magnitude (kV)  (``.<node>`` allowed)
* ``Bus.<bus>.kv`` / ``kv_base``  bus base voltage (kV, line-to-neutral)
* ``Bus.<bus>.vang``    voltage angle (degrees) of the first selected node
  (``.<node>`` allowed)
* ``<Type>.<Name>.p_kw`` / ``q_kvar``  total power into terminal 1 summed over
  phases (``p2_kw`` / ``q2_kvar`` for terminal 2), ``losses_kw``,
  ``i_a`` (largest phase current at terminal 1), ``loading_percent``
  (``i_a / NormAmps``) for any circuit element (Line, Load, Generator, ...)
* ``Circuit.losses_kw``, ``Circuit.losses_kvar``, ``Circuit.total_power_kw``,
  ``Circuit.total_power_kvar`` (source power, negative = delivered into circuit)
"""

from __future__ import annotations

import os
import re
import weakref
from typing import Any, Dict, List, Optional

from .base import (
    SimulatorAdapter, SimulatorElementNotFoundError, SimulatorVariableError, EnergysimError,
)

_ID = re.compile(r'^[A-Za-z0-9_\-]+$')
_PROP = re.compile(r'^[A-Za-z][A-Za-z0-9_\-]*$')
_TYPE = re.compile(r'^[A-Za-z][A-Za-z0-9_]*$')
_BAD_STR = re.compile(r'["\'\r\n\x00;]')

# Process-wide singleton guard
_ACTIVE: Optional["weakref.ref"] = None


def _active_owner():
    return _ACTIVE() if _ACTIVE is not None else None


class opendss_adapter(SimulatorAdapter):
    """
    Adapter for OpenDSS distribution system simulation via OpenDSSDirect.py.

    Parameters
    ----------
    dss_name : str
        Unique identifier for the OpenDSS simulator.
    dss_location : str
        Path to the primary OpenDSS script (e.g. Master.dss).
    inputs, outputs : list of str, optional
        See module docstring for the naming scheme.
    step_size : float, optional
        Micro time step size in seconds.
    mode : str
        Only ``'ac'`` (snapshot) is supported.
    """

    SET_PARAMS_BEFORE_INIT = False

    def __init__(
        self,
        dss_name: str,
        dss_location: Optional[str] = None,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        step_size: float = 900.0,
        mode: str = 'ac',
        pf: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.dss_name = dss_name
        target = model if model is not None else dss_location
        self.dss_location = os.path.abspath(target) if target else ""
        self.inputs = list(inputs or [])
        self.outputs = list(outputs or [])
        self.step_size = step_size

        raw_mode = (mode if (mode != 'ac' or pf is None) else pf) or 'ac'
        raw_mode = raw_mode.lower()
        if raw_mode in ('ac', 'pf', 'snapshot'):
            self.mode = 'ac'
        elif raw_mode in ('dc', 'dcpf', 'direct'):
            raise ValueError(
                "OpenDSS does not provide a DC power flow: mode='dc' is not supported. "
                "Use mode='ac' (snapshot power flow)."
            )
        else:
            raise ValueError(
                f"Unsupported calculation mode '{raw_mode}'. "
                f"energysim only supports 'ac' and 'dc' power flow modes "
                f"(OpenDSS: 'ac' only)."
            )
        self.pf = self.mode

        # validate declared identifiers early (no OpenDSS needed)
        for item in self.inputs:
            self._parse_input(item)

        self._dss = None
        self._input_cache: Dict[str, Any] = {}
        self._owns = False
        self._static_cache = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_dss(self):
        """Lazy loader for opendssdirect module."""
        if self._dss is None:
            try:
                import opendssdirect as dss
                self._dss = dss
            except ImportError as err:
                raise ImportError(
                    "OpenDSS simulation requires 'opendssdirect.py'. "
                    "Please install it using: pip install opendssdirect.py"
                ) from err
        return self._dss

    @staticmethod
    def _parse_input(param):
        parts = param.split('.') if isinstance(param, str) else []
        if len(parts) != 3:
            raise SimulatorVariableError(
                f"Malformed OpenDSS input '{param}'. Expected 'Type.Name.Property' (e.g. 'Load.ld1.kW')."
            )
        etype, name, prop = parts
        if not _TYPE.match(etype) or not _ID.match(name) or not _PROP.match(prop):
            raise SimulatorVariableError(
                f"Invalid OpenDSS identifier in '{param}': type must match [A-Za-z][A-Za-z0-9_]*, "
                f"name [A-Za-z0-9_-]+ and property [A-Za-z][A-Za-z0-9_-]*."
            )
        return etype, name, prop

    @staticmethod
    def _format_value(val):
        if isinstance(val, bool):
            return 'yes' if val else 'no'
        if isinstance(val, str):
            if _BAD_STR.search(val):
                raise SimulatorVariableError(f"Illegal characters in OpenDSS value {val!r}.")
            try:
                return repr(float(val))
            except ValueError:
                return f'"{val}"'
        try:
            import numpy as np
            if isinstance(val, np.ndarray):
                if val.size != 1:
                    raise SimulatorVariableError("Array values are not supported for OpenDSS inputs.")
                val = val.reshape(-1)[0]
        except ImportError:  # pragma: no cover
            pass
        try:
            f = float(val)
        except (TypeError, ValueError) as exc:
            raise SimulatorVariableError(f"OpenDSS input value {val!r} is not numeric.") from exc
        if f != f or f in (float('inf'), float('-inf')):
            raise SimulatorVariableError(f"OpenDSS input value {val!r} is not finite.")
        return repr(f)

    def _command(self, dss, cmd, context):
        """Run a DSS command; OpenDSSDirect errors (DSSException) become EnergysimError."""
        try:
            dss.Command(cmd)
        except EnergysimError:
            raise
        except Exception as exc:
            raise EnergysimError(f"OpenDSS error during {context}: {exc}") from exc
        self._check_error(dss, context)

    def _check_error(self, dss, context):
        try:
            if dss.Error.Number() != 0:
                raise EnergysimError(f"OpenDSS error during {context}: {dss.Error.Description()}")
        except AttributeError:  # older opendssdirect versions
            pass

    # ------------------------------------------------------------------
    # Life-cycle
    # ------------------------------------------------------------------
    def init(self) -> None:
        """Initialize OpenDSS circuit and run initial solve."""
        global _ACTIVE
        owner = _active_owner()
        if owner is not None and owner is not self:
            raise EnergysimError(
                f"OpenDSS is a process-wide singleton and the simulator '{owner.dss_name}' is already "
                f"active. Call its cleanup() (or finish its simulation) before initialising "
                f"'{self.dss_name}'. Only one OpenDSS circuit can be simulated per Python process."
            )
        if not self.dss_location:
            raise ValueError(f"No OpenDSS script specified for simulator '{self.dss_name}'.")
        if not os.path.exists(self.dss_location):
            raise FileNotFoundError(f"OpenDSS model file not found: {self.dss_location}")
        if '"' in self.dss_location:
            raise ValueError("OpenDSS script path must not contain double quotes.")
        dss = self._get_dss()
        _ACTIVE = weakref.ref(self)
        self._owns = True
        try:
            self._command(dss, "Clear", "Clear")
            self._command(dss, f'Redirect "{self.dss_location}"', f"loading '{self.dss_location}'")
            self._command(dss, "Set mode=Snapshot", "Set mode")
            self._solve(dss)
        except Exception:
            self._release()
            raise

    def _solve(self, dss):
        try:
            dss.Solution.Solve()
        except Exception as exc:
            raise EnergysimError(f"OpenDSS solve failed for simulator '{self.dss_name}': {exc}") from exc
        if not dss.Solution.Converged():
            raise EnergysimError(f"OpenDSS power flow did not converge for simulator '{self.dss_name}'.")

    def _release(self):
        global _ACTIVE
        if self._owns:
            try:
                if self._dss is not None:
                    self._dss.Command("Clear")
            except Exception:
                pass
            if _active_owner() is self:
                _ACTIVE = None
            self._owns = False

    def cleanup(self) -> None:
        """Clear the OpenDSS circuit (only if this adapter owns it) and release the singleton."""
        self._release()
        self._input_cache.clear()

    def _require_active(self):
        if not self._owns or _active_owner() is not self:
            raise EnergysimError(
                f"OpenDSS simulator '{self.dss_name}' is not initialised (call init() first)."
            )
        return self._dss

    # ------------------------------------------------------------------
    # set / get
    # ------------------------------------------------------------------
    def set_value(self, parameters: List[str], values: List[Any]) -> None:
        """Apply inputs to OpenDSS elements via ``edit Type.Name Property=value``."""
        dss = self._require_active()
        for param, val in zip(parameters, values):
            etype, name, prop = self._parse_input(param)
            if dss.Circuit.SetActiveElement(f"{etype}.{name}") < 0:
                raise SimulatorElementNotFoundError(
                    f"OpenDSS element '{etype}.{name}' not found (simulator '{self.dss_name}')."
                )
            formatted = self._format_value(val)
            self._command(dss, f"edit {etype}.{name} {prop}={formatted}", f"setting {param}")
            self._input_cache[param] = val

    def _activate_element(self, dss, etype, name, param):
        if not _TYPE.match(etype) or not _ID.match(name):
            raise SimulatorVariableError(f"Invalid OpenDSS identifier in '{param}'.")
        if dss.Circuit.SetActiveElement(f"{etype}.{name}") < 0:
            raise SimulatorElementNotFoundError(f"OpenDSS element '{etype}.{name}' not found.")

    def _bus_value(self, dss, parts, param):
        if len(parts) < 3:
            raise SimulatorVariableError(
                f"Bus variable query '{param}' missing property. Expected 'Bus.<busname>.<prop>[.<node>]'."
            )
        bus_name, prop = parts[1], parts[2].lower()
        node = None
        if len(parts) == 4:
            try:
                node = int(parts[3])
            except ValueError as exc:
                raise SimulatorVariableError(f"Invalid node selector in '{param}'.") from exc
        elif len(parts) > 4:
            raise SimulatorVariableError(f"Malformed bus variable '{param}'.")
        if not _ID.match(bus_name):
            raise SimulatorVariableError(f"Invalid bus name in '{param}'.")
        if dss.Circuit.SetActiveBus(bus_name) < 0:
            raise SimulatorElementNotFoundError(f"OpenDSS bus '{bus_name}' not found.")

        nodes = list(dss.Bus.Nodes())
        if node is not None:
            if node not in nodes:
                raise SimulatorVariableError(
                    f"Node {node} does not exist on bus '{bus_name}' (nodes: {nodes}).")
            sel = [nodes.index(node)]
        else:
            sel = [i for i, nd in enumerate(nodes) if nd in (1, 2, 3)] or list(range(len(nodes)))

        if prop in ('kv', 'kv_base', 'kvbase'):
            return float(dss.Bus.kVBase())
        if prop in ('vm_pu', 'vmag_pu', 'v_pu'):
            if dss.Bus.kVBase() <= 0:
                raise SimulatorVariableError(
                    f"Bus '{bus_name}' has no voltage base; set 'voltagebases' and run "
                    f"'calcvoltagebases' in the DSS script to obtain per-unit voltages."
                )
            pu = dss.Bus.puVmagAngle()  # [mag1, ang1, mag2, ang2, ...]
            mags = [pu[2 * i] for i in sel]
            return float(sum(mags) / len(mags))
        if prop in ('vm_kv', 'vmag_kv'):
            va = dss.Bus.VMagAngle()
            mags = [va[2 * i] for i in sel]
            return float(sum(mags) / len(mags) / 1000.0)
        if prop in ('vang', 'va_deg', 'va'):
            va = dss.Bus.VMagAngle()
            return float(va[2 * sel[0] + 1])
        raise SimulatorVariableError(f"Unknown bus property '{parts[2]}' in '{param}'.")

    def _element_value(self, dss, etype, name, prop, param):
        self._activate_element(dss, etype, name, param)
        prop = prop.lower()
        powers = list(dss.CktElement.Powers())  # per terminal: nconds * (P, Q)
        nph = max(int(dss.CktElement.NumPhases()), 1)
        ncond = max(int(dss.CktElement.NumConductors()), nph)
        nterm = max(int(dss.CktElement.NumTerminals()), 1)

        def term_sum(t, which):
            base = t * ncond * 2
            return float(sum(powers[base + 2 * k + which] for k in range(nph)
                             if base + 2 * k + which < len(powers)))

        if prop in ('p_kw', 'p'):
            return term_sum(0, 0)
        if prop in ('q_kvar', 'q'):
            return term_sum(0, 1)
        if prop in ('p2_kw',):
            if nterm < 2:
                raise SimulatorVariableError(f"'{etype}.{name}' has only one terminal.")
            return term_sum(1, 0)
        if prop in ('q2_kvar',):
            if nterm < 2:
                raise SimulatorVariableError(f"'{etype}.{name}' has only one terminal.")
            return term_sum(1, 1)
        if prop in ('losses_kw',):
            return float(dss.CktElement.Losses()[0]) / 1000.0
        if prop in ('i_a', 'loading_percent'):
            cur = list(dss.CktElement.CurrentsMagAng())
            mags = [cur[2 * k] for k in range(nph) if 2 * k < len(cur)]
            i_a = float(max(mags)) if mags else 0.0
            if prop == 'i_a':
                return i_a
            norm = float(dss.CktElement.NormalAmps())
            if norm <= 0:
                raise SimulatorVariableError(f"'{etype}.{name}' has no NormAmps rating; cannot compute loading.")
            return i_a / norm * 100.0
        raise SimulatorVariableError(
            f"Unknown property '{prop}' in '{param}'. Valid: p_kw, q_kvar, p2_kw, q2_kvar, "
            f"losses_kw, i_a, loading_percent."
        )

    def get_value(self, parameters: List[str], time: Optional[float] = None) -> List[float]:
        """Query values from the solved OpenDSS circuit."""
        dss = self._require_active()
        results: List[float] = []
        for param in parameters:
            parts = param.split('.')
            target_type = parts[0].lower()
            if target_type == 'circuit':
                prop = parts[1].lower() if len(parts) > 1 else ''
                if prop in ('losses_kw', 'losses'):
                    results.append(float(dss.Circuit.Losses()[0]) / 1000.0)
                elif prop == 'losses_kvar':
                    results.append(float(dss.Circuit.Losses()[1]) / 1000.0)
                elif prop in ('total_power_kw', 'power_kw'):
                    results.append(float(dss.Circuit.TotalPower()[0]))
                elif prop in ('total_power_kvar', 'power_kvar'):
                    results.append(float(dss.Circuit.TotalPower()[1]))
                else:
                    raise SimulatorVariableError(
                        f"Unknown circuit property in '{param}'. Valid: losses_kw, losses_kvar, "
                        f"total_power_kw, total_power_kvar.")
            elif target_type == 'bus':
                results.append(self._bus_value(dss, parts, param))
            else:
                if len(parts) != 3:
                    raise SimulatorVariableError(
                        f"Cannot resolve OpenDSS query '{param}'. Expected 'Type.Name.Property'.")
                results.append(self._element_value(dss, parts[0], parts[1], parts[2], param))
        return results

    def step(self, time: float) -> None:
        """Re-solve the circuit.  Simulation time is not advanced (see module docs)."""
        dss = self._require_active()
        self._solve(dss)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    _STATIC_RE = re.compile(r'^\s*(?:new|edit)\s+([A-Za-z_]\w*)\.([\w\-]+)', re.I)
    _BUS_RE = re.compile(r'\bbus[12]?\s*=\s*([\w\-]+)', re.I)
    _REDIR_RE = re.compile(r'^\s*(?:redirect|compile)\s+"?([^"\r\n]+?)"?\s*$', re.I)

    def _scan_static(self, path, seen, found):
        path = os.path.abspath(path)
        if path in seen or not os.path.isfile(path):
            return
        seen.add(path)
        try:
            with open(path, 'r', errors='ignore') as fh:
                lines = fh.readlines()
        except OSError:
            return
        base = os.path.dirname(path)
        for line in lines:
            line = line.split('!')[0]
            m = self._REDIR_RE.match(line)
            if m:
                self._scan_static(os.path.join(base, m.group(1).strip()), seen, found)
                continue
            m = self._STATIC_RE.match(line)
            if m:
                found['elements'].setdefault(m.group(1).capitalize(), []).append(m.group(2))
                for b in self._BUS_RE.findall(line):
                    found['buses'].add(b)

    def get_available_variables(self) -> Dict[str, List[str]]:
        """Known variables: declared ones plus elements found in the circuit
        (live if active, otherwise by statically scanning the DSS script)."""
        elements: Dict[str, List[str]] = {}
        buses: List[str] = []
        if self._owns and _active_owner() is self:
            dss = self._dss
            for etype, getter in (('Load', 'Loads'), ('Generator', 'Generators'),
                                  ('PVSystem', 'PVsystems'), ('Storage', 'Storages'),
                                  ('Line', 'Lines'), ('Transformer', 'Transformers')):
                mod = getattr(dss, getter, None)
                try:
                    names = list(mod.AllNames()) if mod is not None else []
                except Exception:
                    names = []
                if names:
                    elements[etype] = names
            buses = list(dss.Circuit.AllBusNames())
        else:
            if self._static_cache is None:
                found = {'elements': {}, 'buses': set()}
                if self.dss_location:
                    self._scan_static(self.dss_location, set(), found)
                self._static_cache = found
            elements = {k: list(dict.fromkeys(v)) for k, v in self._static_cache['elements'].items()}
            buses = sorted(self._static_cache['buses'])

        inputs = list(self.inputs)
        outputs = list(self.outputs)
        for etype in ('Load', 'Generator', 'PVSystem', 'Storage'):
            for n in elements.get(etype, []):
                inputs += [f'{etype}.{n}.kW', f'{etype}.{n}.kvar']
        for etype, names in elements.items():
            for n in names:
                outputs += [f'{etype}.{n}.p_kw', f'{etype}.{n}.q_kvar', f'{etype}.{n}.losses_kw']
        for b in buses:
            outputs += [f'Bus.{b}.vm_pu', f'Bus.{b}.vm_kv', f'Bus.{b}.vang', f'Bus.{b}.kv']
        if elements or buses:
            outputs += ['Circuit.losses_kw', 'Circuit.losses_kvar',
                        'Circuit.total_power_kw', 'Circuit.total_power_kvar']
        inputs = list(dict.fromkeys(inputs))
        outputs = list(dict.fromkeys(outputs))
        return {'inputs': inputs, 'outputs': outputs, 'all': list(dict.fromkeys(inputs + outputs))}
